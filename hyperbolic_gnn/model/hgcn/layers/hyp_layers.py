import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.module import Module
import math
import torch.nn.functional as F

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """
    def __init__(self, latent_dim,  num_layers,manifold,curve):
        super(HyperbolicGraphConvolution, self).__init__()
        self.in_features = latent_dim
        self.num_layers = num_layers
        self.manifold=manifold
        self.curve=curve

    def denseGCN(self, x, adj):
        output = [x]
        for i in range(self.num_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


    def forward(self, x, adj):
        x=self.manifold.logmap0(x,self.curve)
        output = self.denseGCN(x, adj)
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in,c_out):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c_in = c_in
        self.c_out = c_out
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, 0.2,training=True)
        # 将在曲率为c_in的x映射到tangent space做矩阵乘法，然后映射到曲率为c_out的空间
        mv = self.manifold.multi_curve_mobius_matvec(drop_weight, x, self.c_in,self.c_out)
        res = self.manifold.proj(mv, self.c_out)
        bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c_out)
        hyp_bias = self.manifold.expmap0(bias, self.c_out)
        hyp_bias = self.manifold.proj(hyp_bias, self.c_out)
        res = self.manifold.mobius_add(res, hyp_bias, c=self.c_out)
        res = self.manifold.proj(res, self.c_out)

        return res




class LorentzLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 c,
                 bias=True,
                 dropout=0.1,
                 scale=30,
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not False)
        self.c=c
    def forward(self, x):
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1/self.c) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)





class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = nn.ReLU()

    def forward(self, x):
        xt = self.manifold.logmap0(x, c=self.c_in)
        xt = self.act(xt)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, act):
        super(HNNLayer, self).__init__()
        self.linear1 = HypLinear(manifold, in_features, out_features, c)
        self.linear2 = HypLinear(manifold, in_features, out_features, c)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear1.forward(x)
        h = self.hyp_act.forward(h)
        h = self.linear2.forward(h)
        h = self.hyp_act.forward(h)
        return h

class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist, split='train'):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs