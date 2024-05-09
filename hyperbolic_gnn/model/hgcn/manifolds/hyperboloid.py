"""Hyperboloid manifold. Copy from https://github.com/HazyResearch/hgcn """
import torch
from hyperbolic_gnn.model.hgcn.manifolds.base import Manifold
from hyperbolic_gnn.model.hgcn.utils.math_utils import arcosh, cosh, sinh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.
    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K
    c = 1 / K is the hyperbolic curvature.
    """
    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def check_point_on_manifold(self,x,c) :
        return self.minkowski_dot(x,x)

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def hyper_dist(self,c,z_i,z_j):
        K=1. / c
        time=-z_i[:,0]*z_j[:,0]
        space=torch.sum(z_i[:, 1:]*z_j[:, 1:],dim=1)
        prod=time+space
        theta = torch.clamp(-prod / K, min=1.0 + 1e-7)
        sqdist = K * arcosh(theta) ** 2
        return sqdist

    def minkowski_matrix_dot(self,X,Y,keepdim=True):
        rows, cols=X.shape
        X_0=X
        X_0[:, cols - 1] = 0
        Y_0=Y
        Y_0[:, cols - 1] = 0
        res=torch.mm(X, Y.t())-2*torch.mm(X_0, Y_0.t())
        return res

    def matrix_sqdist(self, X, Y, c):
        K = 1. / c
        prod = self.minkowski_matrix_dot(X, Y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[X.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u


    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        # d=dim-1
        d = u.size(-1) - 1
        # 取出向量的后面d维
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def egrad2rgrad(self, x, grad, k, dim=-1):
        grad.narrow(-1, 0, 1).mul_(-1)
        grad = grad.addcmul(self.inner(x, grad, dim=dim, keepdim=True), x / k)
        return grad

    def inner(self, u, v, keepdim: bool = False, dim: int = -1):
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
                dim, 1, d
            ).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
                dim=dim, keepdim=True
            )

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def multi_curve_mobius_matvec(self, m, x, c_in,c_out):
        u = self.logmap0(x, c_in)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c_out)

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_scaler_mul(self,k,x,c):
        return self.expmap0(k*self.logmap0(x,c),c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

    def minkowski_tensor_dot(self,X,Y):
        # [H,B,N,D]
        # 计算两个tensor所有的pair的闵可夫斯基内积
        X_time = X[..., 0].unsqueeze(-1)
        X_space = X[..., 1:]
        Y_time = Y[..., 0].unsqueeze(-1)
        Y_space = Y[..., 1:]
        res = -torch.einsum('...nd,...ld->...nl',X_time,Y_time) + torch.einsum('...nd,...ld->...nl',X_space,Y_space)
        return res

    def hyper_dist(self,tensor1,tensor2,c):
        # 两个tensor，所有pair计算双曲距离
        theta=-self.minkowski_tensor_dot(tensor1,tensor2)*c
        sqdist=(1/torch.sqrt(torch.tensor(c)))*arcosh(theta)
        return sqdist

    def pair_wise_minkowski_tensor_dot(self, X, Y):
        # 计算两个同形状tensor两两对应向量间的闵可夫斯基内积
        X_time = X[..., 0].unsqueeze(-1)
        X_space = X[..., 1:]
        Y_time = Y[..., 0].unsqueeze(-1)
        Y_space = Y[..., 1:]
        res=-(X_time*Y_time)+torch.sum(X_space*Y_space,dim=-1).unsqueeze(-1)
        return res

    def pair_wize_hyper_dist(self,tensor1,tensor2,c):
        # 两个形状相同的tnesor，两两对应向量间的双曲距离
        theta=-self.pair_wise_minkowski_tensor_dot(tensor1,tensor2)*c
        sqdist=(1/torch.sqrt(torch.tensor(c)))*arcosh(theta)
        return sqdist

    def hyperbolic_mean(self,weight_tensor,value_tensor,c):
        eu_weighted_sum=torch.einsum('nhd,nhi->nd',value_tensor,weight_tensor)
        # 每个向量计算一个norm
        x_norm=self.tensor_minkouski_norm(eu_weighted_sum)
        # num_heads*batch_size
        coefficient = (1 /(c ** 0.5)) / torch.sqrt(torch.abs(x_norm))
        hyper_weighted_sum=torch.einsum('nd,ni->nd',eu_weighted_sum,coefficient)
        return hyper_weighted_sum

    def tensor_minkouski_norm(self,x):
        # 所有向量与自己做闵可夫斯基内积,然后元素求和
        x_time=x[..., 0].unsqueeze(-1)
        x_space=x[..., 1:]
        norm=-(x_time*x_time)+torch.sum(x_space*x_space,dim=-1).unsqueeze(-1)
        return norm