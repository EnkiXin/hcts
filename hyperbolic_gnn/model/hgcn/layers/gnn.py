
import dgl.nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv,GATConv,GATv2Conv,SAGEConv
import dgl.nn.functional as fnl
import dgl.function as fn

class GATLayer(nn.Module):
    def __init__(self,
                 g,
                 dim1,
                 dim2):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(dim1, dim2, bias=False)
        self.attn_fc = nn.Linear(dim1*2, 1, bias=False)

    def edge_attention(self, edges):
        z = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z)
        return {"e_unnormalized": F.leaky_relu(a)}


    def forward(self,h):
        self.g.ndata['h']=h
        self.g.apply_edges(self.edge_attention)
        self.g.edata['e'] = fnl.edge_softmax(self.g, self.g.edata['e_unnormalized'])
        self.g.update_all(
            message_func=fn.u_mul_e('h', 'e', 'm'),
            reduce_func=fn.sum('m', 'h')
        )
        return self.g.ndata['h']

class GCN(nn.Module):
    def __init__(self, latent_dim, num_layers,src,dst):
        super(GCN, self).__init__()
        self.src=src
        self.dst=dst
        self.g=self.create_graph()
        self.latent_dim=latent_dim
        self.layers = nn.ModuleList([GraphConv(self.latent_dim,self.latent_dim) for _ in range(num_layers)])

    def create_graph(self):
        g=dgl.to_bidirected(dgl.graph((self.src,self.dst)))
        g=dgl.add_self_loop(g).to('cuda')
        return g
    def forward(self, h):
        self.g.ndata['h']=h
        for layer in self.layers:
            h=layer(self.g,h)
        return h


class GCN(nn.Module):
    def __init__(self, latent_dim, num_layers,src,dst):
        super(GCN, self).__init__()
        self.src=src
        self.dst=dst
        self.g=self.create_graph()
        self.latent_dim=latent_dim
        self.layers = nn.ModuleList([GraphConv(self.latent_dim,self.latent_dim) for _ in range(num_layers)])

    def create_graph(self):
        g=dgl.to_bidirected(dgl.graph((self.src,self.dst)))
        g=dgl.add_self_loop(g).to('cuda')
        return g
    def forward(self, h):
        self.g.ndata['h']=h
        for layer in self.layers:
            h=layer(self.g,h)
        return h

class GAT(nn.Module):
    def __init__(self, latent_dim, num_layers,num_heads,src,dst):
        super(GAT, self).__init__()
        self.src=src
        self.dst=dst
        self.g=self.create_graph()
        self.latent_dim=latent_dim
        self.num_heads=num_heads
        self.layers = nn.ModuleList([GATConv(self.latent_dim,self.latent_dim,num_heads=self.num_heads) for _ in range(num_layers)])

    def create_graph(self):
        g=dgl.to_bidirected(dgl.graph((self.src,self.dst)))
        g=dgl.add_self_loop(g).to('cuda')
        return g
    def forward(self, h):
        self.g.ndata['h']=h
        for layer in self.layers:
            h=layer(self.g,h)
            h=torch.mean(h, dim=1)
        return h

class GATv2(nn.Module):
    def __init__(self, latent_dim, num_layers,num_heads,src,dst):
        super(GATv2, self).__init__()
        self.src=src
        self.dst=dst
        self.g=self.create_graph()
        self.latent_dim=latent_dim
        self.num_heads=num_heads
        self.layers = nn.ModuleList([GATv2Conv(self.latent_dim,self.latent_dim,num_heads=self.num_heads) for _ in range(num_layers)])

    def create_graph(self):
        g=dgl.to_bidirected(dgl.graph((self.src,self.dst)))
        g=dgl.add_self_loop(g).to('cuda')
        return g
    def forward(self, h):
        self.g.ndata['h']=h
        for layer in self.layers:
            h=layer(self.g,h)
            h=torch.mean(h, dim=1)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, latent_dim, num_layers,src,dst):
        super(GraphSAGE, self).__init__()
        self.src=src
        self.dst=dst
        self.g=self.create_graph()
        self.latent_dim=latent_dim
        self.layers = nn.ModuleList([SAGEConv(self.latent_dim,self.latent_dim,'mean') for _ in range(num_layers)])

    def create_graph(self):
        g=dgl.to_bidirected(dgl.graph((self.src,self.dst)))
        g=dgl.add_self_loop(g).to('cuda')
        return g
    def forward(self, h):
        self.g.ndata['h']=h
        for layer in self.layers:
            h=layer(self.g,h)

        return h
