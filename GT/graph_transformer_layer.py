import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

import geoopt
from geoopt import ManifoldParameter


"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


# Model components
class HyperbolicLinear(nn.Module):
        def __init__(self, in_features, out_features, manifold, c):
            super().__init__()
            self.weight = ManifoldParameter(torch.randn(out_features, in_features) * 0.01, manifold=manifold)
            self.bias = ManifoldParameter(torch.zeros(out_features), manifold=manifold)
            self.manifold = manifold
            self.c = c

        def forward(self, x):
            # print("feat  ", x.shape)
            x = self.manifold.mobius_matvec(self.weight, x)
            x = self.manifold.mobius_add(x, self.bias)
            return x


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # self.rotary_emb = RotaryEmbedding(dim = out_dim * num_heads)
        
    
    def propagate_attention(self, g):
        # Compute attention score
        # g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        # Compute attention score
       
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        # print("attention scores:  ", g.edata['score'].shape)

        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, pos_enc):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g)
        
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, manifold, c, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        # self.manifold = geoopt.PoincareBall(c=c)
        self.manifold = manifold
        self.c = c
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        if self.c == 0:
            self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
            self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)
        else:
            self.FFN_layer1 = HyperbolicLinear(out_dim, out_dim*2, self.manifold, c)
            self.FFN_layer2 = HyperbolicLinear(out_dim*2, out_dim, self.manifold, c)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, pos_enc):
        h_in1 = h # for first residual connection
        if self.c != 0:
            h_in1 = self.manifold.expmap0(h_in1)
        
        # multi-head attention out
        attn_out = self.attention(g, h, pos_enc)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        # residual connection
        if self.residual:
            if self.c != 0:
                h = self.manifold.mobius_add(h_in1, h)
                h = self.manifold.logmap0(h)
            else:
                h = h_in1 + h 
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        # for second residual connection
        h_in2 = h 
        if self.c != 0:
            h_in2 = self.manifold.expmap0(h_in2)
        
        # FFN
        if self.c != 0:
            h = self.manifold.expmap0(h)
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)
      
        # residual connection
        if self.residual:
            if self.c != 0:
                h = self.manifold.mobius_add(h_in2, h)
                h = self.manifold.logmap0(h)
            else:
                h = h_in2 + h 
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)