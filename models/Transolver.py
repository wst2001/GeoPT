import torch
import torch.nn as nn
import numpy as np

try:
    from timm.models.layers import trunc_normal_
except ImportError:  # timm is only needed for weight init, torch ships the same op
    from torch.nn.init import trunc_normal_
from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_1D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_2D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_3D
import torch.utils.checkpoint as checkpoint

PHYSICS_ATTENTION = {
    'unstructured': Physics_Attention_Irregular_Mesh,
    'structured_1D': Physics_Attention_Structured_Mesh_1D,
    'structured_2D': Physics_Attention_Structured_Mesh_2D,
    'structured_3D': Physics_Attention_Structured_Mesh_3D
}

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU
}


def masked_mean(feature, mask=None):
    """Average per-point features over the point dim, ignoring invalid points."""
    if mask is None:
        return feature.mean(dim=1)
    w = mask.to(feature.dtype)[:, :, None]
    return (feature * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-5)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transolver encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            geotype='unstructured',
            shapelist=None
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = PHYSICS_ATTENTION[geotype](hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                               dropout=dropout, slice_num=slice_num, shapelist=shapelist)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, return_feature=False, mask=None):
        if mask is None:
            fx = self.Attn(self.ln_1(fx)) + fx
        else:
            fx = self.Attn(self.ln_1(fx), mask=mask) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            out = self.mlp2(self.ln_3(fx))
            if return_feature:
                return out, fx
            return out
        else:
            return fx


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Transolver'
        self.args = args
        ## embedding
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)

        ## models
        self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                      dropout=args.dropout,
                                                      act=args.act,
                                                      mlp_ratio=args.mlp_ratio,
                                                      out_dim=args.out_dim,
                                                      slice_num=args.slice_num,
                                                      last_layer=(_ == args.n_layers - 1),
                                                      geotype=args.geotype,
                                                      shapelist=args.shapelist)
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def structured_geo(self, x, fx, return_feature=False):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        mean_feature = None
        for block in self.blocks:
            if return_feature and block.last_layer:
                out, feature = block(fx, return_feature=True)
                fx = out
                mean_feature = feature.mean(dim=1)
            elif self.args.checkpoint:
                fx = checkpoint.checkpoint(block, fx)
            else:
                fx = block(fx)
        if return_feature:
            return fx, mean_feature
        return fx

    def unstructured_geo(self, x, fx, return_feature=False, mask=None, return_point_feature=False):
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        mean_feature = None
        point_feature = None
        want_feature = return_feature or return_point_feature
        for block in self.blocks:
            if want_feature and block.last_layer:
                out, feature = block(fx, return_feature=True, mask=mask)
                fx = out
                if return_point_feature:
                    point_feature = feature
                if return_feature:
                    mean_feature = masked_mean(feature, mask)
            elif self.args.checkpoint:
                fx = checkpoint.checkpoint(block, fx, False, mask)
            else:
                fx = block(fx, mask=mask)
        if return_feature and return_point_feature:
            return fx, mean_feature, point_feature
        if return_point_feature:
            return fx, point_feature
        if return_feature:
            return fx, mean_feature
        return fx

    def forward(self, x, fx, return_feature=False, mask=None, return_point_feature=False):
        if self.args.geotype == 'unstructured':
            return self.unstructured_geo(x, fx, return_feature=return_feature, mask=mask,
                                         return_point_feature=return_point_feature)
        else:
            if mask is not None or return_point_feature:
                raise NotImplementedError('mask / per-point features are only supported for unstructured geometry')
            return self.structured_geo(x, fx, return_feature=return_feature)
