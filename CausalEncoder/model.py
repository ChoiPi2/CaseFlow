
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear

from gcn_conv import GCNConv


def _safe_bn(bn: BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    if x.dim() > 1 and x.size(0) <= 1 and bn.training:
        return x
    return bn(x)


class BIG(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.bn_input = BatchNorm1d(in_channels)
        self.proj_conv = GCNConv(in_channels, hidden, edge_norm=True, gfn=True)
        self.convs = nn.ModuleList(
            [GCNConv(hidden, hidden, edge_norm=True, gfn=False) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList([BatchNorm1d(hidden) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = _safe_bn(self.bn_input, x)
        x = self.proj_conv(x, edge_index)
        x = F.relu(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(_safe_bn(bn, x), edge_index, edge_weight=None)
            x = F.relu(x)
        return x


class AttCov(nn.Module):
    def __init__(self, hidden: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.node_att_conv = GCNConv(hidden * 2, 2, edge_norm=True, gfn=False)
        self.edge_att_mlp = Linear(hidden * 3, 2)

    def forward(self, h, z_q_per_node, edge_r_h, edge_index):
        node_rep = torch.cat([h, z_q_per_node], dim=-1)
        node_att = F.softmax(
            self.node_att_conv(node_rep, edge_index) / self.temperature,
            dim=-1,
        )

        src, dst = edge_index
        edge_rep = torch.cat([h[src], h[dst], edge_r_h], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        return node_att, edge_att


class GConvBranch(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.bn = BatchNorm1d(hidden)
        self.conv = GCNConv(hidden, hidden, edge_norm=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, alpha):
        z = F.elu(self.conv(_safe_bn(self.bn, x), edge_index, edge_weight=alpha))
        z = self.dropout(z)
        return z


class CausalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        att_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.proj_q = Linear(in_channels, hidden)
        self.proj_r = Linear(in_channels, hidden)
        self.proj_a = Linear(in_channels, hidden)
        self.big = BIG(in_channels, hidden, num_layers, dropout)
        self.att_cov = AttCov(hidden, temperature=att_temperature)
        self.gconv_c = GConvBranch(hidden, dropout)
        self.gconv_s = GConvBranch(hidden, dropout)

    def forward(
        self,
        data,
        return_embeddings: bool = False,
        return_alpha: bool = False,
        return_h: bool = False,
        return_xs: bool = False,
    ):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        z_q_raw = data.z_q
        if z_q_raw.dim() == 1:
            z_q_raw = z_q_raw.unsqueeze(0)
        z_q_h = self.proj_q(z_q_raw)
        z_q_per_node = z_q_h[batch]

        edge_r_raw = getattr(data, "edge_z_r", None)
        if edge_r_raw is None:
            edge_r_raw = x.new_zeros((edge_index.size(1), x.size(-1)))
        if edge_r_raw.dim() == 1:
            edge_r_raw = edge_r_raw.unsqueeze(0)
        if edge_r_raw.size(0) != edge_index.size(1):
            raise ValueError(
                f"edge_z_r length ({edge_r_raw.size(0)}) must match edge_index columns ({edge_index.size(1)})."
            )
        edge_r_h = self.proj_r(edge_r_raw)

        x_h = self.big(x, edge_index)
        node_att, edge_att = self.att_cov(x_h, z_q_per_node, edge_r_h, edge_index)

        ac_node = node_att[:, 0].view(-1, 1)
        as_node = node_att[:, 1].view(-1, 1)
        ac_edge = edge_att[:, 0]
        as_edge = edge_att[:, 1]

        x_c = ac_node * x_h
        x_s = as_node * x_h
        z_c_node = self.gconv_c(x_c, edge_index, ac_edge)
        z_s_node = self.gconv_s(x_s, edge_index, as_edge)

        z_a_proj = None
        if hasattr(data, "z_a") and data.z_a is not None:
            z_a = data.z_a
            if z_a.dim() == 1:
                z_a = z_a.unsqueeze(0)
            z_a_proj = self.proj_a(z_a)

        if return_alpha:
            if return_xs:
                return (
                    z_c_node, z_s_node,
                    ac_node, as_node, ac_edge, as_edge,
                    z_a_proj, x_s, x_h,
                )
            if return_h:
                return z_c_node, z_s_node, ac_node, as_node, ac_edge, as_edge, z_a_proj, x_h
            return z_c_node, z_s_node, ac_node, as_node, ac_edge, as_edge, z_a_proj

        if return_xs:
            return z_c_node, z_s_node, z_a_proj, x_s, x_h
        if return_h:
            return z_c_node, z_s_node, z_a_proj, x_h
        return z_c_node, z_s_node, z_a_proj
