import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros


class GCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_norm: bool = True,
        gfn: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super(GCNConv, self).__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor = None,
    ) -> Tensor:

        x = torch.matmul(x, self.weight)


        if self.gfn:
            return x

        N = x.size(0)

        if edge_weight is not None:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index, edge_weight = add_self_loops(
                edge_index,
                edge_attr=edge_weight,
                fill_value=1.0,
                num_nodes=N,
            )
        else:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            edge_weight = torch.ones(
                edge_index.size(1), dtype=x.dtype, device=x.device
            )

        row, col = edge_index

        if self.edge_norm:
            deg = torch.zeros(N, dtype=x.dtype, device=x.device)
            deg.scatter_add_(0, row, edge_weight)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = edge_weight

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out: Tensor) -> Tensor:

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
