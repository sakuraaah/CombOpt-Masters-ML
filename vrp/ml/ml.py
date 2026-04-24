import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv


class GNNMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        # message passing
        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            beta=True,
            dropout=dropout,
            edge_dim=hidden_dim,
        )
        self.conv_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(hidden_dim)

        # node mlp
        self.node_ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # edge mlp
        self.edge_ff = nn.Sequential(
            nn.Linear(3 * hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # message passing
        node_msg = self.conv(node_h, edge_index, edge_h)
        node_h = node_h + self.conv_dropout(node_msg)
        node_h = self.conv_norm(node_h)

        # node mlp
        node_delta = self.node_ff(node_h)
        node_h = node_h + self.node_dropout(node_delta)
        node_h = self.node_norm(node_h)

        # edge mlp
        src, dst = edge_index
        edge_repr = torch.cat(
            [node_h[src], node_h[dst], edge_h],
            dim=-1,
        )
        edge_delta = self.edge_ff(edge_repr)
        edge_h = edge_h + self.edge_dropout(edge_delta)
        edge_h = self.edge_norm(edge_h)

        return node_h, edge_h


class VRPAnalysisModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        # node entry encoding
        self.node_in = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # edge entry encoding
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # message passing (num_layers times)
        self.blocks = nn.ModuleList(
            [
                GNNMessagePassingLayer(hidden_dim, heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # final edge mlp (return single logit)
        self.edge_head = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        # node entry encoding
        node_h = self.node_in(data.x)

        # edge entry encoding
        edge_h = self.edge_in(data.edge_attr)

        # message passing (num_layers times)
        for block in self.blocks:
            node_h, edge_h = block(node_h, data.edge_index, edge_h)

        # final edge mlp (return single logit)
        src, dst = data.edge_index
        edge_repr = torch.cat(
            [
                node_h[src],
                node_h[dst],
                edge_h,
                torch.abs(node_h[src] - node_h[dst]),
                node_h[src] * node_h[dst],
            ],
            dim=-1,
        )

        return self.edge_head(edge_repr).squeeze(-1)
