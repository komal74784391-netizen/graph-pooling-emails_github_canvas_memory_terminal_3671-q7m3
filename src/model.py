import torch
import torch.nn as nn

from .pooling import LearnedPooling


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth=3):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [LearnedPooling(hidden_dim) for _ in range(depth)]
        )
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        h = self.act(self.input(x))
        keep = []
        for block in self.blocks:
            h, mask = block(h, edge_index)
            keep.append(mask)
        return h, keep
