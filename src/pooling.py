import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPooling(nn.Module):
    """Differentiable coarsening operator with a learned score head."""

    def __init__(self, channels, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.score = nn.Linear(channels, 1)
        self.update = nn.Linear(channels, channels)

    def forward(self, h, edge_index):
        scores = self.score(h).squeeze(-1)
        k = max(1, int(h.shape[0] * self.ratio))
        mask = torch.topk(scores, k).indices
        pooled = self.update(h[mask])
        return F.gelu(pooled), mask
