from typing import List

import torch
from torch import nn

from .config import EMB_DIM_CAP


class MultiTaskQuantileNet(nn.Module):
    def __init__(
        self,
        num_dim: int,
        cat_cardinals: List[int],
        emb_cap: int,
        hidden: int,
        layers: int,
        dropout: float,
        n_targets: int,
        n_quants: int,
    ):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_dims = []
        for card in cat_cardinals:
            dim = min(emb_cap, max(4, int(round(card ** 0.25) * 4)))
            self.embs.append(nn.Embedding(card + 1, dim, padding_idx=0))
            emb_dims.append(dim)
        in_dim = num_dim + sum(emb_dims)

        mlp = []
        d = in_dim
        for _ in range(layers):
            mlp += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(d, n_quants) for _ in range(n_targets)])

    def forward(self, x_num, x_cat):
        h = x_num if x_num.numel() != 0 else None
        if x_cat.numel() != 0:
            emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embs)]
            emb = torch.cat(emb, dim=1) if emb else None
            h = emb if h is None else torch.cat([h, emb], dim=1)
        z = self.trunk(h) if isinstance(self.trunk, nn.Sequential) else h
        outs = [head(z) for head in self.heads]
        return outs


def pinball_loss(pred, target, taus):
    diff = target.unsqueeze(1) - pred
    losses = []
    for i, q in enumerate(taus):
        e = diff[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.mean(torch.stack(losses, dim=1))


def dlog_to_level(idx_now: torch.Tensor, dlog: torch.Tensor) -> torch.Tensor:
    return torch.expm1(torch.log1p(idx_now) + dlog)
