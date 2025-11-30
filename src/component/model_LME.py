# src/component/model.py
from __future__ import annotations
import torch
from torch import nn
from typing import Tuple, List

class IntrinsicPriceNet(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout_prob: float = 0.0):
        super().__init__()
        mods: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            mods += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU()]
            if dropout_prob > 0: mods.append(nn.Dropout(dropout_prob))
            last = h
        mods.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
