# src/component/surface.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from sklearn.neighbors import KDTree

class KernelSurface:
    def __init__(self, S_train_std: np.ndarray, K: int = 40, q: float = 1.0):
        self.S = S_train_std.astype(np.float32)
        self.n, _ = self.S.shape
        self.K, self.q = int(K), float(q)
        self.tree = KDTree(self.S, leaf_size=40)

    def _neighbors(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        d, ind = self.tree.query(self.S[i:i+1], k=self.K+1)
        ind = ind[0]; d = d[0]
        mask = ind != i
        return ind[mask][:self.K], (d[mask][:self.K] ** 2)

    def _adapt_weights(self, d2_i: np.ndarray) -> np.ndarray:
        sigma2 = float(np.median(d2_i) + 1e-12)
        w = np.exp(- self.q * d2_i / (2.0 * sigma2))
        return w / (w.sum() + 1e-12)

    def build_U_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        U = []
        for i in range(self.n):
            idxs, d2 = self._neighbors(i)
            w = self._adapt_weights(d2)
            U.append((idxs.astype(np.int32), w.astype(np.float32)))
        return U

    def interpolate_one(self, x_new_std: np.ndarray, D: np.ndarray) -> float:
        d, ind = self.tree.query(x_new_std.reshape(1,-1), k=self.K)
        ind = ind[0]; d2 = (d[0] ** 2)
        w = self._adapt_weights(d2)
        return float(np.dot(w, D[ind].astype(np.float64)))

class LaplacianOp:
    def __init__(self, surf: KernelSurface, K_lap: Optional[int] = None, q: float = 1.0):
        self.S, self.tree, self.n = surf.S, surf.tree, surf.n
        self.K, self.q = int(K_lap or surf.K), float(q)
        self.lap_list = []
        for i in range(self.n):
            d, ind = self.tree.query(self.S[i:i+1], k=self.K+1)
            ind = ind[0]; d = d[0]
            mask = ind != i
            idxs = ind[mask][:self.K]; d2 = (d[mask][:self.K] ** 2)
            sigma2 = float(np.median(d2) + 1e-12)
            w = np.exp(- self.q * d2 / (2.0 * sigma2))
            w = w / (w.sum() + 1e-12)
            self.lap_list.append((idxs.astype(np.int32), w.astype(np.float32)))

    def apply(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v, dtype=np.float64)
        for i, (idxs, w) in enumerate(self.lap_list):
            out[i] += np.sum(w * (v[i] - v[idxs]))
        return out
