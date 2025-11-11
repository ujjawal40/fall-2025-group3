# src/lme_model.py
# --------------------------------------------------------
# Latent Manifold Estimation (Chopra et al. style)
# Two trainable components:
#   1) Parametric intrinsic price model  G(W, x)
#   2) Non-parametric desirability model H(D, x)
# Plus an EM-like outer loop to learn both.
#
# Assumes there is a CSV at one of:
#   - "src/data/sub_sample.csv"
#   - "data/src/sub_sample.csv"
#   - "data/sub_sample.csv"
#   - "sub_sample.csv"
#
# You can run:  python -m src.lme_model
# after adjusting paths/column names.
# --------------------------------------------------------
from __future__ import annotations

import os
from typing import List, Tuple, Optional, Literal, Dict

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_preprocessor import DataPreprocessor  # both in src/

# ----------------------------
# 1. PARAMETRIC COMPONENT
# ----------------------------
class IntrinsicPriceNet(nn.Module):
    """
    G(W, x): parametric model that predicts the intrinsic log-price m_i
    We keep it close to the paper: 2 hidden layers (80, 40) → 1 output.
    """
    def __init__(self, in_dim: int, h1: int = 80, h2: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,) not (batch,1)


# ----------------------------
# 2. NON-PARAMETRIC BASE
# ----------------------------
class BaseDesirabilitySurface:
    """
    Common utilities for H(D, x):
    - compute K nearest neighbours
    - compute kernel weights
    We always return a *sparse* representation of U_i:
       U_i = (indices, weights) so that h_i = sum_j weights[j] * d[indices[j]]
    """
    def __init__(
        self,
        X_np: np.ndarray,
        K: int = 13,
        q: float = 1.0,
    ):
        """
        X_np: (n_samples, n_features) *standardized* features used for distance
        K:    number of neighbors N(x)
        q:    kernel sharpness in exp(-q ||x - x_j||^2)
        """
        self.X = X_np
        self.n, self.d = X_np.shape
        self.K = K
        self.q = q

        # precompute pairwise distances once (O(n^2)), fine for sub_sample.csv
        # then keep top-K per point
        self.nn_indices = self._build_all_neighbors()

    def _build_all_neighbors(self) -> List[np.ndarray]:
        """
        For every i, return indices of K+1 nearest points (we'll drop self later).
        """
        # pairwise squared distances
        X2 = np.sum(self.X ** 2, axis=1, keepdims=True)
        # (n, n) distance matrix
        dists = X2 + X2.T - 2 * (self.X @ self.X.T)
        np.fill_diagonal(dists, np.inf)
        # argsort along each row
        nn_idx = np.argsort(dists, axis=1)[:, : self.K]
        return [nn_idx[i] for i in range(self.n)]

    def _kernel_weights(self, x_i: np.ndarray, neigh_idx: np.ndarray) -> np.ndarray:
        neigh = self.X[neigh_idx]
        d2 = np.sum((neigh - x_i) ** 2, axis=1)
        w = np.exp(-self.q * d2)
        w = w / (w.sum() + 1e-12)
        return w

    def build_U_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        This will be overridden by subclasses to produce
        U_i = (indices, weights) so that h_i = sum weights * d[indices]
        """
        raise NotImplementedError


# ----------------------------
# 2a. KERNEL-BASED VERSION
# ----------------------------
class KernelDesirabilitySurface(BaseDesirabilitySurface):
    """
    H(D, x_i) = sum_{j in N(i)} Ker(x_i, x_j) * d_j
    exactly eq. (1)-(2) from the paper.
    """
    def build_U_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        U_list = []
        for i in range(self.n):
            x_i = self.X[i]
            neigh_idx = self.nn_indices[i]

            # IMPORTANT per paper: remove self if present
            neigh_idx = neigh_idx[neigh_idx != i]

            weights = self._kernel_weights(x_i, neigh_idx)
            U_list.append((neigh_idx, weights))
        return U_list


# ----------------------------
# 2b. WEIGHTED LOCAL LINEAR REGRESSION VERSION
# ----------------------------
class LLRDesirabilitySurface(BaseDesirabilitySurface):
    """
    For each x_i, fit a local weighted linear reg on its neighbors
    with targets = desirabilities of neighbors.
    BUT during phase 1 we *don't* know D yet. The paper notes (remark 1)
    that h_i can still be written as
         h_i = sum_{j in N(i)} a_ij * d_j
    where a_ij depends only on X, not on D.
    So here we precompute those a_ij.
    """
    def build_U_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        U_list = []
        for i in range(self.n):
            x_i = self.X[i]  # (d,)
            neigh_idx = self.nn_indices[i]
            neigh_idx = neigh_idx[neigh_idx != i]

            X_neigh = self.X[neigh_idx]  # (k, d)
            # kernel weights for LLR
            w = self._kernel_weights(x_i, neigh_idx)  # (k,)

            # Design matrix Z = [1, x_j]
            ones = np.ones((X_neigh.shape[0], 1))
            Z = np.concatenate([ones, X_neigh], axis=1)  # (k, d+1)

            # Weighted least squares to get the linear map from d_neigh → h_i
            # theta = (Z^T W Z)^-1 Z^T W d_neigh
            # h_i = [1, x_i]^T theta
            #    = [1, x_i]^T (Z^T W Z)^-1 Z^T W d_neigh
            # Let B = [1, x_i]^T (Z^T W Z)^-1 Z^T W  => shape (1, k)
            W = np.diag(w)
            ZTW = Z.T @ W
            M = ZTW @ Z  # (d+1, d+1)
            # regularize a bit for numerical stability
            M = M + 1e-6 * np.eye(M.shape[0])
            M_inv = np.linalg.inv(M)
            xi_aug = np.concatenate([[1.0], x_i])  # (d+1,)
            B = xi_aug @ M_inv @ ZTW  # (k,)

            # B are the coefficients a_ij in eq. (5)
            U_list.append((neigh_idx, B))
        return U_list


# ----------------------------
# 3. CONJUGATE GRADIENT FOR D
# ----------------------------
def cg_solve(
    apply_A,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Conjugate gradient on A x = b, where A is given as a matrix-free operator.
    """
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - apply_A(x)
    p = r.copy()
    rsold = np.dot(r, r)

    for _ in range(max_iter):
        Ap = apply_A(p)
        alpha = rsold / (np.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew
    return x


# ----------------------------
# 4. LME TRAINER
# ----------------------------
class LMETrainer:
    """
    Orchestrates the 2-phase optimization:
    Phase 1: solve for D using fixed G(W, .)
    Phase 2: train G with fixed D
    """
    def __init__(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        model: IntrinsicPriceNet,
        surface: BaseDesirabilitySurface,
        reg_r: float = 1e-2,
        device: str = "cpu",
    ):
        self.X_np = X_np
        self.y_np = y_np
        self.n = X_np.shape[0]

        self.model = model.to(device)
        self.surface = surface
        self.reg_r = reg_r
        self.device = device

        # initialize desirability vector D = [d1 ... dn]
        self.D = np.zeros(self.n, dtype=np.float64)

    # ---------- Phase 1: update D ----------
    def _update_D(self, U_list: List[Tuple[np.ndarray, np.ndarray]], m_np: np.ndarray):
        """
        Solve:
            L(D) = r/2 ||D||^2 + 1/2 sum_i (y_i - (m_i + U_i^T D))^2
        → (r I + sum_i U_i U_i^T) D = sum_i (y_i - m_i) U_i
        We'll build b and a matrix-free A·v.
        """
        n = self.n
        y = self.y_np
        r = self.reg_r

        # build b
        b = np.zeros(n, dtype=np.float64)
        for i, (idxs, weights) in enumerate(U_list):
            residual = y[i] - m_np[i]
            b[idxs] += residual * weights

        # matrix-free A·v
        def apply_A(v: np.ndarray) -> np.ndarray:
            out = r * v
            for idxs, weights in U_list:
                # (U_i^T v)
                coeff = np.dot(v[idxs], weights)
                out[idxs] += coeff * weights
            return out

        D_new = cg_solve(apply_A, b, x0=self.D, tol=1e-4, max_iter=300)
        self.D = D_new

    # ---------- Phase 2: update W (parametric) ----------
    def _update_W(
        self,
        X_torch: torch.Tensor,
        y_torch: torch.Tensor,
        U_list: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 64,
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        dataset = TensorDataset(X_torch, y_torch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        # we need fast access to h_i = U_i^T D for all i
        h_all = np.zeros(self.n, dtype=np.float32)
        for i, (idxs, weights) in enumerate(U_list):
            h_all[i] = np.dot(weights, self.D[idxs])
        h_all_t = torch.from_numpy(h_all).to(self.device)

        for _ in range(epochs):
            for batch_idx, (xb, yb) in enumerate(loader):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # indices of this batch in original array
                # (DataLoader shuffles, but dataset is sequential, so we can get it like this)
                start = batch_idx * batch_size
                end = start + xb.size(0)
                h_b = h_all_t[start:end]

                m_b = self.model(xb)
                pred_b = m_b + h_b
                loss = 0.5 * torch.mean((yb - pred_b) ** 2)

                optim.zero_grad()
                loss.backward()
                optim.step()

    def fit(
        self,
        outer_iters: int = 5,
        llr: bool = False,
    ):
        """
        outer_iters: number of EM-like passes
        """
        X_t = torch.from_numpy(self.X_np).float().to(self.device)
        y_t = torch.from_numpy(self.y_np).float().to(self.device)

        for it in range(outer_iters):
            # 1) build U_i from current surface definition
            U_list = self.surface.build_U_list()

            # 2) forward pass through param model to get m_i
            with torch.no_grad():
                m_t = self.model(X_t).cpu().numpy()

            # 3) phase 1: update D
            self._update_D(U_list, m_t)

            # 4) phase 2: update W
            self._update_W(X_t, y_t, U_list, epochs=5)

            # debug
            with torch.no_grad():
                m_t2 = self.model(X_t).cpu().numpy()
            # build h_i for train set and print loss
            h_np = np.zeros(self.n, dtype=np.float64)
            for i, (idxs, weights) in enumerate(U_list):
                h_np[i] = np.dot(weights, self.D[idxs])
            preds = m_t2 + h_np
            mse = 0.5 * np.mean((self.y_np - preds) ** 2)
            print(f"[outer {it+1}] training energy = {mse:.6f}")

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        For now, just do parametric prediction + kernel interpolation
        w.r.t. **training** desirabilities.
        """
        self.model.eval()
        with torch.no_grad():
            m_new = self.model(torch.from_numpy(X_new).float().to(self.device)).cpu().numpy()

        # for simplicity, we reuse the surface’s X to interpolate
        preds = []
        for x in X_new:
            # naive: do kernel interpolation on the fly (kernel version)
            # if you're using LLR, you'd need to call the LLR builder again with x
            neigh_idx = np.argsort(np.sum((self.surface.X - x) ** 2, axis=1))[: self.surface.K]
            neigh_idx = neigh_idx[neigh_idx != -1]
            w = self.surface._kernel_weights(x, neigh_idx)
            h = np.dot(w, self.D[neigh_idx])
            preds.append(h)
        preds = np.array(preds)
        return m_new + preds


# ----------------------------
# 5. DATA LOADING
# ----------------------------
def load_sub_sample() -> pd.DataFrame:
    candidate_paths = [
        "src/data/sub_sample.csv",
        "data/src/sub_sample.csv",
        "data/sub_sample.csv",
        "sub_sample.csv",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Could not find sub_sample.csv in expected locations.")


def prepare_xy(
    df: pd.DataFrame,
    target_cols: List[str] = ("log_price", "price", "SalePrice"),
) -> Tuple[np.ndarray, np.ndarray]:
    # find target
    target_name = None
    for c in target_cols:
        if c in df.columns:
            target_name = c
            break
    if target_name is None:
        # fallback: last column
        target_name = df.columns[-1]

    y_raw = df[target_name].to_numpy().astype(np.float64)
    # log if named "price"
    if target_name.lower() in ("price", "saleprice"):
        y = np.log(y_raw + 1e-9)
    else:
        y = y_raw

    # drop target from features
    X = df.drop(columns=[target_name]).to_numpy().astype(np.float64)

    # simple standardization
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-9
    X_std = (X - mean) / std

    return X_std, y


# ----------------------------
# 6. MAIN (example)
# ----------------------------
if __name__ == "__main__":
    df = load_sub_sample()
    X_np, y_np = prepare_xy(df)

    # choose which H(D, x) to use:
    # surface = KernelDesirabilitySurface(X_np, K=13, q=1.0)
    surface = LLRDesirabilitySurface(X_np, K=20, q=1.0)

    model = IntrinsicPriceNet(in_dim=X_np.shape[1])
    trainer = LMETrainer(
        X_np=X_np,
        y_np=y_np,
        model=model,
        surface=surface,
        reg_r=1e-2,
        device="cpu",
    )

    # run a few EM iterations
    trainer.fit(outer_iters=3)

    # example prediction on training itself
    preds = trainer.predict(X_np[:10])
    print("sample preds:", preds[:5])
