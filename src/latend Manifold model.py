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

# ========================================================
# 2. NON-PARAMETRIC SURFACES
# ========================================================
# ========================================================
# 2. NON-PARAMETRIC SURFACES (KD-tree version)
# ========================================================
 # make sure scikit-learn is installed

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

    # helper for prediction time
    def interpolate_one(self, x_new: np.ndarray, D: np.ndarray) -> float:
        """
        Simple kernel interpolation of D for a *new* point x_new
        using training X as support.
        """
        d2 = np.sum((self.X - x_new) ** 2, axis=1)
        neigh_idx = np.argsort(d2)[: self.K]
        w = np.exp(-self.q * d2[neigh_idx])
        w = w / (w.sum() + 1e-12)
        return float(np.dot(w, D[neigh_idx]))

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

    hist["iters"] = max_iter
    return x, hist

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
        model: IntrinsicPriceNet,
        reg_r: float = 1e-2,
        device: str = "cpu",
        zpid: Optional[np.ndarray] = None,
    ):
        self.X_np = X_np
        self.y_np = y_np
        self.n = X_np.shape[0]

        self.model = model.to(device)
        self.surface = surface
        self.model = model.to(device)
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
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        U_list: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 64,
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        dataset = TensorDataset(X_t, y_t)
        # IMPORTANT: keep shuffle=False so we can slice h_all by position
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

                start = end  # move window

    # ------------------ Outer loop ------------------
    def fit(self, outer_iters: int = 4):
        X_t = torch.from_numpy(self.X_param).float().to(self.device)
        y_t = torch.from_numpy(self.y).float().to(self.device)

        # we’ll log stuff in here
        em_losses: list[float] = []
        train_losses_per_iter: list[list[float]] = []

        # 1) pretrain like your notebook
        pretrain_losses = self._pretrain_model(X_t, y_t, epochs=3)

        best_loss = float("inf")

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
                h_np[i] = np.dot(self.D[idxs], weights)
            preds = m_np2 + h_np
            energy = 0.5 * np.mean((self.y - preds) ** 2)
            print(f"[outer {it+1}] training energy = {energy:.6f}")

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict on new rows.
        - X_new: parametric features (same as training X_param)
        - surf_X_new: surface features (e.g. lat/lon) for new rows;
                      if None, we fall back to using X_new in surface space.
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

        h_new = []
        for x in surf_X_new:
            h_new.append(self.surface.interpolate_one(x, self.D))
        h_new = np.array(h_new, dtype=np.float32)
        return m_new + h_new


# ========================================================
# 5. DATA LOADING VIA OUR PREPROCESSOR
# ========================================================
@dataclass
class HParams:
    test_size: float = 0.20
    random_state: int = 42
    em_iters: int = 6
    warmup_epochs: int = 5
    mstep_epochs: int = 3
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0 # (not used by default Adam)
    K: int = 25
    q: float = 0.20
    device: str = "cpu"
    verbose: bool = True
    max_train_rows: Optional[int] = None  # keep None to use all


def load_with_preprocessor() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Find the CSV, run Snowflake-like preprocessing, and return:
      - X_param  (for NN)
      - y        (log-price)
      - extras   (zpid, spatial, etc.)
    """
    candidate_paths = [
        "src/data/sub_sample.csv",
        "data/src/sub_sample.csv",
        "data/sub_sample.csv",
        "sub_sample.csv",
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError("Could not find sub_sample.csv in expected locations.")

    pre = DataPreprocessor(dataset_path=csv_path)
    raw_df = pre.load_data()               # raw CSV
    clean_df = pre.clean_and_engineer(raw_df)
    clean_df = clean_df.sample(n=10000, random_state = 42)
    X_param, y, feature_names, extras = pre.prepare_features(
        clean_df,
        target="LOG_PRICE",  # match your Snowflake LME which log-priced
        clip_ppsqft_quantile=0.995,
    )
    return X_param, y, extras

    # simple standardization
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-9
    X_std = (X - mean) / std

    return X_std, y

    return X_param_std, y, extras


# ========================================================
# 5b. SPATIAL TRAIN/TEST SPLIT
# ========================================================
def spatial_train_test_split(
    spatial: np.ndarray,
    test_frac: float = 0.2,
    min_cells: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple spatial split:
    - bin lat/lon into a grid
    - pick cells for test until we hit ~test_frac
    If everything collapses into one cell (common!), we fall back to random split.
    """
    rng = np.random.default_rng(random_state)

    n, d = spatial.shape
    if d < 2:
        # caller will probably fallback anyway
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(test_frac * n)
        return idx[cut:], idx[:cut]

    lat = spatial[:, 0]
    lon = spatial[:, 1]

    n_bins = 20
    lat_bins = np.linspace(lat.min(), lat.max(), n_bins + 1)
    lon_bins = np.linspace(lon.min(), lon.max(), n_bins + 1)

    lat_ids = np.digitize(lat, lat_bins) - 1
    lon_ids = np.digitize(lon, lon_bins) - 1

    cells: Dict[Tuple[int, int], list[int]] = {}
    for i, (la, lo) in enumerate(zip(lat_ids, lon_ids)):
        key = (la, lo)
        cells.setdefault(key, []).append(i)

    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)

    test_idx: list[int] = []
    target = test_frac * n

    for ck in cell_keys:
        test_idx.extend(cells[ck])
        if len(test_idx) >= target and len(test_idx) >= min_cells:
            break

    test_idx = np.array(test_idx, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_idx = np.where(mask)[0]

    # 👇 safety: if everything went to test, fallback to random
    if len(train_idx) == 0 or len(test_idx) == 0:
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int((1 - test_frac) * n)
        return idx[:cut], idx[cut:]

    return train_idx, test_idx



# ========================================================
# 6. MAIN SCRIPT
# ========================================================
if __name__ == "__main__":
    # 1) load + preprocess
    X_param, y, extras = load_with_preprocessor()
    print(f"parametric X shape: {X_param.shape}, y shape: {y.shape}")

    # 2) build a NaN-safe surface feature matrix
    spatial = extras.get("spatial")
    spatial_cols = extras.get("spatial_cols", [])

    if spatial is not None and spatial.size > 0:
        # prefer just LATITUDE/LONGITUDE if both exist
        if "LATITUDE" in spatial_cols and "LONGITUDE" in spatial_cols:
            lat_idx = spatial_cols.index("LATITUDE")
            lon_idx = spatial_cols.index("LONGITUDE")
            X_surface = spatial[:, [lat_idx, lon_idx]].astype(np.float32).copy()
        else:
            X_surface = spatial.astype(np.float32).copy()

        # drop columns that are entirely NaN
        keep_cols = ~np.isnan(X_surface).all(axis=0)
        X_surface = X_surface[:, keep_cols]

        # fill remaining NaNs with column means
        if np.isnan(X_surface).any():
            col_means = np.nanmean(X_surface, axis=0)
            rows, cols = np.where(np.isnan(X_surface))
            X_surface[rows, cols] = col_means[cols]
    else:
        # fallback to parametric features for distance
        mean = X_param.mean(axis=0, keepdims=True)
        std = X_param.std(axis=0, keepdims=True) + 1e-9
        X_surface_std = (X_param - mean) / std

    # 3) standardize surface features for KDTree
    surf_mean = X_surface.mean(axis=0, keepdims=True)
    surf_std = X_surface.std(axis=0, keepdims=True) + 1e-9
    X_surface_std = (X_surface - surf_mean) / surf_std
    print(f"[data] surface X shape (std): {X_surface_std.shape}")

    # 4) build surface (kernel or LLR)
    # surface = KernelDesirabilitySurface(X_surface_std, K=15, q=1.0)
    surface = LLRDesirabilitySurface(X_surface_std, K=15, q=1.0)

    # 5) build parametric model
    model = IntrinsicPriceNet(in_dim=X_param.shape[1])

    # fallback: random split
    idx_all = rng.permutation(n)
    cutoff = int((1.0 - hp.test_size) * n)
    train_idx = idx_all[:cutoff]
    test_idx = idx_all[cutoff:]

    X_tr = X_param[train_idx]
    X_te = X_param[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # use param X as surface
    surf_tr_raw = X_tr.astype(np.float32)
    surf_te_raw = X_te.astype(np.float32)

    m_tr = surf_tr_raw.mean(axis=0, keepdims=True)
    s_tr = surf_tr_raw.std(axis=0, keepdims=True) + 1e-9
    surf_tr = (surf_tr_raw - m_tr) / s_tr

    m_te = surf_te_raw.mean(axis=0, keepdims=True)
    s_te = surf_te_raw.std(axis=0, keepdims=True) + 1e-9
    surf_te = (surf_te_raw - m_te) / s_te

    print(f"[split] train size: {len(train_idx)}, test size: {len(test_idx)}")
    print(f"[data] surface train shape (std): {surf_tr.shape}")
    print(f"[data] surface test shape (std): {surf_te.shape}")

    return X_tr, X_te, y_tr, y_te, surf_tr, surf_te


# =========================================================
# 7. Metrics
# =========================================================
def compute_paper_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_rel = np.abs(y_pred - y_true) / np.clip(y_true, 1e-9, None)
    return {
        "within_5": float((abs_rel < 0.05).mean()),
        "within_10": float((abs_rel < 0.10).mean()),
        "within_15": float((abs_rel < 0.15).mean()),
        "median_abs_rel": float(np.median(abs_rel)),
    }


# =========================================================
# 8. main
# =========================================================
if __name__ == "__main__":
    hparams = LMEHyperparams(
        test_size=0.20,
        random_state=42,
        max_rows=None,          # set to 100_000 if you want a hard cap
        em_iters=3,
        warmup_epochs=5,
        mstep_epochs=5,
        batch_size=512,
        lr=3e-4,
        weight_decay=1e-4,
        patience=3,
        hidden_layers=(256, 128, 64, 32),
        dropout_prob=0.1,
        K=15,
        q=1.0,
        reg_r=1e-2,
        hidden_layers=(80, 40),
        dropout_prob=0.0,
        device="cpu",
        zpid=extras.get("zpid"),
    )

    # 1) load & preprocess
    X_param_all, y_all, extras = load_with_preprocessor(hp)
    if hp.verbose:
        print(f"[data] parametric X shape: {X_param_all.shape}, y shape: {y_all.shape}")

    # 2) surface matrix
    X_surface_all, surf_cols = build_surface_matrix(extras, X_param_all)

    # 3) spatial split
    # drop rows with NaN in surface (should already be filled, but guard anyway)
    mask_valid = ~np.isnan(X_surface_all).any(axis=1)
    if not mask_valid.all() and hp.verbose:
        print(f"[split] dropping {(~mask_valid).sum()} rows with NaN surface")
    X_param_all = X_param_all[mask_valid]
    y_all = y_all[mask_valid]
    X_surface_all = X_surface_all[mask_valid]

    train_mask, test_mask = spatial_train_test_split(X_surface_all, test_size=hp.test_size, seed=hp.random_state)
    if hp.verbose:
        print(f"[split] train size: {train_mask.sum()}, test size: {test_mask.sum()}")

    X_tr, X_te = X_param_all[train_mask], X_param_all[test_mask]
    y_tr, y_te = y_all[train_mask], y_all[test_mask]
    S_tr, S_te = X_surface_all[train_mask], X_surface_all[test_mask]

    # 4) lat/lon -> meters (only if we truly have lat/lon)
    have_ll = surf_cols[:2] == ["LATITUDE", "LONGITUDE"]
    if have_ll:
        R = 6_371_000.0  # meters
        lat_tr = np.deg2rad(S_tr[:, 0]); lon_tr = np.deg2rad(S_tr[:, 1])
        lat0 = lat_tr.mean()
        x_tr = R * (lon_tr - lon_tr.mean()) * np.cos(lat0)
        y_tr_m = R * (lat_tr - lat0)
        S_tr_m = np.c_[y_tr_m, x_tr].astype(np.float32)

        lat_te = np.deg2rad(S_te[:, 0]); lon_te = np.deg2rad(S_te[:, 1])
        x_te = R * (lon_te - lon_tr.mean()) * np.cos(lat0)  # anchor to TRAIN mean
        y_te_m = R * (lat_te - lat0)
        S_te_m = np.c_[y_te_m, x_te].astype(np.float32)

    # use TRAIN means to anchor both train and test (prevents train/test shift)
    lat0 = np.deg2rad(S_tr[:, 0]).mean()
    lon0 = np.deg2rad(S_tr[:, 1]).mean()

    # train → meters
    lat_tr = np.deg2rad(S_tr[:, 0])
    lon_tr = np.deg2rad(S_tr[:, 1])
    x_tr = R * (lon_tr - lon0) * np.cos(lat0)
    y_tr = R * (lat_tr - lat0)
    S_tr_m = np.c_[y_tr, x_tr].astype(np.float32)

    # 8) evaluate on the same 10k (since we sampled)
    y_pred_log = trainer.predict(
        X_new=X_param,
        surf_X_new=X_surface_std,
    )
    y_true_train = np.exp(y_train)
    y_pred_train = np.exp(y_pred_log_train)
    abs_rel_train = np.abs(y_pred_train - y_true_train) / y_true_train
    print("\n=== Paper-style metrics (TRAIN) ===")
    print(f"within 5%:  {(abs_rel_train < 0.05).mean():.4f}")
    print(f"within 10%: {(abs_rel_train < 0.10).mean():.4f}")
    print(f"within 15%: {(abs_rel_train < 0.15).mean():.4f}")
    print(f"median abs rel: {np.median(abs_rel_train):.4f}")

    if use_ppsqft:
        sqft_tr = np.clip(sqft_all[train_mask], 1.0, None)
        sqft_te = np.clip(sqft_all[test_mask], 1.0, None)
        # log-ppsqft = log_price - log(sqft)
        y_tr_pp = y_tr_raw - np.log(sqft_tr)
        y_te_pp = y_te_raw - np.log(sqft_te)
        # train-only clip in PPSQFT space
        pp_clip = np.quantile(np.exp(y_tr_pp), 0.995)
        y_tr = np.log(np.clip(np.exp(y_tr_pp), 0.0, pp_clip))
        y_te = np.log(np.clip(np.exp(y_te_pp), 0.0, pp_clip))
        extras["ppsqft_clip"] = float(pp_clip)
        # sample weights from PPSQFT distribution
        q50, q90 = np.quantile(np.exp(y_tr), [0.50, 0.90])
        w_tr = np.ones_like(y_tr, dtype=np.float32)
        w_tr[np.exp(y_tr) >= q50] = 1.2
        w_tr[np.exp(y_tr) >= q90] = 1.6
    else:
        # LOG_PRICE target fallback
        y_tr, y_te = y_tr_raw.copy(), y_te_raw.copy()
        q50, q90 = np.quantile(np.exp(y_tr), [0.50, 0.90])
        w_tr = np.ones_like(y_tr, dtype=np.float32)
        w_tr[np.exp(y_tr) >= q50] = 1.2
        w_tr[np.exp(y_tr) >= q90] = 1.6
        sqft_tr = np.ones_like(y_tr)
        sqft_te = np.ones_like(y_te)

    # 5) Build model + trainer
    model = IntrinsicPriceNet(in_dim=X_tr_std.shape[1])
    trainer = LMETrainer(
        X_param=X_tr_std,
        y=y_tr,
        surface=surface,
        model=model,
        reg_r=1e-1,
        device=hp.device,
    )

    # 6) Train (pretrain + EM)
    history = trainer.fit(
        outer_iters=hp.em_iters,
        warmup_epochs=hp.warmup_epochs,
        mstep_epochs=hp.mstep_epochs,
        batch_size=hp.batch_size,
        lr=hp.lr,
    )

    # 7) Predictions (log-space)
    y_pred_log_tr = trainer.predict(X_tr_std, surf_tr)
    y_pred_log_te = trainer.predict(X_te_std, surf_te)

    # Guard against overflow when exponentiating
    y_pred_log_tr = np.clip(y_pred_log_tr, y_tr.min() - 1.0, y_tr.max() + 1.0)
    y_pred_log_te = np.clip(y_pred_log_te, y_te.min() - 1.0, y_te.max() + 1.0)

    # 8) Paper-style metrics (TRAIN / TEST)
    def summarize(name: str, y_log_true: np.ndarray, y_log_pred: np.ndarray):
        y_true = np.exp(y_log_true)
        y_pred = np.exp(y_log_pred)
        abs_rel = np.abs(y_pred - y_true) / y_true
        within_5  = (abs_rel < 0.05).mean()
        within_10 = (abs_rel < 0.10).mean()
        within_15 = (abs_rel < 0.15).mean()
        med_err   = np.median(abs_rel)
        print(f"\n=== Paper-style metrics ({name}) ===")
        print(f"within 5%:  {within_5:.4f}")
        print(f"within 10%: {within_10:.4f}")
        print(f"within 15%: {within_15:.4f}")
        print(f"median abs rel: {med_err:.4f}")
        return y_true, y_pred, abs_rel

    # 9) plots like notebook — save to files (headless friendly)

    # 9a) EM loss curve
    if "em_losses" in history and history["em_losses"]:
        plt.figure(figsize=(6, 4))
        plt.plot(history["em_losses"], marker="o")
        plt.title("EM iteration loss")
        plt.xlabel("EM iteration")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("em_loss.png", dpi=150)

    # 9b) M-step losses per EM
    if "train_losses_per_iter" in history and history["train_losses_per_iter"]:
        plt.figure(figsize=(6, 4))
        for i, losses in enumerate(history["train_losses_per_iter"]):
            plt.plot(losses, label=f"EM {i+1}")
        plt.title("M-step (NN) losses per EM")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("mstep_losses.png", dpi=150)

    # 9c) predicted vs true
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=4)
    mn, mx = y_true.min(), y_true.max()
    plt.plot([mn, mx], [mn, mx], color="red")
    plt.title("Predicted vs True Price")
    plt.xlabel("True price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig("pred_vs_true.png", dpi=150)

    # 9d) abs relative error hist
    plt.figure(figsize=(6, 4))
    plt.hist(abs_rel, bins=50)
    plt.title("Absolute Relative Error")
    plt.xlabel("abs_rel")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("abs_rel_hist.png", dpi=150)

    # 9e) optional desirability map if we have lat/lon
    if extras.get("spatial") is not None and "LATITUDE" in spatial_cols and "LONGITUDE" in spatial_cols:
        lat_idx = spatial_cols.index("LATITUDE")
        lon_idx = spatial_cols.index("LONGITUDE")
        coords = extras["spatial"]
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(
            coords[:, lon_idx],
            coords[:, lat_idx],
            c=trainer.D,
            s=5,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(sc, label="desirability (D)")
        plt.title("Learned desirability field")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig("desirability.png", dpi=150)

