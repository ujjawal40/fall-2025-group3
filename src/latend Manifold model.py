# lme_model.py
# ============================================================
# Latent Manifold Estimation (paper-aligned) with:
# - PPSQFT target (preferred) or LOG_PRICE (fallback)
# - Spatial splits + inner spatial VAL for early stopping
# - KDTree surface in meters w/ adaptive RBF bandwidth
# - Weighted E- and M- steps, Laplacian smoothing on D (optional)
# - BatchNorm+Dropout intrinsic net, warmup+cosine LR
# ============================================================
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from data_preprocessor_LME import DataPreprocessor


# ------------------------- Hyperparams -------------------------
@dataclass
class HParams:
    # Data / splits
    test_size: float = 0.20
    inner_val_frac: float = 0.12        # VAL proportion INSIDE TRAIN (spatial)
    random_state: int = 42
    max_rows: Optional[int] = None

    # EM loop
    em_iters: int = 6
    warmup_epochs: int = 8
    mstep_epochs: int = 12
    patience: int = 10                  # M-step early stop on VAL (full energy)

    # Optimizer
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 5e-4

    # Surface / neighbors
    K: int = 40                         # neighbors for desirability interpolation
    q: float = 1.0                      # base sharpness (scaled by adaptive sigma)
    K_lap: Optional[int] = None         # neighbors for Laplacian; None => K
    lap_lambda: float = 0.02            # λ for Laplacian smoothing (0 disables)

    # Regularization on D (ridge)
    reg_r: float = 5e-2

    # Intrinsic model
    hidden_layers: Tuple[int, ...] = (256, 128, 64, 32)
    dropout_prob: float = 0.25

    # Compute
    device: str = "cpu"

    # CG (E-step) stopping
    cg_rel_tol: float = 1e-5
    cg_max_iter: int = 300
    cg_patience: int = 10

    # Logging
    verbose: bool = True


# --------------------- Intrinsic Price Network ---------------------
class IntrinsicPriceNet(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout_prob: float = 0.0):
        super().__init__()
        mods: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            mods.append(nn.Linear(last, h))
            mods.append(nn.BatchNorm1d(h))     # BatchNorm (important)
            mods.append(nn.ReLU())
            if dropout_prob > 0:
                mods.append(nn.Dropout(dropout_prob))
            last = h
        mods.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------- KD-tree surface (meters) ---------------------
def latlon_to_xy_meters(latlon: np.ndarray, lat0_rad: float) -> np.ndarray:
    """
    latlon: (n,2) with columns [LAT, LON] in degrees
    lat0_rad: reference latitude (radians), use TRAIN mean
    returns (n,2) [y, x] meters
    """
    R = 6_371_000.0
    lat = np.deg2rad(latlon[:, 0])
    lon = np.deg2rad(latlon[:, 1])
    x = R * (lon - lon.mean()) * np.cos(lat0_rad)
    y = R * (lat - lat0_rad)
    return np.c_[y, x].astype(np.float32)


class KernelSurface:
    """
    KD-tree surface with adaptive bandwidth:
    - For each i, sigma_i^2 = median(d2 to its K neighbors) + eps
    - weights_ij ∝ exp( - d2_ij / (2 * sigma_i^2) )
    """
    def __init__(self, S_train_std: np.ndarray, K: int = 40, q: float = 1.0):
        self.S = S_train_std.astype(np.float32)
        self.n, self.d = self.S.shape
        self.K = int(K)
        self.q = float(q)
        self.tree = KDTree(self.S, leaf_size=40)

    def _neighbors(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        # return neighbor indices and squared distances (drop self)
        d, ind = self.tree.query(self.S[i:i+1], k=self.K + 1)
        ind = ind[0]
        d = d[0]
        mask = ind != i
        return ind[mask][:self.K], (d[mask][:self.K] ** 2)

    def _adapt_weights(self, d2_i: np.ndarray) -> np.ndarray:
        sigma2 = float(np.median(d2_i) + 1e-12)
        w = np.exp(- self.q * d2_i / (2.0 * sigma2))
        s = w.sum() + 1e-12
        return w / s

    def build_U_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        U: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n):
            idxs, d2 = self._neighbors(i)
            w = self._adapt_weights(d2)
            U.append((idxs, w.astype(np.float32)))
        return U

    def interpolate_one(self, x_new_std: np.ndarray, D: np.ndarray) -> float:
        # query neighbors in TRAIN space
        d, ind = self.tree.query(x_new_std.reshape(1, -1), k=self.K)
        ind = ind[0]
        d2 = (d[0] ** 2)
        w = self._adapt_weights(d2)
        return float(np.dot(w, D[ind].astype(np.float64)))


# --------------------- Graph Laplacian (optional) ---------------------
class LaplacianOp:
    """Matrix-free Laplacian-vector multiply using the same KDTree."""
    def __init__(self, surf: KernelSurface, K_lap: Optional[int] = None, q: float = 1.0):
        self.S = surf.S
        self.tree = surf.tree
        self.n = surf.n
        self.K = int(K_lap or surf.K)
        self.q = float(q)
        # Prebuild neighbor lists for Laplacian
        self.lap_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n):
            d, ind = self.tree.query(self.S[i:i+1], k=self.K + 1)
            ind = ind[0]; d = d[0]
            mask = ind != i
            idxs = ind[mask][:self.K]
            d2 = (d[mask][:self.K] ** 2)
            # symmetric-ish weights (normalized)
            sigma2 = float(np.median(d2) + 1e-12)
            w = np.exp(- self.q * d2 / (2.0 * sigma2))
            w = w / (w.sum() + 1e-12)
            self.lap_list.append((idxs.astype(np.int32), w.astype(np.float32)))

    def apply(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v, dtype=np.float64)
        for i, (idxs, w) in enumerate(self.lap_list):
            vi = v[i]
            out[i] += np.sum(w * (vi - v[idxs]))
        return out


# --------------------- Conjugate Gradient (QP) ---------------------
def cg_solve_qp(apply_A, b: np.ndarray, x0: Optional[np.ndarray] = None,
                rel_tol: float = 1e-5, max_iter: int = 300, patience: int = 10):
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64) if x0 is None else x0.astype(np.float64).copy()
    r = b - apply_A(x)
    p = r.copy()
    r0 = np.linalg.norm(r) + 1e-12
    rsold = np.dot(r, r)

    def obj(xv: np.ndarray) -> float:
        Ax = apply_A(xv)
        return 0.5 * float(np.dot(xv, Ax)) - float(np.dot(b, xv))

    best_obj = np.inf
    stale = 0

    for k in range(max_iter):
        Ap = apply_A(p)
        denom = np.dot(p, Ap) + 1e-18
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)

        rel_res = float(np.sqrt(rsnew) / r0)
        J = obj(x)
        if J + 1e-12 < best_obj:
            best_obj, stale = J, 0
        else:
            stale += 1

        if rel_res <= rel_tol or stale >= patience:
            return x
        p = r + (rsnew / (rsold + 1e-18)) * p
        rsold = rsnew
    return x


# --------------------- Trainer ---------------------
class LMETrainer:
    def __init__(
        self,
        X_tr: np.ndarray,               # standardized param (TRAIN)
        y_tr: np.ndarray,               # TRAIN target in log space (log_ppsqft or log_price)
        w_tr: np.ndarray,               # TRAIN sample weights (>=0)
        S_tr_std: np.ndarray,           # TRAIN surface std
        S_val_std: np.ndarray,          # VAL surface std
        model: IntrinsicPriceNet,
        surf: KernelSurface,
        lap: Optional[LaplacianOp],
        hp: HParams,
    ):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.w_tr = w_tr.astype(np.float32)
        self.S_tr_std = S_tr_std
        self.S_val_std = S_val_std
        self.model = model.to(hp.device)
        self.surf = surf
        self.lap = lap
        self.hp = hp

        self.n = X_tr.shape[0]
        self.D = np.zeros(self.n, dtype=np.float64)  # desirabilities

        # Precompute neighbor weights for TRAIN once
        self.U_list = self.surf.build_U_list()

    # ---------- utilities ----------
    def _compute_h(self, D: np.ndarray, U_list: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        h = np.zeros(self.n, dtype=np.float64)
        for i, (idxs, w) in enumerate(U_list):
            h[i] = float(np.dot(D[idxs], w))
        return h

    def _h_on_val(self, D: np.ndarray) -> np.ndarray:
        # interpolate D (TRAIN) on VAL coordinates using TRAIN KDTree
        h_val = [self.surf.interpolate_one(sv, D) for sv in self.S_val_std]
        return np.array(h_val, dtype=np.float64)

    def _pretrain(self) -> List[float]:
        if self.hp.verbose:
            print(f"[pretrain] starting for {self.hp.warmup_epochs} epochs...")
        ds = TensorDataset(
            torch.from_numpy(self.X_tr).float(),
            torch.from_numpy(self.y_tr).float(),
            torch.from_numpy(self.w_tr).float(),
        )
        loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
        loss_fn = nn.MSELoss(reduction="none")
        losses: List[float] = []

        for ep in range(self.hp.warmup_epochs):
            running, wsum = 0.0, 0.0
            for xb, yb, wb in loader:
                xb = xb.to(self.hp.device); yb = yb.to(self.hp.device); wb = wb.to(self.hp.device)
                pred = self.model(xb)
                per = loss_fn(pred, yb) * wb
                denom = wb.sum().clamp_min(1e-6)
                loss = per.sum() / denom
                opt.zero_grad(); loss.backward(); opt.step()
                running += per.sum().item(); wsum += float(denom)
            avg = running / max(1e-6, wsum)
            losses.append(float(avg))
            if self.hp.verbose:
                print(f"[pretrain] epoch {ep+1}/{self.hp.warmup_epochs} - loss: {avg:.4f}")
        return losses

    # ---------- E-step: solve D with CG (weighted + Laplacian + gauge) ----------
    def _update_D(self):
        y = self.y_tr
        w = self.w_tr.astype(np.float64)
        r = float(self.hp.reg_r)

        X_t = torch.from_numpy(self.X_tr).float().to(self.hp.device)
        with torch.no_grad():
            m_np = self.model(X_t).cpu().numpy().astype(np.float64)

        # b = sum_i w_i (y_i - m_i) U_i
        b = np.zeros(self.n, dtype=np.float64)
        for i, (idxs, weights) in enumerate(self.U_list):
            b[idxs] += float(w[i]) * (y[i] - m_np[i]) * weights.astype(np.float64)

        # A v = r v + sum_i w_i (U_i U_i^T) v + λ L v
        lap = self.lap
        lam = float(self.hp.lap_lambda) if lap is not None else 0.0

        def apply_A(v: np.ndarray) -> np.ndarray:
            out = r * v
            for i, (idxs, weights) in enumerate(self.U_list):
                coeff = float(np.dot(v[idxs], weights.astype(np.float64)))
                out[idxs] += w[i] * coeff * weights.astype(np.float64)
            if lam > 0.0:
                out += lam * lap.apply(v)
            return out

        D_new = cg_solve_qp(
            apply_A, b, x0=self.D,
            rel_tol=self.hp.cg_rel_tol,
            max_iter=self.hp.cg_max_iter,
            patience=self.hp.cg_patience
        )
        # gauge: mean-zero to keep m vs h identifiable
        D_new -= D_new.mean()
        self.D = D_new

    # ---------- M-step: train W on residuals with spatial VAL early-stop ----------
    def _mstep(self, X_val: np.ndarray, y_val: np.ndarray) -> List[float]:
        # Precompute h on TRAIN and VAL with current D
        h_tr = self._compute_h(self.D, self.U_list).astype(np.float32)
        h_val = self._h_on_val(self.D).astype(np.float32)

        ds_tr = TensorDataset(
            torch.from_numpy(self.X_tr).float(),
            torch.from_numpy(self.y_tr).float(),
            torch.from_numpy(h_tr).float(),
            torch.from_numpy(self.w_tr).float()
        )
        loader = DataLoader(ds_tr, batch_size=self.hp.batch_size, shuffle=False)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
        loss_fn = nn.MSELoss(reduction="none")

        # simple warmup+cosine across mstep_epochs
        def lr_factor(epoch: int) -> float:
            warm = min(1.0, (epoch + 1) / max(1, self.hp.warmup_epochs))
            prog = max(0.0, (epoch + 1 - self.hp.warmup_epochs) / max(1, self.hp.mstep_epochs - self.hp.warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return warm * cosine

        best_val = float("inf")
        wait = 0
        losses: List[float] = []
        best_state = None

        X_val_t = torch.from_numpy(X_val).float().to(self.hp.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.hp.device)
        h_val_t = torch.from_numpy(h_val).float().to(self.hp.device)

        for ep in range(self.hp.mstep_epochs):
            for g in opt.param_groups:
                g["lr"] = self.hp.lr * lr_factor(ep)

            running, wsum = 0.0, 0.0
            self.model.train()
            for xb, yb, hb, wb in loader:
                xb = xb.to(self.hp.device); yb = yb.to(self.hp.device)
                hb = hb.to(self.hp.device); wb = wb.to(self.hp.device)
                # residual target
                y_res = yb - hb
                pred_m = self.model(xb)
                per = loss_fn(pred_m, y_res) * wb
                denom = wb.sum().clamp_min(1e-6)
                loss = per.sum() / denom
                opt.zero_grad(); loss.backward(); opt.step()
                running += per.sum().item(); wsum += float(denom)

            avg = running / max(1e-6, wsum)
            losses.append(float(avg))
            if self.hp.verbose:
                print(f"[m-step] epoch {ep+1}/{self.hp.mstep_epochs} - loss: {avg:.6f}")

            # ---- VAL check: full energy on VAL (y_val - (m_val + h_val)) ----
            self.model.eval()
            with torch.no_grad():
                m_val = self.model(X_val_t)
                full_res = y_val_t - (m_val + h_val_t)
                val_energy = 0.5 * torch.mean(full_res ** 2).item()

            if val_energy + 1e-6 < best_val:
                best_val, wait = val_energy, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.hp.patience:
                    if self.hp.verbose:
                        print("[m-step] early stopping (VAL)")
                    break

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return losses

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        history = {"pretrain_losses": [], "em_losses": [], "mstep_losses_per_iter": []}

        history["pretrain_losses"] = self._pretrain()

        for it in range(self.hp.em_iters):
            if self.hp.verbose:
                print(f"[em] iteration {it+1}/{self.hp.em_iters}")

            # E-step: D
            self._update_D()

            # M-step: W with inner VAL early-stop
            m_losses = self._mstep(X_val, y_val)
            history["mstep_losses_per_iter"].append(m_losses)

            # report train energy after this EM iteration
            with torch.no_grad():
                m_tr = self.model(torch.from_numpy(self.X_tr).float().to(self.hp.device)).cpu().numpy()
            h_tr = self._compute_h(self.D, self.U_list)
            energy = 0.5 * np.mean((self.y_tr - (m_tr + h_tr)) ** 2)
            history["em_losses"].append(float(energy))
            if self.hp.verbose:
                print(f"[em]   loss: {energy:.6f}  |  improvement: "
                      f"{(history['em_losses'][-2] - energy):.6f}" if len(history["em_losses"]) > 1 else
                      f"[em]   loss: {energy:.6f}  |  improvement: inf")

        return history

    # Predict on any standardized surface coords (uses TRAIN KDTree)
    def predict(self, X_std: np.ndarray, S_std: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            m = self.model(torch.from_numpy(X_std).float().to(self.hp.device)).cpu().numpy()
        h = np.array([self.surf.interpolate_one(s, self.D) for s in S_std], dtype=np.float64)
        return m + h


# --------------------- Data + splits + metrics ---------------------
def load_with_preprocessor(hp: HParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    # Locate CSV
    for p in ["src/data/sub_sample.csv", "data/src/sub_sample.csv", "data/sub_sample.csv", "sub_sample.csv"]:
        if os.path.exists(p):
            csv_path = p; break
    else:
        raise FileNotFoundError("Could not find sub_sample.csv in expected locations.")

    pre = DataPreprocessor(dataset_path=csv_path)
    raw_df = pre.load_data()
    clean_df = pre.clean_and_engineer(raw_df)
    if hp.max_rows is not None and len(clean_df) > hp.max_rows:
        clean_df = clean_df.sample(n=hp.max_rows, random_state=hp.random_state)

    X_raw, y_log_price, feature_names, extras = pre.prepare_features(
        clean_df, target="LOG_PRICE", clip_ppsqft_quantile=0.995
    )

    # Try to get raw SQFT from features if not explicitly in extras
    sqft_idx = None
    sqft_names = {"SQFT", "SQUARE_FEET", "LIVING_AREA", "TOTAL_SQFT", "FINISHED_SQ_FT", "AREA_SQFT"}
    if "feature_names" in extras:
        fn = np.array(extras["feature_names"])
    else:
        fn = np.array(feature_names)
    for nm in fn:
        if str(nm).upper() in sqft_names:
            sqft_idx = int(np.where(fn == nm)[0][0]); break

    SQFT_raw = None
    if sqft_idx is not None:
        SQFT_raw = X_raw[:, sqft_idx].astype(np.float64)
        extras["SQFT_RAW"] = SQFT_raw

    # Standardize param features (keep means for test)
    x_mean = X_raw.mean(axis=0, keepdims=True)
    x_std = X_raw.std(axis=0, keepdims=True) + 1e-9
    X_std = (X_raw - x_mean) / x_std

    extras["x_mean"] = x_mean; extras["x_std"] = x_std; extras["feature_names"] = feature_names
    return X_std.astype(np.float32), y_log_price.astype(np.float32), X_raw.astype(np.float32), extras


def build_surface_and_splits(
    X_std: np.ndarray,
    X_raw: np.ndarray,
    y_log_price: np.ndarray,
    extras: Dict[str, Any],
    hp: HParams
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, KernelSurface, Optional[LaplacianOp],
    Dict[str, Any]
]:
    # Prefer LAT/LON
    spatial = extras.get("spatial")
    spatial_cols = extras.get("spatial_cols", [])
    if spatial is None or spatial.size == 0 or "LATITUDE" not in spatial_cols or "LONGITUDE" not in spatial_cols:
        raise RuntimeError("LATITUDE / LONGITUDE not found in extras['spatial']; can't do spatial split.")

    lat_idx = spatial_cols.index("LATITUDE")
    lon_idx = spatial_cols.index("LONGITUDE")
    S_all_deg = spatial[:, [lat_idx, lon_idx]].astype(np.float32)

    # Drop rows with NaN lat/lon
    valid = ~np.isnan(S_all_deg).any(axis=1)
    if not valid.all() and hp.verbose:
        print(f"[split] dropping {(~valid).sum()} rows with NaN lat/lon")
    X_std = X_std[valid]; X_raw = X_raw[valid]
    y_log_price = y_log_price[valid]; S_all_deg = S_all_deg[valid]

    # Spatial TRAIN/TEST split by coarse grid cells
    train_mask, test_mask = spatial_grid_split(S_all_deg, test_size=hp.test_size, seed=hp.random_state)
    if hp.verbose:
        print(f"[split] train size: {train_mask.sum()}, test size: {test_mask.sum()}")

    X_tr, X_te = X_std[train_mask], X_std[test_mask]
    Xr_tr, Xr_te = X_raw[train_mask], X_raw[test_mask]
    y_tr_lp, y_te_lp = y_log_price[train_mask], y_log_price[test_mask]
    S_tr_deg, S_te_deg = S_all_deg[train_mask], S_all_deg[test_mask]

    # meters projection using TRAIN reference latitude
    lat0_rad = float(np.deg2rad(S_tr_deg[:, 0]).mean())
    S_tr_m = latlon_to_xy_meters(S_tr_deg, lat0_rad)
    S_te_m = latlon_to_xy_meters(S_te_deg, lat0_rad)

    # standardize surface by TRAIN stats
    S_tr_std, S_te_std, s_mean, s_std = standardize_by_train(S_tr_m, S_te_m)

    # Build surface + Laplacian operator (on TRAIN only)
    surf = KernelSurface(S_tr_std, K=hp.K, q=hp.q)
    lap = LaplacianOp(surf, K_lap=hp.K_lap or hp.K, q=hp.q) if hp.lap_lambda > 0 else None

    meta = {"lat0_rad": lat0_rad, "s_mean": s_mean, "s_std": s_std, "spatial_cols": ["LATITUDE", "LONGITUDE"]}

    return X_tr, X_te, Xr_tr, Xr_te, y_tr_lp, y_te_lp, S_tr_std, S_te_std, train_mask, test_mask, surf, lap, meta


def spatial_grid_split(S_deg: np.ndarray, test_size=0.2, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    n = S_deg.shape[0]
    lat, lon = S_deg[:, 0], S_deg[:, 1]
    lat_bins = np.linspace(lat.min(), lat.max(), 101)
    lon_bins = np.linspace(lon.min(), lon.max(), 101)
    lat_id = np.digitize(lat, lat_bins) - 1
    lon_id = np.digitize(lon, lon_bins) - 1
    cell = lat_id * 100 + lon_id
    uniq = np.unique(cell)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cut = int(test_size * len(uniq))
    test_cells = set(uniq[:cut])
    train_mask = ~np.isin(cell, list(test_cells))
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        idx = rng.permutation(n)
        cut = int((1 - test_size) * n)
        train_mask = np.zeros(n, dtype=bool); train_mask[idx[:cut]] = True
        test_mask = ~train_mask
    return train_mask, test_mask


def standardize_by_train(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-9
    return (X_tr - mean) / std, (X_te - mean) / std, mean, std


def make_inner_spatial_val(S_tr_deg_like: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial VAL split inside TRAIN."""
    n = S_tr_deg_like.shape[0]
    lat, lon = S_tr_deg_like[:, 0], S_tr_deg_like[:, 1]
    lat_bins = np.linspace(lat.min(), lat.max(), 61)
    lon_bins = np.linspace(lon.min(), lon.max(), 61)
    lat_id = np.digitize(lat, lat_bins) - 1
    lon_id = np.digitize(lon, lon_bins) - 1
    cell = lat_id * 60 + lon_id
    uniq = np.unique(cell)
    rng = np.random.RandomState(seed + 123)
    rng.shuffle(uniq)
    cut = max(1, int(frac * len(uniq)))
    val_cells = set(uniq[:cut])
    val_mask = np.isin(cell, list(val_cells))
    train_inner = ~val_mask
    return train_inner, val_mask


def to_ppsqft_from_logs(y_log_price: np.ndarray, sqft: np.ndarray) -> np.ndarray:
    price = np.exp(np.clip(y_log_price, -20.0, 20.0))
    sqft_safe = np.clip(sqft.astype(np.float64), 1.0, 1e12)
    return price / sqft_safe


def price_metrics_from_logs(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> Dict[str, float]:
    clip = 20.0
    y_true = np.exp(np.clip(y_log_true, -clip, clip))
    y_pred = np.exp(np.clip(y_log_pred, -clip, clip))
    abs_rel = np.abs(y_pred - y_true) / (y_true + 1e-12)
    return {
        "within_5": float((abs_rel < 0.05).mean()),
        "within_10": float((abs_rel < 0.10).mean()),
        "within_15": float((abs_rel < 0.15).mean()),
        "median_abs_rel": float(np.median(abs_rel)),
    }


# --------------------- Main ---------------------
if __name__ == "__main__":
    import math

    hp = HParams()  # use defaults as discussed

    # 1) load & preprocess
    X_std_all, y_log_price_all, X_raw_all, extras = load_with_preprocessor(hp)
    if hp.verbose:
        print(f"[data] parametric X shape: {X_std_all.shape}, y shape: {y_log_price_all.shape}")

    # 2) spatial split & surface
    (X_tr, X_te, Xr_tr, Xr_te, y_tr_lp, y_te_lp,
     S_tr_std, S_te_std, train_mask, test_mask, surf, lap, meta) = build_surface_and_splits(
        X_std_all, X_raw_all, y_log_price_all, extras, hp
    )

    # 3) choose target: PPSQFT (preferred) or LOG_PRICE
    SQFT_tr = extras.get("SQFT_RAW", None)
    if SQFT_tr is None:
        # Try to recover SQFT from raw TRAIN subset by name index if available
        pass  # we already tried above; nothing else to do here

    use_ppsqft = SQFT_tr is not None
    if use_ppsqft:
        # extract SQFT for train/test from global mask
        SQFT_all = extras["SQFT_RAW"]
        SQFT_tr = SQFT_all[train_mask]; SQFT_te = SQFT_all[test_mask]

        # transform to log_ppsqft
        sqft_tr_safe = np.clip(SQFT_tr, 1.0, 1e12)
        sqft_te_safe = np.clip(SQFT_te, 1.0, 1e12)
        y_tr = (y_tr_lp - np.log(sqft_tr_safe)).astype(np.float32)
        y_te = (y_te_lp - np.log(sqft_te_safe)).astype(np.float32)

        # sample weights by PPSQFT quantiles (linear space)
        ppsqft_tr = to_ppsqft_from_logs(y_tr_lp, SQFT_tr)
        q50, q90 = np.quantile(ppsqft_tr, [0.5, 0.9])
        w_tr = np.ones_like(ppsqft_tr, dtype=np.float32)
        w_tr[ppsqft_tr >= q50] *= 1.2
        w_tr[ppsqft_tr >= q90] *= 1.6
        target_name = "PPSQFT"
    else:
        print("[warn] SQFT missing/unusable; using LOG_PRICE target.")
        y_tr = y_tr_lp.astype(np.float32)
        y_te = y_te_lp.astype(np.float32)
        w_tr = np.ones_like(y_tr, dtype=np.float32)
        target_name = "LOG_PRICE"

    # 4) INNER spatial VAL split inside TRAIN (for M-step early stop)
    #    We re-use original degrees coords from train_mask subset to pick cells.
    #    Recover TRAIN lat/lon degrees for split: inverse standardization not needed for split.
    #    (We approximate using the standardized meters back to "like" degrees layout by reusing the same mask on raw spatial.)
    spatial = extras.get("spatial"); spatial_cols = extras.get("spatial_cols", [])
    lat_idx = spatial_cols.index("LATITUDE"); lon_idx = spatial_cols.index("LONGITUDE")
    S_all_deg = spatial[:, [lat_idx, lon_idx]]
    S_tr_deg = S_all_deg[train_mask]
    tr_inner_mask, val_mask = make_inner_spatial_val(S_tr_deg, frac=hp.inner_val_frac, seed=hp.random_state)

    X_tr_in, X_val = X_tr[tr_inner_mask], X_tr[val_mask]
    y_tr_in, y_val = y_tr[tr_inner_mask], y_tr[val_mask]
    w_tr_in = w_tr[tr_inner_mask]

    # Corresponding surface rows (already std): TRAIN uses S_tr_std
    S_tr_in_std, S_val_std = S_tr_std[tr_inner_mask], S_tr_std[val_mask]

    # 5) model + trainer
    model = IntrinsicPriceNet(in_dim=X_tr.shape[1], hidden=hp.hidden_layers, dropout_prob=hp.dropout_prob)
    surf_in = KernelSurface(S_tr_in_std, K=hp.K, q=hp.q)
    lap_in = LaplacianOp(surf_in, K_lap=hp.K_lap or hp.K, q=hp.q) if hp.lap_lambda > 0 else None

    trainer = LMETrainer(
        X_tr=X_tr_in, y_tr=y_tr_in, w_tr=w_tr_in,
        S_tr_std=S_tr_in_std, S_val_std=S_val_std,
        model=model, surf=KernelSurface(S_tr_in_std, K=hp.K, q=hp.q),
        lap=LaplacianOp(KernelSurface(S_tr_in_std, K=hp.K, q=hp.q), K_lap=hp.K_lap or hp.K, q=hp.q) if hp.lap_lambda > 0 else None,
        hp=hp
    )

    # 6) fit
    history = trainer.fit(X_val=X_val, y_val=y_val)

    # 7) evaluate on full TRAIN (outer) and TEST
    # Build a surface object on FULL TRAIN (std) for inference
    surf_full = KernelSurface(S_tr_std, K=hp.K, q=hp.q)
    trainer_full = trainer  # reuse model and D trained on inner-train; D length matches inner-train only
    # For fairness, recompute D for FULL TRAIN with current model and then predict
    # (Cheap alternative: predict using inner-train D; here we solve once on FULL TRAIN)
    # ---- Refit D on full TRAIN (single E-step) ----
    # Assemble a temp trainer for FULL TRAIN E-step with same model weights
    tmp = LMETrainer(
        X_tr=X_tr, y_tr=y_tr, w_tr=w_tr,
        S_tr_std=S_tr_std, S_val_std=S_val_std[:1],  # dummy val
        model=model, surf=surf_full, lap=LaplacianOp(surf_full, K_lap=hp.K_lap or hp.K, q=hp.q) if hp.lap_lambda > 0 else None,
        hp=hp
    )
    tmp.U_list = surf_full.build_U_list()
    tmp._update_D()  # one E-step at the end

    # TRAIN predictions (full train)
    y_pred_tr = tmp.predict(X_tr, S_tr_std).astype(np.float64)
    # TEST predictions: standardize test surface already done as S_te_std
    y_pred_te = tmp.predict(X_te, S_te_std).astype(np.float64)

    # Metrics (log space -> price or ppsqft space absolute-relative)
    tr_metrics = price_metrics_from_logs(y_tr, y_pred_tr)
    te_metrics = price_metrics_from_logs(y_te, y_pred_te)

    print(f"\n=== Paper-style metrics (TRAIN, {target_name}) ===")
    print(f"within 5%:  {tr_metrics['within_5']:.4f}")
    print(f"within 10%: {tr_metrics['within_10']:.4f}")
    print(f"within 15%: {tr_metrics['within_15']:.4f}")
    print(f"median abs rel: {tr_metrics['median_abs_rel']:.4f}")

    print(f"\n=== Paper-style metrics (TEST, {target_name}) ===")
    print(f"within 5%:  {te_metrics['within_5']:.4f}")
    print(f"within 10%: {te_metrics['within_10']:.4f}")
    print(f"within 15%: {te_metrics['within_15']:.4f}")
    print(f"median abs rel: {te_metrics['median_abs_rel']:.4f}")

    # 8) plots saved to files
    if history.get("em_losses"):
        plt.figure(figsize=(6, 4))
        plt.plot(history["em_losses"], marker="o")
        plt.title("EM iteration loss (train energy on inner-train)")
        plt.xlabel("EM iteration"); plt.ylabel("loss")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("em_loss.png", dpi=150)

    if history.get("mstep_losses_per_iter"):
        plt.figure(figsize=(6, 4))
        for i, losses in enumerate(history["mstep_losses_per_iter"]):
            plt.plot(losses, label=f"EM {i+1}")
        plt.title("M-step (NN) losses per EM (inner-train)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("mstep_losses.png", dpi=150)

    # TEST scatter
    clip = 20.0
    y_true_te = np.exp(np.clip(y_te, -clip, clip))
    y_pred_te_lin = np.exp(np.clip(y_pred_te, -clip, clip))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_te, y_pred_te_lin, s=4)
    mn, mx = y_true_te.min(), y_true_te.max()
    plt.plot([mn, mx], [mn, mx], color="red")
    plt.title(f"Predicted vs True ({target_name}, TEST)")
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.tight_layout(); plt.savefig("pred_vs_true_test.png", dpi=150)

    # TEST abs-rel hist
    abs_rel = np.abs(y_pred_te_lin - y_true_te) / (y_true_te + 1e-12)
    plt.figure(figsize=(6, 4))
    plt.hist(abs_rel, bins=50)
    plt.title(f"Absolute Relative Error ({target_name}, TEST)")
    plt.xlabel("abs_rel"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig("abs_rel_hist_test.png", dpi=150)
