# src/component/trainer.py
from __future__ import annotations
import math, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from .config_LME import HParams
from .surface import KernelSurface, LaplacianOp

def cg_solve_qp(apply_A, b: np.ndarray, x0: Optional[np.ndarray] = None,
                rel_tol: float = 1e-5, max_iter: int = 300, patience: int = 10):
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64) if x0 is None else x0.astype(np.float64).copy()
    r = b - apply_A(x); p = r.copy()
    r0 = np.linalg.norm(r) + 1e-12; rsold = np.dot(r, r)

    def obj(xv: np.ndarray) -> float:
        Ax = apply_A(xv); return 0.5 * float(np.dot(xv, Ax)) - float(np.dot(b, xv))

    best_obj, stale = np.inf, 0
    for _ in range(max_iter):
        Ap = apply_A(p); denom = np.dot(p, Ap) + 1e-18
        alpha = rsold / denom; x = x + alpha * p; r = r - alpha * Ap
        rsnew = np.dot(r, r); rel_res = float(np.sqrt(rsnew) / r0)
        if obj(x) + 1e-12 < best_obj: best_obj, stale = obj(x), 0
        else: stale += 1
        if rel_res <= rel_tol or stale >= patience: return x
        p = r + (rsnew / (rsold + 1e-18)) * p; rsold = rsnew
    return x

class LMETrainer:
    def __init__(self, X_tr, y_tr, w_tr, S_tr_std, S_val_std,
                 model, surf: KernelSurface, lap: Optional[LaplacianOp], hp: HParams):
        self.X_tr, self.y_tr = X_tr, y_tr
        self.w_tr = w_tr.astype(np.float32)
        self.S_tr_std, self.S_val_std = S_tr_std, S_val_std
        self.model, self.surf, self.lap, self.hp = model.to(hp.device), surf, lap, hp
        self.n = X_tr.shape[0]
        self.D = np.zeros(self.n, dtype=np.float64)
        self.U_list = self.surf.build_U_list()

    def _compute_h(self, D, U_list):  # TRAIN h
        h = np.zeros(self.n, dtype=np.float64)
        for i, (idxs, w) in enumerate(U_list):
            h[i] = float(np.dot(D[idxs], w))
        return h

    def _h_on_val(self, D):
        return np.array([self.surf.interpolate_one(s, D) for s in self.S_val_std], dtype=np.float64)

    def _pretrain(self) -> List[float]:
        if self.hp.verbose: print(f"[pretrain] starting for {self.hp.warmup_epochs} epochs...")
        ds = TensorDataset(torch.from_numpy(self.X_tr).float(),
                           torch.from_numpy(self.y_tr).float(),
                           torch.from_numpy(self.w_tr).float())
        loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
        loss_fn = nn.MSELoss(reduction="none")
        losses = []
        for ep in range(self.hp.warmup_epochs):
            running, wsum = 0.0, 0.0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.hp.device), yb.to(self.hp.device), wb.to(self.hp.device)
                pred = self.model(xb); per = loss_fn(pred, yb) * wb
                denom = wb.sum().clamp_min(1e-6); loss = per.sum() / denom
                opt.zero_grad(); loss.backward(); opt.step()
                running += per.sum().item(); wsum += float(denom)
            avg = running / max(1e-6, wsum); losses.append(float(avg))
            if self.hp.verbose: print(f"[pretrain] epoch {ep+1}/{self.hp.warmup_epochs} - loss: {avg:.4f}")
        return losses

    def _update_D(self):
        y = self.y_tr; w = self.w_tr.astype(np.float64); r = float(self.hp.reg_r)
        with torch.no_grad():
            m_np = self.model(torch.from_numpy(self.X_tr).float().to(self.hp.device)).cpu().numpy().astype(np.float64)

        b = np.zeros(self.n, dtype=np.float64)
        for i, (idxs, weights) in enumerate(self.U_list):
            b[idxs] += float(w[i]) * (y[i] - m_np[i]) * weights.astype(np.float64)

        lam = float(self.hp.lap_lambda) if self.lap is not None else 0.0
        def apply_A(v: np.ndarray) -> np.ndarray:
            out = r * v
            for i, (idxs, weights) in enumerate(self.U_list):
                coeff = float(np.dot(v[idxs], weights.astype(np.float64)))
                out[idxs] += w[i] * coeff * weights.astype(np.float64)
            if lam > 0.0: out += lam * self.lap.apply(v)
            return out

        D_new = cg_solve_qp(apply_A, b, x0=self.D,
                            rel_tol=self.hp.cg_rel_tol, max_iter=self.hp.cg_max_iter,
                            patience=self.hp.cg_patience)
        D_new -= D_new.mean()
        self.D = D_new

    def _mstep(self, X_val, y_val) -> List[float]:
        h_tr = self._compute_h(self.D, self.U_list).astype(np.float32)
        h_val = self._h_on_val(self.D).astype(np.float32)

        ds_tr = TensorDataset(torch.from_numpy(self.X_tr).float(),
                              torch.from_numpy(self.y_tr).float(),
                              torch.from_numpy(h_tr).float(),
                              torch.from_numpy(self.w_tr).float())
        loader = DataLoader(ds_tr, batch_size=self.hp.batch_size, shuffle=False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
        loss_fn = nn.MSELoss(reduction="none")

        def lr_factor(epoch: int) -> float:
            warm = min(1.0, (epoch+1)/max(1,self.hp.warmup_epochs))
            prog = max(0.0, (epoch+1 - self.hp.warmup_epochs) / max(1, self.hp.mstep_epochs - self.hp.warmup_epochs))
            return warm * (0.5 * (1.0 + math.cos(math.pi * prog)))

        best_val, wait, losses, best_state = float("inf"), 0, [], None
        X_val_t = torch.from_numpy(X_val).float().to(self.hp.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.hp.device)
        h_val_t = torch.from_numpy(h_val).float().to(self.hp.device)

        for ep in range(self.hp.mstep_epochs):
            for g in opt.param_groups: g["lr"] = self.hp.lr * lr_factor(ep)
            running, wsum = 0.0, 0.0
            self.model.train()
            for xb, yb, hb, wb in loader:
                xb, yb, hb, wb = xb.to(self.hp.device), yb.to(self.hp.device), hb.to(self.hp.device), wb.to(self.hp.device)
                y_res = yb - hb
                pred_m = self.model(xb); per = loss_fn(pred_m, y_res) * wb
                denom = wb.sum().clamp_min(1e-6); loss = per.sum() / denom
                opt.zero_grad(); loss.backward(); opt.step()
                running += per.sum().item(); wsum += float(denom)
            avg = running / max(1e-6, wsum); losses.append(float(avg))
            if self.hp.verbose: print(f"[m-step] epoch {ep+1}/{self.hp.mstep_epochs} - loss: {avg:.6f}")

            self.model.eval()
            with torch.no_grad():
                m_val = self.model(X_val_t)
                full_res = y_val_t - (m_val + h_val_t)
                val_energy = 0.5 * torch.mean(full_res ** 2).item()
            if val_energy + 1e-6 < best_val:
                best_val, wait = val_energy, 0
                best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.hp.patience:
                    if self.hp.verbose: print("[m-step] early stopping (VAL)")
                    break

        if best_state is not None: self.model.load_state_dict(best_state)
        return losses

    def fit(self, X_val, y_val) -> Dict[str, Any]:
        history = {"pretrain_losses": [], "em_losses": [], "mstep_losses_per_iter": []}
        history["pretrain_losses"] = self._pretrain()
        for it in range(self.hp.em_iters):
            if self.hp.verbose: print(f"[em] iteration {it+1}/{self.hp.em_iters}")
            prev = history["em_losses"][-1] if history["em_losses"] else None
            self._update_D()
            m_losses = self._mstep(X_val, y_val)
            history["mstep_losses_per_iter"].append(m_losses)
            with torch.no_grad():
                m_tr = self.model(torch.from_numpy(self.X_tr).float().to(self.hp.device)).cpu().numpy()
            h_tr = self._compute_h(self.D, self.U_list)
            energy = 0.5 * np.mean((self.y_tr - (m_tr + h_tr)) ** 2)
            history["em_losses"].append(float(energy))
            if self.hp.verbose:
                print(f"[em]   loss: {energy:.6f}  |  improvement: {('inf' if prev is None else (prev-energy))}")
        return history

    def predict(self, X_std, S_std):
        self.model.eval()
        with torch.no_grad():
            m = self.model(torch.from_numpy(X_std).float().to(self.hp.device)).cpu().numpy()
        h = np.array([self.surf.interpolate_one(s, self.D) for s in S_std], dtype=np.float64)
        return m + h
