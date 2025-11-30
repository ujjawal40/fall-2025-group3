from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEVICE, MIN_LEVEL, SUPPRESS_WIDTH_PCT, BATCH_SIZE
from .model import dlog_to_level


def _infer_price_scale(idx_vals: np.ndarray) -> float:
    med = float(np.nanmedian(idx_vals)) if idx_vals.size else np.nan
    if np.isfinite(med) and med < 10_000.0:
        return 1_000.0
    return 1.0


@torch.no_grad()
def eval_split(model, loader, taus, head_ix: int) -> Dict[str, Any]:
    model.eval()
    try:
        i10 = taus.index(0.1)
        i50 = taus.index(0.5)
        i90 = taus.index(0.9)
    except ValueError:
        i10, i50, i90 = 0, len(taus) // 2, -1

    y_list, p10_list, p50_list, p90_list, idx_list = [], [], [], [], []

    for xnum, xcat, y, idx_now, w in loader:
        if xnum.numel():
            xnum = xnum.to(DEVICE)
        if xcat.numel():
            xcat = xcat.to(DEVICE)
        y = y.to(DEVICE)
        idx_now = idx_now.to(DEVICE)

        outs = model(xnum, xcat)[head_ix]
        y_true = y[:, head_ix]
        p10 = outs[:, i10]
        p50 = outs[:, i50]
        p90 = outs[:, i90]

        mask = (
            torch.isfinite(y_true)
            & torch.isfinite(p50)
            & torch.isfinite(idx_now)
            & (idx_now > -1.0)
        )
        if not mask.any():
            continue

        y_list.append(y_true[mask].detach().cpu())
        p10_list.append(p10[mask].detach().cpu())
        p50_list.append(p50[mask].detach().cpu())
        p90_list.append(p90[mask].detach().cpu())
        idx_list.append(idx_now[mask].detach().cpu())

    if not y_list:
        return dict(
            mae=np.nan,
            r2=np.nan,
            wape=np.nan,
            mdape=np.nan,
            pct10=np.nan,
            p90_p10_cover=np.nan,
            rel_width=np.nan,
        )

    y_true = torch.cat(y_list)
    p10 = torch.cat(p10_list)
    p50 = torch.cat(p50_list)
    p90 = torch.cat(p90_list)
    idx = torch.cat(idx_list)

    true_lvl = dlog_to_level(idx, y_true).cpu().numpy()
    pred_lvl = dlog_to_level(idx, p50).cpu().numpy()
    p10_lvl = dlog_to_level(idx, p10).cpu().numpy()
    p90_lvl = dlog_to_level(idx, p90).cpu().numpy()

    scale = _infer_price_scale(idx.cpu().numpy())
    true_d = true_lvl * scale
    pred_d = pred_lvl * scale
    p10_d = p10_lvl * scale
    p90_d = p90_lvl * scale

    finite = np.isfinite(true_d) & np.isfinite(pred_d)
    if finite.sum() == 0:
        return dict(
            mae=np.nan,
            r2=np.nan,
            wape=np.nan,
            mdape=np.nan,
            pct10=np.nan,
            p90_p10_cover=np.nan,
            rel_width=np.nan,
        )

    yv = true_d[finite]
    pv = pred_d[finite]

    mae = float(np.nanmean(np.abs(yv - pv)))
    if len(yv) > 1 and np.nanvar(yv) > 0:
        ss_res = np.nansum((yv - pv) ** 2)
        ss_tot = np.nansum((yv - np.nanmean(yv)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r2 = np.nan

    denom = np.nansum(np.abs(yv))
    wape_val = float(np.nansum(np.abs(yv - pv)) / denom) if denom > 0 else np.nan
    mdape_val = float(
        np.nanmedian(np.abs((yv - pv) / np.clip(np.abs(yv), 1e-9, None)))
    )
    pct10_val = float(np.nanmean(np.abs(pv - yv) <= 0.10 * np.abs(yv)))

    finite_pi = (
        np.isfinite(true_d)
        & np.isfinite(p10_d)
        & np.isfinite(p90_d)
        & np.isfinite(pred_d)
    )
    if finite_pi.sum() == 0:
        cover = np.nan
        rel_w = np.nan
    else:
        y_pi = true_d[finite_pi]
        p10_pi = p10_d[finite_pi]
        p90_pi = p90_d[finite_pi]
        p50_pi = pred_d[finite_pi]
        width = np.maximum(np.abs(p90_pi - p10_pi), 1e-9)
        cover = float(np.mean((y_pi >= p10_pi) & (y_pi <= p90_pi)))
        rel_w = float(
            np.mean(width / np.clip(np.abs(p50_pi), MIN_LEVEL, None))
        )

    return dict(
        mae=mae,
        r2=r2,
        wape=wape_val,
        mdape=mdape_val,
        pct10=pct10_val,
        p90_p10_cover=cover,
        rel_width=rel_w,
    )


@torch.no_grad()
def suppression_report(model, ds, taus, head_ix):
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    n, n_suppr = 0, 0

    try:
        i10 = taus.index(0.1)
        i50 = taus.index(0.5)
        i90 = taus.index(0.9)
    except ValueError:
        i10, i50, i90 = 0, len(taus) // 2, -1

    for xnum, xcat, y, idx_now, w in dl:
        if xnum.numel():
            xnum = xnum.to(DEVICE)
        if xcat.numel():
            xcat = xcat.to(DEVICE)
        idx_now = idx_now.to(DEVICE)

        outs = model(xnum, xcat)[head_ix]
        p10 = outs[:, i10]
        p50 = outs[:, i50]
        p90 = outs[:, i90]

        mask = (
            torch.isfinite(p10)
            & torch.isfinite(p50)
            & torch.isfinite(p90)
            & torch.isfinite(idx_now)
            & (idx_now > -1.0)
        )
        if not mask.any():
            continue

        idx_m = idx_now[mask]
        p10_m = p10[mask]
        p50_m = p50[mask]
        p90_m = p90[mask]

        pred_lvl = dlog_to_level(idx_m, p50_m).cpu().numpy()
        p10_lvl = dlog_to_level(idx_m, p10_m).cpu().numpy()
        p90_lvl = dlog_to_level(idx_m, p90_m).cpu().numpy()

        scale = _infer_price_scale(idx_m.cpu().numpy())
        pv = pred_lvl * scale
        lo = p10_lvl * scale
        hi = p90_lvl * scale

        finite = np.isfinite(pv) & np.isfinite(lo) & np.isfinite(hi)
        if finite.sum() == 0:
            continue

        pv = pv[finite]
        lo = lo[finite]
        hi = hi[finite]
        width = np.abs(hi - lo)
        rel_width = width / np.clip(np.abs(pv), MIN_LEVEL, None)

        m = rel_width > SUPPRESS_WIDTH_PCT
        n += len(m)
        n_suppr += int(m.sum())

    return dict(suppressed=n_suppr, total=n, rate=(float(n_suppr) / max(n, 1) if n else np.nan))
