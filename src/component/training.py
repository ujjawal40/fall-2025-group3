from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .config import (
    DEVICE,
    EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    PINBALL_WEIGHT,
    L1_MEDIAN_WEIGHT,
    RELIABILITY_C,
)
from .evaluation import eval_split
from .model import pinball_loss


def train_one(model, ds_trn, ds_val, taus, head_weights=(1.0, 1.0)):
    dl_trn = DataLoader(ds_trn, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, bad = {"score": 1e18, "state": None, "epoch": -1}, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xnum, xcat, y, idx_now, w in dl_trn:
            xnum, xcat, y, w = xnum.to(DEVICE), xcat.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            outs = model(xnum, xcat)
            loss = 0.0
            for head_ix, out in enumerate(outs):
                pl = pinball_loss(out, y[:, head_ix], taus) * PINBALL_WEIGHT
                med_ix = taus.index(0.5)
                l1m = torch.mean(torch.abs(out[:, med_ix] - y[:, head_ix])) * L1_MEDIAN_WEIGHT
                if head_ix == 0 and RELIABILITY_C is not None:
                    pl = pl * w.mean()
                    l1m = l1m * w.mean()
                loss = loss + head_weights[head_ix] * (pl + l1m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())

        v1 = eval_split(model, dl_val, taus, head_ix=0)
        v2 = eval_split(model, dl_val, taus, head_ix=1)
        score = v1["mae"] + v2["mae"]
        print(
            f"[ep {ep:02d}] trn_loss={total/len(dl_trn):.5f} | "
            f"val H1: MAE=${v1['mae']:.0f} R2={v1['r2']:.3f} WAPE={v1['wape']:.3f} | "
            f"H2: MAE=${v2['mae']:.0f} R2={v2['r2']:.3f} WAPE={v2['wape']:.3f}"
        )
        if score < best["score"]:
            best = {
                "score": score,
                "state": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": ep,
            }
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(
                    f"Early stopping at epoch {ep} (no improv {PATIENCE}). "
                    f"Best epoch={best['epoch']}."
                )
                break

    model.load_state_dict(best["state"])
    return model
