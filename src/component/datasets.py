from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Wraps a ZIP×month panel into tensors:
      - Numeric features (scaled)
      - Categorical features (indices)
      - Multi-task targets [Y_H1, Y_H2]
      - Current IDX level for dlog→level conversion
      - Optional sample weights
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rows_mask: pd.Series,
        num_cols: List[str],
        cat_cols: List[str],
        cat_maps: Dict[str, Dict[Any, int]],
        targets: List[str],
        weights_col: Optional[str] = None,
    ):
        ix = np.where(rows_mask.values)[0]
        self.df = df.iloc[ix].copy().reset_index(drop=True)

        ycols = [t + ("_WZ" if (t + "_WZ") in self.df.columns else "") for t in targets]

        idx_col = "IDX_EFF" if "IDX_EFF" in self.df.columns else "IDX"

        good = (
            pd.to_numeric(self.df[idx_col], errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .values
        )
        for yc in ycols:
            yv = (
                pd.to_numeric(self.df[yc], errors="coerce")
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .values
            )
            good &= yv
        self.df = self.df.loc[good].reset_index(drop=True)

        self.idx_col = idx_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.targets = targets
        self.weights_col = weights_col

        if len(num_cols) > 0:
            self.df.loc[:, num_cols] = (
                self.df[num_cols]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

        self.X_num = (
            torch.tensor(
                self.df[num_cols].astype(np.float32).values,
                dtype=torch.float32,
            )
            if num_cols
            else None
        )

        self.X_cat = []
        for c in cat_cols:
            mp = cat_maps[c]
            idx = (
                self.df[c].astype("string")
                .map(mp)
                .fillna(0)
                .astype(np.int64)
                .values
            )
            self.X_cat.append(torch.tensor(idx, dtype=torch.long))
        self.X_cat = torch.stack(self.X_cat, dim=1) if len(self.X_cat) > 0 else None

        ys = []
        for yc in ycols:
            yv = (
                pd.to_numeric(self.df[yc], errors="coerce")
                .astype(np.float32)
                .values
            )
            ys.append(torch.tensor(yv, dtype=torch.float32))
        self.y = torch.stack(ys, dim=1)

        self.idx_now = torch.tensor(
            pd.to_numeric(self.df[self.idx_col], errors="coerce")
            .astype(np.float32)
            .values,
            dtype=torch.float32,
        )

        if self.weights_col and self.weights_col in self.df.columns:
            wv = (
                pd.to_numeric(self.df[self.weights_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
                .astype(np.float32)
                .values
            )
            self.w = torch.tensor(wv, dtype=torch.float32)
        else:
            self.w = torch.ones(len(self.df), dtype=torch.float32)

    def __len__(self):
        return len(self.idx_now)

    def __getitem__(self, i):
        return (
            (self.X_num[i] if self.X_num is not None else torch.empty(0)),
            (self.X_cat[i] if self.X_cat is not None else torch.empty(0, dtype=torch.long)),
            self.y[i],
            self.idx_now[i],
            self.w[i],
        )
