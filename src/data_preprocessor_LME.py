"""Reusable data preprocessing utilities for house price / LME models
with Snowflake-inspired feature engineering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class DataPreprocessor:
    """
    Handle data loading, cleaning, and feature engineering steps.

    Additions over the basic version:
    - Snowflake-style boolean normalization via YES/NO token lists
    - Snowflake-style top-k one-hot for selected string columns
    - Best-effort PRICEHISTORY aggregation in pandas
    - Exposes spatial bundle + zpid for desirability models
    """

    dataset_path: Path | None = None
    extra_numeric_cols: List[str] | None = None

    # runtime metadata
    target_clip_value: float | None = None
    target_clip_quantile: float | None = None

    # config
    YES_TOKENS: tuple = ("yes", "true", "y", "1")
    NO_TOKENS: tuple = ("no", "false", "n", "0")

    # columns we’ll try to return for spatial / kNN
    SPATIAL_CANDIDATES: tuple = (
        "LATITUDE",
        "LONGITUDE",
        "ZIPCODE",
        "FIPS",
        "STATE_FIPS",
        "COUNTY_FIPS",
    )

    def __post_init__(self) -> None:
        # original groupings
        self.COLS_1A = [
            "SQFT",
            "BEDROOMS",
            "BATHROOMS",
            "STORIES",
            "LEVELS",
            "LOT",
            "PARKING",
            "POOLFEATURES",
            "BASEMENT",
            "STRUCTURETYPE",
            "HOMETYPE",
            "PROPERTYCONDITION",
            "COOLINGFEATURES",
            "HEATINGFEATURES",
            "SENIORLIVING",
            "NEWCONSTRUCTIONFLAG",
        ]

        self.COLS_1B = ["YEARBUILT", "CREATEDAT_YEAR", "CREATEDAT_MONTH"]

        self.COLS_1C = [
            "ELEMNTARYSCHOOLRATING",
            "MIDDLESCHOOLRATING",
            "HIGHSCHOOLRATING",
            "MONTHLY_UNEMPLOYMENT_RATE",
            "MONTHLY_AVG_MORTGAGE_RATE",
            "HOTNESS_SCORE",
            "SUPPLY_SCORE",
            "DEMAND_SCORE",
            "MEDIAN_DAYS_ON_MARKET",
        ]

        self.COLS_1D = ["STATE_FIPS", "COUNTY_FIPS"]

        self.KEEP_COLS = [
            "ZPID",
            *self.COLS_1A,
            *self.COLS_1B,
            *self.COLS_1C,
            *self.COLS_1D,
            "PRICE",
        ]

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load_data(self, file_path: str | Path | None = None) -> pd.DataFrame:
        requested = Path(file_path) if file_path is not None else self.dataset_path
        if requested is None:
            raise ValueError("A dataset path must be provided before loading data.")


        if not isinstance(requested, Path):
            requested = Path(requested)

       
        search_roots = []
        if requested.is_absolute():
            search_roots.append(requested)
        else:
            search_roots.extend(
                [
                    requested,
                    Path.cwd() / requested,
                    Path(__file__).resolve().parent / requested,
                ]
            )

        resolved_path: Path | None = None
        for candidate in search_roots:
            if candidate.is_file():
                resolved_path = candidate
                break

        if resolved_path is None:
            raise FileNotFoundError(f"File not found: {requested}")

        df = pd.read_csv(resolved_path)
        self.dataset_path = resolved_path

        if resolved_path != requested:
            print(f"Resolved dataset path to: {resolved_path}")

        print(f"Loaded data shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # helpers from Snowflake notebook
    # ------------------------------------------------------------------
    def _normalize_bool_tokens(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Map many string-ish Yes/No tokens to 1/0 like in Snowflake."""
        for c in cols:
            if c not in df.columns:
                continue
            ser = df[c].astype(str).str.strip().str.lower()
            is_yes = ser.isin(self.YES_TOKENS)
            is_no = ser.isin(self.NO_TOKENS)
            df[c] = np.where(is_yes, 1.0, np.where(is_no, 0.0, np.nan))
        return df

    def _one_hot_topk(self, df: pd.DataFrame, src: str, prefix: str, k: int = 5) -> pd.DataFrame:
        """
        Snowflake-like: keep top-k frequent categories (lowercased/trimmed),
        everything else → "other", and create one-hot columns.
        """
        if src not in df.columns:
            return df

        norm_col = (
            df[src]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown", "nan": "unknown"})
        )

        vc = norm_col.value_counts(dropna=False)
        top_vals = list(vc.head(k).index)

        cat_col = np.where(norm_col.isin(top_vals), norm_col, "other")
        df[f"{src}__CAT"] = cat_col

        for val in top_vals + ["other"]:
            safe = str(val).replace(" ", "_")
            df[f"{prefix}_{safe}"] = (df[f"{src}__CAT"] == val).astype(np.float32)

        df.drop(columns=[f"{src}__CAT"], inplace=True)
        return df

    def _price_history_agg(
        self,
        df: pd.DataFrame,
        ph_col: str = "PRICEHISTORY",
        list_date_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Best-effort pandas version of the Snowflake price_history_agg().
        Expects PRICEHISTORY to be a JSON array of objects like:
            [{"date": "...", "price": 350000, "priceChangeRate": -0.02}, ...]
        """
        if ph_col not in df.columns:
            return df

        if list_date_col is None:
            # prefer DATEPOSTED, else a year/month combo, else nothing
            if "DATEPOSTED" in df.columns:
                list_date_col = "DATEPOSTED"
            elif "CREATEDAT_YEAR" in df.columns and "CREATEDAT_MONTH" in df.columns:
                # synthesize a date, e.g. year-month-01
                df["_LISTDATE_SYNTH"] = pd.to_datetime(
                    df["CREATEDAT_YEAR"].astype(str) + "-" + df["CREATEDAT_MONTH"].astype(str) + "-01",
                    errors="coerce",
                )
                list_date_col = "_LISTDATE_SYNTH"

        ph_features = {
            "ph_n_events": [],
            "ph_n_price_cuts": [],
            "ph_last_change_date": [],
            "ph_max_list_price": [],
            "ph_min_list_price": [],
            "ph_price_volatility": [],
            "ph_avg_change_rate": [],
        }

        list_dates = pd.to_datetime(df[list_date_col], errors="coerce") if list_date_col and list_date_col in df.columns else None

        for idx, raw in df[ph_col].items():
            if pd.isna(raw):
                # no history
                for k in ph_features:
                    ph_features[k].append(np.nan)
                continue

            try:
                hist = json.loads(raw)
                if not isinstance(hist, list):
                    hist = []
            except Exception:
                hist = []

            if not hist:
                for k in ph_features:
                    ph_features[k].append(np.nan)
                continue

            # turn into DataFrame
            hdf = pd.DataFrame(hist)
            if "date" in hdf.columns:
                hdf["date"] = pd.to_datetime(hdf["date"], errors="coerce")

            # filter to <= listing date
            if list_dates is not None:
                ld = list_dates.iloc[idx]
                if pd.notna(ld):
                    hdf = hdf[hdf["date"] <= ld]

            # now aggregate
            n_events = len(hdf)
            price_col = hdf.get("price")
            chg_col = hdf.get("priceChangeRate")

            # count price cuts
            if chg_col is not None:
                n_cuts = (chg_col < 0).sum()
                avg_chg = chg_col.mean()
            else:
                n_cuts = np.nan
                avg_chg = np.nan

            last_change = hdf["date"].max() if "date" in hdf else np.nan
            max_price = price_col.max() if price_col is not None else np.nan
            min_price = price_col.min() if price_col is not None else np.nan
            price_vol = price_col.std() if price_col is not None else np.nan

            ph_features["ph_n_events"].append(n_events)
            ph_features["ph_n_price_cuts"].append(n_cuts)
            ph_features["ph_last_change_date"].append(last_change)
            ph_features["ph_max_list_price"].append(max_price)
            ph_features["ph_min_list_price"].append(min_price)
            ph_features["ph_price_volatility"].append(price_vol)
            ph_features["ph_avg_change_rate"].append(avg_chg)

        # attach back
        for k, vals in ph_features.items():
            df[k] = vals

        # days since change (like Snowflake)
        if list_dates is not None:
            df["ph_days_since_change"] = (list_dates - pd.to_datetime(df["ph_last_change_date"], errors="coerce")).dt.days
        else:
            df["ph_days_since_change"] = np.nan

        return df
    @staticmethod
    def num_from_text(series: pd.Series, allow_comma: bool = True) -> pd.Series:
        """
        Extract the first number from a text-y series like "3 bd" or "1,234 sqft".
        This is the same helper we used earlier.
        """
        pattern = r"([-+]?\d[\d,]*\.?\d*)"
        cleaned = series.astype(str).str.extract(pattern, expand=False)

        if allow_comma:
            cleaned = cleaned.str.replace(",", "", regex=False)

        return pd.to_numeric(cleaned, errors="coerce")

    # ------------------------------------------------------------------
    # main clean + engineer
    # ------------------------------------------------------------------
    def clean_and_engineer(
        self,
        df: pd.DataFrame,
        *,
        one_hot: bool = True,
        max_categories: int | None = 60,
        min_frequency: float = 0.005,
    ) -> pd.DataFrame:
        df = df.copy()

        # 0) try to derive price-history features if present
        df = self._price_history_agg(df)

        # 1) numeric-from-text like before
        numeric_text_cols = [
            "BEDROOMS",
            "BATHROOMS",
            "STORIES",
            "LEVELS",
            "LOT",
            "PARKING",
            "PARKINGTOTALSPACES",
            "YEARBUILT",
        ]
        for col in numeric_text_cols:
            if col in df.columns:
                df[col] = self.num_from_text(df[col])

        # 2) LOT small → acres→sqft
        if "LOT" in df.columns:
            mask = df["LOT"] < 10
            df.loc[mask, "LOT"] = df.loc[mask, "LOT"] * 43_560

        # 3) normalize parking
        raw_park = "PARKING" if "PARKING" in df.columns else "PARKINGTOTALSPACES"
        if raw_park in df.columns:
            df["GARAGE_SPACES"] = df[raw_park]
            df.drop(columns=[raw_park], inplace=True)

        # 4) to numeric
        for col in [
            "BEDROOMS",
            "BATHROOMS",
            "STORIES",
            "LEVELS",
            "GARAGE_SPACES",
            "YEARBUILT",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 5) Snowflake-style bool normalization for known yes/no-ish columns
        boolish = ["NEWCONSTRUCTIONFLAG", "SENIORLIVING", "FIREPLACE", "WATERVIEWYN", "BASEMENTYN"]
        df = self._normalize_bool_tokens(df, boolish)

        # 6) property age
        if "CREATEDAT_YEAR" in df.columns and "YEARBUILT" in df.columns:
            df["PROPERTY_AGE"] = df["CREATEDAT_YEAR"] - df["YEARBUILT"]
            df.loc[df["PROPERTY_AGE"] < 0, "PROPERTY_AGE"] = np.nan

        # 7) zero/negative → NaN
        for col in ["SQFT", "LOT", "MEDIAN_DAYS_ON_MARKET", "PRICE"]:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan

        # 8) logs
        if "SQFT" in df.columns:
            df["LOG_SQFT"] = np.log1p(df["SQFT"])
        if "LOT" in df.columns:
            df["LOG_LOT"] = np.log1p(df["LOT"])
        if "MEDIAN_DAYS_ON_MARKET" in df.columns:
            df["LOG_DOM"] = np.log1p(df["MEDIAN_DAYS_ON_MARKET"])

        # 9) month sin/cos
        if "CREATEDAT_MONTH" in df.columns:
            two_pi = 2 * np.pi
            df["MONTH_SIN"] = np.sin(two_pi * df["CREATEDAT_MONTH"] / 12)
            df["MONTH_COS"] = np.cos(two_pi * df["CREATEDAT_MONTH"] / 12)

        # 10) base imputations
        base_imp = ["SQFT", "LOT", "GARAGE_SPACES", "PROPERTY_AGE"]
        for col in base_imp:
            if col in df.columns:
                df[f"MISS_{col}"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())

        # 11) neighborhood fills
        neigh_cols = [
            "HOTNESS_SCORE",
            "SUPPLY_SCORE",
            "DEMAND_SCORE",
            "MONTHLY_UNEMPLOYMENT_RATE",
            "MONTHLY_AVG_MORTGAGE_RATE",
            "MEDIAN_DAYS_ON_MARKET",
        ]
        for col in neigh_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # 12) Snowflake-ish top-k one-hot for some known text cols
        # these mimic your `one_hot_topk` calls
        df = self._one_hot_topk(df, "BASEMENT", "BASEMENT", k=5)
        df = self._one_hot_topk(df, "PROPERTYCONDITION", "COND", k=4)

        # 13) keep extra numeric cols if user asked
        if self.extra_numeric_cols:
            for col in self.extra_numeric_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # 14) generic object handling → one-hot
        remain_obj = df.select_dtypes("object").columns
        truly_cat: list[str] = []
        for col in remain_obj:
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notna().mean() >= 0.80:
                df[col] = conv
            else:
                truly_cat.append(col)

        if one_hot and truly_cat:
            n_rows = len(df)

            def prune(col: str) -> pd.Series:
                series = df[col].astype(str).fillna("nan")
                vc = series.value_counts(dropna=False)
                keepers = set(vc.head(max_categories).index if max_categories else vc.index)
                if min_frequency > 0:
                    thr = min_frequency * n_rows
                    keepers.update(vc[vc >= thr].index)
                return series.where(series.isin(keepers), "OTHER")

            for col in truly_cat:
                df[col] = prune(col)

            df = pd.get_dummies(df, columns=truly_cat, dummy_na=True, prefix_sep="==", dtype=np.float32)
        else:
            df.drop(columns=truly_cat, inplace=True)

        return df

    # ------------------------------------------------------------------
    # feature prep (same as before, plus extras)
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        df: pd.DataFrame,
        *,
        target: str = "LOG_PPSQFT",
        clip_ppsqft_quantile: float | None = 0.995,
        max_ppsqft: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # mask before filtering so we can return it
        mask_valid = (df["PRICE"] > 0) & (df["SQFT"] > 0)
        df = df[mask_valid].copy()
        if df.empty:
            raise ValueError("No rows remain after filtering invalid PRICE/SQFT values.")

        # PPSQFT + clipping
        df["PPSQFT"] = df["PRICE"] / df["SQFT"].clip(lower=1)
        clip_value: float | None = None
        valid_ppsqft = df["PPSQFT"].replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_ppsqft.empty and clip_ppsqft_quantile is not None:
            q = float(np.clip(clip_ppsqft_quantile, 0.0, 1.0))
            clip_value = float(valid_ppsqft.quantile(q))
        if max_ppsqft is not None:
            clip_value = float(min(clip_value, max_ppsqft)) if clip_value is not None else float(max_ppsqft)
        if clip_value is not None and np.isfinite(clip_value):
            df["PPSQFT"] = df["PPSQFT"].clip(upper=clip_value)
            self.target_clip_value = clip_value
            self.target_clip_quantile = clip_ppsqft_quantile
        else:
            self.target_clip_value = None
            self.target_clip_quantile = None

        df["LOG_PPSQFT"] = np.log1p(df["PPSQFT"])

        # build numeric feature df
        numeric_df = df.select_dtypes(include=[np.number])
        drop_cols = ["PRICE", "PPSQFT", "LOG_PPSQFT", "ZPID"]
        features_df = numeric_df.drop(columns=drop_cols, errors="ignore")
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # median impute
        for i in range(features_df.shape[1]):
            col = features_df.iloc[:, i]
            if col.isna().all():
                col = col.fillna(0.0)
            else:
                col = col.fillna(col.median())
            features_df.iloc[:, i] = col

        features_df = features_df.astype(np.float32)
        features_df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

        # select target
        tkey = target.upper()
        if tkey == "LOG_PPSQFT":
            y = df["LOG_PPSQFT"].to_numpy(dtype=np.float64)
        elif tkey == "PRICE":
            y = df["PRICE"].to_numpy(dtype=np.float64)
        elif tkey == "LOG_PRICE":
            y = np.log(df["PRICE"].to_numpy(dtype=np.float64))
        else:
            raise ValueError("Unsupported target. Use 'PRICE', 'LOG_PRICE', or 'LOG_PPSQFT'.")

        X_param = features_df.to_numpy(dtype=np.float32)
        feature_names = features_df.columns.to_numpy()

        # build extras
        zpid = df["ZPID"].to_numpy(dtype=np.int64) if "ZPID" in df.columns else None
        spatial_cols = [c for c in self.SPATIAL_CANDIDATES if c in df.columns]
        spatial = df[spatial_cols].to_numpy(dtype=np.float32) if spatial_cols else None

        extras: Dict[str, Any] = {
            "zpid": zpid,
            "spatial": spatial,
            "spatial_cols": spatial_cols,
            "kept_mask": mask_valid.to_numpy(),
            "target_stats": {
                "min_price": float(df["PRICE"].min()),
                "max_price": float(df["PRICE"].max()),
                "mean_price": float(df["PRICE"].mean()),
                "n_rows": int(len(df)),
                "target_type": tkey,
            },
        }

        return X_param, y, feature_names, extras
