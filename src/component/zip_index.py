import math
import numpy as np
import pandas as pd

from .geo_tiling import build_geo_tiling


class ZipMonthIndexBuilder:
    """
    Build ZIP×month panel with:
      - IDX_RAW: median sale/list per ZIP×YM (sale > list)
      - Liquidity: N_SOLD, N_LIST
      - Reliability: N_MONTHS_TO_DATE, W_H1_COMBINED
      - Pooling: county/state medians
      - IDX: effective index
      - Labels: Y_H1, Y_H2 as Δlog IDX
    """

    def __init__(
        self,
        combined_events: pd.DataFrame,
        min_start_date: str = "2022-01-01",
        min_sold_per_zip_m: int = 10,
        min_list_per_zip_m: int = 20,
    ):
        self.ce = combined_events
        self.min_start_date = min_start_date
        self.min_sold = int(min_sold_per_zip_m)
        self.min_list = int(min_list_per_zip_m)

    def build(self) -> pd.DataFrame:
        needed_cols = [
            "ZPID", "ZIPCODE", "STATE", "COUNTY",
            "EVT_DATE", "evt_date",
            "EVT_PRICE", "evt_price",
            "EVT_TYPE", "evt_type",
            "EVT_IS_RENTAL", "evt_is_rental",
            "WEEKLY_AVERAGE_MORTGAGE_RATE", "UNEMPLOYMENT_RATE",
        ]
        keep = [c for c in needed_cols if c in self.ce.columns]
        ce = self.ce[keep].copy()

        if "EVT_DATE" in ce.columns:
            ce["EVT_DATE"] = pd.to_datetime(ce["EVT_DATE"], errors="coerce")
        elif "evt_date" in ce.columns:
            ce["EVT_DATE"] = pd.to_datetime(ce["evt_date"], errors="coerce")
        else:
            raise KeyError("Combined events missing EVT_DATE/evt_date")

        ce = ce[ce["EVT_DATE"].notna()]
        ce = ce[ce["EVT_DATE"] >= pd.to_datetime(self.min_start_date)]
        print(f"[ZipMonthIndexBuilder] events after date filter: {len(ce):,}")

        ce["YM"] = ce["EVT_DATE"].values.astype("datetime64[M]")
        ce["DAY_FOR_SPLIT"] = ce["EVT_DATE"]

        ce["ZIPCODE"] = ce.get("ZIPCODE", pd.Series(index=ce.index, dtype="object")).astype(str)
        ce["STATE_MODE"] = ce.get("STATE", pd.Series(index=ce.index, dtype="object")).astype(str)
        ce["COUNTY_MODE"] = ce.get("COUNTY", pd.Series(index=ce.index, dtype="object")).astype(str)

        price_col = "evt_price" if "evt_price" in ce.columns else "EVT_PRICE"
        evt_type_col = "evt_type" if "evt_type" in ce.columns else "EVT_TYPE"
        rent_col = (
            "evt_is_rental" if "evt_is_rental" in ce.columns
            else ("EVT_IS_RENTAL" if "EVT_IS_RENTAL" in ce.columns else None)
        )

        raw_price = ce[price_col].astype(str)
        clean_price = raw_price.str.replace(r"[,\s\$%]", "", regex=True)
        ce["EVT_PRICE"] = pd.to_numeric(clean_price, errors="coerce")

        ce["EVT_TYPE"] = ce[evt_type_col].astype(str).str.lower()

        if rent_col is not None:
            ce["EVT_IS_RENTAL"] = ce[rent_col].fillna(False).astype(bool)
        else:
            ce["EVT_IS_RENTAL"] = False

        ce_nr = ce[~ce["EVT_IS_RENTAL"]].copy()
        print(f"[ZipMonthIndexBuilder] non-rental events: {len(ce_nr):,}")

        listing_like = ce_nr["EVT_TYPE"].isin(
            ["listing", "for sale", "listed for sale", "price change"]
        )
        sold_like = ce_nr["EVT_TYPE"].isin(["sold", "sale", "closed"])

        print("[ZipMonthIndexBuilder] EVT_TYPE sample:")
        print(ce_nr["EVT_TYPE"].value_counts().head(10))

        print(f"[ZipMonthIndexBuilder] listing_like rows: {listing_like.sum():,}")
        print(f"[ZipMonthIndexBuilder] sold_like rows:    {sold_like.sum():,}")

        keys = ["ZIPCODE", "STATE_MODE", "COUNTY_MODE", "YM"]
        has_price = ce_nr["EVT_PRICE"].notna()

        sold_df = ce_nr[sold_like & has_price].groupby(keys, as_index=False).agg(
            N_SOLD=("EVT_PRICE", "size"),
            SOLD_MEDIAN=("EVT_PRICE", "median"),
        )
        list_df = ce_nr[listing_like & has_price].groupby(keys, as_index=False).agg(
            N_LIST=("EVT_PRICE", "size"),
            LIST_MEDIAN=("EVT_PRICE", "median"),
        )

        if sold_df.empty and list_df.empty:
            print("[ZipMonthIndexBuilder] WARNING: no sold/list rows with price – fallback ANY price.")
            any_price = ce_nr[has_price].copy()
            base = any_price.groupby(keys, as_index=False).agg(
                N_SOLD=("EVT_PRICE", "size"),
                SOLD_MEDIAN=("EVT_PRICE", "median"),
                N_LIST=("EVT_PRICE", "size"),
                LIST_MEDIAN=("EVT_PRICE", "median"),
            )
        else:
            base = pd.merge(sold_df, list_df, on=keys, how="outer")

        if base.empty:
            print("[ZipMonthIndexBuilder] base is EMPTY after grouping – check EVT_PRICE parsing.")
            return base

        base["N_SOLD"] = base["N_SOLD"].fillna(0).astype(int)
        base["N_LIST"] = base["N_LIST"].fillna(0).astype(int)
        base["SOLD_MEDIAN"] = base["SOLD_MEDIAN"].astype(float)
        base["LIST_MEDIAN"] = base["LIST_MEDIAN"].astype(float)
        base["IDX_RAW"] = base["SOLD_MEDIAN"].fillna(base["LIST_MEDIAN"])

        print(f"[ZipMonthIndexBuilder] base rows after grouping: {len(base):,}")

        base = base.sort_values(["ZIPCODE", "YM"])
        base["N_MONTHS_TO_DATE"] = base.groupby("ZIPCODE").cumcount() + 1

        n_tx = np.maximum(
            base["N_SOLD"].fillna(0).astype(float),
            base["N_LIST"].fillna(0).astype(float),
        )
        w_hist_raw = np.minimum(base["N_MONTHS_TO_DATE"], 24).astype(float)
        w_hist = np.log1p(w_hist_raw) / np.log1p(12.0)
        w_hist = np.maximum(w_hist, 0.2)
        w_tx = np.log1p(n_tx) / 3.0
        w_tx = np.maximum(w_tx, 0.2)
        base["W_H1_COMBINED"] = np.minimum(w_hist * w_tx, 1.0)

        macro_cols = [c for c in ["WEEKLY_AVERAGE_MORTGAGE_RATE", "UNEMPLOYMENT_RATE"] if c in ce_nr.columns]
        if macro_cols:
            macro = ce_nr[has_price].groupby(keys, as_index=False)[macro_cols].median()
            base = base.merge(macro, on=keys, how="left")

        county_base = base.groupby(["COUNTY_MODE", "YM"], as_index=False)["IDX_RAW"].median()
        county_base = county_base.rename(columns={"IDX_RAW": "IDX_COUNTY_MED"})

        state_base = base.groupby(["STATE_MODE", "YM"], as_index=False)["IDX_RAW"].median()
        state_base = state_base.rename(columns={"IDX_RAW": "IDX_STATE_MED"})

        agg = base.merge(county_base, on=["COUNTY_MODE", "YM"], how="left")
        agg = agg.merge(state_base, on=["STATE_MODE", "YM"], how="left")

        agg["IDX_EFF"] = agg["IDX_RAW"]
        mask_na = agg["IDX_EFF"].isna()
        agg.loc[mask_na, "IDX_EFF"] = agg.loc[mask_na, "IDX_COUNTY_MED"]
        mask_na = agg["IDX_EFF"].isna()
        agg.loc[mask_na, "IDX_EFF"] = agg.loc[mask_na, "IDX_STATE_MED"]

        agg["IDX_REL_COUNTY"] = agg["IDX_EFF"] / agg["IDX_COUNTY_MED"].replace(0, np.nan)
        agg["IDX_REL_STATE"]  = agg["IDX_EFF"] / agg["IDX_STATE_MED"].replace(0, np.nan)

        agg["IDX"] = agg["IDX_EFF"]

        agg = agg.sort_values(["ZIPCODE", "YM"])
        grp = agg.groupby("ZIPCODE", group_keys=False)
        agg["IDX_FUTURE_H1"] = grp["IDX"].shift(-1)
        agg["IDX_FUTURE_H2"] = grp["IDX"].shift(-2)

        for col_src, col_y in [("IDX_FUTURE_H1", "Y_H1"), ("IDX_FUTURE_H2", "Y_H2")]:
            agg[col_y] = np.log1p(agg[col_src]) - np.log1p(agg["IDX"])

        print(f"[ZipMonthIndexBuilder] final agg rows: {len(agg):,}")
        return agg.reset_index(drop=True)


class ZipIndexFeatureizer:
    def __init__(
        self,
        idx_df: pd.DataFrame,
        add_macro: bool = True,
        monthly_periods: tuple = (12, 6),
        start_origin: str = "2015-01-01",
    ):
        self.idx_df = idx_df.copy()
        self.add_macro = add_macro
        self.periods = monthly_periods
        self.origin = pd.to_datetime(start_origin)

    def _add_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["YM"] = pd.to_datetime(df["YM"], errors="coerce")
        origin_year = self.origin.year
        origin_month = self.origin.month
        df["GLOBAL_MONTH_INDEX"] = (
            (df["YM"].dt.year - origin_year) * 12
            + (df["YM"].dt.month - origin_month)
        )

        for p in self.periods:
            angle = 2.0 * math.pi * (df["GLOBAL_MONTH_INDEX"] / float(p))
            df[f"SEAS_SIN_{p}"] = np.sin(angle)
            df[f"SEAS_COS_{p}"] = np.cos(angle)
        return df

    def _add_idx_lags_mom(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ZIPCODE", "YM"]).copy()
        grp = df.groupby("ZIPCODE")
        for lag in [1, 3, 6]:
            df[f"IDX_LAG_{lag}"] = grp["IDX"].shift(lag)

        df["IDX_PCT_D1"] = df["IDX"] / df["IDX_LAG_1"].replace(0, np.nan) - 1.0
        df["IDX_MOM_3"]   = df["IDX"] / df["IDX_LAG_3"].replace(0, np.nan) - 1.0
        df["IDX_MOM_6"]   = df["IDX"] / df["IDX_LAG_6"].replace(0, np.nan) - 1.0

        df["IDX_ROLL_STD_3"] = (
            grp["IDX"].rolling(window=3, min_periods=1).std()
            .reset_index(level=0, drop=True)
        )
        df["IDX_ROLL_STD_6"] = (
            grp["IDX"].rolling(window=6, min_periods=1).std()
            .reset_index(level=0, drop=True)
        )
        return df

    def _add_macro_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.add_macro:
            return df
        df = df.sort_values(["ZIPCODE", "YM"]).copy()
        grp = df.groupby("ZIPCODE")

        for base, alias in [
            ("WEEKLY_AVERAGE_MORTGAGE_RATE", "MORTGAGE_RATE_M"),
            ("UNEMPLOYMENT_RATE", "UNEMPLOYMENT_RATE_M"),
        ]:
            if base not in df.columns:
                continue
            s = pd.to_numeric(df[base], errors="coerce")
            df[alias] = s
            for lag in [1, 3, 12]:
                df[f"{alias}_L{lag}"] = grp[alias].shift(lag)
                df[f"{alias}_D{lag}"] = df[alias] - grp[alias].shift(lag)
        return df

    def build(self) -> pd.DataFrame:
        df = self.idx_df.copy()
        df = self._add_temporal(df)
        df = self._add_idx_lags_mom(df)
        df = self._add_macro_lags(df)
        return df


def build_all_features(
    combined_events: pd.DataFrame,
    h3_resolutions=(6, 7, 8, 9),
    min_start_date="2022-01-01",
    min_sold_per_zip_m=10,
    min_list_per_zip_m=20,
) -> pd.DataFrame:
    print("Building ZIP×month index & aggregates (pandas)…")
    idx_builder = ZipMonthIndexBuilder(
        combined_events=combined_events,
        min_start_date=min_start_date,
        min_sold_per_zip_m=min_sold_per_zip_m,
        min_list_per_zip_m=min_list_per_zip_m,
    )
    idx_df = idx_builder.build()

    print("Computing geo-tiling features (approx)…")
    geo_df = build_geo_tiling(combined_events, resolutions=h3_resolutions)

    print("Assembling feature matrix (ZipIndexFeatureizer)…")
    feat_df = idx_df.copy()
    if geo_df is not None:
        geo_cols = [c for c in geo_df.columns if c not in {"STATE_MODE", "COUNTY_MODE"}]
        feat_df = feat_df.merge(
            geo_df[geo_cols],
            on=["ZIPCODE", "YM"],
            how="left",
        )

    feat_df = ZipIndexFeatureizer(feat_df, add_macro=True, monthly_periods=(12, 6)).build()

    for c in ("YM", "DAY_FOR_SPLIT"):
        if c in feat_df.columns:
            feat_df[c] = pd.to_datetime(feat_df[c], errors="coerce")

    must_haves = ["YM", "IDX", "Y_H1", "Y_H2"]
    missing = [c for c in must_haves if c not in feat_df.columns]
    if missing:
        print("WARNING — Expected columns missing on feat_df:", missing)

    return feat_df
