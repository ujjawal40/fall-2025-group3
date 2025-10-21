"""Local pipeline for ZIP-level time-series modeling.

This script adapts the original Snowflake-based notebook pipeline to run fully
on a local machine by using pandas and NumPy instead of Snowpark. It keeps the
same class/function names to ease cross-environment parity while replacing the
data access layers with local CSV reads (``src/sub_sample.csv`` by default).
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401 - parity with original dependencies
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # noqa: F401 - kept for parity

import lightgbm as lgb
import xgboost as xgb


# --------------------------------------------------------------------------------------
# Configuration (mirrors notebook constants)
# --------------------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_default_raw_table() -> Path:
    """Locate ``sub_sample.csv`` relative to common entry points.

    Users sometimes execute the script from the repository root (``python src/…``)
    or from inside ``src`` directly. We check the most common directories so the
    pipeline can discover the CSV regardless of the current working directory.
    """

    candidates = [
        SCRIPT_DIR / "sub_sample.csv",
        SCRIPT_DIR.parent / "src" / "sub_sample.csv",
        SCRIPT_DIR.parent / "sub_sample.csv",
        Path.cwd() / "src" / "sub_sample.csv",
        Path.cwd() / "sub_sample.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback to the script-local location even if it does not yet exist so we
    # surface a consistent error message in ``CombinedEventsBuilder``.
    return SCRIPT_DIR / "sub_sample.csv"


RAW_TABLE_PATH = _resolve_default_raw_table()

MIN_START_DATE = "2022-01-01"
HOLDOUT_DAYS = 60
MIN_SOLD_PER_ZIP_M = 20
MIN_LIST_PER_ZIP_M = 40
TOPK_HOMETYPES = 6
CHUNK_LIMIT_ROWS = 100_000
RANDOM_SEED = 42

KEY_COLS = [
    "ZPID",
    "URL",
    "STREETADDRESS",
    "CITY",
    "STATE",
    "COUNTY",
    "ZIPCODE",
    "FIPS",
    "FIPSCODE",
    "STATEFIPS",
    "COUNTYFIPS",
    "MetroAreaID",
    "MetroAreaTitle",
    "DivisionCode",
    "REGION",
    "DIVISION",
    "CITYTYPE",
]

FREE_TEXT_EXCLUDE = ["URL", "STREETADDRESS", "DESCRIPTION"]
VARIANT_COLS = ["PRICEHISTORY"]

EXPLICIT_DROPS = set(FREE_TEXT_EXCLUDE + VARIANT_COLS + KEY_COLS)

REQUIRED_COLS = {
    "ZPID",
    "EVT_DATE",
    "EVT_TYPE",
    "EVT_PRICE",
    "EVT_IS_RENTAL",
    "ZIPCODE",
    "STATE",
    "COUNTY",
    "HOMETYPE",
    "WEEKLY_AVERAGE_MORTGAGE_RATE",
    "UNEMPLOYMENT_RATE",
}

COORDS_NUMERIC = ["LONGITUDE", "LATITUDE"]

NUM_COLS_NUMERIC = [
    "PRICE",
    "SQFT",
    "LOTSQFT",
    "WALKSCORE",
    "TRANSITSCORE",
    "UNEMPLOYMENT_RATE",
    "MEDIAN_DAYS_ON_MARKET",
    "MEDIAN_LISTING_PRICE",
    "SUPPLY_SCORE",
    "ACTIVE_LISTING_COUNT",
    "HOTNESS_SCORE",
    "DEMAND_SCORE",
    "HOTNESS_RANK",
    "FM_HPI",
    "PROPERTY_AGE",
    "WEEKLY_AVERAGE_MORTGAGE_RATE",
    "MONTH",
    "YEAR",
    "CREATEDAT_MONTH",
    "CREATEDAT_YEAR",
    "POPULATION",
    "HOUSEHOLDSPERZIPCODE",
    "WHITEPOPULATION",
    "BLACKPOPULATION",
    "HISPANICPOPULATION",
    "ASIANPOPULATION",
    "HAWAIIANPOPULATION",
    "INDIANPOPULATION",
    "OTHERPOPULATION",
    "MALEPOPULATION",
    "FEMALEPOPULATION",
    "PERSONSPERHOUSEHOLD",
    "AVERAGEHOUSEVALUE",
    "INCOMEPERHOUSEHOLD",
    "MEDIANAGE",
    "MEDIANAGEMALE",
    "MEDIANAGEFEMALE",
    "NUMBEROFBUSINESSES",
    "NUMBEROFEMPLOYEES",
    "BUSINESSANNUALPAYROLL",
    "GROWTHRANK",
    "GROWTHINCREASENUMBER",
    "GROWTHINCREASEPERCENTAGE",
    "POPULATIONESTIMATE",
    "LANDAREA",
    "WATERAREA",
    "VALUE_2_UNITS_REP_M",
    "UNITS_3_4_UNITS_REP_M",
    "VALUE_5_UNITS_REP_M",
    "UNITS_1_UNIT_REP_M",
    "UNITS_2_UNITS_REP_M",
    "VALUE_1_UNIT_REP_M",
    "UNITS_5_UNITS_REP_M",
    "RN",
]

NUM_COLS_TEXT_TO_NUMERIC = [
    "BEDROOMS",
    "BATHROOMS",
    "FULLBATHROOMS",
    "HALFBATHROOMS",
    "YEARBUILT",
    "HOAFEE",
    "PARKINGTOTALSPACES",
    "ELEMENTARYSCHOOLDISTANCE",
    "MIDDLESCHOOLDISTANCE",
    "HIGHSCHOOLDISTANCE",
    "ELEMENTARYSCHOOLRATING",
    "MIDDLESCHOOLRATING",
    "HIGHSCHOOLRATING",
    "TOURVIEWCOUNT",
]

BIN_COLS_NUMERIC_01 = [
    "FIREPLACEYN",
    "NEWCONSTRUCTIONFLAG",
    "HASHOA",
    "SENIORLIVING",
    "ONSTREETPARKING",
    "GARAGEPARKING",
    "ATTACHEDPARKING",
    "DETACHEDPARKING",
    "DRIVEWAY",
    "OFFSTREETPARKING",
    "NOPARKING",
    "RADIATORHEATING",
    "CENTRALHEATING",
    "FORCEDAIRHEATING",
    "SOLARHEATING",
    "ELECTRICHEATING",
    "ZONEDHEATING",
    "HOTWATERHEATING",
    "OILHEATING",
    "PROPANEHEATING",
    "NATURALGASHEATING",
    "NOHEATINGINFO",
    "CENTRALCOOLING",
    "WINDOWUNITACCOOLING",
    "WALLUNITACCOOLING",
    "MULTIUNITCOOLING",
    "ZONEDCOOLING",
    "NOCOOLINGINFO",
    "NOPOOLFEATURES",
    "ABOVEGROUNDPOOL",
    "INGROUNDPOOL",
    "PERSONALPOOL",
    "FENCEDPOOL",
    "INDOORPOOL",
    "HEATEDPOOL",
    "FILTEREDPOOL",
    "SALTWATERPOOL",
    "POOLMATERIALVINYL",
    "POOLMATERIALCONCRETE",
    "POOLMATERIALGUNITE",
    "SINGLEFAMILY",
    "TOWNHOUSE",
]

BIN_COLS_TEXT_YN = ["basementYN"]


def _normalize_name(name: str) -> str:
    """Collapse a column name to lowercase alphanumerics for fuzzy matching."""

    if name is None:
        return ""
    cleaned = ''.join(ch for ch in str(name).lower() if ch.isalnum())
    return cleaned


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns flexibly so downstream steps find the expected fields."""

    canonical_targets = sorted(
        set(
            list(KEY_COLS)
            + list(REQUIRED_COLS)
            + COORDS_NUMERIC
            + NUM_COLS_NUMERIC
            + NUM_COLS_TEXT_TO_NUMERIC
            + BIN_COLS_NUMERIC_01
            + BIN_COLS_TEXT_YN
        )
    )
    name_map = {_normalize_name(name): name for name in canonical_targets}
    alias_map: Dict[str, str] = {
        # Mortgage / macro aliases that show up in CSV extracts
        "weeklyavgmortgagerate": "WEEKLY_AVERAGE_MORTGAGE_RATE",
        "weeklyaveragemortgagerate": "WEEKLY_AVERAGE_MORTGAGE_RATE",
        "monthlyavgmortgagerate": "WEEKLY_AVERAGE_MORTGAGE_RATE",
        "monthlyaveragemortgagerate": "WEEKLY_AVERAGE_MORTGAGE_RATE",
        "avgmortgagerate": "WEEKLY_AVERAGE_MORTGAGE_RATE",
        "mortgagerate": "WEEKLY_AVERAGE_MORTGAGE_RATE",
        # Unemployment variants
        "monthlyunemploymentrate": "UNEMPLOYMENT_RATE",
        "unemploymentrate": "UNEMPLOYMENT_RATE",
        "avgunemploymentrate": "UNEMPLOYMENT_RATE",
        # County fallbacks – common exports only ship FIPS codes
        "countyfips": "COUNTY",
        "countyname": "COUNTY",
    }
    rename: Dict[str, str] = {}
    already_has: set[str] = {col for col in df.columns}
    for col in df.columns:
        normalized = _normalize_name(col)
        target = name_map.get(normalized)
        if target:
            if target == col:
                continue
            if target in already_has:
                # If the canonical name already exists, do not clobber it.
                continue
            rename[col] = target
            continue

        alias_target = alias_map.get(normalized)
        if alias_target:
            if alias_target in df.columns:
                # Only backfill missing values so we do not overwrite real data.
                mask = df[alias_target].isna()
                if mask.any():
                    df.loc[mask, alias_target] = df.loc[mask, col]
            else:
                df[alias_target] = df[col]
    if rename:
        df = df.rename(columns=rename)
    return df


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == "object":
            nun = s.nunique(dropna=False)
            if nun and nun / max(len(s), 1) <= 0.4:
                df[col] = df[col].astype("category")
    return df


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), float)
    y_pred = np.asarray(list(y_pred), float)
    denom = np.abs(y_true).sum()
    return math.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom


def mdape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), float)
    y_pred = np.asarray(list(y_pred), float)
    pct = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))
    return float(np.nanmedian(pct))


def pct_within(y_true: Iterable[float], y_pred: Iterable[float], pct: float = 0.10) -> float:
    y_true = np.asarray(list(y_true), float)
    y_pred = np.asarray(list(y_pred), float)
    ok = np.abs(y_pred - y_true) <= (pct * np.abs(y_true))
    return float(np.mean(ok)) if len(ok) else math.nan


def safe_to_double(series: pd.Series) -> pd.Series:
    regex = r"^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)$"
    as_str = series.astype(str)
    mask = as_str.str.match(regex, na=False)
    return pd.to_numeric(series.where(mask), errors="coerce")


def safe_to_binary_from_text(series: pd.Series) -> pd.Series:
    mapping = {
        "Y": 1,
        "YES": 1,
        "TRUE": 1,
        "T": 1,
        "1": 1,
        "N": 0,
        "NO": 0,
        "FALSE": 0,
        "F": 0,
        "0": 0,
    }
    return series.astype(str).str.upper().map(mapping)


def safe_to_binary_from_number(series: pd.Series) -> pd.Series:
    num = safe_to_double(series)
    return num.where(num.isin([0, 1]))


def _slugify(value: str) -> str:
    cleaned = "" if value is None else str(value).strip()
    cleaned = cleaned.replace("/", " ").replace("-", " ")
    cleaned = "_".join(filter(None, [chunk.upper() for chunk in cleaned.split()]))
    return cleaned or "UNK"


# --------------------------------------------------------------------------------------
# CombinedEventsBuilder – CSV + pandas adaptation
# --------------------------------------------------------------------------------------


class CombinedEventsBuilder:
    """Creates a tidy events DataFrame joined with the latest property snapshot."""

    def __init__(
        self,
        raw_table: Path,
        zpid_col: str = "ZPID",
        pricehistory_col: str = "PRICEHISTORY",
        scrape_ts_col: str = "SCRAPEDAT",
    ) -> None:
        self.raw_table = Path(raw_table).expanduser()
        self.zpid_col = zpid_col
        self.pricehistory_col = pricehistory_col
        self.scrape_ts_col = scrape_ts_col

        self.c_zpid_key = "zpid_key"
        self.c_evt_date = "EVT_DATE"
        self.c_evt_ts = "EVT_TS"
        self.c_evt_type = "EVT_TYPE"
        self.c_evt_price = "EVT_PRICE"
        self.c_evt_price_psf = "EVT_PRICE_PSF"
        self.c_evt_is_rent = "EVT_IS_RENTAL"
        self.c_evt_source = "EVT_SOURCE"
        self.c_evt_mls_id = "EVT_MLS_ID"
        self.c_evt_mls_name = "EVT_MLS_NAME"

        self.c_sort_ts = "sort_ts"
        self.c_event_seq = "event_seq"
        self.c_days_prev = "days_since_prev"
        self.c_days_first = "days_since_first"

        self.c_base_zpid = "BASE_ZPID"
        self.c_base_zpid_key = "BASE_ZPID_KEY"

    def _load_base(self) -> pd.DataFrame:
        if not self.raw_table.exists():
            raise FileNotFoundError(
                "Raw table CSV not found: "
                f"{self.raw_table}. If you relocated sub_sample.csv, pass its "
                "path to run_pipeline(raw_table=...) or set the RAW_TABLE_PATH "
                "constant."
            )
        df = pd.read_csv(self.raw_table)
        return _canonicalize_columns(df)

    def build(self) -> pd.DataFrame:
        base = self._load_base()
        events = self._flatten_events(base)
        events = self._add_sequence(events)
        base_snap = self._make_base_snapshot(base)
        combined = events.merge(
            base_snap,
            left_on=self.c_zpid_key,
            right_on=self.c_base_zpid_key,
            how="left",
        )
        if self.c_base_zpid_key in combined.columns:
            combined = combined.drop(columns=[self.c_base_zpid_key])
        return _canonicalize_columns(combined)

    def _flatten_events(self, base_df: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for row in base_df.itertuples(index=False):
            zpid = getattr(row, self.zpid_col, None)
            price_history = getattr(row, self.pricehistory_col, None)
            if pd.isna(price_history):
                continue
            try:
                events = json.loads(price_history)
                if isinstance(events, dict):
                    events = events.get("events", [])
            except json.JSONDecodeError:
                continue

            for idx, evt in enumerate(events or []):
                if not isinstance(evt, dict):
                    continue
                evt_date = pd.to_datetime(evt.get("date"), errors="coerce")
                evt_time_raw = evt.get("time")
                evt_ts = pd.to_datetime(evt_time_raw, unit="ms", errors="coerce")
                attribute_source = evt.get("attributeSource") or {}

                record = {
                    self.zpid_col: zpid,
                    self.c_zpid_key: str(zpid) if pd.notna(zpid) else None,
                    self.c_evt_date: evt_date,
                    self.c_evt_ts: evt_ts,
                    self.c_evt_type: str(evt.get("event", "")).lower() if evt.get("event") else None,
                    self.c_evt_price: float(evt.get("price")) if evt.get("price") is not None else math.nan,
                    self.c_evt_price_psf: float(evt.get("pricePerSquareFoot")) if evt.get("pricePerSquareFoot") is not None else math.nan,
                    self.c_evt_is_rent: bool(evt.get("postingIsRental")) if evt.get("postingIsRental") is not None else False,
                    self.c_evt_source: evt.get("source"),
                    self.c_evt_mls_id: attribute_source.get("infoString1"),
                    self.c_evt_mls_name: attribute_source.get("infoString2"),
                    "raw_time_ms_str": str(evt_time_raw) if evt_time_raw is not None else None,
                    "json_index": idx,
                }
                sort_ts = evt_ts if pd.notna(evt_ts) else evt_date
                record[self.c_sort_ts] = sort_ts
                if pd.isna(evt_date) and pd.isna(sort_ts):
                    continue
                records.append(record)

        if not records:
            return pd.DataFrame(columns=[self.zpid_col, self.c_zpid_key])

        events_df = pd.DataFrame.from_records(records)
        events_df[self.c_evt_date] = _ensure_datetime(events_df[self.c_evt_date])
        events_df[self.c_evt_ts] = _ensure_datetime(events_df[self.c_evt_ts])
        events_df[self.c_sort_ts] = _ensure_datetime(events_df[self.c_sort_ts])
        return events_df

    def _add_sequence(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty:
            return events_df
        events_df = events_df.sort_values(
            [self.c_zpid_key, self.c_evt_date, self.c_sort_ts, "json_index"],
            kind="mergesort",
        )
        events_df[self.c_event_seq] = events_df.groupby(self.c_zpid_key).cumcount() + 1
        group = events_df.groupby(self.c_zpid_key)
        sort_ts = _ensure_datetime(events_df[self.c_sort_ts])
        prev_ts = group[self.c_sort_ts].shift(1)
        first_ts = group[self.c_sort_ts].transform("first")
        events_df[self.c_days_prev] = (sort_ts - _ensure_datetime(prev_ts)).dt.days
        events_df[self.c_days_first] = (sort_ts - _ensure_datetime(first_ts)).dt.days
        return events_df

    def _make_base_snapshot(self, base_df: pd.DataFrame) -> pd.DataFrame:
        cols_no_json = [c for c in base_df.columns if c != self.pricehistory_col]
        base_no_json = base_df[cols_no_json].copy()
        if self.scrape_ts_col in base_no_json.columns:
            scrape = _ensure_datetime(base_no_json[self.scrape_ts_col])
            base_no_json = base_no_json.assign(_scrape_ts=scrape)
            base_no_json = base_no_json.sort_values([self.zpid_col, "_scrape_ts"], ascending=[True, False])
            base_latest = base_no_json.drop_duplicates(subset=[self.zpid_col], keep="first").drop(columns=["_scrape_ts"])
        else:
            base_latest = base_no_json.drop_duplicates(subset=[self.zpid_col], keep="first")
        base_latest = base_latest.rename(columns={self.zpid_col: self.c_base_zpid})
        base_latest[self.c_base_zpid_key] = base_latest[self.c_base_zpid].astype(str)
        return base_latest


# --------------------------------------------------------------------------------------
# ZipMonthIndexBuilder – pandas adaptation
# --------------------------------------------------------------------------------------


@dataclass
class ZipMonthIndexBuilder:
    ce: pd.DataFrame
    min_sold: int = MIN_SOLD_PER_ZIP_M
    min_list: int = MIN_LIST_PER_ZIP_M
    alpha_sold: float = 10.0
    alpha_msa: float = 50.0

    def __post_init__(self) -> None:
        # Ensure column canonicalization even if the caller bypassed CombinedEventsBuilder.
        self.ce = _canonicalize_columns(self.ce.copy())

    def _base(self) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLS if c not in self.ce.columns]
        if missing:
            available = ", ".join(sorted(self.ce.columns))
            raise ValueError(
                "Missing columns in combined_events: "
                f"{missing}. Available columns: {available}"
            )
        base = self.ce[self.ce["EVT_IS_RENTAL"] == 0].copy()
        base["EVT_DAY"] = _ensure_datetime(base["EVT_DATE"]).dt.date
        base["YM"] = _ensure_datetime(base["EVT_DATE"]).dt.to_period("M").dt.to_timestamp()
        base = base[base["EVT_DAY"] >= pd.to_datetime(MIN_START_DATE).date()]
        return base

    def _filter_extreme_values(self, base: pd.DataFrame) -> pd.DataFrame:
        filtered = base.copy()
        for col in ["EVT_PRICE", "SQFT", "LOTSQFT"]:
            if col in filtered.columns:
                mask = filtered[col].isna() | ((filtered[col] < 1e12) & (filtered[col] > -1e9))
                filtered = filtered[mask]
        return filtered

    def _index_core(self, base: pd.DataFrame) -> pd.DataFrame:
        base = base.copy()
        base["EVT_TYPE"] = base["EVT_TYPE"].astype(str).str.lower()
        sold = (
            base[base["EVT_TYPE"] == "sold"].groupby(["ZIPCODE", "YM"])["EVT_PRICE"].agg(["median", "count"])
        )
        sold = sold.rename(columns={"median": "SOLD_MEDIAN", "count": "N_SOLD"})
        list_like = (
            base[base["EVT_TYPE"].isin(["listed for sale", "price change", "price increased", "price reduced"])].groupby(["ZIPCODE", "YM"])["EVT_PRICE"].agg(["median", "count"])
        )
        list_like = list_like.rename(columns={"median": "LIST_MEDIAN", "count": "N_LIST"})
        macro = (
            base.groupby(["ZIPCODE", "YM"])[["WEEKLY_AVERAGE_MORTGAGE_RATE", "UNEMPLOYMENT_RATE"]].mean().rename(columns={"WEEKLY_AVERAGE_MORTGAGE_RATE": "MORTGAGE_RATE_M", "UNEMPLOYMENT_RATE": "UNEMPLOYMENT_RATE_M"})
        )
        idx0 = sold.join(list_like, how="outer").join(macro, how="left")
        idx0 = idx0.fillna({"N_SOLD": 0, "N_LIST": 0})
        return idx0.reset_index()

    def _pick_col(self, candidates: Iterable[str]) -> Optional[str]:
        lower_map = {c.lower(): c for c in self.ce.columns}
        for cand in candidates:
            match = lower_map.get(cand.lower())
            if match:
                return match
        return None

    def _geo_modes(self, base: pd.DataFrame) -> pd.DataFrame:
        msa_id = self._pick_col(["MetroAreaID", "MSA_ID", "METROAREAID", "MSAID"])
        msa_name = self._pick_col(["MetroAreaTitle", "MSA_TITLE", "METROAREATITLE"])
        cols = ["ZIPCODE", "YM", "STATE", "COUNTY"]
        if msa_id:
            cols.append(msa_id)
        if msa_name:
            cols.append(msa_name)
        grouped = base[cols].copy()
        grouped["CNT"] = 1
        modes = (
            grouped.groupby(cols)["CNT"].sum().reset_index().sort_values(["ZIPCODE", "YM", "CNT"], ascending=[True, True, False]).drop_duplicates(subset=["ZIPCODE", "YM"], keep="first")
        )
        out = modes[["ZIPCODE", "YM"]].copy()
        out["STATE_MODE"] = modes["STATE"].astype(str)
        out["COUNTY_MODE"] = modes["COUNTY"].astype(str)
        out["MSA_ID"] = modes[msa_id].astype(str) if msa_id else None
        out["MSA_TITLE"] = modes[msa_name].astype(str) if msa_name else None
        return out

    def _index_with_shrinkage(self, base: pd.DataFrame) -> pd.DataFrame:
        idx0 = self._index_core(base)
        geo = self._geo_modes(base)
        idx = idx0.merge(geo, on=["ZIPCODE", "YM"], how="left")
        w_sold = idx["N_SOLD"] / (idx["N_SOLD"] + self.alpha_sold)

        def _blend(row: pd.Series) -> float:
            sold = row.get("SOLD_MEDIAN")
            listed = row.get("LIST_MEDIAN")
            weight = row.get("w_sold")
            if pd.notna(sold) and pd.notna(listed):
                return weight * sold + (1 - weight) * listed
            if pd.notna(sold):
                return sold
            return listed

        idx["w_sold"] = w_sold.fillna(0)
        idx["IDX"] = idx.apply(_blend, axis=1)
        idx = idx.drop(columns=["w_sold"])
        return idx

    def _msa_state_baselines(self, idx_zip: pd.DataFrame) -> pd.DataFrame:
        idx_zip = idx_zip.copy()
        nat = idx_zip.groupby("YM")["IDX"].median().rename("NAT_IDX_RAW")
        state_idx = idx_zip.groupby(["STATE_MODE", "YM"])["IDX"].median().rename("STATE_IDX_RAW")
        msa_idx = (
            idx_zip[idx_zip["MSA_ID"].notna()].groupby(["MSA_ID", "YM"]).agg({"IDX": "median", "N_SOLD": "sum"}).rename(columns={"IDX": "MSA_IDX_RAW", "N_SOLD": "MSA_NSOLD"})
        )
        z = idx_zip.merge(state_idx.reset_index(), on=["STATE_MODE", "YM"], how="left")
        z = z.merge(nat.reset_index(), on="YM", how="left")
        z = z.merge(msa_idx.reset_index(), on=["MSA_ID", "YM"], how="left")
        msa_nsold = z["MSA_NSOLD"].fillna(0)
        w_msa = msa_nsold / (msa_nsold + self.alpha_msa)
        z["MSA_IDX_BLEND"] = w_msa * z["MSA_IDX_RAW"].fillna(0) + (1 - w_msa) * z["STATE_IDX_RAW"].fillna(z["NAT_IDX_RAW"])
        z["MSA_IDX_BLEND"] = z["MSA_IDX_BLEND"].where(z["MSA_IDX_RAW"].notna(), z["STATE_IDX_RAW"].fillna(z["NAT_IDX_RAW"]))
        z["STATE_IDX_BLEND"] = z["STATE_IDX_RAW"].fillna(z["NAT_IDX_RAW"])
        keep = list(idx_zip.columns) + ["MSA_IDX_BLEND", "STATE_IDX_BLEND"]
        return z[keep]

    def _hometype_share(self, base: pd.DataFrame) -> pd.DataFrame:
        base_ht = base.copy()
        base_ht["HOMETYPE_TXT"] = base_ht["HOMETYPE"].fillna("UNK").astype(str)
        counts = base_ht.groupby(["ZIPCODE", "YM", "HOMETYPE_TXT"]).size().rename("CNT").reset_index()
        denom = counts.groupby(["ZIPCODE", "YM"])["CNT"].sum().rename("N_ALL").reset_index()
        counts = counts.merge(denom, on=["ZIPCODE", "YM"], how="left")
        counts["SHARE"] = counts["CNT"] / counts["N_ALL"].replace(0, np.nan)
        pivot = counts.pivot_table(index=["ZIPCODE", "YM"], columns="HOMETYPE_TXT", values="SHARE")
        if pivot.empty:
            return counts[["ZIPCODE", "YM"]].drop_duplicates()
        pivot = pivot.rename(columns=lambda c: f"HT_SHARE__{_slugify(c)}")
        return pivot.reset_index()

    def _numeric_and_binary_aggregates(self, base: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        groups = ["ZIPCODE", "YM"]
        num_frames: List[pd.DataFrame] = []
        for col in NUM_COLS_NUMERIC + NUM_COLS_TEXT_TO_NUMERIC + COORDS_NUMERIC:
            if col not in base.columns:
                continue
            values = pd.to_numeric(base[col], errors="coerce")
            agg = base.assign(_val=values).groupby(groups)["_val"].agg(["mean", "median", "std"])
            agg = agg.rename(columns={"mean": f"{col}__MEAN", "median": f"{col}__MEDIAN", "std": f"{col}__STD"})
            num_frames.append(agg)
        if num_frames:
            num_agg = pd.concat(num_frames, axis=1).reset_index()
        else:
            num_agg = base[groups].drop_duplicates()

        bin_frames: List[pd.DataFrame] = []
        for col in BIN_COLS_NUMERIC_01:
            if col not in base.columns:
                continue
            values = safe_to_binary_from_number(base[col])
            agg = base.assign(_val=values).groupby(groups)["_val"].mean().rename(f"{col}__SHARE")
            bin_frames.append(agg)
        for col in BIN_COLS_TEXT_YN:
            if col not in base.columns:
                continue
            values = safe_to_binary_from_text(base[col])
            agg = base.assign(_val=values).groupby(groups)["_val"].mean().rename(f"{col}__SHARE")
            bin_frames.append(agg)
        if bin_frames:
            bin_agg = pd.concat(bin_frames, axis=1).reset_index()
        else:
            bin_agg = base[groups].drop_duplicates()
        return num_agg, bin_agg

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base = self._base()
        base = self._filter_extreme_values(base)
        idx_zip = self._index_with_shrinkage(base)
        idx_sp = self._msa_state_baselines(idx_zip)
        ht_share_sp = self._hometype_share(base)
        num_agg_sp, bin_agg_sp = self._numeric_and_binary_aggregates(base)
        return idx_sp, ht_share_sp, num_agg_sp, bin_agg_sp


# --------------------------------------------------------------------------------------
# ZipIndexFeatureizer – pandas adaptation
# --------------------------------------------------------------------------------------


@dataclass
class ZipIndexFeatureizer:
    idx_sp: pd.DataFrame
    ht_share_sp: pd.DataFrame
    num_agg_sp: pd.DataFrame
    bin_agg_sp: pd.DataFrame

    def build(self) -> pd.DataFrame:
        feats = self.idx_sp.merge(self.ht_share_sp, on=["ZIPCODE", "YM"], how="left")
        feats = feats.merge(self.num_agg_sp, on=["ZIPCODE", "YM"], how="left")
        feats = feats.merge(self.bin_agg_sp, on=["ZIPCODE", "YM"], how="left")
        feats = feats.sort_values(["ZIPCODE", "YM"]).reset_index(drop=True)
        group = feats.groupby("ZIPCODE", group_keys=False)
        for lag, name in [(1, "IDX_LAG_1"), (2, "IDX_LAG_2"), (3, "IDX_LAG_3"), (6, "IDX_LAG_6")]:
            feats[name] = group["IDX"].shift(lag)
        feats["IDX_ROLL_STD_3"] = group["IDX"].transform(lambda s: s.rolling(window=3, min_periods=1).std())
        feats["IDX_ROLL_STD_6"] = group["IDX"].transform(lambda s: s.rolling(window=6, min_periods=1).std())
        feats["IDX_PCT_D1"] = (feats["IDX"] / feats["IDX_LAG_1"]) - 1
        feats["IDX_MOM_3"] = (feats["IDX"] / feats["IDX_LAG_3"]) - 1
        feats["IDX_MOM_6"] = (feats["IDX"] / feats["IDX_LAG_6"]) - 1
        feats["MONTH_NUM"] = _ensure_datetime(feats["YM"]).dt.month
        feats["MONTH_SIN"] = np.sin(feats["MONTH_NUM"] * 2 * np.pi / 12.0)
        feats["MONTH_COS"] = np.cos(feats["MONTH_NUM"] * 2 * np.pi / 12.0)
        for lag, prefix in [(1, "L1"), (3, "L3"), (12, "L12")]:
            feats[f"MORTGAGE_{prefix}"] = group["MORTGAGE_RATE_M"].shift(lag)
            feats[f"UNEMP_{prefix}"] = group["UNEMPLOYMENT_RATE_M"].shift(lag)
            feats[f"MORTGAGE_D{lag}"] = feats["MORTGAGE_RATE_M"] - feats[f"MORTGAGE_{prefix}"]
            feats[f"UNEMP_D{lag}"] = feats["UNEMPLOYMENT_RATE_M"] - feats[f"UNEMP_{prefix}"]
        if "AVERAGEHOUSEVALUE__MEAN" in feats.columns and "INCOMEPERHOUSEHOLD__MEAN" in feats.columns:
            denom = feats["INCOMEPERHOUSEHOLD__MEAN"].replace({0: np.nan})
            feats["AFFORD_RATIO"] = feats["AVERAGEHOUSEVALUE__MEAN"] / denom
        else:
            feats["AFFORD_RATIO"] = np.nan
        feats["N_SOLD_SAFE"] = feats["N_SOLD"].fillna(0)
        feats["N_LIST_SAFE"] = feats["N_LIST"].fillna(0)
        feats["LIQ_LOG"] = np.log1p(feats["N_SOLD_SAFE"])
        feats["AFFORD_X_MORT_D12"] = feats["AFFORD_RATIO"] * feats["MORTGAGE_D12"]
        feats["MOM6_X_UNEMP_D12"] = feats["IDX_MOM_6"] * feats["UNEMP_D12"]
        feats["IDX_FUTURE_H1"] = group["IDX"].shift(-1)
        feats["IDX_FUTURE_H2"] = group["IDX"].shift(-2)
        feats["Y_H1"] = np.log1p(feats["IDX_FUTURE_H1"]) - np.log1p(feats["IDX"])
        feats["Y_H2"] = np.log1p(feats["IDX_FUTURE_H2"]) - np.log1p(feats["IDX"])
        feats["DAY_FOR_SPLIT"] = _ensure_datetime(feats["YM"]).dt.floor("D")
        feats = feats[feats["IDX"].notna()].copy()
        return feats


# --------------------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------------------


def _train_only_winsorize(df: pd.DataFrame, label_col: str, group_cols: List[str], k: float, trn_mask: pd.Series) -> None:
    train_df = df.loc[trn_mask, group_cols + [label_col]].dropna(subset=[label_col])
    fences: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    if not train_df.empty:
        grouped = train_df.groupby(group_cols, dropna=False)[label_col]
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        fences = pd.concat([lo.rename("lo"), hi.rename("hi")], axis=1).to_dict("index")

    def _clip(row: pd.Series) -> float:
        key = tuple(row[c] for c in group_cols)
        val = row[label_col]
        if pd.isna(val):
            return val
        fence = fences.get(key)
        if fence is None:
            return val
        return float(np.clip(val, fence["lo"], fence["hi"]))

    df[label_col] = df.apply(_clip, axis=1)


def _eval_on_level(idx_now: np.ndarray, y_ld_true: np.ndarray, y_ld_hat: np.ndarray) -> Dict[str, float]:
    idx_now = np.asarray(idx_now, float)
    y_ld_true = np.asarray(y_ld_true, float)
    y_ld_hat = np.asarray(y_ld_hat, float)
    mask = np.isfinite(idx_now) & np.isfinite(y_ld_true) & np.isfinite(y_ld_hat)
    if not mask.any():
        return dict(mae=np.nan, r2=np.nan, wape=np.nan, mdape=np.nan, pct10=np.nan)
    true_future = np.expm1(np.log1p(idx_now[mask]) + y_ld_true[mask])
    pred_future = np.expm1(np.log1p(idx_now[mask]) + y_ld_hat[mask])
    mae = mean_absolute_error(true_future, pred_future)
    r2 = r2_score(true_future, pred_future) if len(true_future) > 1 else np.nan
    denom = np.abs(true_future).sum()
    wape_val = np.nan if denom == 0 else np.abs(true_future - pred_future).sum() / denom
    mdape_val = np.nanmedian(np.abs((true_future - pred_future) / np.clip(np.abs(true_future), 1e-9, None)))
    pct10_val = float(np.mean(np.abs(pred_future - true_future) <= 0.10 * np.abs(true_future)))
    return dict(mae=mae, r2=r2, wape=wape_val, mdape=mdape_val, pct10=pct10_val)


def _corr_prune(df_train: pd.DataFrame, cols: List[str], thr: float = 0.98, protected: Optional[List[str]] = None, keep_medians: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, int]]:
    protected = set(protected or [])
    keep_medians = set(keep_medians or [])
    cols = [c for c in cols if c in df_train.columns]
    if not cols:
        return cols, {"before": 0, "after": 0}
    X = df_train[cols].astype(float)
    nun = X.nunique()
    keep = nun[nun > 1].index.tolist()
    X = X[keep]
    corr = X.corr(method="pearson", min_periods=100)
    to_drop: set[str] = set()
    order = list(X.columns)
    for i, ci in enumerate(order):
        if ci in to_drop:
            continue
        for j in range(i + 1, len(order)):
            cj = order[j]
            if cj in to_drop:
                continue
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= thr:
                if cj in protected or cj in keep_medians:
                    to_drop.add(ci)
                    break
                if ci in protected or ci in keep_medians:
                    to_drop.add(cj)
                else:
                    vi = np.nanvar(X[ci].values)
                    vj = np.nanvar(X[cj].values)
                    if vi >= vj:
                        to_drop.add(cj)
                    else:
                        to_drop.add(ci)
                        break
    kept = [c for c in cols if (c in X.columns and c not in to_drop)]
    return kept, {"before": len(cols), "after": len(kept)}


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def run_pipeline(raw_table: Path = RAW_TABLE_PATH) -> None:
    builder = CombinedEventsBuilder(raw_table=raw_table)
    combined_events = builder.build()
    if combined_events.empty:
        raise RuntimeError("Combined events DataFrame is empty. Check PRICEHISTORY parsing.")
    preview = combined_events.head(5)
    print("Combined events preview (first 5 rows):")
    with pd.option_context("display.max_columns", None):
        print(preview.to_string(index=False))
    print("Starting ALL-FEATURES pipeline (ZIP index + Aggregates) with XGBoost…")
    idx_sp, ht_share_sp, num_agg_sp, bin_agg_sp = ZipMonthIndexBuilder(combined_events).build()
    feat_sp = ZipIndexFeatureizer(idx_sp, ht_share_sp, num_agg_sp, bin_agg_sp).build()
    max_evt_day = pd.to_datetime(combined_events["EVT_DATE"], errors="coerce").max()
    effective_train_end = max_evt_day - pd.Timedelta(days=HOLDOUT_DAYS)
    holdout_start = effective_train_end + pd.Timedelta(days=1)
    print(
        f"max_day={max_evt_day.date()} | effective_train_end={effective_train_end.date()} | holdout=[{holdout_start.date()} … {max_evt_day.date()}]"
    )
    s = feat_sp[feat_sp["DAY_FOR_SPLIT"] <= effective_train_end].copy()
    df = s.reset_index(drop=True)
    base_col = None
    for cand in ["MSA_IDX_BLEND", "STATE_IDX_BLEND"]:
        if cand in df.columns and df[cand].notna().any():
            base_col = cand
            break
    if base_col:
        df = df.sort_values(["ZIPCODE", "YM"]).reset_index(drop=True)
        df["_BASE_NOW"] = pd.to_numeric(df[base_col], errors="coerce")
        df["_BASE_FWD1"] = df.groupby("ZIPCODE")["_BASE_NOW"].shift(-1)
        df["_BASE_FWD2"] = df.groupby("ZIPCODE")["_BASE_NOW"].shift(-2)
        df["BASE_DLOG_H1"] = np.log1p(df["_BASE_FWD1"]) - np.log1p(df["_BASE_NOW"])
        df["BASE_DLOG_H2"] = np.log1p(df["_BASE_FWD2"]) - np.log1p(df["_BASE_NOW"])
    cutoff_start = effective_train_end - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    trn_mask = df["DAY_FOR_SPLIT"] < cutoff_start
    hld_mask = ~trn_mask
    group_cols = ["STATE_MODE", "YM"] if "STATE_MODE" in df.columns else ["YM"]
    if "Y_H1" in df.columns:
        _train_only_winsorize(df, "Y_H1", group_cols, k=1.5, trn_mask=trn_mask)
    if "Y_H2" in df.columns:
        _train_only_winsorize(df, "Y_H2", group_cols, k=3.0, trn_mask=trn_mask)
    label_cols = {"Y_H1", "Y_H2", "IDX_FUTURE_H1", "IDX_FUTURE_H2"}
    non_feature_keys = {"ZIPCODE", "YM", "STATE_MODE", "COUNTY_MODE", "DAY_FOR_SPLIT"}
    all_cols = set(df.columns)
    feat_cols = sorted([c for c in all_cols if c not in (EXPLICIT_DROPS | non_feature_keys | label_cols)])
    future_cols = ["BASE_DLOG_H1", "BASE_DLOG_H2", "_BASE_FWD1", "_BASE_FWD2", "_BASE_NOW"]
    feat_cols = [c for c in feat_cols if c not in future_cols]
    for col in feat_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feat_cols] = df[feat_cols].fillna(0.0)
    feat_cols = [c for c in feat_cols if df.loc[trn_mask, c].nunique() > 1]
    print(f"Pre-prune features: {len(feat_cols)}")
    boost_feats = {
        "MORTGAGE_RATE_M": 1.4,
        "UNEMPLOYMENT_RATE_M": 1.2,
        "MORTGAGE_D1": 1.2,
        "UNEMP_D1": 1.1,
        "MORTGAGE_D3": 1.2,
        "UNEMP_D3": 1.1,
        "MORTGAGE_D12": 1.2,
        "UNEMP_D12": 1.1,
        "IDX_LAG_1": 1.3,
        "IDX_LAG_2": 1.2,
    }
    for col in feat_cols:
        if col.startswith("HT_SHARE__"):
            boost_feats[col] = boost_feats.get(col, 1.15)
    prune_cfg = {
        "H1": dict(
            threshold=0.98,
            protected=["AVERAGEHOUSEVALUE__MEAN", "INCOMEPERHOUSEHOLD__MEAN", "MORTGAGE_RATE_M", "UNEMPLOYMENT_RATE_M"],
            keep_medians=["MEDIAN_LISTING_PRICE__MEDIAN"],
        ),
        "H2": dict(
            threshold=0.995,
            protected=["AVERAGEHOUSEVALUE__MEAN", "INCOMEPERHOUSEHOLD__MEAN", "MORTGAGE_RATE_M", "UNEMPLOYMENT_RATE_M"],
            keep_medians=["MEDIAN_LISTING_PRICE__MEDIAN", "AVERAGEHOUSEVALUE__MEDIAN"],
        ),
    }
    results: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Optional[xgb.Booster]] = {}
    dims: Dict[str, Dict[str, int]] = {}
    prune_logs: Dict[str, Dict[str, Any]] = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    param_grid_tried: Dict[str, List[Dict[str, Any]]] = {}
    for h, ycol, tag in [(1, "Y_H1", "H1"), (2, "Y_H2", "H2")]:
        print(f"\n[H={h*30}] training window ≤ {effective_train_end.date()} | holdout last {HOLDOUT_DAYS} days")
        has_y_trn = trn_mask & df[ycol].notna() & df["IDX"].notna()
        has_y_hld = hld_mask & df[ycol].notna() & df["IDX"].notna()
        X_trn = df.loc[has_y_trn, feat_cols].copy()
        y_trn = df.loc[has_y_trn, ycol].astype(float).values
        X_hld = df.loc[has_y_hld, feat_cols].copy()
        y_hld_ld = df.loc[has_y_hld, ycol].astype(float).values
        idx_now_hld = df.loc[has_y_hld, "IDX"].astype(float).values
        resid_used = False
        y_val_label = y_hld_ld
        base_hld_slice = None
        m_hld = None
        if tag == "H2" and base_col and f"BASE_DLOG_H{h}" in df.columns:
            base_trn = pd.to_numeric(df.loc[has_y_trn, f"BASE_DLOG_H{h}"], errors="coerce").values
            base_hld = pd.to_numeric(df.loc[has_y_hld, f"BASE_DLOG_H{h}"], errors="coerce").values
            m_trn = np.isfinite(base_trn) & np.isfinite(y_trn)
            m_hld = np.isfinite(base_hld) & np.isfinite(y_hld_ld)
            cov_trn = float(m_trn.mean())
            cov_hld = float(m_hld.mean())
            print(f"[H=60] Baseline coverage — train={100*cov_trn:.1f}% | holdout={100*cov_hld:.1f}%")
            if cov_trn >= 0.90 and cov_hld >= 0.90:
                X_trn = X_trn.loc[m_trn].copy()
                y_trn = (y_trn[m_trn] - base_trn[m_trn]).astype(float)
                X_hld = X_hld.loc[m_hld].copy()
                y_hld_ld_resid = (y_hld_ld[m_hld] - base_hld[m_hld]).astype(float)
                idx_now_hld = idx_now_hld[m_hld]
                base_hld_slice = base_hld[m_hld]
                y_val_label = y_hld_ld_resid
                resid_used = True
                print("[H=60] Residualized labels to MSA/State baseline (robust).")
            else:
                print("[H=60] Residualization disabled (insufficient baseline coverage).")
        dims[tag] = dict(train_rows=int(len(X_trn)), holdout_rows=int(len(X_hld)), n_features=int(X_trn.shape[1]))
        print(f"[H={h*30}] rows — train={len(X_trn):,} | holdout={len(X_hld):,} | features(pre-prune)={X_trn.shape[1]}")
        if len(X_trn) == 0 or len(X_hld) == 0:
            print(f"[H={h*30}] No labeled rows; skipping.")
            results[tag] = dict(mae=np.nan, r2=np.nan, wape=np.nan, mdape=np.nan, pct10=np.nan)
            models[tag] = None
            continue
        pcfg = prune_cfg[tag]
        kept_cols, logp = _corr_prune(X_trn, list(X_trn.columns), thr=pcfg["threshold"], protected=pcfg["protected"], keep_medians=pcfg["keep_medians"])
        print(f"Correlation prune {tag} @|r|≥{pcfg['threshold']}: {X_trn.shape[1]} → {len(kept_cols)}")
        prune_logs[tag] = dict(enabled=True, threshold=pcfg["threshold"], protected=pcfg["protected"], keep_medians=pcfg["keep_medians"], before=logp["before"], after=logp["after"])
        X_trn = X_trn[kept_cols]
        X_hld = X_hld[kept_cols]
        fw = np.ones(len(kept_cols), dtype=float)
        col_to_idx = {c: i for i, c in enumerate(kept_cols)}
        for name, weight in boost_feats.items():
            if name in col_to_idx:
                fw[col_to_idx[name]] = float(weight)
        if tag == "H1":
            ns = df.loc[has_y_trn, "N_SOLD"].astype(float).fillna(0).values
            if len(ns) != len(X_trn):
                ns = ns[: len(X_trn)]
            w_row_trn = np.clip(np.log1p(ns) / 2.5, 0.10, 1.0)
            print("Reliability weights: using w = clip(log1p(N_SOLD)/2.5, 0.10, 1)")
        else:
            w_row_trn = np.ones(len(X_trn), dtype=float)
            print("Reliability weights: disabled for H2.")
        if tag == "H1":
            grid = [
                dict(min_child_weight=mcw, gamma=g, reg_lambda=rl, eta=0.05, max_depth=6)
                for mcw in [1, 5, 10]
                for g in [0.0, 0.1, 0.3]
                for rl in [1.0, 3.0, 5.0]
            ]
        else:
            grid = [
                dict(min_child_weight=mcw, gamma=g, reg_lambda=rl, eta=0.03, max_depth=7)
                for mcw in [1, 5, 10]
                for g in [0.0, 0.1, 0.3]
                for rl in [1.0, 3.0, 5.0]
            ]
        best = {"mae": np.inf, "iter": None, "params": None, "yhat": None, "model": None}
        tried: List[Dict[str, Any]] = []
        dval = xgb.DMatrix(X_hld.values, label=y_val_label, feature_names=kept_cols)
        for params in grid:
            xgb_params = dict(
                objective="reg:squarederror",
                eval_metric="mae",
                eta=params["eta"],
                max_depth=params["max_depth"],
                subsample=0.8,
                colsample_bytree=0.9,
                min_child_weight=params["min_child_weight"],
                gamma=params["gamma"],
                reg_lambda=params["reg_lambda"],
                reg_alpha=0.0,
                tree_method="hist",
                seed=RANDOM_SEED,
            )
            dtrn = xgb.DMatrix(
                X_trn.values,
                label=y_trn,
                feature_names=kept_cols,
                feature_weights=fw,
                weight=w_row_trn,
            )
            model = xgb.train(
                xgb_params,
                dtrn,
                num_boost_round=6000,
                evals=[(dval, "val")],
                early_stopping_rounds=400,
                verbose_eval=False,
            )
            yhat_ld = model.predict(dval, iteration_range=(0, (model.best_iteration or 0) + 1))
            if tag == "H2" and resid_used:
                ytrue_eval = df.loc[has_y_hld, "Y_H2"].astype(float).values[m_hld]
                yhat_eval = yhat_ld + base_hld_slice
            else:
                ytrue_eval = y_hld_ld
                yhat_eval = yhat_ld

            metrics = _eval_on_level(idx_now_hld, ytrue_eval, yhat_eval)
            tried.append({"params": params, "val_mae": float(metrics["mae"]), "best_iter": int(model.best_iteration or 0)})
            if metrics["mae"] < best["mae"]:
                best.update(mae=float(metrics["mae"]), iter=int(model.best_iteration or 0), params=params.copy(), yhat=yhat_ld, model=model)

        param_grid_tried[tag] = tried
        models[tag] = best["model"]
        best_params[tag] = dict(p=best["params"], best_iter=best["iter"])
        print(f"[0]\tval-mae: (grid best so far) {best['mae']:.5f}")
        print(f"[H={h*30}] best params @ depth={best['params']['max_depth']}: {best['params']} | iter={best['iter']}")

        if tag == "H2":
            if resid_used:
                ytrue_eval = df.loc[has_y_hld, "Y_H2"].astype(float).values[m_hld]
                yhat_eval = best["yhat"] + base_hld_slice
            else:
                ytrue_eval = y_hld_ld
                yhat_eval = best["yhat"]
            met_final = _eval_on_level(idx_now_hld, ytrue_eval, yhat_eval)
            met_final["used_blend"] = False
            final_yhat = yhat_eval
        else:
            try:
                lgb_model = lgb.LGBMRegressor(
                    objective="huber",
                    learning_rate=0.05,
                    n_estimators=5000,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    num_leaves=63,
                    reg_lambda=best["params"]["reg_lambda"],
                    random_state=RANDOM_SEED,
                )
                lgb_model.fit(
                    X_trn,
                    y_trn,
                    eval_set=[(X_hld, y_val_label)],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
                )
                yhat_ld_lgb = lgb_model.predict(X_hld, num_iteration=lgb_model.best_iteration_)
                yhat_ld_blend = 0.70 * best["yhat"] + 0.30 * yhat_ld_lgb
                met_xgb = _eval_on_level(idx_now_hld, y_val_label, best["yhat"])
                met_blend = _eval_on_level(idx_now_hld, y_val_label, yhat_ld_blend)
                use_blend = (met_blend["mae"] <= met_xgb["mae"]) or (met_blend["mdape"] < met_xgb["mdape"])
                final_yhat = yhat_ld_blend if use_blend else best["yhat"]
                met_final = met_blend if use_blend else met_xgb
                met_final["used_blend"] = bool(use_blend)
                met_final["mae_xgb_only"] = float(met_xgb["mae"])
                met_final["mae_blend"] = float(met_blend["mae"])
                met_final["blend_weights"] = [0.70, 0.30]
            except Exception as exc:  # pragma: no cover - defensive fallback
                met_final = _eval_on_level(idx_now_hld, y_val_label, best["yhat"])
                met_final["used_blend"] = False
                met_final["blend_error"] = str(exc)
                final_yhat = best["yhat"]

        try:
            idx_now = df.loc[has_y_hld, "IDX"].astype(float).values
            if tag == "H2":
                y_ld_true_eval = ytrue_eval
                y_ld_hat_eval = final_yhat
            else:
                y_ld_true_eval = y_val_label
                y_ld_hat_eval = final_yhat
            true_lvl = np.expm1(np.log1p(idx_now) + y_ld_true_eval)
            pred_lvl = np.expm1(np.log1p(idx_now) + y_ld_hat_eval)
            res = pd.DataFrame(
                {
                    "STATE": df.loc[has_y_hld, "STATE_MODE"].astype(str).values if "STATE_MODE" in df.columns else "NA",
                    "N_SOLD": df.loc[has_y_hld, "N_SOLD"].astype(float).values if "N_SOLD" in df.columns else np.nan,
                    "TRUE": true_lvl,
                    "PRED": pred_lvl,
                }
            )
            res["ABS_ERR"] = np.abs(res["TRUE"] - res["PRED"])
            if "N_SOLD" in res.columns:
                res["LIQ_DEC"] = pd.qcut(res["N_SOLD"].fillna(-1), q=10, duplicates="drop")
            res["P_DEC"] = pd.qcut(res["TRUE"], q=10, duplicates="drop")
            by_state = res.groupby("STATE")["ABS_ERR"].mean().sort_values(ascending=False).head(8)
            by_liq = res.groupby("LIQ_DEC")["ABS_ERR"].mean() if "LIQ_DEC" in res else pd.Series()
            by_price = res.groupby("P_DEC")["ABS_ERR"].mean()
            print(f"[H={h*30}] Residual slice — worst states (MAE):\n{by_state.to_string()}")
            if not by_liq.empty:
                print(f"[H={h*30}] Residual slice — MAE by liquidity decile (N_SOLD):\n{by_liq.to_string()}")
            print(f"[H={h*30}] Residual slice — MAE by price decile:\n{by_price.to_string()}")
            met_final["slices"] = {"state_top8": by_state.to_dict()}
        except Exception:
            pass

        results[tag] = met_final

    ts = time.strftime("%Y%m%d-%H%M%S")
    runname = "zipmonth_xgb_h2_residualized"
    Path("runlog").mkdir(exist_ok=True)
    artifacts = dict(
        models_meta={k: (v.attributes() if v is not None else None) for k, v in models.items()},
        feature_columns=feat_cols,
        dims=dims,
        results=results,
        split=dict(max_day=str(max_evt_day.date()), effective_train_end=str(effective_train_end.date()), holdout_start=str(holdout_start.date())),
        corr_prune=prune_logs,
        best_params=best_params,
        param_grid_sizes={k: len(v) for k, v in param_grid_tried.items()},
    )
    json_path = Path("runlog") / f"{ts}__{runname}.json"
    json_path.write_text(json.dumps(artifacts, indent=2))
    print(f"Saved artifacts → {json_path}")
    model_files = {}
    for tag in ("H1", "H2"):
        if models.get(tag) is not None:
            mp = Path("runlog") / f"{ts}__{runname}__{tag}_xgb.json"
            models[tag].save_model(str(mp))
            model_files[tag] = str(mp)
    print("Saved model files:", model_files)
    print("\n=== Dimensions ==="); print(dims)
    print("\n=== Results (per horizon; 'used_blend' means Huber blend beat XGB) ==="); print(results)
    print("\n=== Split ==="); print(dict(max_day=str(max_evt_day.date()), effective_train_end=str(effective_train_end.date()), holdout_start=str(holdout_start.date())))
    print("\n=== Corr-prune log (H1/H2) ==="); print(prune_logs)
    print("\n=== Best XGB params (per horizon) ==="); print(best_params)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()

