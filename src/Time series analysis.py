# ============================================
# IMPORTS & GLOBAL CONFIG
# ============================================
import os, gc, math, re, warnings, json, time, random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.window import Window
from snowflake.snowpark import Session, DataFrame as SnowparkDF
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark.functions import (
    col as sp_col,
    parse_json, flatten,
    count, avg, sum as sf_sum,
    max as sf_max, min as sf_min,
    stddev_samp, sql_expr, when, lower, trim, to_date
)
from snowflake.snowpark.types import FloatType, StringType, BooleanType

from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from sklearn.metrics import (
    mean_absolute_error, r2_score, roc_auc_score, log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

session = get_active_session()
RAW_TABLE = "APIFY_FOR_SALE_ENCODED_DEDUPED"

# ----------------- CONFIG -----------------
MIN_START_DATE      = '2022-01-01'  # enforce 2022+
HOLDOUT_DAYS        = 60
MIN_SOLD_PER_ZIP_M  = 20
MIN_LIST_PER_ZIP_M  = 40
TOPK_HOMETYPES      = 6
CHUNK_LIMIT_ROWS    = 800_000
RANDOM_SEED         = 42

# ----------------- FEATURE CATALOGS -----------------
KEY_COLS = [
    "ZPID","URL","STREETADDRESS","CITY","STATE","COUNTY","ZIPCODE",
    "FIPS","FIPSCODE","STATEFIPS","COUNTYFIPS",
    "MetroAreaID","MetroAreaTitle","DivisionCode","REGION","DIVISION","CITYTYPE"
]

EXPLICIT_DROPS = set([
    "ZPID_KEY","EVT_SOURCE","EVT_MLS_ID","EVT_MLS_NAME","RAW_TIME_MS_STR","JSON_INDEX",
    "BASE_ZPID","URL","STREETADDRESS","HOMESTATUSTEXT","KEYSTONEHOMESTATUS","LOT","basement",
    "LEVELS","PARKING","PARKINGFEATURES","COOLINGFEATURES","HEATINGFEATURES","FIREPLACE",
    "DESCRIPTION","ELEMENTARYSCHOOLNAME","MIDDLESCHOOLNAME","HIGHSCHOOLNAME"
])

REQUIRED_COLS = {
    "ZPID","EVT_DATE","EVT_TYPE","EVT_PRICE","EVT_IS_RENTAL",
    "ZIPCODE","STATE","COUNTY","HOMETYPE",
    "WEEKLY_AVERAGE_MORTGAGE_RATE","UNEMPLOYMENT_RATE"
}
COORDS_NUMERIC = ["LONGITUDE","LATITUDE"]

NUM_COLS_NUMERIC = [
    "PRICE","SQFT","LOTSQFT",
    "WALKSCORE","TRANSITSCORE","UNEMPLOYMENT_RATE",
    "MEDIAN_DAYS_ON_MARKET","MEDIAN_LISTING_PRICE",
    "SUPPLY_SCORE","ACTIVE_LISTING_COUNT",
    "HOTNESS_SCORE","DEMAND_SCORE","HOTNESS_RANK",
    "FM_HPI","PROPERTY_AGE","WEEKLY_AVERAGE_MORTGAGE_RATE",
    "MONTH","YEAR","CREATEDAT_MONTH","CREATEDAT_YEAR",
    "POPULATION","HOUSEHOLDSPERZIPCODE",
    "WHITEPOPULATION","BLACKPOPULATION","HISPANICPOPULATION",
    "ASIANPOPULATION","HAWAIIANPOPULATION","INDIANPOPULATION","OTHERPOPULATION",
    "MALEPOPULATION","FEMALEPOPULATION","PERSONSPERHOUSEHOLD","AVERAGEHOUSEVALUE",
    "INCOMEPERHOUSEHOLD","MEDIANAGE","MEDIANAGEMALE","MEDIANAGEFEMALE",
    "NUMBEROFBUSINESSES","NUMBEROFEMPLOYEES","BUSINESSANNUALPAYROLL",
    "GROWTHRANK","GROWTHINCREASENUMBER","GROWTHINCREASEPERCENTAGE",
    "POPULATIONESTIMATE","LANDAREA","WATERAREA",
    "VALUE_2_UNITS_REP_M","UNITS_3_4_UNITS_REP_M","VALUE_5_UNITS_REP_M",
    "UNITS_1_UNIT_REP_M","UNITS_2_UNITS_REP_M","VALUE_1_UNIT_REP_M","UNITS_5_UNITS_REP_M",
    "RN"
]

NUM_COLS_TEXT_TO_NUMERIC = [
    "BEDROOMS","BATHROOMS","FULLBATHROOMS","HALFBATHROOMS",
    "YEARBUILT","HOAFEE","PARKINGTOTALSPACES",
    "ELEMENTARYSCHOOLDISTANCE","MIDDLESCHOOLDISTANCE","HIGHSCHOOLDISTANCE",
    "ELEMENTARYSCHOOLRATING","MIDDLESCHOOLRATING","HIGHSCHOOLRATING",
    "TOURVIEWCOUNT"
]

BIN_COLS_NUMERIC_01 = [
    "FIREPLACEYN","NEWCONSTRUCTIONFLAG","HASHOA","SENIORLIVING",
    "ONSTREETPARKING","GARAGEPARKING","ATTACHEDPARKING","DETACHEDPARKING",
    "DRIVEWAY","OFFSTREETPARKING","NOPARKING",
    "RADIATORHEATING","CENTRALHEATING","FORCEDAIRHEATING","SOLARHEATING",
    "ELECTRICHEATING","ZONEDHEATING","HOTWATERHEATING","OILHEATING",
    "PROPANEHEATING","NATURALGASHEATING","NOHEATINGINFO",
    "CENTRALCOOLING","WINDOWUNITACCOOLING","WALLUNITACCOOLING",
    "MULTIUNITCOOLING","ZONEDCOOLING","NOCOOLINGINFO",
    "NOPOOLFEATURES","ABOVEGROUNDPOOL","INGROUNDPOOL","PERSONALPOOL",
    "FENCEDPOOL","INDOORPOOL","HEATEDPOOL","FILTEREDPOOL","SALTWATERPOOL",
    "POOLMATERIALVINYL","POOLMATERIALCONCRETE","POOLMATERIALGUNITE",
    "SINGLEFAMILY","TOWNHOUSE"
]

BIN_COLS_TEXT_YN = ["basementYN"]

CATEGORICAL_TEXT = [
    "HOMETYPE","HOMESTATUS","KEYSTONEHOMESTATUS","HOMESTATUSTEXT",
    "CITY","STATE","COUNTY","ZIPCODE",
    "ZONING","PARCELNUMBER","CONSTRUCTIONMATERIAL","FOUNDATION","ROOF",
    "ARCHITECTURALSTYLE","POOLFEATURES","PARKINGFEATURES",
    "COOLINGFEATURES","HEATINGFEATURES","FIREPLACE",
    "ELEMENTARYSCHOOLNAME","MIDDLESCHOOLNAME","HIGHSCHOOLNAME",
    "CITYTYPE","REGION","DIVISION",
    "MetroAreaTitle","DivisionCode",
    "STATEFIPS","COUNTYFIPS","FIPSCODE","FIPS"
]

DATE_LIKE_TEXT = ["CREATEDAT","DATEPOSTED"]
FREE_TEXT_EXCLUDE = ["URL","STREETADDRESS","DESCRIPTION"]
VARIANT_COLS = ["PRICEHISTORY"]

OPTIONAL_PARSE_LATER = [
    "LEVELS","STORIES","PARKING",
    "CONSTRUCTIONMATERIAL","FOUNDATION","ROOF","ARCHITECTURALSTYLE",
    "POOLFEATURES","COOLINGFEATURES","HEATINGFEATURES",
    "ELEMENTARYSCHOOLGRADES","MIDDLESCHOOLGRADES","HIGHSCHOOLGRADES"
]

EXPLICIT_DROPS = set(FREE_TEXT_EXCLUDE + VARIANT_COLS + KEY_COLS)

NON_FEATURE_KEYS = {"ZIPCODE","YM","STATE_MODE","COUNTY_MODE","DAY_FOR_SPLIT"}

# ============================================
# HELPERS / METRICS
# ============================================
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s):
            df[c] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == "object":
            nun = s.nunique(dropna=False)
            if nun and nun / max(len(s),1) <= 0.4:
                df[c] = df[c].astype("category")
    return df

def wape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return np.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom

def mdape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    pct = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))
    return np.nanmedian(pct)

def pct_within(y_true, y_pred, pct=0.10):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ok = np.abs(y_pred - y_true) <= (pct * np.abs(y_true))
    return float(np.mean(ok)) if len(ok) else np.nan

# ---------- Snowpark-safe coercion gates ----------
def safe_to_double(col):
    s = F.to_varchar(col)
    s = F.regexp_replace(s, r'[,\s\$%]', '')
    return F.call_function("TRY_TO_DOUBLE", s)

def safe_to_binary_from_text(col: F.Column) -> F.Column:
    u = F.upper(F.to_varchar(col))
    return (
        F.iff(u.in_(F.lit("Y"),F.lit("YES"),F.lit("TRUE"),F.lit("T"),F.lit("1")), F.lit(1),
        F.iff(u.in_(F.lit("N"),F.lit("NO"),F.lit("FALSE"),F.lit("F"),F.lit("0")), F.lit(0), F.lit(None)))
    )

def safe_to_binary_from_number(col: F.Column) -> F.Column:
    x = safe_to_double(col)
    return F.iff(x == F.lit(1), F.lit(1),
           F.iff(x == F.lit(0), F.lit(0), F.lit(None)))

# ============================================
# COMBINED EVENTS BUILDER
# ============================================
class CombinedEventsBuilder:
    """
    Creates a time-series 'combined_events' Snowpark DataFrame:
      - one row per PRICEHISTORY event per ZPID
      - event columns are prefixed with 'evt_' to avoid name collisions
      - joined to a single 'base snapshot' row per ZPID (latest by SCRAPEDAT if available)
    """

    def __init__(
        self,
        raw_table: str,
        zpid_col: str = "ZPID",
        pricehistory_col: str = "PRICEHISTORY",
        scrape_ts_col: str = "SCRAPEDAT"
    ):
        self.raw_table = raw_table
        self.zpid_col = zpid_col
        self.pricehistory_col = pricehistory_col
        self.scrape_ts_col = scrape_ts_col

        self.c_zpid_key     = "zpid_key"
        self.c_evt_date     = "evt_date"
        self.c_evt_ts       = "evt_ts"
        self.c_evt_type     = "evt_type"
        self.c_evt_price    = "evt_price"
        self.c_evt_price_psf= "evt_price_psf"
        self.c_evt_is_rent  = "evt_is_rental"
        self.c_evt_source   = "evt_source"
        self.c_evt_mls_id   = "evt_mls_id"
        self.c_evt_mls_name = "evt_mls_name"

        self.c_sort_ts       = "sort_ts"
        self.c_event_seq     = "event_seq"
        self.c_days_prev     = "days_since_prev"
        self.c_days_first    = "days_since_first"

        self.c_base_zpid     = "BASE_ZPID"
        self.c_base_zpid_key = "BASE_ZPID_KEY"

    def build(self):
        base = session.table(self.raw_table)
        events = self._flatten_events(base)
        events = self._add_sequence(events)
        base_snap = self._make_base_snapshot(base)
        combined = (
            events.join(
                base_snap,
                events[self.c_zpid_key] == base_snap[self.c_base_zpid_key],
                how="left",
            )
            .drop(self.c_base_zpid_key)
        )
        return combined

    def _flatten_events(self, base_df):
        base_core = (
            base_df
            .select(
                F.col(self.zpid_col).alias(self.zpid_col),
                F.parse_json(F.col(self.pricehistory_col)).alias("PH_JSON"),
            )
            .filter(F.col("PH_JSON").is_not_null())
            .with_column(self.c_zpid_key, F.col(self.zpid_col).cast("string"))
        )
        flat[self.c_evt_price_psf] = pd.to_numeric(
            v.apply(lambda d: get_val(d, "pricePerSquareFoot", None)),
            errors="coerce"
        )
        flat[self.c_evt_is_rent] = v.apply(
            lambda d: bool(get_val(d, "postingIsRental", False))
        )
        flat[self.c_evt_source] = v.apply(lambda d: get_val(d, "source", None))
        flat[self.c_evt_mls_id] = v.apply(
            lambda d: get_val(get_val(d, "attributeSource", {}) or {}, "infoString1", None)
        )
        flat[self.c_evt_mls_name] = v.apply(
            lambda d: get_val(get_val(d, "attributeSource", {}) or {}, "infoString2", None)
        )
        flat["RAW_TIME_MS_STR"] = v.apply(lambda d: get_val(d, "time", None))

        ms_num  = F.call_function("TO_NUMBER", F.col("raw_time_ms_str"))
        epoch0  = F.to_timestamp_ntz(F.lit("1970-01-01 00:00:00"))
        evt_ts  = F.call_function("DATEADD", F.lit("millisecond"), ms_num, epoch0)

        flat = (
            flat
            .with_column(self.c_evt_ts, evt_ts)
            .with_column(self.c_sort_ts, F.coalesce(F.col(self.c_evt_ts), F.to_timestamp_ntz(F.col(self.c_evt_date))))
            .filter(
                F.coalesce(F.col(self.c_evt_date).is_not_null(), F.col(self.c_evt_ts).is_not_null())
            )
        )
        return flat

    def _add_sequence(self, events_df):
        w_order = Window.partition_by(self.c_zpid_key).order_by(
            F.col(self.c_evt_date), F.col(self.c_sort_ts), F.col("json_index")
        )
        w_all = Window.partition_by(self.c_zpid_key)

        prev_ts  = F.lag(F.col(self.c_sort_ts), 1).over(w_order)
        first_ts = F.min(F.col(self.c_sort_ts)).over(w_all)

        return (
            events_df
            .with_column(self.c_event_seq, F.row_number().over(w_order))
            .with_column(self.c_days_prev,  F.call_function("DATEDIFF", F.lit("day"), prev_ts,  F.col(self.c_sort_ts)))
            .with_column(self.c_days_first, F.call_function("DATEDIFF", F.lit("day"), first_ts, F.col(self.c_sort_ts)))
        )

    def _make_base_snapshot(self, base_df):
        cols_no_json: List[str] = [c for c in base_df.columns if c != self.pricehistory_col]
        base_no_json = base_df.select([F.col(c) for c in cols_no_json])

        if self.scrape_ts_col in base_no_json.columns:
            scrape_ts = F.to_timestamp_ntz(F.col(self.scrape_ts_col).cast("string"))
            w_latest = Window.partition_by(self.zpid_col).order_by(scrape_ts.desc_nulls_last())
            base_latest = (
                base_no_json
                .with_column("_rnk", F.row_number().over(w_latest))
                .filter(F.col("_rnk") == 1)
                .drop("_rnk")
            )
            base_no_json = base_no_json.sort_values(
                [self.zpid_col, self.scrape_ts_col],
                ascending=[True, False]
            )
            base_latest = base_no_json.drop_duplicates(subset=[self.zpid_col], keep="first")
        else:
            agg_exprs = [F.any_value(F.col(c)).alias(c) for c in cols_no_json if c != self.zpid_col]
            base_latest = base_no_json.group_by(self.zpid_col).agg(*agg_exprs)

        base_latest = (
            base_latest
            .with_column_renamed(self.zpid_col, self.c_base_zpid)
            .with_column(self.c_base_zpid_key, F.col(self.c_base_zpid).cast("string"))
        )
        return base_latest

# ============================================
# ZIP×MONTH INDEX BUILDER
# ============================================
class ZipMonthIndexBuilder:
    """
    Build ZIP×month panel with:
      - IDX_RAW: median sale/list level per ZIP×YM (uses sale first, then list)
      - Liquidity: N_SOLD, N_LIST, SOLD_MEDIAN, LIST_MEDIAN
      - Reliability: N_MONTHS_TO_DATE, W_H1_COMBINED
      - Pooling: county/state medians and relatives
      - IDX_EFF: coalesce ZIP/COUNTY/STATE
      - Labels: Y_H1, Y_H2 as Δlog on IDX_EFF
    """

    def __init__(
        self,
        combined_events: SnowparkDF,
        min_start_date: str = "2022-01-01",
        min_sold_per_zip_m: int = 10,
        min_list_per_zip_m: int = 20,
    ):
        self.ce = combined_events
        self.min_start_date = min_start_date
        self.min_sold = int(min_sold_per_zip_m)
        self.min_list = int(min_list_per_zip_m)

    def build(self) -> pd.DataFrame:
        ce = self.ce.copy()

        # ---------------------------------
        # Canonical event date
        # ---------------------------------
        if "EVT_DATE" in ce.columns:
            ce["EVT_DATE"] = pd.to_datetime(ce["EVT_DATE"], errors="coerce")
        elif "evt_date" in ce.columns:
            ce["EVT_DATE"] = pd.to_datetime(ce["evt_date"], errors="coerce")
        else:
            raise KeyError("Combined events missing EVT_DATE/evt_date")

        ce = ce[ce["EVT_DATE"].notna()]
        ce = ce[ce["EVT_DATE"] >= pd.to_datetime(self.min_start_date)]

        # ---------------------------------
        # Month index + split date
        # ---------------------------------
        ce["YM"] = ce["EVT_DATE"].values.astype("datetime64[M]")
        ce["DAY_FOR_SPLIT"] = ce["EVT_DATE"]

        # ---------------------------------
        # Safe creation of ZIPCODE / STATE_MODE / COUNTY_MODE
        # ---------------------------------
        # ZIPCODE: if missing, fill with NA series
        if "ZIPCODE" in ce.columns:
            zip_series = ce["ZIPCODE"]
        else:
            zip_series = pd.Series(pd.NA, index=ce.index)

        # STATE: if missing, fill with NA series
        if "STATE" in ce.columns:
            state_series = ce["STATE"]
        else:
            state_series = pd.Series(pd.NA, index=ce.index)

        # COUNTY: if missing, fill with NA series
        if "COUNTY" in ce.columns:
            county_series = ce["COUNTY"]
        else:
            county_series = pd.Series(pd.NA, index=ce.index)

        # Convert to string dtype; pd.NA is handled correctly inside a Series
        ce["ZIPCODE"]     = zip_series.astype("string")
        ce["STATE_MODE"]  = state_series.astype("string")
        ce["COUNTY_MODE"] = county_series.astype("string")

        # ---------------------------------
        # Event type / price / rental flag
        # ---------------------------------
        price_col = "evt_price" if "evt_price" in ce.columns else "EVT_PRICE"
        evt_type_col = "evt_type" if "evt_type" in ce.columns else "EVT_TYPE"
        rent_col = "evt_is_rental" if "evt_is_rental" in ce.columns else (
            "EVT_IS_RENTAL" if "EVT_IS_RENTAL" in ce.columns else None
        )

        ce["EVT_PRICE"] = pd.to_numeric(ce[price_col], errors="coerce")
        ce["EVT_TYPE"] = ce[evt_type_col].astype(str).str.lower()

        if rent_col is not None:
            ce["EVT_IS_RENTAL"] = ce[rent_col].fillna(False).astype(bool)
        else:
            ce["EVT_IS_RENTAL"] = False

        # Filter out rentals
        ce_nr = ce[~ce["EVT_IS_RENTAL"]].copy()
        print(f"[ZipMonthIndexBuilder] non-rental events: {len(ce_nr):,}")

        # see what EVT_TYPE actually looks like
        print("[ZipMonthIndexBuilder] EVT_TYPE sample:")
        print(ce_nr["EVT_TYPE"].value_counts().head(15))

        listing_like = F.col("EVT_TYPE").in_(
            F.lit("listing"), F.lit("for sale"), F.lit("listed for sale"), F.lit("price change")
        )
        sold_like    = F.col("EVT_TYPE").in_(F.lit("sold"), F.lit("sale"), F.lit("closed"))

        base = (
            ce.filter(F.col("EVT_IS_RENTAL") == F.lit(False))
              .group_by("ZIPCODE","STATE_MODE","COUNTY_MODE","YM")
              .agg(
                  F.sum(F.iff(sold_like & F.col("EVT_PRICE").is_not_null(), F.lit(1), F.lit(0))).alias("N_SOLD"),
                  F.sum(F.iff(listing_like & F.col("EVT_PRICE").is_not_null(), F.lit(1), F.lit(0))).alias("N_LIST"),
                  F.median(F.iff(sold_like & F.col("EVT_PRICE").is_not_null(), F.col("EVT_PRICE"), F.lit(None))).alias("SOLD_MEDIAN"),
                  F.median(F.iff(listing_like & F.col("EVT_PRICE").is_not_null(), F.col("EVT_PRICE"), F.lit(None))).alias("LIST_MEDIAN"),
              )
              .with_column("IDX_RAW", F.coalesce(F.col("SOLD_MEDIAN"), F.col("LIST_MEDIAN")))
        )

        w_zip = Window.partition_by("ZIPCODE").order_by(F.col("YM"))
        base = base.with_column("N_MONTHS_TO_DATE", F.row_number().over(w_zip))

        n_tx = F.greatest(F.coalesce(F.col("N_SOLD"), F.lit(0)), F.coalesce(F.col("N_LIST"), F.lit(0)))
        w_hist_raw = F.least(F.col("N_MONTHS_TO_DATE"), F.lit(24))
        w_hist = F.greatest(
            F.call_function("LN", w_hist_raw + F.lit(1.0)) / F.call_function("LN", F.lit(12.0) + F.lit(1.0)),
            F.lit(0.2)
        )
        w_tx = F.greatest(F.call_function("LN", n_tx + F.lit(1.0)) / F.lit(3.0), F.lit(0.2))
        base = base.with_column("W_H1_COMBINED", F.least(w_hist * w_tx, F.lit(1.0)))

        county_base = (
            base.group_by("COUNTY_MODE","YM")
                .agg(F.median(F.col("IDX_RAW")).alias("IDX_COUNTY_MED"))
        )
        state_base = (
            base.group_by("STATE_MODE","YM")
                .agg(F.median(F.col("IDX_RAW")).alias("IDX_STATE_MED"))
        )

        agg = (
            base.join(county_base, on=["COUNTY_MODE","YM"], how="left")
                .join(state_base,  on=["STATE_MODE","YM"],  how="left")
        )

        agg = agg.with_column(
            "IDX_EFF",
            F.coalesce(F.col("IDX_RAW"), F.col("IDX_COUNTY_MED"), F.col("IDX_STATE_MED"))
        )

        agg = (
            agg.with_column("IDX_REL_COUNTY",
                F.iff(F.col("IDX_COUNTY_MED").is_not_null(), F.col("IDX_EFF")/F.col("IDX_COUNTY_MED"), F.lit(None)))
               .with_column("IDX_REL_STATE",
                F.iff(F.col("IDX_STATE_MED").is_not_null(),  F.col("IDX_EFF")/F.col("IDX_STATE_MED"),  F.lit(None)))
        )

        idx_eff_lead1 = F.lead(F.col("IDX_EFF"), 1).over(w_zip)
        idx_eff_lead2 = F.lead(F.col("IDX_EFF"), 2).over(w_zip)
        agg = (
            agg
            .with_column("IDX_FUTURE_H1", idx_eff_lead1)
            .with_column("IDX_FUTURE_H2", idx_eff_lead2)
            .with_column("Y_H1",
                F.call_function("LN", F.col("IDX_FUTURE_H1") + F.lit(1.0)) - F.call_function("LN", F.col("IDX_EFF") + F.lit(1.0)))
            .with_column("Y_H2",
                F.call_function("LN", F.col("IDX_FUTURE_H2") + F.lit(1.0)) - F.call_function("LN", F.col("IDX_EFF") + F.lit(1.0)))
        )

        return agg

# ============================================
# VARIANT PRICE FEATURES (full class; may be unused in slim path)
# ============================================
class VariantPriceFeatures:
    """
    Derives listing-history + home-fact features from CombinedEventsBuilder output.

    Returns:
      - pm_sp: one row per (ZPID, YM)
      - zm_sp: one row per (ZIPCODE, YM)
    """
    def __init__(self, combined_events: SnowparkDF, min_start_date: str = "2022-01-01"):
        assert isinstance(combined_events, SnowparkDF)
        self.sess = combined_events.session
        self.events = combined_events
        self.min_start_date = min_start_date

        self.c_zpid_key  = "ZPID_KEY"
        self.c_evt_date  = "EVT_DATE"
        self.c_evt_ts    = "EVT_TS"
        self.c_evt_type  = "EVT_TYPE"
        self.c_evt_price = "EVT_PRICE"
        self.c_is_rent   = "EVT_IS_RENTAL"

        self.NUM_COLS_NUMERIC        = set(globals().get("NUM_COLS_NUMERIC", []))
        self.NUM_COLS_TEXT_TO_NUM    = set(globals().get("NUM_COLS_TEXT_TO_NUMERIC", []))
        self.BIN_COLS_NUMERIC_01     = set(globals().get("BIN_COLS_NUMERIC_01", []))
        self.BIN_COLS_TEXT_YN        = set(globals().get("BIN_COLS_TEXT_YN", []))
        self.CATEGORICAL_TEXT        = set(globals().get("CATEGORICAL_TEXT", []))

        self.DROP_IN_VPF = (
            set(globals().get("FREE_TEXT_EXCLUDE", []))
            | set(globals().get("VARIANT_COLS", []))
        )

        self.KEEPERS = {
            "ZPID","ZIPCODE","YM","EVT_DATE","EVT_TS","EVT_TYPE","EVT_PRICE",
            "PREV_PRICE","PRICE_DOWN_FLG","PRICE_UP_FLG","CUT_AMT","RAISE_AMT",
            "RN_IN_MONTH_ASC","RN_IN_MONTH_DESC"
        }

    @staticmethod
    def _month_col(dt_col: F.Column) -> F.Column:
        return F.to_date(F.date_trunc("month", dt_col))

    def _events_clean(self) -> SnowparkDF:
        base = self.events.filter(
            F.coalesce(F.col(self.c_is_rent).cast(T.BooleanType()), F.lit(False)) == F.lit(False)
        )

        base_names = {"ZPID","ZIPCODE","EVT_DATE","EVT_TS","YM","EVT_TYPE","EVT_PRICE"}
        present_cols = set(base.columns)
        cand_all = (
            self.NUM_COLS_NUMERIC
            | self.NUM_COLS_TEXT_TO_NUM
            | self.BIN_COLS_NUMERIC_01
            | self.BIN_COLS_TEXT_YN
            | self.CATEGORICAL_TEXT
        )
        skip = base_names | {"PRICEHISTORY","URL","STREETADDRESS","DESCRIPTION", self.c_is_rent}
        pass_through = [c for c in cand_all if (c in present_cols and c not in skip)]

        ev = (
            base
            .select(
                F.col(self.c_zpid_key).cast(T.StringType()).alias("ZPID"),
                F.col("ZIPCODE").cast(T.StringType()).alias("ZIPCODE"),
                F.to_date(F.col(self.c_evt_date)).alias("EVT_DATE"),
                F.coalesce(F.col(self.c_evt_ts), F.to_timestamp_ntz(F.col(self.c_evt_date))).alias("EVT_TS"),
                self._month_col(F.to_date(F.col(self.c_evt_date))).alias("YM"),
                F.col(self.c_evt_type).cast(T.StringType()).alias("EVT_TYPE"),
                F.col(self.c_evt_price).cast(T.DoubleType()).alias("EVT_PRICE"),
                *[F.col(c) for c in pass_through]
            )
            .filter(F.col("EVT_DATE").is_not_null())
            .filter(F.col("EVT_DATE") >= F.to_date(F.lit(self.min_start_date)))
        )

        w_evt = Window.partition_by("ZPID").order_by(
            F.col("EVT_DATE").asc_nulls_first(), F.col("EVT_TS").asc_nulls_first()
        )
        ev = (
            ev
            .with_column("PREV_PRICE", F.lag(F.col("EVT_PRICE")).over(w_evt))
            .with_column(
                "PRICE_DOWN_FLG",
                F.iff((F.col("EVT_PRICE") < F.col("PREV_PRICE")) & F.col("PREV_PRICE").is_not_null(), F.lit(1), F.lit(0))
            )
            .with_column(
                "PRICE_UP_FLG",
                F.iff((F.col("EVT_PRICE") > F.col("PREV_PRICE")) & F.col("PREV_PRICE").is_not_null(), F.lit(1), F.lit(0))
            )
            .with_column("CUT_AMT",   F.iff(F.col("PRICE_DOWN_FLG")==1, F.col("PREV_PRICE") - F.col("EVT_PRICE"), F.lit(0.0)))
            .with_column("RAISE_AMT", F.iff(F.col("PRICE_UP_FLG")==1,   F.col("EVT_PRICE") - F.col("PREV_PRICE"), F.lit(0.0)))
        )

        w_m = Window.partition_by("ZPID", "YM").order_by(F.col("EVT_DATE").asc_nulls_first(), F.col("EVT_TS").asc_nulls_first())
        w_m_desc = Window.partition_by("ZPID", "YM").order_by(F.col("EVT_DATE").desc_nulls_last(), F.col("EVT_TS").desc_nulls_last())
        ev = (
            ev
            .with_column("RN_IN_MONTH_ASC",  F.row_number().over(w_m))
            .with_column("RN_IN_MONTH_DESC", F.row_number().over(w_m_desc))
        )

        for c in [c for c in self.NUM_COLS_NUMERIC if c in ev.columns]:
            ev = ev.with_column(c, F.col(c).cast(T.DoubleType()))
        if "safe_to_double" in globals():
            for c in [c for c in self.NUM_COLS_TEXT_TO_NUM if c in ev.columns]:
                ev = ev.with_column(c, globals()["safe_to_double"](F.col(c)))
        if "safe_to_binary_from_number" in globals():
            for c in [c for c in self.BIN_COLS_NUMERIC_01 if c in ev.columns]:
                ev = ev.with_column(c, globals()["safe_to_binary_from_number"](F.col(c)))
        if "safe_to_binary_from_text" in globals():
            for c in [c for c in self.BIN_COLS_TEXT_YN if c in ev.columns]:
                ev = ev.with_column(c, globals()["safe_to_binary_from_text"](F.col(c)))

        drop_cols = [c for c in self.DROP_IN_VPF if c in ev.columns and c not in self.KEEPERS]
        if drop_cols:
            ev = ev.drop(*drop_cols)

        ev = ev.select(
            "ZPID","ZIPCODE","YM","EVT_DATE","EVT_TS","EVT_TYPE","EVT_PRICE",
            "PREV_PRICE","PRICE_DOWN_FLG","PRICE_UP_FLG","CUT_AMT","RAISE_AMT",
            "RN_IN_MONTH_ASC","RN_IN_MONTH_DESC",
            *[c for c in ev.columns if c not in {
                "ZPID","ZIPCODE","YM","EVT_DATE","EVT_TS","EVT_TYPE","EVT_PRICE",
                "PREV_PRICE","PRICE_DOWN_FLG","PRICE_UP_FLG","CUT_AMT","RAISE_AMT",
                "RN_IN_MONTH_ASC","RN_IN_MONTH_DESC"
            }]
        )
        return ev

    def build_property_month(self) -> SnowparkDF:
        ev = self._events_clean()

        first_in_m = (
            ev.filter(F.col("RN_IN_MONTH_ASC") == 1)
              .select(
                  F.col("ZPID").alias("F_ZPID"),
                  F.col("YM").alias("F_YM"),
                  F.col("ZIPCODE").alias("ZIP_FIRST_M"),
                  F.col("EVT_DATE").alias("FIRST_SEEN_DATE_M"),
                  F.col("EVT_PRICE").alias("LIST_PRICE_FIRST_M"),
              )
        )
        last_in_m = (
            ev.filter(F.col("RN_IN_MONTH_DESC") == 1)
              .select(
                  F.col("ZPID").alias("L_ZPID"),
                  F.col("YM").alias("L_YM"),
                  F.col("EVT_DATE").alias("LAST_SEEN_DATE_M"),
                  F.col("EVT_PRICE").alias("LIST_PRICE_LAST_M"),
              )
        )

        pm_agg = (
            ev.group_by("ZPID","YM")
              .agg(
                  F.count(F.lit(1)).alias("N_EVENTS_M"),
                  F.sum(F.col("PRICE_DOWN_FLG")).alias("N_PRICE_DROPS_M"),
                  F.sum(F.col("PRICE_UP_FLG")).alias("N_PRICE_RAISES_M"),
                  F.sum(F.col("CUT_AMT")).alias("PRICE_CUT_SUM_M"),
                  F.sum(F.col("RAISE_AMT")).alias("PRICE_RAISE_SUM_M"),
                  F.min(F.col("EVT_DATE")).alias("FIRST_SEEN_ANY_M"),
                  F.max(F.col("EVT_DATE")).alias("LAST_SEEN_ANY_M"),
                  F.lit(1).alias("PRESENT_IN_MONTH"),
              )
        )

        pm = (
            pm_agg
            .join(first_in_m, (F.col("ZPID")==F.col("F_ZPID")) & (F.col("YM")==F.col("F_YM")), "left")
            .join(last_in_m,  (F.col("ZPID")==F.col("L_ZPID")) & (F.col("YM")==F.col("L_YM")), "left")
            .drop("F_ZPID","F_YM","L_ZPID","L_YM")
            .with_column("ZIPCODE", F.col("ZIP_FIRST_M"))
            .with_column(
                "DAYS_SINCE_LIST_M",
                F.iff(
                    F.col("FIRST_SEEN_DATE_M").is_not_null(),
                    F.call_function("DATEDIFF", F.lit("day"), F.col("FIRST_SEEN_DATE_M"), F.col("LAST_SEEN_ANY_M")),
                    F.lit(None)
                )
            )
            .drop("ZIP_FIRST_M")
            .with_column("HAS_CUT_IN_M",   F.iff(F.col("N_PRICE_DROPS_M")  > 0, F.lit(1), F.lit(0)))
            .with_column("HAS_RAISE_IN_M", F.iff(F.col("N_PRICE_RAISES_M") > 0, F.lit(1), F.lit(0)))
        )

        key_like = {
            "ZPID","ZIPCODE","YM","EVT_DATE","EVT_TS","EVT_TYPE","EVT_PRICE",
            "PREV_PRICE","PRICE_DOWN_FLG","PRICE_UP_FLG","CUT_AMT","RAISE_AMT",
            "RN_IN_MONTH_ASC","RN_IN_MONTH_DESC",
            "FIRST_SEEN_DATE_M","LAST_SEEN_DATE_M","FIRST_SEEN_ANY_M","LAST_SEEN_ANY_M",
            "N_EVENTS_M","N_PRICE_DROPS_M","N_PRICE_RAISES_M","PRICE_CUT_SUM_M","PRICE_RAISE_SUM_M",
            "PRESENT_IN_MONTH","DAYS_SINCE_LIST_M","LIST_PRICE_FIRST_M","LIST_PRICE_LAST_M"
        }

        num_bool_cols: List[str] = []
        for f in ev.schema.fields:
            c = f.name
            if c in key_like:
                continue
            if isinstance(
                f.datatype,
                (T.IntegerType, T.LongType, T.FloatType, T.DoubleType, T.DecimalType, T.BooleanType),
            ):
                num_bool_cols.append(c)

        if num_bool_cols:
            med_aliases  = [f"{c}_MED"  for c in num_bool_cols]
            mean_aliases = [f"{c}_MEAN" for c in num_bool_cols]

        # reliability weights
        n_tx = np.maximum(
            base["N_SOLD"].fillna(0).astype(float),
            base["N_LIST"].fillna(0).astype(float)
        )
        w_hist_raw = np.minimum(base["N_MONTHS_TO_DATE"], 24).astype(float)
        w_hist = np.log1p(w_hist_raw) / np.log1p(12.0)
        w_hist = np.maximum(w_hist, 0.2)
        w_tx = np.log1p(n_tx) / 3.0
        w_tx = np.maximum(w_tx, 0.2)
        base["W_H1_COMBINED"] = np.minimum(w_hist * w_tx, 1.0)

            pm_more = (
                ev.group_by("ZPID","YM").agg(*agg_exprs)
                  .select(
                      F.col("ZPID").alias("J_ZPID"),
                      F.col("YM").alias("J_YM"),
                      *[F.col(a) for a in (med_aliases + mean_aliases)]
                  )
            )

            pm = (
                pm.join(
                    pm_more,
                    (F.col("ZPID")==F.col("J_ZPID")) & (F.col("YM")==F.col("J_YM")),
                    "left"
                )
                .drop("J_ZPID","J_YM")
            )
            base = base.merge(macro, on=keys, how="left")

        pm = pm.select(
            "ZPID","YM","ZIPCODE",
            *[c for c in pm.columns if c not in {"ZPID","YM","ZIPCODE"}]
        )

        return pm

    def build_zip_month(self, pm_sp: SnowparkDF) -> SnowparkDF:
        assert isinstance(pm_sp, SnowparkDF)
        pm_sp = pm_sp.filter(F.col("ZIPCODE").is_not_null())

        schema = pm_sp.schema
        from snowflake.snowpark.types import IntegerType, LongType, FloatType, DoubleType, DecimalType, BooleanType

        def is_num(dt): return isinstance(dt, (IntegerType, LongType, FloatType, DoubleType, DecimalType))
        def is_bool(dt): return isinstance(dt, BooleanType)

        ignore = {"ZPID","ZIPCODE","YM","FIRST_SEEN_DATE_M","LAST_SEEN_DATE_M","FIRST_SEEN_ANY_M","LAST_SEEN_ANY_M"}
        agg_exprs = []

        for f in schema.fields:
            c = f.name
            if c in ignore:
                continue
            dt = f.datatype
            if is_bool(dt):
                agg_exprs.append(F.avg(F.col(c).cast("double")).alias(f"PM_{c}_SHARE"))
            elif is_num(dt):
                agg_exprs.append(F.avg(F.col(c)).alias(f"PM_{c}_MEAN"))
                agg_exprs.append(F.median(F.col(c)).alias(f"PM_{c}_MED"))

        if "HAS_CUT_IN_M" in pm_sp.columns:
            agg_exprs.append(F.sum(F.col("HAS_CUT_IN_M")).alias("N_LIST_WITH_CUT_PM"))
        if "HAS_RAISE_IN_M" in pm_sp.columns:
            agg_exprs.append(F.sum(F.col("HAS_RAISE_IN_M")).alias("N_LIST_WITH_RAISE_PM"))

        agg_exprs.append(F.count_distinct(F.col("ZPID")).alias("N_LISTINGS_PM"))
        zm = pm_sp.group_by("ZIPCODE","YM").agg(*agg_exprs)

        if {"N_LISTINGS_PM","N_LIST_WITH_CUT_PM"}.issubset(set(zm.columns)):
            zm = zm.with_column(
                "RATIO_WITH_CUT_PM",
                F.iff(F.col("N_LISTINGS_PM") > 0, F.col("N_LIST_WITH_CUT_PM")/F.col("N_LISTINGS_PM"), F.lit(None))
            )
        if {"N_LISTINGS_PM","N_LIST_WITH_RAISE_PM"}.issubset(set(zm.columns)):
            zm = zm.with_column(
                "RATIO_WITH_RAISE_PM",
                F.iff(F.col("N_LISTINGS_PM") > 0, F.col("N_LIST_WITH_RAISE_PM")/F.col("N_LISTINGS_PM"), F.lit(None))
            )

        return zm

# ============================================
# GEO TILING FEATURES
# ============================================
class GeoTilingFeatures:
    """
    Add H3/S2 tiling columns to a pre-aggregated ZIP×YM frame.
    """

    def __init__(
        self,
        df: SnowparkDF,
        udf_name: str | None = None,
        resolutions=(6, 7, 8, 9),
        prefix: str = "H3_R",
        udf_signature: str = "latlon",
    ):
        assert isinstance(df, SnowparkDF)
        self.df = df
        self.udf_name = udf_name
        self.resolutions = list(resolutions)
        self.prefix = prefix
        self.udf_signature = udf_signature

        needed = ["ZIPCODE", "YM", "LATITUDE", "LONGITUDE", "STATE_MODE", "COUNTY_MODE"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"GeoTilingFeatures: missing required columns: {missing}")

        self.df = (
            self.df
            .with_column("ZIPCODE",     F.col("ZIPCODE").cast(T.StringType()))
            .with_column("YM",          F.to_date(F.col("YM")))
            .with_column("LATITUDE",    F.col("LATITUDE").cast(T.DoubleType()))
            .with_column("LONGITUDE",   F.col("LONGITUDE").cast(T.DoubleType()))
            .with_column("STATE_MODE",  F.col("STATE_MODE").cast(T.StringType()))
            .with_column("COUNTY_MODE", F.col("COUNTY_MODE").cast(T.StringType()))
        )

    def _tile_expr(self, r: int):
        lat = F.col("LATITUDE")
        lon = F.col("LONGITUDE")

        if self.udf_name:
            if self.udf_signature == "lonlat":
                expr = F.call_function(self.udf_name, lon, lat, F.lit(r))
            else:
                expr = F.call_function(self.udf_name, lat, lon, F.lit(r))
            return expr.cast(T.StringType())

        return F.concat(
            F.lit(f"H3R{r}"),
            F.to_varchar(F.round(lat, 3)),
            F.to_varchar(F.round(lon, 3)),
        ).cast(T.StringType())

    def build(self) -> SnowparkDF:
        df = self.df
        added_cols = []

        for r in self.resolutions:
            col_name = f"{self.prefix}{r}"
            df = df.with_column(col_name, self._tile_expr(r))
            added_cols.append(col_name)

        out_cols = ["ZIPCODE", "YM", "STATE_MODE", "COUNTY_MODE", *added_cols]
        return df.select(*[F.col(c) for c in out_cols])

# ============================================
# SLIM VARIANT FEATURES (PM / ZM) FROM combined_events
# ============================================
assert 'CombinedEventsBuilder' in globals()

builder = CombinedEventsBuilder(
    raw_table=RAW_TABLE,
    zpid_col="ZPID",
    pricehistory_col="PRICEHISTORY",
    scrape_ts_col="SCRAPEDAT"
)

combined_events = builder.build()

from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark.window import Window
from snowflake.snowpark import DataFrame as SnowparkDF

assert isinstance(combined_events, SnowparkDF)

START_DATE = "2022-01-01"
PM_TMP = "__TMP_PM_SLIM"
ZM_TMP = "__TMP_ZM_SLIM"

sess = combined_events.session

def as_double_safe(cname: str):
    return safe_to_double(F.col(cname))

KEEP_ATTRS = [
    "PRICE", "EVT_PRICE_PSF",
    "SQFT", "LOTSQFT", "LOT",
    "BEDROOMS", "BATHROOMS", "FULLBATHROOMS", "HALFBATHROOMS",
    "YEARBUILT", "PROPERTY_AGE",
    "CENTRALHEATING", "FORCEDAIRHEATING", "ELECTRICHEATING", "NATURALGASHEATING",
    "CENTRALCOOLING", "WINDOWUNITACCOOLING",
    "GARAGEPARKING", "ATTACHEDPARKING", "DETACHEDPARKING", "DRIVEWAY", "OFFSTREETPARKING",
    "INGROUNDPOOL", "INDOORPOOL", "HEATEDPOOL",
    "UNEMPLOYMENT_RATE", "MEDIAN_LISTING_PRICE", "MEDIAN_DAYS_ON_MARKET",
    "ACTIVE_LISTING_COUNT", "WEEKLY_AVERAGE_MORTGAGE_RATE", "FM_HPI",
]

present = set(combined_events.columns)

ce = combined_events.filter(
    F.coalesce(F.col("EVT_IS_RENTAL").cast(T.BooleanType()), F.lit(False)) == F.lit(False)
)

base_cols = ["ZPID_KEY","ZPID","ZIPCODE","EVT_DATE","EVT_TS","EVT_TYPE","EVT_PRICE"]
base_cols += [c for c in KEEP_ATTRS if c in present]
base_cols = list(dict.fromkeys([c for c in base_cols if c in present]))
base = ce.select(*[F.col(c) for c in base_cols])

evt_str = F.col("EVT_DATE").cast(T.StringType())
evt_num = F.call_function("TRY_TO_NUMBER", evt_str)

evt_day_from_date = F.call_function("TRY_TO_DATE", evt_str)
evt_day_from_ts   = F.call_function("TO_DATE", F.call_function("TRY_TO_TIMESTAMP_NTZ", evt_str))
evt_day_from_ms   = F.call_function(
    "TO_DATE",
    F.call_function(
        "TO_TIMESTAMP_NTZ",
        F.when(evt_num.is_not_null() & (evt_num > F.lit(1_000_000_000_000)),
               evt_num / F.lit(1000)).otherwise(F.lit(None))
    )
)
evt_day_from_sec  = F.call_function("TO_DATE", F.call_function("TO_TIMESTAMP_NTZ", evt_num))

EVT_DAY_CANON = (
    F.when(evt_day_from_date.is_not_null(), evt_day_from_date)
     .when(evt_day_from_ts.is_not_null(),   evt_day_from_ts)
     .when(evt_num.is_not_null() & (evt_num > F.lit(1_000_000_000_000)), evt_day_from_ms)
     .when(evt_num.is_not_null() & (evt_num >= F.lit(1_000_000_000)) & (evt_num <= F.lit(1_000_000_000_000)), evt_day_from_sec)
     .otherwise(F.lit(None))
)

ev = (
    base
    .with_column("EVT_DAY_CANON", EVT_DAY_CANON)
    .with_column("YM", F.to_date(F.date_trunc("month", F.col("EVT_DAY_CANON"))))
    .with_column("EVT_PRICE_NUM", safe_to_double(F.col("EVT_PRICE")))
    .drop("EVT_PRICE")
)

ev = ev.filter(
    F.col("ZIPCODE").is_not_null()
    & F.col("EVT_DAY_CANON").is_not_null()
    & (F.col("EVT_DAY_CANON") >= F.to_date(F.lit(START_DATE)))
)

print("Slim VARIANT base (events) — filtering & type cleanup…")
try:
    ev.limit(3).show()
except Exception:
    pass

w_evt = Window.partition_by("ZPID").order_by(F.col("EVT_DAY_CANON").asc_nulls_first(), F.col("EVT_TS").asc_nulls_first())
ev = (
    ev
    .with_column("PREV_PRICE", F.lag(F.col("EVT_PRICE_NUM")).over(w_evt))
    .with_column("PRICE_DOWN_FLG", F.iff((F.col("EVT_PRICE_NUM") < F.col("PREV_PRICE")) & F.col("PREV_PRICE").is_not_null(), F.lit(1), F.lit(0)))
    .with_column("PRICE_UP_FLG",   F.iff((F.col("EVT_PRICE_NUM") > F.col("PREV_PRICE")) & F.col("PREV_PRICE").is_not_null(), F.lit(1), F.lit(0)))
    .with_column("CUT_AMT",        F.iff(F.col("PRICE_DOWN_FLG")==1, F.col("PREV_PRICE") - F.col("EVT_PRICE_NUM"), F.lit(0.0)))
    .with_column("RAISE_AMT",      F.iff(F.col("PRICE_UP_FLG")==1,   F.col("EVT_PRICE_NUM") - F.col("PREV_PRICE"), F.lit(0.0)))
)

w_m_asc  = Window.partition_by("ZPID","YM").order_by(F.col("EVT_DAY_CANON").asc_nulls_first(), F.col("EVT_TS").asc_nulls_first())
w_m_desc = Window.partition_by("ZPID","YM").order_by(F.col("EVT_DAY_CANON").desc_nulls_last(), F.col("EVT_TS").desc_nulls_last())
ev = (
    ev
    .with_column("RN_IN_MONTH_ASC",  F.row_number().over(w_m_asc))
    .with_column("RN_IN_MONTH_DESC", F.row_number().over(w_m_desc))
)

first_in_m = (
    ev.filter(F.col("RN_IN_MONTH_ASC")==1)
      .select(
          F.col("ZPID").alias("F_ZPID"),
          F.col("YM").alias("F_YM"),
          F.col("ZIPCODE").alias("ZIP_FIRST_M"),
          F.col("EVT_DAY_CANON").alias("FIRST_SEEN_DATE_M"),
          F.col("EVT_PRICE_NUM").alias("LIST_PRICE_FIRST_M"),
      )
)

last_in_m = (
    ev.filter(F.col("RN_IN_MONTH_DESC")==1)
      .select(
          F.col("ZPID").alias("L_ZPID"),
          F.col("YM").alias("L_YM"),
          F.col("EVT_DAY_CANON").alias("LAST_SEEN_DATE_M"),
          F.col("EVT_PRICE_NUM").alias("LIST_PRICE_LAST_M"),
      )
)

pm_agg = (
    ev.group_by("ZPID","YM")
      .agg(
          F.count(F.lit(1)).alias("N_EVENTS_M"),
          F.sum(F.col("PRICE_DOWN_FLG")).alias("N_PRICE_DROPS_M"),
          F.sum(F.col("PRICE_UP_FLG")).alias("N_PRICE_RAISES_M"),
          F.sum(F.col("CUT_AMT")).alias("PRICE_CUT_SUM_M"),
          F.sum(F.col("RAISE_AMT")).alias("PRICE_RAISE_SUM_M"),
          F.min(F.col("EVT_DAY_CANON")).alias("FIRST_SEEN_ANY_M"),
          F.max(F.col("EVT_DAY_CANON")).alias("LAST_SEEN_ANY_M"),
          F.median(F.col("EVT_PRICE_NUM")).alias("MED_LIST_PRICE_M"),
      )
)

pm = (
    pm_agg
    .join(first_in_m, (F.col("ZPID")==F.col("F_ZPID")) & (F.col("YM")==F.col("F_YM")), "left")
    .join(last_in_m,  (F.col("ZPID")==F.col("L_ZPID")) & (F.col("YM")==F.col("L_YM")), "left")
    .drop("F_ZPID","F_YM","L_ZPID","L_YM")
    .with_column("ZIPCODE", F.col("ZIP_FIRST_M"))
    .with_column(
        "DAYS_SINCE_LIST_M",
        F.iff(
            F.col("FIRST_SEEN_DATE_M").is_not_null(),
            F.call_function("DATEDIFF", F.lit("day"), F.col("FIRST_SEEN_DATE_M"), F.col("LAST_SEEN_ANY_M")),
            F.lit(None)
        )
    )
    .drop("ZIP_FIRST_M")
    .with_column("HAS_CUT_IN_M",   F.iff(F.col("N_PRICE_DROPS_M")  > 0, F.lit(1), F.lit(0)))
    .with_column("HAS_RAISE_IN_M", F.iff(F.col("N_PRICE_RAISES_M") > 0, F.lit(1), F.lit(0)))
)

num_like, bin_like = [], []
ev_schema = {f.name: f.datatype for f in ev.schema.fields}
for c in KEEP_ATTRS:
    if c not in ev_schema:
        continue
    dt = ev_schema[c]
    if isinstance(dt, T.BooleanType):
        bin_like.append(c)
    elif ("POOL" in c) or ("PARKING" in c) or ("HEATING" in c) or ("COOLING" in c):
        bin_like.append(c)
    else:
        num_like.append(c)

agg_exprs = []
for c in num_like:
    if c in ev.columns:
        agg_exprs += [
            F.median(as_double_safe(c)).alias(f"{c}_MED_PM"),
            F.avg(as_double_safe(c)).alias(f"{c}_MEAN_PM"),
        ]
for c in bin_like:
    if c in ev.columns:
        agg_exprs += [
            F.avg(as_double_safe(c)).alias(f"{c}_SHARE_PM"),
        ]

if agg_exprs:
    pm_more = ev.group_by("ZPID","YM").agg(*agg_exprs)
    pm = pm.join(pm_more, ["ZPID","YM"], "left")

pm.create_or_replace_temp_view(PM_TMP)

print(f"[pm_slim] TEMP view {PM_TMP} created.")
print("Rows:")
sess.table(PM_TMP).select(F.count(F.lit(1)).alias("N")).show()
print("Columns:", len(sess.table(PM_TMP).columns))

pm_tv = sess.table(PM_TMP)

agg_zm = [
    F.count_distinct(F.col("ZPID")).alias("N_PROP_PM"),
    F.sum(F.col("HAS_CUT_IN_M")).alias("N_PROP_WITH_CUT_PM"),
    F.sum(F.col("HAS_RAISE_IN_M")).alias("N_PROP_WITH_RAISE_PM"),
]

for c in ["N_EVENTS_M","N_PRICE_DROPS_M","N_PRICE_RAISES_M","PRICE_CUT_SUM_M","PRICE_RAISE_SUM_M",
          "DAYS_SINCE_LIST_M","MED_LIST_PRICE_M","LIST_PRICE_FIRST_M","LIST_PRICE_LAST_M"]:
    if c in pm_tv.columns:
        agg_zm += [
            F.avg(F.col(c)).alias(f"PM_{c}_MEAN"),
            F.median(F.col(c)).alias(f"PM_{c}_MED"),
        ]

for c in pm_tv.columns:
    if c.endswith("_MEAN_PM") or c.endswith("_MED_PM") or c.endswith("_SHARE_PM"):
        agg_zm.append(F.avg(F.col(c)).alias(f"PM_{c.replace('_PM','')}_ZIPMEAN"))

zm = (
    pm_tv.group_by("ZIPCODE","YM")
         .agg(*agg_zm)
         .with_column("SHARE_PROP_WITH_CUT_PM",
                      F.iff(F.col("N_PROP_PM")>0, F.col("N_PROP_WITH_CUT_PM")/F.col("N_PROP_PM"), F.lit(None)))
         .with_column("SHARE_PROP_WITH_RAISE_PM",
                      F.iff(F.col("N_PROP_PM")>0, F.col("N_PROP_WITH_RAISE_PM")/F.col("N_PROP_PM"), F.lit(None)))
)

zm.create_or_replace_temp_view(ZM_TMP)

print(f"[zm_slim] TEMP view {ZM_TMP} created.")
print("Rows:")
sess.table(ZM_TMP).select(F.count(F.lit(1)).alias("N")).show()
print("Columns:", len(sess.table(ZM_TMP).columns))

try:
    sess.table(ZM_TMP).sort(F.col("YM").asc()).limit(5).show()
except Exception:
    pass

# ============================================
# ZIP INDEX FEATUREIZER + BUILD_ALL_FEATURES
# ============================================
class ZipIndexFeatureizer:
    def __init__(self, idx_df: SnowparkDF, variant_pm: SnowparkDF | None,
                 variant_zm: SnowparkDF | None, geo_df: SnowparkDF | None,
                 add_macro: bool = True, monthly_periods: tuple = (12, 6),
                 start_origin: str = "2015-01-01"):
        self.idx_df = idx_df; self.pm = variant_pm; self.zm = variant_zm; self.geo = geo_df
        self.add_macro = add_macro; self.periods = monthly_periods; self.origin = start_origin

    def _add_temporal(self, df):
        origin = F.to_date(F.lit(self.origin))
        months_since = F.call_function("DATEDIFF","month", origin, F.col("YM"))
        df = df.with_column("GLOBAL_MONTH_INDEX", months_since.cast(T.IntegerType()))
        for p in self.periods:
            rad = F.lit(2.0 * 3.141592653589793) * (months_since / F.lit(float(p)))
            df = df.with_column(f"SEAS_SIN_{p}", F.call_function("SIN", rad))
            df = df.with_column(f"SEAS_COS_{p}", F.call_function("COS", rad))
        return df

    def _add_idx_lags_mom(self, df):
        w_zip = Window.partition_by("ZIPCODE").order_by(F.col("YM"))
        s = F.col("IDX").cast(T.DoubleType())
        lag1 = F.lag(s,1).over(w_zip); lag3 = F.lag(s,3).over(w_zip); lag6 = F.lag(s,6).over(w_zip)
        nz1 = F.call_function("NULLIF", lag1, F.lit(0.0))
        nz3 = F.call_function("NULLIF", lag3, F.lit(0.0))
        nz6 = F.call_function("NULLIF", lag6, F.lit(0.0))
        df = (df
              .with_column("IDX_LAG_1", lag1)
              .with_column("IDX_LAG_3", lag3)
              .with_column("IDX_LAG_6", lag6)
              .with_column("IDX_PCT_D1", s / nz1 - F.lit(1.0))
              .with_column("IDX_MOM_3", s / nz3 - F.lit(1.0))
              .with_column("IDX_MOM_6", s / nz6 - F.lit(1.0))
              .with_column("IDX_ROLL_STD_3", F.stddev_samp(s).over(w_zip.rows_between(-2, 0)))
              .with_column("IDX_ROLL_STD_6", F.stddev_samp(s).over(w_zip.rows_between(-5, 0)))
        )
        return df

    def _add_macro_lags(self, df):
        if not self.add_macro: return df
        w_zip = Window.partition_by("ZIPCODE").order_by(F.col("YM"))
        for base, alias in [("WEEKLY_AVERAGE_MORTGAGE_RATE","MORTGAGE_RATE_M"),
                            ("UNEMPLOYMENT_RATE","UNEMPLOYMENT_RATE_M")]:
            if base not in df.columns: continue
            s = F.col(base).cast(T.DoubleType())
            df = (df
                  .with_column(alias, s)
                  .with_column(f"{alias}_L1",  F.lag(s, 1).over(w_zip))
                  .with_column(f"{alias}_L3",  F.lag(s, 3).over(w_zip))
                  .with_column(f"{alias}_L12", F.lag(s, 12).over(w_zip))
                  .with_column(f"{alias}_D1",  s - F.lag(s,1).over(w_zip))
                  .with_column(f"{alias}_D3",  s - F.lag(s,3).over(w_zip))
                  .with_column(f"{alias}_D12", s - F.lag(s,12).over(w_zip))
                 )
        return df

    def build(self) -> SnowparkDF:
        feat = self.idx_df
        if isinstance(self.zm, SnowparkDF):
            feat = feat.join(self.zm, on=["ZIPCODE","YM"], how="left")
        if isinstance(self.geo, SnowparkDF):
            geo_cols = [c for c in self.geo.columns if c.upper().startswith("H3_R")]
            if geo_cols:
                feat = feat.join(self.geo.select("ZIPCODE","YM", *geo_cols), on=["ZIPCODE","YM"], how="left")
        feat = self._add_temporal(feat)
        feat = self._add_idx_lags_mom(feat)
        feat = self._add_macro_lags(feat)
        return feat

def _is_df(x):
    try:
        return isinstance(x, SnowparkDF)
    except Exception:
        return False

def _unpack_idx_builder(out) -> Tuple[SnowparkDF, Optional[SnowparkDF], Optional[SnowparkDF], Optional[SnowparkDF]]:
    if _is_df(out):
        return out, None, None, None
    if isinstance(out, (list, tuple)):
        dfs = [x for x in out if _is_df(x)]
        if not dfs:
            raise ValueError("ZipMonthIndexBuilder.build() returned no DataFrames.")
        idx_sp = dfs[0]
        ht_share_sp = next((d for d in dfs[1:] if any("HT_SHARE" in c for c in d.columns)), None)
        num_agg_sp  = dfs[2] if len(dfs) >= 4 else None
        bin_agg_sp  = dfs[3] if len(dfs) >= 4 else None
        return idx_sp, ht_share_sp, num_agg_sp, bin_agg_sp
    raise ValueError(f"Unsupported ZipMonthIndexBuilder.build() output type: {type(out)}")

def build_all_features(
    combined_events: SnowparkDF,
    h3_udf: str | None = None,
    h3_resolutions=(6,7,8,9),
    min_start_date="2022-01-01",
    min_sold_per_zip_m=10,
    min_list_per_zip_m=20,
) -> SnowparkDF:
    assert isinstance(combined_events, SnowparkDF)
    sess = combined_events.session

    print("Building ZIP×month index & aggregates (robust)…")
    idx_builder = ZipMonthIndexBuilder(
        combined_events=combined_events,
        min_start_date=min_start_date,
        min_sold_per_zip_m=min_sold_per_zip_m,
        min_list_per_zip_m=min_list_per_zip_m,
    )
    idx_out = idx_builder.build()
    idx_sp, ht_share_sp, num_agg_sp, bin_agg_sp = _unpack_idx_builder(idx_out)

    cols = set(idx_sp.columns)
    have_idx      = "IDX"      in cols
    have_idx_eff  = "IDX_EFF"  in cols
    have_idx_raw  = "IDX_RAW"  in cols

    if not have_idx:
        co = F.coalesce(
            F.col("IDX_EFF") if have_idx_eff else F.lit(None),
            F.col("IDX_RAW") if have_idx_raw else F.lit(None),
        )
        idx_sp = idx_sp.with_column("IDX", co)
        print("[normalize] Added IDX from coalesce(IDX_EFF, IDX_RAW).")

    pm_sp = None
    zm_sp = None
    try:
        zm_sp = sess.table("__TMP_ZM_SLIM")
        print("Using precomputed slim VARIANT ZIP×month from __TMP_ZM_SLIM.")
    except Exception as e:
        print("Slim VARIANT view __TMP_ZM_SLIM not found — continuing without VARIANT ZM. Details:", repr(e))

    geo_sp = None
    evt_date_col = "EVT_DATE" if "EVT_DATE" in combined_events.columns else ("evt_date" if "evt_date" in combined_events.columns else None)
    latlon_ok = all(c in combined_events.columns for c in ["ZIPCODE","LATITUDE","LONGITUDE","STATE","COUNTY"]) and evt_date_col is not None

    if latlon_ok:
        print("Computing geo-tiling features (H3/S2)…")
        base_latlon = (
            combined_events
            .select(
                F.col("ZIPCODE").cast(T.StringType()).alias("ZIPCODE"),
                F.to_date(F.col(evt_date_col)).alias("EVT_DATE"),
                F.to_date(F.date_trunc("month", F.col(evt_date_col))).alias("YM"),
                F.col("LATITUDE").cast(T.DoubleType()).alias("LATITUDE"),
                F.col("LONGITUDE").cast(T.DoubleType()).alias("LONGITUDE"),
                F.col("STATE").cast(T.StringType()).alias("STATE"),
                F.col("COUNTY").cast(T.StringType()).alias("COUNTY"),
            )
            .filter(F.col("ZIPCODE").is_not_null() & F.col("YM").is_not_null())
        )

        latlon_agg = (
            base_latlon
            .group_by("ZIPCODE","YM")
            .agg(
                F.median(F.col("LATITUDE")).alias("LATITUDE"),
                F.median(F.col("LONGITUDE")).alias("LONGITUDE"),
                F.max(F.col("STATE")).alias("STATE_MODE"),
                F.max(F.col("COUNTY")).alias("COUNTY_MODE"),
            )
        )

        try:
            latlon = latlon_agg.select(
                F.col("ZIPCODE"), F.col("YM"),
                F.col("LATITUDE"), F.col("LONGITUDE"),
                F.col("STATE_MODE"), F.col("COUNTY_MODE")
            )
        except Exception as e:
            print(f"Planner balked on pure SELECT lat/lon aggregation; using TEMP VIEW fallback. Details: {e}")
            latlon_view = "__TMP_LATLON_ZIPYM"
            latlon_agg.create_or_replace_temp_view(latlon_view)
            latlon = sess.table(latlon_view).select(
                F.col("ZIPCODE"), F.col("YM"),
                F.col("LATITUDE"), F.col("LONGITUDE"),
                F.col("STATE_MODE"), F.col("COUNTY_MODE")
            )

        geo_sp = GeoTilingFeatures(
            latlon,
            udf_name=h3_udf,
            resolutions=h3_resolutions,
            prefix="H3_R",
            udf_signature="latlon",
        ).build()
    else:
        print("LAT/LON/EVT_DATE not present — skipping geo-tiling.")

    print("Assembling feature matrix (ZipIndexFeatureizer)…")
    try:
        feat_sp = ZipIndexFeatureizer(idx_sp, pm_sp, zm_sp, geo_sp).build()
    except Exception as e1:
        try:
            feat_sp = ZipIndexFeatureizer(idx_sp, ht_share_sp, num_agg_sp, bin_agg_sp).build()
        except Exception as e2:
            raise RuntimeError(
                "Failed to build features with both ZipIndexFeatureizer signatures.\n"
                f"New-signature error: {repr(e1)}\n"
                f"Old-signature error: {repr(e2)}"
            )

    must_haves = ["YM", "IDX", "Y_H1", "Y_H2"]
    missing = [c for c in must_haves if c not in feat_sp.columns]
    if missing:
        print("WARNING — Expected columns missing on feat_sp:", missing)

    return feat_sp

print("Orchestrating full feature build…")
feat_sp = build_all_features(
    combined_events=combined_events,
    h3_udf=globals().get("H3_UDF_NAME", None),
    h3_resolutions=globals().get("H3_RESOLUTIONS", (6,7,8,9)),
    min_start_date=globals().get("MIN_START_DATE", "2022-01-01"),
    min_sold_per_zip_m=globals().get("MIN_SOLD_PER_ZIP_M", 3),
    min_list_per_zip_m=globals().get("MIN_LIST_PER_ZIP_M", 5),
)
print("feat_sp ready. Columns:", len(feat_sp.columns))

# ============================================
# TRAINING CONFIG & MODEL DEFINITIONS
# ============================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta

torch.set_num_threads(max(1, os.cpu_count() // 2))

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RUN_DIR = Path("runlog"); RUN_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

TARGETS = ["Y_H1", "Y_H2"]
QUANTILES = [0.1, 0.5, 0.9]
PINBALL_WEIGHT = 1.0
L1_MEDIAN_WEIGHT = 0.5

RELIABILITY_C = 2.5
VAL_FRACTION_OF_TRAIN_TIME = 1/6

EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 2048

HIDDEN = 384
LAYERS = 3
DROPOUT = 0.15
EMB_DIM_CAP = 64

SUPPRESS_WIDTH_PCT = 0.15
MIN_LEVEL = 1e0

CATEGORICAL_COLS = [c for c in [
    "H3_R6","H3_R7","H3_R8","H3_R9",
    "STATE_MODE","COUNTY_MODE","ZIPCODE"
] if c]

NON_FEATURE_KEYS = {"ZIPCODE","YM","STATE_MODE","COUNTY_MODE","DAY_FOR_SPLIT"}
LABEL_COLS = {"Y_H1","Y_H2","IDX_FUTURE_H1","IDX_FUTURE_H2"}
FUTUREISH_COLS = {"BASE_DLOG_H1","BASE_DLOG_H2","_BASE_FWD1","_BASE_FWD2","_BASE_NOW"}

class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, rows_mask: pd.Series,
                 num_cols: List[str], cat_cols: List[str], cat_maps: Dict[str,Dict[Any,int]],
                 targets: List[str], weights_col: str | None = None):
        ix = np.where(rows_mask.values)[0]
        self.df = df.iloc[ix].copy().reset_index(drop=True)

        ycols = [t + ("_WZ" if (t + "_WZ") in self.df.columns else "") for t in targets]

        idx_col = "IDX_EFF" if "IDX_EFF" in self.df.columns else "IDX"

        good = pd.to_numeric(self.df[idx_col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).notna().values
        for yc in ycols:
            yv = pd.to_numeric(self.df[yc], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).notna().values
            good &= yv
        self.df = self.df.loc[good].reset_index(drop=True)

        self.idx_col = idx_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.targets  = targets
        self.weights_col = weights_col

        if len(num_cols) > 0:
            self.df.loc[:, num_cols] = self.df[num_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.X_num = torch.tensor(self.df[num_cols].astype(np.float32).values, dtype=torch.float32) if num_cols else None

        self.X_cat = []
        for c in cat_cols:
            mp = cat_maps[c]
            idx = self.df[c].astype("string").map(mp).fillna(0).astype(np.int64).values
            self.X_cat.append(torch.tensor(idx, dtype=torch.long))
        self.X_cat = torch.stack(self.X_cat, dim=1) if len(self.X_cat)>0 else None

        ys = []
        for yc in ycols:
            yv = pd.to_numeric(self.df[yc], errors="coerce").astype(np.float32).values
            ys.append(torch.tensor(yv, dtype=torch.float32))
        self.y = torch.stack(ys, dim=1)

        self.idx_now = torch.tensor(pd.to_numeric(self.df[self.idx_col], errors="coerce").astype(np.float32).values, dtype=torch.float32)

        if self.weights_col and self.weights_col in self.df.columns:
            wv = pd.to_numeric(self.df[self.weights_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(np.float32).values
            self.w = torch.tensor(wv, dtype=torch.float32)
        else:
            self.w = torch.ones(len(self.df), dtype=torch.float32)

    def __len__(self): return len(self.idx_now)
    def __getitem__(self, i):
        return (
            (self.X_num[i] if self.X_num is not None else torch.empty(0)),
            (self.X_cat[i] if self.X_cat is not None else torch.empty(0, dtype=torch.long)),
            self.y[i], self.idx_now[i], self.w[i]
        )

class MultiTaskQuantileNet(nn.Module):
    def __init__(self, num_dim: int, cat_cardinals: List[int], emb_cap: int,
                 hidden: int, layers: int, dropout: float, n_targets: int, n_quants: int):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_dims = []
        for card in cat_cardinals:
            dim = min(emb_cap, max(4, int(round(card**0.25)*4)))
            self.embs.append(nn.Embedding(card+1, dim, padding_idx=0))
            emb_dims.append(dim)
        in_dim = num_dim + sum(emb_dims)

        mlp = []
        d = in_dim
        for _ in range(layers):
            mlp += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(d, n_quants) for _ in range(n_targets)])

    def forward(self, x_num, x_cat):
        h = x_num if x_num.numel() != 0 else None
        if x_cat.numel() != 0:
            emb = [emb_layer(x_cat[:, i]) for i,emb_layer in enumerate(self.embs)]
            emb = torch.cat(emb, dim=1) if emb else None
            h = emb if h is None else torch.cat([h, emb], dim=1)
        z = self.trunk(h) if isinstance(self.trunk, nn.Sequential) else h
        outs = [head(z) for head in self.heads]
        return outs

def pinball_loss(pred, target, taus):
    diff = target.unsqueeze(1) - pred
    losses = []
    for i, q in enumerate(taus):
        e = diff[:, i]
        losses.append(torch.maximum(q*e, (q-1)*e))
    return torch.mean(torch.stack(losses, dim=1))

# ============================================
# DLOG → LEVEL & EVAL / SUPPRESSION
# ============================================
def dlog_to_level(idx_now: torch.Tensor, dlog: torch.Tensor) -> torch.Tensor:
    return torch.expm1(torch.log1p(idx_now) + dlog)

def _infer_price_scale(idx_vals: np.ndarray) -> float:
    med = float(np.nanmedian(idx_vals)) if idx_vals.size else np.nan
    if np.isfinite(med) and med < 10_000.0:
        return 1_000.0
    return 1.0

@torch.no_grad()
def eval_split(model, loader, taus, head_ix: int):
    model.eval()
    try:
        i10 = taus.index(0.1); i50 = taus.index(0.5); i90 = taus.index(0.9)
    except ValueError:
        i10, i50, i90 = 0, len(taus)//2, -1

    y_list, p10_list, p50_list, p90_list, idx_list = [], [], [], [], []

    for xnum, xcat, y, idx_now, w in loader:
        if xnum.numel(): xnum = xnum.to(DEVICE)
        if xcat.numel(): xcat = xcat.to(DEVICE)
        y       = y.to(DEVICE)
        idx_now = idx_now.to(DEVICE)

        outs = model(xnum, xcat)[head_ix]
        y_true = y[:, head_ix]
        p10 = outs[:, i10]; p50 = outs[:, i50]; p90 = outs[:, i90]

        mask = torch.isfinite(y_true) & torch.isfinite(p50) & torch.isfinite(idx_now) & (idx_now > -1.0)
        if not mask.any():
            continue

        y_list.append(y_true[mask].detach().cpu())
        p10_list.append(p10[mask].detach().cpu())
        p50_list.append(p50[mask].detach().cpu())
        p90_list.append(p90[mask].detach().cpu())
        idx_list.append(idx_now[mask].detach().cpu())

    if not y_list:
        return dict(mae=np.nan, r2=np.nan, wape=np.nan, mdape=np.nan,
                    pct10=np.nan, p90_p10_cover=np.nan, rel_width=np.nan)

    y_true = torch.cat(y_list)
    p10    = torch.cat(p10_list)
    p50    = torch.cat(p50_list)
    p90    = torch.cat(p90_list)
    idx    = torch.cat(idx_list)

    true_lvl = dlog_to_level(idx, y_true).cpu().numpy()
    pred_lvl = dlog_to_level(idx, p50).cpu().numpy()
    p10_lvl  = dlog_to_level(idx, p10).cpu().numpy()
    p90_lvl  = dlog_to_level(idx, p90).cpu().numpy()

    scale = _infer_price_scale(idx.cpu().numpy())
    true_d = true_lvl * scale
    pred_d = pred_lvl * scale
    p10_d  = p10_lvl  * scale
    p90_d  = p90_lvl  * scale

    finite = np.isfinite(true_d) & np.isfinite(pred_d)
    if finite.sum() == 0:
        return dict(mae=np.nan, r2=np.nan, wape=np.nan, mdape=np.nan,
                    pct10=np.nan, p90_p10_cover=np.nan, rel_width=np.nan)

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
    wape = float(np.nansum(np.abs(yv - pv)) / denom) if denom > 0 else np.nan
    mdape = float(np.nanmedian(np.abs((yv - pv) / np.clip(np.abs(yv), 1e-9, None))))
    pct10 = float(np.nanmean(np.abs(pv - yv) <= 0.10 * np.abs(yv)))

    finite_pi = np.isfinite(true_d) & np.isfinite(p10_d) & np.isfinite(p90_d) & np.isfinite(pred_d)
    if finite_pi.sum() == 0:
        cover = np.nan
        rel_w = np.nan
    else:
        y_pi   = true_d[finite_pi]
        p10_pi = p10_d[finite_pi]
        p90_pi = p90_d[finite_pi]
        p50_pi = pred_d[finite_pi]
        width  = np.maximum(np.abs(p90_pi - p10_pi), 1e-9)
        cover  = float(np.mean((y_pi >= p10_pi) & (y_pi <= p90_pi)))
        rel_w  = float(np.mean(width / np.clip(np.abs(p50_pi), MIN_LEVEL, None)))

    return dict(mae=mae, r2=r2, wape=wape, mdape=mdape,
                pct10=pct10, p90_p10_cover=cover, rel_width=rel_w)

@torch.no_grad()
def suppression_report(model, ds, taus, head_ix):
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    n, n_suppr = 0, 0

    try:
        i10 = taus.index(0.1); i50 = taus.index(0.5); i90 = taus.index(0.9)
    except ValueError:
        i10, i50, i90 = 0, len(taus)//2, -1

    for xnum, xcat, y, idx_now, w in dl:
        if xnum.numel(): xnum = xnum.to(DEVICE)
        if xcat.numel(): xcat = xcat.to(DEVICE)
        idx_now = idx_now.to(DEVICE)

        outs = model(xnum, xcat)[head_ix]
        p10 = outs[:, i10]; p50 = outs[:, i50]; p90 = outs[:, i90]

        mask = torch.isfinite(p10) & torch.isfinite(p50) & torch.isfinite(p90) & torch.isfinite(idx_now) & (idx_now > -1.0)
        if not mask.any():
            continue

        idx_m   = idx_now[mask]
        p10_m   = p10[mask]
        p50_m   = p50[mask]
        p90_m   = p90[mask]

        pred_lvl = dlog_to_level(idx_m, p50_m).cpu().numpy()
        p10_lvl  = dlog_to_level(idx_m, p10_m).cpu().numpy()
        p90_lvl  = dlog_to_level(idx_m, p90_m).cpu().numpy()

        scale = _infer_price_scale(idx_m.cpu().numpy())
        pv = pred_lvl * scale
        lo = p10_lvl  * scale
        hi = p90_lvl  * scale

        finite = np.isfinite(pv) & np.isfinite(lo) & np.isfinite(hi)
        if finite.sum() == 0:
            continue

        pv = pv[finite]; lo = lo[finite]; hi = hi[finite]
        width = np.abs(hi - lo)
        rel_width = width / np.clip(np.abs(pv), MIN_LEVEL, None)

        m = rel_width > SUPPRESS_WIDTH_PCT
        n += len(m)
        n_suppr += int(m.sum())

    return dict(suppressed=n_suppr, total=n, rate=(float(n_suppr)/max(n,1) if n else np.nan))

def train_one(model, ds_trn, ds_val, taus, head_weights=(1.0, 1.0)):
    dl_trn = DataLoader(ds_trn, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, bad = {"score": 1e18, "state": None, "epoch": -1}, 0

    for ep in range(1, EPOCHS+1):
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
                    pl = (pl * w.mean())
                    l1m = (l1m * w.mean())
                loss = loss + head_weights[head_ix] * (pl + l1m)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss.detach().cpu())
        v1 = eval_split(model, dl_val, QUANTILES, head_ix=0)
        v2 = eval_split(model, dl_val, QUANTILES, head_ix=1)
        score = v1["mae"] + v2["mae"]
        print(f"[ep {ep:02d}] trn_loss={total/len(dl_trn):.5f} | "
              f"val H1: MAE=${v1['mae']:.0f} R2={v1['r2']:.3f} WAPE={v1['wape']:.3f} | "
              f"H2: MAE=${v2['mae']:.0f} R2={v2['r2']:.3f} WAPE={v2['wape']:.3f}")
        if score < best["score"]:
            best = {"score": score, "state": {k:v.cpu() for k,v in model.state_dict().items()}, "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {ep} (no improv {PATIENCE}). Best epoch={best['epoch']}.")
                break
    model.load_state_dict(best["state"])
    return model

# ============================================
# PULL feat_sp TO PANDAS, WINSORIZE, BUILD FEATURES
# ============================================
from snowflake.snowpark import functions as F, types as T

START_DATE = "2022-01-01"
HOLDOUT_DAYS = int(globals().get("HOLDOUT_DAYS", 60))
VAL_FRACTION_OF_TRAIN_TIME = globals().get("VAL_FRACTION_OF_TRAIN_TIME", 1/6)
RELIABILITY_C = globals().get("RELIABILITY_C", 2.5)

if "EVT_IS_RENTAL" in feat_sp.columns:
    feat_sp = feat_sp.drop("EVT_IS_RENTAL")

date_col = "DAY_FOR_SPLIT" if "DAY_FOR_SPLIT" in feat_sp.columns else "YM"
dtype = next(f.datatype for f in feat_sp.schema.fields if f.name == date_col)

if isinstance(dtype, T.DateType):
    pred = F.col(date_col) >= F.to_date(F.lit(START_DATE))
elif isinstance(dtype, T.TimestampType):
    pred = F.col(date_col) >= F.to_timestamp_ntz(F.lit(START_DATE))
else:
    pred = F.to_date(F.col(date_col)) >= F.to_date(F.lit(START_DATE))

fat_text = [c for c in ("URL", "STREETADDRESS", "DESCRIPTION") if c in feat_sp.columns]
base = feat_sp.select(*[c for c in feat_sp.columns if c not in fat_text]).filter(pred)

diag = base.select(
    F.min(F.col(date_col)).alias("MIN_D"),
    F.max(F.col(date_col)).alias("MAX_D"),
    F.count(F.lit(1)).alias("N_ROWS")
).collect()[0]
print(f"[CELL11:DIAG] {date_col} range: {diag['MIN_D']} .. {diag['MAX_D']} | rows={diag['N_ROWS']:,}")

pdf_parts = []
for b in base.to_pandas_batches():
    pdf_parts.append(b)
pdf = pd.concat(pdf_parts, ignore_index=True) if pdf_parts else base.to_pandas()

for c in ("YM","DAY_FOR_SPLIT"):
    if c in pdf.columns:
        pdf[c] = pd.to_datetime(pdf[c], errors="coerce")

tcol = "DAY_FOR_SPLIT" if "DAY_FOR_SPLIT" in pdf.columns else "YM"
max_evt_day = pd.to_datetime(pdf[tcol]).max()
max_evt_day = pd.Timestamp(max_evt_day)
effective_train_end = max_evt_day - pd.Timedelta(days=HOLDOUT_DAYS)
holdout_start = effective_train_end + pd.Timedelta(days=1)
print(f"[CELL11:SPLIT] max_day={max_evt_day.date()} | train_end={effective_train_end.date()} "
      f"| holdout=[{holdout_start.date()} … {max_evt_day.date()}]")

if "DAY_FOR_SPLIT" in pdf.columns:
    trn_mask = pdf["DAY_FOR_SPLIT"] <= effective_train_end - pd.Timedelta(days=HOLDOUT_DAYS-1)
    hld_mask = ~trn_mask
else:
    trn_mask = pdf["YM"] <= effective_train_end
    hld_mask = pdf["YM"] > effective_train_end

df_trn = pdf.loc[trn_mask].copy()

def _winsor_train_only(df_trn: pd.DataFrame, df_full: pd.DataFrame, ycol: str, k: float) -> pd.DataFrame:
    key = ["STATE_MODE","YM"] if "STATE_MODE" in df_full.columns else ["YM"]
    fences: Dict[Any, Any] = {}
    for gk, g in df_trn.groupby(key, dropna=False):
        s = pd.to_numeric(g[ycol], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        fences[gk] = (q1 - k*iqr, q3 + k*iqr)

    def clamp(row):
        gk = tuple(row[k] if k in df_full.columns else np.nan for k in key)
        lo, hi = fences.get(gk, (-np.inf, np.inf))
        v = row[ycol]
        return np.clip(v, lo, hi) if pd.notna(v) else v

    df_full[ycol+"_WZ"] = df_full.apply(clamp, axis=1)
    return df_full

LABEL_COLS = set(globals().get("LABEL_COLS", {"Y_H1","Y_H2","IDX_FUTURE_H1","IDX_FUTURE_H2"}))
for ycol, k in [("Y_H1", 1.5), ("Y_H2", 3.0)]:
    if ycol in pdf.columns:
        pdf = _winsor_train_only(df_trn[["STATE_MODE","YM",ycol]] if {"STATE_MODE","YM"}.issubset(df_trn.columns)
                                 else df_trn[[c for c in ["YM", ycol] if c in df_trn.columns]],
                                 pdf, ycol, k)

NON_FEATURE_KEYS = set(globals().get("NON_FEATURE_KEYS", {"ZIPCODE","YM","STATE_MODE","COUNTY_MODE","DAY_FOR_SPLIT"}))
FUTUREISH_COLS   = set(globals().get("FUTUREISH_COLS", {"BASE_DLOG_H1","BASE_DLOG_H2","_BASE_FWD1","_BASE_FWD2","_BASE_NOW"}))
CATEGORICAL_COLS = list(globals().get("CATEGORICAL_COLS", [
    "H3_R6","H3_R7","H3_R8","H3_R9","STATE_MODE","COUNTY_MODE","ZIPCODE"
]))

all_cols   = set(pdf.columns)
drop_never = NON_FEATURE_KEYS | LABEL_COLS | FUTUREISH_COLS
cand_feats = sorted([c for c in all_cols if c not in drop_never])

for c in cand_feats:
    if c in CATEGORICAL_COLS:
        continue
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

pdf["W_H1"] = 1.0
if "N_SOLD" in pdf.columns:
    pdf.loc[trn_mask & pdf["N_SOLD"].notna(), "W_H1"] = 1.0

if "W_H1_COMBINED" in pdf.columns:
    pdf["W_H1_COMBINED"] = pd.to_numeric(pdf["W_H1_COMBINED"], errors="coerce").clip(0.2, 1.0).fillna(1.0)
    pdf["W_H1"] = pdf["W_H1"] * pdf["W_H1_COMBINED"]

if "DAY_FOR_SPLIT" in pdf.columns:
    trange = pdf.loc[trn_mask, "DAY_FOR_SPLIT"]
else:
    trange = pdf.loc[trn_mask, "YM"]

t0, t1 = pd.to_datetime(trange.min()), pd.to_datetime(trange.max())
cut = t0 + (t1 - t0) * (1 - VAL_FRACTION_OF_TRAIN_TIME)

if "DAY_FOR_SPLIT" in pdf.columns:
    trn_in = (pdf["DAY_FOR_SPLIT"] <= cut) & trn_mask
    val_in_tmp = (pdf["DAY_FOR_SPLIT"] >  cut) & trn_mask
else:
    trn_in = (pdf["YM"] <= cut) & trn_mask
    val_in_tmp = (pdf["YM"] >  cut) & trn_mask

cat_maps: Dict[str, Dict[Any,int]] = {}
for c in [c for c in CATEGORICAL_COLS if c in pdf.columns]:
    vals = pd.Index(pdf.loc[trn_mask, c].astype("string").fillna("<NA>").unique())
    cat_maps[c] = {v:i+1 for i,v in enumerate(vals)}
    for split_mask in [trn_in, val_in_tmp, hld_mask]:
        pdf.loc[split_mask, c] = pdf.loc[split_mask, c].astype("string").fillna("<NA>")

num_cols = [c for c in cand_feats if c not in CATEGORICAL_COLS]
scaler = StandardScaler()
if len(num_cols) > 0 and np.sum(trn_in) > 0:
    scaler.fit(pdf.loc[trn_in, num_cols])
    for split_mask in [trn_in, val_in_tmp, hld_mask]:
        pdf.loc[split_mask, num_cols] = scaler.transform(pdf.loc[split_mask, num_cols])

X_cols_num = num_cols
X_cols_cat = [c for c in CATEGORICAL_COLS if c in pdf.columns]

print(f"[CELL11:DONE] rows={len(pdf):,} | F_num={len(X_cols_num)} | F_cat={len(X_cols_cat)} | "
      f"totalF={len(X_cols_num)+len(X_cols_cat)} | dropped_text={len(fat_text)}")

# ============================================
# FIT NATIONAL MODEL, EVAL HOLDOUT, SAVE
# ============================================
pdf["YM"] = pd.to_datetime(pdf["YM"])
if "DAY_FOR_SPLIT" in pdf.columns:
    pdf["DAY_FOR_SPLIT"] = pd.to_datetime(pdf["DAY_FOR_SPLIT"])
else:
    pdf["DAY_FOR_SPLIT"] = pdf["YM"]

has_labels = pdf["Y_H1"].notna() & pdf["Y_H2"].notna() & pdf["IDX"].notna()

last_ym = pdf.loc[has_labels, "YM"].max()
m0 = last_ym
m1 = last_ym - relativedelta(months=1)
m2 = last_ym - relativedelta(months=2)
m3 = last_ym - relativedelta(months=3)

hld_mask = has_labels & (pdf["YM"].isin([m0, m1]))
val_in   = has_labels & (pdf["YM"].isin([m2, m3]))
trn_in   = has_labels & (pdf["YM"] <  m3)

if trn_in.sum() < 0.25 * has_labels.sum():
    m0 = last_ym
    m1 = last_ym - relativedelta(months=1)
    hld_mask = has_labels & (pdf["YM"] == m0)
    val_in   = has_labels & (pdf["YM"] == m1)
    trn_in   = has_labels & (pdf["YM"] <  m1)

n_trn = int(trn_in.sum()); n_val = int(val_in.sum()); n_hld = int(hld_mask.sum())
steps_trn = int(np.ceil(n_trn / BATCH_SIZE)) if n_trn else 0
steps_val = int(np.ceil(n_val / BATCH_SIZE)) if n_val else 0
print(f"[CELL11:DATA] train={n_trn:,} ({steps_trn} steps/epoch) | val={n_val:,} ({steps_val} steps) | holdout={n_hld:,}")

cat_cardinals = []
for c in X_cols_cat:
    card = len(cat_maps[c])
    cat_cardinals.append(card)

ds_trn = TabularDataset(pdf, trn_in, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col="W_H1")
ds_val = TabularDataset(pdf, val_in, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col="W_H1")
ds_hld = TabularDataset(pdf, hld_mask, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col=None)

dl_trn = DataLoader(ds_trn, batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

print(f"[CELL11:DIMS] F_num={len(X_cols_num)} | F_cat={len(X_cols_cat)} | train/val/hld={len(ds_trn)}/{len(ds_val)}/{len(ds_hld)}")

model = MultiTaskQuantileNet(
    num_dim=len(X_cols_num),
    cat_cardinals=cat_cardinals,
    emb_cap=EMB_DIM_CAP,
    hidden=HIDDEN,
    layers=LAYERS,
    dropout=DROPOUT,
    n_targets=len(TARGETS),
    n_quants=len(QUANTILES),
).to(DEVICE)

model = train_one(model, ds_trn, ds_val, QUANTILES)

dl_hld = DataLoader(ds_hld, batch_size=BATCH_SIZE, shuffle=False)
h1 = eval_split(model, dl_hld, QUANTILES, head_ix=0)
h2 = eval_split(model, dl_hld, QUANTILES, head_ix=1)

print("\n=== HOLDOUT METRICS ===")
print("H1:", h1)
print("H2:", h2)

print("\n=== SUPPRESSION (confidence gating) ===")
print("H1:", suppression_report(model, ds_hld, QUANTILES, head_ix=0))
print("H2:", suppression_report(model, ds_hld, QUANTILES, head_ix=1))

ts = time.strftime("%Y%m%d-%H%M%S")
runname = "zipmonth_tabmtl_quantile"

train_max_ym = pd.to_datetime(pdf.loc[trn_in, "YM"]).max() if n_trn else pd.NaT
val_ym_range = pd.to_datetime(pdf.loc[val_in, "YM"]).sort_values().unique() if n_val else []
hld_ym_range = pd.to_datetime(pdf.loc[hld_mask, "YM"]).sort_values().unique() if n_hld else []

split_meta = dict(
    last_ym=str(last_ym.date()) if pd.notna(last_ym) else None,
    train_max_ym=str(train_max_ym.date()) if pd.notna(train_max_ym) else None,
    val_months=[str(pd.to_datetime(x).date()) for x in val_ym_range],
    holdout_months=[str(pd.to_datetime(x).date()) for x in hld_ym_range],
    sizes=dict(train=n_trn, val=n_val, holdout=n_hld),
)

torch.save({
    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "config": dict(
        HIDDEN=HIDDEN, LAYERS=LAYERS, DROPOUT=DROPOUT,
        QUANTILES=QUANTILES, CATEGORICAL_COLS=X_cols_cat, NUMERIC_COLS=X_cols_num
    ),
    "cat_maps": cat_maps,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "split": split_meta,
    "metrics_holdout": {"H1": h1, "H2": h2},
}, RUN_DIR / f"{ts}__{runname}.pt")

with open(RUN_DIR / f"{ts}__{runname}.json", "w") as f:
    json.dump({
        "metrics_holdout": {"H1": h1, "H2": h2},
        "split": split_meta,
        "shapes": dict(train=len(ds_trn), val=len(ds_val), holdout=len(ds_hld),
                       F_num=len(X_cols_num), F_cat=len(X_cols_cat))
    }, f, indent=2)

print("Saved:", RUN_DIR / f"{ts}__{runname}.pt")

n_trn = len(ds_trn); n_val = len(ds_val); n_hld = len(ds_hld)
steps_trn = int(np.ceil(n_trn / BATCH_SIZE)) if n_trn else 0
steps_val = int(np.ceil(n_val / BATCH_SIZE)) if n_val else 0
print(f"[CELL11:DATA] train={n_trn:,} ({steps_trn} steps/epoch) | val={n_val:,} ({steps_val} steps) | holdout={n_hld:,}")

# ============================================
# ROLLING BACKTEST
# ============================================
def rolling_backtest(pdf, n_folds=3, fold_len_days=60):
    days = pd.to_datetime(pdf["DAY_FOR_SPLIT"] if "DAY_FOR_SPLIT" in pdf.columns else pdf["YM"])
    tmin, tmax = days.min(), days.max()
    folds = []
    for i in range(n_folds):
        hold_end = tmax - pd.Timedelta(days=i*fold_len_days)
        hold_start = hold_end - pd.Timedelta(days=fold_len_days-1)
        train_end = hold_start - pd.Timedelta(days=1)
        folds.append((train_end, hold_start, hold_end))
    results = []
    for k,(train_end, hold_start, hold_end) in enumerate(folds[::-1], 1):
        print(f"\n[Fold {k}] train ≤ {train_end.date()} | holdout=[{hold_start.date()} … {hold_end.date()}]")
        tcol = "DAY_FOR_SPLIT" if "DAY_FOR_SPLIT" in pdf.columns else "YM"
        trn_mask = pdf[tcol] <= train_end
        hld_mask = (pdf[tcol] >= hold_start) & (pdf[tcol] <= hold_end)
        ds_trn = TabularDataset(pdf, trn_mask, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col="W_H1")
        ds_hld = TabularDataset(pdf, hld_mask, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col=None)
        model = MultiTaskQuantileNet(len(X_cols_num), [len(cat_maps[c]) for c in X_cols_cat], EMB_DIM_CAP, HIDDEN, LAYERS, DROPOUT, len(TARGETS), len(QUANTILES))
        model = train_one(model, ds_trn, ds_trn, QUANTILES)  # quick
        dl_h = DataLoader(ds_hld, batch_size=BATCH_SIZE, shuffle=False)
        h1 = eval_split(model, dl_h, QUANTILES, 0)
        h2 = eval_split(model, dl_h, QUANTILES, 1)
        results.append(dict(fold=k, H1=h1, H2=h2))
    return results

rolling_results = rolling_backtest(pdf, n_folds=3, fold_len_days=60)
print("\nRolling backtest results:", rolling_results)
