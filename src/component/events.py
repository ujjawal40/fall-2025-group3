import json
import numpy as np
import pandas as pd


class CombinedEventsBuilder:
    """
    LOCAL version:
      - Input: pandas DataFrame with at least ZPID, PRICEHISTORY, SCRAPEDAT, etc.
      - Output: one row per PRICEHISTORY event per ZPID + merged base snapshot.
    """

    def __init__(
        self,
        base_df: pd.DataFrame,
        zpid_col: str = "ZPID",
        pricehistory_col: str = "PRICEHISTORY",
        scrape_ts_col: str = "SCRAPEDAT",
    ):
        self.base_df = base_df
        self.zpid_col = zpid_col
        self.pricehistory_col = pricehistory_col
        self.scrape_ts_col = scrape_ts_col

        self.c_zpid_key     = "ZPID_KEY"
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

    # ---------- PUBLIC ----------
    def build(self) -> pd.DataFrame:
        base = self.base_df.copy()
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
        return combined

    # ---------- INTERNAL ----------
    def _flatten_events(self, base_df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in base_df.columns if c in {self.zpid_col, self.pricehistory_col}]
        base_core = base_df[cols].copy()
        base_core = base_core[base_core[self.pricehistory_col].notna()]
        base_core[self.c_zpid_key] = base_core[self.zpid_col].astype(str)

        def parse_ph(x):
            if isinstance(x, (list, dict)):
                return x
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return np.nan
            return np.nan

        base_core["PH_JSON"] = base_core[self.pricehistory_col].apply(parse_ph)
        base_core = base_core[base_core["PH_JSON"].notna()]

        flat = base_core.explode("PH_JSON").rename(columns={"PH_JSON": "VAL"})
        flat["JSON_INDEX"] = flat.groupby(self.c_zpid_key).cumcount()
        v = flat["VAL"]

        def get_val(d, k, default=None):
            return d.get(k, default) if isinstance(d, dict) else default

        flat[self.c_evt_date] = pd.to_datetime(
            v.apply(lambda d: get_val(d, "date", None)),
            errors="coerce",
        )
        flat[self.c_evt_type] = v.apply(lambda d: str(get_val(d, "event", "")).lower())
        flat[self.c_evt_price] = pd.to_numeric(
            v.apply(lambda d: get_val(d, "price", None)),
            errors="coerce",
        )
        flat[self.c_evt_price_psf] = pd.to_numeric(
            v.apply(lambda d: get_val(d, "pricePerSquareFoot", None)),
            errors="coerce",
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

        ms = pd.to_numeric(flat["RAW_TIME_MS_STR"], errors="coerce")
        evt_ts = pd.to_datetime(ms, unit="ms", origin="unix", errors="coerce")
        flat[self.c_evt_ts] = evt_ts

        flat[self.c_sort_ts] = np.where(
            flat[self.c_evt_ts].notna(),
            flat[self.c_evt_ts],
            flat[self.c_evt_date],
        ).astype("datetime64[ns]")

        flat = flat[(flat[self.c_evt_date].notna()) | (flat[self.c_evt_ts].notna())]
        flat = flat.reset_index(drop=True)
        return flat

    def _add_sequence(self, events_df: pd.DataFrame) -> pd.DataFrame:
        df = events_df.sort_values(
            [self.c_zpid_key, self.c_evt_date, self.c_sort_ts, "JSON_INDEX"]
        ).copy()

        df[self.c_event_seq] = df.groupby(self.c_zpid_key).cumcount() + 1
        df["prev_ts"] = df.groupby(self.c_zpid_key)[self.c_sort_ts].shift(1)
        df["first_ts"] = df.groupby(self.c_zpid_key)[self.c_sort_ts].transform("min")

        df[self.c_days_prev] = (df[self.c_sort_ts] - df["prev_ts"]).dt.days
        df[self.c_days_first] = (df[self.c_sort_ts] - df["first_ts"]).dt.days
        df = df.drop(columns=["prev_ts", "first_ts"])
        return df

    def _make_base_snapshot(self, base_df: pd.DataFrame) -> pd.DataFrame:
        cols_no_json = [c for c in base_df.columns if c != self.pricehistory_col]
        base_no_json = base_df[cols_no_json].copy()

        if self.scrape_ts_col in base_no_json.columns:
            base_no_json[self.scrape_ts_col] = pd.to_datetime(
                base_no_json[self.scrape_ts_col],
                errors="coerce",
            )
            base_no_json = base_no_json.sort_values(
                [self.zpid_col, self.scrape_ts_col],
                ascending=[True, False],
            )
            base_latest = base_no_json.drop_duplicates(subset=[self.zpid_col], keep="first")
        else:
            base_latest = base_no_json.drop_duplicates(subset=[self.zpid_col], keep="first")

        base_latest = base_latest.rename(columns={self.zpid_col: self.c_base_zpid})
        base_latest[self.c_base_zpid_key] = base_latest[self.c_base_zpid].astype(str)
        return base_latest
