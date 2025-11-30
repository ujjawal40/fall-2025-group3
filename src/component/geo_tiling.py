import numpy as np
import pandas as pd


def build_geo_tiling(
    combined_events: pd.DataFrame,
    resolutions=(6, 7, 8, 9),
    prefix: str = "H3_R",
):
    """
    Approximate H3-style tiling:
      - 1 row per ZIPCODE×YM
      - median LAT/LON
      - mode STATE/COUNTY
      - 'H3_R{r}' ≈ f"H3R{r}{lat_3dp}_{lon_3dp}"
    """
    required = ["ZIPCODE", "LATITUDE", "LONGITUDE"]
    if not all(c in combined_events.columns for c in required):
        return None

    df = combined_events.copy()

    if "EVT_DATE" in df.columns:
        df["EVT_DATE"] = pd.to_datetime(df["EVT_DATE"], errors="coerce")
        df["YM"] = df["EVT_DATE"].values.astype("datetime64[M]")
    elif "YM" in df.columns:
        df["YM"] = pd.to_datetime(df["YM"], errors="coerce").values.astype("datetime64[M]")
    else:
        return None

    df["ZIPCODE"] = df["ZIPCODE"].astype(str)
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    subset = df[["ZIPCODE", "YM", "LATITUDE", "LONGITUDE", "STATE", "COUNTY"]].dropna(
        subset=["ZIPCODE", "YM"]
    )
    if subset.empty:
        return None

    def mode_or_first(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else x.iloc[0]

    agg = subset.groupby(["ZIPCODE", "YM"], as_index=False).agg(
        LATITUDE=("LATITUDE", "median"),
        LONGITUDE=("LONGITUDE", "median"),
        STATE_MODE=("STATE", mode_or_first),
        COUNTY_MODE=("COUNTY", mode_or_first),
    )

    for r in resolutions:
        col_name = f"{prefix}{r}"
        agg[col_name] = (
            "H3R" + str(r)
            + agg["LATITUDE"].round(3).astype(str)
            + "_"
            + agg["LONGITUDE"].round(3).astype(str)
        )

    cols = ["ZIPCODE", "YM", "STATE_MODE", "COUNTY_MODE"] + [f"{prefix}{r}" for r in resolutions]
    return agg[cols]
