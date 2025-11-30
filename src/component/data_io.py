import pandas as pd

from .config import (
    RAW_CSV_PATH,
    MIN_START_DATE,
    MIN_SOLD_PER_ZIP_M,
    MIN_LIST_PER_ZIP_M,
)
from .utils import downcast_df
from .events import CombinedEventsBuilder
from .zip_index import build_all_features


def load_raw_and_build_features():
    """
    Full I/O + preprocessing:
      1) Read RAW_CSV_PATH (sub_sample.csv)
      2) Downcast dtypes
      3) Build combined_events from PRICEHISTORY
      4) Build ZIP×month feature matrix feat_df
    """
    print(f"Loading raw data from {RAW_CSV_PATH} …")
    raw_df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    raw_df = downcast_df(raw_df)

    print("Building combined_events from PRICEHISTORY…")
    builder = CombinedEventsBuilder(
        base_df=raw_df,
        zpid_col="ZPID",
        pricehistory_col="PRICEHISTORY",
        scrape_ts_col="SCRAPEDAT",
    )
    combined_events = builder.build()
    print("combined_events shape:", combined_events.shape)

    print("Orchestrating full feature build…")
    feat_df = build_all_features(
        combined_events=combined_events,
        h3_resolutions=(6, 7, 8, 9),
        min_start_date=MIN_START_DATE,
        min_sold_per_zip_m=MIN_SOLD_PER_ZIP_M,
        min_list_per_zip_m=MIN_LIST_PER_ZIP_M,
    )
    print("feat_df ready. Columns:", len(feat_df.columns), " | rows:", len(feat_df))

    return combined_events, feat_df
