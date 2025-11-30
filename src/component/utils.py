import numpy as np
import pandas as pd


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes & convert some objects to category to save RAM."""
    if df.empty:
        return df
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
            if nun and nun / max(len(s), 1) <= 0.4:
                df[c] = df[c].astype("category")
    return df


def safe_to_double(series: pd.Series) -> pd.Series:
    """Clean '1,234', '$123', '5%' → float."""
    s = series.astype(str).str.replace(r"[,\s\$%]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def safe_to_binary_from_text(series: pd.Series) -> pd.Series:
    u = series.astype(str).str.upper().str.strip()
    mapping = {
        "Y": 1, "YES": 1, "TRUE": 1, "T": 1, "1": 1,
        "N": 0, "NO": 0, "FALSE": 0, "F": 0, "0": 0,
    }
    return u.map(mapping).astype("float32")


def safe_to_binary_from_number(series: pd.Series) -> pd.Series:
    x = safe_to_double(series)
    out = pd.Series(np.nan, index=series.index, dtype="float32")
    out[x == 1] = 1
    out[x == 0] = 0
    return out
