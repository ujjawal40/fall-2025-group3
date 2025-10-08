"""Reusable data preprocessing utilities for house price models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class DataPreprocessor:
    """Handle data loading, cleaning, and feature engineering steps."""

    dataset_path: Path | None = None

    def __post_init__(self) -> None:
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
    # Loading helpers
    # ------------------------------------------------------------------
    def load_data(self, file_path: str | Path | None = None) -> pd.DataFrame:
        """Load the CSV dataset and report its shape."""

        path = Path(file_path) if file_path is not None else self.dataset_path
        if path is None:
            raise ValueError("A dataset path must be provided before loading data.")

        try:
            df = pd.read_csv(path)
        except FileNotFoundError as exc:  # pragma: no cover - passthrough for CLI UX
            raise FileNotFoundError(f"File not found: {path}") from exc

        print(f"Loaded data shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------
    @staticmethod
    def num_from_text(series: pd.Series, allow_comma: bool = True) -> pd.Series:
        """Extract the first number from mixed text columns."""

        pattern = r"([-+]?\d[\d,]*\.?\d*)"
        cleaned = series.astype(str).str.extract(pattern, expand=False)

        if allow_comma:
            cleaned = cleaned.str.replace(",", "", regex=False)

        return pd.to_numeric(cleaned, errors="coerce")

    def clean_and_engineer(self, df: pd.DataFrame, one_hot: bool = True) -> pd.DataFrame:
        """Replicate the neural-network feature engineering workflow."""

        df = df.copy()

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

        if "LOT" in df.columns:
            mask = df["LOT"] < 10
            df.loc[mask, "LOT"] = df.loc[mask, "LOT"] * 43_560

        raw_park = "PARKING" if "PARKING" in df.columns else "PARKINGTOTALSPACES"
        if raw_park in df.columns:
            df["GARAGE_SPACES"] = df[raw_park]
            df.drop(columns=[raw_park], inplace=True)

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

        yn = {"Yes": 1, "No": 0}
        if "NEWCONSTRUCTIONFLAG" in df.columns:
            df["NEWCONSTRUCTIONFLAG"] = df["NEWCONSTRUCTIONFLAG"].map(yn)
        if "SENIORLIVING" in df.columns:
            df["SENIORLIVING"] = df["SENIORLIVING"].map(yn)

        if "CREATEDAT_YEAR" in df.columns and "YEARBUILT" in df.columns:
            df["PROPERTY_AGE"] = df["CREATEDAT_YEAR"] - df["YEARBUILT"]
            df.loc[df["PROPERTY_AGE"] < 0, "PROPERTY_AGE"] = np.nan

        for col in ["SQFT", "LOT", "MEDIAN_DAYS_ON_MARKET", "PRICE"]:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan

        if "SQFT" in df.columns:
            df["LOG_SQFT"] = np.log1p(df["SQFT"])
        if "LOT" in df.columns:
            df["LOG_LOT"] = np.log1p(df["LOT"])
        if "MEDIAN_DAYS_ON_MARKET" in df.columns:
            df["LOG_DOM"] = np.log1p(df["MEDIAN_DAYS_ON_MARKET"])

        if "CREATEDAT_MONTH" in df.columns:
            two_pi = 2 * np.pi
            df["MONTH_SIN"] = np.sin(two_pi * df["CREATEDAT_MONTH"] / 12)
            df["MONTH_COS"] = np.cos(two_pi * df["CREATEDAT_MONTH"] / 12)

        base_imp = ["SQFT", "LOT", "GARAGE_SPACES", "PROPERTY_AGE"]
        for col in base_imp:
            if col in df.columns:
                df[f"MISS_{col}"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())

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

        remain_obj = df.select_dtypes("object").columns
        truly_cat: list[str] = []
        for col in remain_obj:
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notna().mean() >= 0.80:
                df[col] = conv
            else:
                truly_cat.append(col)

        if one_hot and truly_cat:
            df = pd.get_dummies(df, columns=truly_cat, dummy_na=True, prefix_sep="==")
        else:
            df.drop(columns=truly_cat, inplace=True)

        return df

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    def prepare_features(
        self, df: pd.DataFrame, *, target: str = "LOG_PPSQFT"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector using the cleaned frame."""

        df = df[(df["PRICE"] > 0) & (df["SQFT"] > 0)].copy()
        if df.empty:
            raise ValueError("No rows remain after filtering invalid PRICE/SQFT values.")

        df["PPSQFT"] = df["PRICE"] / df["SQFT"].clip(lower=1)
        df["LOG_PPSQFT"] = np.log1p(df["PPSQFT"])

        df = df[np.isfinite(df["LOG_PPSQFT"])].copy()

        numeric_df = df.select_dtypes(include=[np.number])
        drop_cols = ["PRICE", "PPSQFT", "LOG_PPSQFT", "ZPID"]
        features_df = numeric_df.drop(columns=drop_cols, errors="ignore")
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for col in features_df.columns:
            if features_df[col].isna().all():
                features_df[col] = 0.0
            else:
                features_df[col] = features_df[col].fillna(features_df[col].median())

        features_df = features_df.astype(np.float64)
        features_df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

        target_key = target.upper()
        if target_key == "LOG_PPSQFT":
            y = df["LOG_PPSQFT"].to_numpy(dtype=np.float64)
        elif target_key == "PRICE":
            y = df["PRICE"].to_numpy(dtype=np.float64)
        else:
            raise ValueError(
                "Unsupported target specified. Use 'PRICE' or 'LOG_PPSQFT'."
            )

        X = features_df.to_numpy(dtype=np.float64)
        feature_names = features_df.columns.to_numpy()

        if not np.isfinite(X).all():
            raise ValueError("Features contain non-finite values after preprocessing.")
        if not np.isfinite(y).all():
            raise ValueError("Target contains non-finite values after preprocessing.")

        return X, y, feature_names

