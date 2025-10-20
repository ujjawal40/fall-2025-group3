"""Utilities for inspecting dataset structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass
class ColumnSummary:
    """Summary information for a single column in a dataset."""

    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    unique_values: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the summary to a serialisable dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "non_null_count": self.non_null_count,
            "missing_count": self.missing_count,
            "unique_values": self.unique_values,
        }


class DataInspector:
    """Inspect tabular data sources to understand their structure."""

    def __init__(self, dataframe: pd.DataFrame):
        """Initialise the inspector with a dataframe to analyse."""
        self._df = dataframe.copy()

    @classmethod
    def from_csv(
        cls, file_path: str, *, nrows: Optional[int] = None, encoding: Optional[str] = None
    ) -> "DataInspector":
        """Create an inspector instance by reading a CSV file."""
        dataframe = pd.read_csv(file_path, nrows=nrows, encoding=encoding)
        return cls(dataframe)

    def column_summaries(self) -> List[ColumnSummary]:
        """Produce per-column summaries including missingness and type information."""
        summaries: List[ColumnSummary] = []
        for column in self._df.columns:
            series = self._df[column]
            summaries.append(
                ColumnSummary(
                    name=column,
                    dtype=str(series.dtype),
                    non_null_count=int(series.notna().sum()),
                    missing_count=int(series.isna().sum()),
                    unique_values=int(series.nunique(dropna=True)) if series.ndim == 1 else None,
                )
            )
        return summaries

    def dataset_shape(self) -> Dict[str, int]:
        """Return the dataset shape as a dictionary for easier serialisation."""
        rows, columns = self._df.shape
        return {"rows": int(rows), "columns": int(columns)}

    def basic_statistics(self, include: Iterable[str] = ("number", "object")) -> Mapping[str, pd.DataFrame]:
        """Return descriptive statistics by dtype category."""
        stats: Dict[str, pd.DataFrame] = {}
        for dtype_group in include:
            try:
                stats[dtype_group] = self._df.describe(include=[dtype_group]).transpose()
            except ValueError:
                # Raised when no columns of the requested dtype are present.
                continue
        return stats

    def missing_value_report(self) -> pd.Series:
        """Return the fraction of missing values per column."""
        return self._df.isna().mean().sort_values(ascending=False)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first *n* records for manual inspection."""
        return self._df.head(n)
