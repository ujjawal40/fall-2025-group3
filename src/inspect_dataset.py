"""Command line utility to inspect the structure of the housing dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_inspector import DataInspector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the CSV file to inspect")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to preview")
    args = parser.parse_args()

    inspector = DataInspector.from_csv(str(args.csv))
    shape = inspector.dataset_shape()
    print(f"Rows: {shape['rows']:,} | Columns: {shape['columns']:,}")

    print("\nMissing value report (sorted):")
    report = inspector.missing_value_report()
    print("Column                      Missing %")
    print("-------------------------- ----------")
    for column, fraction in report.items():
        print(f"{column:<26} {fraction * 100:>9.2f}")

    print("\nColumn summaries:")
    for summary in inspector.column_summaries():
        print(
            f"- {summary.name} | dtype={summary.dtype} | non-null={summary.non_null_count:,} "
            f"| missing={summary.missing_count:,} | unique={summary.unique_values}"
        )

    print("\nPreview:")
    print(inspector.head(args.head))
