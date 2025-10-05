"""Utility helpers and a command line runner for Tweedie GLM regression."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class GLMConfig:
    """Configuration for the Tweedie-based generalised linear regressor."""

    test_size: float = 0.2
    random_state: int = 42
    power: float = 1.5
    alpha: float = 0.0
    l1_ratio: Optional[float] = None
    max_iter: int = 1000


@dataclass
class GLMWorkflow:
    """End-to-end training workflow for a Tweedie GLM."""

    config: GLMConfig
    scaler: Optional[StandardScaler] = None
    model: Optional[TweedieRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model and return validation labels and predictions."""

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        self.model = TweedieRegressor(
            power=self.config.power,
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            max_iter=self.config.max_iter,
        )
        self.model.fit(X_train_scaled, y_train)

        predictions = self.model.predict(X_valid_scaled)
        return y_valid, predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with the fitted GLM."""

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model must be fitted before predicting.")
        transformed = self.scaler.transform(X)
        return self.model.predict(transformed)


def load_dataset(
    csv_path: Path,
    target_column: str,
    *,
    feature_columns: Optional[Iterable[str]] = None,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the CSV file and return features, targets, and the feature names."""

    dataframe = pd.read_csv(csv_path)
    if dropna:
        dataframe = dataframe.dropna(subset=[target_column])

    if feature_columns is None:
        feature_columns = [
            column
            for column in dataframe.columns
            if column != target_column
            and np.issubdtype(dataframe[column].dtype, np.number)
        ]
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No numeric feature columns were provided for training.")

    features = dataframe[feature_columns].to_numpy(dtype=float)
    target = dataframe[target_column].to_numpy(dtype=float)
    return features, target, feature_columns


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics for validation output."""

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    *,
    title: str = "Actual vs Predicted",
) -> Path:
    """Create a scatter plot that compares predictions to ground truth."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="black")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal fit")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "glm_actual_vs_predicted.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot the residual distribution for diagnostic checks."""

    residuals = y_true - y_pred
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("GLM residual distribution")
    fig.tight_layout()
    output_path = output_dir / "glm_residuals.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the training CSV file.")
    parser.add_argument("target", help="Name of the target column in the CSV file.")
    parser.add_argument(
        "--features",
        nargs="*",
        help="Optional list of feature columns to use. Defaults to all numeric columns.",
    )
    parser.add_argument(
        "--no-dropna",
        action="store_true",
        help="Keep rows with missing target values instead of dropping them.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=GLMConfig.test_size,
        help="Fraction of the dataset to use for validation (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=GLMConfig.random_state,
        help="Random seed used for train/validation splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saving evaluation plots (default: artifacts).",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=GLMConfig.power,
        help="Tweedie power parameter (default: 1.5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=GLMConfig.alpha,
        help="Regularisation strength for the GLM (default: 0.0).",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=GLMConfig.l1_ratio,
        help="Elastic-net mixing parameter when alpha > 0 (default: None).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=GLMConfig.max_iter,
        help="Maximum number of iterations for fitting (default: 1000).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the command line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = GLMConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        power=args.power,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
    )
    features, target, _feature_names = load_dataset(
        args.csv_path,
        args.target,
        feature_columns=args.features,
        dropna=not args.no_dropna,
    )

    workflow = GLMWorkflow(config=config)
    y_valid, y_pred = workflow.fit(features, target)
    metrics = regression_metrics(y_valid, y_pred)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    output_dir = args.output_dir
    actual_plot = plot_predictions(y_valid, y_pred, output_dir)
    residual_plot = plot_residuals(y_valid, y_pred, output_dir)

    print(f"Saved actual vs predicted plot to: {actual_plot}")
    print(f"Saved residual plot to: {residual_plot}")


if __name__ == "__main__":
    main()
