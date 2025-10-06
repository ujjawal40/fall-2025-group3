"""Generalised linear model utilities for regression tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

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
class GeneralizedLinearRegressionModel:
    """Train and evaluate a Tweedie-based generalised linear regression model."""

    config: GLMConfig = field(default_factory=GLMConfig)
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    model: Optional[TweedieRegressor] = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[TweedieRegressor, Dict[str, float]]:
        """Fit the GLM to the provided dataset and return validation metrics."""
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        model = TweedieRegressor(
            power=self.config.power,
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            max_iter=self.config.max_iter,
        )
        model.fit(X_train_scaled, y_train)
        self.model = model

        predictions = model.predict(X_valid_scaled)
        metrics = self._compute_metrics(y_valid, predictions)
        return model, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted GLM."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        transformed = self.scaler.transform(X)
        return self.model.predict(transformed)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute RMSE, MAE, and :math:`R^2` metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}


def load_dataset(
    csv_path: Path,
    target_column: str,
    *,
    feature_columns: Optional[Iterable[str]] = None,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str]]:
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

    X = dataframe[feature_columns].to_numpy(dtype=np.float64)
    y = dataframe[target_column].to_numpy(dtype=np.float64)
    return X, y, feature_columns


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output: Optional[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    max_value = max(np.max(y_true), np.max(y_pred))
    min_value = min(np.min(y_true), np.min(y_pred))
    ax.plot([min_value, max_value], [min_value, max_value], "k--", label="Ideal")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    fig.tight_layout()

    if output is None:
        plt.show()
    else:
        fig.savefig(output)
        plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output: Optional[Path]) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    fig.tight_layout()

    if output is None:
        plt.show()
    else:
        fig.savefig(output)
        plt.close(fig)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GLM regressor from a CSV file.")
    parser.add_argument("data", type=Path, help="Path to the CSV dataset")
    parser.add_argument("target", help="Name of the target column in the dataset")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit list of feature columns. Defaults to numeric columns.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=GLMConfig.test_size,
        help="Fraction of the dataset to reserve for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=GLMConfig.random_state,
        help="Random seed used for train/validation splitting.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=GLMConfig.power,
        help="Tweedie power parameter.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=GLMConfig.alpha,
        help="Regularisation strength for the Tweedie regressor.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help="Elastic-net mixing parameter (0=L2, 1=L1).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=GLMConfig.max_iter,
        help="Maximum iterations for the optimiser.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store generated plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display (useful for CI environments).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    config = GLMConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        power=args.power,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
    )
    model = GeneralizedLinearRegressionModel(config)

    X, y, _ = load_dataset(
        args.data,
        args.target,
        feature_columns=args.features,
    )

    _, metrics = model.fit(X, y)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = args.output_dir / "glm_pred_vs_actual.png"
        resid_path = args.output_dir / "glm_residuals.png"
    else:
        pred_path = None
        resid_path = None

    if args.no_show:
        plt.ioff()

    y_pred_all = model.predict(X)
    plot_predictions(y, y_pred_all, pred_path)
    plot_residuals(y, y_pred_all, resid_path)

    if pred_path is not None:
        print(f"Saved prediction plot to {pred_path}")
    if resid_path is not None:
        print(f"Saved residual plot to {resid_path}")


if __name__ == "__main__":
    main()
