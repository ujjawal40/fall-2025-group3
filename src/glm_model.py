"""Generalised linear model utilities and command line interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
class GeneralizedLinearRegressionModel:
    """Train and evaluate a Tweedie-based generalised linear regression model."""

    config: GLMConfig = field(default_factory=GLMConfig)
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    model: Optional[TweedieRegressor] = field(default=None, init=False)
    last_metrics: Optional[Dict[str, float]] = field(default=None, init=False)
    _validation_features: Optional[np.ndarray] = field(default=None, init=False)
    _validation_targets: Optional[np.ndarray] = field(default=None, init=False)
    _validation_predictions: Optional[np.ndarray] = field(default=None, init=False)
    feature_names_: Optional[List[str]] = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[TweedieRegressor, Dict[str, float]]:
        """Fit the GLM to the provided dataset and return validation metrics."""
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self.feature_names_: Optional[List[str]] = (
            list(feature_names) if feature_names is not None else None
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
        self.last_metrics = metrics
        self._validation_features = X_valid
        self._validation_targets = y_valid
        self._validation_predictions = predictions
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

    def validation_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return validation features, targets, and predictions for plotting."""

        if (
            self._validation_features is None
            or self._validation_targets is None
            or self._validation_predictions is None
        ):
            raise RuntimeError("Validation results are not available. Fit the model first.")

        return (
            self._validation_features,
            self._validation_targets,
            self._validation_predictions,
        )


def _prepare_dataset(
    csv_path: Path,
    target_column: str,
    *,
    dropna: bool = True,
    feature_columns: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a dataset from *csv_path* and split it into features and target arrays."""

    dataframe = pd.read_csv(csv_path)

    if dropna:
        dataframe = dataframe.dropna(subset=[target_column])

    if feature_columns is None:
        feature_columns = [
            column
            for column in dataframe.columns
            if column != target_column and np.issubdtype(dataframe[column].dtype, np.number)
        ]
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No numeric feature columns available for training.")

    features = dataframe[feature_columns].to_numpy(dtype=float)
    target = dataframe[target_column].to_numpy(dtype=float)
    return features, target, list(feature_columns)


def _plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    title: str,
    filename: str,
) -> Path:
    """Create a scatter plot comparing actual versus predicted values."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    ax.plot([line_min, line_max], [line_min, line_max], "r--", label="Ideal fit")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    filename: str,
) -> Path:
    """Plot the residual distribution for diagnostic purposes."""

    residuals = y_true - y_pred
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("GLM residual distribution")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the GLM command line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the input CSV dataset")
    parser.add_argument("target", type=str, help="Name of the target column")
    parser.add_argument(
        "--test-size",
        type=float,
        default=GLMConfig.test_size,
        help="Fraction of data reserved for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=GLMConfig.power,
        help="Tweedie power parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=GLMConfig.alpha,
        help="Regularisation strength (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=GLMConfig.max_iter,
        help="Maximum number of solver iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/glm"),
        help="Directory to store generated plots (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for running the GLM training from the command line."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    X, y, feature_names = _prepare_dataset(args.csv, args.target)

    config = GLMConfig(
        test_size=args.test_size,
        power=args.power,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )

    model = GeneralizedLinearRegressionModel(config=config)
    _, metrics = model.fit(X, y, feature_names=feature_names)

    print("Validation metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper():<4}: {value:.4f}")

    if model.model is not None and model.feature_names_ is not None:
        coefficients = model.model.coef_
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        print("Top feature coefficients (absolute magnitude):")
        for idx in sorted_indices[: min(10, len(coefficients))]:
            name = model.feature_names_[idx] if idx < len(model.feature_names_) else f"f{idx}"
            print(f"  {name:<20} {coefficients[idx]: .4f}")

    _, y_true, y_pred = model.validation_results()
    output_dir = args.output_dir

    scatter_path = _plot_actual_vs_predicted(
        y_true,
        y_pred,
        output_dir,
        title="GLM validation predictions",
        filename="glm_actual_vs_predicted.png",
    )
    print(f"Saved actual vs predicted plot to {scatter_path}")

    residual_path = _plot_residual_distribution(
        y_true,
        y_pred,
        output_dir,
        filename="glm_residual_distribution.png",
    )
    print(f"Saved residual distribution plot to {residual_path}")


if __name__ == "__main__":
    main()
