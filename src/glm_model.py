"""Command line utilities and helpers for Tweedie GLM regression workflows."""

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
class GLMTrainingArtifacts:
    """Container for validation data and diagnostic artefacts."""

    metrics: Dict[str, float]
    y_valid: np.ndarray
    predictions: np.ndarray
    feature_names: Sequence[str]


@dataclass
class GeneralizedLinearRegressionModel:
    """Train and evaluate a Tweedie-based generalised linear regression model."""

    config: GLMConfig = field(default_factory=GLMConfig)
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    model: Optional[TweedieRegressor] = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[TweedieRegressor, Dict[str, float]]:
        """Fit the GLM to the provided dataset and return validation metrics."""
        model, metrics, _y_valid, _predictions = self._fit_internal(X, y)
        return model, metrics

    def fit_with_artifacts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> GLMTrainingArtifacts:
        """Fit the GLM and retain artefacts for visualisation."""
        model, metrics, y_valid, predictions = self._fit_internal(X, y)
        return GLMTrainingArtifacts(
            metrics=metrics,
            y_valid=y_valid,
            predictions=predictions,
            feature_names=feature_names,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted GLM."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        transformed = self.scaler.transform(X)
        return self.model.predict(transformed)

    def _fit_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[TweedieRegressor, Dict[str, float], np.ndarray, np.ndarray]:
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
        return model, metrics, y_valid, predictions

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
    """Load a CSV file and return feature/target arrays with feature names."""
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


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output: Path) -> None:
    """Plot predicted versus actual targets and save to *output*."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    max_value = max(np.max(y_true), np.max(y_pred))
    min_value = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], "k--", label="Ideal")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output: Path) -> None:
    """Plot residuals for diagnostic analysis."""
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_coefficients(model: TweedieRegressor, feature_names: Sequence[str], output: Path) -> None:
    """Visualise the learned GLM coefficients."""
    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, model.coef_)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Coefficient value")
    plt.title("GLM Coefficients")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def run_training(
    csv_path: Path,
    target_column: str,
    *,
    feature_columns: Optional[Iterable[str]],
    output_dir: Path,
    config: Optional[GLMConfig] = None,
) -> GLMTrainingArtifacts:
    """Convenience wrapper to train the GLM and emit diagnostic plots."""
    X, y, feature_names = load_dataset(
        csv_path,
        target_column,
        feature_columns=feature_columns,
    )

    model = GeneralizedLinearRegressionModel(config or GLMConfig())
    artifacts = model.fit_with_artifacts(X, y, feature_names=feature_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_predictions(artifacts.y_valid, artifacts.predictions, output_dir / "pred_vs_actual.png")
    plot_residuals(artifacts.y_valid, artifacts.predictions, output_dir / "residuals.png")
    if model.model is not None:
        plot_coefficients(model.model, artifacts.feature_names, output_dir / "coefficients.png")

    return artifacts


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Execute the GLM regression workflow from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the training dataset (CSV)")
    parser.add_argument("target", help="Name of the target column")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of feature column names. Defaults to all numeric columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/glm"),
        help="Directory to store generated plots.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction for validation")
    parser.add_argument("--power", type=float, default=1.5, help="Tweedie power parameter")
    parser.add_argument("--alpha", type=float, default=0.0, help="Overall regularisation strength")
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help="Elastic-net mixing parameter (None for Ridge-style penalty)",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum optimiser iterations")

    args = parser.parse_args(argv)

    config = GLMConfig(
        test_size=args.test_size,
        power=args.power,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
    )

    artifacts = run_training(
        args.csv,
        args.target,
        feature_columns=args.features,
        output_dir=args.output_dir,
        config=config,
    )

    print("Validation metrics:")
    for name, value in artifacts.metrics.items():
        print(f"  {name.upper():<4}: {value:.4f}")


if __name__ == "__main__":
    main()
