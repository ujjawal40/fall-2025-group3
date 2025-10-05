"""Utility helpers and a command line runner for XGBoost regression."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@dataclass
class XGBoostConfig:
    """Configuration options used by :class:`xgboost.XGBRegressor`."""

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    eval_metric: str = "rmse"
    tree_method: str = "hist"

    def to_model_kwargs(self) -> Dict[str, float | int | str]:
        """Return keyword arguments for constructing the regressor."""

        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "objective": "reg:squarederror",
            "tree_method": self.tree_method,
        }


@dataclass
class XGBoostWorkflow:
    """End-to-end training and evaluation workflow for XGBoost."""

    config: XGBoostConfig
    scaler: Optional[StandardScaler] = None
    model: Optional[xgb.XGBRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Train the model and return validation targets and predictions."""

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        self.model = xgb.XGBRegressor(**self.config.to_model_kwargs())
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_valid_scaled, y_valid)],
            eval_metric=self.config.eval_metric,
            verbose=False,
        )

        predictions = self.model.predict(X_valid_scaled)
        return y_valid, predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted model."""

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
    """Load a CSV dataset and return the feature matrix and target vector."""

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
    """Compute RMSE, MAE, and R² metrics for regression predictions."""

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
    """Create a scatter plot comparing actual values with predictions."""

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
    output_path = output_dir / "xgboost_actual_vs_predicted.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot residual distribution for diagnostics."""

    residuals = y_true - y_pred
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual distribution")
    fig.tight_layout()
    output_path = output_dir / "xgboost_residuals.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_feature_importance(
    feature_names: Sequence[str],
    model: xgb.XGBRegressor,
    output_dir: Path,
) -> Optional[Path]:
    """Plot feature importances if available."""

    if not hasattr(model, "feature_importances_"):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[index] for index in sorted_indices]
    sorted_importances = importances[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_names[::-1], sorted_importances[::-1])
    ax.set_xlabel("Importance score")
    ax.set_title("XGBoost feature importance")
    fig.tight_layout()
    output_path = output_dir / "xgboost_feature_importance.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by :func:`main`."""

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
        default=XGBoostConfig.test_size,
        help="Fraction of data reserved for validation (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=XGBoostConfig.random_state,
        help="Random seed used for train/validation splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saving evaluation plots (default: artifacts).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=XGBoostConfig.learning_rate,
        help="Learning rate for boosting (default: 0.05).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=XGBoostConfig.n_estimators,
        help="Number of trees to train (default: 400).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=XGBoostConfig.max_depth,
        help="Maximum depth of each tree (default: 6).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the command line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = XGBoostConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    features, target, feature_names = load_dataset(
        args.csv_path,
        args.target,
        feature_columns=args.features,
        dropna=not args.no_dropna,
    )

    workflow = XGBoostWorkflow(config=config)
    y_valid, y_pred = workflow.fit(features, target)
    metrics = regression_metrics(y_valid, y_pred)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    output_dir = args.output_dir
    actual_plot = plot_predictions(y_valid, y_pred, output_dir)
    residual_plot = plot_residuals(y_valid, y_pred, output_dir)
    importance_plot = None
    if workflow.model is not None:
        importance_plot = plot_feature_importance(feature_names, workflow.model, output_dir)

    print(f"Saved actual vs predicted plot to: {actual_plot}")
    print(f"Saved residual plot to: {residual_plot}")
    if importance_plot is not None:
        print(f"Saved feature importance plot to: {importance_plot}")


if __name__ == "__main__":
    main()
