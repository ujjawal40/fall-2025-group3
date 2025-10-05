"""XGBoost regression utilities and command line interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@dataclass
class XGBoostConfig:
    """Configuration parameters for the XGBoost regressor."""

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    eval_metric: str = "rmse"

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for :class:`xgboost.XGBRegressor`."""
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
            "tree_method": "hist",
        }


@dataclass
class XGBoostRegressorModel:
    """Train and evaluate an XGBoost regression model."""

    config: XGBoostConfig = field(default_factory=XGBoostConfig)
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    model: Optional[xgb.XGBRegressor] = field(default=None, init=False)
    last_metrics: Optional[Dict[str, float]] = field(default=None, init=False)
    last_evals_result: Optional[Dict[str, Dict[str, List[float]]]] = field(
        default=None, init=False
    )
    _validation_features: Optional[np.ndarray] = field(default=None, init=False)
    _validation_targets: Optional[np.ndarray] = field(default=None, init=False)
    _validation_predictions: Optional[np.ndarray] = field(default=None, init=False)
    feature_names_: Optional[List[str]] = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set_fraction: Optional[float] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """Fit the model and return evaluation metrics on the validation split."""
        if eval_set_fraction is None:
            eval_set_fraction = self.config.test_size

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self.feature_names_ = list(feature_names) if feature_names is not None else None

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        model = xgb.XGBRegressor(**self.config.to_model_kwargs())
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_valid_scaled, y_valid)],
            eval_metric=self.config.eval_metric,
            verbose=False,
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

        self.model = model

        predictions = model.predict(X_valid_scaled)
        metrics = self._compute_metrics(y_valid, predictions)
        self.last_metrics = metrics
        self.last_evals_result = model.evals_result()
        self._validation_features = X_valid
        self._validation_targets = y_valid
        self._validation_predictions = predictions
        return model, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted model."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        transformed = self.scaler.transform(X)
        return self.model.predict(transformed)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics for evaluation."""
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


def _plot_evaluation_history(
    evals_result: Optional[Dict[str, Dict[str, List[float]]]],
    metric: str,
    output_dir: Path,
    filename: str,
) -> Optional[Path]:
    """Plot the evaluation metric history captured during training."""

    if not evals_result:
        return None

    metric_history_train = evals_result.get("validation_0", {}).get(metric)
    metric_history_valid = evals_result.get("validation_1", {}).get(metric)

    if metric_history_train is None and metric_history_valid is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    rounds = np.arange(1, len(metric_history_valid or metric_history_train) + 1)
    if metric_history_train is not None:
        ax.plot(rounds[: len(metric_history_train)], metric_history_train, label="Training")
    if metric_history_valid is not None:
        ax.plot(rounds[: len(metric_history_valid)], metric_history_valid, label="Validation")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"XGBoost {metric.upper()} history")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: Optional[List[str]],
    output_dir: Path,
    filename: str,
) -> Optional[Path]:
    """Plot feature importances reported by the trained model."""

    if model is None:
        return None

    importances = model.feature_importances_
    if importances is None or not len(importances):
        return None

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_indices = np.argsort(importances)[::-1]
    ax.bar(np.array(feature_names)[sorted_indices], importances[sorted_indices])
    ax.set_ylabel("Importance score")
    ax.set_title("XGBoost feature importance")
    ax.set_xticklabels(np.array(feature_names)[sorted_indices], rotation=45, ha="right")
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the command line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the input CSV dataset")
    parser.add_argument("target", type=str, help="Name of the target column")
    parser.add_argument(
        "--test-size",
        type=float,
        default=XGBoostConfig.test_size,
        help="Fraction of data reserved for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=XGBoostConfig.learning_rate,
        help="Learning rate for boosting (default: %(default)s)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=XGBoostConfig.n_estimators,
        help="Number of boosting rounds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=XGBoostConfig.max_depth,
        help="Maximum tree depth (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/xgboost"),
        help="Directory to store generated plots (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for running XGBoost training from the command line."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    X, y, feature_names = _prepare_dataset(args.csv, args.target)

    config = XGBoostConfig(
        test_size=args.test_size,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    model = XGBoostRegressorModel(config=config)
    _, metrics = model.fit(X, y, feature_names=feature_names)

    print("Validation metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper():<4}: {value:.4f}")

    if model.model is not None and model.feature_names_ is not None:
        importances = model.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("Top feature importances:")
        for idx in sorted_indices[: min(10, len(importances))]:
            name = model.feature_names_[idx] if idx < len(model.feature_names_) else f"f{idx}"
            print(f"  {name:<20} {importances[idx]:.4f}")

    _, y_true, y_pred = model.validation_results()
    output_dir = args.output_dir

    scatter_path = _plot_actual_vs_predicted(
        y_true,
        y_pred,
        output_dir,
        title="XGBoost validation predictions",
        filename="xgboost_actual_vs_predicted.png",
    )
    print(f"Saved actual vs predicted plot to {scatter_path}")

    history_path = _plot_evaluation_history(
        model.last_evals_result,
        model.config.eval_metric,
        output_dir,
        filename="xgboost_learning_curve.png",
    )
    if history_path:
        print(f"Saved learning curve plot to {history_path}")

    importance_path = _plot_feature_importance(
        model.model,
        model.feature_names_,
        output_dir,
        filename="xgboost_feature_importance.png",
    )
    if importance_path:
        print(f"Saved feature importance plot to {importance_path}")


if __name__ == "__main__":
    main()
