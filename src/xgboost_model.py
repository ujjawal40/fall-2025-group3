"""Command line entry-point and helpers for the XGBoost regression workflow.

The module keeps the original :class:`XGBoostRegressorModel` API that existed
before the CLI refactor so downstream notebooks or scripts can continue to
import it.  A thin ``main`` function now wraps the workflow so developers can
run the model end-to-end from the terminal while still getting useful plots and
metric summaries for debugging merge conflicts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

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
class TrainingArtifacts:
    """Return value from :func:`run_training` for downstream analysis."""

    metrics: Dict[str, float]
    y_valid: np.ndarray
    predictions: np.ndarray
    evals_result: Dict[str, Dict[str, Sequence[float]]]
    feature_names: Sequence[str]


@dataclass
class XGBoostRegressorModel:
    """Train and evaluate an XGBoost regression model."""

    config: XGBoostConfig = field(default_factory=XGBoostConfig)
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    model: Optional[xgb.XGBRegressor] = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set_fraction: Optional[float] = None,
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
        return model, metrics

    def fit_with_artifacts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> TrainingArtifacts:
        """Fit the model and return evaluation metrics plus plotting artefacts."""
        model, metrics, y_valid, predictions, evals_result = self._fit_internal(X, y)
        return TrainingArtifacts(
            metrics=metrics,
            y_valid=y_valid,
            predictions=predictions,
            evals_result=evals_result,
            feature_names=feature_names,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted model."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        transformed = self.scaler.transform(X)
        return self.model.predict(transformed)

    def _fit_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float], np.ndarray, np.ndarray, Dict[str, Dict[str, Sequence[float]]]]:
        """Shared implementation between :meth:`fit` and :meth:`fit_with_artifacts`."""
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

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
        return model, metrics, y_valid, predictions, model.evals_result()

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics for evaluation."""
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
    """Load a CSV dataset and return feature/target arrays plus feature names."""
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


def plot_learning_curve(
    evals_result: Dict[str, Dict[str, Sequence[float]]],
    output: Path,
) -> None:
    """Plot the training and validation metric history."""
    plt.figure(figsize=(6, 4))
    for dataset_name, metric_history in evals_result.items():
        for metric_name, values in metric_history.items():
            label = f"{dataset_name} - {metric_name}"
            plt.plot(values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Metric value")
    plt.title("Evaluation history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: Sequence[str],
    output: Path,
) -> None:
    """Plot feature importances reported by the trained model."""
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(
        [feature_names[idx] for idx in sorted_indices],
        importance[sorted_indices],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance score")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def run_training(
    csv_path: Path,
    target_column: str,
    *,
    feature_columns: Optional[Iterable[str]],
    output_dir: Path,
    config: Optional[XGBoostConfig] = None,
) -> TrainingArtifacts:
    """Convenience wrapper used by :func:`main` and tests."""
    X, y, feature_names = load_dataset(
        csv_path,
        target_column,
        feature_columns=feature_columns,
    )

    model = XGBoostRegressorModel(config or XGBoostConfig())
    artifacts = model.fit_with_artifacts(X, y, feature_names=feature_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_predictions(artifacts.y_valid, artifacts.predictions, output_dir / "pred_vs_actual.png")
    plot_residuals(artifacts.y_valid, artifacts.predictions, output_dir / "residuals.png")
    plot_learning_curve(artifacts.evals_result, output_dir / "evaluation_history.png")
    plot_feature_importance(model.model, artifacts.feature_names, output_dir / "feature_importance.png")

    return artifacts


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the XGBoost regression workflow from the command line."""
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
        default=Path("reports/xgboost"),
        help="Directory to store generated plots.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction for validation")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=6, help="Tree depth")
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio for the training instances",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Subsample ratio for columns when constructing trees",
    )
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularisation term")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularisation term")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Rounds of no improvement before early stopping",
    )
    parser.add_argument(
        "--eval-metric",
        default="rmse",
        help="Evaluation metric reported by XGBoost during training",
    )

    args = parser.parse_args(argv)

    config = XGBoostConfig(
        test_size=args.test_size,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_metric=args.eval_metric,
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
