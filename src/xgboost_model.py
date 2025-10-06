"""XGBoost regression utilities for house-price modelling."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
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
        params: Dict[str, Any] = {
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
            "eval_metric": self.eval_metric,
        }
        return params


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

        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_valid_scaled, y_valid)],
            "verbose": False,
        }

        signature = inspect.signature(model.fit)
        if (
            self.config.early_stopping_rounds
            and "early_stopping_rounds" in signature.parameters
        ):
            fit_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds

        model.fit(X_train_scaled, y_train, **fit_kwargs)

        self.model = model

        predictions = model.predict(X_valid_scaled)
        metrics = self._compute_metrics(y_valid, predictions)
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


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output: Optional[Path]) -> None:
    """Plot predicted versus actual targets and save to *output* if provided."""
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
    """Plot residuals for diagnostic analysis."""
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
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regressor from a CSV file or a sample dataset.")
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Optional path to a CSV dataset. If omitted, a sample diabetes dataset from "
            "scikit-learn is used."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Name of the target column in the dataset (required when providing a CSV path).",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit list of feature columns. Defaults to numeric columns.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=XGBoostConfig.test_size,
        help="Fraction of the dataset to reserve for validation.",
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
        default=None,
        help="Optional directory to store generated plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display (useful for CI environments).",
    )
    args = parser.parse_args(argv)

    if args.data is None and args.target is not None:
        parser.error("The target column cannot be set without providing a dataset path.")
    if args.data is not None and args.target is None:
        parser.error("Please supply the target column name when providing a dataset path.")

    return args


def _load_sample_dataset() -> Tuple[np.ndarray, np.ndarray, Sequence[str], str]:
    """Return the diabetes regression dataset bundled with scikit-learn."""

    dataset = load_diabetes()
    feature_columns = list(dataset.feature_names)

    target_names_attr = getattr(dataset, "target_names", None)
    if isinstance(target_names_attr, (list, tuple)) and target_names_attr:
        target_name = str(target_names_attr[0])
    elif isinstance(target_names_attr, str):
        target_name = target_names_attr
    else:
        target_name = "target"

    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.float64)
    return X, y, feature_columns, target_name


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    config = XGBoostConfig(test_size=args.test_size, random_state=args.random_state)
    model = XGBoostRegressorModel(config)

    if args.data is None:
        print("No dataset provided. Using the sample diabetes dataset bundled with scikit-learn.")
        X, y, feature_names, target_name = _load_sample_dataset()
    else:
        X, y, feature_names = load_dataset(
            args.data,
            args.target,
            feature_columns=args.features,
        )
        target_name = args.target

    _, metrics = model.fit(X, y)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    if args.data is None:
        print(
            "Trained against the sample dataset with target column "
            f"'{target_name}' and {len(feature_names)} features: {', '.join(feature_names)}"
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = args.output_dir / "xgboost_pred_vs_actual.png"
        resid_path = args.output_dir / "xgboost_residuals.png"
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
