"""XGBoost regression utilities aligned with the neural-network pipeline."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from data_preprocessor import DataPreprocessor


DEFAULT_DATASET = Path(__file__).with_name("sub_sample.csv")


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
    target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = field(
        default=None, init=False, repr=False
    )
    target_inverse: Optional[Callable[[np.ndarray], np.ndarray]] = field(
        default=None, init=False, repr=False
    )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set_fraction: Optional[float] = None,
        target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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

        if target_transform is not None:
            y_train_model = target_transform(y_train)
            y_valid_model = target_transform(y_valid)
        else:
            y_train_model = y_train
            y_valid_model = y_valid

        self.target_transform = target_transform
        self.target_inverse = target_inverse

        print(
            "Training split details:"  # user requested dimension info
        )
        print(
            f"  Train rows: {X_train.shape[0]} | Validation rows: {X_valid.shape[0]} | Features: {X_train.shape[1]}"
        )
        print(
            f"  Target mean (train): {np.mean(y_train):.2f} | Target mean (valid): {np.mean(y_valid):.2f}"
        )

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

        model.fit(X_train_scaled, y_train_model, **fit_kwargs)

        self.model = model

        predictions_model = model.predict(X_valid_scaled)

        y_eval = y_valid_model
        preds_eval = predictions_model

        if target_inverse is not None:
            y_eval = target_inverse(y_valid_model)
            preds_eval = target_inverse(predictions_model)

        metrics = self._compute_metrics(y_eval, preds_eval)

        if target_transform is not None and target_inverse is not None:
            transformed_metrics = self._compute_metrics(y_valid_model, predictions_model)
            metrics.update(
                {
                    "rmse_transformed": transformed_metrics["rmse"],
                    "mae_transformed": transformed_metrics["mae"],
                    "r2_transformed": transformed_metrics["r2"],
                }
            )

        return model, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted model."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        transformed = self.scaler.transform(X)
        preds = self.model.predict(transformed)
        if self.target_inverse is not None:
            return self.target_inverse(preds)
        return preds

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics for evaluation."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}


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
        description="Train an XGBoost regressor on the house-price dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Path to the training CSV. Defaults to 'sub_sample.csv' located alongside this script."
        ),
    )
    parser.add_argument(
        "--target",
        default="PRICE",
        help="Target column to model (case-insensitive). Use 'PRICE' or 'LOG_PPSQFT'.",
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

    return args


def _prepare_dataset(data_path: Path, target: str) -> Tuple[np.ndarray, np.ndarray, Sequence[str]]:
    """Load and preprocess the house-price dataset using the shared pipeline."""

    preprocessor = DataPreprocessor(dataset_path=data_path)
    raw_df = preprocessor.load_data()
    print(f"Columns available: {list(raw_df.columns)}")

    clean_df = preprocessor.clean_and_engineer(raw_df, one_hot=True)
    print(f"Cleaned data shape: {clean_df.shape}")

    X, y, feature_names = preprocessor.prepare_features(clean_df, target=target)
    print(f"Feature matrix shape: {X.shape} | Target shape: {y.shape}")
    print(f"Features used ({len(feature_names)}):")
    for name in feature_names:
        print(f"  - {name}")

    return X, y, feature_names


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    config = XGBoostConfig(test_size=args.test_size, random_state=args.random_state)
    model = XGBoostRegressorModel(config)

    data_path = args.data
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please supply the correct --data path."
        )

    X, y, feature_names = _prepare_dataset(data_path, args.target)

    target_key = args.target.upper()
    target_transform: Optional[Callable[[np.ndarray], np.ndarray]]
    target_inverse: Optional[Callable[[np.ndarray], np.ndarray]]

    if target_key == "PRICE":
        print("Applying log1p transform to PRICE target for modelling while reporting metrics in dollars.")
        target_transform = np.log1p
        target_inverse = np.expm1
    else:
        target_transform = None
        target_inverse = None

    _, metrics = model.fit(
        X,
        y,
        target_transform=target_transform,
        target_inverse=target_inverse,
    )

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

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
