"""XGBoost regression utilities for house-price modelling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
            eval_set=[(X_valid_scaled, y_valid)],
            eval_metric=self.config.eval_metric,
            verbose=False,
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

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
