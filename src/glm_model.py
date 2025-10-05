"""Generalised linear model utilities for regression tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
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
