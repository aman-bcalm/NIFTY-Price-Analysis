from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RiskModelConfig:
    horizon_days: int
    fwd_return_threshold: float
    fwd_max_drawdown_threshold: float
    min_train_years: int
    retrain_frequency: str  # pandas offset alias, e.g. "M"
    regularization_C: float


def walkforward_logistic_probabilities(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: RiskModelConfig,
) -> pd.Series:
    """
    Walk-forward monthly retraining logistic regression.

    - Trains only on rows with known labels (non-NaN)
    - Predicts probabilities for subsequent period until next retrain date
    """
    if not X.index.equals(y.index):
        y = y.reindex(X.index)

    idx = X.index
    probs = pd.Series(index=idx, dtype="float64", name="risk_off_prob")

    retrain_dates = (
        X.resample(cfg.retrain_frequency).last().index  # month ends present in data
    )
    if len(retrain_dates) == 0:
        return probs

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    C=float(cfg.regularization_C),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    )

    def has_min_history(train_index: pd.DatetimeIndex, train_end: pd.Timestamp) -> bool:
        if len(train_index) == 0:
            return False
        earliest = train_index.min()
        return earliest <= (train_end - pd.Timedelta(days=int(cfg.min_train_years * 365.25)))

    for i, train_end in enumerate(retrain_dates):
        # Predict window: (train_end, next_train_end]
        next_end = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else idx.max()
        predict_mask = (idx > train_end) & (idx <= next_end)
        if not predict_mask.any():
            continue

        train_mask = (idx <= train_end) & y.notna()
        train_idx = idx[train_mask]
        if not has_min_history(train_idx, train_end):
            continue

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask].astype(int)

        # Require both classes present
        if y_train.nunique(dropna=True) < 2:
            continue

        model.fit(X_train.values, y_train.values)

        X_pred = X.loc[predict_mask]
        p = model.predict_proba(X_pred.values)[:, 1]
        probs.loc[predict_mask] = p

    # For convenience, also fill probabilities on retrain_dates themselves using the model
    # from the previous period (if it exists). Leave NaN otherwise.
    return probs

