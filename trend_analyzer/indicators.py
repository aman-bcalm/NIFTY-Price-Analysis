from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def log_returns(price: pd.Series) -> pd.Series:
    return np.log(price).diff()


def realized_vol(price: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    lr = log_returns(price)
    vol = lr.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252.0)
    return vol


def drawdown_from_rolling_high(price: pd.Series, window: int) -> pd.Series:
    rolling_high = price.rolling(window).max()
    return price / rolling_high - 1.0


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def clamp(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def forward_return(price: pd.Series, horizon_days: int) -> pd.Series:
    return price.shift(-horizon_days) / price - 1.0


def forward_max_drawdown(price: pd.Series, horizon_days: int) -> pd.Series:
    # Compute max drawdown over the *future* window [t..t+horizon]
    # Vectorized across horizon (small: ~21 days).
    future = pd.concat([price.shift(-i) for i in range(horizon_days + 1)], axis=1)
    running_max = future.cummax(axis=1)
    dd = future / running_max - 1.0
    fmdd = dd.min(axis=1)
    # Require a full future window; otherwise return NaN (matches forward_return behavior).
    full_window_available = price.shift(-horizon_days).notna()
    return fmdd.where(full_window_available)

