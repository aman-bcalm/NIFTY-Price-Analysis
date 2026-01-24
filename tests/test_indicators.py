import numpy as np
import pandas as pd

from trend_analyzer.indicators import (
    drawdown_from_rolling_high,
    ema,
    forward_max_drawdown,
    forward_return,
    realized_vol,
    rsi_wilder,
    zscore,
)


def _price_series(n: int = 400) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # gentle uptrend with noise, always positive
    rng = np.random.default_rng(0)
    px = 100 + np.cumsum(rng.normal(0, 1, size=n)) + np.linspace(0, 20, n)
    px = np.maximum(px, 1.0)
    return pd.Series(px, index=idx)


def test_ema_shape_and_nan_handling():
    s = _price_series()
    out = ema(s, span=20)
    assert len(out) == len(s)
    assert out.isna().sum() == 0


def test_rsi_bounds():
    s = _price_series()
    out = rsi_wilder(s, period=14)
    # allow NaNs early, but all finite should be within [0,100]
    finite = out.dropna()
    assert (finite >= 0).all()
    assert (finite <= 100).all()


def test_realized_vol_non_negative():
    s = _price_series()
    vol = realized_vol(s, window=20)
    finite = vol.dropna()
    assert (finite >= 0).all()


def test_drawdown_is_leq_zero():
    s = _price_series()
    dd = drawdown_from_rolling_high(s, window=252)
    finite = dd.dropna()
    assert (finite <= 0).all()


def test_zscore_centered_roughly():
    s = _price_series()
    z = zscore(s, window=60).dropna()
    # not exactly 0 mean due to rolling, but should be small
    assert abs(z.mean()) < 0.25


def test_forward_return_and_fwd_mdd_shapes():
    s = _price_series()
    fr = forward_return(s, horizon_days=21)
    fmdd = forward_max_drawdown(s, horizon_days=21)
    assert fr.index.equals(s.index)
    assert fmdd.index.equals(s.index)
    # last horizon days will be NaN due to shift
    assert fr.tail(21).isna().all()
    assert fmdd.tail(21).isna().all()

