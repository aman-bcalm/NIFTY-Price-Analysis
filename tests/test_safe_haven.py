import numpy as np
import pandas as pd

from trend_analyzer.scoring import compute_safe_haven_stretch


def _series(n: int = 600, seed: int = 0, drift: float = 0.0) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    # Use a log-random-walk so the series stays positive and doesn't clip at 1.0,
    # which can cause RSI/zscore degeneracy in tests.
    log_steps = rng.normal(loc=drift / 100.0, scale=0.01, size=n)
    px = 100.0 * np.exp(np.cumsum(log_steps))
    return pd.Series(px, index=idx)


def test_safe_haven_score_bounds():
    s = _series()
    out = compute_safe_haven_stretch(s, rsi_period=14, z_windows=(60, 252))
    sc = out["safe_score"].dropna()
    assert (sc >= 0).all()
    assert (sc <= 100).all()


def test_safe_haven_label_present():
    s = _series()
    out = compute_safe_haven_stretch(s)
    assert "safe_label" in out.columns
    # Should have at least some non-empty labels after warmup
    assert out["safe_label"].astype(str).nunique() > 1


def test_safe_haven_trending_up_is_more_overbought_than_trending_down():
    up = _series(seed=1, drift=0.2)
    down = _series(seed=2, drift=-0.2)
    oup = compute_safe_haven_stretch(up)
    odn = compute_safe_haven_stretch(down)
    # Compare late-period averages (after warmup)
    a = oup["safe_score"].tail(100).mean()
    b = odn["safe_score"].tail(100).mean()
    assert a > b

