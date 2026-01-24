import numpy as np
import pandas as pd

from trend_analyzer.features import cross_asset_features, equity_features


def _series(n: int = 400, start: str = "2020-01-01", seed: int = 0, base: float = 100.0) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(seed)
    px = base + np.cumsum(rng.normal(0, 1, size=n))
    px = np.maximum(px, 1.0)
    return pd.Series(px, index=idx)


def test_equity_features_columns_present():
    px = _series()
    f = equity_features(
        px,
        ema_fast=50,
        ema_slow=200,
        rsi=14,
        dd_window=252,
        vol_windows=[20, 60],
        z_window=252,
        price_z_window=60,
    )
    for col in ["px", "ema50", "ema200", "ema_ratio", "ema_slope", "d200", "rsi", "price_z", "dd", "vol20", "vol60"]:
        assert col in f.columns


def test_cross_asset_features_graceful_when_missing_inputs():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    aligned = pd.DataFrame({"eq_nifty50": _series(300).reindex(idx)}, index=idx)
    x = cross_asset_features(aligned, z_window=60)
    # No crash; should be empty because inputs missing
    assert isinstance(x, pd.DataFrame)


def test_cross_asset_features_some_outputs():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    aligned = pd.DataFrame(
        {
            "eq_nifty50": _series(400, seed=1, base=15000).reindex(idx),
            "eq_midcap100": _series(400, seed=8, base=20000).reindex(idx),
            "eq_smallcap100": _series(400, seed=9, base=12000).reindex(idx),
            "ro_gold": _series(400, seed=2, base=1500).reindex(idx),
            "ro_silver": _series(400, seed=3, base=20).reindex(idx),
            "ro_usdinr": _series(400, seed=4, base=75).reindex(idx),
            "ro_vix": _series(400, seed=5, base=20).reindex(idx),
            "y_us10y": _series(400, seed=6, base=3.0).reindex(idx),
            "y_us3m": _series(400, seed=7, base=0.5).reindex(idx),
        },
        index=idx,
    )
    x = cross_asset_features(aligned, z_window=120)
    assert "gold_vs_nifty_z" in x.columns
    assert "silver_vs_gold_z" in x.columns
    assert "usdinr_mom20_z" in x.columns
    assert "vix_level_z" in x.columns
    assert "us_curve_slope_z" in x.columns
    assert "midcap_vs_nifty_z" in x.columns
    assert "smallcap_vs_nifty_z" in x.columns
    assert "riskoff_composite" in x.columns

