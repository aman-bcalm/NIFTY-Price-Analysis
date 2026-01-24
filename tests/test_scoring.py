import numpy as np
import pandas as pd

from trend_analyzer.scoring import assemble_final_score, compute_score_components


def _equity_feat(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Minimal columns needed by compute_score_components
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "ema_slope_z": rng.normal(size=n),
            "ema_ratio_z": rng.normal(size=n),
            "d200_z": rng.normal(size=n),
            "dd_z": rng.normal(size=n),
            "price_z": rng.normal(size=n),
            "rsi": np.clip(50 + rng.normal(scale=10, size=n), 0, 100),
        },
        index=idx,
    )
    return df


def test_score_bounds_and_labels():
    f = _equity_feat()
    comps = compute_score_components(f, trend_score_max=60.0, reversion_adj_max=20.0)
    p = pd.Series(0.3, index=comps.index, name="risk_off_prob")
    out = assemble_final_score(
        components=comps,
        risk_off_prob=p,
        risk_penalty_max=20.0,
        neutral_shift=20.0,
    )
    assert "score" in out.columns
    assert "label" in out.columns
    assert out["score"].between(0, 100).all()


def test_label_boundaries():
    from trend_analyzer.scoring import score_to_label

    assert score_to_label(0) == "oversold_panic"
    assert score_to_label(19.999) == "oversold_panic"
    assert score_to_label(20.0) == "bearish_below_trend"
    assert score_to_label(39.999) == "bearish_below_trend"
    assert score_to_label(40.0) == "neutral_fair_vs_trend"
    assert score_to_label(59.999) == "neutral_fair_vs_trend"
    assert score_to_label(60.0) == "bullish_overboughtish"
    assert score_to_label(79.999) == "bullish_overboughtish"
    assert score_to_label(80.0) == "euphoric_very_overbought"

