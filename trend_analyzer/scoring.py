from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import clamp


def score_to_label(score: float) -> str:
    if score < 20:
        return "oversold_panic"
    if score < 40:
        return "bearish_below_trend"
    if score < 60:
        return "neutral_fair_vs_trend"
    if score < 80:
        return "bullish_overboughtish"
    return "euphoric_very_overbought"


def compute_score_components(
    equity_feat: pd.DataFrame,
    *,
    trend_score_max: float,
    reversion_adj_max: float,
) -> pd.DataFrame:
    """
    Build TrendScore (0..trend_score_max) and ReversionAdjustment (-reversion_adj_max..+reversion_adj_max)
    from per-index equity features. Uses bounded transforms to keep outputs stable.
    """
    f = equity_feat.copy()

    # --- TrendScore ---
    # Raw trend is a mix of z-scored EMA slope and EMA ratio, plus being above/below EMA200 (d200_z),
    # and penalize large drawdown.
    trend_raw = (
        0.45 * f["ema_slope_z"]
        + 0.25 * f["ema_ratio_z"]
        + 0.20 * f["d200_z"]
        - 0.10 * f["dd_z"]
    )
    # Map via tanh to [0, trend_score_max]
    trend_unit = (np.tanh(trend_raw / 2.0) + 1.0) / 2.0
    trend_score = trend_unit * float(trend_score_max)

    # --- ReversionAdjustment ---
    # RSI: (0..100) -> (-1..+1) around 50, but invert: high RSI reduces score.
    rsi_unit = ((f["rsi"] - 50.0) / 50.0).clip(-1.0, 1.0)
    # price_z: high -> overbought -> reduce score
    pz_unit = (f["price_z"] / 3.0).clip(-1.0, 1.0)
    reversion_raw = 0.6 * rsi_unit + 0.4 * pz_unit
    reversion_adj = -reversion_raw * float(reversion_adj_max)

    out = pd.DataFrame(index=f.index)
    out["trend_score"] = trend_score
    out["reversion_adj"] = reversion_adj
    out["rsi_unit"] = rsi_unit
    out["pricez_unit"] = pz_unit
    out["trend_raw"] = trend_raw
    return out


def assemble_final_score(
    *,
    components: pd.DataFrame,
    risk_off_prob: pd.Series,
    risk_penalty_max: float,
    neutral_shift: float,
) -> pd.DataFrame:
    p = risk_off_prob.reindex(components.index)
    # If the regime model isn't available yet (insufficient history),
    # treat it as "no extra penalty" but keep the probability column as-is.
    p_for_penalty = p.fillna(0.0)
    risk_penalty = p_for_penalty * float(risk_penalty_max)

    score = (
        components["trend_score"]
        + float(neutral_shift)
        + components["reversion_adj"]
        - risk_penalty
    )
    score = clamp(score, 0.0, 100.0)

    out = components.copy()
    out["risk_off_prob"] = p
    out["risk_penalty"] = risk_penalty
    out["score"] = score
    out["label"] = score.apply(lambda x: score_to_label(float(x)) if pd.notna(x) else "insufficient_data")
    return out

