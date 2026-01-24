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
    impulse_adj_max: float,
) -> pd.DataFrame:
    """
    Build TrendScore (0..trend_score_max) and ReversionAdjustment (-reversion_adj_max..+reversion_adj_max)
    from per-index equity features. Uses bounded transforms to keep outputs stable.
    """
    f = equity_feat.copy()

    # --- TrendScore ---
    # Raw trend is a mix of z-scored EMA slope and EMA ratio, plus being above/below EMA200 (d200_z),
    # and penalize large drawdown.
    # Add a medium-term momentum term to react faster after sharp regime changes.
    trend_raw = (
        0.35 * f["ema_slope_z"]
        + 0.20 * f["ema_ratio_z"]
        + 0.15 * f["d200_z"]
        - 0.15 * f["dd_z"]
        + 0.15 * f["mom20_z"]
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

    # --- ImpulseAdjustment (fast selloff / fast buying sensitivity) ---
    impulse_raw = 0.6 * f["mom5_vs_sigma_z"] + 0.4 * f["mom20_z"]
    impulse_unit = np.tanh(impulse_raw / 2.0)  # [-1, +1]
    impulse_adj = impulse_unit * float(impulse_adj_max)

    out = pd.DataFrame(index=f.index)
    out["trend_score"] = trend_score
    out["reversion_adj"] = reversion_adj
    out["impulse_adj"] = impulse_adj
    out["rsi_unit"] = rsi_unit
    out["pricez_unit"] = pz_unit
    out["trend_raw"] = trend_raw
    out["impulse_raw"] = impulse_raw
    return out


def assemble_final_score(
    *,
    components: pd.DataFrame,
    risk_off_prob: pd.Series,
    riskoff_composite: pd.Series | None = None,
    risk_penalty_max: float,
    neutral_shift: float,
) -> pd.DataFrame:
    p = risk_off_prob.reindex(components.index)
    # If the regime model isn't available yet (insufficient history),
    # treat it as "no extra penalty" but keep the probability column as-is.
    p_for_penalty = p.fillna(0.0)
    risk_penalty = p_for_penalty * float(risk_penalty_max)

    # Risk context used to dampen "oversold bounce" effects during strong risk-off backdrops.
    # This makes March-like crash regimes read more bearish, while allowing April rebounds
    # to recover faster once risk-off composite eases.
    risk_context = p_for_penalty.copy()
    if riskoff_composite is not None:
        roc = riskoff_composite.reindex(components.index)
        # Map roughly: <=0.25 -> 0, >=1.25 -> 1
        rc2 = ((roc - 0.25) / 1.0).clip(lower=0.0, upper=1.0).fillna(0.0)
        risk_context = np.maximum(risk_context, rc2)

    # Damp positive (bullish) adjustments when risk context is high.
    rev = components["reversion_adj"]
    rev_eff = rev.clip(lower=0.0) * (1.0 - 0.70 * risk_context) + rev.clip(upper=0.0)

    imp = components["impulse_adj"]
    imp_eff = imp.clip(lower=0.0) * (1.0 - 0.50 * risk_context) + imp.clip(upper=0.0)

    score = (
        components["trend_score"]
        + float(neutral_shift)
        + rev_eff
        + imp_eff
        - risk_penalty
    )
    score = clamp(score, 0.0, 100.0)

    out = components.copy()
    out["risk_off_prob"] = p
    out["risk_context"] = risk_context
    out["reversion_adj_eff"] = rev_eff
    out["impulse_adj_eff"] = imp_eff
    out["risk_penalty"] = risk_penalty
    out["score"] = score
    out["label"] = score.apply(lambda x: score_to_label(float(x)) if pd.notna(x) else "insufficient_data")
    return out

