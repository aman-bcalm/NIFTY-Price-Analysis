from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import clamp, rsi_wilder, zscore


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


def safe_haven_score_to_label(score: float) -> str:
    """Label for safe-haven *stretch* (mean-reversion) score."""
    if score < 20:
        return "oversold"
    if score < 40:
        return "weak"
    if score < 60:
        return "neutral"
    if score < 80:
        return "strong"
    return "overbought"


def compute_safe_haven_stretch(
    price: pd.Series,
    *,
    rsi_period: int = 14,
    z_windows: tuple[int, int] = (60, 252),
    weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> pd.DataFrame:
    """
    Compute mean-reversion stretch score for a single asset series.

    Returns a DataFrame with:
    - safe_score (0..100): higher = more overbought (stretched up), lower = more oversold
    - safe_label: oversold/weak/neutral/strong/overbought
    - rsi, price_z_short, price_z_long (diagnostics)
    """
    px = price.astype("float64")
    out = pd.DataFrame(index=px.index)
    out["rsi"] = rsi_wilder(px, period=rsi_period)

    z1, z2 = z_windows
    # Use log-price for z-scores to better reflect long-run multiplicative drift.
    log_px = np.log(px.replace(0.0, np.nan))
    out["price_z_short"] = zscore(log_px, window=int(z1))
    out["price_z_long"] = zscore(log_px, window=int(z2))

    # Normalize to roughly [-1, +1]
    rsi_unit = ((out["rsi"] - 50.0) / 50.0).clip(-1.0, 1.0)
    z_short_unit = (out["price_z_short"] / 3.0).clip(-1.0, 1.0)
    z_long_unit = (out["price_z_long"] / 3.0).clip(-1.0, 1.0)

    w_rsi, w_zs, w_zl = weights
    # Weighted average but allow missing components (e.g., long-window z early in history).
    units = pd.concat([rsi_unit.rename("rsi"), z_short_unit.rename("z_short"), z_long_unit.rename("z_long")], axis=1)
    w = np.array([w_rsi, w_zs, w_zl], dtype="float64")
    valid_w = units.notna().astype("float64").mul(w, axis=1)
    denom = valid_w.sum(axis=1)
    stretch = (units.fillna(0.0).mul(w, axis=1)).sum(axis=1) / denom.replace(0.0, np.nan)

    # Map to 0..100
    # Linear mapping makes 'overbought' reachable for strong multi-signal stretches.
    score = ((stretch + 1.0) / 2.0) * 100.0
    score = clamp(score, 0.0, 100.0)

    out["safe_score"] = score
    out["safe_label"] = out["safe_score"].apply(
        lambda x: safe_haven_score_to_label(float(x)) if pd.notna(x) else "insufficient_data"
    )
    return out


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

