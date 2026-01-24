from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import (
    drawdown_from_rolling_high,
    ema,
    log_returns,
    realized_vol,
    rsi_wilder,
    zscore,
)


def equity_features(price: pd.Series, *, ema_fast: int, ema_slow: int, rsi: int, dd_window: int,
                   vol_windows: list[int], z_window: int, price_z_window: int) -> pd.DataFrame:
    """Per-index features used for trend/mean-reversion/stress scoring."""
    px = price.astype("float64")
    lr = log_returns(px)

    ema_fast_s = ema(px, ema_fast)
    ema_slow_s = ema(px, ema_slow)

    out = pd.DataFrame(index=px.index)
    out["px"] = px
    out["lr"] = lr

    out[f"ema{ema_fast}"] = ema_fast_s
    out[f"ema{ema_slow}"] = ema_slow_s
    out["ema_ratio"] = ema_fast_s / ema_slow_s - 1.0

    # Approx trend slope: percent change in slow EMA over 60 trading days (annualized-ish)
    slope_days = 60
    out["ema_slope"] = (ema_slow_s / ema_slow_s.shift(slope_days) - 1.0) * (252.0 / slope_days)

    out["d200"] = px / ema_slow_s - 1.0
    out["rsi"] = rsi_wilder(px, period=rsi)
    out["price_z"] = zscore(px, window=price_z_window)

    out["dd"] = drawdown_from_rolling_high(px, window=dd_window)

    for w in vol_windows:
        out[f"vol{w}"] = realized_vol(px, window=w, annualize=True)

    # --- Fast move / impulse features (sensitive to rapid selloffs and rebounds) ---
    out["mom5"] = lr.rolling(5).sum()
    out["mom20"] = lr.rolling(20).sum()
    if "vol20" in out.columns:
        daily_sigma = out["vol20"] / np.sqrt(252.0)
        mom5_sigma = daily_sigma * np.sqrt(5.0)
        out["mom5_vs_sigma"] = out["mom5"] / mom5_sigma.replace(0.0, np.nan)
    else:
        out["mom5_vs_sigma"] = np.nan

    # Standardized versions for scoring
    out["ema_slope_z"] = zscore(out["ema_slope"], window=z_window)
    out["ema_ratio_z"] = zscore(out["ema_ratio"], window=z_window)
    out["d200_z"] = zscore(out["d200"], window=z_window)
    out["dd_z"] = zscore(out["dd"], window=z_window)
    out["mom20_z"] = zscore(out["mom20"], window=z_window)
    out["mom5_vs_sigma_z"] = zscore(out["mom5_vs_sigma"], window=z_window)

    return out


def cross_asset_features(aligned_prices: pd.DataFrame, *, z_window: int) -> pd.DataFrame:
    """
    Cross-asset risk-off features (shared):
    expects columns like: eq_nifty50, ro_gold, ro_silver, ro_usdinr, ro_vix, y_us10y, y_us3m, etc.
    """
    df = aligned_prices.copy()

    def _z(x: pd.Series) -> pd.Series:
        return zscore(x.astype("float64"), window=z_window)

    out = pd.DataFrame(index=df.index)

    # Guarded fetches (series may be missing depending on ticker availability)
    px_nifty = df.get("eq_nifty50")
    px_midcap = df.get("eq_midcap100")
    px_smallcap = df.get("eq_smallcap100")
    gold = df.get("ro_gold")
    silver = df.get("ro_silver")
    usdinr = df.get("ro_usdinr")
    vix = df.get("ro_vix")
    us10y = df.get("y_us10y")
    us3m = df.get("y_us3m")

    # --- Intra-equity relative strength (divergence inside equities) ---
    if px_nifty is not None and px_midcap is not None:
        out["midcap_vs_nifty"] = np.log(px_midcap.astype("float64")) - np.log(px_nifty.astype("float64"))
        out["midcap_vs_nifty_z"] = _z(out["midcap_vs_nifty"])
        out["midcap_vs_nifty_mom60_z"] = _z(out["midcap_vs_nifty"].diff(60))

    if px_nifty is not None and px_smallcap is not None:
        out["smallcap_vs_nifty"] = np.log(px_smallcap.astype("float64")) - np.log(px_nifty.astype("float64"))
        out["smallcap_vs_nifty_z"] = _z(out["smallcap_vs_nifty"])
        out["smallcap_vs_nifty_mom60_z"] = _z(out["smallcap_vs_nifty"].diff(60))

    if px_nifty is not None and gold is not None:
        out["gold_vs_nifty"] = np.log(gold) - np.log(px_nifty)
        out["gold_vs_nifty_z"] = _z(out["gold_vs_nifty"])
        # Divergence: gold outperforming over medium term
        out["gold_vs_nifty_mom60_z"] = _z(out["gold_vs_nifty"].diff(60))

    if gold is not None and silver is not None:
        out["silver_vs_gold"] = np.log(silver) - np.log(gold)
        out["silver_vs_gold_z"] = _z(out["silver_vs_gold"])
        out["silver_vs_gold_mom60_z"] = _z(out["silver_vs_gold"].diff(60))

    if usdinr is not None:
        lfx = np.log(usdinr.astype("float64"))
        out["usdinr_mom20"] = lfx.diff(20)
        out["usdinr_vol20"] = lfx.diff().rolling(20).std() * np.sqrt(252.0)
        out["usdinr_mom20_z"] = _z(out["usdinr_mom20"])
        out["usdinr_vol20_z"] = _z(out["usdinr_vol20"])
        out["usdinr_mom60_z"] = _z(lfx.diff(60))

    if vix is not None:
        out["vix_level_z"] = _z(vix.astype("float64"))
        out["vix_chg20_z"] = _z(vix.astype("float64").pct_change(20))
        out["vix_chg60_z"] = _z(vix.astype("float64").pct_change(60))

    if us10y is not None:
        out["us10y_level_z"] = _z(us10y.astype("float64"))
        out["us10y_chg20_z"] = _z(us10y.astype("float64").diff(20))
        out["us10y_chg60_z"] = _z(us10y.astype("float64").diff(60))

    if us10y is not None and us3m is not None:
        out["us_curve_slope"] = us10y.astype("float64") - us3m.astype("float64")
        out["us_curve_slope_z"] = _z(out["us_curve_slope"])

    # --- Rolling correlations vs Nifty returns (regime/divergence signal) ---
    if px_nifty is not None:
        nifty_lr = log_returns(px_nifty.astype("float64"))
        if gold is not None:
            out["corr60_nifty_gold"] = nifty_lr.rolling(60).corr(log_returns(gold.astype("float64")))
            out["corr60_nifty_gold_z"] = _z(out["corr60_nifty_gold"])
        if usdinr is not None:
            out["corr60_nifty_usdinr"] = nifty_lr.rolling(60).corr(log_returns(usdinr.astype("float64")))
            out["corr60_nifty_usdinr_z"] = _z(out["corr60_nifty_usdinr"])
        if vix is not None:
            vix_ret = vix.astype("float64").pct_change()
            out["corr60_nifty_vix"] = nifty_lr.rolling(60).corr(vix_ret)
            out["corr60_nifty_vix_z"] = _z(out["corr60_nifty_vix"])
        if us10y is not None:
            ychg = us10y.astype("float64").diff()
            out["corr60_nifty_us10y"] = nifty_lr.rolling(60).corr(ychg)
            out["corr60_nifty_us10y_z"] = _z(out["corr60_nifty_us10y"])

    # --- Risk-off composite (divergence “breadth” style feature) ---
    # Sign convention: higher means more risk-off.
    riskoff_terms: list[pd.Series] = []
    for col in ["gold_vs_nifty_z", "gold_vs_nifty_mom60_z", "usdinr_mom20_z", "usdinr_mom60_z", "vix_chg20_z", "vix_chg60_z", "us10y_chg20_z", "us10y_chg60_z"]:
        if col in out.columns:
            riskoff_terms.append(out[col])
    if "us_curve_slope_z" in out.columns:
        # Curve inversion (negative slope) is risk-off -> flip sign
        riskoff_terms.append(-out["us_curve_slope_z"])
    if "smallcap_vs_nifty_mom60_z" in out.columns:
        # Smallcaps lagging Nifty tends to be risk-off -> flip sign
        riskoff_terms.append(-out["smallcap_vs_nifty_mom60_z"])
    if "midcap_vs_nifty_mom60_z" in out.columns:
        riskoff_terms.append(-out["midcap_vs_nifty_mom60_z"])

    if riskoff_terms:
        out["riskoff_composite"] = pd.concat(riskoff_terms, axis=1).mean(axis=1)
    else:
        out["riskoff_composite"] = np.nan

    return out

