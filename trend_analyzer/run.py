from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .data_loader import align_series, load_or_download_series
from .features import cross_asset_features, equity_features
from .indicators import forward_max_drawdown, forward_return
from .regime_model import RiskModelConfig, walkforward_logistic_probabilities
from .scoring import assemble_final_score, compute_score_components
from .util import ensure_dir


def _as_candidates(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return []


def _try_load_candidates(
    *,
    name: str,
    candidates: list[str],
    cache_dir: Path,
    start: str | None,
    end: str | None,
    refresh: bool,
    required: bool = False,
) -> pd.Series | None:
    if not candidates:
        if required:
            raise RuntimeError(f"Missing required series '{name}': no candidates configured.")
        return None
    last_err: Exception | None = None
    for t in candidates:
        try:
            s = load_or_download_series(
                ticker=t,
                cache_dir=cache_dir,
                start=start,
                end=end,
                refresh=refresh,
            ).rename(name)
            if s is not None and not s.dropna().empty:
                return s
        except Exception as e:  # noqa: BLE001 - CLI tool, best-effort fallbacks
            last_err = e
            continue
    if required:
        raise RuntimeError(f"Failed to load required series '{name}' from candidates={candidates}") from last_err
    print(f"Warning: could not load series '{name}' from candidates={candidates}. Skipping.")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trend Analyzer (Nifty) - local runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached market data")
    args = parser.parse_args(argv)

    cfg = AppConfig.load(args.config)

    cache_dir = Path(cfg.get("data", "cache_dir", default="data/cache"))
    start = cfg.get("data", "start", default=None)
    end = cfg.get("data", "end", default=None)
    max_ff = int(cfg.get("data", "max_forward_fill_days", default=3))

    eq = cfg.get("tickers", "equities", default={}) or {}
    risk = cfg.get("tickers", "risk_off", default={}) or {}
    yields = cfg.get("tickers", "yields", default={}) or {}

    series_map: dict[str, pd.Series] = {}
    # Equities (allow list of fallback tickers)
    s = _try_load_candidates(
        name="eq_nifty50",
        candidates=_as_candidates(eq.get("nifty50")),
        cache_dir=cache_dir,
        start=start,
        end=end,
        refresh=args.refresh,
        required=True,
    )
    if s is not None:
        series_map["eq_nifty50"] = s

    for key in ["midcap100", "smallcap100"]:
        s = _try_load_candidates(
            name=f"eq_{key}",
            candidates=_as_candidates(eq.get(key)),
            cache_dir=cache_dir,
            start=start,
            end=end,
            refresh=args.refresh,
            required=False,
        )
        if s is not None:
            series_map[f"eq_{key}"] = s

    # Risk-off assets (single tickers)
    for k, v in risk.items():
        s = _try_load_candidates(
            name=f"ro_{k}",
            candidates=_as_candidates(v),
            cache_dir=cache_dir,
            start=start,
            end=end,
            refresh=args.refresh,
            required=False,
        )
        if s is not None:
            series_map[f"ro_{k}"] = s

    # Yields (US)
    for k in ["us10y", "us3m"]:
        s = _try_load_candidates(
            name=f"y_{k}",
            candidates=_as_candidates(yields.get(k)),
            cache_dir=cache_dir,
            start=start,
            end=end,
            refresh=args.refresh,
            required=False,
        )
        if s is not None:
            series_map[f"y_{k}"] = s

    # Best-effort India 10Y yield selection (no-auth sources can be unreliable).
    india10y_candidates = _as_candidates(yields.get("india10y_candidates"))
    india_bond_proxy = yields.get("india_bond_proxy")
    s = _try_load_candidates(
        name="y_india10y",
        candidates=india10y_candidates,
        cache_dir=cache_dir,
        start=start,
        end=end,
        refresh=args.refresh,
        required=False,
    )
    if s is not None:
        series_map["y_india10y"] = s
    else:
        s = _try_load_candidates(
            name="y_india_bond_proxy",
            candidates=_as_candidates(india_bond_proxy),
            cache_dir=cache_dir,
            start=start,
            end=end,
            refresh=args.refresh,
            required=False,
        )
        if s is not None:
            series_map["y_india_bond_proxy"] = s

    aligned = align_series(series_map, max_forward_fill_days=max_ff)

    out_dir = ensure_dir(args.out_dir)
    (out_dir / "aligned_prices.csv").write_text(aligned.to_csv(index=True), encoding="utf-8")

    # --- Feature config ---
    feat_cfg = cfg.get("features", default={}) or {}
    ema_fast = int(feat_cfg.get("ema_fast", 50))
    ema_slow = int(feat_cfg.get("ema_slow", 200))
    rsi = int(feat_cfg.get("rsi", 14))
    z_window = int(feat_cfg.get("z_window", 252))
    price_z_window = int(feat_cfg.get("price_z_window", 60))
    vol_windows = list(feat_cfg.get("vol_windows", [20, 60]))
    dd_window = int(feat_cfg.get("dd_window", 252))

    # --- Per-index equity features ---
    eq_prices = {
        "nifty50": aligned.get("eq_nifty50"),
        "midcap100": aligned.get("eq_midcap100"),
        "smallcap100": aligned.get("eq_smallcap100"),
    }
    eq_feats: dict[str, pd.DataFrame] = {}
    for k, s in eq_prices.items():
        if s is None:
            continue
        eq_feats[k] = equity_features(
            s,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            dd_window=dd_window,
            vol_windows=vol_windows,
            z_window=z_window,
            price_z_window=price_z_window,
        )

    # --- Cross-asset features (shared) ---
    xasset = cross_asset_features(aligned, z_window=z_window)

    # --- Risk-off model (trained on Nifty 50 anchor) ---
    rm_cfg_raw = cfg.get("risk_model", default={}) or {}
    rm_cfg = RiskModelConfig(
        horizon_days=int(rm_cfg_raw.get("horizon_days", 21)),
        fwd_return_threshold=float(rm_cfg_raw.get("fwd_return_threshold", -0.05)),
        fwd_max_drawdown_threshold=float(rm_cfg_raw.get("fwd_max_drawdown_threshold", -0.07)),
        min_train_years=int(rm_cfg_raw.get("min_train_years", 5)),
        retrain_frequency=str(rm_cfg_raw.get("retrain_frequency", "M")),
        regularization_C=float(rm_cfg_raw.get("regularization_C", 1.0)),
    )

    nifty_feat = eq_feats.get("nifty50")
    if nifty_feat is None:
        raise RuntimeError("Missing Nifty 50 data; cannot train risk regime model.")

    # Label definition
    nifty_px = nifty_feat["px"]
    fwd_ret = forward_return(nifty_px, horizon_days=rm_cfg.horizon_days)
    fwd_mdd = forward_max_drawdown(nifty_px, horizon_days=rm_cfg.horizon_days)
    risk_off = (fwd_ret < rm_cfg.fwd_return_threshold) | (fwd_mdd < rm_cfg.fwd_max_drawdown_threshold)
    risk_off = risk_off.astype("float64").where(fwd_ret.notna() & fwd_mdd.notna())

    # Model features = Nifty equity context + cross-asset context
    model_X = pd.concat(
        [
            nifty_feat[
                [
                    "ema_slope_z",
                    "ema_ratio_z",
                    "d200_z",
                    "dd_z",
                    "price_z",
                    "rsi",
                    "vol20",
                    "vol60",
                ]
            ],
            xasset,
        ],
        axis=1,
    )
    model_X = model_X.replace([pd.NA, pd.NaT], float("nan")).dropna(how="all")

    # Extra explicit divergence interaction: equities trending up while risk-off composite rises.
    if "riskoff_composite" in model_X.columns:
        trend_up = ((nifty_feat["ema_slope_z"] > 0) & (nifty_feat["ema_ratio_z"] > 0)).astype("float64")
        model_X["divergence_equity_up_riskoff"] = trend_up.reindex(model_X.index) * model_X["riskoff_composite"]

    y = risk_off.reindex(model_X.index)
    probs = walkforward_logistic_probabilities(X=model_X, y=y, cfg=rm_cfg)

    # --- Scoring ---
    sc_cfg = cfg.get("scoring", default={}) or {}
    trend_score_max = float(sc_cfg.get("trend_score_max", 60.0))
    risk_penalty_max = float(sc_cfg.get("risk_penalty_max", 20.0))
    reversion_adj_max = float(sc_cfg.get("reversion_adj_max", 20.0))
    impulse_adj_max = float(sc_cfg.get("impulse_adj_max", 20.0))
    neutral_shift = float(sc_cfg.get("neutral_shift", 20.0))

    scores_long: list[pd.DataFrame] = []

    # Divergence outputs (more granular, for transparency)
    riskoff_comp = xasset.get("riskoff_composite")
    if riskoff_comp is not None:
        nifty_trend_up = (nifty_feat["ema_slope_z"] > 0) & (nifty_feat["ema_ratio_z"] > 0)

        # Short-term direction filters so we don't label a down day as a "divergence".
        # Use log-return vs recent daily sigma (vol20 is annualized).
        lr1 = nifty_feat["lr"].reindex(riskoff_comp.index)
        daily_sigma = (nifty_feat["vol20"] / (252.0 ** 0.5)).reindex(riskoff_comp.index)
        equity_down = lr1 < 0
        equity_up = lr1 > 0
        heavy_down = (lr1 < (-2.0 * daily_sigma)) | (lr1 < -0.02)

        riskoff_high = riskoff_comp > 0.75
        riskoff_low = riskoff_comp < -0.75

        # Bear-market bounce detector:
        # risk-off composite stays high, but equities bounce after a big down day.
        lr_prev = lr1.shift(1)
        sigma_prev = daily_sigma.shift(1)
        prev_heavy_down = (lr_prev < (-2.0 * sigma_prev)) | (lr_prev < -0.02)

        # Categorical divergence state
        divergence_state = pd.Series("normal", index=riskoff_comp.index, dtype="object")
        # Priority order (more specific first):
        mask_crash_day = riskoff_high & equity_down & heavy_down
        mask_selloff = riskoff_high & equity_down & (~heavy_down)
        mask_bear_bounce = riskoff_high & equity_up & prev_heavy_down

        trend_up = nifty_trend_up.reindex(riskoff_comp.index)
        # True divergence = risk-off high while not down today, trend-up, and NOT just a bounce after a heavy down day.
        mask_divergence = riskoff_high & (~equity_down) & trend_up & (~mask_bear_bounce)
        mask_downtrend = riskoff_high & (~equity_down) & (~trend_up) & (~mask_bear_bounce)

        mask_selloff_without_riskoff = riskoff_low & heavy_down

        divergence_state[mask_crash_day] = "riskoff_crash_day"
        divergence_state[mask_selloff] = "riskoff_selloff"
        divergence_state[mask_bear_bounce] = "riskoff_bear_bounce"
        divergence_state[mask_divergence] = "divergence_trendup_riskoff"
        divergence_state[mask_downtrend] = "riskoff_downtrend"
        divergence_state[mask_selloff_without_riskoff] = "selloff_without_riskoff"

        # Keep a boolean flag for the specific divergence case only:
        divergence_flag = divergence_state.eq("divergence_trendup_riskoff")
    else:
        divergence_flag = None
        divergence_state = None

    for name, f in eq_feats.items():
        comps = compute_score_components(
            f,
            trend_score_max=trend_score_max,
            reversion_adj_max=reversion_adj_max,
            impulse_adj_max=impulse_adj_max,
        )
        scored = assemble_final_score(
            components=comps,
            risk_off_prob=probs,
            riskoff_composite=riskoff_comp if riskoff_comp is not None else None,
            risk_penalty_max=risk_penalty_max,
            neutral_shift=neutral_shift,
        )
        # Attach key regime/divergence context to each row (same for all indices).
        if riskoff_comp is not None:
            scored["riskoff_composite"] = riskoff_comp.reindex(scored.index)
        if divergence_state is not None:
            scored["divergence_state"] = divergence_state.reindex(scored.index).fillna("unknown")
        if divergence_flag is not None:
            scored["divergence_flag"] = divergence_flag.reindex(scored.index).fillna(False)
        else:
            scored["divergence_flag"] = False
        scored = scored.assign(index=name)

        # Keep existing column order stable (Excel column letters) by pushing new diagnostics to the far right.
        base_cols = [
            "trend_score",
            "reversion_adj",
            "rsi_unit",
            "pricez_unit",
            "trend_raw",
            "risk_off_prob",
            "risk_penalty",
            "score",
            "label",
            "riskoff_composite",
            "divergence_state",
            "divergence_flag",
            "index",
        ]
        extra_cols = [c for c in scored.columns if c not in base_cols]
        scored = scored[base_cols + extra_cols]
        scores_long.append(scored.reset_index(names="date"))

    scores = pd.concat(scores_long, ignore_index=True).sort_values(["date", "index"])
    scores_path = out_dir / "scores.csv"
    scores.to_csv(scores_path, index=False)

    # Optionally write per-index features
    out_cfg = cfg.get("output", default={}) or {}
    if bool(out_cfg.get("write_features", False)):
        for name, f in eq_feats.items():
            f.to_csv(out_dir / f"features_{name}.csv", index=True)
        xasset.to_csv(out_dir / "features_cross_asset.csv", index=True)
        model_X.to_csv(out_dir / "features_model_X.csv", index=True)

    print(f"Wrote {out_dir / 'aligned_prices.csv'} ({aligned.shape[0]} rows, {aligned.shape[1]} cols)")
    print(f"Wrote {scores_path} ({scores.shape[0]} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

