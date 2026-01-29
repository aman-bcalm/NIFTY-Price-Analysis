from __future__ import annotations

"""
One-off helper script to inspect which series we need from Yahoo Finance
and up to which date data is actually available in the current cache.

Usage (from project root, inside .venv):

    python sandbox/data_availability_probe.py

This DOES NOT change any cached files; it just reuses whatever
`trend_analyzer.data_loader` has already downloaded.
"""

from pathlib import Path
from typing import Any, Dict, List
import sys

import pandas as pd

# Ensure the project root (parent of this sandbox folder) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trend_analyzer.config import AppConfig
from trend_analyzer.data_loader import load_or_download_series


def _as_candidates(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return []


def _try_first_candidate(
    *,
    logical_name: str,
    candidates: List[str],
    cache_dir: Path,
    start: str | None,
    end: str | None,
    refresh: bool,
) -> tuple[pd.Series | None, str | None, Exception | None]:
    """
    Try each candidate ticker until one returns a non-empty series.
    Returns (series_or_none, chosen_ticker_or_none, last_error_or_none).
    """
    last_err: Exception | None = None
    for t in candidates:
        try:
            s = load_or_download_series(
                ticker=t,
                cache_dir=cache_dir,
                start=start,
                end=end,
                refresh=refresh,
            ).rename(logical_name)
            if s is not None and not s.dropna().empty:
                return s, t, None
        except Exception as e:  # noqa: BLE001 - diagnostic helper
            last_err = e
            continue
    return None, None, last_err


def main() -> int:
    cfg = AppConfig.load("config.yaml")

    cache_dir = Path(cfg.get("data", "cache_dir", default="data/cache"))
    start = cfg.get("data", "start", default=None)
    end = cfg.get("data", "end", default=None)

    eq_cfg: Dict[str, Any] = cfg.get("tickers", "equities", default={}) or {}
    risk_cfg: Dict[str, Any] = cfg.get("tickers", "risk_off", default={}) or {}
    yld_cfg: Dict[str, Any] = cfg.get("tickers", "yields", default={}) or {}

    today = pd.Timestamp.today().normalize()
    print(f"Today (local system date): {today.date().isoformat()}")
    print(f"Data start from config:   {start!r}")
    print(f"Data end from config:     {end!r}  (None = up to latest available)")
    print()

    logical_to_candidates: Dict[str, List[str]] = {}

    # Equities (lists of candidates)
    logical_to_candidates["eq_nifty50"] = _as_candidates(eq_cfg.get("nifty50"))
    logical_to_candidates["eq_midcap100"] = _as_candidates(eq_cfg.get("midcap100"))
    logical_to_candidates["eq_smallcap100"] = _as_candidates(eq_cfg.get("smallcap100"))

    # Risk-off (single tickers but we still treat as 1-candidate lists)
    for k, v in (risk_cfg or {}).items():
        logical_to_candidates[f"ro_{k}"] = _as_candidates(v)

    # Yields
    for k, v in (yld_cfg or {}).items():
        logical_to_candidates[f"y_{k}"] = _as_candidates(v)

    print("Logical series and their configured Yahoo tickers:")
    for lname, cands in logical_to_candidates.items():
        print(f"  {lname}: {cands}")
    print()

    rows: list[dict[str, Any]] = []

    for lname, cands in logical_to_candidates.items():
        if not cands:
            rows.append(
                {
                    "logical_name": lname,
                    "chosen_ticker": "",
                    "status": "NO_CANDIDATES",
                    "first_date": "",
                    "last_date": "",
                    "days_lag_vs_today": "",
                    "n_points": "",
                    "note": "",
                }
            )
            continue

        s, chosen, err = _try_first_candidate(
            logical_name=lname,
            candidates=cands,
            cache_dir=cache_dir,
            start=start,
            end=end,
            refresh=False,  # IMPORTANT: only use existing cache; do not hit network here.
        )

        if s is None or s.dropna().empty:
            rows.append(
                {
                    "logical_name": lname,
                    "chosen_ticker": chosen or "",
                    "status": "EMPTY_OR_MISSING",
                    "first_date": "",
                    "last_date": "",
                    "days_lag_vs_today": "",
                    "n_points": "",
                    "note": str(err) if err is not None else "no non-empty series in cache",
                }
            )
            continue

        s = s.dropna()
        first = s.index.min()
        last = s.index.max()
        lag = (today - last.normalize()).days

        rows.append(
            {
                "logical_name": lname,
                "chosen_ticker": chosen or "",
                "status": "OK",
                "first_date": first.date().isoformat() if isinstance(first, pd.Timestamp) else str(first),
                "last_date": last.date().isoformat() if isinstance(last, pd.Timestamp) else str(last),
                "days_lag_vs_today": lag,
                "n_points": int(s.shape[0]),
                "note": "",
            }
        )

    df = pd.DataFrame(rows).sort_values(["logical_name"])
    print("DATA AVAILABILITY (from current cache):")
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

