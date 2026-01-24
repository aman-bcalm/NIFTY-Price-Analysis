from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .util import ensure_dir, safe_filename


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    ticker: str


def _parse_date(x: str | None) -> str | None:
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    # Let yfinance parse ISO-ish strings; validate a bit.
    datetime.fromisoformat(x)
    return x


def download_daily_adj_close(
    tickers: Iterable[str],
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    df = yf.download(
        tickers=list(tickers),
        start=_parse_date(start),
        end=_parse_date(end),
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned from data source.")

    # yfinance returns either:
    # - columns: [Open, High, Low, Close, Volume] for single ticker
    # - MultiIndex columns: (field, ticker) for multiple
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
    else:
        close = df[["Close"]].copy()
        close.columns = [tickers.__iter__().__next__()]  # best-effort

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close = close.dropna(how="all")
    return close


def cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{safe_filename(ticker)}.csv"


def load_or_download_series(
    ticker: str,
    cache_dir: str | Path,
    start: str | None,
    end: str | None,
    refresh: bool = False,
) -> pd.Series:
    cache_dir_p = ensure_dir(cache_dir)
    # yfinance maintains a timezone cache in a SQLite DB under the user profile by default.
    # In restricted/sandboxed environments this can fail, so we pin it to our project cache dir.
    # This is safe for local use too.
    try:
        yf.set_tz_cache_location(str(cache_dir_p))
    except Exception:
        # If the API changes, we can still run without forcing the location.
        pass
    p = cache_path(cache_dir_p, ticker)

    if p.exists() and not refresh:
        df = pd.read_csv(p, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        s = df["Close"].astype("float64")
        return s

    close = download_daily_adj_close([ticker], start=start, end=end)
    s = close.iloc[:, 0].rename("Close")
    out = s.to_frame()
    out.index.name = "Date"
    out.to_csv(p, index=True)
    return s


def align_series(
    series_map: dict[str, pd.Series],
    max_forward_fill_days: int = 3,
) -> pd.DataFrame:
    df = pd.DataFrame({k: v for k, v in series_map.items()})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # Limited forward fill across holidays/weekends; then leave remaining missing.
    if max_forward_fill_days and max_forward_fill_days > 0:
        df = df.ffill(limit=max_forward_fill_days)
    return df

