import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trend_analyzer.data_loader import align_series, load_or_download_series


def test_align_series_forward_fill_limit():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    # Missing in the middle
    a = pd.Series([1.0, None, None, 4.0, None, 6.0], index=idx)
    b = pd.Series([10.0, 11.0, None, None, None, 15.0], index=idx)

    df = align_series({"a": a, "b": b}, max_forward_fill_days=1)

    # a: only one-day forward fill should fill idx[1] but not idx[2]
    assert df.loc[idx[1], "a"] == 1.0
    assert pd.isna(df.loc[idx[2], "a"])

    # b: idx[2] filled from idx[1], idx[3] should remain NaN due to limit
    assert df.loc[idx[2], "b"] == 11.0
    assert pd.isna(df.loc[idx[3], "b"])


def test_load_or_download_series_incremental_fetch():
    """Test that incremental fetching works: cache + new data merged correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Create initial cache with data up to 2026-01-23
        cache_dates = pd.date_range("2026-01-20", "2026-01-23", freq="D")
        cache_data = pd.Series([100.0, 101.0, 102.0, 103.0], index=cache_dates, name="Close")
        cache_df = cache_data.to_frame()
        cache_df.index.name = "Date"
        cache_path = cache_dir / f"{ticker}.csv"
        cache_df.to_csv(cache_path, index=True)

        # Mock yfinance to return new data from 2026-01-24 to 2026-01-26
        new_dates = pd.date_range("2026-01-24", "2026-01-26", freq="D")
        new_data = pd.DataFrame(
            {"Close": [104.0, 105.0, 106.0]},
            index=new_dates,
        )

        with patch("trend_analyzer.data_loader.download_daily_adj_close", return_value=new_data):
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start=None,
                end=None,
                refresh=False,
            )

        # Should have merged cache (4 days) + new (3 days) = 7 days total
        assert len(result) == 7
        assert result.index.min() == pd.Timestamp("2026-01-20")
        assert result.index.max() == pd.Timestamp("2026-01-26")
        assert result.loc["2026-01-23"] == 103.0  # Last cached value
        assert result.loc["2026-01-24"] == 104.0  # First new value
        assert result.loc["2026-01-26"] == 106.0  # Last new value

        # Cache file should be updated with merged data
        updated_cache = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
        assert len(updated_cache) == 7
        assert updated_cache.index.max() == pd.Timestamp("2026-01-26")


def test_load_or_download_series_cache_only():
    """Test that when cache exists and covers requested range, no fetch happens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Create cache with data up to 2026-01-25
        cache_dates = pd.date_range("2026-01-20", "2026-01-25", freq="D")
        cache_data = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=cache_dates, name="Close")
        cache_df = cache_data.to_frame()
        cache_df.index.name = "Date"
        cache_path = cache_dir / f"{ticker}.csv"
        cache_df.to_csv(cache_path, index=True)

        # Request data up to 2026-01-23 (before cache end)
        with patch("trend_analyzer.data_loader.download_daily_adj_close") as mock_download:
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start=None,
                end="2026-01-23",
                refresh=False,
            )

        # Should return cache without calling download
        mock_download.assert_not_called()
        assert len(result) == 6  # All cached data
        assert result.index.max() == pd.Timestamp("2026-01-25")


def test_load_or_download_series_refresh_flag():
    """Test that refresh=True forces full re-download ignoring cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Create old cache
        cache_dates = pd.date_range("2026-01-20", "2026-01-23", freq="D")
        cache_data = pd.Series([100.0, 101.0, 102.0, 103.0], index=cache_dates, name="Close")
        cache_df = cache_data.to_frame()
        cache_df.index.name = "Date"
        cache_path = cache_dir / f"{ticker}.csv"
        cache_df.to_csv(cache_path, index=True)

        # Mock fresh download (different data)
        fresh_dates = pd.date_range("2026-01-20", "2026-01-27", freq="D")
        fresh_data = pd.DataFrame(
            {"Close": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0]},
            index=fresh_dates,
        )

        with patch("trend_analyzer.data_loader.download_daily_adj_close", return_value=fresh_data):
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start="2026-01-20",
                end="2026-01-27",
                refresh=True,
            )

        # Should have fresh data, not cached
        assert len(result) == 8
        assert result.loc["2026-01-20"] == 200.0  # Fresh value, not cached 100.0
        assert result.index.max() == pd.Timestamp("2026-01-27")


def test_load_or_download_series_no_cache_first_run():
    """Test that first run (no cache) downloads everything."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Mock download for first run
        dates = pd.date_range("2026-01-20", "2026-01-25", freq="D")
        data = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=dates,
        )

        with patch("trend_analyzer.data_loader.download_daily_adj_close", return_value=data):
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start="2026-01-20",
                end="2026-01-25",
                refresh=False,
            )

        # Should have downloaded data
        assert len(result) == 6
        assert result.index.min() == pd.Timestamp("2026-01-20")
        assert result.index.max() == pd.Timestamp("2026-01-25")

        # Cache should be created
        cache_path = cache_dir / f"{ticker}.csv"
        assert cache_path.exists()


def test_load_or_download_series_no_new_data_available():
    """Test that when yfinance returns no new data, cached data is returned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Create cache
        cache_dates = pd.date_range("2026-01-20", "2026-01-23", freq="D")
        cache_data = pd.Series([100.0, 101.0, 102.0, 103.0], index=cache_dates, name="Close")
        cache_df = cache_data.to_frame()
        cache_df.index.name = "Date"
        cache_path = cache_dir / f"{ticker}.csv"
        cache_df.to_csv(cache_path, index=True)

        # Mock yfinance returning empty (no new data available)
        with patch("trend_analyzer.data_loader.download_daily_adj_close", side_effect=RuntimeError("No data returned")):
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start=None,
                end=None,
                refresh=False,
            )

        # Should return cached data
        assert len(result) == 4
        assert result.index.max() == pd.Timestamp("2026-01-23")
        assert result.loc["2026-01-23"] == 103.0


def test_load_or_download_series_handles_overlap():
    """Test that if new data overlaps with cache, newer values win."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Cache has data up to 2026-01-23
        cache_dates = pd.date_range("2026-01-20", "2026-01-23", freq="D")
        cache_data = pd.Series([100.0, 101.0, 102.0, 103.0], index=cache_dates, name="Close")
        cache_df = cache_data.to_frame()
        cache_df.index.name = "Date"
        cache_path = cache_dir / f"{ticker}.csv"
        cache_df.to_csv(cache_path, index=True)

        # New data overlaps: starts from 2026-01-23 (same as cache end) with different value
        new_dates = pd.date_range("2026-01-23", "2026-01-25", freq="D")
        new_data = pd.DataFrame(
            {"Close": [999.0, 104.0, 105.0]},  # 2026-01-23 has different value
            index=new_dates,
        )

        with patch("trend_analyzer.data_loader.download_daily_adj_close", return_value=new_data):
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start=None,
                end=None,
                refresh=False,
            )

        # Overlapping date should use NEW value (keep="last" in merge)
        assert result.loc["2026-01-23"] == 999.0  # New value, not cached 103.0
        assert result.loc["2026-01-24"] == 104.0
        assert result.loc["2026-01-25"] == 105.0


def test_load_or_download_series_cache_up_to_today():
    """Test that when cache is already up to today, no fetch happens (end=None)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        ticker = "TEST"

        # Create cache with data up to today
        today = pd.Timestamp.today().normalize()
        cache_dates = pd.date_range(today - pd.Timedelta(days=5), today, freq="D")
        cache_data = pd.Series(range(100, 100 + len(cache_dates)), index=cache_dates, name="Close")
        cache_df = cache_data.to_frame()
        cache_df.index.name = "Date"
        cache_path = cache_dir / f"{ticker}.csv"
        cache_df.to_csv(cache_path, index=True)

        # Request data with end=None (should use today)
        with patch("trend_analyzer.data_loader.download_daily_adj_close") as mock_download:
            result = load_or_download_series(
                ticker=ticker,
                cache_dir=cache_dir,
                start=None,
                end=None,
                refresh=False,
            )

        # Should return cache without calling download (cache already covers today)
        mock_download.assert_not_called()
        assert len(result) == len(cache_dates)
        assert result.index.max() == today

