import pandas as pd

from trend_analyzer.data_loader import align_series


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

