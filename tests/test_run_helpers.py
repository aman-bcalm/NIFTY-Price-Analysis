import pandas as pd
import pytest

import trend_analyzer.run as runmod


def test_try_load_candidates_raises_when_required_and_empty():
    with pytest.raises(RuntimeError):
        runmod._try_load_candidates(
            name="eq_nifty50",
            candidates=[],
            cache_dir=None,  # should fail before using cache_dir
            start=None,
            end=None,
            refresh=False,
            required=True,
        )


def test_try_load_candidates_fallback(monkeypatch, tmp_path):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    good = pd.Series([1, 2, 3, 4, 5], index=idx, dtype="float64")

    calls = {"n": 0}

    def fake_loader(*, ticker, cache_dir, start, end, refresh):
        calls["n"] += 1
        if ticker == "BAD":
            raise RuntimeError("boom")
        return good

    monkeypatch.setattr(runmod, "load_or_download_series", fake_loader)

    out = runmod._try_load_candidates(
        name="x",
        candidates=["BAD", "GOOD"],
        cache_dir=tmp_path,
        start=None,
        end=None,
        refresh=False,
        required=True,
    )
    assert out is not None
    assert out.name == "x"
    assert calls["n"] == 2

