"""
Unit tests for the overnight (ON) / intraday (ID) decomposition.

We build small synthetic OHLC series with KNOWN returns, so the expected
values can be computed by hand. This protects the core math against
regressions (e.g. an accidental shift, a sign flip, or a wrong cumulation).

Run:
    pip install pytest
    pytest -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the script importable when tests live in tests/ and the module is at the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sp500_overnight_intraday import compute_on_id, summarize_ticker  # noqa: E402


def _make_ohlc(on, id_, n_days, close0=100.0):
    """Build a synthetic OHLC frame where every day has a fixed ON and ID return.

    Open_t  = Close_{t-1} * (1 + on)
    Close_t = Open_t      * (1 + id_)
    The first row only carries Close0 (no previous close -> dropped by compute_on_id).
    """
    idx = pd.date_range("2020-01-01", periods=n_days + 1, freq="B")
    opens, highs, lows, closes = [], [], [], []
    prev_close = close0
    for i in range(n_days + 1):
        if i == 0:
            o = c = close0
        else:
            o = prev_close * (1 + on)
            c = o * (1 + id_)
        opens.append(o)
        closes.append(c)
        highs.append(max(o, c))
        lows.append(min(o, c))
        prev_close = c
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000_000] * (n_days + 1),
        },
        index=idx,
    )


def test_on_id_basic_values():
    """ON and ID must match the hand-computed gap/session returns."""
    df = _make_ohlc(on=0.10, id_=0.05, n_days=3)
    d = compute_on_id(df)
    # First row is dropped (no previous close).
    assert len(d) == 3
    np.testing.assert_allclose(d["ON"].values, 0.10, rtol=1e-12)
    np.testing.assert_allclose(d["ID"].values, 0.05, rtol=1e-12)


def test_log_additivity():
    """log(1+ON) + log(1+ID) must equal log(1 + close-to-close)."""
    df = _make_ohlc(on=0.02, id_=-0.01, n_days=10)
    d = compute_on_id(df)
    c2c = d["Close"] / d["Close"].shift(1) - 1.0
    lhs = (d["logON"] + d["logID"]).iloc[1:]
    rhs = np.log1p(c2c).iloc[1:]
    np.testing.assert_allclose(lhs.values, rhs.values, rtol=1e-12, atol=1e-12)


def test_summarize_cumulative():
    """Cumulative ON/ID must equal the geometric product (1+r)^n - 1."""
    on, id_, n = 0.001, 0.0005, 60
    df = _make_ohlc(on=on, id_=id_, n_days=n)
    d = compute_on_id(df)
    s = summarize_ticker("TEST", d)
    assert s is not None
    assert s["n_days"] == n
    np.testing.assert_allclose(s["cum_ON"], (1 + on) ** n - 1, rtol=1e-9)
    np.testing.assert_allclose(s["cum_ID"], (1 + id_) ** n - 1, rtol=1e-9)
    # ON > ID here, so the log difference must be positive.
    assert s["diff_log"] > 0
    np.testing.assert_allclose(
        s["diff_log"], n * (np.log1p(on) - np.log1p(id_)), rtol=1e-9
    )


def test_summarize_too_short_returns_none():
    """Series shorter than 20 valid rows should be skipped (returns None)."""
    df = _make_ohlc(on=0.01, id_=0.01, n_days=5)
    d = compute_on_id(df)
    assert summarize_ticker("TEST", d) is None


def test_annualization_sign_and_consistency():
    """Annualized ON should beat ID when daily ON > daily ID; diff matches formula."""
    on, id_, n = 0.0008, 0.0002, 80
    df = _make_ohlc(on=on, id_=id_, n_days=n)
    s = summarize_ticker("TEST", compute_on_id(df))
    assert s["ann_ON_pct"] > s["ann_ID_pct"]
    np.testing.assert_allclose(
        s["ann_diff_log"], (np.log1p(on) - np.log1p(id_)) * 252, rtol=1e-9
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", __file__]))
