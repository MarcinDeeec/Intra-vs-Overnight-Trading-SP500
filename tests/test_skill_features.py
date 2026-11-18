"""Tests for the quick/--json modes, beta, yfinance layouts and the CLI.

Everything runs offline: yfinance and Wikipedia are replaced with synthetic data.
"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sp500_overnight_intraday as mod  # noqa: E402


def _series(n=300, seed=0, drift_on=0.0004, drift_id=0.0, noise=0.01):
    """Random-walk OHLCV with a chosen average overnight/intraday drift."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    on = drift_on + rng.normal(0, noise / 2, n)
    idr = drift_id + rng.normal(0, noise, n)
    opens, closes, prev = [], [], 100.0
    for i in range(n):
        o = prev * (1 + on[i])
        c = o * (1 + idr[i])
        opens.append(o)
        closes.append(c)
        prev = c
    o, c = np.array(opens), np.array(closes)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c), "Low": np.minimum(o, c),
                         "Close": c, "Volume": 1_000_000}, index=idx)


# ---------------------------------------------------------------- core maths
def test_beta_matches_known_value():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2022-01-03", periods=200)
    mkt = pd.Series(rng.normal(0.0005, 0.01, 200), index=idx)
    close = 100 * (1 + 1.5 * mkt).cumprod()
    df = pd.DataFrame({"Open": close.shift(1).fillna(100.0), "Close": close}, index=idx)
    s = mod.summarize_ticker("X", mod.compute_on_id(df), mkt)
    assert s["beta"] == pytest.approx(1.5, rel=1e-6)


def test_beta_is_nan_without_market_or_with_short_overlap():
    d = mod.compute_on_id(_series(n=80))
    assert np.isnan(mod.summarize_ticker("X", d)["beta"])
    short = pd.Series(0.001, index=d.index[:10])
    assert np.isnan(mod.summarize_ticker("X", d, short)["beta"])


@pytest.mark.parametrize("x, label", [(0.10, "ON"), (-0.10, "ID"), (0.01, "similar"), (-0.019, "similar")])
def test_dominance_labels(x, label):
    assert mod.dominance(x) == label


def test_group_tests_detect_separated_groups_and_skip_missing_columns():
    rng = np.random.default_rng(2)
    n = 60
    tech = np.arange(n) < 30
    df = pd.DataFrame({
        "sector": np.where(tech, "Information Technology", "Utilities"),
        "mean_ON_bps": np.where(tech, 8.0, 1.0) + rng.normal(0, 0.5, n),
        "market_cap": np.nan,
        "beta": np.nan,
    })
    out = mod.group_tests(df)
    assert list(out["group"]) == ["Tech (IT)"]          # no market cap / beta -> only the sector test
    assert out.loc[0, "welch_p"] < 1e-6 and out.loc[0, "mannwhitney_p"] < 1e-6


# ------------------------------------------------------- yfinance layouts
def _multi(df, tickers, ticker_first):
    frames = {t: df for t in tickers}
    out = pd.concat(frames, axis=1)                      # columns: (ticker, field)
    return out if ticker_first else out.swaplevel(axis=1)


@pytest.mark.parametrize("ticker_first", [True, False])
def test_extract_multiindex_layouts_single_and_many(ticker_first):
    df = _series(n=30)
    for tickers in (["AAA"], ["AAA", "BBB"]):
        data = _multi(df, tickers, ticker_first)
        sub = mod._extract(data, "AAA", single=len(tickers) == 1)
        assert list(sub.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert mod._extract(data, "ZZZ", single=False) is None


def test_extract_flat_frame_only_for_single_batch():
    df = _series(n=30)
    assert mod._extract(df, "AAA", single=True) is not None
    assert mod._extract(df, "AAA", single=False) is None


def test_download_ohlc_uses_extract_for_single_ticker_multiindex(monkeypatch):
    df = _series(n=30)
    fake = types.ModuleType("yfinance")
    fake.download = lambda batch, **kw: _multi(df, batch, ticker_first=True)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = mod.download_ohlc(["NVDA"], start="2022-01-01", pause=0)
    assert list(out) == ["NVDA"]


def test_fetch_market_caps_falls_back_to_info_and_survives_errors(monkeypatch):
    class Ticker:
        def __init__(self, t):
            self.t = t

        @property
        def fast_info(self):
            if self.t == "BAD":
                raise RuntimeError("boom")
            return {"market_cap": 5e9} if self.t == "AAA" else {}

        @property
        def info(self):
            return {"marketCap": 7e9}

    fake = types.ModuleType("yfinance")
    fake.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = mod.fetch_market_caps(["AAA", "BBB", "BAD"]).set_index("ticker")["market_cap"]
    assert out["AAA"] == 5e9 and out["BBB"] == 7e9 and np.isnan(out["BAD"])


# ------------------------------------------------------------------- CLI
def _fake_download(missing=()):
    def fake(tickers, start, end=None, chunk=50, pause=1.0):
        out = {}
        for i, t in enumerate(tickers):
            if t in missing:
                continue
            out[t] = _series(seed=abs(hash(t)) % 1000, drift_on=0.0006 if i % 2 else 0.0)
        return out
    return fake


def test_quick_mode_json_stdout_is_pure_json_and_reports_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(mod, "download_ohlc", _fake_download(missing={"NOPE"}))
    assert mod.main(["--tickers", "nvda, mu,NOPE,brk.b", "--json"]) == 0
    cap = capsys.readouterr()
    payload = json.loads(cap.out)                         # stdout must be valid JSON only
    assert payload["mode"] == "tickers" and payload["missing"] == ["NOPE"]
    assert {r["ticker"] for r in payload["results"]} == {"NVDA", "MU", "BRK-B"}
    assert all(r["dominant"] in {"ON", "ID", "similar"} for r in payload["results"])
    assert "not investment advice" in payload["disclaimer"]
    assert "Downloading OHLC" in cap.err                 # progress went to stderr
    assert not list(tmp_path.iterdir())                   # quick mode writes nothing by default


def test_quick_mode_plain_output_and_optional_csv(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(mod, "download_ohlc", _fake_download())
    out_dir = tmp_path / "nested" / "dir"
    assert mod.main(["--tickers", "AAA,BBB", "--outdir", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "ann_ON_pct" in out and "not investment advice" in out
    assert (out_dir / "tickers_on_id.csv").exists()


def _fake_constituents(n=40):
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(n)], "name": "n",
        "sector": ["Information Technology" if i < n // 2 else "Utilities" for i in range(n)],
        "sub_industry": "s"})


def test_full_scan_json_creates_outdir_and_files_without_meta(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(mod, "download_ohlc", _fake_download())
    monkeypatch.setattr(mod, "get_sp500_constituents", _fake_constituents)
    out_dir = tmp_path / "does" / "not" / "exist"
    assert mod.main(["--skip-meta", "--json", "--charts", "--topx", "5", "--outdir", str(out_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "scan" and payload["n_companies"] == 40
    assert len(payload["on_dominates"]) == len(payload["id_dominates"]) == 5
    names = {g["group"] for g in payload["group_tests"]}
    assert "Tech (IT)" in names and "High beta (top quartile)" in names   # beta now comes from prices
    assert "Small cap (bottom mcap quartile)" not in names                # --skip-meta
    for f in payload["files"]:
        assert (out_dir / f).exists()
    assert "sp500_overnight_dashboard.png" in payload["files"]


def test_full_scan_with_market_caps_runs_small_cap_test(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(mod, "download_ohlc", _fake_download())
    monkeypatch.setattr(mod, "get_sp500_constituents", _fake_constituents)
    monkeypatch.setattr(mod, "fetch_market_caps", lambda ts: pd.DataFrame(
        {"ticker": ts, "market_cap": np.linspace(1e9, 5e11, len(ts))}))
    assert mod.main(["--json", "--outdir", str(tmp_path)]) == 0
    names = {g["group"] for g in json.loads(capsys.readouterr().out)["group_tests"]}
    assert "Small cap (bottom mcap quartile)" in names


def test_no_data_exits_with_clear_json_error(monkeypatch, capsys):
    monkeypatch.setattr(mod, "download_ohlc", lambda *a, **k: {})
    with pytest.raises(SystemExit) as e:
        mod.main(["--tickers", "AAA", "--json"])
    assert e.value.code == 2
    assert "no price data" in json.loads(capsys.readouterr().out)["error"]


def test_constituents_failure_is_reported(monkeypatch, capsys):
    def boom():
        raise OSError("offline")
    monkeypatch.setattr(mod, "get_sp500_constituents", boom)
    with pytest.raises(SystemExit) as e:
        mod.main(["--json"])
    assert e.value.code == 2
    assert "constituents" in json.loads(capsys.readouterr().out)["error"]
