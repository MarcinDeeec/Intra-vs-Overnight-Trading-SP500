#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S&P 500 overnight (ON) vs intraday (ID) decomposition.

    ON_t = Open_t / Close_{t-1} - 1        ID_t = Close_t / Open_t - 1

Quick mode (a few tickers, ~seconds):
    python sp500_overnight_intraday.py --tickers NVDA,MU,AAPL [--json]
Full scan of the current S&P 500 (minutes):
    python sp500_overnight_intraday.py --start 2015-01-01 --charts --outdir out
Useful flags: --skip-meta (no market caps, faster), --limit N (smoke test),
--common-start YYYY-MM-DD, --topx N, --json (machine-readable stdout).

Needs: pip install yfinance pandas numpy scipy lxml pyarrow matplotlib
Adjusted prices, gross returns (no costs); full scans use TODAY's constituents
(survivorship bias). Educational analysis, not investment advice.
"""

import argparse
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

UA = ("sp500-overnight-intraday/1.2 "
      "(+https://github.com/MarcinDeeec/Intra-vs-Overnight-Trading-SP500)")
SIMILAR_BAND = 0.02  # |ON - ID| < 2 pp/yr (log) is labelled "similar"
DISCLAIMER = ("Gross returns on adjusted Yahoo Finance prices; full scans use today's "
              "S&P 500 members (survivorship bias). Educational, not investment advice.")
RES_COLS = ["ticker", "sector", "cum_ON", "cum_ID", "ann_ON_pct", "ann_ID_pct",
            "ann_diff_log", "beta", "dominant"]


def log(*a):
    """Progress/diagnostics go to stderr so stdout stays clean for --json."""
    print(*a, file=sys.stderr, flush=True)


def fail(msg, as_json=False, code=2):
    if as_json:
        print(json.dumps({"error": msg}))
    log("ERROR: " + msg)
    sys.exit(code)


# --- 1. S&P 500 constituents ---
def _fetch_url_text(url, timeout=30):
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_sp500_constituents():
    """Return DataFrame: ticker, name, sector, sub_industry."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df = pd.read_html(io.StringIO(_fetch_url_text(url)))[0].copy()
    except Exception as e:
        log(f"  (Wikipedia unavailable: {e}; using CSV mirror)")
        csv_url = ("https://raw.githubusercontent.com/datasets/"
                   "s-and-p-500-companies/main/data/constituents.csv")
        df = pd.read_csv(io.StringIO(_fetch_url_text(csv_url)))
    df = df.rename(columns={"Symbol": "ticker", "Security": "name",
                            "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()  # BRK.B -> BRK-B
    return df[["ticker", "name", "sector", "sub_industry"]].drop_duplicates("ticker")


# --- 2. OHLC download ---
def _extract(data, t, single):
    """One ticker's OHLCV frame from a yfinance download (handles every column layout)."""
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if t in cols.get_level_values(0):
            sub = data[t]
        elif t in cols.get_level_values(1):
            sub = data.xs(t, axis=1, level=1)
        else:
            return None
    elif single:
        sub = data
    else:
        return None
    sub = sub.dropna(how="all")
    if sub.empty or "Open" not in sub.columns or "Close" not in sub.columns:
        return None
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in sub.columns]
    return sub[keep].copy()


def download_ohlc(tickers, start, end=None, chunk=50, pause=1.0, quiet=False):
    """Adjusted daily OHLC in batches -> dict[ticker] -> DataFrame."""
    import yfinance as yf

    out = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        if not quiet:
            log(f"  downloading {i + 1}-{i + len(batch)} / {len(tickers)} ...")
        try:
            data = yf.download(batch, start=start, end=end, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:
            log(f"  (batch failed: {e})")
            continue
        for t in batch:
            try:
                sub = _extract(data, t, single=len(batch) == 1)
            except Exception:
                sub = None
            if sub is not None:
                out[t] = sub
        if i + chunk < len(tickers):
            time.sleep(pause)
    return out


# --- 3-4. ON / ID, cumulative, beta ---
def dominance(ann_diff_log):
    if abs(ann_diff_log) < SIMILAR_BAND:
        return "similar"
    return "ON" if ann_diff_log > 0 else "ID"


def compute_on_id(df):
    """For one company: ON, ID and their logs (first row is dropped: no previous close)."""
    d = df.copy()
    d["prev_close"] = d["Close"].shift(1)
    d["ON"] = d["Open"] / d["prev_close"] - 1.0
    d["ID"] = d["Close"] / d["Open"] - 1.0
    d = d.dropna(subset=["ON", "ID"])
    d["logON"] = np.log1p(d["ON"])
    d["logID"] = np.log1p(d["ID"])
    return d


def summarize_ticker(t, d, mkt_ret=None):
    """Cumulative/statistical metrics for one company (None if < 20 sessions).

    mkt_ret: optional Series of market (SPY) close-to-close returns -> beta.
    """
    n = len(d)
    if n < 20:
        return None
    ann = 252.0
    s_on, s_id = float(d["logON"].sum()), float(d["logID"].sum())
    c2c = d["Close"].pct_change()
    beta = np.nan
    if mkt_ret is not None:
        j = pd.concat([c2c, mkt_ret], axis=1, join="inner").dropna()
        if len(j) >= 60 and j.iloc[:, 1].var() > 0:
            beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    ann_diff = (s_on - s_id) / n * ann
    return {
        "ticker": t,
        "n_days": n,
        "cum_ON": float(np.expm1(s_on)),
        "cum_ID": float(np.expm1(s_id)),
        "sum_logON": s_on,
        "sum_logID": s_id,
        "diff_log": s_on - s_id,                       # > 0 => ON dominates
        "ann_ON_pct": float(np.expm1(s_on / n * ann) * 100),
        "ann_ID_pct": float(np.expm1(s_id / n * ann) * 100),
        "ann_diff_log": float(ann_diff),
        "mean_ON_bps": float(d["ON"].mean() * 1e4),
        "mean_ID_bps": float(d["ID"].mean() * 1e4),
        "vol_ON": float(d["ON"].std() * np.sqrt(ann)),
        "vol_ID": float(d["ID"].std() * np.sqrt(ann)),
        "vol_close2close": float(c2c.std() * np.sqrt(ann)),
        "beta": beta,
        "dominant": dominance(ann_diff),
    }


# --- 6. Market caps (beta is computed from prices, see summarize_ticker) ---
def fetch_market_caps(tickers, workers=8):
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            tk = yf.Ticker(t)
            mc = tk.fast_info.get("market_cap")
            if not mc:
                mc = tk.info.get("marketCap")
            return t, float(mc) if mc else np.nan
        except Exception:
            return t, np.nan

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, tickers))
    return pd.DataFrame(rows, columns=["ticker", "market_cap"])


# --- 7. Group tests ---
def group_tests(df, value_col="mean_ON_bps"):
    """Welch t-test + Mann-Whitney: tech / small cap / high beta vs the rest."""
    from scipy import stats

    results = []

    def run(name, mask):
        a = df.loc[mask, value_col].dropna()
        b = df.loc[~mask, value_col].dropna()
        if len(a) < 5 or len(b) < 5:
            return
        tt = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        results.append({"group": name, "n_group": len(a), "n_rest": len(b),
                        "mean_group": float(a.mean()), "mean_rest": float(b.mean()),
                        "welch_t": float(tt.statistic), "welch_p": float(tt.pvalue),
                        "mannwhitney_p": float(mw.pvalue)})

    run("Tech (IT)", df["sector"].eq("Information Technology"))
    if df["market_cap"].notna().sum() > 20:
        run("Small cap (bottom mcap quartile)", df["market_cap"] <= df["market_cap"].quantile(0.25))
    if df["beta"].notna().sum() > 20:
        run("High beta (top quartile)", df["beta"] >= df["beta"].quantile(0.75))
    return pd.DataFrame(results)


# --- Charts ---
def make_charts(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = res.copy()

    def corr(col):
        s = df[["ann_diff_log", col]].dropna()
        return np.corrcoef(s.ann_diff_log, s[col])[0, 1] if len(s) > 2 else float("nan")

    def scatter(ax, col, color, title, xlabel):
        s = df[["ann_diff_log", col]].dropna()
        ax.axhline(0, color="gray", ls=":")
        if len(s) > 2:
            ax.scatter(s[col], s.ann_diff_log, c=color, alpha=0.55, s=22)
            xs = np.linspace(s[col].min(), s[col].max(), 50)
            ax.plot(xs, np.polyval(np.polyfit(s[col], s.ann_diff_log, 1), xs), "k-", lw=2)
        ax.set_title(f"{title} (r={corr(col):.2f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Annual ON-minus-ID advantage (log)")

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("S&P 500: overnight (ON) vs intraday (ID) decomposition - per-company annualization",
                 fontsize=14, fontweight="bold")
    a = ax[0, 0]
    a.hist(df.ann_ON_pct.clip(-20, 60), bins=40, alpha=0.6, label="Overnight (ON)", color="#2c7fb8")
    a.hist(df.ann_ID_pct.clip(-20, 60), bins=40, alpha=0.6, label="Intraday (ID)", color="#de2d26")
    a.axvline(df.ann_ON_pct.median(), color="#2c7fb8", ls="--", lw=2)
    a.axvline(df.ann_ID_pct.median(), color="#de2d26", ls="--", lw=2)
    a.set_title(f"Annualized return (median ON={df.ann_ON_pct.median():.1f}%, "
                f"ID={df.ann_ID_pct.median():.1f}%)")
    a.set_xlabel("Annualized return [%]")
    a.set_ylabel("Number of companies")
    a.legend()
    scatter(ax[0, 1], "beta", "#d95f02", "Beta vs ON advantage", "Beta (vs SPY, daily)")
    scatter(ax[1, 0], "vol_close2close", "#1b9e77", "Volatility vs ON advantage", "Annualized volatility")
    d = ax[1, 1]
    g = df.groupby("sector").agg(ON=("ann_ON_pct", "mean"), ID=("ann_ID_pct", "mean")).sort_values("ON")
    y, h = np.arange(len(g)), 0.4
    d.barh(y + h / 2, g.ON, height=h, color="#2c7fb8", label="ON")
    d.barh(y - h / 2, g.ID, height=h, color="#de2d26", label="ID")
    d.set_yticks(y)
    d.set_yticklabels(g.index, fontsize=9)
    d.set_title("Mean annual ON vs ID by sector")
    d.set_xlabel("Annualized return [%]")
    d.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=130)
    plt.close(fig)


# --- Output helpers ---
def to_records(df, cols=RES_COLS):
    cols = [c for c in cols if c in df.columns]
    return json.loads(df[cols].round(4).to_json(orient="records"))


def _num(x):
    return None if pd.isna(x) else round(float(x), 4)


def _now_et():
    return pd.Timestamp.now(tz="America/New_York")


def _drop_open_session(ohlc, now=None):
    """Drop today's bar while the US market is still open (it holds a live price, not a close).

    Returns (ohlc, dropped_date_or_None). Cutoff 16:30 ET = close plus Yahoo's delay.
    """
    try:
        now = now if now is not None else _now_et()
    except Exception:
        return ohlc, None
    if now.weekday() >= 5 or now.hour * 60 + now.minute >= 16 * 60 + 30:
        return ohlc, None
    out, dropped = {}, None
    for t, df in ohlc.items():
        if len(df) and pd.Timestamp(df.index[-1]).date() == now.date():
            df, dropped = df.iloc[:-1], str(now.date())
        if len(df):
            out[t] = df
    return out, dropped


def _window(ohlc):
    idx = [df.index for df in ohlc.values()]
    return {"data_from": str(min(i.min() for i in idx).date()),
            "data_to": str(max(i.max() for i in idx).date())}


def _compute_all(ohlc, mkt_ret, common_start):
    rows = []
    for t, df in ohlc.items():
        if common_start:
            df = df[df.index >= pd.Timestamp(common_start)]
            if len(df) < 21:
                continue
        s = summarize_ticker(t, compute_on_id(df), mkt_ret)
        if s:
            rows.append(s)
    return pd.DataFrame(rows)


# --- Main ---
def build_parser():
    ap = argparse.ArgumentParser(description="S&P 500 overnight vs intraday decomposition")
    ap.add_argument("--tickers", default=None, help="comma-separated tickers (quick mode, skips the full scan)")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--topx", type=int, default=25)
    ap.add_argument("--outdir", default=None, help="output dir (full scan default: out; quick mode: no files)")
    ap.add_argument("--skip-meta", action="store_true", help="skip downloading market caps")
    ap.add_argument("--common-start", default=None, help="truncate all series to this date (YYYY-MM-DD)")
    ap.add_argument("--charts", action="store_true", help="save PNG dashboard (full scan)")
    ap.add_argument("--limit", type=int, default=None, help="limit number of companies (debug)")
    ap.add_argument("--json", action="store_true", help="print a compact JSON summary to stdout")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    quick = bool(args.tickers)
    cons = None
    steps = [0, 2 if quick else (5 if args.skip_meta else 6)]

    def step(msg):
        steps[0] += 1
        log(f"[{steps[0]}/{steps[1]}] {msg}")

    if quick:
        tickers = list(dict.fromkeys(t.strip().upper().replace(".", "-")
                                     for t in args.tickers.split(",") if t.strip()))
        if not tickers:
            fail("--tickers is empty", args.json, 1)
    else:
        step("S&P 500 constituents ...")
        try:
            cons = get_sp500_constituents()
        except Exception as e:
            fail(f"could not fetch S&P 500 constituents ({e}); check the network or use --tickers", args.json)
        if args.limit:
            cons = cons.head(args.limit)
        tickers = cons["ticker"].tolist()
        log(f"      {len(tickers)} companies")

    step("Downloading OHLC ...")
    ohlc = download_ohlc(tickers, start=args.start, end=args.end)
    if not ohlc:
        fail("no price data returned - network problem, Yahoo Finance rate limit, or wrong tickers", args.json)
    missing = [t for t in tickers if t not in ohlc]
    if missing:
        log(f"      no data for: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    spy = download_ohlc(["SPY"], start=args.start, end=args.end, quiet=True).get("SPY")
    ohlc, partial = _drop_open_session(ohlc)
    if partial:
        log(f"      dropped today's unfinished session ({partial}): US market still open")
    if spy is not None:
        spy = _drop_open_session({"SPY": spy})[0].get("SPY")
    mkt_ret = spy["Close"].pct_change() if spy is not None else None
    if mkt_ret is None:
        log("      (SPY unavailable: beta will be empty)")

    step("Computing ON / ID ...")
    res = _compute_all(ohlc, mkt_ret, args.common_start)
    if res.empty:
        fail("no ticker had enough history (need >= 20 sessions)", args.json)
    meta = {**_window(ohlc), "partial_session_dropped": partial}

    if quick:
        res = res.sort_values("ann_diff_log", ascending=False)
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            res.to_csv(os.path.join(args.outdir, "tickers_on_id.csv"), index=False)
        if args.json:
            print(json.dumps({"mode": "tickers", **meta, "missing": missing,
                              "results": to_records(res), "disclaimer": DISCLAIMER}, indent=1))
        else:
            pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
            cols = ["ticker", "n_days", "cum_ON", "cum_ID", "ann_ON_pct", "ann_ID_pct",
                    "mean_ON_bps", "mean_ID_bps", "beta", "dominant"]
            print(res[cols].to_string(index=False))
            print(f"\n{DISCLAIMER}")
        return 0

    res = res.merge(cons, on="ticker", how="left")
    if args.skip_meta:
        res["market_cap"] = np.nan
    else:
        step("Market caps ...")
        res = res.merge(fetch_market_caps(res["ticker"].tolist()), on="ticker", how="left")

    step("Rankings ...")
    res = res.sort_values("ann_diff_log", ascending=False)
    on_dom = res.head(args.topx).copy()
    id_dom = res.sort_values("ann_diff_log").head(args.topx).copy()
    similar = (res.assign(abs_diff=res["ann_diff_log"].abs()).sort_values("abs_diff")
               .head(args.topx).drop(columns="abs_diff"))

    step("Statistical tests ...")
    try:
        tests = group_tests(res)
    except Exception as e:
        log(f"      (tests skipped: {e})")
        tests = pd.DataFrame()
    note = None
    if tests.empty:
        note = ("no group tests ran: each group needs >= 5 companies, and the beta / market-cap "
                "tests need > 20 values - run a larger scan")
        log("      " + note)

    od = args.outdir or "out"
    os.makedirs(od, exist_ok=True)
    files = {"sp500_on_id_all.csv": res, "rank_ON_dominates.csv": on_dom,
             "rank_ID_dominates.csv": id_dom, "rank_similar.csv": similar}
    if not tests.empty:
        files["group_tests.csv"] = tests
    for name, frame in files.items():
        frame.to_csv(os.path.join(od, name), index=False)
    try:
        res.to_parquet(os.path.join(od, "sp500_on_id_all.parquet"))
    except Exception:
        pass
    if args.charts:
        try:
            make_charts(res, os.path.join(od, "sp500_overnight_dashboard.png"))
            files["sp500_overnight_dashboard.png"] = None
        except Exception as e:
            log(f"      (charts skipped: {e})")

    if args.json:
        print(json.dumps({
            "mode": "scan", **meta, "n_companies": len(res), "missing_count": len(missing),
            "median_ann_ON_pct": _num(res["ann_ON_pct"].median()),
            "median_ann_ID_pct": _num(res["ann_ID_pct"].median()),
            "on_dominates": to_records(on_dom), "id_dominates": to_records(id_dom),
            "similar": to_records(similar),
            "group_tests": to_records(tests, list(tests.columns)) if not tests.empty else [],
            "group_tests_note": note,
            "outdir": od, "files": sorted(files), "disclaimer": DISCLAIMER}, indent=1))
    else:
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        cols = ["ticker", "sector", "cum_ON", "cum_ID", "ann_diff_log", "vol_close2close"]
        print("\n=== TOP: ON >> ID (overnight dominates) ===")
        print(on_dom[cols].to_string(index=False))
        print("\n=== TOP: ID >> ON (intraday dominates) ===")
        print(id_dom[cols].to_string(index=False))
        print("\n=== Similar (ON ~ ID) ===")
        print(similar[cols].to_string(index=False))
        if not tests.empty:
            print("\n=== Group tests on mean daily overnight return (bps) ===")
            print(tests.to_string(index=False))
        print(f"\nDone. Files saved in: {od}/\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
