#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S&P 500 — Overnight (ON) vs Intraday (ID) decomposition
=======================================================

What this script does:
  1. Fetches the CURRENT S&P 500 constituents (from Wikipedia) with GICS sector.
  2. Downloads daily adjusted OHLC from Yahoo Finance (yfinance).
  3. For each session t computes:
        ON_t = Open_t / Close_{t-1} - 1     (overnight)
        ID_t = Close_t / Open_t   - 1       (intraday)
  4. Computes CUMULATIVE ON and ID per company (product of (1+r)).
  5. Builds rankings:
        - ON >> ID  (like Micron / MU)
        - ID >> ON
        - ON ~ ID   (similar)
  6. Adds metadata: sector, market cap, annualized volatility, beta.
  7. Statistical tests: do groups (e.g. tech, small caps, high beta) have a
     significantly different overnight profile (Welch t-test + Mann-Whitney)?
  8. Saves results to CSV/Parquet, optional charts, and prints a summary.

Requirements:
    pip install yfinance pandas numpy scipy lxml pyarrow matplotlib

Usage:
    python sp500_overnight_intraday.py --start 2015-01-01 --topx 25 --charts --outdir out

Methodology notes:
  - We use ADJUSTED prices (auto_adjust=True), so splits/dividends do not
    artificially distort overnight gaps. ON and ID are then scaled consistently.
  - 'Cumulative' = geometric product of (1+r). We also report the sum of log
    returns, since log(1+ON)+log(1+ID) = log(1+close-to-close) is additive.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# 1. S&P 500 constituents
# ----------------------------------------------------------------------------
def _fetch_url_text(url, timeout=30):
    """Fetch URL content with a browser User-Agent (Wikipedia blocks missing UA -> 403)."""
    import urllib.request

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_sp500_constituents():
    """Return DataFrame: ticker, name, sector, sub_industry."""
    import io

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = _fetch_url_text(url)
        tables = pd.read_html(io.StringIO(html))
        df = tables[0].copy()
    except Exception as e:
        # Fallback: stable CSV mirror (datahub) with the same columns.
        print(f"      (Wikipedia unavailable: {e}; using CSV mirror)", flush=True)
        csv_url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/main/data/constituents.csv"
        )
        df = pd.read_csv(io.StringIO(_fetch_url_text(csv_url)))
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    # Yahoo uses '-' instead of '.' (e.g. BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    return df[["ticker", "name", "sector", "sub_industry"]].drop_duplicates("ticker")


# ----------------------------------------------------------------------------
# 2. OHLC download
# ----------------------------------------------------------------------------
def download_ohlc(tickers, start, end=None, chunk=50, pause=1.0):
    """Download adjusted OHLC in batches. Returns dict[ticker] -> DataFrame(Open,High,Low,Close,Volume)."""
    import yfinance as yf

    out = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        print(f"  downloading {i+1}-{i+len(batch)} / {len(tickers)} ...", flush=True)
        data = yf.download(
            batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        for t in batch:
            try:
                if len(batch) == 1:
                    sub = data
                else:
                    sub = data[t]
                sub = sub.dropna(how="all")
                if sub.empty or "Open" not in sub or "Close" not in sub:
                    continue
                out[t] = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception:
                continue
        time.sleep(pause)
    return out


# ----------------------------------------------------------------------------
# 3-4. ON / ID and cumulative
# ----------------------------------------------------------------------------
def compute_on_id(df):
    """For a single company return a DataFrame with ON, ID and their logs."""
    d = df.copy()
    d["prev_close"] = d["Close"].shift(1)
    d["ON"] = d["Open"] / d["prev_close"] - 1.0
    d["ID"] = d["Close"] / d["Open"] - 1.0
    d = d.dropna(subset=["ON", "ID"])
    d["logON"] = np.log1p(d["ON"])
    d["logID"] = np.log1p(d["ID"])
    return d


def summarize_ticker(t, d):
    """Return a dict of cumulative/statistical metrics for one company."""
    n = len(d)
    if n < 20:
        return None
    cum_on = float(np.expm1(d["logON"].sum()))   # geometric cumulative ON
    cum_id = float(np.expm1(d["logID"].sum()))   # geometric cumulative ID
    sum_log_on = float(d["logON"].sum())
    sum_log_id = float(d["logID"].sum())
    ann = 252.0
    vol_on = float(d["ON"].std() * np.sqrt(ann))
    vol_id = float(d["ID"].std() * np.sqrt(ann))
    vol_c2c = float((d["Close"].pct_change().std()) * np.sqrt(ann))
    return {
        "ticker": t,
        "n_days": n,
        "cum_ON": cum_on,
        "cum_ID": cum_id,
        "sum_logON": sum_log_on,
        "sum_logID": sum_log_id,
        "diff_log": sum_log_on - sum_log_id,   # >0 => ON dominates
        # per-company annualization: mean daily log return * 252
        "ann_ON_pct": float(np.expm1(sum_log_on / n * ann) * 100),
        "ann_ID_pct": float(np.expm1(sum_log_id / n * ann) * 100),
        "ann_diff_log": float((sum_log_on - sum_log_id) / n * ann),
        "mean_ON_bps": float(d["ON"].mean() * 1e4),
        "mean_ID_bps": float(d["ID"].mean() * 1e4),
        "vol_ON": vol_on,
        "vol_ID": vol_id,
        "vol_close2close": vol_c2c,
    }


# ----------------------------------------------------------------------------
# 6. Metadata (market cap, beta)
# ----------------------------------------------------------------------------
def fetch_meta(tickers, pause=0.0):
    """Market cap + beta from yfinance (fast_info / info). Can be slow."""
    import yfinance as yf

    rows = []
    for i, t in enumerate(tickers):
        mc, beta = np.nan, np.nan
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", {}) or {}
            mc = fi.get("market_cap", np.nan)
            if mc is None or (isinstance(mc, float) and np.isnan(mc)):
                info = tk.info
                mc = info.get("marketCap", np.nan)
                beta = info.get("beta", np.nan)
        except Exception:
            pass
        rows.append({"ticker": t, "market_cap": mc, "beta": beta})
        if (i + 1) % 50 == 0:
            print(f"  metadata {i+1}/{len(tickers)}", flush=True)
        if pause:
            time.sleep(pause)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. Statistical tests between groups
# ----------------------------------------------------------------------------
def group_tests(df, value_col="sum_logON"):
    """Compare overnight profile between groups: tech vs rest, small vs large, high vs low beta."""
    from scipy import stats

    results = []

    def run(name, mask):
        a = df.loc[mask, value_col].dropna()
        b = df.loc[~mask, value_col].dropna()
        if len(a) < 5 or len(b) < 5:
            return
        tt = stats.ttest_ind(a, b, equal_var=False)   # Welch
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        results.append(
            {
                "group": name,
                "n_group": len(a),
                "n_rest": len(b),
                "mean_group": float(a.mean()),
                "mean_rest": float(b.mean()),
                "welch_t": float(tt.statistic),
                "welch_p": float(tt.pvalue),
                "mannwhitney_p": float(mw.pvalue),
            }
        )

    # tech
    run("Tech (IT)", df["sector"].eq("Information Technology"))
    # small caps = bottom market-cap quartile
    if df["market_cap"].notna().sum() > 20:
        q = df["market_cap"].quantile(0.25)
        run("Small cap (bottom mcap quartile)", df["market_cap"] <= q)
    # high beta
    if df["beta"].notna().sum() > 20:
        bq = df["beta"].quantile(0.75)
        run("High beta (top quartile)", df["beta"] >= bq)

    return pd.DataFrame(results)


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def make_charts(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = res.copy()
    sub_b = df[["ann_diff_log", "beta"]].dropna()
    r_beta = np.corrcoef(sub_b.ann_diff_log, sub_b.beta)[0, 1] if len(sub_b) > 2 else float("nan")
    sub_v = df[["ann_diff_log", "vol_close2close"]].dropna()
    r_vol = np.corrcoef(sub_v.ann_diff_log, sub_v.vol_close2close)[0, 1] if len(sub_v) > 2 else float("nan")

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("S&P 500: overnight (ON) vs intraday (ID) decomposition — per-company annualization",
                 fontsize=14, fontweight="bold")

    a = ax[0, 0]
    lo, hi = -20, 60
    a.hist(df.ann_ON_pct.clip(lo, hi), bins=40, alpha=0.6, label="Overnight (ON)", color="#2c7fb8")
    a.hist(df.ann_ID_pct.clip(lo, hi), bins=40, alpha=0.6, label="Intraday (ID)", color="#de2d26")
    a.axvline(df.ann_ON_pct.median(), color="#2c7fb8", ls="--", lw=2)
    a.axvline(df.ann_ID_pct.median(), color="#de2d26", ls="--", lw=2)
    a.set_title(f"Annualized return distribution (median ON={df.ann_ON_pct.median():.1f}%, ID={df.ann_ID_pct.median():.1f}%)")
    a.set_xlabel("Annualized return [%]"); a.set_ylabel("Number of companies"); a.legend()

    b = ax[0, 1]
    if df.beta.notna().sum() > 5:
        hi_beta = df.beta.quantile(0.75)
        col = np.where(df.beta >= hi_beta, "#d95f02", "#7570b3")
        b.scatter(df.beta, df.ann_diff_log, c=col, alpha=0.6, s=22)
        m = df[["beta", "ann_diff_log"]].dropna()
        z = np.polyfit(m.beta, m.ann_diff_log, 1)
        xs = np.linspace(m.beta.min(), m.beta.max(), 50)
        b.plot(xs, np.polyval(z, xs), "k-", lw=2)
    b.axhline(0, color="gray", ls=":")
    b.set_title(f"Beta vs ON advantage (r={r_beta:.2f}); orange = high beta")
    b.set_xlabel("Beta"); b.set_ylabel("Annual ON-minus-ID advantage (log)")

    c = ax[1, 0]
    c.scatter(df.vol_close2close, df.ann_diff_log, c="#1b9e77", alpha=0.55, s=22)
    m2 = df[["vol_close2close", "ann_diff_log"]].dropna()
    z2 = np.polyfit(m2.vol_close2close, m2.ann_diff_log, 1)
    xs2 = np.linspace(m2.vol_close2close.min(), m2.vol_close2close.max(), 50)
    c.plot(xs2, np.polyval(z2, xs2), "k-", lw=2)
    c.axhline(0, color="gray", ls=":")
    c.set_title(f"Volatility vs ON advantage (r={r_vol:.2f})")
    c.set_xlabel("Annualized volatility"); c.set_ylabel("Annual ON-minus-ID advantage (log)")

    d = ax[1, 1]
    g = df.groupby("sector").agg(ON=("ann_ON_pct", "mean"), ID=("ann_ID_pct", "mean")).sort_values("ON")
    y = np.arange(len(g)); h = 0.4
    d.barh(y + h / 2, g.ON, height=h, color="#2c7fb8", label="ON")
    d.barh(y - h / 2, g.ID, height=h, color="#de2d26", label="ID")
    d.set_yticks(y); d.set_yticklabels(g.index, fontsize=9)
    d.set_title("Mean annual ON vs ID by sector"); d.set_xlabel("Annualized return [%]"); d.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{outdir}/sp500_overnight_dashboard.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--topx", type=int, default=25)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--skip-meta", action="store_true", help="skip downloading market cap/beta")
    ap.add_argument("--common-start", default=None, help="common window: truncate all series to this date (YYYY-MM-DD)")
    ap.add_argument("--charts", action="store_true", help="save PNG charts (dashboard)")
    ap.add_argument("--limit", type=int, default=None, help="limit number of companies (debug)")
    args = ap.parse_args()

    print("[1/6] S&P 500 constituents ...", flush=True)
    cons = get_sp500_constituents()
    if args.limit:
        cons = cons.head(args.limit)
    tickers = cons["ticker"].tolist()
    print(f"      {len(tickers)} companies", flush=True)

    print("[2/6] Downloading OHLC ...", flush=True)
    ohlc = download_ohlc(tickers, start=args.start, end=args.end)
    print(f"      data downloaded for {len(ohlc)} companies", flush=True)

    print("[3/6] ON / ID + cumulative ...", flush=True)
    summ = []
    for t, df in ohlc.items():
        if args.common_start:
            df = df[df.index >= pd.Timestamp(args.common_start)]
            if len(df) < 21:
                continue
        d = compute_on_id(df)
        s = summarize_ticker(t, d)
        if s:
            summ.append(s)
    res = pd.DataFrame(summ)
    res = res.merge(cons, on="ticker", how="left")

    if not args.skip_meta:
        print("[4/6] Metadata (market cap, beta) ...", flush=True)
        meta = fetch_meta(res["ticker"].tolist())
        res = res.merge(meta, on="ticker", how="left")
    else:
        res["market_cap"] = np.nan
        res["beta"] = np.nan

    # Classify ON vs ID based on the log-return difference
    print("[5/6] Rankings ...", flush=True)
    res = res.sort_values("diff_log", ascending=False)
    on_dom = res.head(args.topx).copy()                          # ON >> ID
    id_dom = res.sort_values("diff_log").head(args.topx).copy()   # ID >> ON
    # similar: smallest |diff|
    res["abs_diff"] = res["diff_log"].abs()
    similar = res.sort_values("abs_diff").head(args.topx).copy()

    print("[6/6] Statistical tests ...", flush=True)
    try:
        tests = group_tests(res, value_col="sum_logON")
    except Exception as e:
        print(f"      (tests skipped: {e})", flush=True)
        tests = pd.DataFrame()

    # Save
    od = args.outdir.rstrip("/")
    res.drop(columns=["abs_diff"]).to_csv(f"{od}/sp500_on_id_all.csv", index=False)
    on_dom.to_csv(f"{od}/rank_ON_dominates.csv", index=False)
    id_dom.to_csv(f"{od}/rank_ID_dominates.csv", index=False)
    similar.to_csv(f"{od}/rank_similar.csv", index=False)
    if not tests.empty:
        tests.to_csv(f"{od}/group_tests.csv", index=False)
    try:
        res.drop(columns=["abs_diff"]).to_parquet(f"{od}/sp500_on_id_all.parquet")
    except Exception:
        pass

    # Console summary
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    cols = ["ticker", "sector", "cum_ON", "cum_ID", "diff_log", "vol_close2close"]
    print("\n=== TOP: ON >> ID (overnight dominates, like MU) ===")
    print(on_dom[cols].to_string(index=False))
    print("\n=== TOP: ID >> ON (intraday dominates) ===")
    print(id_dom[cols].to_string(index=False))
    print("\n=== Similar (ON ~ ID) ===")
    print(similar[cols].to_string(index=False))
    if not tests.empty:
        print("\n=== Tests: does the group have a different overnight profile (sum_logON) ===")
        print(tests.to_string(index=False))

    if args.charts:
        print("      saving charts ...", flush=True)
        try:
            make_charts(res, od)
            print("      saved: sp500_overnight_dashboard.png", flush=True)
        except Exception as e:
            print(f"      (charts skipped: {e})", flush=True)

    print(f"\nDone. Files saved in: {od}/")
    print("  sp500_on_id_all.csv / .parquet, rank_*.csv, group_tests.csv")


if __name__ == "__main__":
    main()
