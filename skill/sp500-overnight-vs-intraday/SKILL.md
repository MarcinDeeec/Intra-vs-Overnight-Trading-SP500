---
name: sp500-overnight-vs-intraday
description: Splits stock returns into overnight (close-to-open) and intraday (open-to-close) components and tells you which one drives a stock's performance. Use whenever the user asks if a stock or the S&P 500 earns its return overnight or during the session, about the overnight/gap anomaly, "close to open" returns, ON vs ID, or wants rankings and sector, beta or market-cap comparisons across S&P 500 companies - even if they never say "overnight". Runs a bundled Python script on free Yahoo Finance data (no API key). Quick mode checks a few tickers in seconds; full mode scans the whole index.
license: MIT
---

# S&P 500: overnight vs intraday returns

For each trading day `t` the script computes

- **ON** (overnight) = `Open_t / Close_{t-1} - 1`
- **ID** (intraday) = `Close_t / Open_t - 1`

and compounds them per company, so you can see whether a stock's return is earned
while the market is closed or during the session. Full definitions, the statistics
and all limitations are in `references/methodology.md` (read it before explaining
results in depth or answering methodology questions).

The script is `scripts/sp500_overnight_intraday.py`. It is one self-contained file:
if your agent only has the script text, write it unchanged to a temporary file and run that.

## 1. Setup (once per environment)

Python 3.11 or 3.12 is recommended (3.13+ may lack prebuilt wheels for some packages).

```bash
python -c "import yfinance, pandas, numpy, scipy, lxml, pyarrow, matplotlib" 2>/dev/null \
  || pip install yfinance pandas numpy scipy lxml pyarrow matplotlib
```

Prefer a virtual environment if the user has one; do not install packages globally without asking.
The script only talks to Yahoo Finance (prices), Wikipedia (index members) and, as a fallback,
a public CSV on raw.githubusercontent.com. It needs no API keys and uploads nothing.

## 2. Pick a mode

**Quick mode - the user names specific tickers (default choice).** Takes seconds.

```bash
python sp500_overnight_intraday.py --tickers NVDA,MU,AAPL --start 2018-01-01 --json
```

**Full scan - the user wants rankings, sectors, beta or "which S&P 500 stocks...".**
Downloads ~500 tickers and takes minutes, so:

1. Smoke test first: `python sp500_overnight_intraday.py --limit 10 --skip-meta --json`
2. Then the real run, in the background or with a long command timeout:

```bash
python sp500_overnight_intraday.py --start 2015-01-01 --common-start 2016-01-01 \
  --topx 25 --charts --outdir out --json > scan.json
```

Add `--skip-meta` to skip market caps (faster; the small-cap test is then skipped, the beta
test still runs). Use `--common-start` when comparing companies fairly across the same window.
Progress and warnings go to stderr; with `--json` stdout is a single JSON document.

## 3. Read the result

Quick mode JSON: `results[]` with, per ticker, `cum_ON`, `cum_ID` (cumulative returns, 0.5 = +50%),
`ann_ON_pct`, `ann_ID_pct` (annualized, %), `mean_ON_bps`, `mean_ID_bps` (average daily, basis points),
`beta` (vs SPY, daily), and `dominant`: `"ON"`, `"ID"` or `"similar"` (within 2 pp/year of each other).
Also `missing[]` (tickers with no data - usually typos or delisted), `data_from`, `data_to`, and
`partial_session_dropped` (a date, or null): while the US market is open the script drops today's unfinished
bar automatically, because a live price is not a close. Mention it if it is set.

Scan JSON: `on_dominates`, `id_dominates`, `similar` (top-N lists, ranked by `ann_diff_log`, the annualized
ON-minus-ID advantage, so history length does not matter), `group_tests` (tech, small cap,
high beta vs the rest: Welch and Mann-Whitney p-values on the mean daily overnight return;
if empty, `group_tests_note` says why, usually a scan that was too small),
`median_ann_ON_pct` / `median_ann_ID_pct`, `n_companies`, and `files` written to `outdir`
(CSV/Parquet, and `sp500_overnight_dashboard.png` with `--charts`).

## 4. Report it honestly

- Lead with the numbers and the date window (`data_from` - `data_to`); say which side dominates per ticker.
- Always state the caveats that apply: results are **gross** (no costs, spreads or taxes; trading exactly at
  the open and close is not achievable), prices are **adjusted** Yahoo data, and full scans use **today's**
  index members (survivorship bias inflates results).
- Large overnight returns, especially for growth stocks (e.g. NVDA, MU, AMD), often reflect earnings releases and
  news that come out while the market is closed, i.e. a few big gaps rather than a steady premium. Say so
  when ON dominates strongly.
- A "significant" p-value describes this sample, not the future. Several tests are run; do not oversell them.
- Present this as analysis, never as a trade recommendation or investment advice.
- If tickers are in `missing`, tell the user which ones and suggest checking the spelling
  (share classes use a dash, e.g. `BRK-B`).

## Troubleshooting

- `no price data returned`: no network, Yahoo rate-limited the request, or all tickers are wrong. Wait a minute,
  retry with fewer tickers, check the symbols.
- Very slow full scan: use `--limit`, `--skip-meta`, or a later `--start`.
- Package build errors: use Python 3.11/3.12 in a fresh virtual environment.
- Wikipedia unreachable: the script falls back to a CSV mirror automatically.
