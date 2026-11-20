# S&P 500 — Overnight vs Intraday

[![tests](https://github.com/MarcinDeeec/Intra-vs-Overnight-Trading-SP500/actions/workflows/tests.yml/badge.svg)](https://github.com/MarcinDeeec/Intra-vs-Overnight-Trading-SP500/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Decomposes S&P 500 stock returns into an **overnight (ON)** and **intraday (ID)** component, with rankings, metadata (sector, market cap, volatility, beta) and statistical tests between groups.

The project replicates a well-known market anomaly: historically, most of a stock's return is earned **outside trading hours** (between the close and the next open), not during the session.

## Definitions

For each session `t`:

```
ON_t = Open_t / Close_{t-1} - 1     # overnight (the overnight gap)
ID_t = Close_t / Open_t   - 1       # intraday (the trading session)
```

Cumulative returns are computed as the geometric product of `(1 + r)` (and also as the sum of log returns, since `log(1+ON) + log(1+ID) = log(1 + close-to-close)`).

## Features

- Fetches the **current S&P 500 constituents** from Wikipedia (with a CSV-mirror fallback) including GICS sector.
- Downloads daily **adjusted OHLC** from Yahoo Finance (`yfinance`).
- Computes cumulative and **annualized** ON and ID per company.
- Builds 3 rankings: `ON >> ID`, `ID >> ON`, `ON ~ ID` (ranked by the annualized ON-minus-ID advantage `ann_diff_log`, so history length does not matter).
- Adds sector, market cap, volatility (ON/ID/close-to-close) and **beta vs SPY** (computed from daily close-to-close returns, so it needs no extra per-ticker requests).
- Statistical tests (Welch t-test + Mann-Whitney) on the mean daily overnight return: do tech / small caps / high beta have a different overnight profile.
- **Quick mode** (`--tickers`) for a handful of tickers in seconds, and `--json` for machine-readable output (used by the AI-agent skill below).
- Generates a summary chart (`--charts`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Recommended Python **3.11 / 3.12**. On the newest versions (3.13/3.14) some libraries may not yet ship prebuilt wheels.

## Usage

```bash
# full run with charts, results written to out/
python sp500_overnight_intraday.py --start 2015-01-01 --charts --outdir out

# common window (apples-to-apples) and top 25 in rankings
python sp500_overnight_intraday.py --start 2015-01-01 --common-start 2016-01-01 --topx 25 --charts --outdir out

# quick test without market caps, on 10 companies
python sp500_overnight_intraday.py --limit 10 --skip-meta --outdir out

# quick mode: just a few tickers, no files written, compact table
python sp500_overnight_intraday.py --tickers NVDA,MU,AAPL --start 2018-01-01

# same, as JSON on stdout (progress/warnings go to stderr)
python sp500_overnight_intraday.py --tickers NVDA,MU --json
```

| Flag | Description |
|---|---|
| `--tickers` | comma-separated tickers: quick mode (skips the full S&P 500 scan, no files unless `--outdir` is given) |
| `--start` | OHLC history start (default 2015-01-01) |
| `--end` | history end (default: today) |
| `--common-start` | common window — truncate all series to this date |
| `--topx` | ranking size (default 25) |
| `--skip-meta` | skip downloading market caps (beta is always computed from prices) |
| `--charts` | save the PNG dashboard |
| `--limit` | limit the number of companies (debug) |
| `--json` | print a compact JSON summary to stdout |
| `--outdir` | output directory (full scan default `out`, created if missing) |

## Outputs

- `sp500_on_id_all.csv` / `.parquet` — full per-company metrics
- `rank_ON_dominates.csv`, `rank_ID_dominates.csv`, `rank_similar.csv`
- `group_tests.csv` — statistical test results
- `sp500_overnight_dashboard.png` — charts
- quick mode with `--outdir`: `tickers_on_id.csv`

![dashboard](examples/sp500_overnight_dashboard.png)

## Methodology and caveats

- Uses **adjusted** prices (`auto_adjust=True`), so splits/dividends do not artificially distort overnight gaps.
- **Survivorship bias**: we use *today's* index membership, so results are upward-biased (companies that dropped out of the S&P 500 are missing).
- Results are **gross** — no transaction costs, taxes or slippage; executing at the exact open/close prices is not achievable in practice.
- **Earnings and news gaps**: companies usually report outside trading hours, so a large overnight return (typical for growth stocks) can come from a few earnings-day gaps rather than a steady premium.
- Today's **unfinished US session** is dropped automatically (a live price is not a close); the date is reported as `partial_session_dropped`.
- Companies with different history lengths: for comparisons use `--common-start` or the annualized columns. The group tests use the *average daily* overnight return, which does not depend on history length.
- Beta is the slope of daily close-to-close returns on SPY's over the sample window - not Yahoo's published 5-year monthly beta.

## Testing

Tests run fully offline on synthetic data (yfinance and Wikipedia are mocked). They cover the core ON/ID math (returns, log-additivity, cumulation, annualization), beta, the different yfinance column layouts, group tests, the quick/`--json`/full-scan CLI paths and error handling, plus a lint of the skill package.

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Use it from an AI agent (Agent Skill)

The folder [`skill/`](skill/) packages this tool as a `SKILL.md` skill for agents such as Claude Code, Codex CLI or Cursor: the agent runs `--tickers ... --json` and explains the result, with the caveats above built in. The script stays the single source of truth; build the ZIP with:

```bash
python skill/build_skill.py      # -> dist/sp500-overnight-vs-intraday.zip
```

The build lints the package (name format, script under 20 KB, no risky patterns). Listing copy for marketplaces is in [`skill/LISTING.md`](skill/LISTING.md).

## License

Released under the **MIT** license — you may freely use, modify and distribute it, keeping the copyright notice. See the [LICENSE](LICENSE) file.

## Disclaimer

Educational/analytical project. **This is not investment advice.** Data comes from Yahoo Finance and is subject to their terms of use — it is not redistributed in this repository.
