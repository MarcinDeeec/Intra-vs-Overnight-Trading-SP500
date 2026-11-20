# Methodology and limitations

## Definitions
For each session `t` (daily, split/dividend-adjusted prices from Yahoo Finance, `auto_adjust=True`):

```
ON_t = Open_t / Close_{t-1} - 1     overnight (the gap between sessions)
ID_t = Close_t / Open_t   - 1       intraday (the session itself)
```

`(1 + ON_t)(1 + ID_t) = 1 + close-to-close return`, so the two parts always add up (in logs: exactly).
Weekends and holidays are part of the overnight gap.

## Aggregates (per company)
- `cum_ON`, `cum_ID`: geometric cumulation, `prod(1 + r) - 1`. Computed as `expm1(sum(log1p(r)))`.
- `diff_log` = `sum(log ON) - sum(log ID)`: positive means overnight dominates over the whole sample. It grows
  with history length, so it is kept in the CSV but not used for rankings.
- `ann_ON_pct`, `ann_ID_pct`: mean daily log return x 252, converted back to a percentage.
  Use these (not the cumulative values) to compare companies with different history lengths.
- `ann_diff_log`: `diff_log` annualized the same way. `dominant` is `"ON"` / `"ID"` when
  `|ann_diff_log| >= 0.02` (2 percentage points per year, log), otherwise `"similar"`. The 2 pp band is a
  convenience threshold, not a statistical test.
- `mean_ON_bps`, `mean_ID_bps`: average simple daily return in basis points.
- `vol_ON`, `vol_ID`, `vol_close2close`: annualized standard deviation (x sqrt(252)).
- `beta`: slope of the company's daily close-to-close returns on SPY's, over the same window.
  Needs at least 60 overlapping days. This is *not* Yahoo's published 5-year monthly beta.
- Companies with fewer than 20 sessions are skipped.

## Rankings (full scan)
All three use `ann_diff_log` (the annualized ON-minus-ID advantage, same measure as the `dominant` label):
`on_dominates` = top N; `id_dominates` = bottom N; `similar` = N smallest `|ann_diff_log|`.
If no group test could run, `group_tests_note` explains why.

## Group tests (full scan)
Compares the mean daily overnight return (`mean_ON_bps`) of a group with the rest of the index using
Welch's t-test and the Mann-Whitney U test. Groups: Information Technology sector; bottom market-cap
quartile (needs market caps, i.e. no `--skip-meta`); top beta quartile. A group needs at least 5 members.
These are exploratory cross-sectional comparisons: companies are correlated, three groups are tested,
and no multiple-testing correction is applied.

## Limitations
- **Survivorship bias**: the full scan uses the *current* index constituents, so companies that left the
  index are missing and results look better than a real investor would have seen.
- **Gross returns**: no commissions, bid-ask spreads, slippage, borrow costs or taxes. A strategy that
  buys every close and sells every open trades twice a day and is dominated by costs in practice.
- **Data quality**: Yahoo's daily `Open` is not guaranteed to be the official opening-auction price, and
  adjusted history can be revised. Different vendors give slightly different ON/ID splits.
- **History**: results depend on the chosen window; the anomaly is not stable across periods or markets.
- **Earnings and news gaps**: companies usually report results outside trading hours, so a large ON for
  growth stocks can be driven by a handful of earnings-day gaps rather than a steady overnight premium.
- **Unfinished session**: while the US market is open (before 16:30 ET) Yahoo's bar for today holds a live
  price, not a close. The script drops it automatically and reports the date in `partial_session_dropped`.
- **Not advice**: educational analysis only.
