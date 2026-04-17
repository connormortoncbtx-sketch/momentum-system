# Tier 4 — Signal quality & execution correctness

Fixes behavioral bugs that have materially corrupted every signal the system
has produced. Two of these (regime SMA200, 12-1 month momentum) are
**deploy-ASAP priority** — they've been silently distorting alpha scores since
day one.

## What's in this bundle

```
tier4_signal_quality/
├── pipeline/
│   ├── 02_regime.py              SMA200 lookback + strict above_sma
│   └── signals/
│       └── momentum.py           fetch_history calendar→trading days fix
├── automation/
│   ├── collect_returns.py        persist 14 sub-signals to perf_log
│   ├── retrain.py                C1 feature selection + C2 mask-before-fillna
│   ├── tz_utils.py               NYE year-boundary observance
│   └── execution_tracker.py      H1 percent display + H2 fill-window
├── docs/
│   └── trade_log.html            H1 XSS escape on user-typed fields
├── diffs/                        unified diffs vs original for every file
└── README.md                     this file
```

## Critical-priority fixes

### pipeline/02_regime.py — SMA200 unreachable

`LOOKBACK_DAYS` was 90 calendar days → ~75 trading days fetched. The
`score_trend()` function calls `above_sma(spy, 200)`, which needs 200 trading
days. The function's fallback was:

```python
if len(series) < n:
    return True
```

So `above_sma(spy, 200)` was returning `True` unconditionally on every run,
adding a phantom `+0.15` to the trend score. Every regime classification
since inception has been biased risk-on.

**Fix**: `LOOKBACK_DAYS = 300` (≥215 trading days after weekend/holiday buffer)
and `above_sma` now raises `ValueError` on insufficient data — optimistic
fallbacks have no place in a function whose output maps to real trading
decisions.

**Magnitude**: every regime composite score has been ~0.15 too positive.
On the 5-bucket regime scale (-1 to +1 mapped to risk_off_severe →
trending_mixed → risk_on), that's enough to flip a borderline
`choppy_neutral` into `trending_mixed`, or a `trending_mixed` into `risk_on`.
Each regime has different weight multipliers in `config/weights.json`, so the
corruption cascaded to every ticker's alpha score.

### pipeline/signals/momentum.py — 12-1 month momentum always zero

`LOOKBACK = 252` was commented as trading days but `fetch_history` did
`timedelta(days=LOOKBACK + 30)` — that's calendar days. 282 calendar days ≈
201 trading days returned. The `rs_return` function then had:

```python
r_12_1 = (close.iloc[-21] / close.iloc[-252]) - 1 if len(close) >= 252 else 0.0
```

The `len(close) >= 252` branch was **unreachable**. Every ticker got
`r_12_1 = 0.0`.

**Fix**: Explicitly convert trading-day target to calendar window with a
`7/5` ratio plus 30-day buffer = 382 calendar days. Now fetches ~273 trading
days, the 12-1 branch runs, and the 40%-weighted momentum term is alive again.

**Magnitude**: `rs_return = 0.40 × r_12_1 + 0.35 × r_6_1 + 0.25 × r_3_1`. With
`r_12_1` always zero, 40% of the signal was dead weight, effectively
redistributing to `r_6_1` and `r_3_1`. The 12-1 month horizon is the single
strongest momentum predictor in the cross-sectional literature
(Jegadeesh-Titman 1993, Asness et al. 2013). Every pick since inception was
missing this input.

## Retrain sub-system fix (three-layer)

The retrain module had a chain of nested bugs that together meant it could
never have produced a working drop-in model:

1. **Layer 1 — sub-signals never written**: Stage 4 already had the right code
   (`[c for c in df.columns if c.startswith("sig_")]`) but current
   `scores_final.csv` is stale from older runs. Nothing to do here — next
   Friday's pipeline will include sub-signals automatically.

2. **Layer 2 — sub-signals not in perf_log**: `collect_returns.build_rows` only
   pulled 4 coarse composites + 4 `_adj` composites, dropping 14 sub-signals
   that retrain needs. **Fixed** — added `SUB_SIGNAL_COLS` list, now captured
   via `sr.get(col)` into each perf_log row. Missing columns resolve to `None`
   (retrain drops those rows via the new notna mask).

3. **Layer 3 — retrain C1 (unused features) and C2 (vacuous mask)**:
   - **C1**: `build_training_data` used only the 4 coarse composites. Stage 4's
     production model is 18-feature. Retrain would have produced a 4-feature
     model that `pipeline/04_model.py` then silently routes to composite-mode,
     degrading the signal set end-to-end.
   - **C2**: `X = log_df[available].fillna(0.5)` ran **before** `mask =
     y.notna() & X.notna().all(axis=1)`. Post-fillna there are no NaNs, so the
     mask was vacuous. Rows with missing signals trained as synthetic
     "neutral-at-0.5 across every feature" samples — roughly 40k such rows on
     the current perf_log, vs 3k real rows.

   **Fix**: Prefer `RETRAIN_FEATURES` (18-feature set) when all present,
   fall back to 4 composites with warning. Mask **before** fillna. Empty-frame
   guard in `run()` so clean skip when masking drops everything.

**Verification against current data**: post-fix, `build_training_data` returns
2,975 rows (all 4-composite fallback rows where every field is populated).
That's below `MIN_ROWS=5000` so retrain correctly refuses to fire. Pre-fix
behavior would have "trained" on ~43k rows, most of which were synthetic 0.5s
— that model would have been strictly worse than no retrain.

## Other fixes

### automation/tz_utils.py — NYE year-boundary

When Jan 1 of year Y+1 falls on Saturday, NYSE observes the holiday on Friday
Dec 31 of year Y. The old code added that observance to Y+1's holiday set, but
`is_trading_day(date(Y, 12, 31))` asked for Y's set → got a false negative and
would have run the pipeline.

**Fix**: compute next year's Jan 1 observance, and if it rolls back into the
current year, add it to the current year's set too.

**Next trigger**: Jan 1 2028 is Saturday → Dec 31 2027 is the observed
holiday. Without this fix, the pipeline would have tried to run that Friday.
Verified across 2021, 2022, 2023, 2025, 2027.

### automation/execution_tracker.py — two bugs

**H1 — display under-percent**: The signal-quality alert formatted fractions
as percent strings:

```python
# r1 stored as 0.025 (meaning 2.5%), alert displayed:
f"rank 1-3 {r1:.2f}%"  # → "rank 1-3 0.02%"  (wrong!)
```

**Fix**: multiply by 100 in the format string to match every other percent
display in the module.

**H2 — Friday fills excluded**: Alpaca's `list_orders(until=YYYY-MM-DD)` is
exclusive — "before midnight UTC of that date". Friday-close MOC sell orders
fill at 20:00-21:00 UTC on Friday, *after* `until="Friday"` excludes them.
Every exit fill was silently dropped.

**Fix**: `until_inclusive = (parse(week_end) + 1 day).isoformat()` so the
boundary lands on Saturday 00:00 UTC. All Friday fills captured.

### docs/trade_log.html — XSS escape

The trade-log UI renders user-typed fields via `innerHTML`:

```javascript
`<td class="sym-cell">${t.ticker}${badge}</td>`
```

If a user types `<script>alert(1)</script>` as a ticker, it executes. Same
risk for `exit_reason`, `regime`, and `id` (used in onclick). Worse, trades
are committed via a PAT, so a malicious `trades.json` push from any source
with write access would XSS the next page load.

**Fix**: Added `esc()` helper and wrapped every string field (`ticker`,
`entry_date`, `exit_date`, `exit_reason`, `regime`, `id`) before
interpolation. Numeric fields (`toFixed` output) are already safe.

### Not in this tier

**trade_log.html H2 — PAT scope**: The embedded GitHub PAT probably has the
`repo` scope (full write access to all repos). Should be regenerated as a
fine-grained token scoped to `contents:write` on only `momentum-system`. That's
a configuration change in GitHub Settings, not a code edit — documented for
when you regenerate the token next.

## Deploy guidance

If you want to ship only the highest-impact items from this tier:

1. **`pipeline/02_regime.py`** — every regime classification fixes immediately
2. **`pipeline/signals/momentum.py`** — 12-1 momentum signal restored on next
   pipeline run
3. **Everything else is optional for this cycle**, though all are
   improvements.

These two fixes alone will change your next alpha_score for every ticker. You
may want to watch a week or two of picks under the corrected signals before
deploying the retrain changes (which produce different downstream effects as
real sub-signal data accumulates in perf_log).

## Line count summary

```
pipeline/02_regime.py                              +22       -2
pipeline/signals/momentum.py                       +17       -2
automation/collect_returns.py                     +156      -74
automation/retrain.py                             +156      -59
automation/tz_utils.py                             +13       -0
automation/execution_tracker.py                   +140      -82
docs/trade_log.html                                +25       -6
```

## Verification performed

- All 6 Python files: `py_compile` clean
- `docs/trade_log.html`: HTML parser clean
- `pipeline/02_regime.py`: tested `above_sma` raises on short series, returns
  True/False correctly on 215-point series
- `pipeline/signals/momentum.py`: verified `_CAL_LOOKBACK_DAYS = 382` → ~273
  trading days (well above 252 required)
- `automation/tz_utils.py`: verified NYE observance for 2021, 2022, 2023,
  2025, 2027 — all correct
- `automation/retrain.py`: run against live `performance_log.csv` → 2,975
  training rows after masking (was implicitly 40k+ pre-fix), correctly skips
  since below `MIN_ROWS=5000`
