# Followup Patches — first-cycle issues

Three small fixes for issues surfaced by the first live Friday run after Tiers 1-4.

## Files

```
followup_patches/
├── automation/
│   ├── collect_returns.py     composite_rank dtype coercion
│   └── retrain.py             partial sub-signal tolerance
├── .github/workflows/
│   └── friday_learning.yml    Alpaca SDK install added
└── diffs/                     unified diffs vs shipped Tier 3/4 versions
```

## What each patch does

### `collect_returns.py` — composite_rank dtype fix

**Symptom**: `TypeError: Column 'composite_rank' has dtype object, cannot use method 'nsmallest' with this dtype`

**Cause**: When pd.read_csv reads `performance_log.csv`, rows from before Tier 2 have NaN for `composite_rank` while new rows have integers. pandas stores the mixed column as `object` dtype, which `nsmallest` can't sort.

**Fix**: Coerce to numeric with `errors="coerce"` before `nsmallest`, then drop rows that couldn't be ranked.

**Impact**: The crash happened *after* the successful data write, so this week's perf_log rows are fine. Fix prevents the crash on future runs.

### `retrain.py` — partial sub-signal tolerance

**Symptom**: `Training data: 0 rows after masking (85,995 dropped for missing features/label)`

**Cause**: The Tier 4 mask-before-fillna logic required ALL 18 sub-signal features to be non-null for a row to be kept. In practice, many rows have 1-2 sub-signals missing (e.g. insider data unavailable for a given ticker that week) so requiring 100% coverage zeroed out the training set.

**Fix**: Require ≥60% of sub-signal columns to be present in perf_log, and ≥70% of features to be non-null per row. Apply fillna(0.5) only to the minority NaN in kept rows, after masking. This is still meaningfully stricter than the original pre-Tier-4 behavior (which treated 100% NaN rows as synthetic-neutral training data) while being practical for real-world signal coverage.

**Impact**: Retrain will start producing usable models once you have ~5000 rows with adequate sub-signal coverage. Should take 2-3 weeks to accumulate given current fill rate.

### `friday_learning.yml` — Alpaca SDK install

**Symptom**: `WARNING  Could not retrieve Alpaca fills: No module named 'alpaca_trade_api'`

**Cause**: `execution_tracker.py` imports `alpaca_trade_api` but the workflow job only runs `pip install -r requirements.txt`, which doesn't include the Alpaca SDK. The entry/exit workflows install it explicitly; the learning loop didn't.

**Fix**: Add explicit `pip install "alpaca-trade-api>=3.0.0" "aiohttp>=3.9.0"` step after the requirements.txt install, mirroring the pattern in `alpaca_entry.yml`.

**Impact**: Once Monday's paper trades fill and Friday's exit completes, execution_tracker will successfully query Alpaca for fills and record slippage/ADV metrics. Without this, every Friday would have silently returned zero execution data even with real fills happening.

## Deploy

Drop the three files in place over the current repo. No data migration needed. All three changes are additive/corrective and don't affect data already on disk.

## Verification done

- All three files pass syntax check (py_compile for Python, yaml.safe_load for YAML)
- No test run was possible since the failure requires Friday learning loop conditions
- The retrain change is the only one that could produce a behavior change on the next run; the other two fix error paths

## What to watch next Friday

- `collect_returns`: should complete without the `nsmallest` crash, log the entry-day comparison for top-20% composite picks
- `retrain`: should log a row count > 0 after masking (exact count depends on how many sub-signals are populated in new rows)
- `execution_tracker`: should log fill count — though still expected to be zero until Monday entries actually fill
