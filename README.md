# Tier 3 — Observability & Workflow Resilience

Stops the silent failures. After this tier, every weekly module emits structured
log events, every capital-at-risk event fires a push notification, and the Friday
learning loop survives partial failures without cascading.

## What's in this bundle

```
tier3_observability/
├── automation/          12 Python modules (11 changed + system_logger fix)
├── run_pipeline.py      orchestrator with per-stage observability
├── .github/workflows/   8 workflow YAMLs (concurrency + NTFY_CHANNEL)
├── diffs/               unified diffs vs original for every file
└── README.md            this file
```

## Before you deploy: add `NTFY_CHANNEL` to GitHub Secrets

All the new `notify_*` calls read `os.environ["NTFY_CHANNEL"]`. If unset they
silently no-op (per `notifier.py` lines 45–48) — the system still works, you
just won't see alerts.

1. Pick an unguessable channel name, e.g. `momentum-alpha-7k9f2q`.
2. Add it as a repo secret named `NTFY_CHANNEL`.
3. Install the Ntfy app on your phone, subscribe to that channel name on `ntfy.sh`.
4. The workflow YAMLs in this bundle already pipe `NTFY_CHANNEL` into every
   relevant job's env block — nothing else to configure.

Test with: `python -m automation.notifier` from a machine with the env var set.

## Deploy order

1. Drop Python files over the top of the repo (preserves paths).
2. Drop workflow YAMLs over `.github/workflows/`.
3. Add the `NTFY_CHANNEL` secret.
4. Push. No data migration required — `system_log.jsonl` creates itself on first
   write.

## What changed, by theme

### Bug fixes (previously silent or incorrect)

- **system_logger** — `str(LogStatus.SUCCESS)` was serializing as
  `"LogStatus.SUCCESS"` (Python 3.11+ Enum behavior). Now uses `.value` so logs
  contain `{"status": "success"}` as designed. Affects every downstream
  consumer (`read_logs`, `format_logs_for_review`, `health_check`).
- **analyze_winners H1** — `get_last_completed_week` returned a stringified
  Timestamp (`"2026-04-03 00:00:00"`) but `log_df["week_of"].astype(str)`
  returned just `"2026-04-03"`. The two never matched, so winners lookup
  silently returned 0 every run. Fixed with `strftime("%Y-%m-%d")`. Verified
  against current perf_log: now returns 10 winners for 2026-03-27.
- **analyze_winners H2** — required `forward_return_1w.notna()` only, which
  passed on backfill-heavy logs where alpha_score was never populated. Now
  requires both. MIN_WEEKS now counts live weeks, not total weeks.
- **health_check H1** — empty-log-file path was a silent self-log. Now fires
  HIGH-priority push notify with "OBSERVABILITY GAP" title. This is the single
  alert that would have caught every prior silent failure mode.

### Observability wiring (new log_event + notify calls)

Every `run()` in every automation module now follows the pattern:
- Starts with `log_event(<module>, INFO, "Starting...")`
- Every non-success return path emits `log_event` with reason
- Notifies (via `notify_alert` or `notify_error`) for events the user needs to
  see — not for routine skips
- Top-level try/except wraps the body; unhandled exceptions log ERROR + notify
  + re-raise so the workflow still fails loudly
- Success path emits `log_event` with metrics dict, plus `notify_success` for
  capital-at-risk completions

Notifications are asymmetric by design: a quiet monitor run emits no push, a
Phase 2 activation fires immediately. Scaling threshold breaches (from
execution_tracker) now surface as alerts instead of buried workflow logs.

### Learning-loop resilience

`friday_learning.yml` now has `continue-on-error: true` on each of the five
steps (collect_returns, execution_tracker, analyze_winners, self_refine,
retrain). Previously a bad collect_returns meant no self_refine, no retrain,
no winners analysis for the week. Now each runs independently; its own log
events surface what broke. The commit step still runs and captures whatever
partial outputs each produced.

### Workflow concurrency

Every scheduled workflow now has a `concurrency` key. Prevents:
- Manual `workflow_dispatch` overlap with scheduled crons (double entry = bad)
- DST dual-crons from both firing on the boundary day
- Back-to-back pipeline runs corrupting `scores_final.csv`

`alpaca_monitor.yml` uses `group: alpaca-monitor-${{ github.event.schedule ||
github.run_id }}` so the three overlapping intraday jobs still coexist by
design, while same-schedule duplicates are blocked.

### Capital-at-risk observability (new push notifications)

| Event                                   | Priority |
|-----------------------------------------|----------|
| Entry: all orders filled                | SUCCESS  |
| Entry: any order failed                 | ALERT    |
| Exit: all closed + weekly P&L           | SUCCESS  |
| Exit: any close failed                  | ALERT    |
| Hard stop placed (post-fill)            | log only |
| Hard stop submission FAILED             | ERROR    |
| Phase 2 activated (gain locked in)      | SUCCESS  |
| Phase 2 upgrade failed                  | ERROR    |
| Pipeline stage failed                   | ERROR    |
| collect_returns skipped by 5-day guard  | ALERT    |
| self_refine rejected proposed weights   | ALERT    |
| Execution_tracker scaling breach        | ALERT    |
| Pre-market final check: ≥5% gaps        | ALERT    |
| Health check: empty log file            | HIGH     |
| Monitor poll error rate >10%            | ALERT    |

### Commit-step hardening (H3 fix)

`weekly_pipeline.yml` and `weekend_refresh.yml` had commit messages that
embedded `$(python -c "...")` interpolations. If `scores_final.csv` was
corrupted, the Python one-liner crashed, the commit command aborted, and
whatever partial state the pipeline produced was never committed. Now every
interpolation is wrapped with `|| echo "unknown"` so the commit step always
completes.

## NOT in this tier (deferred to Tier 4/5)

- Stage 1 W/U/R lookback length-gate mismatch
- Stage 2 SMA200 LOOKBACK_DAYS bump
- execution_tracker H1 (double-percent) and H2 (fill-date window)
- retrain C1/C2 (RETRAIN_FEATURES masking)
- trade_log.html HTML-escape + PAT scope reduction
- tz_utils year-boundary NYE observance
- DST dual-cron guards (mostly mitigated by in-script hour checks and new
  concurrency keys; formal guard step is Tier 5 polish)
- README rewrite, dead dependency cleanup, file deduplication

## Line count summary

```
automation/system_logger.py                         +5       -1
automation/health_check.py                         +13       -0
automation/self_refine.py                          +29       -0
automation/collect_returns.py                     +128      -72
automation/retrain.py                              +86      -51
automation/analyze_winners.py                     +167     -105
automation/execution_tracker.py                   +119      -79
automation/weekend_refresh.py                     +167      -87
automation/alpaca_trader.py                       +150      -45
automation/alpaca_monitor.py                      +171       -4
automation/premarket_monitor.py                   +183     -130
run_pipeline.py                                    +42       -3
.github/workflows/friday_learning.yml              +20       -0
.github/workflows/weekly_pipeline.yml              +17       -3
.github/workflows/weekend_refresh.yml              +17       -2
.github/workflows/alpaca_entry.yml                  +9       -0
.github/workflows/alpaca_exit.yml                   +9       -0
.github/workflows/alpaca_monitor.yml               +17       -6
.github/workflows/premarket_monitor.yml            +10       -0
.github/workflows/weekly_health_check.yml           +6       -0
```

Many of the `+N -M` pairs where added ≈ removed reflect rewrites-in-place
(e.g. `run()` wrapped in try/except) rather than pure additions.
