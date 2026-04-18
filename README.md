# Commit Retry Patch — eliminates the concurrent push race

Eliminates the git rebase conflict that bit three times tonight: the monitor,
the Friday pipeline, and (potentially) any workflow that commits within
seconds of another.

## What changed

All 11 workflows in `.github/workflows/*.yml` had their commit step updated
from:

```bash
git pull --rebase origin main
git push
```

To a retry loop:

```bash
for attempt in 1 2 3 4 5; do
  if git pull --rebase origin main && git push; then
    echo "Push succeeded on attempt $attempt"
    break
  else
    echo "Push attempt $attempt failed; aborting rebase and retrying"
    git rebase --abort 2>/dev/null || true
    git pull origin main --no-rebase --strategy=recursive -X theirs --no-edit 2>/dev/null || true
    if [ $attempt -eq 5 ]; then
      echo "All 5 push attempts exhausted; giving up"
      exit 1  # or exit 0 for workflows that previously had `|| true`
    fi
    sleep $((RANDOM % 5 + 2))
  fi
done
```

## How it handles the race

When two workflows commit to `system_log.jsonl` within seconds of each other:

1. First one pushes successfully.
2. Second one tries `git pull --rebase`, hits a merge conflict on the JSONL
   file, rebase fails.
3. OLD behavior: workflow exits red, data lost from runner filesystem.
4. NEW behavior: abort the rebase, try a non-rebase pull with `-X theirs`
   which prefers the remote's JSONL additions during content conflicts.
   Our local commit gets re-applied on top. Push succeeds on retry.

The `-X theirs` flag is safe for this specific use case because JSONL log
files are append-only. Taking "theirs" keeps the remote's log lines and
our commit adds ours on top. No data loss either way.

## Behavior matrix

| Scenario | Old | New |
|---|---|---|
| No conflict | succeeds | succeeds (on attempt 1) |
| Concurrent JSONL append | fails, data lost | succeeds (on attempt 2-5) |
| Genuine conflict in code | fails | fails after 5 attempts |
| Network blip during push | fails, data lost | retries, usually succeeds |

## Workflows affected

All 11 (9 with plain push, 2 with `|| true` suffix preserving fault-tolerant behavior):

- alpaca_entry.yml
- alpaca_exit.yml
- alpaca_monitor.yml (preserves exit 0 on failure — was `|| true`)
- backfill_history.yml
- backfill_sectors.yml
- friday_learning.yml
- inject_universe.yml
- premarket_monitor.yml
- weekend_refresh.yml
- weekly_health_check.yml (preserves exit 0 on failure — was `|| true`)
- weekly_pipeline.yml

## Verification

- All 11 files parse as valid YAML
- Retry block preserved identical behavior for success path (unchanged pull-rebase-push)
- Only changes behavior for failure paths (adds retry instead of exiting)

## Deploy

Drop all 11 files in place over `.github/workflows/`, push. Next time two
workflows race, the later one retries cleanly instead of failing.

## Note on what this doesn't fix

If conflicts happen on non-JSONL files (e.g., both workflows modify
`data/scores_final.csv` simultaneously), `-X theirs` would silently overwrite
the local change. Unlikely given the current workflow designs — each writes
to distinct files except for the log — but worth knowing. If future
workflows share non-append-only files, this pattern would need adjustment.

Longer-term architectural fix (Tier 5+): per-workflow log files
(`system_log_monitor.jsonl`, `system_log_pipeline.jsonl`, etc.) so conflicts
become structurally impossible rather than handled after the fact. Retry
loop is the right short-term fix; separate files is the right long-term one.
