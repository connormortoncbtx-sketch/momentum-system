# Exact-time scheduler setup

**Why:** GitHub's `schedule:` trigger is best-effort. From May–Oct 2026 the
2:45 PM CT entry/exit jobs ran 1–5 hours late; 43 of 45 entries fired after
the close. Timing-critical workflows are now triggered by an external
scheduler calling GitHub's `workflow_dispatch` API at the exact minute.

## 1. Create a GitHub token

GitHub → Settings → Developer settings → Fine-grained personal access tokens → Generate.

- Repository access: **Only** `momentum-system`
- Permissions: **Actions → Read and write** (nothing else)
- Expiration: as long as you're comfortable with. **Put the renewal date on
  your calendar** — an expired token stops all trading triggers.

## 2. Create the jobs (cron-job.org or any scheduler with timezone + headers)

Set every job's timezone to **America/Chicago** (handles DST — no more
CDT/CST duplicate crons).

Every job uses the same request, changing only the workflow file name:

```
POST https://api.github.com/repos/connormortoncbtx-sketch/momentum-system/actions/workflows/<WORKFLOW_FILE>/dispatches

Authorization: Bearer <TOKEN>
Accept: application/vnd.github+json
X-GitHub-Api-Version: 2022-11-28
Content-Type: application/json

{"ref":"main"}
```

A successful dispatch returns **HTTP 204**. Turn on failure notifications in
the scheduler so a non-204 emails you.

| Workflow file | Days | Time (CT) | Notes |
|---|---|---|---|
| `premarket_monitor.yml` | Mon, Tue | 6:00 AM | Script's 6 AM guard now passes |
| `alpaca_monitor.yml` | Mon–Fri | 8:30 AM | Duration auto-computed |
| `alpaca_monitor.yml` | Mon–Fri | 11:15 AM | |
| `alpaca_monitor.yml` | Mon–Fri | 2:00 PM | |
| `alpaca_entry.yml` | Mon, Tue | 2:45 PM | Script picks Mon, or Tue if Mon is a holiday |
| `alpaca_place_stops.yml` | Mon, Tue | 3:10 PM | |
| `alpaca_exit.yml` | Thu, Fri | 2:45 PM | Script picks Fri, or Thu if Fri is a holiday |

`alpaca_exit.yml` and `alpaca_place_stops.yml` keep their GitHub crons as a
**backstop**. If the dispatcher misses an exit, the late cron still sells
(queued for next open) and alerts you. If the dispatcher worked, the late
cron finds no positions and does nothing.

## 3. Test

1. Dispatch `alpaca_monitor.yml` once by hand from the scheduler → confirm
   a run appears in the Actions tab within ~1 minute.
2. Outside market hours, dispatch `alpaca_entry.yml` → it should **refuse**
   ("outside pre-close window") and send a phone alert. That proves the
   clock guard works without placing orders.

## Safety rails now in code (`automation/alpaca_trader.py`)

- Entry submits only with 4–45 min to close (Alpaca market clock).
- Entry no-ops if BUY orders are already working (duplicate trigger).
- State is saved only if at least one order was accepted — a failed run
  can no longer wipe position metadata.
- `place_stops` alerts loudly if Alpaca holds positions the state doesn't
  know about (previously a silent skip that left positions unstopped).
- Skipped entries (stray positions, low cash) now alert instead of only logging.
