# Exact-time scheduler setup (Netlify)

**Why:** GitHub's `schedule:` trigger is best-effort. From May–Oct 2026 the
2:45 PM CT entry/exit jobs ran 1–5 hours late; 43 of 45 entries fired after
the close. Timing-critical workflows are now triggered by a Netlify scheduled
function (`netlify-dispatcher/`) that calls GitHub's `workflow_dispatch` API
at the exact minute, on America/Chicago time (DST handled in code).

## 1. Create a GitHub token

GitHub → Settings → Developer settings → Fine-grained personal access tokens → Generate.

- Repository access: **Only** `momentum-system`
- Permissions: **Actions → Read and write** (nothing else)
- Expiration: as long as you're comfortable with. **Put the renewal date on
  your calendar.** An expired token makes dispatches fail with 401 (you'll
  get a phone alert), and no trades fire until it's renewed.

## 2. Create the Netlify site

1. Netlify → **Add new project → Import an existing project → GitHub** →
   pick `momentum-system`.
2. Build settings:
   - **Branch to deploy:** `main`
   - **Base directory:** `netlify-dispatcher` ← important
   - Leave build command / publish directory blank (`netlify.toml` sets them).
3. **Environment variables** (Site configuration → Environment variables):
   - `GH_DISPATCH_TOKEN` = the token from step 1 (mark as secret)
   - `NTFY_TOPIC` = same value as your `NTFY_CHANNEL` GitHub secret
4. Deploy. The site is a placeholder page; the work is the `dispatch` function.

**Credits:** `netlify.toml` skips every build unless a file inside
`netlify-dispatcher/` changed, so the bot's data commits don't deploy.
Expect one deploy now (~15 of 300 monthly credits) plus negligible function
compute. Check Usage once after the first trading week to confirm skipped
builds aren't being charged.

## 3. Test

1. Netlify → Logs → Functions → `dispatch`. Within ~15 minutes on a weekday
   between 6 AM and 3:10 PM CT you'll see lines like
   `Chicago Tue 10:45 -> nothing due`. That confirms the schedule is live.
2. First real dispatch: at the next slot (8:30 / 11:15 / 2:00 CT) an
   "Alpaca Intraday Monitor" run should appear in GitHub → Actions within
   about a minute, with event **workflow_dispatch**.
3. Optional clock-guard check (**Monday after 3:00 PM CT only**): Actions →
   Alpaca Entry → Run workflow. It should **refuse** ("outside pre-close
   window") and send a phone alert. On other days the script exits earlier
   with "not entry day", which doesn't exercise the guard. Never run it by
   hand on Mon/Tue between 2:15 and 2:56 PM CT; that is the live window.

## Schedule (edit `SCHEDULE` in `dispatch.mjs` to change)

| Workflow file | Days | Time (CT) | Notes |
|---|---|---|---|
| `premarket_monitor.yml` | Mon, Tue | 6:00 AM | |
| `alpaca_monitor.yml` | Mon–Fri | 8:30 AM, 11:15 AM, 2:00 PM | Duration auto-computed |
| `alpaca_entry.yml` | Mon, Tue | 2:45 PM | Script picks Mon, or Tue if Mon is a holiday |
| `alpaca_place_stops.yml` | Mon, Tue | 3:10 PM | |
| `alpaca_exit.yml` | Thu, Fri | 2:45 PM | Script picks Fri, or Thu if Fri is a holiday |

`alpaca_exit.yml` and `alpaca_place_stops.yml` keep their GitHub crons as a
**backstop**. If the dispatcher misses an exit, the late cron still sells
(queued for next open) and alerts you. If the dispatcher worked, the late
cron finds no positions and does nothing.

**Fallback:** if Netlify ever gives trouble, any scheduler that supports a
timezone and custom headers (e.g. cron-job.org) can do the same job:
`POST https://api.github.com/repos/connormortoncbtx-sketch/momentum-system/actions/workflows/<file>/dispatches`
with headers `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`
and body `{"ref":"main"}`. Success is HTTP 204.

## Safety rails in code (`automation/alpaca_trader.py`)

- Entry submits only with 4–45 min to close (Alpaca market clock).
- Entry no-ops if BUY orders are already working (duplicate trigger).
- State is saved only if at least one order was accepted. A failed run
  can no longer wipe position metadata.
- `place_stops` alerts loudly if Alpaca holds positions the state doesn't
  know about (previously a silent skip that left positions unstopped).
- Skipped entries (stray positions, low cash) now alert instead of only logging.
