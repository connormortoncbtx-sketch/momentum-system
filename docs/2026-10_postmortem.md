# Post-mortem: unattended run, May – Oct 2026

Written 2026-10-06. Source: GitHub Actions run history, `data/system_log.jsonl`,
`alpaca_state.json` git history, `performance_log.csv`, `execution_log.csv`.

## Account

~$100.6k (Jun 18) → ~$68.8k (Oct 5 week-open). Worst weeks: Jul 2 (−7.1%),
Jul 17 (−6.1%), Aug 21 (−5.8%), Sep 11 (−13.0%).

## What broke

**1. Every "close" trade ran after the close.** GitHub `schedule:` drift was
1–5 hours (median first-run lateness ~60 min May–Jul, ~140+ min Sep–Oct).
Only 2 of ~45 entry days (Aug 18, Aug 21 exit) ran before 3:00 PM CT.
Market DAY orders queued for the next open: entries filled Tuesday open,
exits filled **Monday** open. Every basket was held over the weekend.

**2. Stops were off from Jun 22 onward.** The duplicate CST-offset cron fired
while the first run's orders were still queued (not yet positions). It saw
zero positions, reset `alpaca_state.json`, failed all 20 orders on buying
power, and saved empty state. `place_stops` then skipped ("state has no
positions") and the monitor had nothing to manage. Mid-week state showed 0
tracked positions in every sampled week from Jun 24 to Sep 30.

**3. Two weeks untraded.** Sep 14 and Sep 21 entries were silently skipped —
a stranded CLBK position tripped the "already holding positions" guard. It
was finally sold in the Sep 25 exit.

**4. Premarket monitor never ran** — its 6 AM CT guard rejected every late run.

**5. LLM credits exhausted ~Sep 4.** Synthesis, self_refine, analyze_winners
and health_check failed. Synthesis falls back to neutral silently, so
ranking ran rule-based only for ~5 weeks. Not a driver of the drawdown —
most losses predate it.

## What actually drove the losses

The ranking. Using `performance_log.csv` (Mon close → Fri close, 12 weeks
from late June):

| Bucket | Compounded |
|---|---|
| Top 10 by composite_rank | −23.6% |
| Top 50 | −15.3% |
| Universe median | +0.9% |
| Bottom half | −0.8% |

Mean weekly Spearman IC ≈ 0.02, positive in 5 of 12 weeks. A rough
counterfactual on the actual baskets (designed timing + 7% hard stop, using
weekly lows) was about as negative as what happened. The plumbing failures
removed the risk controls and added weekend gap risk, but the designed
strategy would also have lost money in this stretch.

## Fixes shipped 2026-10-06

- Entry/monitor/premarket moved to an exact-time Netlify scheduled-function dispatcher
  (`docs/scheduler_setup.md`); exit and place_stops keep crons as backstop.
- Market-clock guard on entry/exit; entry idempotent on working BUY orders;
  exit idempotent on working SELL orders.
- State only written if an order is accepted.
- Loud alerts for: unstoppable positions, stray-position entry blocks,
  low-cash skips, LLM pass producing zero results.

## Open questions (not fixed — need a decision)

- Signal decay since late June. Re-check IC by regime before trusting it
  with real timing restored; consider pausing entries until a few weeks of
  shadow results look positive.
- Whether 7% hard stops help or hurt this basket profile (rough sim was mixed).
