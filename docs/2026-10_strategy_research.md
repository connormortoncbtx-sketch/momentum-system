# Strategy research: does the weekly momentum basket have an edge?

October 2026. Full tables: `research/results/report.md`. Reproduce:
`python research/run_experiments.py --data <research-data branch checkout>`.

## Setup

- **Data:** Alpaca SIP daily bars, Jan 2016 – Oct 2, 2026, condensed to weekly.
  14,406 symbols including 2,040 delisted names. 417 delisted names were
  unavailable, so a little survivorship bias remains.
- **Universe each week:**
  - Unadjusted price ≥ $5 and average daily dollar volume ≥ $2M.
  - At least one year of history.
  - No funds or ETFs.
  - No reverse split in the past 52 weeks.
  - About 2,400 stocks.
- **Test period:** 508 weeks, Jan 2017 – Sep 2026.
- **Default trade:** Friday signal, buy Monday close, sell Friday close,
  top-10 equal weight, 15 bps per side.
- **Signals:** only price-based. Catalyst, fundamental and sentiment
  signals have no history to test against. The live momentum score is
  reproduced from its RS, trend and 52-week-high parts, covering 80% of
  its weight.

## Benchmarks

| | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| SPY, held | 15.0% | 17.5% | 0.89 | −33% |
| SPY, Mon close → Fri close only | 12.0% | 14.9% | 0.83 | −18% |
| Equal-weight universe, same timing | 9.8% | 18.2% | 0.61 | −23% |

## Findings

**1. The signals are real but tiny.** The 12-1 momentum, live-RS and
52-week-high signals rank next week's returns with weekly IC of
0.015–0.02 (t ≈ 1.7–2.5). Last week's and last month's winners
underperform, which is the short-term reversal effect.

**2. Weekly full turnover destroys the edge.** The live-momentum proxy
(top-10, weekly) by trading cost per side:

| Cost per side | CAGR |
|---|---|
| 5 bps | +1.2% |
| 15 bps | −8.8% |
| 30 bps | −22.0% |

Selling everything Friday and rebuying Monday costs about 1 point of
annual return per bp of cost, before any signal. Sitting in cash from
Friday close to Monday close also gives up about 3%/yr (SPY: 15.0% held
vs 12.0% with the system's timing).

**3. A top-10 basket is too concentrated.** The highest-momentum names
run 40–45% annualized volatility with drawdowns of 70–80%. Volatility
drag turns a slightly positive average week into a negative compounded
return.

**4. Holding longer helps the most.** Live-momentum proxy at 15 bps:

| Rebalance every | CAGR | Sharpe | Max DD |
|---|---|---|---|
| 1 week | −9.0% | −0.22 | −72% |
| **4 weeks** | **+14.6%** | **0.62** | **−42%** |
| 13 weeks | +5.3% | 0.31 | −52% |

Keeping any holding that stays in the top 30 instead of selling
everything weekly: 12-1 momentum +8.2% at 15 bps (Sharpe 0.41).

**5. Some fixes that seem obvious don't help:**
- 7% hard stop: worse (−10.4% vs −8.8%).
- Going to cash when SPY is below its 40-week average: no meaningful change.
- $1 minimum price (the live setting) instead of $5: worse (−17.2%).
- Top-30 instead of top-10: about the same CAGR, lower volatility.

**6. Last 12 months:** every momentum variant collapsed (−50% to −63%)
except the blended live proxy (about flat). This was a historically bad
year for momentum, not a defect peculiar to your system.

## What this changes in the Oct 2026 post-mortem

- The after-hours timing bug **did not hurt**. Tuesday-open-in,
  Monday-open-out did slightly *better* than the designed timing
  (−4.1% vs −8.8% CAGR), because it holds through the weekend.
- The missing stops **did not hurt on average** (finding 5).
- The losses came from the strategy itself: weekly full turnover, a
  10-name basket, and a momentum-hostile year.

## Bottom line

No variant tested beats simply holding SPY on a risk-adjusted basis over
2017–2026. The best one (monthly rebalance, +14.6%, Sharpe 0.62) roughly
matches SPY's return with about 1.6× the volatility and a deeper drawdown.
It was also the best of roughly 40 variants tried, so its result is
optimistic.

## Caveats

- Costs are a flat assumption, not a fill model.
- About 417 delisted names are missing.
- Catalyst, fundamental and sentiment signals are untested (their 17 live
  weeks showed IC ≈ 0).
- Monday-close fills are assumed to be at the closing print.
