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

---

# Round 2 (same day): restructured core + new data

**Rule:** every design was chosen on 2017–2023 data and scored once on
2024–2026 (insiders: through Jun 2026, when the SEC data ends).

| Family (best of grid) | Designs tried | Dev Sharpe | Dev SPY | Holdout CAGR | Holdout SPY |
|---|---|---|---|---|---|
| Momentum v3 (RS, 20 names, monthly, inv-vol) | 144 | 0.74 | 0.76 | 8.9% | 20.6% |
| Earnings drift (SEC 8-K 2.02, gap-up reactions, 13-wk window) | 144 | 0.77 | 0.76 | 19.1% | 20.6% |
| Insider purchases (Form 4 $/volume + 12-1 momentum) | 120 | 0.66 | 0.76 | 9.7% | 21.3% |

**Event studies (dev years):**
- Earnings drift: the best-reaction fifth trails SPY by 0.7% over 12 weeks;
  the worst fifth trails by 0.8%. There is no meaningful spread.
- Insiders: cluster buys show +1.2–1.6% at 12 weeks, but negative at 8 and
  26 weeks. The sign flips, so this is noise (no t-stat above 2).

**Company size explains part of the gap to SPY.** The equal-weight
universe trailed SPY by 3.7%/yr in dev and 7.2%/yr in the holdout
(mega-caps dominated the decade). Measured against the universe the
strategies actually pick from, holdout alphas are momentum −7.0%/yr,
earnings drift +4.6%/yr and insiders −6.7%/yr. None of the t-stats
exceeds 1, so there is no detectable stock-picking skill.

Data added to the `research-data` branch:
- `earnings_events.parquet`: 126k events, 4,165 companies.
- `insider_buys.parquet`: 293k purchases, 8,585 tickers.

---

# Round 3: large caps, simulator fix, AI news test

## Simulator fix (applies to every portfolio search)

Holdings whose next-week price was missing had been booked at −50%.
Checking showed these were almost all **acquisitions** (Acceleron, Pivotal,
Genesee & Wyoming, Ellie Mae and others), which stop trading near the deal
price. They now book 0%. Dev results rose (momentum best dev Sharpe 0.74 →
0.85), but every pre-chosen design and its holdout score was unchanged.

## Large caps only (500 most-traded names each week)

| Family | Dev Sharpe (SPY 0.76) | Holdout CAGR | SPY | Beta | Alpha vs SPY, beta-adjusted |
|---|---|---|---|---|---|
| Momentum (live proxy, 30 names, monthly) | 0.81 | **33.2%** | 20.6% | 1.46 | +5.2%/yr (t 0.34) |
| Insider $ + 12-1 momentum (20 names) | 0.81 | 24.2% | 21.3% | 1.25 | −1.1%/yr (t −0.12) |
| Earnings drift (gap-up reactions) | 0.68 | 15.8% | 20.6% | 1.08 | −3.9%/yr (t −0.59) |

- **Large-cap momentum** beat SPY in raw return every holdout year (56.5% /
  17.8% / 18.7% vs 27.1% / 16.9% / 12.2%). However, SPY held at the same
  1.46× exposure would have made 30.3%. It is mostly amplified market
  exposure. Gains were spread across rotating leaders (APP, PLTR, MSTR in
  2024; BE, KGC, HOOD in 2025; WDC, LITE, CIEN in 2026), not one lucky name.
- **Large-cap insider pick:** the first pick (officer buys) was investable
  in only 199 of 365 dev weeks. A rule requiring ≥90% of weeks invested
  was added before any holdout return existed, and the pick was redone.
- **Large-cap event studies:** no earnings drift. Officer purchases were
  followed by *under*performance (8 weeks, t = −2.3).

## AI news classification: partial

Sample: 2,200 events chosen at random in advance. Claude Sonnet 5.5 read
anonymized headlines and summaries.

- **Status:** 1,458 events labeled for $5.20 before the Anthropic account ran
  out of credit. Dev is complete; 169 of 500 holdout and 0 of 300
  post-cutoff events are done. The run resumes from where it stopped.
- **Dev results** (can be inflated by the model remembering outcomes):
  - H1, the fundamental score: +0.61% per point at 8 weeks (t 1.89).
  - H2, "under-reacted" plus a strong score: −2.2% vs −0.35% for the rest.
    This is the wrong direction.
  - H3, raised vs lowered guidance: +0.2% vs −1.9%.
- **Partial holdout** (n = 168): the H1 coefficient flips to −1.33 (t −1.31).
- Not conclusive until the post-cutoff group runs.

## Holdout looks so far

Six designs have been scored on the holdout (3 families × 2 universes). With
that many looks, one beating SPY on raw return by luck is expected. None
shows beta-adjusted alpha with t > 1.
