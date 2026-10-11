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

---

# Round 4: your manual trades, intraday rules, AI news (complete)

## Your manual trade log (69 trades, Apr 7 – Jun 5, 2026)

- **Verified:** +4.52% average per trade, 65% win rate. These were the live
  model's top ranks (mostly 1–15), not discretionary picks. Compounded over
  7 trading weeks: **+35.8%**, vs SPY +8.4% and the price-only proxy +5.3%.
  The full live model (catalyst, fundamentals, sentiment, LLM) did something
  in that window that price signals alone did not.
- **What followed:** over the next 10 weeks the live model's top 10 lost
  19.9% while SPY gained 0.7% (performance_log). Weekly excess vs SPY across
  all 16 logged weeks: −0.9%/wk (t −0.7).
- **Stops and trails in your own trades:** the same 67 entries held to
  Friday close would have averaged +5.38% per trade, vs +4.72% actual.
  Stops filled at −10.2% on average against 7–10% settings, because stocks
  gapped through them.

## Intraday rules on 5-minute bars (2017–2026)

- **Data:** 2.66M bars, the price-only proxy's top-10 picks each week.
  5,004 paths; matches daily data with correlation 0.9986.
- **Test:** 135 rule sets, covering entry timing, hard stops (5–15%),
  trailing stops (8–25% trigger / 4–10% trail) and selling half at the
  trigger. Gap-through fills use the bar's open. Costs 15 bps per side.

| Rule | Dev per trade | Holdout per trade |
|---|---|---|
| Hold Mon close → Fri close (baseline) | −0.39% | −0.10% |
| Best dev: trail on at +8%, trails 4%, sell half at trigger | −0.24% | −0.12% |
| Live-style: 10% stop, trail on at +17% / 8%, sell half | −0.46% | −0.09% |
| Enter Tue open (hold) | −0.55% | — |
| Enter Mon open (hold) | −0.68% | — |

Intraday management moves results by about 0.1–0.2% per trade and does not
help on the holdout. Monday close is the best entry time. **Which stocks get
picked decides the outcome; intraday handling barely changes it.**

## AI news classification (complete: 2,200 events, about $7.50)

| | Dev | Holdout | Post-cutoff (clean) |
|---|---|---|---|
| Fundamental score, return per point | +0.61% (t 1.9) | −0.56% (t −1.0) | −0.12% (t −0.3) |
| Guidance raised vs lowered | +0.2 / −1.9% | +1.5 / −3.0% | −3.4 / +0.9% |

The only positive result is in the period the model may remember. It
disappears on events after its training cutoff.

## Running now

`shadow_lc.yml` / `automation/shadow_lc_momentum.py` runs a shadow of the
large-cap monthly momentum design with no orders. It writes one row per week
to `data/shadow_lc_ledger.jsonl`: shadow value, live Alpaca equity and SPY,
all indexed to 100 from Oct 5, 2026.

---

# Round 5: market regime

## What the live system does today

`02_regime.py` labels each week (risk_on … risk_off_severe). `config/weights.json`
then multiplies the signal weights by that label.

**`self_refine.py` has rewritten those weights about weekly since June**, each
time from a few recent weeks of results per regime:

- Catalyst weight: 0.38 (Jun) → 0.60 (Jul) → 0.05 (Aug).
- Fundamentals weight: 0.10 → 0.43.
- Momentum now makes up just 9% of the score.

The system that produced the April–May trades no longer exists in its
original form.

## Regime replica, 2016–2026

The classifier was rebuilt on weekly ETF closes. VIX is approximated by SPY
realized volatility + 3 points. It tracks the live classifier with composite
correlation 0.83 and 61% label agreement over 23 overlapping weeks.

## Stage 1: do the labels predict which stock types win? (dev 2017–2023)

No. Across six stock types and five labels, every t-stat is below 1.8. The
live momentum blend's excess return is about zero in every label. The
momentum-crash rule (SPY 2-year return < 0) had only 10 dev weeks and no
effect.

Factor momentum ran the **opposite** way for momentum stocks. After 8 weeks
of beating the universe, 12-1 momentum underperformed the next week
(−0.13%). After 8 weeks of lagging, it outperformed (+0.54%, t 2.6, dev).

## Stage 2: regime-aware strategies (dev pick, holdout once)

| Design | Dev Sharpe | Holdout CAGR | Holdout Sharpe | Holdout maxDD |
|---|---|---|---|---|
| SPY | 0.76 | 20.6% | 1.31 | −17% |
| Large-cap momentum, always on | 0.42 | 33.5% | 1.06 | −28% |
| **Large-cap momentum only after an 8-wk momentum lag, else SPY** (added after Stage 1) | 0.76 | **33.7%** | **1.43** | −23% |
| Your labels as on/off | 0.30 | — | — | — |
| Your labels choose the stock type | 0.42 | — | — | — |
| Rotate to best trailing stock type | 0.16 | — | — | — |

The contrarian switch is the first design with a holdout Sharpe above SPY's.
Robustness is mixed:

- **Replication:** after-lag beat after-run in the holdout for the broad
  universe (t 1.8), but not within large caps (t −1.0).
- **Lookbacks:** switching on the large-cap factor itself underperformed SPY
  in dev at every lookback from 4 to 26 weeks, and outperformed in holdout at
  every one. That looks like a good 2024–26 for large-cap momentum, not a
  stable rule.
- **Beta-adjusted alpha:** +10.2%/yr, t 1.1.
- It was chosen after many holdout looks across this research.

**Conclusion:** promising enough to track live, not proven enough to trade.
The regime labels as currently built do not carry predictive information for
stock selection. Weekly re-tuning of regime multipliers fits noise.

---

# Actions taken (Oct 6, 2026)

1. **Weights frozen** (on `main`). `self_refine.py` now only logs proposals
   to `refinements/`. Switch back with `_meta.self_refine_mode: "apply"`.
2. **Shadow #2 live:** the contrarian regime switch
   (`automation/shadow_regime_switch.py`), running beside shadow #1
   (large-cap monthly momentum) in `shadow_lc.yml`. It reuses the research
   code exactly (verified to match on 6 historical weeks). Ledgers:
   `data/shadow_regime_ledger.jsonl` and `data/shadow_lc_ledger.jsonl`.
   First reading (Oct 5): **OFF / in SPY** (momentum factor +5.0% over the
   trailing 8 weeks).
3. **PR #2, pending approval:**
   - Regime multipliers set to neutral.
   - The four `*_adj` inputs removed from LightGBM.
   - One-time retrain on 14 stable signals, then retraining is report-only.

   Walk-forward IC is about zero for both old and new models; the change
   removes a distortion rather than adding an edge.

**Review date:** after about 12 weeks (early January 2027), compare the live
system, both shadows and SPY on the same Monday closes. Any regime rule that
gets promoted must also hold up in the 10-year history.

---

# Pass 3b: richer regime diagnosis (Oct 7, 2026)

**Script:** `research/regime_v2.py`.

**Features (12):** dispersion, breadth, breadth change, average correlation,
market volatility and its trend, SPY trend, SPY 52-week return, credit
(HYG−TLT), size (IWM−SPY), momentum crowding, and trailing momentum-factor
return. Each is z-scored using only past data.

**Choices:** six top-500 baskets (live proxy, 12-1 momentum, near 52-week
high, low vol, high vol, last week's losers) or SPY.

**Stage A (dev 2017–2023):** 10 of 72 feature/basket links had |t| > 2,
against about 3.6 expected by chance. There was a coherent "stress" cluster:
high dispersion, volatility and correlation, or weak credit, was followed by
high-vol stocks and last week's losers beating SPY.

**Stability check** (Newey-West t by period):

| Link | 2017–20 | 2021–23 | 2024–26 |
|---|---|---|---|
| dispersion → high-vol | +3.4 | −0.9 | −1.6 |
| dispersion → last week's losers | +3.3 | −0.4 | −1.9 |
| dispersion → 12-1 momentum | +2.7 | −0.4 | −1.9 |
| market vol → high-vol | +2.6 | −0.3 | +5.1 |
| credit → low-vol | +3.3 | +0.2 | +1.7 |
| trailing momentum factor → live proxy (shadow #2's input) | −1.9 | +0.3 | +0.0 |

The cluster comes almost entirely from the 2020 crash and rebound. It flips
sign or disappears afterward.

**Stage B** (dev comparison 2020–2023, holdout scored once):
- Ridge on all 12 features: Sharpe 0.31, worse than always-on (0.49).
- The best-single-feature rule changed its feature between refits
  (credit → dispersion), a sign of instability.
- Holdout: 13.2% CAGR (Sharpe 0.52), vs always-on large-cap momentum
  33.5% (1.06) and SPY 20.6% (1.31).

**Conclusion:** none of these 12 price- and market-based diagnoses gives a
stable regime signal for choosing what to hold. **Shadow #2's input also
looks weak:** its relationship came from 2017–20 and has been flat since.
Shadow #2 keeps running as cheap live evidence, but expectations should be
low.

---

# Pass 3c: macro, options and flow data (Oct 7, 2026)

**Data** (`research/fetch_macro.py`, all free):
- **FRED:** rates and yield curve, Baa/Aaa credit spreads, CPI,
  unemployment, industrial production, initial claims, Chicago Fed NFCI,
  real yields, fed funds.
- **CBOE:** VIX, VIX3M, VIX9D, SKEW; put/call ratios, 2006–2019 only.
- **Ken French daily momentum factor (UMD)**, from 1926.
- **FINRA daily short-sale volume**, from 2009. This is the off-exchange
  ("dark pool") data that DIX-style indexes are built from.

Publication lags are applied: CPI, unemployment and IP lag one month;
claims and NFCI lag 7 days. The NBER recession flag is excluded because
it is only declared after the fact.

## 60 years of momentum vs macro regime (`research/macro_regime.py`)

**Setup:** next-month momentum-factor return. Fit 1972–2004, choose on
2005–2016, score the holdout (2017–2026) once.

**No single macro measure predicts momentum.** 0 of 16 were significant in
1972–2004; none carried into 2005–2016. Macro quadrants (growth ×
inflation) were not stable either: "deflation" was the best quadrant in
1972–2004 (+1.6%/month) and the worst in 2005–2016 (−1.1%).

**Strategies on the momentum factor:**

| | 1972–2004 Sharpe | 2005–2016 Sharpe | Holdout 2017–2026 |
|---|---|---|---|
| Always on | 0.72 | 0.04 | Sharpe 0.27, max DD −35% |
| Volatility-managed (size down when momentum is volatile) | 1.03 | 0.59 | *diagnostic only:* 0.53, −12% |
| **Vol-managed + 16-feature macro on/off (pre-registered pick)** | 1.03 | 0.78 | **Sharpe 0.74, max DD −5.6%** |

**Attribution** (diagnostic only; the pick was already scored):
- The macro model was "on" in 90% of holdout months.
- Next-month momentum averaged **+0.55%** in the 102 "on" months and
  **−2.28%** in the 11 "off" months.
- The macro inputs add something on top of volatility management, but
  the evidence rests on only 11 "off" months.

**Translated to the long-only top-500 momentum basket** (exposure = rule
weight, rest in SPY):

| | 2017–2023 Sharpe | 2024–2026 Sharpe | Max DD (24–26) |
|---|---|---|---|
| Always in momentum basket | 0.42 | 1.05 | −28% |
| Rule-sized momentum + SPY | 0.64 | 1.15 | −22% |
| SPY | 0.76 | 1.32 | −17% |

The rule improves the basket's risk-adjusted return in both periods but
does not beat SPY.

**Live-use caveat:** Ken French data posts about a month late. A live
version needs our own momentum-factor volatility, which the weekly
panel already computes.

## Options, credit, macro and short volume on our baskets (`research/flow_regime.py`)

**Setup:** next-4-week (basket − SPY) on each input. An input must hold
with |t| > 2 and the same sign in both 2017–2020 and 2021–2023.

**Result (without short volume):** 19 of 54 links were significant in
2017–2020, but 7 in 2021–2023 and **0 in both**. VIX, VIX term structure,
SKEW, put/call, credit, curve, claims and NFCI all fall into the same
COVID-driven pattern seen in pass 3b.

**FINRA short volume.** The public archive starts Dec 29, 2017, so the
2017–2020 half could not be tested. In 2021–2023, a rising off-exchange
short ratio (vs its 26-week mean) preceded momentum beating SPY:
t 4.3 (live proxy) and t 3.6 (12-1 momentum). Both pairs were fixed in
advance and scored once on 2024–2026:

| Pair | 2021–23 t | Holdout t |
|---|---|---|
| Short-ratio change → live-proxy basket | 4.33 | −0.11 |
| Short-ratio change → 12-1 momentum basket | 3.59 | 1.46 (same sign, +2.2% per sd over 4 wks) |

Neither pair cleared |t| > 2 out of sample. The 12-1 link kept its sign,
so it goes on the watch list; it is not a rule.

## Pass 3c conclusion

- **Combining macro inputs works modestly; no single input does.**
  Volatility-managed momentum plus a 16-input macro on/off model passed a
  60-year fit / 12-year selection / 10-year holdout test.
- **On the long-only basket** it raises the Sharpe (0.42 → 0.64 in
  2017–23; 1.05 → 1.15 in 2024–26) and cuts drawdowns, but stays below
  SPY.
- **Options, credit and short-volume measures** give no stable
  short-horizon regime signal 2017–2026.

## Shadow #3 live (Oct 7, 2026)

`automation/shadow_macro_sized.py` (in `shadow_lc.yml`) runs the pass-3c rule
with no orders:
- **Monthly:** momentum weight = vol scale × macro on/off, capped at 1.
- **Weekly:** that share of NAV goes in the top-20 large-cap momentum
  basket; the rest is in SPY.

**Verification:**
- `research/macro_signal.py` reproduces the research weights exactly
  (549 months).
- With Ken French data at its real 1–2 month publication lag, the holdout
  Sharpe is 0.69–0.77 vs 0.27 always-on.

**Bug fixed during the build:** monthly FRED series lost their newest value
when lagged, and the Oct-2025 shutdown gap blanked some months. The
research holdout moved 0.74 → 0.77 (n 112 → 115); the pick did not change.

**First reading (Oct 5):** macro **ON**. Momentum volatility is high (24%),
so exposure is **27% momentum basket / 73% SPY**. The ledger is
`data/shadow_macro_ledger.jsonl`. You get a phone alert when the macro
switch flips.

**Three shadows now run beside the live system**, all indexed to Oct 5 = 100:
1. Large-cap monthly momentum (always on).
2. Contrarian regime switch (currently in SPY).
3. Macro + vol-sized momentum (currently 27/73).

Review in early January 2027.

## Kalshi daily-high temperature markets (Oct 8, 2026) — `research/kalshi/`

**Question:** does a free public forecast beat Kalshi's prices after fees?

**Data** (`research_data/kalshi/` on the research-data branch):
- 54,276 settled markets, 9,569 city-days, 14 cities, Aug 2021 – Oct 2026.
- Hourly bid/ask for 54,090 of them.
- GFS MOS and NBM text forecasts from the Iowa Environmental Mesonet.

**Model:**
- high ~ Normal(a + b·forecast, σ(season)).
- Fitted per station on dev (pre-2025) by interval-censored maximum likelihood.
- Uses only forecast runs that were public at the decision time: 9pm local the night before, or 9am local on the day.
- Only NYC, Chicago, Austin and Miami have enough dev history.
- Forecast error (NBM vs actual high): about 2°F (1.4°F Miami, 2.7°F Chicago).
- Buckets are 2°F wide.

**Results.** Dev = 2021–24, holdout = 2025–26, ~15k quoted markets each. Taker fills at bid/ask plus the fee.

| | Market log loss | Model | Market+model blend |
|---|---|---|---|
| 9pm, dev | 0.415 | 0.440 | 0.410 |
| 9pm, holdout | **0.333** | 0.379 | 0.333 |
| 9am, dev | 0.374 | 0.432 | 0.370 |
| 9am, holdout | **0.289** | 0.369 | 0.291 |

- **The market is well calibrated.** At every price bucket, outcomes land within about 2–4 percentage points of the price.
- **The market got sharper over time:** holdout log loss is about 20% lower than dev.
- **Trading on the model loses** in the holdout, −2 to −2.5¢ per contract (t −4 to −8).
- **The blend made 2.5–3.8¢ per contract in dev** (in-sample, weights fit on dev), but −2¢ to +0.1¢ in the holdout. It does not survive.
- **Longshot bias** (buy NO on markets priced under 5¢): −1 to −1.5¢ per contract in dev, +0.4¢ in the holdout (t 1.9). This is noise after the spread.
- The Aug–Oct 2026 slice of the holdout was seen once in a debug run; nothing was tuned on it.

**Conclusion:** no edge from public point forecasts. Traders already price NBM/GFS and do better. If the Kalshi idea continues, the only plausible paths are:
- (a) trading the same day on live station readings: the max so far plus the HRRR model. The edge there is speed, and the window is short.
- (b) market making (posting quotes, earning the spread), which needs order-book data and an API account.

Both are a different kind of system from "many small bets on a forecast".

## Rank tranches and hold length (Oct 10, 2026)

Scripts:
- `research/rank_tranches_live.py`
- `research/rank_tranches.py`
- `research/hold_conditions.py`

Results: `research/results/hold_conditions.csv`.

### Does rank 1–3 underperform rank 4–13?

**Live history:** 21 weeks with composite ranks (Apr 17 – Oct 2, 2026). Excess return is measured vs. the universe, Tue open → Fri close.

| Ranks | All 21 wks | First 7 wks | Last 14 wks (after the Jun 5 scoring fix) |
|---|---|---|---|
| 1–3 | −1.00% (t −0.8) | −2.36% | −0.32% |
| 4–6 | +0.51% (t 0.3) | +3.44% | −0.95% |
| 7–10 | −0.04% | +3.14% | −1.64% |
| 11–13 | −1.69% (t −1.6) | −0.02% | −2.53% |

- The June observation (1–3 bad, 4–10 good) came from the first 7 weeks.
- In the 14 weeks since, every bucket in the top 20 trailed the universe.
- Nothing is significant.

**10 years (2017–2026) on the momentum proxy, 12-1 momentum, and the live RS blend:**
- The 1–3 vs 4–6 vs 7–13 differences are all |t| < 1.5.
- The sign flips between dev and holdout, and between signals.
- **There is no robust "skip the top 3" effect.**
- What the very top does carry is the biggest recent spike. Median last-week return is +7.0% for ranks 1–3, +5.2% for 4–13, and +0.2% for the rest. Short-term reversal is real (decile table).

### Hold length

Staggered tranches, 15 bps per side. Momentum proxy, top 20.

| Hold | 1 wk | 2 wk | 4 wk | 8 wk | 13 wk |
|---|---|---|---|---|---|
| Dev CAGR | −1.6% | 3.7% | 7.6% | 12.1% | 14.1% |
| Dev Sharpe | 0.07 | 0.27 | 0.42 | 0.57 | 0.64 |
| Holdout CAGR | 6.8% | 12.7% | 23.5% | 16.8% | 16.2% |

The live score's edge is slow, and weekly turnover throws it away. This is the most consistent result of the whole review.

### Condition-based exits and entries

106 designs. Dev 2017–23 picks, holdout 2024–26 once.

- **Hold while the stock stays in the top-B by rank** (exit on a condition, not a date).
  - 12-1 momentum, top 10, hold while in the top 100:
    - Dev: 26.4%/yr, Sharpe 0.75, max DD −46%.
    - Holdout: 38.1%/yr, Sharpe 0.85, max DD −42%.
    - SPY: dev 13.0% / 0.76; holdout 20.6% / 1.31.
  - Median hold is 21 weeks. It beat SPY in 7 of 10 years and lagged in 2021, 2023 and 2026 YTD.
  - Neighbouring designs agree: 12-1 with a top-50 to top-200 exit gives holdout 11–38%/yr.
  - The live proxy and the live RS blend do worse under the same rules.
  - It was best of 106 designs, so treat the level as optimistic. The direction (slow momentum, rank-based exit) is robust.
- **Trailing stops** (10–20% off the peak weekly close): dev 15–21%, holdout −13% to +6%. They don't help.
- **Entry filters:**
  - Skip last week's top-10% movers: dev 19.5%, holdout 20.9% (12-1, n20).
  - Buy only after a down week: similar.
  - Modest help, not a standalone edge.

**Conclusion:** the tranche idea is noise. The N-length idea is the real lead:
- Buy 12-1 momentum.
- Hold until the name falls out of the top ~100.
- Accept −40% drawdowns.

Candidate for shadow #4.

## Full-spectrum rank bands, earnings weeks, large-cap reversal (Oct 10, 2026)

Scripts:
- `research/rank_bands.py`
- `research/earnings_week.py`
- `research/reversal_lc.py`

Results:
- `research/results/rank_bands*.csv`
- `research/results/reversal_lc.csv`

### Rank bands

**Setup:**
- 10 scores: live proxy, 12-1, 6-1, 3-1, RS blend, 52-week high, trend, last week, last month, low vol.
- 61 bands each: 11 fine bands over ranks 1–500, plus 50 two-percent bins over the whole list.
- Horizons of 1/2/4/8/13 weeks.
- Plus size, market-trend and dispersion cuts.
- 3,050 band tests in all; Newey-West t-stats.

**Results:**
- **Fewer bands were significant than chance.** 3% had |t| > 2 in dev, versus 5% expected by luck.
  - Of those, 22% kept the sign with |t| > 1 in the holdout, versus about 16% by chance.
  - Bands at the head of the list carry essentially no information.
- **No stable sweet spot at the top.**
  - For every momentum score, the profile over ranks 1–500 has near-zero or negative rank correlation between dev and holdout.
  - The "skip 1–3, buy 4–13" shape does not exist in 10 years of data.
  - Live proxy at 13 weeks: dev ranks 1–50 all earn +1.2 to +1.8% excess; holdout is mixed.
- **The broad, coarse momentum profile is stable only at long horizons.**
  - Dev/holdout correlation of the 50-bin profile: RS blend 0.74–0.80 at 8–13 weeks, versus 0.13 at 1 week.
  - Live proxy: 0.61–0.69 versus −0.01.
  - This is the same message as the hold-length test.
- **One band effect is robust: the extreme top of recent gainers.** The 5 biggest gainers of the past week, out of ~2,400:

| Horizon | 2 wk | 4 wk | 8 wk | 13 wk |
|---|---|---|---|---|
| Dev excess | −2.7% (t −4.8) | −2.5% | −3.7% | −3.8% |
| Holdout excess | −2.4% (t −2.1) | −3.7% | −7.7% | −9.2% |

- Ranks 6–10 show about half the effect, and ranks 11+ are flat.
- Last month's top 5 gainers behave the same way: −3.8% to −14.1% over 2–13 weeks in the holdout.
- The effect holds in large and small names and in most market states.
- Only about 7% of the live system's top-10 picks fall in that zone, so filtering them helps a little.

### Earnings weeks

The live system excludes holding through an earnings report.

| Group | Dev | Holdout |
|---|---|---|
| All stocks reporting | −0.06%/wk (t −0.5) | −0.04%/wk |
| Top-2% momentum, reporting | **+1.06%/wk** (t 1.9) | **+1.52%/wk** (t 1.4) |
| Top-2% momentum, not reporting | −0.07% | −0.05% |
| Large caps (top 500), reporting | −0.19% | −0.66% (t −2.0) |

- No general earnings premium.
- Strong-momentum names, however, beat the universe in the weeks they report, in both periods.
- Single-stock risk doubles: the weekly return spread is 9.6% versus 4.5%.
- Effect on the weekly top-20 proxy basket of excluding earnings:
  - Dev: −9.6% CAGR with the exclusion, −7.1% without.
  - Holdout: −11.0% with, −3.3% without.
- **The exclusion rule has cost money on average.** It reduces single-name blow-ups (FSLY), not losses.

### Large-cap short-term reversal

Buy last week's biggest losers among the most-traded names. 48 designs.

- Best dev design: 1,000 names, 20 losers, skip earnings-driven drops, 5 bps.
  - Dev: 30.4%/yr, Sharpe 0.84.
  - Holdout: **−8.5%**.
- No design holds up in the holdout at 15 bps.
- **Dead.** The effect has decayed since 2024.
