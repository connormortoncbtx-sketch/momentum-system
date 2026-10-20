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

## Post-scoring filters (Oct 10, 2026)

V1 filtered the scored list on prior-week return and Monday's move, and bought at Tuesday's open.

Scripts: `research/post_filters.py` and `research/post_filters_portfolio.py`. Results: `research/results/post_filters.csv`.

**Setup:**
- Each week, take the top 50 by score. Split them into fifths by a factor and measure each fifth against the pool average.
- 23 factors, all known at decision time:
  - **Recent moves:** prior week, prior 4 weeks, Monday's return, Monday's open gap, Monday intraday, Tuesday open gap, stretch above the 10-week average, close vs. the week's high.
  - **Trend and momentum:** 52-week-high distance, trend count, 12-1, 3-1.
  - **Risk:** 12-week vol, lottery (max week), 52-week beta, idiosyncratic vol.
  - **Size and liquidity:** dollar volume, volume surge, price level.
  - **Events:** earnings last week, earnings this week, insider buying in the last 4 weeks.
- Four pools:
  - Live proxy, bought Tue open (V1 timing).
  - Live proxy, bought Mon close.
  - 12-1 momentum held 4 weeks.
  - 12-1 momentum held 13 weeks.

**Result: filters are noise, with one exception.**
- 80 top-minus-bottom tests. 5 had |t| > 2 in dev, about what chance gives (4).
- Only one held up in both periods: **reporting earnings during the hold week is GOOD.** It beats the pool by +1.0% (t 2.0) in dev and +2.3% (t 2.2) in the holdout, per week. The live rule removes exactly these names.
- **V1's Monday filter has the right sign in both periods but is weak.** The fifth that ran most on Monday trails the pool, Tue open → Fri close: −0.14% dev, −0.30% holdout per week. The fifth that fell most on Monday leads: +0.08%, +0.26%.
- Tuesday-gap and prior-week filters point the same way, with smaller and less consistent effects.
- **Size, price level and beta look strong in the holdout only.** Big, high-priced, high-beta names won in 2024–26, the mega-cap era. They show nothing in dev, so they are a regime effect, not a filter.

**Top-10 weekly portfolios** (live proxy, Tue open → Fri close, 15 bps):

| Variant | Dev CAGR / Sharpe | Holdout CAGR / Sharpe |
|---|---|---|
| No filter | −17.0% / −0.68 | −2.3% / 0.09 |
| Skip top-20% Monday runners | −11.8% / −0.45 | −1.3% / 0.11 |
| Skip Monday + prior-week runners | −13.6% / −0.62 | +8.4% / 0.44 |
| Skip top-20% Tuesday gap | −12.3% / −0.49 | −0.5% / 0.13 |
| Current rule: skip earnings in hold week | −19.5% / −0.83 | −8.3% / −0.13 |

The Monday filter helps a few points a year, but cannot rescue a weekly-turnover strategy.

**Live-only factors** (21 weeks of the live top 50): catalyst, fundamentals and sentiment sub-signals, conviction, EV, predicted vol, sector. Nothing reaches |t| > 1.2; there is too little history.

Factors with no history here (short interest, options flow, analyst revisions, news) can only be tested forward.

## Novel-angle program, part 1 (Oct 10–11, 2026)

**New data on the research-data branch:**
- `sec/filings_<yr>.parquet`: 4.16M filings, 7,860 current issuers, 2016+, with 8-K item codes.
- `sec/fundamentals_<yr>.parquet`: 10.2M XBRL values, 13,144 companies including delisted ones, with acceptance times.
- `alt/*`: FINRA per-stock short volume, SEC fails-to-deliver, Wikipedia page views. Queued.

**Timestamp caveat (verified with `research/probes/sec_time*.py`):**
- The SEC submissions API's `acceptanceDateTime` is sometimes true UTC and sometimes UTC plus one extra Eastern offset.
- The stored Eastern value is therefore either right or 4–5 hours late, never early. That is safe to use: it can delay a signal but never leaks the future.
- Studies that need the exact time read the filing's index page instead.

### Forced flows (`research/forced_flows.py`)

Weekly data. Excess vs the equal-weight universe. Dev = events before 2023, holdout = 2023 on.

**Follow-on offerings** (424B4/424B5 from companies listed more than 6 months; first prospectus per 30 days):

| Week vs. filing | 424B5 dev | 424B5 holdout | 424B4 dev | 424B4 holdout |
|---|---|---|---|---|
| −1 | +1.61% (t 8.5) | +1.04% (t 4.3) | −2.0% | −7.1% |
| 0 | −1.74% (t −11.4) | −2.38% (t −9.6) | −7.5% | −12.8% |
| +1 to +4, per week | −0.03 to −0.27% | −0.29 to −0.75% (t −2 to −4.5) | mixed | mixed |

- Companies raise money right after a run-up, then lag for weeks.
- **This is the first new effect that is robust in both periods.** Use it as a hard filter (never hold a recent issuer) and as a candidate short signal.
- The next step is predicting issuance before it happens: a shelf registration, cash burn and a recent rally.

**IPO lock-up expiry:**
- 1,477 IPOs identified (first trade plus a 424B4 within 3 weeks).
- There is no clean lock-up-week effect.
- Recent IPOs drift lower generally, about −0.5 to −1%/week around month 6, in both periods.

**Tax-loss / January** (YTD-loser decile, bought around Dec 20):
- Dev: +2.8% to +3.2% over 4 weeks (t ≈ 0.9, 6 years).
- Holdout: −0.3% to −0.5% (3 years).
- Not reliable.

### LLM filing reader (`research/llm_sample.py`, `research/llm_filings.py`)

**Sample:**
- Small-cap 8-Ks: $30M–$2B market cap, ≥ $300k/day traded, price ≥ $1.
- Items 1.01–8.01, excluding earnings (2.02).
- 5,000 dev filings (2023 – Mar 2025) and 12,000 holdout filings (Apr 2025 – Sep 2026, after the model's training cutoff, so it cannot know outcomes).

**Method:**
- Text is anonymised: company name and ticker are removed.
- Claude Haiku 4.5, via the Message Batches API, judges type, $ value, direction, materiality relative to company size, novelty, and red flags.
- Entry is at the first open after the authoritative acceptance time.

Results pending.

### Spike anatomy, step 1 (`research/spike_catalog.py`)

**Definitions:**
- Spike = a week ≥ +25% above the equal-weight universe; crash = ≤ −25%.
- Universe: price ≥ $2, ≥ $500k/day traded, ≥ 26 weeks listed.
- 2017 to Sep 2026: **9,945 spikes and 6,827 crashes** (about 1,000 spikes a year; 1,300–1,650 a year since 2020).
- Catalysts come from the SEC filings index plus the earnings table.

**Spikes:**

| Catalyst | Share | Next 4 weeks: median excess | % up |
|---|---|---|---|
| No filing (news or flow) | 39% | −7.3% | 38% |
| Earnings (scheduled) | 21% | −0.7% | 48% |
| Other filing (10-Q, S-3, 13G…) | 16% | −7.5% | 36% |
| Press release 8-K | 6% | −6.1% | 39% |
| Offering / dilution in the same week | 5% | −7.7% | 36% |
| Material agreement | 2% | −5.0% | 37% |
| M&A forms (under-counted: acquired targets are delisted) | 1.5% | −13.3% | 34% |
| Accounting / listing problem | 0.5% | −22.3% | 33% |

**Crashes:** offering-driven crashes keep falling (−14.1% median, only 26% up). Earnings crashes stabilise (−0.9%).

**Takeaways:**
1. **Chasing spikes after the fact loses.** Every unscheduled type fades by a median of 5–13% over the next month. Only earnings spikes hold.
2. **The catalyst decides what comes next.** Classifying the catalyst fast is worth something on its own: hold real scheduled news, fade unexplained spikes.
3. **39% of spikes have no filing behind them.** Explaining those needs news data (GDELT next). These are also the ones that fade.
4. **"Event minus one" therefore matters most for spikes we can enter early AND exit near the top.** Step 2 looks for signatures in the 1–8 weeks before, against matched controls.

### Steady risers at the 3%/week run rate (`research/steady_risers.py`, `research/results/steady_risers.txt`)

Run rate: 3%/wk = +12.6%/4 wk = +46.9%/13 wk.

**A. How common:**
- A steady run is 13 weeks at ≥ +46.9% with ≥ 9 up weeks, no week above +20%, and an in-window drawdown no worse than −12%.
- There were **3,323 such episodes since 2017** (about 100–700 a year). On a typical week 0.47% of the universe is in one.
- Any-path runs at the run rate are about 6× more common.

**B. Can a steady run be caught after 8 weeks?**
- Take stocks up ≥ 20–30% in 8 weeks with ≥ 6 up weeks and no week above +15%.
- They hit the run rate over the NEXT 13 weeks only 4–7% of the time. All stocks: 4–5%.
- Their 13-week excess return is +0.7% to +2.4%, the same sign in dev and holdout but small.
- **A steady path alone does not predict continuation.**
- Jumpy 8-week risers hit the run rate more often (12%) but crash more (29–38% fall below −20%). Holdout excess −3.3%.

**C. Smooth vs. choppy winners** (top 10% by 26-week return, split by share of up weeks):
- Smooth beats choppy over 13 weeks: +1.9% vs +0.3% in dev, +1.5% vs −3.4% in the holdout. t ≈ 1–1.7.
- Consistent direction, weak strength. This is the known "frog in the pan" effect.

### 8-K item codes (`research/item_events.py`, `research/results/item_events.csv`)

**Setup:**
- Every 8-K since 2017 from current filers, weekly data.
- Excess return vs the equal-weight universe ($2, $500k/day, 26+ weeks listed) in the 4 weeks AFTER the filing week.

**Baseline:**
- Conditioning on being in the universe at t gives a small negative drift, because losers later drop out of the universe.
- Over 4 weeks, current 8-K filers average −0.60% (dev) and −1.33% (holdout).
- **Read the item effects against that baseline.**

| Item | 4-wk after, dev (vs −0.60) | Holdout (vs −1.33) |
|---|---|---|
| 3.02 unregistered share sale | −2.79% (t −7.0) | −5.83% (t −9.7) |
| 3.01 listing deficiency | −2.51% | −7.76% |
| 3.03 holder rights modified (often reverse splits) | −1.19% | −5.15% |
| 5.01 change of control | −2.69% | −11.69% |
| 5.03 bylaws/charter (often split authorisations) | −1.15% | −2.65% |
| 1.01 material agreement (many are financing deals) | −0.99% (small caps −1.24%) | −2.64% (small caps −3.50%) |
| 2.01 acquisition/disposal completed | −1.17% | −3.60% |
| 2.02 earnings, 7.01 Reg FD, 8.01 other | ≈ baseline | ≈ baseline to −0.8% below |

- No item code predicts outperformance.
- Codes tied to **dilution and distress** predict underperformance well beyond baseline in both periods. That gives a second robust avoid/short list alongside the offering result.
- Item 1.01 mixes good contracts with financing deals. Separating them is the LLM reader's job.

### XBRL fundamental factors (`research/xbrl_factors.py`, `research/results/xbrl_factors.csv`)

**Setup:**
- Top-minus-bottom quintile within the $5 / $2M universe, 4- and 13-week horizons.
- 7,065 symbols matched by CIK. Survivorship caveat: delisted companies don't map to tickers.

**Results:**
- **No factor is robust.**
- Gross profitability: dev +1.6%/13 wk (t 2.7), holdout −3.5%.
- Revenue acceleration: dev +0.9%/13 wk (t 3.6), holdout +0.1%.
- Net issuance, low asset growth, ROA, earnings yield, book-to-market: |t| < 2 in both periods.
- The accruals test is broken (1% coverage): cash flow is reported year-to-date and needs a different build. This is low priority given the rest.
- Classic published factors add nothing usable here. That is consistent with their post-publication decay and with the 2024–26 rally in low-quality stocks.

### LLM filing reader: results (`research/llm_analyze.py`, `research/results/llm_filings.txt`)

**Run:**
- 16,996 of 17,000 filings judged.
- Cost: 38.5M input + 2.4M output tokens with Claude Haiku 4.5 Batches ≈ **$25 (≈ $1.50 per 1,000 filings)**.
- Returns are measured from the FIRST OPEN after the authoritative acceptance time, IWM-adjusted.
- The score was designed on dev (2023 – Mar 2025). The holdout (Apr 2025 – Sep 2026, after the model's training cutoff) was scored once.
- Anonymisation leaked some names (e.g. a summary named Tilray). That matters only for dev; the holdout is post-cutoff regardless.

**Bad news keeps drifting down after the open, in both periods:**

| Holdout | n | Day 1 | Day 5 | Day 20 |
|---|---|---|---|---|
| LLM direction −1 | 2,314 | −1.14% | −1.58% (t −5.2) | −2.41% |
| LLM direction −2 | 339 | −2.33% | −1.12% | −2.62% |
| 4+ red flags | 4,595 | −1.01% | −1.23% (t −5.2) | −2.96% |
| Dev, direction −1 | 972 | −0.76% | −1.17% (t −2.7) | −2.80% |

**Good news is priced by the open and then fades:**
- Direction +2: gap +7.6% at the open, then −0.4% (5d) and −1.1% (20d) in the holdout.
- Big positive deals (stated value > 20% of market cap) fade −2.8% to −4.4% over 20 days.

**Event types with the same sign in both periods (5-day / 20-day):**

| Event | Dev | Holdout |
|---|---|---|
| **Contract win** (the one positive) | +5.1% / +6.4% (t 2.4) | +2.6% / +5.0% (t 1.9; n = 207, ~12/month) |
| Clinical "positive" | −3.6% | −2.0% / −3.4% |
| Company announces an acquisition | −0.7% / −1.5% | −2.0% / −4.8% (t −4.1) |
| Equity financing | −0.8% / −3.1% | −1.7% / −3.4% |
| Executive departure | −0.6% | −1.1% |
| Partnership | −1.3% | −1.3% |

**Overall rank correlation** of the score with 1–20 day returns: +0.02 to +0.05.

**Conclusion:**
- Reading small-cap filings gives **a strong, cheap, consistent bad-news signal** (avoid/short) and **one long signal, contract wins**.
- It does not find broad positive under-reaction. Good news is already in the opening price.
- Shorting small caps is limited by borrow cost and availability, so the bad-news side is most usable as a filter on any long book. It can be a short book only where borrow is cheap.

### Spike anatomy step 2: pre-event signatures (`research/signatures.py`, `research/signatures_model.py`)

**Setup:**
- 1.56M stock-weeks (2017 – Sep 2026), 30 features known at the Friday close:
  - price and volume
  - FINRA short-volume share
  - fails-to-deliver (only files already published)
  - Wikipedia attention
  - filing counts and types
  - insider buys
  - earnings due next week
- LightGBM trained on 2017–2023 and scored once on 2024–2026.

**Results** (holdout; lift = the event's rate in the model's top 1% ÷ its base rate):

| Target | AUC | Base rate | Top-1% predicted rate |
|---|---|---|---|
| Spike next week (≥ +25% excess) | 0.89 | 0.86% | 9.8% (11×) |
| Crash next week (≤ −25%) | 0.93 | 0.64% | 17.6% (27×) |
| Run-rate quarter (≥ +46.9% in 13 wk) | 0.77 | 5.2% | 17.1% (3.3×) |

**What drives it:**
- Most of the predictive power comes from volatility. Top features by gain: 12-week volatility (21–47%), distance below the 52-week high, low price, and an earnings report due.
- Distress filings (3.01/3.03/5.01) and share sales (3.02) raise the odds of a spike 3× — and of a crash too.

**Payoff:**
- **Buy the 10 likeliest spikes each week:** catches about 1 spike a week but returns −1.1%/wk excess (median −3.2%).
- **Rank by P(spike) − P(crash):** about −0.2%/wk.
- **Top 20 by P(run-rate quarter), held 13 weeks:**
  - Mean +7.8% vs universe +3.6%. Median −5.8%.
  - 24% of picks lose more than 30% (universe: 8%).
  - Non-overlapping t-stats 0.5–1.3.
  - That is a lottery ticket, not an edge. It does beat a pure top-volatility basket (−12.3%).

**Conclusion:**
- Public pre-event signatures predict THAT a big move is coming (AUC ~0.9), not WHICH WAY.
- "Event minus one" is real for volatility. Direction must come from content: the filing reader's bad-news side, and contract wins.
- The natural instrument for "a big move is coming, direction unknown" is options (straddles). They are priced by the same crowd, so the test would be whether our predicted-move probability beats the options' implied move. That needs options history we do not have free.

### Avoid list and spike filter on slow momentum (`research/avoid_filter.py`)

**Setup:** 12-1 momentum, hold while ranked ≤ 100, entry filters only.

**Results:**
- **Avoid list** (3.02, 3.01/3.03/5.01, or a 424B prospectus in the last 8 weeks — 19% of top-50 momentum names):
  - Top 10: dev 26.4% → 21.4%, holdout 38.1% → 39.6% (max DD −42% → −34%).
  - Top 20: dev 18.3% → 17.8%, holdout 26.6% → 15.7%.
  - Not robust.
- **Spike filter** (skip the top-10 gainers over 1 or 4 weeks):
  - Top 10: dev 26.4% → 19.1%, holdout 38.1% → 34.6%.
  - **Worse.** It was removed from shadow #4 before its first run, so the shadow matches the backtested design.

**Takeaway:** the avoid signals help event-driven, short-horizon trading. They do not help a slow momentum book, whose winners often issue stock into strength and keep running.

### Shadow #6 live (Oct 11, 2026): `automation/shadow_filings.py`, `.github/workflows/shadow_filings.yml`

**Schedule:** twice each weekday, about 9:30pm ET and 8:40am ET.

**What it does:**
- Reads EDGAR's latest-8-K feed (feed parsing verified in Actions with `research/probes/edgar_feed.py`).
- Filters to small caps: $30M–$2B market cap, price ≥ $1, ≥ $300k/day traded, and the study's item list.
- Judges each filing with the study's exact prompt and model.
- Logs every judgement to `data/shadow_filings_judgements.jsonl`.

**Two hypothetical 20-session books, IWM-adjusted, entered at the first open AFTER each judgement:**
1. Contract-win longs.
2. Bad-news shorts (direction ≤ −1 or 4+ red flags), before borrow costs.

Closed trades go to `data/shadow_filings_ledger.jsonl`. Estimated cost is about $0.003 per filing.

**Fix found while wiring it:** the shadow workflows ran `git add` on a list of files that included ledgers not created yet. That makes git stage nothing, so shadows #4/#5 would have silently lost their first state on Monday. Now only existing files are staged.

## Winners first (Oct 11, 2026): `research/winners.py`, `research/results/winners.txt`

Start from the biggest winners and dissect them.

### Before the run

1,213 quarterly top-1% winners, 2017–2026; median next-13-week return +106%. They are compared with the bottom-1% losers (median −63%). Average percentile at the START of the quarter, where 50 = typical stock:

| Trait | Winners | Losers | Typical (middle half) |
|---|---|---|---|
| 12-week volatility | 82 | 85 | 42 |
| Distance below 52-week high | 74 | 78 | 44 |
| Share price | 24 | 22 | 55 |
| $ volume | 36 | 34 | 52 |
| 12-1 momentum | **45** | **36** | 51 |
| Earnings filed in last 13 wk | **47** | **39** | 51 |
| Form 4 activity | **48** | **40** | 50 |

- **Before the run, the biggest winners and the biggest losers look almost the same:** volatile, cheap, small, beaten-down lottery tickets.
- The differences are a few percentile points. Winners have somewhat better 12-1 momentum, are current on their filings, and show more insider-form activity.

### Where they were

- Pharma/biotech (SIC 28) is **29% of all winners, 2.6× its share of the universe**. These are binary clinical and regulatory events.
- Credit/lending (SIC 61) is 3.1× over-represented, and electronics/semis 1.6×.
- In each year, the five industries with the most winners hold 45–69% of that year's winners, against 30–34% of the universe. **Clustering is strong.**

### Is the clustering tradeable in advance?

Next-13-week excess, top vs bottom quintile:

| Signal | Dev spread | Holdout spread |
|---|---|---|
| Industry 13-wk momentum (SIC-3, leave-one-out) | +1.6% (t 1.6) | +0.5% (t 0.2) |
| Own 12-1 momentum | +2.1% (t 1.9) | +1.8% (t 1.7) |
| Strong stock in strong industry (vs universe) | +2.6% (t 2.4) | −0.7% |

- **Industry waves are visible only in hindsight.** Last quarter's hot industry does not reliably stay hot.
- Plain 12-1 momentum is the one trait that keeps working, which is consistent with shadow #4.

### AI post-mortems

`research/winner_postmortems.py`: each 2024–26 top/bottom-1% stock's 8-Ks are read to name the driver, whether its date was known in advance, and any precursor. Queued.

### Earnings run-up for momentum names (`research/earnings_runup.py`, `research/results/earnings_runup.txt`)

Excess return by week relative to the report week (0). Groups are 12-1 momentum percentiles, measured 3 weeks before the report.

| Group | Week −1, dev | Week −1, holdout | Week 0, dev | Week 0, holdout |
|---|---|---|---|---|
| Top 10% | +0.18% | +0.57% | +0.40% | +0.69% |
| 80–90% | −0.07% | +0.03% | +0.08% | +0.18% |
| Bottom 20% | +0.10% | −0.08% | +0.35% | −0.66% |

**The pre-registered rule did not work.** Buying top-20% momentum names one week before their report and holding 2 weeks returned −0.08% in dev and +0.67% in the holdout (t 1.0).

**The positive effect is confined to the top 10%.** That matches the earlier earnings-week result (top-2% reporters +1.1% to +1.5% a week). Narrowing to the top 10% is a post-hoc choice, so it stays a hypothesis for shadow #5, not a finding.

### Queued Oct 11 (parallel research runs)

- `research/fetch_trend.py`: cross-asset ETF and crypto daily bars, for time-series momentum and trend tests.
- `research/fetch_pdufa.py`: an FDA-decision (PDUFA) calendar mined from SEC full-text search, for a biotech run-up test. Pharma/biotech is 29% of the biggest winners.
- `research/fetch_asos.py`: hourly station observations, for the same-day Kalshi test (the high so far vs. market prices).

### Trend following, cross-asset ETFs and crypto (`research/trend_test.py`, `research/results/trend_test.txt`)

**ETFs** (41 ETFs, monthly signal, 10% vol target per asset). CAGR and Sharpe; dev = 2016–22, holdout = 2023–26:

| Book | Dev | Holdout |
|---|---|---|
| Long-only, 63-day | 16.3%, Sharpe 0.71, max DD −50% | 24.7%, Sharpe 1.08, max DD −20% |
| Long-only, 252-day | 10.5%, 0.50 | 28.6%, 1.20 |
| Long/short | 6.8–7.4%, 0.44–0.50 | 1.0–12.5%, 0.14–0.80 |
| SPY | 11.6%, 0.68, max DD −34% | 22.4%, 1.43, max DD −19% |

**Crypto** (16 coins, long when price is above its N-day average, otherwise cash; 25 bps costs):

| Book | Dev (2021–23) | Holdout (2024–26) |
|---|---|---|
| 50-day | 8.6%, Sharpe 0.52, max DD −31% | 14.7%, 0.77, max DD −32% |
| BTC buy & hold | 5.0%, 0.38, max DD −77% | 27.5%, 0.75, max DD −53% |

**Verdict:**
- **Trend following cuts drawdowns but adds no return.** ETF long/short barely makes money. Long-only roughly tracks SPY.
- Crypto trend matches BTC's Sharpe with half the drawdown.
- It is a defensive overlay, not a growth engine.

### Straddles on predicted movers (`research/options_straddles.py`)

The signature model predicts *who* will move (AUC 0.89 spike, 0.93 crash) but not the direction. So the natural trade is to buy a straddle on the names it flags. Alpaca option history only starts in Jan/Feb 2024, so this test is holdout-only. It buys the nearest-expiry ATM straddle at the entry close and holds it to expiry.

| Group | n | Mean P&L / premium | Median | Implied move | Realized move |
|---|---|---|---|---|---|
| Model top picks | 1,045 | −5.3% | −24.1% | 19.6% | 14.3% |
| Control (random universe names) | 1,520 | −6.4% | −23.8% | 6.4% | 4.5% |

**Verdict: no edge.** The options market already prices the moves the model predicts. The flagged names carry triple the implied move, and still realize less than that implied move. The model finds the same volatility the market maker sees.

### PDUFA run-up (`research/fetch_pdufa.py`, `research/pdufa_test.py`, `research/results/pdufa_test.txt`)

804 FDA decision dates (193 companies), mined from SEC full-text search. Only weeks after the date was public are counted. Median lead time is 128 days.

- **Decision week is negative:** −3.6% excess in dev, −2.2% in holdout (medians −1.6% / −0.9%). This is "sell the news" or binary risk.
- **No pre-decision run-up** in the 8 weeks before the decision; every weekly mean is within ±0.7%.
- **Strategy** (buy 4 weeks before, sell before the decision week): dev −0.02% (t −0.03), holdout +1.25% (t 1.4, win rate 54%). Weak and inconsistent.

**Verdict:** not a growth engine. The decision-week drop is a reason to keep known binary events out of a long book.

### Same-day Kalshi weather (`research/kalshi/sameday.py`, `research/results/kalshi_sameday.txt`)

The test asks whether the high observed so far (ASOS hourly data) beats Kalshi prices late in the day. The model takes the high so far plus a historical distribution of afternoon gains.

| Hour | Market log loss (dev / holdout) | Model log loss (dev / holdout) |
|---|---|---|
| 12:00 | 0.162 / 0.213 | 0.207 / 0.294 |
| 14:00 | 0.055 / 0.120 | 0.105 / 0.207 |
| 16:00 | 0.022 / 0.041 | 0.123 / 0.166 |

The market is far better calibrated than the model at every hour. Trades against it lose in both periods (holdout t −3.5 to −8.2). **Closed.** Participants already watch the live observations.

### Winner post-mortems (`research/winner_postmortems.py`)

Haiku read the filings around ~330 top winners and ~300 bottom losers and named each one's driver.

| Group | Unclear | Most common named drivers |
|---|---|---|
| Winners | 253 | earnings/guidance 20, contract/customer 12, acquired/merger 10 |
| Losers | 180 | delisting/distress 58, dilution/offering 50 |

**Verdict:**
- **Most big winners have no filing catalyst.** The filing stream explains losers far better than winners.
- This matches the earlier LLM-reader result: bad news is legible and drifts; good news is either priced at the open or never shows up in filings at all.

#### Post-mortem detail and precursor test (`research/precursor_test.py`, `research/results/precursor_test.txt`)

The sample was 333 winners and 328 losers. Every one had at least one 8-K in its window.

- **Scheduled catalyst:** 8% of winners and 10% of losers had one.
- **Hint before the window:** the model saw a "yes" or "partly" hint for 7% of winners and 13% of losers.
- **Named precursors that are observable as form types:**
  - merger communications (Form 425): AVAV, MOVE
  - proxy solicitation: SDEV
  - preliminary proxy votes on reverse splits or share authorisations: MLGO, TJGC
  - Nasdaq compliance hearings
  - the rest are one-off narratives.

I tested every observable precursor against the base rate, across all universe stocks from 2017 to 2026. Each event is the first such filing in a 4-week gap. Returns are excess returns starting the week after the filing.

| Precursor | 13-week excess (dev / holdout) | Winner rate (dev / holdout) | Base rate |
|---|---|---|---|
| Form 425 merger comms | −2.5% / −4.6% (t −4.3 / −5.4) | 2.3% / 1.9% | 3.8% / 5.2% |
| Proxy solicitation | +0.3% / +0.3% | 4.7% / 6.6% | 3.8% / 5.2% |
| Merger proxy | −4.9% / −6.0% | 3.7% / 2.1% | |
| Tender offer | −0.1% / −3.2% | 3.4% / 6.0% | |
| New 13D stake | −1.1% / −4.3% | 5.7% / 5.6% | |
| Late filer (NT 10-K/Q) | −5.1% / −11.0% (t −4.3 / −5.4) | 5.7% / 7.0% | |
| Preliminary proxy (vote) | −0.1% / −2.8% | 5.4% / 7.6% | |

**Verdict: no precursor is a buy signal.**

- Late filers, preliminary proxies and 13D stakes do raise the odds of becoming a run-rate winner, by about 1.4–1.5× the base rate. But their mean return is negative.
- That is the same lottery-ticket shape as the signature model: these filings predict volatility, not direction.
- Form 425 and merger proxies are reliably negative. Late filers are a strong avoid (−11% over 13 weeks in the holdout).

### Combined book: best Sharpe we can build, and leverage (`research/combined_book.py`, `research/results/combined_book.txt`)

The question: what is the highest Sharpe we can build from the robust pieces, and how far does leverage take it toward 3%/week?

Setup:
- **Long core:** slow 12-1 momentum (shadow #4).
- **Exclusions:** names flagged by filings are dropped. The flags are offering (424B or item 3.02, last 8 weeks), late filer (NT 10-K/Q, last 13 weeks), distress (items 3.01/3.03/5.01, last 8 weeks) and merger (425/PREM14A/DEFM14A, last 13 weeks).
- **Short leg:** new bad-news flags, restricted to price ≥ $5 and ≥ $5M/day so the names are likely borrowable. Borrow costs 4%/yr and trading 15 bps per side. Squeezes are not clipped.
- **Selection:** the long variant and short leg were chosen on dev; the holdout was scored once.

**1. Filing exclusions hurt the momentum core.**

| Long book | Dev CAGR / Sharpe | Holdout CAGR / Sharpe |
|---|---|---|
| Core, no exclusions | 26.4% / 0.75 | 38.1% / 0.85 |
| Excluding offerings | 19.1% / 0.62 | 28.2% / 0.75 |
| Excluding offering + late + distress | 15.2% / 0.54 | 11.5% / 0.46 |

Momentum winners raise money while their stock is hot, and keep rising anyway. The offering drift is a small-cap, non-momentum effect.

**2. The bad-news short leg has no edge among borrowable names.**
- Raw short legs lose 11–20%/yr in both periods.
- Relative to borrowable stocks, the combined flags earn +3.8% to +8.9%/yr excess in dev (t 1.9–3.6), but only −0.3% to +2.6%/yr in the holdout (t ≈ 0). That is before the 4% borrow cost.
- The strong event-study drifts live in names you cannot borrow.

**3. The short leg works only as a hedge.**
- Long plus 0.5× late-filer shorts: Sharpe 0.82 dev / 0.91 holdout, versus 0.75 / 0.85 for long only.
- Correlation between the long book and the short leg is −0.57. The improvement is hedging, not alpha.

**4. Leverage** (financing at 6%/yr):

| Book | Leverage | Dev CAGR / max DD | Holdout CAGR / max DD |
|---|---|---|---|
| Long only | 1.0× | 26.4% / −46% | 38.1% / −42% |
| Long only | full Kelly (1.4–1.5×) | ~28% / −69% | 41.2% / −56% |
| Long + 0.5× short | full Kelly (1.8–1.9×) | 31.2% / −68% | 44.7% / −65% |
| Any | 3× | −8% to +15% / −86% to −98% | |

**Ceiling: about 0.5–0.7%/week geometric**, with 55–70% drawdowns at full Kelly. 3%/week (366%/yr) is about 5× beyond what any tested combination supports, and the core is the best of 106 designs, so live results will be lower.

**Not yet testable: contract-win overlay.**
- LLM judgements cover only a 17k-filing sample. Contract wins are 1.7% of 8-Ks: 281 events, mean +4–6% excess over 20 days, but median about 0 (a lottery-like skew).
- A portfolio test needs the reader run over every small-cap 8-K from 2024 to 2026.

### Contract-win book with full coverage (`research/contract_book.py`, `research/combo_contract.py`)

**Data**
- The LLM reader now covers every qualifying small-cap 8-K from Jan 2024 to Sep 2026: 32,783 filings, about 17.9k of them new. The backfill cost about $26 (Haiku 4.5 via the batch API).
- "Early" = Jan 2024 – Mar 2025, used for choosing. It may overlap the model's training data.
- "Holdout" = Apr 2025 – Sep 2026, scored once.

**Event level** (IWM-adjusted, from the entry open)
- The baseline is bad: the average small-cap 8-K filer trails IWM by −0.9% / −1.0% over 20 days (median −2%).
- All contract wins with direction ≥ 1:
  - Early: +1.3% (n 218).
  - Holdout: +14.1% (n 323), but that figure is driven by one +3,400% name (BNAI). Without it, +3.5% (t 1.9).
  - Removing the best 10 holdout events brings it to 0.
- Contract worth ≥ 10% of market cap: +5.3% early (n 74) and +4.1% holdout (n 100).
  - Against random draws of the same number of 8-Ks, p = 0.03 early and p = 0.04 holdout.
  - The holdout 5-day gain is +5.3%, median +2.0%, t 2.9.
- All other positive news types: −0.9% / +0.05%. Contract wins are the only good news that drifts.

**Portfolio** (buy at the first open after acceptance, hold 20 sessions, equal weight across open positions, 25 bps/side)

The best design on early data: contract value ≥ 10% of market cap, hold 20 sessions, no per-name cap (about 5 positions on average, invested 98% of days).

| | Early | Holdout |
|---|---|---|
| CAGR / Sharpe / max DD | 113% / 1.60 / −29% | 72% / 1.21 / −34% |
| 50 bps per side | 100% / 1.49 | 62% / 1.11 |
| 100 bps per side | 77% / 1.28 | 45% / 0.92 |
| Without the 5 best events (ex post) | 31% / 0.76 | 52% / 1.01 |
| Without the 10 best events | 9% / 0.44 | 29% / 0.73 |
| 20 random 8-K portfolios (mean, best) | Sharpe −0.47, 0.84 | Sharpe −0.04, 0.75 |
| IWM | 0.6% / 0.13 | 25.9% / 1.20 |

**Combined with the slow-momentum core** (weekly, Monday to Monday)
- Weekly correlation between the two is 0.54. Both are small-cap risk-on bets.
- Blending does not beat the contract book alone:

| Book | Leverage | Early CAGR / Sharpe / DD | Holdout CAGR / Sharpe / DD | Holdout per week |
|---|---|---|---|---|
| Momentum core only | 1× | 29% / 0.73 / −28% | 46% / 0.94 / −42% | 0.74% |
| 50/50 blend | 1× | 65% / 1.20 / −21% | 71% / 1.30 / −35% | 1.04% |
| Contract book only | 1× | 90% / 1.25 / −27% | 90% / 1.41 / −31% | 1.24% |
| Contract book only | Kelly from early (1.66×) | 113% / 1.21 / −43% | 137% / 1.37 / −48% | 1.67% |

(At weekly resolution the contract book's CAGR differs from the daily figure above because of how the windows are aligned.)

**Verdict**
- This is the first edge that beats random controls in both periods, survives 50–100 bps costs, and survives removing its best five trades.
- It is still a lottery-skewed book with about 5 names, about 180 qualifying events in 2.75 years (roughly 1.3 per week), and only one regime (a hot small-cap market, 2024–26).
- The ceiling moves from about 0.5–0.7%/week to about 1.0–1.7%/week. 3%/week is still not supported.

**Live check:** shadow #6's records already hold the contract value and market cap, so the ≥ 10% sub-book can be read out of its existing ledger with no change to the shadow.

### Contract wins out of sample, 2017–2023, plus filters, exits, sizing and hedge (`research/contract_filters.py`, `research/llm_counterparty.py`, `research/results/contract_filters.txt`)

**Data**
- The LLM read every qualifying small-cap 8-K from 2017 to 2023: 58,124 filings, about $83. Total coverage is now 93,006 filings from 2017-01 to 2026-09.
- A second read of all 1,577 contract wins (counterparty type, how firm the deal is, timing) cost about $2.
- **Caveat:** the SEC filing set only includes companies that still have a ticker, so 2017–23 is survivor-biased. The random-filing controls share that bias.

**A. Out of sample (2017–2023, never seen by the design)**
- **Event level:** contract wins ≥10% of market cap gained +1.8% (IWM-adjusted, 20 days) versus −0.3% for all 8-Ks (p = 0.04, median −1.1%). For all contract wins it was +0.45% (p = 0.10).
- **Book** (≥10% of market cap, 20-day hold, equal weight, no cap):

| | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Book, 2017–23 | 9.8% | 0.43 | −64% |
| IWM, 2017–23 | 7.3% | 0.41 | |
| Book, 2024–26 | 89% | 1.39 | |

- **Year by year:** −6%, −50%, +141%, +113%, +23%, −35%, 0%, then +165% (2024), +32%, +67%.
- Random-filing books averaged a Sharpe of −0.11 in dev (best 0.26).
- **Verdict:**
  - The edge is real but small: a percent or two over 20 days.
  - The book is a high-beta, small-cap lottery.
  - 2024–26 was an exceptionally kind regime for it.

**B. Filters (terciles chosen on dev)**
- **Adopted on dev** (worst group trails the rest at t < −2):
  - Drop the low third by contract value relative to market cap.
  - Drop US-federal-government customers.
- **On the holdout, both filters hurt:**
  - The federal effect reverses: −4.1% in dev, +3.6% in holdout.
  - Filtered base: holdout Sharpe 0.51, versus 1.39 unfiltered.
  - All contract wins filtered: holdout Sharpe 1.30, versus 1.55 unfiltered.
- No technical, fundamental or counterparty feature separates winners reliably. Holdout tercile means are dominated by a few lottery names.

**C. Exits, sizing, hedge (528 designs, chosen on dev)**
- **What works is breadth plus sizing:**
  - all contract wins, not only the ≥10% ones
  - inverse-volatility weights
  - a 25% cap per name
  - 10-day hold
- Dev-best design: dev Sharpe 1.04 (32%/yr, max DD −40%); holdout Sharpe 1.18 (47%/yr, max DD −35%).
- The median holdout Sharpe of the dev top-20 is 1.02.
- Early loser cuts and trailing stops are neutral.
- The IWM hedge lowers Sharpe (dev 1.04 → 0.96).

**Leverage on the dev-best design** (full Kelly from dev = 2.9×):

| Leverage | Holdout CAGR | Per week | Holdout max DD | Dev max DD |
|---|---|---|---|---|
| 1× | 47% | 0.74% | −35% | |
| 1.45× (half Kelly) | 62% | 0.93% | −48% | −56% |
| 2.9× (full Kelly) | 79% | 1.12% | −78% | −87% |

**Ceiling: about 0.7–1.0%/week at survivable leverage.** 3%/week is not supported by anything tested.

**Live tracking (Oct 11):** shadow #6 now runs a third book on the same judgements, `long_contract_div`. It is all contract wins (direction ≥ 1), held 10 sessions, weighted by 1/vol60 with a 25% cap per name, 25 bps per side. It is tracked as a daily NAV in `data/shadow_filings_state.json` (`div`) plus closed trades in the ledger. There is no extra LLM cost, and no orders are placed.

### Would reverting to the Apr–May version help? (`research/version_replay.py`, `research/probes/alpaca_equity.py`, Oct 16)

**The account itself**
- Apr 17 – May 29: +8.0%, versus SPY +6.5% and IWM +5.3%.
- Almost all of it came from one week: +12.1% in the week ending May 22. The picks that week were an AI/optical/space theme (FCEL +48%, PENG +18%, INTC +13%, MXL, POET, NBIS, LITE, RKLB, VECO and AAOI all up 7–13%). Without that week the stretch is about −4%.
- Peak $111,589 on May 22; −43.6% since.

**There is no single "version" to revert to.** Over Apr 20 – May 29:
- weights changed 8 times (Apr 20: catalyst 0.40 / sentiment 0.25 → May 22: momentum 0.35 / fundamentals 0.27);
- the ranking logic was redesigned on May 4;
- exclusion filters were added on Apr 20;
- the LightGBM model was retrained weekly.

**Replay.** Each snapshot's model, weights and feature list was re-scored on every logged week, using the top-10 forward 1-week return. In sample, every snapshot looks great: the current model averages +8.4%/week on the weeks it was trained on. After each snapshot's date (out of sample):

| Snapshot | Mean per week | Cumulative | Weeks |
|---|---|---|---|
| Apr 20 | −0.42% | −7% | 15 |
| May 08 | +0.04% | −1% | 13 |
| May 22 | −2.59% | −25% | 11 |
| May 29 | −2.45% | −22% | 10 |
| Jun 13 | −1.66% | −13% | 8 |
| Aug 01 | +1.17% | +5% | 4 |
| All names (benchmark) | +0.05% to +0.30% | | |

**Verdict:**
- The good stretch was a rising market plus one theme week, not a better version.
- The weekly self-refine/retrain loop fits the past and fails forward. That is why the stack is frozen.
- Reverting to May 22 would have lost about 25% more than the universe.

**Stops added to the revert test** (`research/stops_replay.py`). Each snapshot's top 10 was replayed out of sample, Mon open → Fri close, with the live rules: hard stop 7/10/15% by volatility, activation at max(3%, 0.75 × average win), then a trailing stop of min(vol/2, 15%).

| Snapshot | No stops | With stops | With stops, pessimistic | Universe |
|---|---|---|---|---|
| May 22, 10 weeks | −1.55%/wk (−15%) | −0.95%/wk (−9%) | −1.33%/wk (−13%) | +0.26%/wk |
| Apr 20, 14 weeks | −1.63%/wk | −0.88%/wk | −1.19%/wk | |

- The May 22 picks had 9 big winners (≥ +10%) and 14 big losers (≤ −10%).
- Stops cut losses by a third to a half but do not turn the book positive.
- In sample, the same snapshots show +4% to +8%/week, which shows how much the weekly retraining overfits.
- The live account also had stops through Jun – Aug and fell from $111.6k to about $74k.

### Theme detection (`research/theme_clusters.py`, `research/results/theme_clusters.txt`, Oct 16)

**Method.** Each week, stocks (≥ $3, ≥ $1M/day) are clustered by k-means on 26 weeks of market-residual weekly returns, about 40 names per cluster, 60–80 clusters, 2017–2026. Heat measures: 4- and 13-week cluster return, breadth, acceleration, breadth change.

**The detector finds real themes.** Snapshot for Oct 2, 2026:

| Theme | Members | 13-week return | Status |
|---|---|---|---|
| Tanker/LPG shipping | FRO, DHT, TNK, HAFN, INSW | +32% | breadth 0.62 |
| Crypto-treasury / AI-health | BMNR, SBET, MSTR, CRCL, TEM, GRAL | +23% | 4w +11% |
| Cloud/security software | NET, OKTA, S, RBRK, GTLB, ESTC | +21% | |
| Genomics tools | TWST, NTRA, ADPT | +14% | |
| Refiners | MPC, VLO, PSX, PBF, DINO | +16% | rolling over: 4w −5%, breadth 0.24 |
| Large SaaS | CRM, WDAY, MSFT, VEEV | | rolling over: 4w −8%, breadth 0.12 |
| IT services | ACN, CTSH, EPAM | | rolling over |

Some clusters are mixed, with no coherent story.

**It doesn't predict.**
- **Cluster level:** the rank IC of heat versus the next 4-week cluster excess return is −0.03 to +0.01 in both periods. The top-minus-bottom quintile spread is −0.3% to +0.3%.
- **Stock level** (Fama-MacBeth): cluster heat adds nothing beyond the stock's own momentum. Holdout 13-week cluster return: −2.8 (t −2.4); hot themes mean-revert.
- **Books** (4-week hold, 25 bps per side):

| Book | Dev 2017–23 | Holdout 2024–26 |
|---|---|---|
| Top 10 by own 13-week return, no clusters | −33%/yr | −58%/yr |
| Inside the hottest clusters | −5% to −10% | −14% to −29% |
| Early themes (accelerating, breadth rising) | −2.6% | +0.4%, max DD −43% |
| Equal-weight universe | +9.4% | +13.9% |

**Verdict.** Themes are easy to see and useful as a dashboard, but by the time a theme shows up in prices, its next month is a coin flip at best. This matches the earlier industry-momentum result. Horizons beyond 4 weeks were not tested.

### Forensic audit: what changed around the end of May? (Oct 16)

**Code, config and workflow changes, May 15 – Jul 5:**
- May 31: holiday guard now checks the *upcoming* week. This was a correct fix; without it, stale scores would have been traded.
- Jun 5–6: `collect_returns` day-count fix (training labels).
- **Jun 5: stage-5 rerank fix.** Live ranking changed from 0.5 × alpha + 0.5 × EV to alpha with a momentum tie-break.
- Weekly weight refinements.
- Model retrains on Jun 1, 6, 13 and 27. Feature importances drift gradually (volume surge / RS / momentum stay on top); there is no break.
- Nothing changed in `alpaca_trader` or `alpaca_monitor` in that window.

**The ranking change did not cause the decline** (`research/rank_formula_test.py`). Both formulas were re-scored on every logged week over the same eligible names; "actual" matches OLD before Jun 5 and NEW after.

| Ranking | Before (7 weeks) | After (11 weeks) | Pick volatility |
|---|---|---|---|
| OLD | +1.1%/wk | −2.04%/wk (−22%) | 13–20% |
| NEW | +2.6%/wk | −1.38%/wk (−15%) | 9–12% |

Reverting the formula would have lost more.

**What did change: the factor the model bets on flipped market-wide.** Top-decile excess return per week, across all logged names (`research/results/version_replay.txt` and inline check):

| Factor | Apr 10 – May 22 | May 29 – Oct 2 |
|---|---|---|
| Short-term RS | +0.61% | −0.89% |
| Regime-adjusted momentum | +0.79% | −0.37% |
| Volume surge | +0.01% | −0.39% |

The account's shape (up, then a steady slide) is the shape of short-term momentum in this period: a rebound rally in which chasing worked, then a market in which it reversed. The May 22 snapshot, frozen with no later changes, also lost −1.55%/wk out of sample.

### Themes at longer horizons (`research/theme_clusters.py` §1b/3b)

- Cluster-heat IC at 13 and 26 weeks is −0.05 to +0.02 in both periods.
- Books held 13–26 weeks do better than 4-week books but at best match the equal-weight universe:

| Book | Dev 2017–23 | Holdout 2024–26 |
|---|---|---|
| Early themes, 26-week hold | 10.2%/yr, Sharpe 0.50, max DD −41% | 8.7%/yr, Sharpe 0.46 |
| Equal-weight universe | 9.4%, Sharpe 0.51 | 13.9%, Sharpe 0.79 |

**Verdict:** "early" detected from price alone carries no edge. Getting in before a theme forms needs non-price information.

### Do the small fixes add up? (`research/avoid_stack.py`, `research/results/avoid_stack.txt`, Oct 16)

**Fixes tested** (each known at formation): skip the 5 biggest 1w/4w gainers, skip the top volatility decile, skip late filers, distress 8-Ks, merger filings, offerings, and LLM bad news; a 10% hard stop; for LIVE only, hold while ranked ≤ 50. Each fix was run alone, then as a greedy stack chosen on dev, and then with all fixes on.

**LIVE proxy** (weekly top 10 by the live-style composite):

| | Dev 2017–23 | Holdout 2024–26 |
|---|---|---|
| Base | −3.4%/yr, Sharpe 0.02, max DD −65% | +0.7%, Sharpe 0.22, max DD −56% |
| All fixes on | +1.4%, Sharpe 0.18, max DD −45% | +12.8%, Sharpe 0.58, max DD −24% |

- No single fix moves dev Sharpe by more than about ±0.07, except the stop, which is −0.22 dev but +0.44 holdout.
- **The fixes do add up**, by about +0.2 to +0.35 Sharpe, but only to roughly market level. They turn a losing book into a mediocre one; they do not create an edge.

**SLOW momentum** (shadow #4):

| | Dev 2017–23 | Holdout 2024–26 |
|---|---|---|
| Base | 26.4%/yr, Sharpe 0.75, max DD −46% | 38.1%, Sharpe 0.85, max DD −42% |
| All fixes on | 12.7%, Sharpe 0.54, max DD −35% | 23.9%, Sharpe 0.81, max DD −23% |
| Greedy pick: 10% stop only | 27.5%, Sharpe 0.81, max DD −34% | 34.9%, Sharpe 0.85, max DD −38% |

- On a book that already works, the filters remove winners: strong momentum names often have offerings, high volatility and bad-news days.
- Only the stop helps robustly, and mostly through drawdown.

### Filing flow as an early theme signal (`research/theme_filing_flow.py`, `research/results/theme_filing_flow.txt`)

**Data:** 20,752 positive and 19,620 negative LLM-judged 8-Ks, 2017–2026, mapped onto the weekly price clusters.
- The share of cluster members with recent good news, its change, bad-news share, and "good news spreading while price is still cold": IC at 4, 13 and 26 weeks is −0.08 to +0.04, with no significant t in either period.
- Book (members with positive filings inside the top-3 "quiet good" clusters, 26-week hold): dev 6.0%/yr (Sharpe 0.35), holdout 12.6% (Sharpe 0.65). Equal-weight universe: 9.4% (Sharpe 0.51) / 13.9% (Sharpe 0.79).

**Verdict:** filing news spreading through a theme does not lead its prices. Company-level contract wins still work (see the contract-win book); theme-level aggregation does not add anything.

### Strategy stack: diversify the surviving books, then lever (`research/strategy_stack.py`, `research/results/strategy_stack.txt`, Oct 16)

**Components** (weekly, 2017–2026, 364 dev weeks / 143 holdout weeks):

| Component | Dev | Holdout |
|---|---|---|
| SLOW (slow momentum + 10% stop) | Sharpe 0.81, 27%/yr | Sharpe 0.86, 36%/yr |
| CONTRACT (dev-chosen contract-win book) | 0.99, 32% | 1.28, 52% |
| ETFTREND (63-day, long-only) | 0.63, 15% | 1.39, 31% |
| CRYPTO (trend 50-day; cash before Jun 2021) | 0.31 | 0.73 |
| LIVEFIX (live proxy + all fixes) | 0.18 | 0.58 |

- Correlations are low: SLOW–CONTRACT 0.28–0.32, CONTRACT–ETF 0.21–0.39, CRYPTO about 0 to everything.

**Mixes** (weights chosen on dev, scored on holdout):

| Mix | Weights | Dev | Holdout |
|---|---|---|---|
| Max Sharpe | SLOW 0.19, CONTRACT 0.35, ETF 0.19, CRYPTO 0.27 | Sharpe 1.18, 22.7%/yr, max DD −21% | Sharpe 1.64, 39.3%/yr, max DD −14% |
| Max Sharpe, no crypto | SLOW 0.26, CONTRACT 0.48, ETF 0.26 | 1.16, 29.7%, max DD −24% | 1.50, 46.4%, max DD −22% |
| Equal weight | | 0.99 | 1.51 |

- LIVEFIX gets zero weight in every optimized mix.

**Leverage** (Kelly from dev; financing 6%/yr):

| Mix | Leverage | Dev | Holdout |
|---|---|---|---|
| Max Sharpe | 2.0× | 37%/yr (0.61%/wk), max DD −42% | 74%/yr (1.08%/wk), max DD −27% |
| Max Sharpe | 2.58× (half Kelly) | 44%/yr (0.70%/wk), max DD −53% | 95%/yr (1.29%/wk), max DD −35% |
| No crypto | 2.0× (half Kelly) | 49%/yr (0.77%/wk), max DD −47% | 87%/yr (1.21%/wk), max DD −42% |

**Caveats**
- The components were included because they survived earlier tests, so holdout Sharpe (1.5–1.6) is optimistic; dev (about 1.15) is the better guide.
- 2024–26 was a strong market for every component.
- The contract book's 2017–23 data has survivorship bias.
- Crypto is zero before 2021, which flatters its dev risk.
- Leverage needs margin; a Roth cannot lever, and small caps have higher margin requirements.

**Verdict:** this is the first combination at or above Sharpe 1.1 on data it was not designed on. About 1%/week becomes plausible at about 2× leverage, with roughly −40% to −50% drawdowns.

### Data checks: universe coverage and intraday stops (Oct 16)

**Research panel: complete.** It is built from Alpaca's full asset list: 15,008 symbols, 12,598 with data (NASDAQ 6,847, NYSE 2,891, AMEX 343, plus ETF venues), active and inactive. **Limit:** Alpaca keeps only about 2,500 delisted names, so 2017–2020 under-represents companies that later died. That flatters every backtest (small caps most), but comparisons between strategies are unaffected.

**Live universe: 23% of eligible stocks were missing.** Of 2,221 eligible names (non-fund, price ≥ $5, ≥ 500k shares/day), 520 were absent. Causes:
- **Suffix filter bug (212 names, since day one).** `01_universe.py` dropped every ticker ending in W, U or R. That rule is meant only for NASDAQ 5-letter warrants, units and rights. It removed MU, PLTR, PANW, INTU, MSTR, UBER, COHR, GLW, NOW, TER, MPWR, SCHW, LOW, KR, PWR and others.
- **Stale sources (about 300 names).** The primary NASDAQ list was a years-old GitHub dataset, so new listings never entered: SPCX, CBRS, HONA, FDXF.
- **Share-volume gate.** It also dropped expensive, heavily traded names such as MELI and FICO.

**Fix (Oct 16):**
- W/U/R rule applied to 5-letter tickers only.
- Alpaca's live asset list (NASDAQ/NYSE/AMEX, tradable, non-fund) unioned in.
- Live nasdaqtrader list read before the stale one.
- A stock also passes the volume gate at ≥ $25M/day.
- Dry run of tonight's exact upsert path (`research/probes/universe_dryrun.py`): 1,811 → **2,438** (+636, −9). All 25 spot checks present. Stage 1 takes 13 minutes.
- Health-check baseline updated.

**Effect on the trial:** the live system and shadows #1, #4 and #5 (which take their universe from `scores_final.csv`) change universe from the Oct 17 run. The Jan 5 live-vs-shadow comparison should start from that run.

**Intraday:** the earlier 5-minute study (`research/intraday_rules.py`, weekly top-15 picks, 2017–2026) found that intraday exit rules move results by only 0.1–0.2% per trade.

**The 10% stop on slow momentum fails on daily bars** (`research/stop_daily_check.py`). It was re-simulated on Alpaca daily OHLC for all 5,070 holding-weeks, 2017–2026; gaps fill at the open, intraday breaches at the stop.
- It triggers in 27.9% of name-weeks: 1,264 intraday, 152 gaps. Gap fills average −14%; the worst is −43%.
- Stopped weeks average a −10.6% fill. Without the stop, those same weeks returned −10.0%, so the stop just locks in the dip.

| SLOW | Dev Sharpe | Holdout Sharpe |
|---|---|---|
| Stop, daily-validated | 0.71 | 0.69 |
| No stop | 0.75 | 0.85 |
| Stop, weekly-low approximation | 0.81 | 0.85 (wrong) |

The stop is dropped. The strategy stack, re-run with SLOW without a stop, is unchanged in substance:
- Max-Sharpe mix: dev 1.16, holdout 1.65.
- At 2×: dev 35.7%/yr, max DD −44%; holdout 75.6%/yr (+1.09%/wk), max DD −27%.

### More books, and hedging out the market (`research/extra_books.py`, `research/hedged_stack.py`, Oct 16)

**Adding PEAD and INSIDER** (pre-registered dev picks, full 2017–2026 series):
- PEAD: Sharpe 0.77 dev / 0.78 holdout. INSIDER: 0.75 / 0.63.
- Combined max-Sharpe is unchanged: 1.16 dev / 1.64 holdout.
- Long US-stock books correlate 0.6–0.8 with SLOW and with each other through shared market exposure. More stock books ≠ more independent books.

**Market-hedged stock books.** Each stock book was regressed on IWM over dev; hedged book = book − β × IWM (short IWM futures or ETF).
- Betas: SLOW 0.80, PEAD 0.70, INSIDER 0.78, CONTRACT 0.74.
- Hedged books keep most of their alpha. CONTRACT hedged: Sharpe 0.91 dev / 1.10 holdout. SLOW hedged: 0.63 / 0.71.
- Hedged-book correlations: CONTRACT_h vs the others 0.03–0.10; ETF trend vs hedged books about 0.

| Mix (dev-chosen weights) | Dev Sharpe | Holdout Sharpe | Max DD (dev / holdout) |
|---|---|---|---|
| Unhedged max-Sharpe (before) | 1.16 | 1.64 | |
| IWM-hedged books + ETF trend + crypto | **1.41** | 1.68 | −11% / −10% |
| IWM-hedged books + ETF trend, no crypto | **1.39** | 1.54 | −12% / −13% |
| SPY-hedged mixes | 1.08–1.17 | | |
| SPY + IWM two-factor hedge | 1.6 | | |

- The SPY-hedged mixes give no gain.
- The two-factor hedge's 1.6 is **rejected**: its collinear betas (long SPY, short more IWM) embed a large-over-small bet that paid in both periods.

**Realistic weekly return**, IWM-hedged mix with no crypto, live Sharpe haircut 20% (about 1.12):

| Leverage | Per week | Max DD |
|---|---|---|
| 2× | 0.42% | −29% |
| 3× | 0.51% | −45% |

- Unhedged at the same haircut: 2× 0.53%/wk, max DD −55%.
- Hedging buys smoothness: about the same return for 10 points less drawdown, or a little more return at the same drawdown. It is not a step change in return.
- Needs a margin account that can short IWM (or IWM futures). A Roth cannot.

### DoD contract awards as a 5th book (`research/dod_awards.py`, Oct 16): **fails**

**Data.** war.gov/defense.gov block cloud servers (403); the RSS gives titles only, back to Aug 2024. History therefore came from USAspending: 47,817 new DoD awards ≥ $7.5M, 2017–2026. The base obligation date is treated as the announcement day; one month-page (2025-09, page 3) failed to fetch.

**Coverage**
- Only 16% of awards mapped to a current ticker, covering 145 companies.
- They are dominated by the big primes: LMT 1,201 events, BA 814, GD 716, LDOS 445, BAH 388.
- 58% of events lack a share count, so award size relative to market cap is unknown.
- Small contractors are mostly private, or already file 8-Ks, which the contract-win book catches.

**Event drift** (IWM-adjusted, 20 days):

| Award size vs market cap | Dev | Holdout |
|---|---|---|
| 0.1–1% | +0.75% (t 2.7) | +0.51% (t 0.7) |
| 1–5% | +1.77% (t 2.7) | −1.14% |
| ≥ 5% | −0.71% | −0.71% |

**Book** (dev-chosen: awards ≥ 0.1% of market cap, 20-day hold): Sharpe 0.53 dev / 0.27 holdout.
- IWM-hedged: 0.31 / −0.24.
- Weekly correlation with the contract-win book: 0.25.

**In the hedged stack**
- The optimizer gave DOD 16–19% weight on its low dev correlation. Dev Sharpe rose (1.39 → 1.44), but holdout fell (1.54 → 1.44 without crypto; 1.68 → 1.59 with crypto). This is the classic sign of a weak book overfitting the mix.
- **Excluded.** `strategy_stack.py` only includes it with `INCLUDE_DOD=1`.
- The live DoD feed question is moot.

### Shadow #7 live: the strategy stack (`automation/shadow_stack.py`, `.github/workflows/shadow_stack.yml`, Oct 16)

**What it tracks.** The four books that exist live, with frozen weights and IWM hedge ratios fitted on 2017–2023 (`research/hedged_stack.py` subset):

| Sleeve | Weight | Hedge |
|---|---|---|
| Slow momentum (shadow #4) | 14.5% | − 0.798 × IWM |
| Contract-win diversified book (shadow #6) | 35.3% | − 0.743 × IWM |
| ETF trend | 29.4% | none |
| Crypto trend | 20.8% | none |

- Backtest of exactly this mix: Sharpe 1.35 dev / 1.72 holdout. Without crypto (rescaled weights): 1.32 / 1.57.
- PEAD and INSIDER are not in it; they have no live implementation.

**How it is marked**
- Weekly, over shadow #4's Monday-to-Monday window, running Tuesday 10:10 ET with a Wednesday backup.
- The ETF and crypto books are recomputed from Alpaca daily bars each run (no state). On research data the replicas give ETF Sharpe 0.69 / 1.26 and crypto 0.52 / 0.77.
- The contract sleeve comes from shadow #6's daily NAV.
- NAV is tracked at 1×, 2× and 3× (6%/yr financing), plus each sleeve.
- Started at shadow #4's Oct 12 mark; first mark about Oct 20. The live data path was dry-run on Oct 16: all 41 ETFs and 16 coins present.

**Review.** The Jan 5, 2027 review task was updated to evaluate #4–#7. It also uses the Oct 19 universe-fix start for the live system and the price-based shadows.

## Kalshi passive market making, weather markets (Oct 16, 2026): `research/kalshi/market_making.py`

**Question.** Can a small account earn the spread by resting orders on Kalshi daily-high temperature markets? The aim is an uncorrelated income stream that funds the stack.

**Data.** 5,950 settled markets (14.7% stratified sample of 40,582, 14 cities, Jun 2024 to Oct 2026): 3.0M trades, 110M contracts. Dev runs to Oct 1, 2025; the holdout runs after. The maker side of each trade is held to settlement. All 14 series use the `quadratic` fee type: takers pay 0.07·p(1−p) and makers pay nothing.

**Maker edge, cents per contract**

| Cut | Dev | Holdout |
|---|---|---|
| All trades | 1.39 | 0.76 |
| Price buckets | 0.5 to 6.6, all positive | 0.4 to 1.8, all positive |
| Hours-left buckets | 1.2 to 2.2, all positive | 0.4 to 1.1, all positive |
| Single trades at the touch | 0.35 | 0.49 |
| Sweeps (taker crosses >1 price level) | 2.90 | 1.29 |
| Deepest level of a sweep | 3.39 | 1.95 |

- **Price impact after the trade.** On single trades the price ten minutes later has barely moved (about 0, maker's view). On sweeps the maker gains about 1.2–2.8¢ by then.
- **By month.** Every month from Jun 2024 to Oct 2026 is positive, but the edge is shrinking: about 2–4¢ in mid-2024, about 1¢ in 2025, about 0.6–0.9¢ in 2026, and 0.06¢ in a partial Oct 2026.
- **By city.** The newer series show losses in the holdout (TBOS, TDC, TSATX, TEWR).

**Passive book vs placebo.** The placebo shuffles the taker side within each market.

| Book | Sharpe, dev | Sharpe, holdout | Return on fill collateral |
|---|---|---|---|
| Real | 4.3 to 7.2 | 4.8 to 7.4 | about 4–9%/month |
| Placebo | −1.9 to +0.3 | −4.0 to +2.0 | about −2% to +2.6%/month |

- The real book wins on about 63–67% of days.
- The real book beats the placebo in every share and cap setting, in both periods.
- This is the first test in this program whose holdout result is not ambiguous.

**Why the backtest overstates it**

1. **The edge is the average maker's.** The book assumes we get our share of every trade. Back-of-queue fills are worse, and the queue at the touch is where pros cancel ahead of news. Single-trade fills at the touch earn only about 0.4¢.
2. **Resting orders tie up collateral.** Kalshi reserves funds for unfilled orders too. The capital figure above counts only fills, so true capital needs are 2–3x higher.
3. **The edge is decaying** as competition grows: from about 2.4¢ to about 0.6¢ in two years.
4. **Fees can change.** If Kalshi starts charging makers 0.0175·p(1−p), that takes about 0.2–0.4¢ off.

**Where the money is.** Most of the edge sits deeper in the book, where impatient sweeps hit. That suggests quoting a few cents behind the touch rather than at it: fewer fills, much better fills, and less racing against pros.

**Realistic size.** After a 50–75% haircut, expect roughly 1–3% a month on deployed capital:
- $2k → about $20–60/month
- $5k → about $50–150/month
- $10k → about $100–300/month

Capacity is not the binding constraint at these sizes. The deep-level pot alone is about $1–2M/yr market-wide.

**Verdict.** The edge is real and robust: positive in every bucket, period and month, and far above the placebo. It is small per contract and shrinking, and its size for a back-of-queue participant is unknown. The next step is a small live test (deep quotes, a few hundred dollars, about 8 weeks), measuring realized cents per contract against the 1.3–1.9¢ deep-level benchmark.

**Bot built (Oct 16):** `kalshi_mm/bot.py`, workflow `kalshi_mm.yml`, run in dry or demo mode only.

**How it quotes.**
- Rests post-only orders 2¢ behind the best bid on both sides.
- 5 contracts per quote, with a cap of ±25 contracts per market.
- $200 total budget, $40 daily loss stop.
- Quotes only 1–48h before close, in the 8 series that were positive in the holdout.
- A KILL file cancels everything. All orders are cancelled on exit.
- Logs quotes, orders, fills and snapshots for the realized-edge measurement.

**Tests.** Offline tests pass for quote logic and Ed25519/RSA signing. A 6-minute dry run against live public books parsed every series and produced about 40 two-sided quotes per cycle (≈$190 collateral if all rested). Touch spreads were typically 1¢.

**Next.** Demo-exchange run once demo API keys are stored as the GitHub secrets KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY. After that comes a funded small live test on an always-on host.

## IRA-compatible stack and risk overlays (Oct 17, 2026): `research/ira_stack.py`

**Caveat found.** The ETF-trend research book runs at a mean 3.2x gross (95th percentile 4.6x). The "1x" stack is therefore really about 1.6x gross, and financing on the extra ~0.6x (about 3–4%/yr) is not charged in the stack backtests or in shadow #7. This should be corrected before any capital decision.

**IRA version.** No margin, no shorting, no crypto, so:
- the IWM hedge becomes an inverse-ETF hedge, which costs cash: w·(1+β) of capital per book;
- crypto becomes BTC+ETH ETFs only;
- the ETF sleeve is capped at 1x gross.

**Results** (dev 2017–23 | holdout 2024–26):

| Version | Dev CAGR | Dev Sharpe | Dev max DD | Holdout CAGR | Holdout Sharpe | Holdout max DD |
|---|---|---|---|---|---|---|
| Taxable 1x | 19.9% | 1.35 | | 34.0% | 1.72 | |
| Taxable 2x | 32.7% | 1.14 | −28.5% | 63.7% | 1.55 | −24.8% |
| IRA 1x, ETF capped | 11.4% | 1.16 | −10.6% | 21.3% | 1.70 | −9.7% |

**IRA conclusion.** Sharpe survives, but returns are about one-third of taxable 2x. After roughly 24% annual tax, taxable 2x still compounds about twice as fast. The Roth is better used for a plain index core; the strategy stays taxable unless the IRA uses futures for leverage.

**Risk overlays on taxable 2x.**
- **Volatility targeting, 8-week lookback (best on dev):** Sharpe 1.14→1.40 dev, 1.55→1.61 holdout. Max drawdown −28.5→−23.5% dev, −24.8→−19.4% holdout. Average leverage in the holdout was 1.47.
- **13-week lookback:** holdout Sharpe 1.77, DD −19%.
- **Halve leverage beyond a 10/15/20% drawdown:** lowers dev Sharpe (1.03–1.10) and gives mixed holdout results. Rejected.
- **Recommendation:** vol targeting (8–13 week lookback) for any leveraged deployment.

## Theme / industry ETF momentum (Oct 17, 2026): `research/etf_momentum.py`

**Setup.** 4,144 ETFs (3,186 active plus delisted fund-named), excluding leveraged, inverse, option-income, buffer and bond funds. Point-in-time liquidity of at least $2M/day and one year of history. Grid of 120 configurations:
- lookback: 1, 3, 6 or 12 months, and 12 months skipping the last month;
- top 5, 10 or 20;
- weekly or monthly rebalance;
- optional absolute filter and de-duplication;
- 10 bps costs.

**Results.**

| | Dev Sharpe | Dev CAGR | Holdout Sharpe | Holdout CAGR | Max DD |
|---|---|---|---|---|---|
| Best on dev (12-1 month, top 20, monthly) | 0.67 | 15% | 0.44 | 10% | about −36% (both periods) |
| Grid median | 0.25 | | 0.73 | | |
| IWM-hedged best | | | 0.20 | | |

- Across configurations, dev and holdout Sharpe correlate at only 0.2: the ranking is mostly luck.
- Every configuration is below buy-and-hold SPY on a risk-adjusted basis in dev, with deep drawdowns (−30 to −45%). The absolute filter made no difference: some ETFs always have positive 12-month returns.

**Verdict.** Rejected. Industry/theme momentum via ETFs does not beat the market, and with the market hedged out it has almost no Sharpe left. That is consistent with the earlier stock-level theme work (detectable but not predictive). Survivorship bias would only make the true result worse.

**Stack check.** Adding the IWM-hedged ETF momentum book to the 4 live books:
- the dev optimizer gives it 9.7%;
- stack Sharpe goes 1.35→1.36 in dev and 1.72→1.64 in the holdout;
- its correlation with ETFTREND is 0.31.

Not added.

**Demo-exchange run (Oct 17, 5 h, $90 budget).**
- 560 cycles; 691 orders placed and 632 replaced as books moved.
- 21 fills (105 contracts), all as maker, zero fees.
- Max position 15 contracts (cap 25); one harmless "post-only would cross" rejection.

**Bugs found and fixed.**
1. Fill records overwrote the log timestamp.
2. Resting orders were not trimmed when fills pushed total exposure past the budget (peak $104 vs $90).
3. Cancel-all left 2 orders behind. It now handles per-order errors, re-checks, and confirms `resting_after: 0`; verified on demo.
4. Settlement records are now logged hourly, for realized P&L per contract.

**Status.** Mechanics verified. Next: a funded real account (a few hundred dollars) plus an always-on host for an 8-week edge measurement.

## Audit of failed and accepted tests (Oct 17, 2026)

Every test in this log was re-read against its code. Findings, most important first.

### Errors that change accepted numbers

**1. The stack's Sharpe was overstated.**
- The ETF-trend book holds 3.2x gross on average. The stack's account gross was 1.5x on average (95th percentile 1.9x) with no financing charged.
- Net of 6% financing on the true gross:

| | Before | Net of financing |
|---|---|---|
| Stack 1x Sharpe (dev / holdout) | 1.35 / 1.72 | 1.14 / 1.56 |
| "2x" Sharpe (dev / holdout) | | 0.93 / 1.40 |
| "2x" CAGR (dev / holdout) | | 25% / 55% |

- "2x" is really about 3x gross on average (max 4.3x).
- All Sharpes in this log are computed on total, not excess-over-cash, returns. Relative comparisons hold; absolute levels are a further ~0.1–0.2 too high.
- **Redo:** re-optimize the stack with the ETF sleeve capped at 1x gross (or financed) and excess-return Sharpes, then update shadow #7 expectations.

**2. Look-ahead in the weekly simulators.** `backtest.py` and `avoid_stack.sim` only consider names whose next-week return exists. That silently skips stocks about to be delisted or halted, and the +150% clip removes huge winners.
- Measured on SLOW (`scratchpad/lookahead.py`):
  - removing the look-ahead and the clip: dev Sharpe 0.78→0.79, holdout 0.96→1.05;
  - booking every non-acquisition-like disappearance (178 of 471 name-weeks) at −100%: dev Sharpe 0.76.
- **Immaterial for SLOW.** The same pattern exists in other weekly studies; with an effect this size it changes no verdict.

### Tests that were incomplete and deserve a better redo

| Test | What was missing | Could the verdict flip? | Redo |
|---|---|---|---|
| Contract-win book 2017–23 (35% of the stack) | The SEC filing set only includes companies that still have a ticker; 2017–23 filers that later delisted are absent | Moderate: dev Sharpe could fall | **Yes.** Add delisted filers via the EDGAR full index, mapped to Alpaca inactive tickers by name |
| Earnings drift (PEAD) | "Surprise" is the 2-day announcement return (EAR), not earnings vs expectation; weekly entry | Could strengthen a hedged PEAD book (currently 0.77/0.78) | **Yes.** XBRL seasonal-EPS surprise plus EAR, daily entry, IWM-hedged |
| Same-day Kalshi weather | Used METAR obs (routine + special), not the 1-minute data the NWS daily max is built from; the true max falls between obs and suffers C→F rounding | Possibly. The market won by a lot, but the model had a handicap traders don't | **Yes, cheap.** IEM 1-minute ASOS plus the 6-hour max groups. Also useful to the MM bot (avoid quoting after the max is effectively known) |
| Insider buying | No routine vs opportunistic split (Cohen-Malloy-Pomorski), the documented source of insider alpha | Possibly a modest book | Yes, lower priority |
| Kalshi cross-market scan | Rate-limit failures cached as "no data" | Unknown; nothing was measured | Rerunning (fixed) |
| XBRL accruals | Broken (1% coverage: cash flow is year-to-date) | Unlikely (factor decay) | Low priority |
| DoD awards | 16% ticker mapping, dominated by primes; obligation date ≠ announcement time | Unlikely: small contractors already 8-K, and those are in the contract book | No |
| Straddles | Traded/VWAP option prices, not bid/ask; holdout only | Unlikely: realized/implied was the same as controls | No |
| IPO lock-ups | Estimated dates, weekly data | Unlikely | No |

### Failures that hold up on review

These were checked and the verdicts stand:
- weekly-turnover momentum;
- regime / macro / options / flow switches (multiple splits, 60-year test);
- rank tranches and bands;
- post-scoring filters;
- large-cap reversal;
- themes and industry momentum (stocks, clusters, filing flow, ETFs; 4–26 week horizons);
- spike signatures (direction);
- precursors;
- avoid filters on slow momentum;
- the 10% stop (daily-validated);
- AI headline classification (clean post-cutoff fail);
- the forecast-based Kalshi weather model;
- ETF theme momentum.

### Systemic limits (not fixable with free data)

- **Survivorship:** about 417 delisted names are missing, and Alpaca keeps limited delisted history. This flatters small-cap long books.
- **One split:** a single dev/holdout split with a short (2.75-year), very bullish holdout that has been looked at many times. Holdout Sharpes are optimistic. Year-by-year (walk-forward) tables and a deflated Sharpe should accompany any capital decision.

## Audit redo 1: the stack, rebuilt honestly (Oct 17): `research/stack_v2.py`

**Changes.**
- ETF and crypto sleeves capped at 1x gross;
- idle cash earns T-bills;
- borrowing at 6%;
- Sharpe on excess-over-T-bill returns;
- IWM hedge betas from dev (SLOW 0.798, CONTRACT 0.743);
- weights re-chosen on dev.

**Sleeves (excess Sharpe, dev / holdout).**

| Sleeve | Dev | Holdout |
|---|---|---|
| SLOW_h | 0.59 | 0.62 |
| CONTRACT_h | 0.85 | 0.97 |
| ETFTREND (1x) | 0.41 | 0.83 |
| CRYPTO (1x) | 0.29 | 0.69 |
| PEAD_h | 0.57 | 0.27 |
| INSIDER_h | 0.55 | 0.02 |

Dev correlations between the four live sleeves are all within ±0.14.

**Mixes at 1x.**

| Mix | Weights | Dev | Holdout |
|---|---|---|---|
| Shadow #7 frozen weights | | exSh 1.08, 15.9%/yr, DD −14% | exSh 1.34, 27.8%/yr |
| Re-optimised live 4 | SLOW 0.10, CONTRACT 0.24, ETF 0.50, CRYPTO 0.15 | exSh 1.12, 12.6%/yr, vol 9.4%, DD −9.7% | exSh 1.38, 22.4%/yr, DD −8.5% |
| Adding PEAD/INSIDER | | 1.19 | 1.30 |

Adding PEAD/INSIDER is the overfitting pattern again; rejected.

**Year by year (re-optimised, 1x, total return %).**

| 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|---|
| +12.0 | +6.6 | +11.0 | +27.3 | +9.3 | +2.1 | +21.9 | +19.6 | +32.8 | +9.8 |

No losing year.

**Leverage (borrowing at 6% lowers excess Sharpe as leverage rises).**

| Leverage | Dev | Holdout |
|---|---|---|
| 2x fixed | 18.4%/yr, exSh 0.90, DD −22% | 39.1%/yr, exSh 1.31, DD −18% |
| **Vol-target, avg 2x on dev (8-week lookback)** | **23.2%/yr, exSh 1.16, DD −16%** | **30.2%/yr, exSh 1.35, DD −13%** (avg leverage 1.41) |

**Deflated Sharpe** (full 2017–26 Sharpe 1.20, assuming trial-Sharpe dispersion of 0.5):

| Designs tried | P(the Sharpe is real) |
|---|---|
| 100 | 0.42 |
| 500 | 0.15 |
| 2,000 | 0.05 |

This research program has tried hundreds of designs. Even honest numbers carry a real chance of being selection luck; shadow #7's live record is the arbiter.

**Bottom line.**
- Realistic, pre-tax: about 20–25%/yr (0.35–0.45%/wk) at vol-targeted ~2x, with drawdowns around −15 to −20%. That is about the previous "realistic" figure, now reached with smaller drawdowns and without hidden leverage.
- Shadow #7 still uses the uncapped ETF sleeve and the old weights. **Proposed change (needs Connor's approval):** cap the ETF sleeve at 1x, use the re-optimised weights, and add 8-week vol targeting.

## Audit redo 2: earnings drift with a real surprise (Oct 17): `research/pead_v2.py`

**Surprise.** XBRL seasonal EPS surprise (XSUE): 61,577 company-quarters, 5,406 companies; usable events 48,988.
- **Event study, 13-week excess by XSUE quintile.** Dev Q5−Q1 is −0.35% (t −0.7); holdout +0.94% (t 1.8). XSUE and EAR both positive: −0.33% vs −0.82% for the rest in dev; −0.38% vs −0.44% in the holdout.
- **Grid (48 designs).** The dev pick is still the old announcement-return score: exSh 0.83 / 0.86, unhedged, beta 0.61. XSUE designs: median 0.60 / 0.80.
- **Hedged, in the honest stack.** Dev 1.13→1.36 but holdout 1.37→1.24: overfit. Rejected.

**Verdict.** With a proper surprise measure, earnings drift adds nothing here. XSUE is only known at the 10-Q filing; 8-K exhibit EPS would be earlier, but the dev spread is not even positive.

## Audit redo 3: same-day Kalshi weather on official-basis temperatures (Oct 17)

**Data.** `research/fetch_asos1min.py`: IEM 1-minute ASOS, 14 stations, Jun 2021 – Oct 2026, stored as 5-minute running averages (ASOS's reporting basis).

**Agreement with the settled high** (full-day max inside the settled bucket):

| Source | Inside | Below | Above |
|---|---|---|---|
| METAR (original test) | 64.7% | 35.2% | 0.1% |
| Raw 1-minute max | 63.8% | | 24.8% |
| **5-minute average** | **87.5%** | 11.9% | 0.6% |

So the original test **was** handicapped: METAR missed the high a third of the time.

**Rerun** (`ASOS_FILE=asos5avg.parquet python research/kalshi/sameday.py`).
- **Model log loss improved sharply** (16:00 holdout 0.13; raw 1-minute was 0.57), but the market is still about 3x better: 0.04 at 16:00, 0.12 at 14:00 (model 0.20).
- **Dev-chosen trade** (14:00, edge threshold 0.10): dev +2.6¢/contract (t 2.5), holdout +0.5¢ (t 0.6).
- 16:00 shows +1.4¢ in the holdout (t 2.2), but it was not the dev pick (dev +0.5¢).

**Verdict.** Much closer to break-even than the original −2 to −3¢, but no robust edge: traders price the live readings better than this model. **Closed fairly this time.**

**Side benefit for the MM bot.** After about 14:00 local, the market knows the day's high. Resting quotes late in the day face informed takers, so the bot's 1-hour cutoff should probably move earlier for markets whose high is already in. Candidate for the live test.

**Shadow #7 v2 (approved by Connor, Oct 17).** `automation/shadow_stack.py` now also tracks the honest stack, beside the original (v1) and from its next mark:
- ETF and crypto sleeves capped at 1x gross, idle cash at about 3.8%;
- weights SLOW_h 0.104 / CONTRACT_h 0.238 / ETF 0.504 / CRYPTO 0.154;
- NAV at 1x and with 8-week vol targeting (leverage = 0.020698 / std of the last 8 weekly returns, clipped 0.5–3x, seeded with research returns Aug 7 – Sep 25), with 6% financing on the borrowed part.

Ledger rows carry a `v2` block; state is in `state["v2"]`. The Jan 5 review should compare v1, v2 at 1x and v2 vol-targeted.

## Bugs that helped: what was really going on (Oct 17)

Connor's rule: when a "bug" beats the corrected version, find out why before discarding it. Every such case in this log:

**1. Weekly-low stop approximation (looked better than the daily-validated stop).** It booked gap-downs at exactly −10%, a fill that cannot happen. Nothing to keep.

**2. ETF-trend book at 3.2x with no financing.**
- Re-tested with financing: leverage inside the low-vol ETF sleeve vs the same total volatility from account-level borrowing gives dev excess Sharpe 1.04 vs 0.95 and holdout 1.34 vs 1.33.
- A marginal risk-parity benefit. The apparent gain was mostly free borrowing. Nothing to adopt.

**3. The live system's mistimed trades (Tue-open in, Mon-open out beat the designed Mon-close timing, Round 1). A real effect.**

On the 5-minute data (the proxy's weekly top-15 picks, 2017–26):

| | Night (close→open) | Day (open→close) |
|---|---|---|
| Dev | +11.8 bp (t 9.4) | −11.7 bp (t −5.7) |
| Holdout | +16.4 bp (t 4.7) | −18.4 bp (t −4.4) |

- Monday daytime is the worst (−18 / −31 bp). This is the documented overnight/intraday "tug of war" in momentum stocks.
- After hedging out IWM's own overnight return: +7.3 / +11.5 bp per night (t 4.2 / 2.5).

**Execution lessons** (`scratchpad/timing.py`, weekly panel):
- **Never buy hot momentum names at the open.** LIVE book, buys at Monday open instead of close: dev −15.2%/yr vs −3.4%, holdout −11.3% vs +0.7%.
- **For slow momentum, sell timing barely matters.** Selling at Monday open hurts (0.75→0.67) because names being sold have already lost momentum. Tuesday-open exits are neutral.

**Overnight-only book (5-minute data, weekend excluded).** Before costs Sharpe 1.98 / 1.85. Very cost-sensitive:

| Cost per side | Dev Sharpe | Holdout Sharpe |
|---|---|---|
| 2 bp | 1.31 | 1.43 |
| 5 bp | 0.30 | 0.80 |

Closing and opening auction orders (MOC/MOO) fill at the official print with no spread, so real costs could be about 1 bp.

**Full test running:** `research/overnight_momentum.py`. Official daily open/close prints, full universe, three signals × top 10/20/50, with and without the weekend night, 0/1/2/5 bp, IWM-hedged, dev pick then holdout once.

## Overnight-only momentum, full test (Oct 17): `research/overnight_momentum.py`

**Setup.** Official daily open/close prints (Alpaca SIP), 2017–26, weekly signals, buy at the close and sell at the next open every night.

**Average night / day per basket (bp, before costs).**

| Basket | Night dev | Night holdout | Day dev | Day holdout | Beta to IWM night |
|---|---|---|---|---|---|
| LIVE top 10 | 12.3 | 12.4 | −6.9 | −6.6 | 0.7 |
| 12-1 top 50 | 13.8 | 19.7 | −2.6 | −11.8 | 1.0 |
| Last-week hot | about 0 | | | | |

IWM's own night is +6.3 bp in both periods.

- The 5-minute preview (+12/+18 bp on weekday nights alone) was overstated: the last 5-minute bar is not the official closing print.
- The weekend night is the biggest single night (LIVE: 16.9 / 26.2 bp vs 11.0 / 8.4 on weekday nights).

**Breadth.**
- The median name earns only about the market's night (+5–9 bp).
- Winsorised 10/90 baskets earn +10–13 bp; 52–54% of names rise overnight.
- The best 20 of 2,456 nights supply 26–31% of the cumulative total.
- **Real but right-tail-heavy.**

**Books** (Sharpe on daily returns; hedged = minus beta × IWM night):

| | Hedged Sharpe dev | Hedged Sharpe holdout | Raw return dev | Raw return holdout |
|---|---|---|---|---|
| Dev pick, 1 bp/side (LIVE top 10, with weekend) | 1.27 | 0.77 | 28%/yr | 26%/yr |
| Dev pick, 2 bp/side | 0.85 | 0.52 | | |
| 12-1 top 50, 1 bp (not the dev pick) | 1.07 | 1.76 | | |

At 0 bp the hedged Sharpe is about 1.7–1.8 in dev; the edge roughly halves with every basis point of cost.

**In the honest stack (weekly, IWM-hedged).**
- At 1 bp the dev optimizer gives it 55%. Dev excess Sharpe 1.12→1.64, but holdout **1.38→1.37**. At 2 bp: holdout 1.38→1.29.
- Correlation with the other books: −0.11 to +0.25.
- Classic dev-only gain: **does not improve the stack out of sample.**

**Against the pre-registered kill criteria.**
1. At least +5 bp hedged: borderline pass (winsorised hedged dev 3–6 bp, holdout 7–8 bp).
2. Survives 2 bp in both periods: weak pass (0.85/0.52).
3. Not a handful of names: partial (right-tail driven, median ≈ market).
4. Dev pick holds in holdout: degraded (1.27→0.77).

**Verdict.** A real, documented effect, but modest here, and decisively dependent on costs of 1 bp or less per side. Not a stack improvement on current evidence. The only way to settle the cost question is real auction fills: a paper shadow (MOC buys / MOO sells of the LIVE top 10 nightly on Alpaca paper) would measure fill price vs the official print. Proposed, not built.

## Audit redo 4: contract-win book with delisted filers (Oct 17)

**Added.** 591 later-delisted issuers (29% of inactive symbols matched by name via the EDGAR master index), 6,544 small-cap 8-Ks 2017–23, judged by the same LLM prompt (`llm_delist/`). Contract wins in dev went to n 1,138.

| | Survivors only | With delisted filers |
|---|---|---|
| Dev contract-win drift (20d, vs all 8-Ks −0.28%) | | +0.48% (p 0.105) |
| Base book (≥10% of mcap, equal weight), dev Sharpe | 0.43 | 0.42 |
| Dev-best design (all wins + filters, 10-day, 1/vol, 25% cap), dev Sharpe | 1.04 | **0.92** |
| Same design, holdout Sharpe | 1.18 | 1.17 (46%/yr) |

- Hedged CONTRACT sleeve: dev excess Sharpe 0.85→0.72, holdout 0.97→0.95.
- **Honest stack:** re-optimised live 4 at 1x goes from 1.12 / 1.38 to **1.04 / 1.37**. Vol-target ~2x: dev 21.1%/yr (DD −16%), holdout 29.0%/yr. Re-optimised weights barely move (0.115 / 0.214 / 0.506 / 0.165 vs shadow v2's 0.104 / 0.238 / 0.504 / 0.154), so shadow #7 v2 is left as is.
- Deflated Sharpe: P(real) is 0.35 at 100 trials and 0.11 at 500.

**Verdict.** Survivorship flattered the contract book's dev result by about 0.1 Sharpe. The holdout is unaffected, and the conclusions hold. Coverage is still partial (71% of delisted symbols unmatched), so the true dev figure is likely a little lower again.

**Audit redos complete:** stack rebuilt; PEAD (real surprise) rejected; same-day weather closed fairly; contract survivorship measured. The cross-market Kalshi scan is re-running after its third bug fix (sampling dropped the window column).

**Shadow #8 live (approved by Connor, Oct 17): overnight-only momentum.** `automation/shadow_overnight.py`, `.github/workflows/shadow_overnight.yml`.
- **Picks:** LIVE-proxy top 10 each Friday, through the same panel builder and research code as #4.
- **Marking:** every trading morning, last official close → today's official open.
- **NAVs:** raw at 1 and 2 bp/side, IWM-hedged (beta 0.67) at 1 bp, the basket's daytime return, and IWM's own night.
- **Schedule:** cron weekdays 11:20 ET plus an 18:20 ET backup; idempotent and catches up missed nights. First entry close Mon Oct 19, first mark Tue Oct 20.
- **No orders:** official prints stand in for auction fills, so fill quality still needs real orders to measure.
- **Dry run on Oct 12–16** (formation Oct 9: MRNA, HPE, DELL, PAYS, TEAM, OKTA, ECO, LITE, HZO, ASX): nights −184, −203, −370, −6 bp; raw NAV 100 → 92.5. Verified against raw prints (DELL 507.93 → 471.21, LITE 896 → 845 overnight on Oct 14–15) as a genuine momentum gap-down week, not a bug. It is a reminder that the left tail is real.
- **Fix found:** the free SIP feed refuses queries ending within 15 minutes of now, so the shadow queries bars up to 20 minutes ago.

**Earnings nights in the overnight book (Oct 17).** Earnings name-nights are those whose open is the first reaction session.

| Basket | Share of name-nights | Share of the cumulative night return | Mean earnings night (sd) |
|---|---|---|---|
| LIVE top 10 | 1.2% | 18% | +179 / +202 bp (sd 7–11%) |
| LIVE top 50 | 1.3% | 11% | |
| 12-1 top 50 | 1.2% | 7% | |

Momentum names mostly beat on earnings.

**Excluding them lowers Sharpe** (LIVE top 10 hedged at 1 bp: 1.27→0.94 dev, 0.47→0.18 holdout) **and does not change the worst night or drawdown** (worst −7.8% / −12.2% either way). The book's tail risk is market/sector-wide momentum gaps (like Oct 14–15), not single-name earnings.

**Keep earnings nights.** Manage the tail with breadth (top 50 was steadier), the IWM hedge and possibly sector caps.

**Sector cap (Oct 17, before shadow #8's first mark; approved by Connor).** At most N picks per SEC SIC 2-digit industry; unknown SIC is never capped.

| Basket | Cap | Hedged Sharpe dev | Hedged Sharpe holdout | Holdout worst night | Holdout max DD |
|---|---|---|---|---|---|
| LIVE top 10 | none | 1.27 | 0.77 | −12.2% | −28% |
| LIVE top 10 | 3 | 1.26 | 0.78 | −8.0% | |
| LIVE top 10 | **2 (dev best)** | **1.39** | **0.87** | **−7.4%** | **−25%** |
| 12-1 top 50 | 10 / 5 | no gain | | | |
| LIVE top 50 | 5 | 1.26 | 1.04 | | |

Shadow #8 now applies cap 2 (SIC from SEC submissions, cached in state; workflow passes SEC_USER_AGENT). Dry run: OKTA (the third SIC-73 name) was replaced by IOVA; the Oct 12–16 week was still −7.2%, a broad momentum gap that sector caps cannot diversify away.

## Drawdown protection beyond stack-level vol targeting (Oct 17)

Connor asked for confidence measures and back-off rules to protect gains. Tested on the honest 4-book stack with the current v2 8-week stack-level vol target (~2x).

| Version | Dev exSh | Dev CAGR | Dev DD | Holdout exSh | Holdout DD |
|---|---|---|---|---|---|
| **Current v2** | **1.06** | **21.1%** | **−16.4%** | **1.31** | −12.5% |
| + per-book vol targeting (8/13/26 wk) | 0.85–0.97 | | −15 to −18% | 0.95–1.25 | −11 to −13% |
| + per-book equity-curve filter (cut a book to 50% when its trailing 13/26-week return < 0) | 0.76–0.86 | | worse, −18 to −21% | 1.25 | |
| Same filter cutting to 0% | 0.34–0.51 | | | | |
| Both combined | 0.67 | | | 1.02 | −9.9% |

The equity-curve filters cut books after losses and miss the rebounds. That matches the factor-momentum finding: momentum outperforms *after* lagging.

**Conclusion.** Every added back-off rule lowers dev Sharpe. Together with the earlier failures (regime labels, 12 stress features, options/credit/short-volume signals, fixed stops, drawdown de-levering) and the successes (vol targeting; vol-managed + macro momentum, shadow #3):
- **drawdowns are not predictable from these signals, but volatility is**;
- the stack-level vol target already in shadow #7 v2 is the protection that works.

## Kalshi cross-market scan: results (Oct 17)

> **Superseded by the full-coverage update below (Oct 18).** The numbers in this section came from 1,970 of 6,654
> series with volume: a sampling bug silently skipped every series whose volume was concentrated in a few markets.: `research/kalshi/mm_scan.py`, `scan_analysis.py`

Coverage: 10,429 of 15,187 series cached (the rest were request failures, not cached; re-dispatched to fill in).
1,970 series had enough settled volume (>= 20k contracts): 18,652 sampled markets, 12.2M trades, 1.45B contracts.
Samples are volume-weighted, 12 markets per (series, first 90 days / last 90 days), up to 3,000 trades per market
(the most recent ones), so the data is mostly 2026 and mostly high-volume markets.

**Average maker edge** (maker side of every trade, held to settlement, clustered by market): +0.71c, t 5.8.
- By series age at close: 0-30 days +1.14c (t 6.5); 31-90d +0.30 (t 1.3); 91-180d +1.17 (t 3.1); 181-365d +0.12 (t 0.4); >1y -0.57.
- Same series, early vs late window (n = 479): median 0.94c -> 0.39c. New-series edge is real but modest, and
  "new" is confounded with one-off political/election series.
- Categories: Sports +0.72 early / +0.52 late (most of the volume); Financials late -6.4c (t -6.1, makers lose);
  Weather +0.2 (small sample here; the dedicated study found 0.76c holdout); Politics +1.4 early / -0.2 late.
- Per-series rankings are mostly noise: of 289 eligible late-window series, 11% have t > 2 and 7% have t < -2
  (12 markets each). The stand-outs with big t are multi-outcome longshot markets (NBA triple double t 13.8,
  Korn Ferry t 9.8, NFL first TD t 8.0, DP World R1 leader t 8.1, F1 pole t 6.3) -- the same longshot bias below.
- Sports by contract type: spreads/totals early +2.1c / +1.2c but negative late; props +0.9 / +1.9; game winners +0.3 / +0.4.
  NHL (-4.9c, t -3.5), Serie A and CS2 negative; MLB +2.7, NBA +1.6 (t ~2.5).

**Calibration (all categories): a strong favourite-longshot bias.**

| YES price | 0-5 | 5-10 | 10-20 | 20-35 | 35-50 | 50-65 | 65-80 | 80-90 | 90-95 | 95-100 |
|---|---|---|---|---|---|---|---|---|---|---|
| settled minus price (c) | -1.3 | -4.3 | -5.9 | -4.9 | -1.0 | -0.7 | +6.1 | +3.7 | +3.0 | +0.5 |

**Longshot fade** (buy NO as a taker when YES is cheap, at the prices where takers actually sold YES, held to
settlement, after the 0.07*p*(1-p) taker fee; clustered by event):

| YES price | events | fade c/contract | t | return on NO cost |
|---|---|---|---|---|
| 0-5c | 9,855 | +0.85 | 5.0 | 0.9% |
| 5-10c | 8,863 | +2.61 | 6.1 | 2.8% |
| 10-20c | 9,449 | +4.17 | 6.6 | 4.9% |
| 20-35c | 9,602 | +3.91 | 3.6 | 5.4% |
| 35-50c | 8,934 | -1.25 | -0.7 | -2.2% |

Positive in 2026 for sports (+2.4 / +3.9 / +3.3c in the 5-10 / 10-20 / 20-35 buckets) and non-sports (+4.1 / +6.6 / +8.7c),
and in 2023-2025 where there is data. Negative only in Weather, Mentions, Commodities.
Caveats: essentially one year (2026) of data; sampled markets are high-volume; returns are per settlement cycle
(days for sports), with a lose-the-stake tail (NO at ~85c pays -85c when the longshot hits); depth at the bid
limits size. This matches the published Kalshi maker/taker findings (cheap contracts are overpriced).

**Proposals (not built; need Connor's approval):**
1. Paper shadow "longshot fade": each day, markets closing within 24h with YES bid 5-35c, ex Weather/Mentions/
   Commodities: record buying NO at the NO ask, fixed $ per event, mark at settlement. 8 weeks before real money.
2. Bot tilt: in the weather MM bot, quote only the NO side when YES is under 35c (sell longshots, don't buy them):
   maker edge plus the bias, no taker fee.
3. Weekly new-series watch (`NEW_ONLY=1`): edge in a series' first 30 days is ~1.1c vs ~0.3c later. Worth a weekly
   run, but the current NEW_ONLY path re-lists all 15k series; it needs a filter on series first-close date
   before fetching trades to be cheap.

### Full-coverage update (Oct 18)

**Bug:** `scan_series` drew markets with pandas' weighted `sample(replace=False)`, which raises when one market carries
more than 1/n of a series' volume. The exception was caught and printed, not cached, so ~4,750 series -- many of the
biggest one-off events, Mentions and soccer markets -- were silently skipped (the follow-up run looked like
"request failures"). Fixed with numpy sequential weighted sampling; code errors now surface as ::warning:: and a
coverage notice. Re-run: 15,187 series attempted, 0 request failures, 0 code errors; 6,654 series with data.

**Maker edge, full coverage:**
- By series age at close: 0-30d +1.16c (t 17.6); 31-90d +0.63 (t 4.5); 91-180d +0.88 (t 4.4); 181-365d +0.33 (t 1.7); >1y +0.63 (t 2.1).
  New series still pay most, but the drop after launch is gentler than the partial sample suggested.
- Sports +1.24c early (t 11.5) / +0.68 late (t 3.1); Mentions +1.37 / +0.99; Entertainment +1.24 early but 0.02 late;
  Weather +0.33 / +0.54; Financials ~0 (the -6.4c late figure was a small-sample artifact).

**Longshot fade, full coverage** (taker-NO prices, after fee, clustered by event):

| YES price | events | fade c/contract | t | return on NO cost |
|---|---|---|---|---|
| 0-5c | 21,771 | +1.06 | 13.3 | 1.1% |
| 5-10c | 19,399 | +3.33 | 10.7 | 3.6% |
| 10-20c | 20,362 | +4.86 | 10.5 | 5.7% |
| 20-35c | 20,236 | +5.36 | 7.4 | 7.4% |
| 35-50c | 18,993 | +0.36 | 0.3 | 0.6% |

YES 5-35c by half-year: 2024-H2 +6.6c, 2025-H1 +5.3, 2025-H2 +4.4, 2026-H1 +4.3, 2026-H2 +3.6 (t 3-7 each; earlier halves are thin
and mixed). Positive every half since 2024 but slowly shrinking -- the same decay seen in the weather maker edge.
By category (5-35c): Sports +4.9c (t 14.1, 36k events), Politics +5.1, Elections +4.5, Entertainment +3.5, Crypto +5.5,
Weather +2.8 (now positive), Mentions +1.7 (weak); Financials, Commodities, Science/Tech and World are zero or negative.

**Verdict:** both findings are stronger on full data. The proposals stand, with the exclusion list updated to
Financials / Commodities / Science-Tech / World (Weather and Mentions no longer excluded, Mentions low priority).

## Longshot fade as a strategy: does NOT hold (Oct 18): `research/kalshi/longshot_fade.py`

The scan's longshot result (+3-5c per contract) was contract/trade-weighted across all trades. Tested as rules one could run:
- **First-time-in-band entry** (per market, $10 NO, max 3 per event): taker -6 to -8% per trade on 2026, t -10 to -16.
  Cause: contracts that end up winning pass through every price band on the way to $1, so "ever traded in band"
  over-selects winners. Same data, per market: in-band YES settled 26% vs ~17c average price; markets that FELL into
  the band settled YES 29%. Market-equal fade -10c (holdout) vs +2c trade-weighted vs +4c contract-weighted.
- **Snapshot rule** (no path selection: price = last trade at close-h, h = 1-96h; bands 5-15 ... 20-40c):
  dev-picked cell 5-15c at 1h: dev +4.7% (t 7.6) -> holdout -0.8% (t -1.7) at a 0.5c half-spread, -1.4% at 1c.
  Holdout is negative in essentially every cell (-0.3% to -6%). Maker version (rest an offer at p, filled only if a
  later taker buys YES at >= p) is worse: -5% holdout at the dev pick -- adverse selection.
- **New-series market-making edge, re-weighted per market:** 0-30d series +0.25c (t 1.9) on 2026 vs +1.21c contract-weighted;
  older series ~0. The contract-weighted figure is driven by large fills a fixed-size bot won't get.

**Verdict:** rejected. The bias exists in aggregate but sits in large trades / paths that a rule can't select in advance.
None of the three proposals (paper fade, bot tilt, new-series watch) go ahead. Lesson for every Kalshi study here:
report per-market (and per-fill) results, not only contract-weighted ones. The weather MM edge (1.39c / 0.76c) was
measured contract-weighted too; the real-money pilot's per-fill P&L is the test that settles it.

Demo vs prod (probe, Oct 18 afternoon): same tickers, but demo 24h volume per weather market 20-390 contracts vs
2,000-23,000 on prod, and demo books were empty while prod had 1,000-4,000 contracts at the best bid. Demo is a
separate, thin exchange: real events and settlements, not real prices or fills.

## Weather MM edge re-measured per fill (Oct 18): `research/kalshi/mm_perfill.py` -- holds

After the longshot result flipped under re-weighting, the weather MM edge was re-measured the way the bot earns it
(bot filters: 1-48h to close, 3-97c; dev < 2025-10-01 <= holdout; 3.0M trades, weather sample).

| holdout | per contract | per trade | per fixed-5 fill | per market (t) |
|---|---|---|---|---|
| maker at the touch | +0.90c | +0.81c | +0.95c | +0.94c (7.8) |
| deep fill, 2c behind (sweep reached our level) | +2.85c | +2.86c | +2.85c | +3.55c (14.4) |

Weighting barely matters here (unlike the longshot fade): the edge is spread evenly across trades and markets.
Deep fills are certain when a sweep trades beyond our level (price priority), so the fill proxy is sound; the edge is
net of the adverse selection in those sweeps. Positive in every price bucket; by time to close, 6-48h carries it
(6-12h +2.2c, 12-24h +3.7c, 24-48h +2.4c) while 3-6h is ~0 (thin, 600 fills) -- consider MIN_HOURS = 6 for the pilot.
Verdict: the real-money pilot is justified; expect ~+2-3c per filled contract before any queue/latency slippage.

## Weather expansion: which other temperature series earn (Oct 19): `research/kalshi/weather_expand.py`

The bot's per-fill deep-fill edge (2c behind, size 5, 6-48h to close) on every daily high/low temperature series,
~52-60 sampled settled markets per series from the last 365 days, full trade histories. Control = the current series
measured identically (Austin was dropped by a filter bug -- `US$` also matched KXHIGHAUS -- so it was not re-tested).

- Control (7 current cities): per-market edge 0.9-5.7c, median 3.5c; t 0.4-3.4 (only ~52 markets each). Dallas 5.7c,
  Philadelphia 3.8, Miami 3.7, NYC 3.5, LA 1.5, Chicago 1.0, Denver 0.9.
- 36 other series with data: 67% positive per market; t > 2 in 3, t < -2 in none (about 1 would pass t > 2 by luck).
  Daily HIGHS median +2.2c per market; daily LOWS +1.0c with thin volume (1-4 fills per market) -- skip the lows.
- Pass (per-market > 1.5c, t > 2.5, recent-180d > 0.5c, >= 5 fills/market): **Atlanta high (KXHIGHTATL) +5.5c t 3.4,
  Boston high (KXHIGHTBOS) +4.8c t 2.7, Las Vegas high (KXHIGHTLV) +4.1c t 3.2** -- volume ~4-7k contracts/market,
  similar to the control; positive in both halves of the year.
- Next tier (positive, not significant): Houston, Minneapolis, Phoenix, Seattle, San Francisco highs (2-3c, t 1.0-1.7).

Verdict: add ATL, BOS, LV. At the current $90 budget the bot is already capital-bound (resting orders fill the budget),
so more series spread the same money over more markets rather than adding profit; the value shows up when the budget grows.

## Intraday strategies: all rejected (Oct 19): `research/intraday_strats.py`

5-minute SIP bars 2017-2026 for SPY/QQQ/IWM/DIA, 10 sector SPDRs, 48 large stocks; costs 1 bp/side (index ETFs),
2 bp (others). Dev 2017-23 pick, holdout 2024-26.
- A intraday momentum (first half hour -> last half hour, index ETFs): dev pick Sharpe -0.37; holdout -2.06, gross -1.3 bp/trade.
  No variant has positive gross returns in the holdout worth trading (best +0.7 bp). The published effect is gone.
- B short-term reversal (stock drops vs SPY, buy, SPY-hedged, hold 15-60 min): dev pick +3.9 bp net (Sharpe 1.35) ->
  holdout -8.5 bp net, gross -2.5 bp. Holdout gross is ~0 across all variants.
- C opening-range breakout: negative in dev and holdout (holdout gross +0.4 bp median, below any realistic cost).
Verdict: at retail costs there's nothing at 5-minute to 1-day horizons here; gross edges are 0-1 bp vs 2-4 bp round-trip costs.

## Intraday follow-ups (Oct 19): `research/intraday_timing.py`

**Overnight execution timing** (live-proxy top-10 picks, 5-min bars, weekday nights; vs buying at the close auction
and selling at the open auction): every alternative is worse or equal. Moving the entry earlier (15:30 / 15:45 / TWAP of
the last 30 min) changes ~0 bp; moving the exit later costs 4-6 bp per night by 09:35-10:00 and ~12 bp by 10:30 (dev
t -2 to -5; holdout same sign). Conditional exits (hold if gapped up / first bar up) are worse still. And leaving the
auctions adds spread. Verdict: the MOC-in / MOO-out design is already the best execution; a live version must use
auction orders (sell ON the open, not a market order a minute after it).

**Multi-timeframe trend + dip** (62 liquid symbols; long when the 60-min trend is up, the prior 15 min dipped and the
current bar turned up; optional 20-day filter; hold 30/60 min or to the close): the dev pick (60m > 0.4%, dip > 0.3%,
hold to close, long only, no daily filter) earned **+3.5 bp gross per trade in dev and +3.7 bp in the holdout** --
the only intraday signal here with a stable pre-cost edge -- but -0.5 / -0.3 bp net at 2 bp per side (break-even
~1.8 bp per side), holdout daily Sharpe -1.7. The 20-day filter and short side hurt. Not tradable at the assumed cost;
the open question is real cost: entry is a market order (half-spread, ~0.3-2 bp across these names) and the exit could
be a closing-auction order (no spread). Worth one cost-realistic re-score before dropping it.

**MTF cost re-check** (`research/mtf_costcheck.py`; same signals, entry half of a 1c spread + 0.2 bp, exit at the closing
auction + 0.2 bp): net +1.9 bp/trade dev, +2.9 bp holdout, but t 0.75 / 0.93 and daily Sharpe -0.3 / -1.1 (returns are
noisy and day-clustered). Rejected -- not distinguishable from zero.

## Live alpha / EV scores, tested directly on the logged weeks (Oct 19)

performance_log.csv: 23 formation weeks with realised forward returns (Sep 2025 - Oct 2026), ~1.8k names/week.
| Score | mean weekly rank IC | t | top-decile excess/wk | top-10 excess/wk |
|---|---|---|---|---|
| alpha (LightGBM) | -0.009 | -0.3 | -0.02% | -0.49% |
| EV | -0.021 | -0.6 | -0.42% | -1.12% |
| 50/50 alpha+EV | -0.015 | -0.5 | +0.08% | -0.94% |
| raw momentum signal | -0.036 | -1.3 | -0.21% | -1.43% |
- alpha is not calibrated: names scored 0.95-1.00 beat the weekly average 44% of the time (excess -0.36%/wk).
- The EV veto points the wrong way: EV > 0 names -0.25%/wk excess vs +0.08% for EV <= 0.
- So no weighting of alpha and EV can help -- neither carries signal in the live period (consistent with the
  postmortem: the factor the model bets on, short-term RS/momentum, reversed from late May). 23 weeks is short, but it
  is every live week there is. The stack's edge comes from the separately tested books, not these scores.

## Alpha / EV rebuilt (Oct 19): `research/alpha_v2.py` -- both versions fail the holdout

Price/volume features a weekly shadow could compute (r1w, r4w, r13w, 12-1 / 6-1 / 3-1 momentum, 52-week-high
distance, 12-week vol, trend, log $ volume, log price), research universe, target = next 4-week excess return.
Books: top 20, 4-week overlapping tranches, 15 bp/side, excess vs the equal-weight universe.

| Book | dev 2017-23 excess/yr (Sharpe) | holdout 2024-26 excess/yr (Sharpe) |
|---|---|---|
| v2a LightGBM (rank target, trained 2017-22, early-stopped 2023) | +20.2% (1.41) | **-6.5% (-0.58)** |
| v2b dev-IC-weighted blend (log price, log $vol, -r4w, -vol12, hi52, -r1w) | +0.9% (0.13) | **-11.2% (-0.90)** |
| 12-1 momentum, same book | +5.6% (0.33) | -4.7% (0.03) |
| LIVE proxy, same book | -3.1% (-0.06) | +5.9% (0.39) |
- Dev 4-week rank ICs are all small (|IC| <= 0.034, t <= 2.3): no price/volume feature predicts 4-week relative returns
  reliably. The ML model's dev Sharpe is fit, not edge.
- Calibration does not survive: holdout names with alpha_prob 0.55-0.60 beat the universe 43% of the time; EV deciles
  invert (dev top decile +1.09% per 4 weeks -> holdout -0.26%). The EV > 0 filter changed nothing (all top-20 had EV > 0).
Verdict: a better alpha/EV at the weekly-to-monthly horizon can't be built from price/volume; the edges that hold are
specific setups (overnight, contract filings, slow momentum's long holds), not a general score. No shadows built.

## Trading after the station has decided a bucket (Oct 19): `research/kalshi/obs_decided.py`

3M weather trades, 14 stations, 2021-2026. A bucket is decided when the high so far already exceeds its top (certain NO)
or a "greater than" threshold (certain YES); readings must be 10 / 30 min old at the trade; margin 0 / 1 F.
- **Defense (maker side):** in decided markets makers are on the losing side for only 1-7% of contracts (vs ~40% in
  open markets) and earn +1.0-1.2c (holdout) vs +0.7c in open markets. No filter needed -- the bot isn't being
  picked off there (and MIN_HOURS=6 already avoids most of it).
- **Sniping with the precise 5-minute averages** (the official basis, but assembled after the fact): certain-NO
  buckets still traded at ~75c on 351 city-days in 2025-26, +22.6c per contract after fees, wrong 0.13%. Looks huge,
  but those readings are not public in real time.
- **Sniping with real-time hourly METAR** (what a bot can actually read): the certain side trades at 98-99c almost
  always; only ~100 trades on 13 city-days in 2025-26 were at <= 97c (+7.7c). Total edge ~+1c/contract at 99c, i.e. the
  market prices public readings within minutes.
Verdict: no snipe with public data. The gap between the two runs is the value of faster/more precise readings than
the hourly METAR (5-minute ASOS feeds exist via NWS/MADIS in whole degrees C, commercial feeds in finer detail) --
untested whether any real-time feed is precise and fast enough; that would be its own project.

## Non-weather expansion (Oct 19): `research/kalshi/other_expand.py` (weather_expand on 147 candidates)

The bot's deep-fill test (2c behind, size 5, held to settlement) on 147 recurring non-sports/politics series
(112 with enough fills), windows 6-48h (the bot's) and any time >= 6h before close (needed for weekly/monthly markets).
Control: the weather highs, same run, median +2.6c per market.
- Broadly positive: 69% of series positive per market; t > 2 in 21, t < -2 in 5 (6h+ window; ~3 each by chance).
  By category (median per market): Commodities +6.2c, Economics +4.0c, Crypto +2.6c, Entertainment +2.5c;
  Financials -2.1c and Science/Tech -4.3c negative.
- Strongest with real fills (t clustered by event; monthly series have few events, so t is generous):
  WTI oil weekly range (KXWTIW) +10.3c t 7.4 (4 fills/market), LA monthly rain (KXRAINLAXM) +6.9c t 7.0 (5),
  Bitcoin monthly one-touch (KXBTCMAXMON) +5.3c t 3.7 (9, high volume), SpaceX launch count (KXSPACEXCOUNT) +3.9c t 3.0
  (34), CPI (KXCPI) +6.5c t 2.3 (22). Mentions (Trump/Fed word counts) also score high but are live-event markets
  with informed traders and thin books -- excluded. Central-bank decisions score high t but < 1 fill per market.
- Practicality: these are weekly/monthly, so the bot would quote them for days to weeks (capital tied up longer) and
  needs a per-series quoting window (MAX_HOURS is global, 48h). Their fills per market are lower than weather's.
Verdict: real second-tier opportunity. Add when the bot's budget outgrows the weather markets (resting + positions
regularly below budget). Shortlist: KXWTIW, KXBTCMAXMON, KXSPACEXCOUNT, KXCPI, KXRAINLAXM. Needs a per-series window.

## Kalshi Liquidity Incentive Program (Oct 19): `research/probes/kalshi_incentives.py`, `lip_capture.py`

GET /incentive_programs (public): 6,735 active liquidity programs across 600 series, ~$100 per market per weekly period
typical; every daily high/low weather market (all cities incl. ATL/BOS/LV and the lows) pays $20 per market per ~26 h
period (~$22/market/day), target size 1,000, discount 50% per tick below the reference price. Runs to Jan 1, 2027.
Scoring (help.kalshi.com): per second, per side, reference price = level where cumulative size from the best bid
reaches target/5; size counts until the target is reached; full credit at/above the reference, 0.5^ticks below;
reward split by share; snapshot void unless both sides have >= target.
Order-book probe (58 open weather markets, 15 min, our hypothetical 5-contract order):
| order placement | reward per contract-day | capital per contract | reward / capital per day |
|---|---|---|---|
| at best bid | 1.28c | 49c | 2.6% |
| 1c behind | 1.30c | 57c | 2.3% |
| 2c behind (the bot today) | 1.26c | 59c | 2.1% |
- The reference price usually sits below the best bid (deep books), so the bot's 2c-behind quotes already qualify.
- By price (at the bid): 5-10c 0.8c/day, 20-80c 1.7-2.1c/day, > 90c 1.25c/day; snapshots void ~30% of the time in
  the tails (thin side).
Implication (unverified until Kalshi pays): the bot's resting orders (~$90 at ~50c = ~180-200 contracts) may earn
~$2-2.5/day from the program on top of the trading edge -- comparable to or larger than the spread edge at this size,
and it rewards keeping the budget fully resting, which argues for more markets (lows, ATL/BOS/LV all pay) before
larger size. Caveats: my reading of the scoring rules; competition changes daily; Kalshi can change/end the program.
Verification: rewards shown in the Kalshi account after payout.

## Polymarket US and cross-venue arbitrage (Oct 19): `research/probes/pmus_markets.py`, `xvenue_arb.py`

- Polymarket US (CFTC-regulated, available in Texas) has a public gateway + authenticated trading API (Ed25519 keys).
  20,000+ open markets: sports (futures, props, game lines) and politics, a few macro (Fed, GDP); essentially no
  weather (1 temperature market) and no CPI. Taker fee 0.0695 p(1-p) (~Kalshi's). Maker rebate (-0.0125) was made
  volume-tiered on Oct 19, 2026: nothing below $10M/month. Liquidity rewards exist (e.g. NBA props $25-300 per period,
  target 1,000-1,500). -> No natural home for the weather bot.
- Cross-venue game-winner arbitrage (75 NBA/NFL/college games matched, 20 min, 1,341 snapshots): PM vs Kalshi mid
  correlation 0.95, median gap 0.5c (p90 1.5c); best net of both taker fees median -6.4c per contract. The only
  "positive" case (+45c, one college game) was a team-code mismatch between venues, not a price gap. No arbitrage.
