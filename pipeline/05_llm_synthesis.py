"""
Stage 5 — LLM Synthesis
========================
Takes the top 50 tickers by alpha score and sends each through
a structured Claude API call that reads all available signal
context and outputs:

    - A final conviction adjustment (+/- 0.0-0.15)
    - A one-line thesis (why this ticker, why this week)
    - Key risk flag (if any)
    - Confidence level (high / medium / low)

For all remaining tickers, a fast rule-based thesis generator
writes a one-liner based on which signals fired — no API call.

Everyone gets a thesis. Top 50 get the LLM version.

Reads:   data/scores.csv, data/regime.json
Writes:  data/synthesis.json  (keyed by symbol)
         data/scores_final.csv (scores.csv + thesis columns)

──────────────────────────────────────────────────────────────────────────
2026-06-05 PATCH NOTES — stage 5 rerank bug fix:

The post-LLM rerank block previously used a 50/50 alpha+EV blend for
composite_rank, which silently reverted the 2026-05-04 stage 4 redesign
(EV as veto gate, alpha-driven ranking with momentum tiebreaker for
saturated alphas).

Stage 4 correctly produced a composite_rank using the alpha+momentum
tiebreaker. Stage 5 then OVERWROTE it with the old 50/50 alpha+EV blend
on every weekly run. The bug was confirmed empirically: every
composite_rank value in the historical perf log matches the broken
stage 5 formula, not the intended stage 4 formula.

The fix replaces the rerank block with logic that mirrors stage 4's
intent:
  1. Recompute alpha_pct_rank and alpha_rank to reflect LLM bumps.
  2. Build composite_key = alpha_pct_rank + 0.001 * sig_momentum_pct
     (same formula as stage 4).
  3. Preserve stage 4's exclusions (detected via NaN composite_rank
     from stage 4 -- tickers excluded by liquidity, EV veto, or
     earnings have NaN there).
  4. Apply the coverage gate (NaN alpha_score) as well.

Backtested impact across 6 weeks of historical perf log data:
  - Old behavior (50/50 alpha+EV blend): mean basket return +2.76%/wk
  - Fixed behavior (alpha+momentum tiebreaker): mean +3.83%/wk
  - Delta: +1.07pp per week, beats prior in 4 of 6 weeks

Downstream notes:
  - The "skip_top_3_v1" shadow strategy was calibrated against the
    broken ranking and is partially invalidated by this fix. Don't
    activate it until 6-8 weeks of post-fix data accumulates.
  - Rank-crowding analysis from prior research sessions reflected the
    broken ranking. The fundamental EV-paradox finding (EV anti-
    predictive within top-100) is still real, but its operational
    manifestation should diminish significantly after this fix.
──────────────────────────────────────────────────────────────────────────
"""

import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

log = logging.getLogger(__name__)

DATA_DIR      = Path("data")
SCORES_CSV    = DATA_DIR / "scores.csv"
REGIME_JSON   = DATA_DIR / "regime.json"
SYNTH_JSON    = DATA_DIR / "synthesis.json"
FINAL_CSV     = DATA_DIR / "scores_final.csv"

LLM_TOP_N     = 50       # tickers that get the full Claude pass
API_SLEEP     = 0.3      # seconds between API calls
MODEL         = "claude-sonnet-4-6"
MAX_TOKENS    = 300


# ── RULE-BASED THESIS (all tickers) ──────────────────────────────────────────

def rule_based_thesis(row: pd.Series, regime: str) -> str:
    """
    Fast heuristic thesis for every ticker.
    Reads which signals are strong and constructs a plain-English sentence.
    """
    parts = []

    mom  = row.get("sig_momentum",     np.nan)
    cat  = row.get("sig_catalyst",     np.nan)
    fund = row.get("sig_fundamentals", np.nan)
    sent = row.get("sig_sentiment",    np.nan)
    rs   = row.get("sig_momentum_rs",  np.nan)
    brk  = row.get("sig_momentum_breakout", np.nan)
    earn = row.get("sig_catalyst_earnings", np.nan)
    ins  = row.get("sig_catalyst_insider",  np.nan)
    vsrg = row.get("sig_momentum_vol_surge", np.nan)

    # Momentum descriptor
    if not np.isnan(rs):
        if rs >= 0.90:
            parts.append("top-decile relative strength")
        elif rs >= 0.75:
            parts.append("strong RS rank")
        elif rs >= 0.50:
            parts.append("above-average momentum")

    # Breakout
    if not np.isnan(brk) and brk >= 0.80:
        parts.append("near 52-week high breakout")

    # Volume confirmation
    if not np.isnan(vsrg) and vsrg >= 0.70:
        parts.append("elevated volume confirmation")

    # Earnings catalyst
    if not np.isnan(earn) and earn >= 0.70:
        parts.append("earnings catalyst within 2 weeks")
    elif not np.isnan(earn) and earn >= 0.45:
        parts.append("upcoming earnings")

    # Insider buying
    if not np.isnan(ins) and ins >= 0.60:
        parts.append("recent insider buying (Form 4)")

    # Fundamental quality
    if not np.isnan(fund):
        if fund >= 0.75:
            parts.append("strong fundamental quality")
        elif fund >= 0.60:
            parts.append("solid fundamentals")

    # Sentiment
    if not np.isnan(sent) and sent >= 0.70:
        parts.append("positive news sentiment")

    # Regime context
    regime_ctx = {
        "risk_on":        "risk-on market",
        "trending_mixed": "mixed but trending market",
        "choppy_neutral": "choppy market — proceed selectively",
        "risk_off_mild":  "defensive rotation underway",
        "risk_off_severe":"risk-off environment",
    }.get(regime, "current market")

    if not parts:
        return f"Moderate signal alignment in {regime_ctx}."

    joined = ", ".join(parts[:3])  # cap at 3 factors for readability
    return f"{joined.capitalize()} in {regime_ctx}."


# ── LLM THESIS (top N tickers) ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative research analyst reviewing weekly stock scoring data.
You will receive signal scores and context for a single ticker.
Your job is to synthesize the signals into a concise, actionable assessment.

CORE PHILOSOPHY: There are winners every single week in every market condition.
A brutal down week for the broad market is irrelevant — your job is to find the
tickers that WIN that specific week. The regime tells you WHERE winners are hiding,
not whether they exist. Never flag a ticker as unattractive simply because the
market regime is difficult. Instead, explain WHY this ticker wins in THIS regime.

Respond ONLY with a JSON object in exactly this format:
{
  "thesis": "one sentence, max 20 words, specific to why this ticker wins this week",
  "conviction_adjustment": 0.00,
  "risk_flag": "one short phrase or null",
  "confidence": "high|medium|low"
}

conviction_adjustment: float between -0.15 and +0.15
  Positive if signals are unusually well-aligned or there is a specific catalyst
  Negative only for genuine red flags: deteriorating fundamentals WITH weak momentum,
    or a risk_flag that materially undermines the thesis
  Never negative simply because the broad market regime is risk-off

risk_flag: only flag genuine ticker-specific risks, not broad market conditions.
  Valid: "earnings miss history", "heavy debt load", "CEO just sold $10M"
  Invalid: "risk-off market", "choppy conditions", "macro uncertainty"

Be direct. No hedging. No generic statements. No market commentary.
The thesis must explain what is DRIVING this specific ticker this specific week.
Bad:  "Strong momentum in a challenging market environment."
Bad:  "Caution warranted given current risk-off conditions."
Good: "Top-decile RS with insider buying 8 days before earnings."
Good: "Short squeeze setup — 31% float short, momentum inflecting, risk-off rotation into sector."
Good: "FDA decision catalyst in 6 days, analyst upgrades accelerating, defensive sector bid."
"""

def build_user_prompt(row: pd.Series, regime_data: dict) -> str:
    """Build the per-ticker prompt with all signal context."""

    def fmt(val, pct=False):
        if pd.isna(val):
            return "n/a"
        if pct:
            return f"{val*100:.1f}%"
        return f"{val:.4f}"

    regime = regime_data["regime"]
    reg_composite = regime_data.get("composite", 0)

    lines = [
        f"TICKER: {row.get('symbol', '?')}",
        f"Sector: {row.get('sector', 'Unknown')}",
        f"Price: ${row.get('last_price', 0):.2f}  MarketCap: ${row.get('market_cap', 0)/1e9:.1f}B",
        f"",
        f"REGIME: {regime} (composite: {reg_composite:+.3f})",
        f"  {regime_data.get('description', '')}",
        f"",
        f"ALPHA SCORE: {fmt(row.get('alpha_score'))}  "
        f"Rank: #{int(row.get('alpha_rank', 0))} of {int(row.get('universe_size', 0))}  "
        f"Conviction: {row.get('conviction', 'n/a')}",
        f"",
        f"SIGNAL BREAKDOWN:",
        f"  Momentum composite:    {fmt(row.get('sig_momentum'))}",
        f"    RS rank:             {fmt(row.get('sig_momentum_rs'), pct=True)}",
        f"    Trend quality:       {fmt(row.get('sig_momentum_trend'))}",
        f"    Volume surge:        {fmt(row.get('sig_momentum_vol_surge'))}",
        f"    Breakout proximity:  {fmt(row.get('sig_momentum_breakout'))}",
        f"",
        f"  Catalyst composite:    {fmt(row.get('sig_catalyst'))}",
        f"    Earnings proximity:  {fmt(row.get('sig_catalyst_earnings'))}",
        f"    Insider buying:      {fmt(row.get('sig_catalyst_insider'))}",
        f"    Analyst signal:      {fmt(row.get('sig_catalyst_analyst'))}",
        f"",
        f"  Fundamentals:          {fmt(row.get('sig_fundamentals'))}",
        f"    Growth:              {fmt(row.get('sig_fund_growth'))}",
        f"    Quality:             {fmt(row.get('sig_fund_quality'))}",
        f"    Profitability:       {fmt(row.get('sig_fund_profitability'))}",
        f"",
        f"  Sentiment:             {fmt(row.get('sig_sentiment'))}",
        f"    News tone:           {fmt(row.get('sig_sentiment_news'))}",
        f"    Analyst trend:       {fmt(row.get('sig_sentiment_analyst'))}",
        f"    Short interest:      {fmt(row.get('sig_sentiment_short'))}",
        f"    Article count:       {int(row.get('sig_sentiment_articles', 0))}",
    ]

    return "\n".join(lines)


def llm_synthesis(row: pd.Series, regime_data: dict,
                  client: Anthropic) -> dict:
    """
    Call Claude API for a single ticker.
    Returns parsed JSON dict or fallback on any error.
    """
    fallback = {
        "thesis":               rule_based_thesis(row, regime_data["regime"]),
        "conviction_adjustment": 0.0,
        "risk_flag":            None,
        "confidence":           "low",
        "source":               "fallback",
    }

    try:
        prompt = build_user_prompt(row, regime_data)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])

        parsed = json.loads(raw)

        # Validate and clamp conviction adjustment
        adj = float(parsed.get("conviction_adjustment", 0.0))
        adj = float(np.clip(adj, -0.15, 0.15))

        return {
            "thesis":               str(parsed.get("thesis", fallback["thesis"]))[:200],
            "conviction_adjustment": round(adj, 4),
            "risk_flag":            parsed.get("risk_flag"),
            "confidence":           parsed.get("confidence", "medium"),
            "source":               "llm",
        }

    except json.JSONDecodeError as e:
        log.debug(f"    JSON parse error for {row.get('symbol')}: {e}")
        return {**fallback, "source": "fallback_parse_error"}
    except Exception as e:
        log.debug(f"    API error for {row.get('symbol')}: {e}")
        return {**fallback, "source": "fallback_api_error"}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading scores...")
    scores = pd.read_csv(SCORES_CSV) if SCORES_CSV.exists() else pd.read_csv(FINAL_CSV)
    universe_size = len(scores)
    scores["universe_size"] = universe_size
    log.info(f"  {universe_size:,} tickers")

    log.info("Loading regime...")
    with open(REGIME_JSON) as f:
        regime_data = json.load(f)
    log.info(f"  Regime: {regime_data['regime']}")

    # ── RULE-BASED THESIS FOR ALL ─────────────────────────────────────────────
    log.info("Generating rule-based thesis for all tickers...")
    regime = regime_data["regime"]
    scores["thesis"]               = scores.apply(
        lambda r: rule_based_thesis(r, regime), axis=1)
    scores["conviction_adjustment"] = 0.0
    scores["risk_flag"]             = None
    scores["confidence"]            = "rule_based"
    scores["thesis_source"]         = "rule_based"

    # ── LLM SYNTHESIS FOR TOP N ───────────────────────────────────────────────
    top_n = scores.head(LLM_TOP_N).copy()
    log.info(f"Running LLM synthesis on top {LLM_TOP_N} tickers...")

    synthesis_results = {}

    try:
        client = Anthropic()
        llm_available = True
    except Exception as e:
        log.warning(f"  Anthropic client unavailable: {e} — skipping LLM pass")
        llm_available = False

    if llm_available:
        for i, (idx, row) in enumerate(top_n.iterrows()):
            sym = row.get("symbol", "?")
            log.info(f"  [{i+1}/{LLM_TOP_N}] {sym}")

            result = llm_synthesis(row, regime_data, client)
            synthesis_results[sym] = result

            # Apply conviction adjustment to alpha score
            adj = result["conviction_adjustment"]
            scores.loc[idx, "alpha_score"] = float(np.clip(
                scores.loc[idx, "alpha_score"] + adj, 0.0, 1.0
            ))
            scores.loc[idx, "thesis"]               = result["thesis"]
            scores.loc[idx, "conviction_adjustment"] = adj
            scores.loc[idx, "risk_flag"]             = result.get("risk_flag")
            scores.loc[idx, "confidence"]            = result["confidence"]
            scores.loc[idx, "thesis_source"]         = result["source"]

            time.sleep(API_SLEEP)

    # ── RE-RANK AFTER LLM ADJUSTMENTS ────────────────────────────────────────
    # LLM synthesis can shift alpha_score via conviction_adjustment. Both
    # alpha_rank AND composite_rank must be recomputed to reflect those
    # shifts -- otherwise composite_rank (which alpaca_trader uses for
    # position selection) lags the post-LLM alpha_score in the same file.
    #
    # 2026-06-05 PATCH: previously this block used a 50/50 alpha+EV blend
    # for composite_rank, which silently reverted the 2026-05-04 stage 4
    # redesign (EV as veto gate, alpha-driven ranking with momentum
    # tiebreaker). The composite produced here would overwrite stage 4's
    # correctly computed composite_rank with the OLD broken formula. The
    # bug was confirmed empirically: every composite_rank value in the
    # historical perf log matches the broken stage 5 formula, not the
    # intended stage 4 formula.
    #
    # The fix mirrors stage 4's composite logic:
    #   1. Recompute alpha_pct_rank and alpha_rank to reflect LLM bumps.
    #   2. Build composite_key = alpha_pct_rank + 0.001 * sig_momentum_pct
    #      (same formula as stage 4).
    #   3. Preserve stage 4's exclusions (detected via NaN composite_rank
    #      from stage 4 -- tickers excluded by liquidity, EV veto, or
    #      earnings have NaN there).
    #   4. Apply the coverage gate (NaN alpha_score) as well.
    #
    # Backtested impact across 6 weeks: +1.07pp per week mean basket
    # return improvement, beats prior behavior in 4 of 6 weeks.
    scores["alpha_pct_rank"] = scores["alpha_score"].rank(pct=True).round(4)
    scores["alpha_rank"]     = scores["alpha_score"].rank(
        ascending=False, method="min").astype("Int64")

    # Preserve stage 4's exclusions: any ticker stage 4 excluded (NaN
    # composite_rank) stays excluded in stage 5. Stage 4 handles three
    # categories of exclusion (liquidity, EV veto, earnings); we don't
    # need to know which one applied to any individual ticker -- we just
    # need to preserve the exclusion.
    stage4_excluded = (
        scores["composite_rank"].isna()
        if "composite_rank" in scores.columns
        else pd.Series(False, index=scores.index)
    )
    coverage_excluded = scores["alpha_score"].isna()
    excluded = stage4_excluded | coverage_excluded

    # Build composite key using stage 4's logic: alpha-driven, momentum
    # tiebreaker. The momentum tiebreaker matters because LightGBM
    # predict_proba can saturate alpha at 1.0 for many tickers; without
    # a tiebreaker, ranking among them is determined by data order
    # (essentially random) -- or, in the prior broken version, by EV
    # (which is anti-predictive within the top-100).
    if "sig_momentum" in scores.columns:
        mom_pct = scores["sig_momentum"].rank(pct=True).fillna(0.5)
        composite_key = scores["alpha_pct_rank"] + 0.001 * mom_pct
    else:
        log.warning("  sig_momentum not available in stage 5 -- using "
                    "alpha_pct_rank alone (saturated alpha will produce ties)")
        composite_key = scores["alpha_pct_rank"]

    # Apply exclusions: NaN out excluded rows so they don't take rank positions
    composite_key = composite_key.where(~excluded, float("nan"))
    scores["composite_rank"] = composite_key.rank(
        ascending=False, method="min"
    ).astype("Int64")

    # Keep composite_rank_score for diagnostic visibility (matches stage 4)
    scores["composite_rank_score"] = composite_key.round(6)

    n_excluded_total = int(excluded.sum())
    n_ranked = int((~excluded).sum())
    log.info(f"  Post-LLM rerank: {n_ranked:,} ranked, "
             f"{n_excluded_total:,} excluded "
             f"(preserving stage 4 exclusions + coverage gate)")

    # Reapply conviction tiers
    p = scores["alpha_pct_rank"]
    scores["conviction"] = pd.cut(
        p,
        bins=[0, 0.50, 0.70, 0.85, 0.93, 1.01],
        labels=["low", "moderate", "elevated", "high", "very_high"],
    )

    # Sort by composite_rank (authoritative ranking for trading). NaN rows
    # (liquidity-excluded, coverage-gated) sort last so they don't pollute
    # the top of the display.
    if "composite_rank" in scores.columns and scores["composite_rank"].notna().any():
        scores = scores.sort_values("composite_rank", na_position="last").reset_index(drop=True)
    else:
        scores = scores.sort_values("alpha_rank", na_position="last").reset_index(drop=True)

    # ── SAVE ──────────────────────────────────────────────────────────────────
    scores.to_csv(FINAL_CSV, index=False)

    with open(SYNTH_JSON, "w") as f:
        json.dump(synthesis_results, f, indent=2)

    llm_count = sum(1 for v in synthesis_results.values()
                    if v.get("source") == "llm")

    log.info(f"LLM synthesis: {llm_count}/{LLM_TOP_N} successful")
    log.info(f"Output → {FINAL_CSV}")
    log.info(f"Synthesis → {SYNTH_JSON}")
    log.info("\nTop 5 final:")
    for _, row in scores.head(5).iterrows():
        flag = f"  [{row['risk_flag']}]" if row.get("risk_flag") else ""
        log.info(f"  #{row['alpha_rank']:<4} {row['symbol']:<8} "
                 f"{row['confidence']:<10} {row['thesis'][:70]}{flag}")

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
