# automation/analyze_winners.py
# Runs every Friday after collect_returns.py.
#
# For the most recently completed week, identifies the top 10 actual gainers
# from the full universe, looks up their pre-week signal scores, and sends
# everything to Claude for pattern analysis.
#
# Outputs: insights/YYYY-MM-DD_winners.md
#
# The goal is to learn what winning stocks look like BEFORE they move --
# including stocks the system ranked poorly -- so signal weights and
# selection logic can improve over time.

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from anthropic import Anthropic

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR  = Path("data")
PERF_LOG  = DATA_DIR / "performance_log.csv"
INSIGHTS  = Path("insights")
MIN_WEEKS = 1   # need at least 1 completed week
TOP_N     = 10  # number of winners to analyze


# -- HELPERS ------------------------------------------------------------------

def get_last_completed_week(log_df: pd.DataFrame) -> str:
    """Return the most recent week_of value that has return data."""
    completed = log_df[log_df["forward_return_1w"].notna()]
    if completed.empty:
        return None
    return str(completed["week_of"].max())


def get_top_winners(log_df: pd.DataFrame, week: str, n: int = 10) -> pd.DataFrame:
    """Return the top N gainers for a given week."""
    week_df = log_df[log_df["week_of"].astype(str) == week].copy()
    week_df = week_df[week_df["forward_return_1w"].notna()]
    return week_df.nlargest(n, "forward_return_1w").reset_index(drop=True)


def sig_fmt(val) -> str:
    """Format a signal value for the prompt."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{float(val):.3f}"


def build_winner_block(row: pd.Series, rank: int) -> str:
    """Build a text block describing one winner for the prompt."""
    ret = row.get("forward_return_1w", 0)
    lines = [
        f"#{rank}  {row.get('symbol', '?')}  +{ret*100:.1f}%  "
        f"sector={row.get('sector', 'unknown')}",
        f"  System rank: composite={row.get('composite_rank', 'n/a')}  "
        f"alpha_rank={row.get('alpha_rank', 'n/a')}  "
        f"ev_rank={row.get('ev_rank', 'n/a')}  "
        f"conviction={row.get('conviction', 'n/a')}",
        f"  Signals: momentum={sig_fmt(row.get('sig_momentum'))}  "
        f"catalyst={sig_fmt(row.get('sig_catalyst'))}  "
        f"fundamentals={sig_fmt(row.get('sig_fundamentals'))}  "
        f"sentiment={sig_fmt(row.get('sig_sentiment'))}",
        f"  Sub-signals: rs={sig_fmt(row.get('sig_momentum_rs'))}  "
        f"trend={sig_fmt(row.get('sig_momentum_trend'))}  "
        f"vol_surge={sig_fmt(row.get('sig_momentum_vol_surge'))}  "
        f"breakout={sig_fmt(row.get('sig_momentum_breakout'))}",
        f"               earnings={sig_fmt(row.get('sig_catalyst_earnings'))}  "
        f"insider={sig_fmt(row.get('sig_catalyst_insider'))}  "
        f"analyst={sig_fmt(row.get('sig_catalyst_analyst'))}",
        f"               news={sig_fmt(row.get('sig_sentiment_news'))}  "
        f"short_int={sig_fmt(row.get('sig_sentiment_short'))}  "
        f"ev_score={sig_fmt(row.get('ev_score'))}  "
        f"weekly_vol={sig_fmt(row.get('weekly_vol'))}",
    ]
    return "\n".join(lines)


# -- PROMPT -------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the pattern analysis engine for a quantitative momentum stock ranking system.\n"
    "\n"
    "You will receive data on the top 10 actual weekly gainers from the most recent week,\n"
    "including their pre-week signal scores and how the system ranked them.\n"
    "\n"
    "Your job is to identify patterns that explain WHY these stocks won and\n"
    "what the system could do better to find them earlier.\n"
    "\n"
    "Focus on:\n"
    "1. Which signals were elevated vs suppressed in the winners\n"
    "2. Whether the system found them (high rank) or missed them (low rank)\n"
    "3. Common patterns across multiple winners -- sector, signal profile, regime fit\n"
    "4. Specific signals that appear predictive this week vs which appear noisy\n"
    "5. Any blind spots -- winner types the system consistently underranks\n"
    "6. Actionable recommendations for signal weight adjustments or new signals to build\n"
    "\n"
    "Be specific and data-driven. Reference actual signal values. Avoid generic statements.\n"
    "If the system ranked a winner well, explain what worked.\n"
    "If it missed a winner, diagnose why and what signal would have caught it.\n"
    "\n"
    "Output a structured markdown analysis with clear sections.\n"
    "Do not include preamble -- start directly with the analysis."
)


def build_prompt(winners: pd.DataFrame, week: str, regime: str,
                 universe_avg: float, universe_median: float) -> str:
    lines = [
        f"WEEKLY WINNERS ANALYSIS -- week of {week}",
        f"Regime: {regime}",
        f"Universe avg return: {universe_avg*100:+.2f}%  "
        f"median: {universe_median*100:+.2f}%",
        f"Top winner: +{winners.iloc[0]['forward_return_1w']*100:.1f}%  "
        f"#10 winner: +{winners.iloc[-1]['forward_return_1w']*100:.1f}%",
        "",
        "TOP 10 ACTUAL GAINERS (with pre-week signal scores):",
        "",
    ]

    for i, (_, row) in enumerate(winners.iterrows(), 1):
        lines.append(build_winner_block(row, i))
        lines.append("")

    # Add context: how many winners were in very_high / high conviction
    found = winners[winners["conviction"].isin(["very_high", "high"])]
    top50 = winners[winners["alpha_rank"].notna() & (winners["alpha_rank"] <= 50)]
    lines += [
        f"System hit rate: {len(found)}/{len(winners)} winners were very_high/high conviction",
        f"Top-50 capture: {len(top50)}/{len(winners)} winners were ranked in top 50",
        "",
        "Provide your pattern analysis below:",
    ]

    return "\n".join(lines)


# -- MAIN ---------------------------------------------------------------------

def run():
    INSIGHTS.mkdir(parents=True, exist_ok=True)

    if not PERF_LOG.exists():
        log.info("No performance log yet -- skipping winners analysis")
        return

    log_df = pd.read_csv(PERF_LOG, low_memory=False, parse_dates=["week_of"])
    n_weeks = log_df["week_of"].nunique()

    if n_weeks < MIN_WEEKS:
        log.info(f"Only {n_weeks} weeks of data -- skipping winners analysis")
        return

    week = get_last_completed_week(log_df)
    if not week:
        log.info("No completed weeks with return data -- skipping")
        return

    # Check if we already ran this week
    out_path = INSIGHTS / f"{week}_winners.md"
    if out_path.exists():
        log.info(f"Winners analysis already exists for {week} -- skipping")
        return

    log.info(f"Analyzing top winners for week of {week}...")

    winners = get_top_winners(log_df, week, TOP_N)
    if len(winners) < 3:
        log.info(f"Not enough return data for week {week} -- skipping")
        return

    # Universe stats for context
    week_df = log_df[log_df["week_of"].astype(str) == week]
    universe_avg    = float(week_df["forward_return_1w"].mean())
    universe_median = float(week_df["forward_return_1w"].median())
    regime = str(week_df["regime"].mode().iloc[0]) if len(week_df) > 0 else "unknown"

    log.info(f"  Week regime: {regime}")
    log.info(f"  Universe avg: {universe_avg*100:+.2f}%  "
             f"median: {universe_median*100:+.2f}%")
    log.info(f"  Top winner: {winners.iloc[0]['symbol']} "
             f"+{winners.iloc[0]['forward_return_1w']*100:.1f}%")

    prompt = build_prompt(winners, week, regime, universe_avg, universe_median)

    log.info("Sending to Claude for pattern analysis...")
    try:
        client   = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = response.content[0].text.strip()
    except Exception as e:
        log.error(f"Claude API call failed: {e} -- skipping analysis")
        return

    # Build the output markdown
    found = winners[winners["conviction"].isin(["very_high", "high"])]
    top50 = winners[winners["alpha_rank"].notna() & (winners["alpha_rank"] <= 50)]

    output_lines = [
        f"# Weekly Winners Analysis -- {week}",
        "",
        f"**Regime:** {regime}  ",
        f"**Universe avg return:** {universe_avg*100:+.2f}%  ",
        f"**Universe median return:** {universe_median*100:+.2f}%  ",
        f"**System hit rate:** {len(found)}/{len(winners)} winners in very_high/high conviction  ",
        f"**Top-50 capture:** {len(top50)}/{len(winners)} winners ranked in top 50",
        "",
        "## Top 10 Actual Gainers",
        "",
    ]

    for i, (_, row) in enumerate(winners.iterrows(), 1):
        ret = row.get("forward_return_1w", 0)
        conv = row.get("conviction", "n/a")
        rank = row.get("composite_rank", row.get("alpha_rank", "n/a"))
        output_lines.append(
            f"{i}. **{row.get('symbol', '?')}** +{ret*100:.1f}%  "
            f"rank={rank}  conviction={conv}  "
            f"sector={row.get('sector', 'unknown')}"
        )

    output_lines += [
        "",
        "## Pattern Analysis",
        "",
        analysis,
        "",
        f"*Generated {datetime.today().strftime('%Y-%m-%d %H:%M')} by analyze_winners.py*",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    log.info(f"Winners analysis -> {out_path}")

    # Log the hit rate summary
    log.info(f"  Hit rate: {len(found)}/{len(winners)} winners were high/very_high conviction")
    log.info(f"  Top-50 capture: {len(top50)}/{len(winners)} winners in top 50")
    for i, (_, row) in enumerate(winners.iterrows(), 1):
        ret = row.get("forward_return_1w", 0)
        rank = row.get("composite_rank", row.get("alpha_rank", "n/a"))
        conv = row.get("conviction", "n/a")
        log.info(f"  #{i:<2} {row.get('symbol','?'):<8} "
                 f"+{ret*100:.1f}%  rank={rank}  {conv}")


if __name__ == "__main__":
    run()
