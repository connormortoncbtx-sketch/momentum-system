# automation/self_refine.py
# Runs every Friday at 6pm CT after collect_returns.py.
# Analyzes performance log and asks Claude to update weights.json.

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from anthropic import Anthropic

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR    = Path("data")
PERF_LOG    = DATA_DIR / "performance_log.csv"
WEIGHTS     = Path("config/weights.json")
REFINEMENTS = Path("refinements")
MIN_WEEKS   = 3


# -- ANALYSIS ------------------------------------------------------------------

def information_coefficient(scores, returns):
    mask = scores.notna() & returns.notna()
    if mask.sum() < 20:
        return float("nan")
    r, _ = stats.spearmanr(scores[mask], returns[mask])
    return round(float(r), 4)


def hit_rate(scores, returns, top_pct=0.20):
    mask = scores.notna() & returns.notna()
    if mask.sum() < 20:
        return float("nan")
    s, r   = scores[mask], returns[mask]
    cutoff = s.quantile(1 - top_pct)
    top    = r[s >= cutoff]
    med    = r.median()
    return round(float((top > med).mean()), 4)


def compute_attribution(log_df):
    signals = ["sig_momentum", "sig_catalyst", "sig_fundamentals", "sig_sentiment"]
    regimes = log_df["regime"].dropna().unique().tolist()
    results = {}
    for sig in signals:
        if sig not in log_df.columns:
            continue
        ic_overall = information_coefficient(log_df[sig], log_df["forward_return_1w"])
        hr_overall = hit_rate(log_df[sig], log_df["forward_return_1w"])
        by_regime = {}
        for regime in regimes:
            sub = log_df[log_df["regime"] == regime]
            if len(sub) < 15:
                continue
            by_regime[regime] = {
                "ic":       information_coefficient(sub[sig], sub["forward_return_1w"]),
                "hit_rate": hit_rate(sub[sig], sub["forward_return_1w"]),
                "n":        int(len(sub)),
            }
        results[sig] = {
            "ic_overall": ic_overall,
            "hr_overall": hr_overall,
            "by_regime":  by_regime,
        }
    return results


def conviction_performance(log_df):
    out = {}
    for conv, grp in log_df.groupby("conviction"):
        rets = grp["forward_return_1w"].dropna()
        if len(rets) < 5:
            continue
        out[conv] = {
            "mean_return":   round(float(rets.mean()), 4),
            "median_return": round(float(rets.median()), 4),
            "hit_rate":      round(float((rets > log_df["forward_return_1w"].median()).mean()), 4),
            "n":             int(len(rets)),
        }
    return out


def weekly_alpha(log_df):
    weeks = []
    for week, grp in log_df.groupby("week_of"):
        rets   = grp["forward_return_1w"].dropna()
        scores = grp["alpha_score"].dropna()
        if len(rets) < 20:
            continue
        top_q   = scores.quantile(0.80)
        top_ret = grp.loc[grp["alpha_score"] >= top_q, "forward_return_1w"].mean()
        all_ret = rets.mean()
        weeks.append({
            "week":      week,
            "regime":    grp["regime"].mode().iloc[0] if len(grp) else "unknown",
            "top_q_ret": round(float(top_ret), 4) if pd.notna(top_ret) else None,
            "avg_ret":   round(float(all_ret), 4),
            "alpha":     round(float(top_ret - all_ret), 4) if pd.notna(top_ret) else None,
        })
    return weeks


# -- PROMPT --------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the self-refinement engine for a quantitative momentum stock ranking system.\n"
    "\n"
    "You will receive a structured performance analysis. Output an updated weights.json.\n"
    "\n"
    "CORE PHILOSOPHY: Winners exist every week in every market condition.\n"
    "The regime determines WHERE winners are found, not WHETHER they exist.\n"
    "Never reduce regime multipliers to near-zero.\n"
    "\n"
    "HARD CONSTRAINTS:\n"
    "- signal_weight values must sum to exactly 1.0\n"
    "- each signal_weight must be between 0.05 and 0.60\n"
    "- each regime_multiplier must be between 0.10 and 2.00\n"
    "- you may only change numeric values, do not add or remove keys\n"
    "- the _meta, constraints, gates, and note fields must be preserved\n"
    "\n"
    "GUIDANCE:\n"
    "- Increase weights for signals with IC > 0.05 and hit rate > 0.55 consistently\n"
    "- Decrease weights for signals with IC < 0.00 or hit rate < 0.45 consistently\n"
    "- In risk-off regimes, shift weight TO catalyst and sentiment\n"
    "- Make conservative changes, max 0.10 change per signal weight per week\n"
    "\n"
    "Respond with ONLY the complete updated weights.json content.\n"
    "No explanation, no markdown fences, no preamble. Just the raw JSON."
)


def build_analysis_prompt(attribution, conviction_perf, weekly, current_weights, n_weeks):
    lines = [
        "PERFORMANCE ANALYSIS - last %d weeks" % n_weeks,
        "",
        "SIGNAL ATTRIBUTION:",
    ]
    for sig, data in attribution.items():
        lines.append("  %s:" % sig)
        lines.append("    overall IC=%s  hit_rate=%s" % (data["ic_overall"], data["hr_overall"]))
        for regime, rd in data.get("by_regime", {}).items():
            lines.append("    %-20s IC=%s  hit_rate=%s  n=%d" % (
                regime, rd["ic"], rd["hit_rate"], rd["n"]))
    lines += ["", "CONVICTION TIER PERFORMANCE:"]
    for conv in ["very_high", "high", "elevated", "moderate", "low"]:
        if conv in conviction_perf:
            d = conviction_perf[conv]
            lines.append("  %-12s mean=%+.4f  hit_rate=%s  n=%d" % (
                conv, d["mean_return"], d["hit_rate"], d["n"]))
    lines += ["", "WEEKLY ALPHA (top quintile vs universe):"]
    for w in weekly[-8:]:
        alpha_str = "%+.4f" % w["alpha"] if w["alpha"] is not None else "n/a"
        lines.append("  %s  %-20s top_q=%s  avg=%s  alpha=%s" % (
            w["week"], w["regime"], w["top_q_ret"], w["avg_ret"], alpha_str))
    lines += ["", "CURRENT WEIGHTS:", json.dumps(current_weights, indent=2)]
    return "\n".join(lines)


# -- VALIDATION ----------------------------------------------------------------

def validate_weights(new_w, current_w):
    try:
        constraints = current_w.get("constraints", {})
        w_min = float(constraints.get("signal_weight_min", 0.05))
        w_max = float(constraints.get("signal_weight_max", 0.60))
        m_min = float(constraints.get("regime_multiplier_min", 0.10))
        m_max = float(constraints.get("regime_multiplier_max", 2.00))

        # Coerce to float -- Claude may return numeric strings
        sw = new_w.get("signal_weights", {})
        sw = {k: float(v) for k, v in sw.items()}
        new_w["signal_weights"] = sw

        total = sum(sw.values())
        if abs(total - 1.0) > 0.001:
            return False, "signal_weights sum to %.4f, not 1.0" % total

        for k, v in sw.items():
            if not (w_min <= v <= w_max):
                return False, "signal_weight %s=%s outside [%s,%s]" % (k, v, w_min, w_max)

        # Coerce regime multipliers to float too
        for regime, mults in new_w.get("regime_multipliers", {}).items():
            coerced = {sig: float(v) for sig, v in mults.items()}
            new_w["regime_multipliers"][regime] = coerced
            for sig, v in coerced.items():
                if not (m_min <= v <= m_max):
                    return False, "regime_multiplier %s.%s=%s outside [%s,%s]" % (
                        regime, sig, v, m_min, m_max)

        current_sigs = set(current_w["signal_weights"].keys())
        new_sigs     = set(sw.keys())
        if current_sigs != new_sigs:
            return False, "signal_weight keys changed: %s" % str(current_sigs ^ new_sigs)

        return True, "ok"
    except Exception as e:
        return False, str(e)


def diff_weights(old, new):
    changes = []
    for sig, old_v in old.get("signal_weights", {}).items():
        new_v = new.get("signal_weights", {}).get(sig, old_v)
        if abs(float(new_v) - float(old_v)) > 0.001:
            arrow = "up" if float(new_v) > float(old_v) else "down"
            changes.append("  signal_weights.%s: %.3f -> %.3f %s" % (
                sig, float(old_v), float(new_v), arrow))
    for regime in old.get("regime_multipliers", {}):
        for sig, old_v in old["regime_multipliers"][regime].items():
            new_v = new.get("regime_multipliers", {}).get(regime, {}).get(sig, old_v)
            if abs(float(new_v) - float(old_v)) > 0.01:
                arrow = "up" if float(new_v) > float(old_v) else "down"
                changes.append("  regime_multipliers.%s.%s: %.2f -> %.2f %s" % (
                    regime, sig, float(old_v), float(new_v), arrow))
    return changes


# -- MAIN ----------------------------------------------------------------------

def run():
    REFINEMENTS.mkdir(parents=True, exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")

    if not PERF_LOG.exists():
        log.info("No performance log yet - skipping refinement")
        return

    log_df = pd.read_csv(PERF_LOG, low_memory=False, parse_dates=["week_of"])
    n_weeks = log_df["week_of"].nunique()

    if n_weeks < MIN_WEEKS:
        log.info("Only %d weeks of data (need %d) - skipping refinement" % (n_weeks, MIN_WEEKS))
        return

    log.info("Analyzing %d weeks, %d rows..." % (n_weeks, len(log_df)))

    with open(WEIGHTS) as f:
        current_weights = json.load(f)

    attribution     = compute_attribution(log_df)
    conviction_perf = conviction_performance(log_df)
    weekly          = weekly_alpha(log_df)

    prompt = build_analysis_prompt(
        attribution, conviction_perf, weekly, current_weights, n_weeks)

    log.info("Sending analysis to Claude for refinement...")

    try:
        client   = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        new_weights = json.loads(raw)
    except Exception as e:
        log.error("Claude API call failed: %s - skipping refinement" % e)
        return

    valid, reason = validate_weights(new_weights, current_weights)
    if not valid:
        log.error("Weight validation failed: %s - keeping current weights" % reason)
        note_path = REFINEMENTS / ("%s.md" % date_str)
        with open(note_path, "w") as f:
            f.write("# Refinement %s - REJECTED\n\n**Reason:** %s\n\nWeeks analyzed: %d\n" % (
                date_str, reason, n_weeks))
        return

    new_weights["_meta"] = {
        **current_weights.get("_meta", {}),
        "last_modified": date_str,
        "modified_by":   "self_refine.py",
    }
    new_weights["constraints"] = current_weights["constraints"]
    new_weights["gates"]       = current_weights["gates"]

    changes = diff_weights(current_weights, new_weights)

    with open(WEIGHTS, "w") as f:
        json.dump(new_weights, f, indent=2)

    log.info("Weights updated -> %s" % WEIGHTS)
    for c in changes:
        log.info(c)
    if not changes:
        log.info("  No significant changes made")

    note_lines = [
        "# Refinement - %s" % date_str, "",
        "**Weeks analyzed:** %d" % n_weeks,
        "**Data points:** %d" % len(log_df), "",
        "## Changes", "",
    ]
    note_lines += changes if changes else ["No changes - current weights performing well."]
    note_lines += ["", "## Signal Attribution Summary", ""]
    for sig, data in attribution.items():
        note_lines.append("- **%s**: IC=%s  hit_rate=%s" % (
            sig, data["ic_overall"], data["hr_overall"]))
    note_lines += ["", "## Weekly Alpha (last 4 weeks)", ""]
    for w in weekly[-4:]:
        alpha_str = "%+.4f" % w["alpha"] if w["alpha"] is not None else "n/a"
        note_lines.append("- %s `%s` - alpha vs universe: %s" % (
            w["week"], w["regime"], alpha_str))

    note_path = REFINEMENTS / ("%s.md" % date_str)
    with open(note_path, "w") as f:
        f.write("\n".join(note_lines))
    log.info("Refinement note -> %s" % note_path)


if __name__ == "__main__":
    run()
