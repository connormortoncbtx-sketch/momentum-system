"""
automation/collect_returns.py
==============================
Runs every Monday morning.

Finds the most recent scores_final.csv snapshot (Friday's scores),
pulls actual Friday-to-Friday returns for every scored ticker,
and appends rows to data/performance_log.csv.

The performance log is the ground truth that retraining uses.

Columns written:
    week_of             Friday date of the score
    symbol
    alpha_score         Score assigned that Friday
    alpha_rank
    conviction
    sector
    regime              Regime label that week
    regime_composite
    forward_return_1w   Actual return from that Friday close to this Friday close
    forward_return_1w_rank  Percentile rank within that week's universe
    sig_momentum        Raw signal values (for attribution analysis)
    sig_catalyst
    sig_fundamentals
    sig_sentiment
"""

import logging
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.tz_utils import now_ct

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR  = Path("data")
PERF_LOG  = DATA_DIR / "performance_log.csv"
SCORES    = DATA_DIR / "scores_final.csv"


def last_friday(from_date=None) -> datetime:
    d = from_date or datetime.today()
    days_back = (d.weekday() - 4) % 7  # 4 = Friday
    return (d - timedelta(days=days_back)).replace(
        hour=0, minute=0, second=0, microsecond=0)


def trading_days_since(from_date: datetime) -> int:
    """
    Count trading days (Mon-Fri, excluding weekends) between
    from_date and today. Used to guard against collecting returns
    before the week has actually played out.
    """
    today = datetime.today()
    if from_date >= today:
        return 0
    days = 0
    current = from_date + timedelta(days=1)
    while current <= today:
        if current.weekday() < 5:  # Mon-Fri
            days += 1
        current += timedelta(days=1)
    return days


def fetch_weekly_returns(symbols: list[str],
                         week_start: datetime,
                         week_end: datetime) -> pd.Series:
    """
    Download Friday-to-Friday returns.
    Returns Series keyed by symbol.
    """
    start = (week_start - timedelta(days=3)).strftime("%Y-%m-%d")
    end   = (week_end   + timedelta(days=3)).strftime("%Y-%m-%d")

    log.info(f"Fetching returns {start} → {end} for {len(symbols)} symbols...")
    batch_size = 200
    returns    = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False)["Close"]

            if len(batch) == 1:
                sym = batch[0]
                if not raw.empty:
                    start_price = raw.iloc[0]
                    end_price   = raw.iloc[-1]
                    if start_price > 0:
                        returns[sym] = float(end_price / start_price) - 1
            else:
                for sym in batch:
                    try:
                        prices = raw[sym].dropna()
                        if len(prices) >= 2:
                            returns[sym] = float(prices.iloc[-1] / prices.iloc[0]) - 1
                    except Exception:
                        continue
        except Exception as e:
            log.warning(f"Batch fetch error: {e}")
            continue

    return pd.Series(returns)


def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Holiday check — skip if not a full 5-day trading week
    from automation.tz_utils import assert_normal_week
    if not assert_normal_week("collect_returns"):
        return

    # Idempotency guard — prevent double-run from dual DST crons
    today_ct   = now_ct().strftime("%Y-%m-%d")
    lock_file  = DATA_DIR / ".collect_lock"
    lock_key   = f"friday_learning_{today_ct}"
    if lock_file.exists() and lock_file.read_text().strip() == lock_key:
        log.info(f"collect_returns already ran today ({today_ct}) — skipping duplicate")
        return
    lock_file.write_text(lock_key)

    # Load last Friday's scores
    if not SCORES.exists():
        log.warning("No scores_final.csv found — nothing to score")
        return

    scores = pd.read_csv(SCORES)
    log.info(f"Loaded scores: {len(scores)} tickers")

    today       = datetime.today()
    this_friday = last_friday(today)
    last_fri    = this_friday - timedelta(weeks=1)

    # Guard: ensure at least 5 trading days have passed since the scores
    # were generated. Prevents collecting returns before the week plays out.
    scored_at = None
    if "scored_at" in scores.columns:
        try:
            scored_at = datetime.strptime(str(scores["scored_at"].iloc[0]), "%Y-%m-%d")
        except Exception:
            pass

    if scored_at:
        elapsed = trading_days_since(scored_at)
        log.info(f"Scores generated: {scored_at.date()}  |  Trading days elapsed: {elapsed}")
        if elapsed < 5:
            log.warning(
                f"Only {elapsed} trading days since scores were generated "
                f"(need 5). Skipping return collection — run again next Monday."
            )
            return
    else:
        log.warning("Could not determine score date — proceeding anyway")

    # Check if we already have this week's returns
    if PERF_LOG.exists():
        existing = pd.read_csv(PERF_LOG)
        week_str = last_fri.strftime("%Y-%m-%d")
        if "week_of" in existing.columns:
            already_logged = (existing["week_of"] == week_str).any()
            if already_logged:
                log.info(f"Returns for {week_str} already logged — skipping")
                return
    else:
        existing = pd.DataFrame()

    # Fetch actual returns
    symbols = scores["symbol"].tolist()
    returns = fetch_weekly_returns(symbols, last_fri, this_friday)

    if returns.empty:
        log.warning("No returns fetched — check market calendar")
        return

    # Build log rows
    log.info(f"Building performance log rows...")
    rows = []
    for _, row in scores.iterrows():
        sym = row["symbol"]
        ret = returns.get(sym, np.nan)
        rows.append({
            "week_of":           last_fri.strftime("%Y-%m-%d"),
            "symbol":            sym,
            "alpha_score":       row.get("alpha_score"),
            "alpha_rank":        row.get("alpha_rank"),
            "conviction":        row.get("conviction"),
            "sector":            row.get("sector"),
            "regime":            row.get("regime"),
            "regime_composite":  row.get("regime_composite"),
            "forward_return_1w": round(ret, 6) if pd.notna(ret) else np.nan,
            "sig_momentum":      row.get("sig_momentum"),
            "sig_catalyst":      row.get("sig_catalyst"),
            "sig_fundamentals":  row.get("sig_fundamentals"),
            "sig_sentiment":     row.get("sig_sentiment"),
        })

    new_df = pd.DataFrame(rows)

    # Add percentile rank within this week
    new_df["forward_return_1w_rank"] = (
        new_df["forward_return_1w"]
        .rank(pct=True)
        .round(4)
    )

    # Append to log
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_csv(PERF_LOG, index=False)

    coverage = new_df["forward_return_1w"].notna().mean()
    avg_ret  = new_df["forward_return_1w"].mean()
    top_ret  = new_df.nlargest(5, "alpha_score")["forward_return_1w"].mean()

    log.info(f"Performance log updated: {len(new_df)} rows added")
    log.info(f"Return coverage: {coverage*100:.1f}%")
    log.info(f"Universe avg return: {avg_ret*100:.2f}%")
    log.info(f"Top-scored avg return: {top_ret*100:.2f}%  ← model signal quality")
    log.info(f"Output → {PERF_LOG}  (total rows: {len(combined)})")


if __name__ == "__main__":
    run()
