"""
automation/collect_intraday.py
==============================
Collects intraday 5-minute OHLCV bars for the just-completed trading week.

Runs Friday after market close, as part of the learning loop. Each run
produces one file: data/intraday/{score_friday}.csv, containing 5m bars
for the union of:

  - Top 50 tickers from the current scores_final.csv (by composite_rank)
  - Held positions from execution_log.csv for the just-completed week
  - Top 50 tickers from the previous week's perf_log (if available)

The union captures both forward-looking instrumentation (what we'll
analyze for the upcoming week's stop-tuning) and retrospective data
(what actually happened to names we just exited).

Why 5m bars:
  Resolution sufficient for stop/trail analysis (5-minute drawdowns are
  the relevant granularity for intra-day decisions) without the volume
  of 1m data or the rolling 7-day yfinance limit on 1m.

Why a separate file per week:
  Keeps individual file sizes manageable (~3-5 MB each), produces clean
  git diffs (new week = new file, no modifications to existing data),
  and lets aggregate analysis use pd.concat([pd.read_csv(f) for f in
  Path('data/intraday').glob('*.csv')]).

Why this script can fail gracefully:
  Intraday data is research instrumentation, not operational. If yfinance
  rate-limits or a ticker is delisted, the script logs and continues.
  The downstream analyzer can work with whatever data does come through.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, time as dtime

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation.tz_utils import now_ct, is_normal_trading_week
from automation.system_logger import log_event, LogStatus
from automation.notifier import notify_alert

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

DATA_DIR        = Path("data")
INTRADAY_DIR    = DATA_DIR / "intraday"
PERF_LOG        = DATA_DIR / "performance_log.csv"
SCORES          = DATA_DIR / "scores_final.csv"
EXECUTION_LOG   = DATA_DIR / "execution_log.csv"

TOP_N            = 50
BATCH_SIZE       = 50    # smaller than collect_returns since 5m is heavier per ticker
INTERVAL         = "5m"

# Regular trading hours in Eastern time (NYSE/NASDAQ).
# yfinance returns bars timestamped in market local time (US/Eastern) when
# the user has tz set; we filter explicitly to be safe.
MARKET_OPEN_ET   = dtime(9, 30)
MARKET_CLOSE_ET  = dtime(16, 0)


def last_friday(from_date: datetime = None) -> datetime:
    """Return most recent Friday at midnight."""
    d = from_date or datetime.today()
    days_back = (d.weekday() - 4) % 7
    return (d - timedelta(days=days_back)).replace(
        hour=0, minute=0, second=0, microsecond=0)


def get_target_symbols() -> tuple[set, dict]:
    """
    Return (symbols_to_fetch, sources_summary).

    Union of:
      - Top 50 from current scores_final.csv (by composite_rank)
      - Held positions from execution_log for the just-completed week
      - Top 50 from previous week's perf_log (by composite_rank)
    """
    symbols = set()
    sources = {"current_top50": 0, "held_positions": 0, "prior_top50": 0}

    # ── Source 1: Current scores_final ──────────────────────────────────
    if SCORES.exists():
        try:
            scores = pd.read_csv(SCORES, low_memory=False)
            scores["composite_rank"] = pd.to_numeric(scores["composite_rank"], errors="coerce")
            ranked = scores.dropna(subset=["composite_rank"])
            top50 = ranked.nsmallest(TOP_N, "composite_rank")
            current_top = set(top50["symbol"].tolist())
            symbols |= current_top
            sources["current_top50"] = len(current_top)
            log.info(f"Source 1: {len(current_top)} symbols from current scores top-{TOP_N}")
        except Exception as e:
            log.warning(f"Could not read scores_final.csv: {e}")
    else:
        log.warning("scores_final.csv missing -- skipping current top-50 source")

    # ── Source 2: Held positions from execution_log ─────────────────────
    if EXECUTION_LOG.exists():
        try:
            exec_log = pd.read_csv(EXECUTION_LOG, low_memory=False)
            if "week_of" in exec_log.columns and len(exec_log) > 0:
                # Get the most recent week's holdings
                latest_week = exec_log["week_of"].max()
                latest_rows = exec_log[exec_log["week_of"] == latest_week]
                if "symbol" in latest_rows.columns:
                    held = set(latest_rows["symbol"].dropna().tolist())
                    new_from_held = held - symbols  # only count truly new
                    symbols |= held
                    sources["held_positions"] = len(held)
                    log.info(f"Source 2: {len(held)} held positions from week {latest_week} "
                             f"({len(new_from_held)} not already in current top-50)")
        except Exception as e:
            log.warning(f"Could not read execution_log.csv: {e}")
    else:
        log.info("execution_log.csv missing -- skipping held positions source")

    # ── Source 3: Previous week's perf_log top-50 ───────────────────────
    if PERF_LOG.exists():
        try:
            perf = pd.read_csv(PERF_LOG, low_memory=False)
            perf["composite_rank"] = pd.to_numeric(perf["composite_rank"], errors="coerce")
            # Most recent fully-completed week with data is the latest week_of
            weeks = sorted(perf["week_of"].dropna().unique().tolist())
            if weeks:
                prior_week = weeks[-1]
                prior_rows = perf[perf["week_of"] == prior_week].dropna(subset=["composite_rank"])
                prior_top = prior_rows.nsmallest(TOP_N, "composite_rank")
                prior_set = set(prior_top["symbol"].tolist())
                new_from_prior = prior_set - symbols
                symbols |= prior_set
                sources["prior_top50"] = len(prior_set)
                log.info(f"Source 3: {len(prior_set)} symbols from prior week ({prior_week}) "
                         f"top-{TOP_N} ({len(new_from_prior)} not already covered)")
        except Exception as e:
            log.warning(f"Could not read performance_log.csv: {e}")
    else:
        log.info("performance_log.csv missing -- skipping prior top-50 source")

    return symbols, sources


def fetch_intraday_batch(symbols: list, start: datetime, end: datetime) -> dict:
    """
    Fetch 5m bars for a list of symbols between start and end (inclusive).
    Returns dict mapping symbol -> DataFrame with columns
    [open, high, low, close, volume] indexed by datetime.

    yfinance's 5m endpoint is more brittle than the daily endpoint. We
    handle missing data per-symbol rather than failing the whole batch.
    """
    result = {}
    start_str = start.strftime("%Y-%m-%d")
    # end is exclusive in yfinance, so push one day forward to capture Friday
    end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        if i > 0:
            log.info(f"  Progress: {i}/{len(symbols)}")
        try:
            raw = yf.download(batch, start=start_str, end=end_str,
                              interval=INTERVAL, auto_adjust=True,
                              progress=False, threads=True, prepost=False)
            if raw is None or raw.empty:
                log.warning(f"  Empty batch at position {i}")
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        df = pd.DataFrame({
                            "open":   raw["Open"][sym],
                            "high":   raw["High"][sym],
                            "low":    raw["Low"][sym],
                            "close":  raw["Close"][sym],
                            "volume": raw["Volume"][sym],
                        }).dropna()
                        if len(df) > 0:
                            result[sym] = df
                    except Exception:
                        # Single-ticker failure -- the column may not exist if
                        # yfinance didn't return data for this ticker. Skip.
                        continue
            else:
                # Single-symbol batch -- raw is flat
                sym = batch[0]
                df = raw.rename(columns=str.lower).dropna()
                if len(df) > 0:
                    result[sym] = df

        except Exception as e:
            log.warning(f"  Batch error at position {i}: {e}")
            continue

    return result


def filter_to_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a per-ticker df to regular market hours bars only."""
    if df.empty:
        return df

    # Ensure the index has a time component we can filter on. yfinance
    # 5m bars come back with a DatetimeIndex; if it's tz-aware, convert
    # to US/Eastern. If tz-naive, treat as already in market local time.
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        try:
            idx_et = idx.tz_convert("US/Eastern")
        except Exception:
            idx_et = idx
    else:
        idx_et = idx

    # Filter: time of day between 09:30 and 16:00 ET, weekday only.
    times = pd.Series(idx_et).dt.time
    weekdays = pd.Series(idx_et).dt.weekday
    mask = (
        (times >= MARKET_OPEN_ET) &
        (times < MARKET_CLOSE_ET) &
        (weekdays < 5)
    )
    return df[mask.values]


def assign_day_label(df: pd.DataFrame, week_monday: datetime) -> pd.DataFrame:
    """
    Add `day` (mon/tue/wed/thu/fri) and `bar_in_day` (1-based) columns.

    bar_in_day starts at 1 for the 09:30 bar and counts forward in 5m
    increments through the trading session (78 bars in a normal day).
    """
    if df.empty:
        return df

    df = df.copy()
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        try:
            idx_et = idx.tz_convert("US/Eastern")
        except Exception:
            idx_et = idx
    else:
        idx_et = idx

    day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    df["day"] = [day_names[d.weekday()] for d in idx_et]

    # bar_in_day: count from the day's open (09:30) in 5m increments
    df["bar_in_day"] = [
        ((d.hour * 60 + d.minute) - (9 * 60 + 30)) // 5 + 1
        for d in idx_et
    ]

    return df


def run():
    log.info("=" * 60)
    log.info("  INTRADAY COLLECTOR")
    log.info("=" * 60)
    log_event("collect_intraday", LogStatus.INFO, "Starting intraday collection")

    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)

    # ── Determine which week we're collecting ───────────────────────────
    # Same convention as collect_returns: score_friday = the Friday BEFORE
    # the trading week we just observed. Today is Friday after market
    # close; the trading week we observed is THIS Mon-Fri; the score
    # friday that labeled it is LAST Friday.
    today = datetime.today()
    this_friday = last_friday(today)
    score_friday = this_friday - timedelta(weeks=1)
    week_str = score_friday.strftime("%Y-%m-%d")

    # ── Holiday check ───────────────────────────────────────────────────
    # The trading week we just observed starts on the Monday after score_friday.
    trading_week_monday = (score_friday + timedelta(days=3)).date()
    if not is_normal_trading_week(ref_date=trading_week_monday, ref_week="current"):
        log.info(f"Holiday week [collect_intraday] (trading week of {trading_week_monday}) — "
                 f"skipping intraday collection.")
        log_event("collect_intraday", LogStatus.INFO,
                  "Skipped: not a normal trading week")
        return

    out_path = INTRADAY_DIR / f"{week_str}.csv"
    if out_path.exists():
        log.info(f"Intraday data for {week_str} already exists at {out_path} -- skipping")
        log_event("collect_intraday", LogStatus.INFO,
                  f"Skipped: {out_path.name} already exists")
        return

    # ── Build the universe of tickers to fetch ──────────────────────────
    symbols, sources = get_target_symbols()
    if not symbols:
        log.warning("No symbols to fetch -- all sources empty")
        log_event("collect_intraday", LogStatus.WARNING,
                  "No symbols to fetch",
                  metrics=sources)
        return
    symbols_list = sorted(symbols)
    log.info(f"Total unique symbols to fetch: {len(symbols_list)}")

    # ── Fetch ───────────────────────────────────────────────────────────
    trading_monday = score_friday + timedelta(days=3)
    trading_friday = score_friday + timedelta(days=7)
    log.info(f"Fetching {INTERVAL} bars from {trading_monday.date()} to "
             f"{trading_friday.date()} for {len(symbols_list)} symbols...")

    raw_data = fetch_intraday_batch(symbols_list, trading_monday, trading_friday)
    log.info(f"Got intraday data for {len(raw_data):,} symbols "
             f"({len(symbols_list) - len(raw_data):,} returned empty)")

    if not raw_data:
        log.error("No intraday data returned from yfinance -- skipping write")
        log_event("collect_intraday", LogStatus.ERROR,
                  "yfinance returned no data",
                  metrics={"symbols_requested": len(symbols_list)})
        notify_alert("collect_intraday",
                     f"yfinance returned no intraday data for {len(symbols_list)} "
                     f"symbols. Rate limit or network?")
        return

    # ── Filter to market hours and assemble output ──────────────────────
    rows = []
    for sym, df in raw_data.items():
        filtered = filter_to_market_hours(df)
        if filtered.empty:
            continue
        labeled = assign_day_label(filtered, trading_monday)
        for ts, r in labeled.iterrows():
            rows.append({
                "week_of":    week_str,
                "symbol":     sym,
                "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
                "day":        r["day"],
                "bar_in_day": int(r["bar_in_day"]),
                "open":       float(r["open"]),
                "high":       float(r["high"]),
                "low":        float(r["low"]),
                "close":      float(r["close"]),
                "volume":     float(r["volume"]) if pd.notna(r["volume"]) else 0,
            })

    if not rows:
        log.warning("After market-hours filtering, no bars remain")
        log_event("collect_intraday", LogStatus.WARNING,
                  "No bars after filtering")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    # ── Summary ─────────────────────────────────────────────────────────
    n_symbols_with_data = out_df["symbol"].nunique()
    bars_per_symbol = out_df.groupby("symbol").size()
    log.info(f"\nWrote {len(out_df):,} bars for {n_symbols_with_data} symbols "
             f"to {out_path}")
    log.info(f"  Bars per symbol: mean={bars_per_symbol.mean():.0f}, "
             f"min={bars_per_symbol.min()}, max={bars_per_symbol.max()}")
    log.info(f"  Days represented: {sorted(out_df['day'].unique().tolist())}")
    log.info(f"  Source breakdown: {sources}")

    log_event("collect_intraday", LogStatus.SUCCESS,
              f"Wrote {len(out_df):,} bars for {n_symbols_with_data} symbols "
              f"({week_str})",
              metrics={
                  "week_of":              week_str,
                  "n_bars":               int(len(out_df)),
                  "n_symbols_with_data":  int(n_symbols_with_data),
                  "n_symbols_requested":  len(symbols_list),
                  "source_current_top50":  sources["current_top50"],
                  "source_held_positions": sources["held_positions"],
                  "source_prior_top50":    sources["prior_top50"],
              })


if __name__ == "__main__":
    run()
