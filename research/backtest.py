"""
research/backtest.py
====================
Weekly cross-sectional backtest over the research-data weekly panel.

Timing (no look-ahead):
  * Features use data through Friday close of week t.
  * Trades happen in week t+1 under one of three timing models:
      designed  : buy Monday CLOSE, sell Friday CLOSE   (what the system intends)
      broken    : buy Tuesday OPEN, sell next Monday OPEN (what it actually did May-Oct 2026)
      open_open : buy Monday OPEN, sell Friday CLOSE
  * Optional hard stop: triggers if any low on days 2..n of the week breaches
    entry*(1-s); fills at the stop, or at Tuesday's open if it gapped through.

Universe at formation (week t):
  price >= MIN_PRICE, 4-week avg daily dollar volume >= MIN_ADV, >= 53 weeks of
  history, not an obvious fund/ETF/warrant/unit/preferred by name. Delisted
  (inactive) symbols are included while they traded.

Known limitations (state them with every result):
  * Alpaca's inactive-asset list is incomplete, so some survivorship bias remains.
  * A stock that stops trading between Friday and Monday is skipped rather than
    booked at a delisting return.
  * Costs are a flat per-side bps assumption, not a fill model.
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FUND_PAT = re.compile(
    r"\b(ETF|ETN|FUND|TRUST|PROSHARES|ISHARES|DIREXION|SPDR|INVESCO|VANGUARD|WISDOMTREE|"
    r"WARRANTS?|RIGHTS?|UNITS?|PREFERRED|NOTES?|DEBENTURES?|PORTFOLIO|INDEX|BITCOIN|ETHER)\b",
    re.I,
)
# Operating REITs / business trusts we don't want to drop just for the word "trust"
KEEP_PAT = re.compile(r"\b(REALTY|PROPERTIES|PROPERTY|HOSPITALITY|RESIDENTIAL|BANCORP|BANK)\b", re.I)


# ── DATA ──────────────────────────────────────────────────────────────────────

@dataclass
class Panel:
    weeks: pd.DatetimeIndex
    symbols: pd.Index
    m: dict = field(default_factory=dict)        # name -> (weeks x symbols) float array
    is_fund: np.ndarray | None = None

    def df(self, name):
        return pd.DataFrame(self.m[name], index=self.weeks, columns=self.symbols)


def load_panel(data_dir: str, last_complete_week: str | None = None) -> Panel:
    files = sorted(glob.glob(f"{data_dir}/weekly_*.parquet"))
    long = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if last_complete_week:
        long = long[long.week <= pd.Timestamp(last_complete_week)]
    assets = pd.read_parquet(f"{data_dir}/assets.parquet").set_index("symbol")

    cols = ["d1_open", "d1_close", "d2_open", "close", "low_after_d1", "high", "dollar_vol", "n_days"]
    wide = {c: long.pivot(index="week", columns="symbol", values=c).sort_index() for c in cols}
    weeks, symbols = wide["close"].index, wide["close"].columns
    p = Panel(weeks=weeks, symbols=symbols)
    for c in cols:
        p.m[c] = wide[c].reindex(index=weeks, columns=symbols).to_numpy(dtype="float64")
    names = assets.reindex(symbols)["name"].fillna("").astype(str)
    p.is_fund = np.array([bool(FUND_PAT.search(n)) and not KEEP_PAT.search(n) for n in names])
    return p


# ── FEATURES (all as of Friday close, week t) ─────────────────────────────────

def features(p: Panel) -> dict[str, pd.DataFrame]:
    C = p.df("close")
    H = p.df("high")
    dv = p.df("dollar_vol")
    nd = p.df("n_days")
    f = {}
    r1 = C / C.shift(1) - 1
    f["r1w"] = r1
    f["r4w"] = C / C.shift(4) - 1
    f["m12_1"] = C.shift(4) / C.shift(52) - 1
    f["m6_1"] = C.shift(4) / C.shift(26) - 1
    f["m3_1"] = C.shift(4) / C.shift(13) - 1
    f["rs_live"] = 0.40 * f["m12_1"] + 0.35 * f["m6_1"] + 0.25 * f["m3_1"]   # live RS formula
    f["hi52"] = C / H.rolling(52, min_periods=40).max()
    f["vol12"] = r1.rolling(12, min_periods=10).std()
    sma10, sma40 = C.rolling(10).mean(), C.rolling(40).mean()
    f["trend"] = (C > sma10).astype(float) + (sma10 > sma40).astype(float) + (C > sma40).astype(float)
    f["adv"] = dv.rolling(4, min_periods=3).sum() / nd.rolling(4, min_periods=3).sum()
    f["hist"] = C.notna().rolling(53, min_periods=1).sum()
    f["price"] = C
    return f


def universe(p: Panel, f, min_price=5.0, min_adv=2e6) -> pd.DataFrame:
    u = (f["price"] >= min_price) & (f["adv"] >= min_adv) & (f["hist"] >= 53)
    u &= ~pd.Series(p.is_fund, index=p.symbols)
    return u


def xrank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank within the universe each week."""
    return df.where(mask).rank(axis=1, pct=True)


# ── NEXT-WEEK RETURNS ─────────────────────────────────────────────────────────

def next_week_returns(p: Panel, timing="designed", stop: float | None = None) -> pd.DataFrame:
    """Return earned in week t+1, aligned to formation week t."""
    d1o, d1c, d2o = p.df("d1_open"), p.df("d1_close"), p.df("d2_open")
    cl, lo, nd = p.df("close"), p.df("low_after_d1"), p.df("n_days")
    if timing == "designed":
        entry, exit_ = d1c, cl
        ok = nd >= 2
    elif timing == "open_open":
        entry, exit_ = d1o, cl
        ok = nd >= 1
    elif timing == "broken":
        entry, exit_ = d2o, d1o.shift(-1)       # Tue open -> next Mon open
        ok = nd >= 2
    else:
        raise ValueError(timing)
    r = (exit_ / entry - 1).where(ok)
    if stop is not None:
        if timing != "designed":
            raise ValueError("stop simulation only modelled for designed timing")
        stop_px = entry * (1 - stop)
        hit = lo <= stop_px
        gap = d2o < stop_px
        stopped = np.where(gap, d2o / entry - 1, -stop)
        r = r.where(~hit, pd.DataFrame(stopped, index=r.index, columns=r.columns))
    # clip absurd prints (bad ticks / reverse-split artifacts)
    r = r.where(r.abs() < 1.5)
    return r.shift(-1)   # align to formation week t


# ── PORTFOLIO SIM ─────────────────────────────────────────────────────────────

def top_n(score: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame, n=10,
          cost_bps=15.0, ascending=False) -> pd.Series:
    """Equal-weight top-N each week; full liquidation weekly => 2 x cost per week."""
    s = score.where(mask & fwd.notna())
    out = {}
    for wk, row in s.iterrows():
        row = row.dropna()
        if len(row) < n:
            continue
        picks = row.nsmallest(n).index if ascending else row.nlargest(n).index
        out[wk] = fwd.loc[wk, picks].mean() - 2 * cost_bps / 1e4
    return pd.Series(out, dtype=float)


def decile_spread(score, fwd, mask, q=10):
    """Mean next-week return by score decile (diagnoses the signal itself)."""
    s = score.where(mask & fwd.notna())
    buckets = s.rank(axis=1, pct=True).mul(q).clip(upper=q - 1e-9).floordiv(1)
    res = {}
    for b in range(q):
        res[b + 1] = fwd.where(buckets == b).mean(axis=1)
    return pd.DataFrame(res)


def weekly_ic(score, fwd, mask):
    s = score.where(mask & fwd.notna())
    return s.rank(axis=1).corrwith(fwd.where(s.notna()).rank(axis=1), axis=1)


# ── STATS ─────────────────────────────────────────────────────────────────────

def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    dd = eq / eq.cummax() - 1
    return {
        "weeks": len(r),
        "CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
        "vol%": r.std() * np.sqrt(52) * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else np.nan,
        "maxDD%": dd.min() * 100,
        "win%": (r > 0).mean() * 100,
        "worst_wk%": r.min() * 100,
    }


def by_year(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
