"""
research/contract_book.py -- portfolio test of the LLM contract-win signal with FULL
coverage of qualifying small-cap 8-Ks, Jan 2024 - Sep 2026 (Oct 11, 2026).

Judgements: llm/ (study sample) + llm_full/part*/ (backfill of every other qualifying
8-K: $30M-$2B market cap, price >= $1, >= $300k/day, study items).
Periods: "early" = Jan 2024 - Mar 2025 (before/around the model's training cutoff --
possible memorised outcomes, used for choosing), "holdout" = Apr 2025 - Sep 2026, scored once.

Book: buy at the first regular-session OPEN after the authoritative acceptance time,
hold H sessions (exit at the close), equal weight across open positions with a
per-name cap (rest in cash); 25 bps per side. Daily bars from Alpaca (SIP, adjusted).
Variants (signal x H x cap) are ranked on "early" only.
"""
import glob, os
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = os.environ.get("OUT") or f"{RD}/results"
COST = 25.0
HOLD0, START = pd.Timestamp("2025-04-01"), pd.Timestamp("2024-01-01")


def load():
    J, E = [], []
    for d in [f"{RD}/llm"] + sorted(glob.glob(f"{RD}/llm_full/part*")):
        if os.path.exists(f"{d}/judgements.parquet") and os.path.exists(f"{d}/events.parquet"):
            J.append(pd.read_parquet(f"{d}/judgements.parquet"))
            E.append(pd.read_parquet(f"{d}/events.parquet"))
    j = pd.concat(J).drop_duplicates("accession"); e = pd.concat(E).drop_duplicates("accession")
    ev = e.merge(j, on="accession")
    ev["t"] = pd.to_datetime(ev.accepted_true)
    return ev[(ev.t >= START) & ev.ok.fillna(False)]


def bars(symbols, start):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    cl = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames = []
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 200):
        try:
            d = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 200], timeframe=TimeFrame.Day,
                                                   start=start.to_pydatetime(), adjustment=Adjustment.ALL,
                                                   feed=DataFeed.SIP)).df
            if len(d):
                frames.append(d.reset_index()[["symbol", "timestamp", "open", "close"]])
        except Exception as ex:
            print(f"bars batch {i // 200}: {ex}")
    b = pd.concat(frames)
    b["date"] = pd.to_datetime(b.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return (b.pivot_table(index="date", columns="symbol", values="open"),
            b.pivot_table(index="date", columns="symbol", values="close"))


def book(ev, O, C, H, cap):
    days = C.index
    open_t = days + pd.Timedelta(hours=9, minutes=30)
    pos = np.searchsorted(open_t.values, ev.t.values.astype("datetime64[ns]"), side="right")
    # daily return contribution of each position: entry day close/open, then close/close
    R = C / C.shift(1) - 1
    R0 = C / O - 1
    contrib = {}          # day index -> list of returns of open positions
    for sym, p in zip(ev.symbol, pos):
        if sym not in C.columns or p >= len(days):
            continue
        if not np.isfinite(O[sym].iloc[p]):
            continue
        for h in range(H + 1):
            if p + h >= len(days):
                break
            r = R0[sym].iloc[p] if h == 0 else R[sym].iloc[p + h]
            if not np.isfinite(r):
                if h > 0:
                    break              # stopped trading: exit flat
                continue
            c = (2 * COST / 1e4) if h == 0 else 0.0      # round trip charged on entry day
            contrib.setdefault(p + h, []).append(r - c)
    out, n = np.zeros(len(days)), np.zeros(len(days))
    for i, rs in contrib.items():
        w = min(1.0 / len(rs), cap)
        out[i] = w * float(np.sum(rs)); n[i] = len(rs)
    s = pd.Series(out, index=days); n = pd.Series(n, index=days)
    return s[s.index >= START], n[n.index >= START]


def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100,
                vol=r.std() * np.sqrt(252) * 100, Sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                maxDD=(eq / eq.cummax() - 1).min() * 100)


class Tee:
    def __init__(self, *fs): self.fs = fs
    def write(self, x): [f.write(x) for f in self.fs]
    def flush(self): [f.flush() for f in self.fs]


def main():
    import sys
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, open(f"{OUT}/contract_book.txt", "w"))
    ev = load()
    print(f"judged 8-Ks since 2024: {len(ev):,}; by period:",
          ev.groupby(np.where(ev.t >= HOLD0, "holdout", "early")).size().to_dict())
    pos_types = ["contract_win", "guidance_raise", "prelim_results_good", "regulatory_approval", "clinical_positive",
                 "partnership"]
    ev["rel_value"] = ev.dollar_value_usd / ev.mcap
    cw = (ev.event_type == "contract_win") & (ev.direction >= 1)
    SIG = {"contract_win dir>=1 (shadow #6)": cw,
           "contract_win dir>=1 materiality>=0.7": cw & (ev.materiality >= 0.7),
           "contract_win dir>=1 novelty>=0.8": cw & (ev.novelty >= 0.8),
           "contract_win dir>=1 value>=10% of mcap": cw & (ev.rel_value >= 0.10),
           "contract_win dir=2": (ev.event_type == "contract_win") & (ev.direction >= 2),
           "any positive type dir>=1": ev.event_type.isin(pos_types) & (ev.direction >= 1)}
    for k, m in SIG.items():
        print(f"  {k}: {int(m.sum())} events ({int((m & (ev.t >= HOLD0)).sum())} holdout)")
    # event-level drift with t-stats (IWM-adjusted, from the entry open)
    print("\nEVENT LEVEL (IWM-adjusted from entry open; %)")
    for k, m in SIG.items():
        for per, pm in (("early", ev.t < HOLD0), ("holdout", ev.t >= HOLD0)):
            x = ev[m & pm]
            for h in ("ab_5", "ab_20"):
                v = x[h].dropna()
                t = v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 10 else np.nan
                print(f"  {k:42s} {per:8s} {h}: n {len(v):4d} mean {v.mean() * 100:6.2f} median {v.median() * 100:6.2f} t {t:5.2f}"
                      f"  win {(v > 0).mean() * 100:4.0f}%")

    sel = ev[np.logical_or.reduce(list(SIG.values()))]
    O, C = bars(list(sel.symbol) + ["IWM", "SPY"], START - pd.Timedelta(days=10))
    rows = []
    for k, m in SIG.items():
        for H in (5, 10, 20, 40):
            for cap in (0.10, 0.20, 1.0):
                r, n = book(ev[m], O, C, H, cap)
                e, h = r[r.index < HOLD0], r[r.index >= HOLD0]
                rows.append(dict(signal=k, H=H, cap=cap, avg_pos=n.mean(), invested_days=(n > 0).mean() * 100,
                                 **{f"early_{a}": b for a, b in st(e).items()},
                                 **{f"hold_{a}": b for a, b in st(h).items()}))
    res = pd.DataFrame(rows).sort_values("early_Sharpe", ascending=False)
    for nm in ("IWM", "SPY"):
        b = (C[nm] / C[nm].shift(1) - 1); b = b[b.index >= START]
        print(nm, "early", {a: round(v, 2) for a, v in st(b[b.index < HOLD0]).items()},
              "holdout", {a: round(v, 2) for a, v in st(b[b.index >= HOLD0]).items()})
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    print("\nPORTFOLIOS (ranked on early Sharpe)")
    print(res.round(2).to_string(index=False))
    res.to_csv(f"{OUT}/contract_book.csv", index=False)
    top = res.iloc[0]
    # robustness of the design chosen on "early": higher costs, without its best events, daily series saved
    global COST
    m = SIG[top.signal]; H, cap = int(top.H), float(top.cap)
    base, _ = book(ev[m], O, C, H, cap)
    pd.DataFrame({"contract_book": base}).to_csv(f"{OUT}/contract_book_daily.csv")
    print(f"\nROBUSTNESS of {top.signal} H{H} cap{cap}")
    for cst in (50.0, 100.0):
        COST = cst; r, _ = book(ev[m], O, C, H, cap)
        print(f"  cost {cst:.0f} bps/side: early", {a: round(v, 2) for a, v in st(r[r.index < HOLD0]).items()},
              "holdout", {a: round(v, 2) for a, v in st(r[r.index >= HOLD0]).items()})
    COST = 25.0
    x = ev[m].copy()
    for k in (1, 3, 5, 10):
        drop = x.sort_values("ab_20", ascending=False).head(k).accession          # ex post: remove the best k events
        r, _ = book(x[~x.accession.isin(drop)], O, C, H, cap)
        print(f"  without best {k:2d} events: early", {a: round(v, 2) for a, v in st(r[r.index < HOLD0]).items()},
              "holdout", {a: round(v, 2) for a, v in st(r[r.index >= HOLD0]).items()})
    # control: random 8-Ks (any type) with the same number of events per period, same book
    rng = np.random.default_rng(3)
    ctrl = []
    pool_e, pool_h = ev[ev.t < HOLD0], ev[ev.t >= HOLD0]
    ne, nh = int((m & (ev.t < HOLD0)).sum()), int((m & (ev.t >= HOLD0)).sum())
    O2, C2 = bars(list(pd.concat([pool_e.sample(min(len(pool_e), ne * 20), random_state=1),
                                  pool_h.sample(min(len(pool_h), nh * 20), random_state=1)]).symbol) + ["IWM"],
                  START - pd.Timedelta(days=10))
    for d in range(20):
        smp = pd.concat([pool_e.sample(min(len(pool_e), ne * 20), random_state=1).iloc[d * ne:(d + 1) * ne],
                         pool_h.sample(min(len(pool_h), nh * 20), random_state=1).iloc[d * nh:(d + 1) * nh]])
        r, _ = book(smp, O2, C2, H, cap)
        ctrl.append((st(r[r.index < HOLD0])["Sharpe"], st(r[r.index >= HOLD0])["Sharpe"],
                     st(r[r.index < HOLD0])["CAGR"], st(r[r.index >= HOLD0])["CAGR"]))
    ctrl = pd.DataFrame(ctrl, columns=["early_Sharpe", "hold_Sharpe", "early_CAGR", "hold_CAGR"])
    print("  CONTROL: 20 random sets of 8-Ks (same counts, same book):")
    print(ctrl.describe().round(2).to_string())
    print(f"\n::notice::contract book: best-on-early {top.signal} H{top.H} cap{top.cap}: early Sharpe {top.early_Sharpe:.2f} "
          f"CAGR {top.early_CAGR:.1f}%, holdout Sharpe {top.hold_Sharpe:.2f} CAGR {top.hold_CAGR:.1f}% DD {top.hold_maxDD:.0f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
