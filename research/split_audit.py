"""
research/split_audit.py -- find stock splits the vendor's adjusted prices missed (Nov 15, 2026; prep for the Jan 5
review: the Nov 13 data check found ~118 weeks with a >= 3x move within 3% of a whole split ratio).

A missed split in adjusted daily bars looks like:
  * one overnight gap, open / previous close within 4% of a split ratio R in {2,3,4,5,6,8,10,12,15,20,25,30,40,50,
    100,200} or 1/R (2:1 .. 1:200), and |gap| >= 1.8x;
  * share volume shifting by ~1/R so DOLLAR volume is continuous: median dollar volume of the 10 sessions after
    vs the 10 before within [1/3, 3] (a real squeeze or collapse usually brings a dollar-volume spike);
  * the new level holds: median close of the next 5 sessions within 0.6-1.6x of the gap-day open (no instant reversal,
    which would be a bad print).
Output: research_data/results/split_events.csv (symbol, date, ratio = new price / old price, gap, dv_ratio, hold).
backtest.load_panel() applies these to the weekly panel (scales earlier prices by the ratio) unless SPLIT_FIX=0.
"""
import glob
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
RATIOS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 200], dtype=float)
CANDS = np.concatenate([RATIOS, 1 / RATIOS])


def nearest_ratio(g):
    """Nearest split ratio to gap g and the relative distance."""
    i = np.abs(np.log(CANDS)[None, :] - np.log(g)[:, None]).argmin(axis=1)
    R = CANDS[i]
    return R, np.abs(g / R - 1)


def main():
    files = sorted(glob.glob(f"{RD}/daily/*ohlcv_*.parquet"))
    cols = ["symbol", "date", "open", "close", "volume"]
    D = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)
    D["date"] = pd.to_datetime(D.date).dt.tz_localize(None).dt.normalize() if not hasattr(D.date.dtype, "tz") or D.date.dt.tz is None \
        else pd.to_datetime(D.date).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    D = D.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    for c in ("open", "close", "volume"):
        D[c] = D[c].astype("float64")
    print(f"daily bars: {len(D):,} rows, {D.symbol.nunique():,} symbols, {D.date.min().date()}..{D.date.max().date()}", flush=True)
    g = D.groupby("symbol", sort=False)
    prev = g.close.shift(1)
    gap = D.open / prev
    cand = D.index[(gap.notna()) & ((gap >= 1.8) | (gap <= 1 / 1.8))]
    print(f"overnight gaps >= 1.8x: {len(cand):,}", flush=True)
    dv = D.close * D.volume
    sym = D.symbol.to_numpy()
    rows = []
    gv = gap.to_numpy()
    dvv = dv.to_numpy()
    cl = D.close.to_numpy()
    op = D.open.to_numpy()
    R, dist = nearest_ratio(gv[cand])
    for k, i in enumerate(cand):
        s = sym[i]
        lo, hi = max(i - 10, 0), min(i + 10, len(D))
        before = [j for j in range(lo, i) if sym[j] == s]
        after = [j for j in range(i + 1, min(i + 11, len(D))) if sym[j] == s]   # skip the event day itself
        nxt = [j for j in range(i, min(i + 5, len(D))) if sym[j] == s]
        if len(before) < 5 or len(after) < 3:
            continue
        dvr = np.nanmedian(dvv[after]) / np.nanmedian(dvv[before]) if np.nanmedian(dvv[before]) > 0 else np.nan
        hold = np.nanmedian(cl[nxt]) / op[i] if op[i] > 0 else np.nan
        rows.append(dict(symbol=s, date=D.date.iat[i].date(), prev_close=round(cl[i - 1], 4), open=round(op[i], 4),
                         gap=round(gv[i], 4), ratio=R[k], dist=round(dist[k], 4), dv_ratio=round(dvr, 3), hold=round(hold, 3)))
    E = pd.DataFrame(rows)
    E["split"] = (E.dist <= 0.04) & E.dv_ratio.between(0.5, 2) & E.hold.between(0.6, 1.6)
    # in the research universe the week before? (raw close >= $5, >= $2M/day, not a fund name)
    import sys; sys.path.insert(0, os.path.dirname(__file__)); import backtest as B
    W = pd.concat((pd.read_parquet(f, columns=["symbol", "week", "raw_close", "dollar_vol", "n_days"])
                   for f in sorted(glob.glob(f"{RD}/weekly_*.parquet"))), ignore_index=True)
    W["ok"] = (W.raw_close >= 5) & (W.dollar_vol / W.n_days >= 2e6)
    names = pd.read_parquet(f"{RD}/assets.parquet").set_index("symbol")["name"].fillna("").astype(str)
    fund = {s for s, n in names.items() if B.is_fund_name(n)}
    W = W[W.ok & ~W.symbol.isin(fund)].sort_values("week")
    E["date"] = pd.to_datetime(E.date).astype("datetime64[ns]")
    W["week"] = W.week.astype("datetime64[ns]")
    E = E.sort_values("date")
    m = pd.merge_asof(E[["symbol", "date"]].reset_index(), W[["symbol", "week"]].rename(columns={"week": "date"}).assign(wk=lambda x: x.date),
                      on="date", by="symbol", allow_exact_matches=False, direction="backward")
    E["in_universe"] = ((E.date - m.set_index("index").wk.reindex(E.index)).dt.days <= 10).fillna(False).to_numpy()
    os.makedirs(f"{RD}/results", exist_ok=True)
    E.to_csv(f"{RD}/results/split_gaps_all.csv", index=False)
    S = E[E.split].drop(columns="split")
    S.to_csv(f"{RD}/results/split_events.csv", index=False)
    print(f"gaps examined {len(E):,}; near a split ratio {int((E.dist <= 0.04).sum()):,}; "
          f"flagged as missed splits {len(S):,} ({S.symbol.nunique():,} symbols)")
    print("by direction: forward (price down) %d, reverse (price up) %d" % ((S.ratio < 1).sum(), (S.ratio > 1).sum()))
    print("by year:", S.groupby(pd.to_datetime(S.date).dt.year).size().to_dict())
    print("most common ratios:", S.ratio.map(lambda r: f"{r:g}" if r >= 1 else f"1/{1 / r:g}").value_counts().head(10).to_dict())
    U = S[S.in_universe]
    print(f"of which in the $5 / $2M universe the week before: {len(U)} ({U.symbol.nunique()} symbols); by year",
          U.groupby(U.date.dt.year).size().to_dict())
    print(U.drop(columns="in_universe").to_string(index=False))


if __name__ == "__main__":
    main()
