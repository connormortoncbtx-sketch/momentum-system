"""
research/contract_pick10.py -- "pick 10 built around contract wins" (Nov 24, 2026; Connor: "pick 10 based around
contract wins, with the remaining [slots] filled in by what would be filled in with the old pick 10 system").

Book (10 slots, 10% each, daily simulation):
  CONTRACT slots  the contract-win events of the live CONTRACT sleeve (research/contract_filters.py dev-best set: all
                  contract wins, direction >= 1, + the filters adopted on 2017-23), bought at the first regular-session
                  OPEN after the 8-K's acceptance time, held 10 sessions (as the live sleeve), 25 bps per side.
                  If more than 10 are open, the 10 most recent are held.
  FILL slots      10 - (open contract names): the top names of the old pick-10 research proxy (selection_lab.py
                  "live_proxy": 0.35 RS + 0.25 trend + 0.20 52-week-high rank; $5 / $2M universe, no funds, no recent
                  reverse splits), formed Friday, bought at Monday's close, re-formed weekly; a contract name is never
                  double-counted. Cost: per-name half-spread table (universe_floor.half_spread_bps) on every change.
Compared with: the pick-10 proxy alone (same engine) and the contract sleeve alone (contract_filters "best" series).
Periods: dev 2017-23, holdout 2024 - Sep 2026 (contract judgements end 2026-09-24).
Also: "1-2 names carry the week" profile -- per week, the top-2 names' contribution vs the other 8.
Caveats: the pick-10 is the research proxy, not the full live scorer; the 2023-26 price data drops most delisted names
(see "Data hole found"), which flatters both books in the holdout.
Aggregates only (public repo) -> results/contract_pick10.csv / .txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import contract_filters as CF  # noqa: E402
import universe_floor as U  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT, START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2017-01-01"), pd.Timestamp("2026-09-30")
SLOTS, H, CCOST = 10, 10, 25.0


def daily_bars(symbols):
    syms = set(symbols)
    fr = []
    for f in sorted(glob.glob(f"{RD}/daily/ohlcv_*.parquet")) + sorted(glob.glob(f"{RD}/delisted/delisted_ohlcv_*.parquet")):
        d = pd.read_parquet(f, columns=["symbol", "date", "open", "close", "volume"])
        fr.append(d[d.symbol.isin(syms)])
    b = pd.concat(fr).drop_duplicates(["symbol", "date"])
    b["date"] = pd.to_datetime(b.date).dt.normalize()
    piv = lambda c: b.pivot_table(index="date", columns="symbol", values=c).astype("float64").sort_index()
    return piv("open"), piv("close"), piv("volume")


def contract_events(M):
    """Replicates contract_filters.main(): all contract wins + filters adopted on dev (worst tercile t < -2)."""
    ev = CF.load()
    ev["period"] = np.where(ev.t < SPLIT, "dev", "holdout")
    cw = (ev.event_type == "contract_win") & (ev.direction >= 1)
    E = ev[cw].join(M.features(ev[cw]))
    feats = ["rel_value", "rel_rev", "mcap", "price", "materiality", "novelty", "n_flags", "pre_mom20", "pre_mom60",
             "dist_hi252", "vol60", "pre_volsurge5", "gap_at_entry"]
    cats = [c for c in ("counterparty_type", "firmness", "counterparty_named", "revenue_within_12m", "repeat_customer",
                        "going_concern") if c in E.columns]
    rules, dev = [], E[E.period == "dev"]
    for c in feats:
        q = dev[c].quantile([1 / 3, 2 / 3]).to_numpy()
        if not np.all(np.isfinite(q)) or q[0] == q[1]:
            continue
        tb = pd.cut(E[c], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
        dtb = tb[E.period == "dev"]
        for lab in ("low", "high"):
            w, r = dev[dtb == lab].ab_20, dev[dtb != lab].ab_20
            if (w.mean() - r.mean()) / np.sqrt(w.var() / w.count() + r.var() / r.count()) < -2:
                rules.append((c, lab, q))
    for c in cats:
        for lv, sub in dev.groupby(c):
            rest = dev[dev[c] != lv].ab_20
            if sub.ab_20.count() >= 20 and (sub.ab_20.mean() - rest.mean()) / np.sqrt(
                    sub.ab_20.var() / sub.ab_20.count() + rest.var() / rest.count()) < -2:
                rules.append((c, lv, None))
    keep = pd.Series(True, index=E.index)
    for c, lab, q in rules:
        keep &= (E[c] != lab) if q is None else (~(E[c] <= q[0]) if lab == "low" else ~(E[c] > q[1]))
    print("adopted filters:", [(r[0], r[1]) for r in rules])
    return E[keep]


def pick_rankings():
    """Weekly ordered list (best first, top 30) of the old pick-10 research proxy, keyed by formation Friday."""
    p = B.load_panel(RD); f = B.features(p)
    base = (f["hist"] >= 53) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols) \
        & ~f["rsplit_52w"].fillna(False).astype(bool)
    m = base & (f["price"] >= 5) & (f["adv"] >= 2e6)
    rk = lambda x: x.where(m).rank(axis=1, pct=True)
    sc = (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80
    out = {}
    for wk in sc.index:
        s = sc.loc[wk].dropna()
        if len(s) >= 100:
            out[wk] = list(s.sort_values(ascending=False).index[:30])
    return out


def simulate(O, C, V, X, ranks, use_contract=True, use_fill=True):
    days = C.index[(C.index >= START - pd.Timedelta(days=10)) & (C.index <= END)]
    O, C, V = O.reindex(days), C.reindex(days), V.reindex(days)
    R = C / C.shift(1) - 1
    adv = (C * V).rolling(20, min_periods=10).mean()
    spread = pd.DataFrame(U.half_spread_bps(C.fillna(0).to_numpy(), adv.fillna(0).to_numpy()), index=days, columns=C.columns)
    nd = len(days)
    # contract positions: (sym, entry_i)
    pos = []
    if use_contract:
        open_t = days + pd.Timedelta(hours=9, minutes=30)
        ent = np.searchsorted(open_t.values, pd.to_datetime(X.t).values.astype("datetime64[ns]"), side="right")
        pos = sorted([(s, int(i)) for s, i in zip(X.symbol, ent) if s in C.columns and i < nd], key=lambda z: z[1])
    # fill list active on day k = ranking formed on the last Friday whose following Monday close is before close k
    fri = sorted(ranks)
    mon_close_i = {wk: np.searchsorted(days.values, (wk + pd.Timedelta(days=3)).to_datetime64(), side="left") for wk in fri}
    rets, cnt, contrib = np.zeros(nd), np.zeros(nd), {}
    prev_fill, j = set(), -1
    for k in range(1, nd):
        while j + 1 < len(fri) and mon_close_i[fri[j + 1]] <= k - 1:
            j += 1
        act = [(s, i) for s, i in pos if i <= k <= i + H and np.isfinite(C[s].iloc[k])]
        act = act[-SLOTS:]
        csyms = {s for s, _ in act}
        n = len(act); cnt[k] = n
        r, w = 0.0, 1.0 / SLOTS
        for s, i in act:
            o0 = O[s].iloc[i]
            if k == i:
                x = C[s].iloc[k] / o0 - 1 - 2 * CCOST / 1e4 if np.isfinite(o0) and o0 > 0 else 0.0
            else:
                x = R[s].iloc[k]
            x = 0.0 if not np.isfinite(x) else x
            r += w * x; contrib[(days[k], s)] = contrib.get((days[k], s), 0.0) + w * x
        fill = []
        if use_fill and j >= 0:
            fill = [s for s in ranks[fri[j]] if s not in csyms and s in C.columns][:SLOTS - n]
        fs = set(fill)
        for s in fill:
            x = R[s].iloc[k]                                   # fill names are bought at the prior close
            x = x if np.isfinite(x) else 0.0
            c =spread[s].iloc[k - 1] / 1e4 if s not in prev_fill else 0.0
            r += w * (x - c); contrib[(days[k], s)] = contrib.get((days[k], s), 0.0) + w * (x - c)
        for s in prev_fill - fs:
            r -= w * spread[s].iloc[k - 1] / 1e4 if s in spread.columns and np.isfinite(spread[s].iloc[k - 1]) else 0.0
        prev_fill = fs
        rets[k] = r
    s = pd.Series(rets, index=days)
    return s[s.index >= START], pd.Series(cnt, index=days)[s.index >= START], contrib


def stats(r):
    r = r.dropna()
    w = (1 + r).resample("W-FRI").prod() - 1
    eq = (1 + r).cumprod()
    return dict(CAGR=(eq.iloc[-1] ** (252 / len(r)) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(252),
                maxDD=(eq / eq.cummax() - 1).min() * 100, pct_weeks_up=(w > 0).mean() * 100,
                median_week=w.median() * 100, worst_week=w.min() * 100, best_week=w.max() * 100)


def carry_profile(contrib, a, z):
    c = pd.Series(contrib)
    c.index = pd.MultiIndex.from_tuples(c.index, names=["d", "s"])
    c = c[(c.index.get_level_values(0) >= a) & (c.index.get_level_values(0) <= z)]
    wk = c.groupby([c.index.get_level_values(0).to_period("W-FRI"), c.index.get_level_values(1)]).sum()
    rows = []
    for w, g in wk.groupby(level=0):
        g = g.sort_values(ascending=False)
        rows.append(dict(total=g.sum(), top2=g.iloc[:2].sum(), rest=g.iloc[2:].sum(), bot2=g.iloc[-2:].sum()))
    P = pd.DataFrame(rows)
    good = P[P.total > 0.02]
    return dict(weeks=len(P), weeks_gt2pct=len(good), carried_share=((good.top2 >= good.total) .mean() * 100) if len(good) else np.nan,
                median_rest_pct=P.rest.median() * 100, median_top2_pct=P.top2.median() * 100,
                median_bot2_pct=P.bot2.median() * 100, weeks_top2_gt4pct=(P.top2 > 0.04).mean() * 100)


def main():
    os.makedirs(f"{RD}/results", exist_ok=True)
    ranks = pick_rankings()
    print(f"pick-10 rankings: {len(ranks)} weeks", flush=True)
    ev = CF.load()
    cw = ev[(ev.event_type == "contract_win") & (ev.direction >= 1)]
    syms = set(cw.symbol) | {s for v in ranks.values() for s in v[:25]} | {"IWM", "SPY"}
    O, C, V = daily_bars(syms)
    print(f"bars: {C.shape}", flush=True)
    M = CF.Market(O, C, V)
    X = contract_events(M)
    print(f"contract events used: {len(X)}", X.groupby(X.t.dt.year).size().to_dict(), flush=True)
    books = {"combo: contract + pick-10 fill": simulate(O, C, V, X, ranks, True, True),
             "pick-10 proxy alone": simulate(O, C, V, X, ranks, False, True),
             "contract alone, 10 slots (rest cash)": simulate(O, C, V, X, ranks, True, False)}
    iwm = (C["IWM"] / C["IWM"].shift(1) - 1)
    iwm = iwm[(iwm.index >= START) & (iwm.index <= END)]
    cs = pd.read_csv(f"{RD}/results/contract_filters_daily.csv", index_col=0, parse_dates=True)["best"]
    per = (("dev 2017-23", "2017-01-01", "2023-12-31"), ("holdout 2024-Sep 2026", "2024-01-01", "2026-09-30"),
           ("2025-10..2026-09 (last 52 wk)", "2025-10-01", "2026-09-30"), ("May-Sep 2026", "2026-05-01", "2026-09-30"))
    rows = []
    for lab, a, z in per:
        for nm, (r, n, cb) in books.items():
            rows.append(dict(period=lab, book=nm, **stats(r.loc[a:z]), avg_contract_slots=n.loc[a:z].mean(),
                             **carry_profile(cb, pd.Timestamp(a), pd.Timestamp(z))))
        rows.append(dict(period=lab, book="contract sleeve as live (1/vol, 25% cap)", **stats(cs.loc[a:z])))
        rows.append(dict(period=lab, book="IWM", **stats(iwm.loc[a:z])))
    T = pd.DataFrame(rows)
    T.to_csv(f"{RD}/results/contract_pick10.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    yr = pd.DataFrame({nm: (1 + v[0]).groupby(v[0].index.year).prod() - 1 for nm, v in books.items()})
    yr["IWM"] = (1 + iwm).groupby(iwm.index.year).prod() - 1
    print((yr * 100).round(1).to_string())


if __name__ == "__main__":
    main()
