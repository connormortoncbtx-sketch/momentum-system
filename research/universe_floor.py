"""
research/universe_floor.py -- should the momentum books reach further down into small / thin stocks? (Nov 12, 2026)

Connor: the system's best stretch came when very small stocks got in; the universe was pruned to guard against slippage
in thin markets, but with the live account in the hundreds, impact is nil -- revisit, maybe the old pick-10.

At a few hundred dollars our orders are far too small to move a price, so the only trading cost that matters is the
bid-ask spread, which IS much wider in small, thin, cheap stocks. The panel has no quotes, so costs are a per-name
half-spread table by dollar volume and price (literature-typical small-order spreads), shown at 1x and 2x.

Universes (weekly, non-fund, >= 53 weeks listed, no reverse split in 52 weeks; ADV = 4-week avg daily $ volume):
  U5_2M    price >= $5,   ADV >= $2M     today's research/SLOW-sleeve universe (live pipeline: $5, 500k sh, $300M cap)
  U2_1M    price >= $2,   ADV >= $1M
  U1_500k  price >= $1,   ADV >= $500k
  U1_250k  price >= $1,   ADV >= $250k
  U05_100k price >= $0.5, ADV >= $100k   (close to no floor)
  SMALL    in U1_250k but NOT in U5_2M   (only the names a looser floor would add)
Signals: live momentum proxy (the pick-10's momentum blend) and 12-1 momentum (the SLOW sleeve).
Books (10 names, equal weight): weekly full turnover (the old pick-10), weekly with a top-30 buffer, monthly (4-week)
with a top-30 buffer. Monday close in, held continuously.
Delistings: a name with no next-week price books MISS (default -30% if it was under $5 or under $2M/day -- small-cap
disappearances are mostly failures; 0% for the rest, mostly takeovers). Shown also with 0% everywhere.
Dev 2017-2023 / holdout 2024 - Sep 2026 (where the panel ends), plus each year. Public data only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)


def half_spread_bps(price, adv):
    """Per-side cost for a tiny order = half the quoted spread (bps), by dollar volume and price."""
    t = np.select([adv >= 20e6, adv >= 5e6, adv >= 2e6, adv >= 1e6, adv >= 5e5, adv >= 2.5e5],
                  [5, 10, 20, 35, 60, 100], 150).astype(float)
    t += np.where(price < 2, 60, np.where(price < 5, 30, 0))
    return t


def sim(score, mask, wk_ret, cost, small, n=10, k=1, buffer=None, miss_small=-0.30, miss_other=0.0):
    w = pd.Series(dtype=float)
    out = {}
    for i, wk in enumerate(score.index):
        c = 0.0
        if i % k == 0:
            s = score.loc[wk].where(mask.loc[wk]).dropna()
            if len(s) >= n:
                if buffer:
                    rk = s.rank(ascending=False)
                    keep = [h for h in w.index if h in rk.index and rk[h] <= buffer]
                    fill = [x for x in s.nlargest(n + len(keep)).index if x not in keep][: n - len(keep)]
                    names = keep + fill
                else:
                    names = list(s.nlargest(n).index)
                tgt = pd.Series(1.0 / n, index=names)
                tr = tgt.sub(w, fill_value=0).abs()
                cc = cost.loc[wk].reindex(tr.index).fillna(150.0)
                c = float((tr * cc).sum()) / 1e4
                w = tgt
        if w.empty:
            continue
        r = wk_ret.loc[wk].reindex(w.index)
        if r.notna().mean() < 0.5:
            break
        miss = r.isna()
        if miss.any():
            sm = small.loc[wk].reindex(w.index).fillna(True)
            r = r.where(~miss, np.where(sm, miss_small, miss_other))
        out[wk] = float((w * r).sum()) - c
        w = w * (1 + r)
        w = w.drop(miss[miss].index, errors="ignore")
        w = w[w > 0]
        w = w / w.sum() if w.sum() > 0 else w
    return pd.Series(out, dtype=float)


def st(r):
    r = r.dropna()
    if len(r) < 20:
        return dict(CAGR=np.nan, Sharpe=np.nan, maxDD=np.nan)
    eq = (1 + r).cumprod(); y = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / y) - 1) * 100, Sharpe=r.mean() / r.std() * np.sqrt(52),
                maxDD=(eq / eq.cummax() - 1).min() * 100)


def main():
    p = B.load_panel(RD)
    f = B.features(p)
    print(f"panel {p.weeks[0].date()} .. {p.weeks[-1].date()}, {len(p.symbols):,} symbols", flush=True)
    base = (f["hist"] >= 53) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols) \
        & ~f["rsplit_52w"].fillna(False).astype(bool)
    px, adv = f["price"], f["adv"]
    U = {"U5_2M": (5, 2e6), "U2_1M": (2, 1e6), "U1_500k": (1, 5e5), "U1_250k": (1, 2.5e5), "U05_100k": (0.5, 1e5)}
    masks = {k: base & (px >= a) & (adv >= b) for k, (a, b) in U.items()}
    masks["SMALL"] = masks["U1_250k"] & ~masks["U5_2M"]
    small = (px < 5) | (adv < 2e6)
    cost1 = pd.DataFrame(half_spread_bps(px.fillna(0).to_numpy(), adv.fillna(0).to_numpy()), index=p.weeks, columns=p.symbols)
    d1c = p.df("d1_close")
    wk_ret = (d1c.shift(-2) / d1c.shift(-1) - 1)
    wk_ret = wk_ret.where(wk_ret.abs() < 3.0)        # small caps can really double; only kill absurd prints
    rk = lambda x: x.rank(axis=1, pct=True)
    sig = {"live_proxy": (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80,
           "m12_1": f["m12_1"]}
    books = {"weekly_pick10": dict(k=1, buffer=None), "weekly_buf30": dict(k=1, buffer=30), "monthly_buf30": dict(k=4, buffer=30)}
    rows, series = [], {}
    for sn, s in sig.items():
        for bn, kw in books.items():
            for un, m in masks.items():
                # rank WITHIN each universe (percentile ranks recomputed inside the mask for the proxy)
                sc = s.where(m)
                if sn == "live_proxy":
                    sc = (0.35 * rk(f["rs_live"].where(m)) + 0.25 * rk(f["trend"].where(m)) + 0.20 * rk(f["hi52"].where(m))) / 0.80
                for cm, cost in (("1x", cost1), ("2x", cost1 * 2)):
                    for dm, ms in (("miss-30%", -0.30), ("miss0", 0.0)):
                        if cm == "2x" and dm == "miss0":
                            continue
                        r = sim(sc, m, wk_ret, cost, small, k=kw["k"], buffer=kw["buffer"], miss_small=ms)
                        series[(sn, bn, un, cm, dm)] = r
                        d, h = st(r[r.index < SPLIT]), st(r[r.index >= SPLIT])
                        l12 = st(r[r.index >= r.index[-1] - pd.Timedelta(weeks=52)])
                        avgsz = float(m.sum(axis=1)[m.index >= SPLIT].mean())
                        rows.append(dict(signal=sn, book=bn, universe=un, cost=cm, delist=dm, names=round(avgsz),
                                         dev_CAGR=d["CAGR"], dev_Sharpe=d["Sharpe"], dev_DD=d["maxDD"],
                                         hold_CAGR=h["CAGR"], hold_Sharpe=h["Sharpe"], hold_DD=h["maxDD"],
                                         last52_CAGR=l12["CAGR"]))
                print(f"  done {sn} {bn} {un}", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/universe_floor.csv", index=False)
    main_ = R[(R.cost == "1x") & (R.delist == "miss-30%")]
    print("\nMAIN (1x spread costs, small-cap delistings -30%). CAGR %, Sharpe, max drawdown %:\n"
          + main_.drop(columns=["cost", "delist"]).round(2).to_string(index=False))
    print("\nSENSITIVITY: same rows at 2x costs, and with delistings at 0%:\n"
          + R[R.cost.eq("2x") | R.delist.eq("miss0")].round(2).to_string(index=False))
    # by year for the headline book
    yr = {}
    for un in masks:
        r = series[("live_proxy", "weekly_pick10", un, "1x", "miss-30%")]
        yr[un] = r.groupby(r.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    print("\nBY YEAR, old pick-10 (live proxy, weekly, 1x costs), % return:\n" + pd.DataFrame(yr).round(1).to_string())
    yr = {}
    for un in masks:
        r = series[("m12_1", "monthly_buf30", un, "1x", "miss-30%")]
        yr[un] = r.groupby(r.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    print("\nBY YEAR, 12-1 momentum monthly top-10 (1x costs), % return:\n" + pd.DataFrame(yr).round(1).to_string())
    # spring 2026, when Connor remembers the best run
    sp = {un: series[("live_proxy", "weekly_pick10", un, "1x", "miss-30%")].loc["2026-03-01":"2026-06-30"] for un in masks}
    print("\nSPRING 2026 (Mar-Jun), old pick-10, compounded %: "
          + ", ".join(f"{k} {((1 + v).prod() - 1) * 100:+.1f}" for k, v in sp.items()))
    h = main_[main_.universe.isin(["U5_2M", "U1_250k"])][["signal", "book", "universe", "hold_CAGR", "hold_Sharpe"]]
    print("::notice::universe floor (holdout CAGR/Sharpe): " + "; ".join(
        f"{r.signal}/{r.book}/{r.universe} {r.hold_CAGR:+.1f}%/{r.hold_Sharpe:.2f}" for r in h.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
