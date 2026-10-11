"""
research/rank_bands.py -- full-spectrum rank-band study (Oct 2026)

For each score, rank the universe every Friday, cut the ranks into bands
(fine at the head, percentile bins across the rest) and measure each band's
forward excess return vs the equal-weight universe over 1/2/4/8/13 weeks.
Dev 2017-2023 finds shapes; holdout 2024-2026 checks them. Newey-West t-stats
(lag = horizon-1) for overlapping horizons.

  PANEL=<research-data checkout> python research/rank_bands.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
OUT = os.environ.get("OUT", "research/results")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
d1c, cl = p.df("d1_close"), p.df("close")
rk = lambda x: B.xrank(x, U)
start, split = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01")

SCORES = {
    "live_proxy": (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80,
    "m12_1": f["m12_1"], "m6_1": f["m6_1"], "m3_1": f["m3_1"], "rs_live": f["rs_live"],
    "hi52": f["hi52"], "trend": f["trend"] + 1e-6 * rk(f["rs_live"]),   # tie-break the 0-3 trend count
    "r1w": f["r1w"], "r4w": f["r4w"], "low_vol": -f["vol12"],
}
H = [1, 2, 4, 8, 13]
FWD = {h: (cl.shift(-h) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 3) for h in H}

HEAD = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 75), (76, 100), (101, 150),
        (151, 200), (201, 300), (301, 500)]
PCT = [(i / 50, (i + 1) / 50) for i in range(50)]          # 2% bins over the whole list


def nw_t(x, lag):
    x = x.dropna().to_numpy(); n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean(); v = e @ e / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def band_series(score, h, mask=None):
    fw = FWD[h]; m = U & fw.notna() & score.notna()
    if mask is not None:
        m &= mask
    s = score.where(m)
    r = s.rank(axis=1, ascending=False, method="first")
    pct = r.div(r.max(axis=1), axis=0)
    ex = fw.where(m).sub(fw.where(m).mean(axis=1), axis=0)
    out = {}
    for lo, hi in HEAD:
        out[f"r{lo}-{hi}"] = ex.where((r >= lo) & (r <= hi)).mean(axis=1)
    for lo, hi in PCT:
        out[f"p{int(lo*100):02d}-{int(hi*100):02d}"] = ex.where((pct > lo) & (pct <= hi)).mean(axis=1)
    return pd.DataFrame(out)


def summarize(bs, h):
    rows = {}
    for col in bs:
        d = bs[col][(bs.index >= start) & (bs.index < split)]
        o = bs[col][bs.index >= split]
        rows[col] = {"dev%": d.mean() * 100, "dev_t": nw_t(d, h - 1),
                     "hold%": o.mean() * 100, "hold_t": nw_t(o, h - 1)}
    return pd.DataFrame(rows).T


def main():
    os.makedirs(OUT, exist_ok=True)
    allrows = []
    for sn, sc in SCORES.items():
        for h in H:
            t = summarize(band_series(sc, h), h)
            t["score"], t["h"] = sn, h
            allrows.append(t)
        print("done", sn, flush=True)
    res = pd.concat(allrows).rename_axis("band").reset_index()
    res.to_csv(f"{OUT}/rank_bands.csv", index=False)

    # conditional cuts for the two main scores at 1 and 4 weeks
    big = f["adv"].where(U).rank(axis=1, ascending=False) <= 500           # top-500 by $ volume
    spy = p.df("close")["SPY"]
    up = (spy > spy.rolling(40).mean())
    up_mask = pd.DataFrame(np.repeat(up.to_numpy()[:, None], len(p.symbols), 1), index=p.weeks, columns=p.symbols)
    vol_hi = (f["r1w"].where(U).std(axis=1))
    disp = vol_hi > vol_hi.rolling(52, min_periods=26).median()
    disp_mask = pd.DataFrame(np.repeat(disp.to_numpy()[:, None], len(p.symbols), 1), index=p.weeks, columns=p.symbols)
    cond = []
    for sn in ("live_proxy", "m12_1", "r1w"):
        for h in (1, 4):
            for cname, msk in (("large500", big), ("small_rest", ~big), ("spy_up", up_mask),
                               ("spy_down", ~up_mask), ("disp_high", disp_mask), ("disp_low", ~disp_mask)):
                t = summarize(band_series(SCORES[sn], h, msk), h)
                t["score"], t["h"], t["cond"] = sn, h, cname
                cond.append(t)
        print("cond done", sn, flush=True)
    pd.concat(cond).rename_axis("band").reset_index().to_csv(f"{OUT}/rank_bands_conditional.csv", index=False)


if __name__ == "__main__":
    main()
