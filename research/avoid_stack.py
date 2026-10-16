"""
research/avoid_stack.py -- do the small "remove a disadvantage" fixes add up? (Oct 16, 2026)

Two base books on the weekly research panel (formation Friday close, trade Monday close,
15 bps per side), universe price >= $5, >= $2M/day (research/backtest.py universe()):
  LIVE  live-system proxy (0.35 RS + 0.25 trend + 0.20 52w-high ranks), top 10, full weekly refresh
  SLOW  12-1 momentum, top 10, hold while ranked <= 100 (shadow #4)

Fixes (each known at the formation Friday):
  no_spike    skip the 5 biggest 1-week and 4-week gainers   (rank-band study: they lag)
  no_topvol   skip the top decile of 12-week volatility       (wildest decile: mean negative)
  no_late     skip NT 10-K / NT 10-Q filers, last 13 weeks     (-5% / -11% over 13 weeks)
  no_distress skip 8-K items 3.01 / 3.03 / 5.01, last 8 weeks
  no_merger   skip Form 425 / PREM14A / DEFM14A, last 13 weeks (-2.5% / -4.6% over 13 weeks)
  no_offering skip 424B1/3/4/5 or item 3.02, last 8 weeks
  no_badnews  skip names with an LLM-judged 8-K of direction <= -1 in the last 2 weeks
  hold50      (LIVE only) keep holdings while ranked <= 50 instead of a full weekly refresh
  stop10      hard stop at -10% from Monday-close entry (weekly low after Monday; approximate)
Each fix alone, then a greedy stack chosen on DEV (2017-2023) Sharpe, each step scored on
HOLDOUT (2024-2026); plus everything on at once.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = f"{P}/sec"
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
W, S = p.weeks, p.symbols
U = B.universe(p, f, 5.0, 2e6)
d1c = p.df("d1_close"); lowd = p.df("low_after_d1")
WR = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
LOWR = (lowd.shift(-1) / d1c.shift(-1) - 1)          # worst dip after entry within the holding week (approx.)
START, SPLIT, COST = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01"), 15.0
rk = lambda x: B.xrank(x, U)
SCORES = {"LIVE": (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80,
          "SLOW": f["m12_1"]}


def week_label(ts, hours=8):
    sh = (pd.to_datetime(ts) + pd.Timedelta(hours=hours)).dt.normalize()
    return sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")


def grid(df, window):
    g = df[df.symbol.isin(set(S)) & df.week.isin(set(W))].groupby(["week", "symbol"]).size().unstack()
    g = g.reindex(index=W, columns=S).fillna(0)
    return g.rolling(window, min_periods=1).sum() > 0


fi = pd.concat(pd.read_parquet(x, columns=["tickers", "form", "items", "accepted_et"])
               for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
fi = fi[fi.accepted_et >= "2015-06-01"]
fi["symbol"] = fi.tickers.fillna("").str.split(",").str[0]
fi["week"] = week_label(fi.accepted_et)
it = fi["items"].fillna("")
has = lambda codes: (fi.form == "8-K") & it.apply(lambda s: any(c in s.split(",") for c in codes))
BLOCK = {"no_late": grid(fi[fi.form.isin(["NT 10-K", "NT 10-Q"])], 13),
         "no_distress": grid(fi[has(["3.01", "3.03", "5.01"])], 8),
         "no_merger": grid(fi[fi.form.isin(["425", "PREM14A", "DEFM14A"])], 13),
         "no_offering": grid(fi[fi.form.isin(["424B1", "424B3", "424B4", "424B5"]) | has(["3.02"])], 8)}
J, E = [], []
for d in [f"{P}/llm"] + sorted(glob.glob(f"{P}/llm_full/part*")) + sorted(glob.glob(f"{P}/llm_hist/part*")):
    if os.path.exists(f"{d}/judgements.parquet"):
        J.append(pd.read_parquet(f"{d}/judgements.parquet", columns=["accession", "direction"]))
        E.append(pd.read_parquet(f"{d}/events.parquet", columns=["accession", "symbol", "accepted_true"]))
ev = pd.concat(E).drop_duplicates("accession").merge(pd.concat(J).drop_duplicates("accession"), on="accession")
ev["week"] = week_label(ev.accepted_true)
BLOCK["no_badnews"] = grid(ev[ev.direction <= -1], 2)
r1rank = f["r1w"].where(U).rank(axis=1, ascending=False)
r4rank = f["r4w"].where(U).rank(axis=1, ascending=False)
BLOCK["no_spike"] = (r1rank <= 5) | (r4rank <= 5)
BLOCK["no_topvol"] = f["vol12"].where(U).rank(axis=1, pct=True) >= 0.9
BLOCK = {k: v.reindex(index=W, columns=S).fillna(False).to_numpy(dtype=bool) for k, v in BLOCK.items()}
print("blocks ready:", {k: round(float(v[W >= START].mean()) * 100, 2) for k, v in BLOCK.items()}, "% of name-weeks", flush=True)


def sim(base, fixes, n=10):
    score = SCORES[base].where(U).to_numpy(); Wr = WR.to_numpy(); Lo = LOWR.to_numpy()
    blk = np.zeros_like(score, dtype=bool)
    for fx in fixes:
        if fx in BLOCK:
            blk |= BLOCK[fx]
    buffer = 100 if base == "SLOW" else (50 if "hold50" in fixes else None)
    stop = 0.10 if "stop10" in fixes else None
    w, out = {}, {}
    for i, wk in enumerate(W):
        if wk < START - pd.Timedelta(days=7 * 60):
            continue
        row = score[i]; ok = ~np.isnan(row) & ~np.isnan(Wr[i])
        if ok.sum() < n * 3:
            continue
        order = np.argsort(-np.where(ok, row, -np.inf))
        rank = np.empty_like(order); rank[order] = np.arange(1, len(order) + 1)
        keep = [c for c in w if buffer and ok[c] and rank[c] <= buffer and not blk[i, c]]
        cands = [c for c in order[: n * 10] if ok[c] and c not in keep and not blk[i, c]]
        new = keep + cands[: n - len(keep)]
        tgt = {c: 1.0 / len(new) for c in new}
        cost = sum(abs(tgt.get(c, 0) - w.get(c, 0)) for c in set(tgt) | set(w)) * COST / 1e4
        r = np.nan_to_num(np.array([Wr[i, c] for c in tgt]))
        if stop:
            lo = np.array([Lo[i, c] for c in tgt])
            hit = np.isfinite(lo) & (lo <= -stop)
            r = np.where(hit, -stop - COST / 1e4, r)
        wt = np.array(list(tgt.values()))
        out[wk] = float((wt * r).sum()) - cost
        nw = wt * (1 + r); w = dict(zip(tgt.keys(), nw / nw.sum()))
    s = pd.Series(out, dtype=float)
    return s[s.index >= START]


def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0,
                Sharpe=r.mean() / r.std() * np.sqrt(52), maxDD=(eq / eq.cummax() - 1).min() * 100)


def both(r):
    d, h = st(r[r.index < SPLIT]), st(r[r.index >= SPLIT])
    return d, h


def fmt(d):
    return f"{d['CAGR']:6.1f}% Sh {d['Sharpe']:5.2f} DD {d['maxDD']:6.1f}%"


FIXES = ["no_spike", "no_topvol", "no_late", "no_distress", "no_merger", "no_offering", "no_badnews", "stop10"]
for base in (("LIVE", "SLOW") if __name__ == "__main__" else ()):
    fixes = FIXES + (["hold50"] if base == "LIVE" else [])
    r0 = sim(base, []); d0, h0 = both(r0)
    print(f"\n=== {base} ===\n  {'base':28s} dev {fmt(d0)} | holdout {fmt(h0)}")
    alone = {}
    for fx in fixes:
        d, h = both(sim(base, [fx])); alone[fx] = d["Sharpe"]
        print(f"  + {fx:26s} dev {fmt(d)} (dSh {d['Sharpe'] - d0['Sharpe']:+.2f}) | holdout {fmt(h)} (dSh {h['Sharpe'] - h0['Sharpe']:+.2f})")
    print("  GREEDY STACK (each step adds the fix that most improves DEV Sharpe; stops when none helps):")
    chosen, cur = [], d0["Sharpe"]
    while True:
        best = None
        for fx in [x for x in fixes if x not in chosen]:
            d, _ = both(sim(base, chosen + [fx]))
            if d["Sharpe"] > cur + 0.01 and (best is None or d["Sharpe"] > best[1]):
                best = (fx, d["Sharpe"])
        if best is None:
            break
        chosen.append(best[0]); cur = best[1]
        d, h = both(sim(base, chosen))
        print(f"    {' + '.join(chosen):60s} dev {fmt(d)} | holdout {fmt(h)}")
    d, h = both(sim(base, fixes))
    print(f"  ALL FIXES ON {'':47s} dev {fmt(d)} | holdout {fmt(h)}")
