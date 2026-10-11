"""
research/combined_book.py -- "best Sharpe we can build" (Oct 11, 2026).

Pieces (all from data with full universe coverage; SEC acceptance times are
conservative, i.e. never early):
  LONG core   slow 12-1 momentum, top 10, hold while ranked <= 100 (shadow #4 design)
              + filing exclusions: a flagged name cannot be bought and is sold at the
              next weekly rebalance
                offering  424B1/3/4/5 prospectus or 8-K item 3.02, last 8 weeks
                late      NT 10-K / NT 10-Q, last 13 weeks
                distress  8-K items 3.01 / 3.03 / 5.01, last 8 weeks
                merger    Form 425 / PREM14A / DEFM14A, last 13 weeks
  SHORT leg   short every name with a NEW late / offering / distress flag that week,
              hold H weeks, equal weight. Borrow realism: price >= $5, >= $5M/day
              (likely easy-to-borrow), borrow fee 4%/yr (sensitivity 10%), 15 bps/side.
              Short returns are NOT clipped at +150% (squeezes count; > +400% treated as a
              data error and capped at +400%).
  COMBINED    long + s * short, s in {0.25, 0.5, 1.0}
  LEVERAGE    weekly returns x k, financing 6%/yr on borrowed long exposure;
              full-Kelly k* = mean / variance of weekly excess return.
Trading: formation Friday close, trade Monday close (as in hold_conditions.py).
Dev 2017-2023 chooses; holdout 2024-2026 scored once.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{P}/sec")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
W, S = p.weeks, p.symbols
U = B.universe(p, f, 5.0, 2e6)
d1c, C = p.df("d1_close"), p.df("close")
WR_RAW = d1c.shift(-2) / d1c.shift(-1) - 1                 # Mon close -> Mon close, aligned to formation week
WR = WR_RAW.where(WR_RAW.abs() < 1.5)                      # long side: as in hold_conditions.py
WR_SHORT = WR_RAW.clip(upper=4.0)                          # short side: keep squeezes
START, SPLIT, COST = pd.Timestamp("2017-01-06"), pd.Timestamp("2024-01-01"), 15.0

# ── filing flags ───────────────────────────────────────────────────────────────
fi = pd.concat(pd.read_parquet(x, columns=["tickers", "form", "items", "accepted_et"])
               for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
fi = fi[fi.accepted_et >= "2015-06-01"]
fi["symbol"] = fi.tickers.fillna("").str.split(",").str[0]
sh = (fi.accepted_et + pd.Timedelta(hours=8)).dt.normalize()
fi["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")
fi = fi[fi.symbol.isin(set(S)) & fi.week.isin(set(W))]
items = fi["items"].fillna("")


def has_item(codes):
    return (fi.form == "8-K") & items.apply(lambda s: any(c in s.split(",") for c in codes))


EVENTS = {"offering": fi.form.isin(["424B1", "424B3", "424B4", "424B5"]) | has_item(["3.02"]),
          "late": fi.form.isin(["NT 10-K", "NT 10-Q"]),
          "distress": has_item(["3.01", "3.03", "5.01"]),
          "merger": fi.form.isin(["425", "PREM14A", "DEFM14A"])}
WINDOW = {"offering": 8, "late": 13, "distress": 8, "merger": 13}
NEW, FLAG = {}, {}
for k, m in EVENTS.items():
    g = fi[m].groupby(["week", "symbol"]).size().unstack().reindex(index=W, columns=S).fillna(0) > 0
    NEW[k] = g
    FLAG[k] = g.astype(float).rolling(WINDOW[k], min_periods=1).sum() > 0


def sim_long(block=None, n=10, buffer=100):
    score = f["m12_1"].where(U).to_numpy(); Wr = WR.to_numpy()
    Bk = None if block is None else block.reindex(index=W, columns=S).to_numpy()
    w, out = {}, {}
    for i, wk in enumerate(W):
        if wk < START - pd.Timedelta(days=7 * 60):
            continue
        row = score[i]; ok = ~np.isnan(row) & ~np.isnan(Wr[i])
        if ok.sum() < n * 3:
            continue
        order = np.argsort(-np.where(ok, row, -np.inf))
        rank = np.empty_like(order); rank[order] = np.arange(1, len(order) + 1)
        bad = (lambda c: False) if Bk is None else (lambda c: bool(Bk[i, c]))
        keep = [c for c in w if ok[c] and rank[c] <= buffer and not bad(c)]
        cands = [c for c in order[: n * 8] if ok[c] and c not in keep and not bad(c)]
        new = keep + cands[: n - len(keep)]
        tgt = {c: 1.0 / len(new) for c in new}
        cost = sum(abs(tgt.get(c, 0) - w.get(c, 0)) for c in set(tgt) | set(w)) * COST / 1e4
        r = np.nan_to_num(np.array([Wr[i, c] for c in tgt]))
        wt = np.array(list(tgt.values()))
        out[wk] = float((wt * r).sum()) - cost
        nw = wt * (1 + r); nw = nw / nw.sum()
        w = dict(zip(tgt.keys(), nw))
    s = pd.Series(out, dtype=float)
    return s[s.index >= START]


def sim_short(kinds, H=4, borrow=0.04, min_price=5.0, min_adv=5e6):
    elig = (f["price"] >= min_price) & (f["adv"] >= min_adv) & U
    trig = sum(NEW[k].astype(int) for k in kinds) > 0
    trig = (trig & elig).to_numpy()
    Wr = WR_SHORT.to_numpy()
    active = np.zeros_like(trig, dtype=bool)
    for h in range(H):                                     # entered in the last H formation weeks
        active[h:] |= trig[: len(trig) - h] if h else trig
    out, npos = {}, {}
    for i, wk in enumerate(W):
        if wk < START:
            continue
        cols = np.nonzero(active[i])[0]
        if len(cols) == 0:
            out[wk] = 0.0; npos[wk] = 0; continue
        r = np.nan_to_num(Wr[i, cols])                     # delisted -> 0 (conservative for shorts)
        out[wk] = -r.mean() - borrow / 52 - 2 * COST / 1e4 / H
        npos[wk] = len(cols)
    return pd.Series(out, dtype=float), pd.Series(npos)


def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0,
                vol=r.std() * np.sqrt(52) * 100, Sharpe=r.mean() / r.std() * np.sqrt(52),
                maxDD=(eq / eq.cummax() - 1).min() * 100)


rows = []


def run(name, r, part):
    d, h = r[r.index < SPLIT], r[r.index >= SPLIT]
    rows.append(dict(part=part, design=name, **{f"dev_{a}": b for a, b in st(d).items()},
                     **{f"hold_{a}": b for a, b in st(h).items()}))
    return r


pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
spy = WR["SPY"].dropna(); spy = spy[spy.index >= START]
run("SPY", spy, "benchmark")

# 1. long core + exclusions
L = {"none": run("core (no exclusions)", sim_long(), "long")}
for k in FLAG:
    L[k] = run(f"core excl {k}", sim_long(FLAG[k]), "long")
allflag = FLAG["offering"] | FLAG["late"] | FLAG["distress"] | FLAG["merger"]
L["all"] = run("core excl all four", sim_long(allflag), "long")
L["neg3"] = run("core excl offering+late+distress", sim_long(FLAG["offering"] | FLAG["late"] | FLAG["distress"]), "long")
print("long done", flush=True)

# 2. short legs
SH = {}
for kinds in (["late"], ["offering"], ["distress"], ["late", "offering", "distress"]):
    for H in (4, 13):
        nm = "+".join(kinds) + f" H{H}"
        s, n = sim_short(kinds, H)
        SH[nm] = run(f"short {nm} (avg {n[n.index >= START].mean():.0f} names)", s, "short")
s10, _ = sim_short(["late", "offering", "distress"], 13, borrow=0.10)
run("short late+offering+distress H13, borrow 10%/yr", s10, "short")
s_lo, _ = sim_short(["late", "offering", "distress"], 13, min_price=2.0, min_adv=1e6)
run("short late+offering+distress H13, looser borrow screen ($2, $1M/day)", s_lo, "short")
print("short done", flush=True)

res = pd.DataFrame(rows)
print(res.round(2).to_string(index=False))

# 3. combined: choose long variant and short leg on DEV Sharpe
lr = res[res.part == "long"].sort_values("dev_Sharpe", ascending=False).iloc[0]
sr = res[res.part == "short"][~res.design.str.contains("borrow")].sort_values("dev_Sharpe", ascending=False).iloc[0]
lkey = {v: k for k, v in {"none": "core (no exclusions)", **{k: f"core excl {k}" for k in FLAG},
                          "all": "core excl all four", "neg3": "core excl offering+late+distress"}.items()}[lr.design]
skey = sr.design.split(" (")[0].replace("short ", "")
print(f"\nchosen on dev: long = {lr.design} | short = {skey}")
rows.clear()
run("SPY", spy, "benchmark")
run(f"long {lr.design}", L[lkey], "combo")
combos = {}
for s in (0.25, 0.5, 1.0):
    c = (L[lkey] + s * SH[skey].reindex(L[lkey].index).fillna(0))
    combos[s] = run(f"long + {s:.2f} x short", c, "combo")
cres = pd.DataFrame(rows)
print(cres.round(2).to_string(index=False))
print("\ncorrelation long vs short (weekly):",
      round(L[lkey].corr(SH[skey].reindex(L[lkey].index)), 3))

# 4. leverage
FIN = 0.06 / 52
print("\nLEVERAGE (gross long exposure x k; financing 6%/yr on the borrowed part)")
best = max(combos, key=lambda s: st(combos[s][combos[s].index < SPLIT])["Sharpe"])
lev_rows = []
for name, r in (("long only", L[lkey]), (f"long + {best:.2f} x short", combos[best])):
    for per, rr in (("dev", r[r.index < SPLIT]), ("holdout", r[r.index >= SPLIT])):
        ex = rr - 0.04 / 52
        kstar = ex.mean() / ex.var()
        for k in (1.0, 1.5, 2.0, 3.0, round(kstar / 2, 2), round(kstar, 2)):
            lr_ = k * rr - (k - 1) * FIN if k > 1 else k * rr
            s_ = st(lr_)
            lev_rows.append(dict(book=name, period=per, leverage=k, CAGR=s_["CAGR"], maxDD=s_["maxDD"],
                                 weekly_geo=((1 + s_["CAGR"] / 100) ** (1 / 52) - 1) * 100, kelly=kstar))
lev = pd.DataFrame(lev_rows).drop_duplicates(["book", "period", "leverage"])
print(lev.round(2).to_string(index=False))
print("\ntarget: 3.00%/week geometric = 366%/yr")
