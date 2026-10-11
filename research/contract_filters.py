"""
research/contract_filters.py -- contract-win book: out-of-sample history, then filters,
exits, sizing and a hedge (Oct 11, 2026).

Data: every qualifying small-cap 8-K read by the LLM, 2017-01 .. 2026-09
(llm/, llm_full/part*, llm_hist/part*), plus llm_counterparty.parquet when present.
Caveat: the SEC filing set is keyed by companies with a CURRENT ticker, so 2017-2023
excludes firms that later delisted (survivorship); the random-filing control shares it.

  A. OUT-OF-SAMPLE: the design chosen on Jan 2024 - Mar 2025 (contract_win, direction >= 1,
     value >= 10% of market cap, hold 20 sessions, equal weight, no cap) scored on 2017-2023,
     which it never saw. Year by year, plus 10 random-filing control books.
  B. FILTERS: event features known at entry; terciles on DEV = 2017-2023; a feature is
     adopted when its worst tercile trails the rest by t < -2 on DEV; the combined filter
     is scored ONCE on HOLDOUT = 2024-2026 (which chose the base design -- not pristine,
     but the filters never saw it).
  C. EXITS / SIZING / HEDGE: chosen on DEV, scored on HOLDOUT.
Entry: first regular-session open after acceptance; costs 25 bps per side.
"""
import glob, os, sys
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = os.environ.get("OUT") or f"{RD}/results"
SPLIT, START = pd.Timestamp("2024-01-01"), pd.Timestamp("2017-01-01")
COST = 25.0


class Tee:
    def __init__(self, *fs): self.fs = fs
    def write(self, x): [f.write(x) for f in self.fs]
    def flush(self): [f.flush() for f in self.fs]


def load():
    J, E = [], []
    for d in [f"{RD}/llm"] + sorted(glob.glob(f"{RD}/llm_full/part*")) + sorted(glob.glob(f"{RD}/llm_hist/part*")):
        if os.path.exists(f"{d}/judgements.parquet") and os.path.exists(f"{d}/events.parquet"):
            J.append(pd.read_parquet(f"{d}/judgements.parquet")); E.append(pd.read_parquet(f"{d}/events.parquet"))
    ev = pd.concat(E).drop_duplicates("accession").merge(pd.concat(J).drop_duplicates("accession"), on="accession")
    ev["t"] = pd.to_datetime(ev.accepted_true)
    ev = ev[(ev.t >= START) & ev.ok.fillna(False)].copy()
    ev["rel_value"] = ev.dollar_value_usd / ev.mcap
    ev["rel_rev"] = ev.dollar_value_usd / ev.revenue_ttm.where(ev.revenue_ttm > 0)
    ev["going_concern"] = ev.red_flags.fillna("").str.contains("going concern", case=False)
    ev["n_flags"] = ev.red_flags.fillna("").apply(lambda s: len([x for x in s.split("|") if x]))
    cp = f"{RD}/llm_counterparty.parquet"
    if os.path.exists(cp):
        ev = ev.merge(pd.read_parquet(cp), on="accession", how="left")
    return ev


def bars(symbols, start):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    cl = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames, syms = [], sorted(set(symbols))
    for i in range(0, len(syms), 150):
        try:
            d = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 150], timeframe=TimeFrame.Day,
                                                   start=start.to_pydatetime(), adjustment=Adjustment.ALL,
                                                   feed=DataFeed.SIP)).df
            if len(d):
                frames.append(d.reset_index()[["symbol", "timestamp", "open", "close", "volume"]])
        except Exception as ex:
            print(f"bars batch {i // 150}: {ex}")
    b = pd.concat(frames)
    b["date"] = pd.to_datetime(b.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    piv = lambda c: b.pivot_table(index="date", columns="symbol", values=c).astype("float32")
    return piv("open"), piv("close"), piv("volume")


class Market:
    def __init__(self, O, C, V):
        self.O, self.C, self.V = O, C, V
        self.days = C.index
        self.R = (C / C.shift(1) - 1)
        self.open_t = self.days + pd.Timedelta(hours=9, minutes=30)

    def entry_pos(self, t):
        return np.searchsorted(self.open_t.values, pd.to_datetime(t).values.astype("datetime64[ns]"), side="right")

    def features(self, ev):
        """Features known BEFORE the entry open (pre_*) or AT it (gap)."""
        p = self.entry_pos(ev.t)
        C, V = self.C, self.V
        out = {k: np.full(len(ev), np.nan) for k in ("pre_mom20", "pre_mom60", "dist_hi252", "vol60", "pre_volsurge5",
                                                      "gap_at_entry")}
        for n, (sym, i) in enumerate(zip(ev.symbol, p)):
            if sym not in C.columns or i < 61 or i >= len(self.days):
                continue
            c = C[sym].to_numpy(); v = V[sym].to_numpy(); o0 = self.O[sym].iloc[i]
            prev = c[i - 1]
            if not np.isfinite(prev) or prev <= 0:
                continue
            out["pre_mom20"][n] = prev / c[i - 21] - 1 if np.isfinite(c[i - 21]) else np.nan
            out["pre_mom60"][n] = prev / c[i - 61] - 1 if np.isfinite(c[i - 61]) else np.nan
            hi = np.nanmax(c[max(0, i - 252):i]); out["dist_hi252"][n] = prev / hi - 1
            r = np.diff(np.log(c[i - 61:i])); out["vol60"][n] = np.nanstd(r) * np.sqrt(252)
            mv = np.nanmedian(v[i - 61:i - 5]); out["pre_volsurge5"][n] = np.nanmean(v[i - 5:i]) / mv if mv > 0 else np.nan
            out["gap_at_entry"][n] = o0 / prev - 1 if np.isfinite(o0) else np.nan
        f = pd.DataFrame(out, index=ev.index)
        f["entry_i"] = p
        return f

    def book(self, ev, H=20, cap=1.0, kill_day=None, kill_below=0.0, trail=None, size="equal", hedge=0.0):
        """Daily returns. Per position: entry open -> exit close. Exits: after H sessions; optionally
        at the close of kill_day if return-since-entry < kill_below; optionally trailing stop
        (close below peak*(1-trail)). Weights: equal or 1/vol60 among open positions, cap per name,
        rest cash. hedge: short IWM = hedge x gross long exposure."""
        days = self.days; nd = len(days)
        paths = {}            # day -> list of (ret, raw_weight)
        P = self.entry_pos(ev.t)
        vols = ev["vol60"].to_numpy() if "vol60" in ev else np.full(len(ev), np.nan)
        for sym, i, vol in zip(ev.symbol, P, vols):
            if sym not in self.C.columns or i >= nd:
                continue
            o0 = self.O[sym].iloc[i]
            if not np.isfinite(o0) or o0 <= 0:
                continue
            c = self.C[sym].to_numpy()
            rw = 1.0 if size == "equal" else (1.0 / max(vol, 0.2) if np.isfinite(vol) else 1.0 / 0.8)
            peak, prev = o0, o0
            for h in range(H + 1):
                k = i + h
                if k >= nd or not np.isfinite(c[k]):
                    break
                r = c[k] / prev - 1 - ((2 * COST / 1e4) if h == 0 else 0.0)
                paths.setdefault(k, []).append((r, rw))
                prev = c[k]; peak = max(peak, c[k])
                if kill_day is not None and h == kill_day and c[k] / o0 - 1 < kill_below:
                    break
                if trail is not None and c[k] < peak * (1 - trail):
                    break
        out = np.zeros(nd); gross = np.zeros(nd)
        for k, lst in paths.items():
            rs = np.array([a for a, _ in lst]); ws = np.array([b for _, b in lst])
            ws = np.minimum(ws / ws.sum(), cap)
            out[k] = float((ws * rs).sum()); gross[k] = ws.sum()
        s = pd.Series(out, index=days)
        if hedge:
            iwm = self.R["IWM"].fillna(0).to_numpy()
            s = s - hedge * pd.Series(gross, index=days).shift(0).fillna(0) * iwm
        return s[s.index >= START]


def st(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, maxDD=np.nan)
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0,
                Sharpe=r.mean() / r.std() * np.sqrt(252), maxDD=(eq / eq.cummax() - 1).min() * 100)


def fmt(d):
    return " ".join(f"{k} {v:6.2f}" for k, v in d.items())


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 10 else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, open(f"{OUT}/contract_filters.txt", "w"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    ev = load()
    ev["period"] = np.where(ev.t < SPLIT, "dev", "holdout")
    print(f"judged 8-Ks: {len(ev):,}", ev.groupby(ev.t.dt.year).size().to_dict())
    cw = (ev.event_type == "contract_win") & (ev.direction >= 1)
    base = cw & (ev.rel_value >= 0.10)
    print(f"contract wins dir>=1: {int(cw.sum())} | base (>=10% of mcap): {int(base.sum())}",
          ev[base].groupby(ev[base].t.dt.year).size().to_dict())

    # ── A. event level, out of sample ────────────────────────────────────────
    print("\nA1. EVENT LEVEL (IWM-adjusted from the entry open, %) vs all judged 8-Ks of the same years")
    rng = np.random.default_rng(5)
    for nm, m in (("all contract wins", cw), ("base >=10% of mcap", base)):
        for per in ("dev", "holdout"):
            x = ev[m & (ev.period == per)].ab_20.dropna(); pool = ev[ev.period == per].ab_20.dropna().to_numpy()
            draws = np.array([rng.choice(pool, len(x), replace=False).mean() for _ in range(3000)])
            print(f"  {nm:20s} {per:8s} n {len(x):4d} ab20 mean {x.mean() * 100:6.2f} median {x.median() * 100:6.2f} "
                  f"| all-8-K mean {pool.mean() * 100:6.2f} | p(random >= signal) {(draws >= x.mean()).mean():.3f} "
                  f"| ab5 mean {ev[m & (ev.period == per)].ab_5.mean() * 100:5.2f}")
    print("  base by year (ab20 mean %, n):",
          {int(y): (round(g.ab_20.mean() * 100, 1), len(g)) for y, g in ev[base].groupby(ev[base].t.dt.year)})

    # bars: all contract wins + control sample + IWM/SPY
    ctrl_pool = {per: ev[ev.period == per] for per in ("dev", "holdout")}
    nb = {per: int((base & (ev.period == per)).sum()) for per in ("dev", "holdout")}
    ctrl = {per: ctrl_pool[per].sample(min(len(ctrl_pool[per]), nb[per] * 10), random_state=7) for per in nb}
    syms = list(ev[cw].symbol) + list(ctrl["dev"].symbol) + list(ctrl["holdout"].symbol) + ["IWM", "SPY"]
    M = Market(*bars(syms, START - pd.Timedelta(days=420)))
    F = M.features(ev[cw]); E = ev[cw].join(F)
    Eb = E[E.rel_value >= 0.10]

    print("\nA2. BASE BOOK (H20, equal weight, no cap) by period and year")
    r0 = M.book(Eb)
    for per, m in (("dev 2017-23", r0.index < SPLIT), ("holdout 2024-26", r0.index >= SPLIT)):
        print(f"  {per}: {fmt(st(r0[m]))}")
    iwm = M.R["IWM"]; iwm = iwm[iwm.index >= START]
    for per, m in (("dev", iwm.index < SPLIT), ("holdout", iwm.index >= SPLIT)):
        print(f"  IWM {per}: {fmt(st(iwm[m]))}")
    yr = pd.DataFrame({"book": r0, "iwm": iwm}).groupby(r0.index.year).apply(
        lambda g: pd.Series({"book_%": ((1 + g.book).prod() - 1) * 100, "iwm_%": ((1 + g.iwm).prod() - 1) * 100}))
    print(yr.round(1).to_string())
    print("  CONTROL: 10 random-filing books per period (same event counts):")
    cs = []
    for d in range(10):
        smp = pd.concat([ctrl[per].iloc[d * nb[per]:(d + 1) * nb[per]] for per in nb])
        r = M.book(smp)
        cs.append({"dev_Sharpe": st(r[r.index < SPLIT])["Sharpe"], "hold_Sharpe": st(r[r.index >= SPLIT])["Sharpe"],
                   "dev_CAGR": st(r[r.index < SPLIT])["CAGR"], "hold_CAGR": st(r[r.index >= SPLIT])["CAGR"]})
    print(pd.DataFrame(cs).describe().loc[["mean", "min", "max"]].round(2).to_string())

    # ── B. filters (terciles on dev) ─────────────────────────────────────────
    print("\nB1. FEATURE TERCILES on all contract wins (ab20 %, by DEV tercile cut points)")
    feats = ["rel_value", "rel_rev", "mcap", "price", "materiality", "novelty", "n_flags", "pre_mom20", "pre_mom60",
             "dist_hi252", "vol60", "pre_volsurge5", "gap_at_entry"]
    cats = [c for c in ("counterparty_type", "firmness", "counterparty_named", "revenue_within_12m", "repeat_customer",
                        "going_concern") if c in E.columns]
    rules = []
    dev = E[E.period == "dev"]
    for c in feats:
        q = dev[c].quantile([1 / 3, 2 / 3]).to_numpy()
        if not np.all(np.isfinite(q)) or q[0] == q[1]:
            continue
        tb = pd.cut(E[c], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
        row = []
        for per in ("dev", "holdout"):
            for lab in ("low", "mid", "high"):
                x = E[(E.period == per) & (tb == lab)].ab_20
                row.append(f"{per[:4]}-{lab} {x.mean() * 100:6.2f} (n{len(x.dropna())})")
        print(f"  {c:14s} " + " | ".join(row))
        dtb = tb[E.period == "dev"]
        for lab in ("low", "high"):
            worst = dev[dtb == lab].ab_20; rest = dev[dtb != lab].ab_20
            diff_t = (worst.mean() - rest.mean()) / np.sqrt(worst.var() / worst.count() + rest.var() / rest.count())
            if diff_t < -2:
                rules.append((c, lab, q, diff_t))
    for c in cats:
        g = E.groupby([c, "period"]).ab_20.agg(["mean", "count"]).unstack()
        print(f"  {c}:\n" + (g * [100, 100, 1, 1]).round(2).to_string())
        for lv, sub in dev.groupby(c):
            rest = dev[dev[c] != lv].ab_20
            if sub.ab_20.count() >= 20:
                dt_ = (sub.ab_20.mean() - rest.mean()) / np.sqrt(sub.ab_20.var() / sub.ab_20.count() + rest.var() / rest.count())
                if dt_ < -2:
                    rules.append((c, lv, None, dt_))
    print("\nB2. ADOPTED on DEV (worst group trails the rest at t < -2):", [(r[0], r[1], round(r[3], 2)) for r in rules])

    def apply_rules(X):
        keep = pd.Series(True, index=X.index)
        for c, lab, q, _ in rules:
            if q is None:
                keep &= X[c] != lab
            elif lab == "low":
                keep &= ~(X[c] <= q[0])
            else:
                keep &= ~(X[c] > q[1])
        return X[keep]

    variants = {"base (>=10% mcap)": Eb, "all contract wins": E,
                "all contract wins + adopted filters": apply_rules(E), "base + adopted filters": apply_rules(Eb)}
    print("\nB3. BOOKS (H20 equal weight)")
    for nm, X in variants.items():
        r = M.book(X)
        print(f"  {nm:40s} n {len(X):4d} | dev {fmt(st(r[r.index < SPLIT]))} | holdout {fmt(st(r[r.index >= SPLIT]))}")

    # ── C. exits / sizing / hedge, chosen on dev ──────────────────────────────
    print("\nC. EXITS / SIZING / HEDGE (on base and on the filtered base)")
    rows = []
    for nm in ("base (>=10% mcap)", "base + adopted filters", "all contract wins + adopted filters"):
        X = variants[nm]
        for H in (10, 20, 40):
            for kill in (None, 3, 5, 10):
                if kill is not None and kill >= H:
                    continue
                for trail in (None, 0.25):
                    for size in ("equal", "invvol"):
                        for cap in (1.0, 0.25):
                            for hedge in (0.0, 1.0):
                                r = M.book(X, H=H, cap=cap, kill_day=kill, trail=trail, size=size, hedge=hedge)
                                d, h = st(r[r.index < SPLIT]), st(r[r.index >= SPLIT])
                                rows.append(dict(set=nm, H=H, kill=kill, trail=trail, size=size, cap=cap, hedge=hedge,
                                                 **{f"dev_{k}": v for k, v in d.items()},
                                                 **{f"hold_{k}": v for k, v in h.items()}))
    res = pd.DataFrame(rows).sort_values("dev_Sharpe", ascending=False)
    res.to_csv(f"{OUT}/contract_filters_grid.csv", index=False)
    print(f"  {len(res)} designs; top 15 by DEV Sharpe:")
    print(res.head(15).round(2).to_string(index=False))
    best = res.iloc[0]
    X = variants[best.set]
    rb = M.book(X, H=int(best.H), cap=best.cap, kill_day=None if pd.isna(best.kill) else int(best.kill),
                trail=None if pd.isna(best.trail) else best.trail, size=best.size, hedge=best.hedge)
    pd.DataFrame({"best": rb, "base": r0}).to_csv(f"{OUT}/contract_filters_daily.csv")
    print("\n  median holdout Sharpe of the dev top-20:", round(res.head(20).hold_Sharpe.median(), 2),
          "| of all designs:", round(res.hold_Sharpe.median(), 2))
    # leverage on the chosen design (Kelly from dev)
    FIN = 0.06 / 252
    d = rb[rb.index < SPLIT]; ex = d - 0.04 / 252
    kst = ex.mean() / ex.var()
    print(f"\n  LEVERAGE of the dev-best design (full Kelly from dev = {kst:.2f}x):")
    for k in sorted({1.0, 1.5, 2.0, round(kst / 2, 2), round(kst, 2)}):
        lr = k * rb - max(k - 1, 0) * FIN
        sh = st(lr[lr.index >= SPLIT]); sd = st(lr[lr.index < SPLIT])
        wk = ((1 + sh["CAGR"] / 100) ** (1 / 52) - 1) * 100
        print(f"    {k:4.2f}x  dev CAGR {sd['CAGR']:7.1f} DD {sd['maxDD']:6.1f} | holdout CAGR {sh['CAGR']:7.1f} "
              f"Sharpe {sh['Sharpe']:.2f} DD {sh['maxDD']:6.1f} -> {wk:.2f}%/week")
    print(f"\n::notice::contract filters: base dev {fmt(st(r0[r0.index < SPLIT]))}; dev-best {best.set} H{best.H} "
          f"kill{best.kill} trail{best.trail} {best['size']} cap{best.cap} hedge{best.hedge}: dev Sharpe {best.dev_Sharpe:.2f} "
          f"holdout Sharpe {best.hold_Sharpe:.2f} CAGR {best.hold_CAGR:.0f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
