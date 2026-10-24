"""
research/options_theta_full.py -- same-day option premium, thorough version (Oct 23, 2026)

Data (ThetaData Standard, personal licence): bid/ask QUOTES every 15 minutes 09:45-16:00 (part of the history was pulled at 5 minutes; all analysed on the 15-minute grid) for every same-day SPY / QQQ
expiry Jan 2016 - Oct 2026, strikes within 6% of the 10:00 price. RAW DATA NEVER LEAVES THE RUNNER: it lives in
RAW_DIR (outside research_data, persisted only in the repo's private Actions cache); only aggregate statistics are
written to research_data/results (and so to the public branch).

Pre-registered grid (7,560 variants), each scored per trade as return on risk (P&L / (width - credit)):
  entry      09:45 10:00 11:00 12:00 13:00 14:00 15:00 ET (underlying = the 15-minute bar open at that time)
  structure  iron condor / put credit spread / call credit spread
  k          short strike(s) k implied moves out, k = 0.5 0.75 1.0 1.5 2.0 (implied move = ATM straddle mid at entry)
  wings      SPY $2 / $5 / $10; QQQ 0.4% / 0.8% / 1.6% of the price (min $1)
  fill       mid / near-mid (mid -/+ 1c per leg) / natural (sell at bid, buy at ask) -- entry AND any early exit
  exit       hold to the close / take profit at 50% of the credit / stop when the loss reaches 2x the credit /
             both -- checked on every 15-minute mid mark after entry, executed at the same fill model
Dev = 2016-2023 (pre-2022 and 2022-23 shown separately), holdout = 2024 - Oct 2026, scored once. Selection: rank by
dev t-stat among variants positive in BOTH dev eras; report the dev top 20 with their holdout, the holdout median of
the dev top 20, and how the holdout ranks the dev winner. All variants are written out (no hidden search).
"""
import datetime as dt, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import options_theta as OT

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/results"
RAW = os.environ.get("RAW_DIR", "/tmp/theta_raw")
T_START = time.time()
QIVL = int(os.environ.get("QIVL", "900000"))   # 15-minute quotes (the first pass used 5-minute; the backtest uses the 15-minute grid)
BUDGET_S = float(os.environ.get("TIME_BUDGET_MIN", "300")) * 60   # stop pulling in time for the cache to be saved
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
START, END = dt.date(2016, 1, 1), dt.date(2026, 10, 22)
ENTRIES = ["09:45", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]
KS = [0.5, 0.75, 1.0, 1.5, 2.0]
STRUCTS = ["condor", "put", "call"]
FILLS = ["mid", "nearmid", "natural"]
EXITS = ["hold", "tp50", "sl2x", "tp50+sl2x"]
note, warn = OT.note, OT.warn
CAT_E, CAT_S = pd.CategoricalDtype(ENTRIES), pd.CategoricalDtype(STRUCTS)
CAT_F, CAT_X = pd.CategoricalDtype(FILLS), pd.CategoricalDtype(EXITS)


def widths(und, S):
    return [2, 5, 10] if und == "SPY" else [max(1, round(f * S)) for f in (0.004, 0.008, 0.016)]


# ------------------------------------------------------------------ pull (quotes, 5-minute)
def quotes_day(und, d, s0):
    ds = d.strftime("%Y%m%d")
    rows = OT.get_v2("/v2/bulk_hist/option/quote", root=und, exp=ds, start_date=ds, end_date=ds, ivl=QIVL,
                     start_time=35100000, end_time=57600000, use_csv="false")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    low = {c.lower(): c for c in df.columns}
    q = pd.DataFrame({"strike": pd.to_numeric(df[low["strike"]], errors="coerce"),
                      "right": df[low["right"]].astype(str).str[0].str.upper(),
                      "ms": pd.to_numeric(df[low["ms_of_day"]], errors="coerce"),
                      "bid": pd.to_numeric(df[low["bid"]], errors="coerce"), "ask": pd.to_numeric(df[low["ask"]], errors="coerce")})
    if q.strike.median() > 5000:
        q["strike"] /= 1000
    q = q[(q.strike >= s0 * 0.94) & (q.strike <= s0 * 1.06)]
    return q.assign(d=pd.Timestamp(d)).astype({"strike": "float32", "bid": "float32", "ask": "float32", "ms": "int32"})


def pull(und):
    path = f"{RAW}/{und.lower()}_q5.parquet"
    have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["d"])
    done = set(pd.to_datetime(have.d).dt.date) if len(have) else set()
    ep = f"{RAW}/{und.lower()}_empty.json"                       # days the API returned nothing for: don't re-pull
    empty = set(dt.date.fromisoformat(x) for x in json.load(open(ep))) if os.path.exists(ep) else set()
    done |= empty
    exps = OT.expirations(und)
    s0, _ = OT.spot(und)
    days = sorted(d for d in exps if START <= d <= END and d in s0.index and d not in done)
    if os.environ.get("LIMIT"):
        days = days[:int(os.environ["LIMIT"])]
    note(f"{und}: {len([d for d in exps if START <= d <= END])} same-day expiries, {len(done)} cached, {len(days)} to pull")
    parts, t0, bad = [], time.time(), 0

    def one(d):
        try:
            return quotes_day(und, d, float(s0[d]))
        except Exception as ex:
            return ex
    with ThreadPoolExecutor(2) as ex:
        for c0 in range(0, len(days), 50):                       # chunks, so the time budget can stop cleanly
            if time.time() - T_START > BUDGET_S:
                warn(f"{und}: time budget reached after {c0} of {len(days)} days -- partial pull cached; re-run to resume")
                raise SystemExit(0)
            chunk = days[c0:c0 + 50]
            for d, r in zip(chunk, ex.map(one, chunk)):
                if isinstance(r, Exception):
                    bad += 1
                    if bad <= 5:
                        warn(f"{und} {d}: {str(r)[:150]}")
                elif r is not None and len(r):
                    parts.append(r)
                else:
                    empty.add(d)
            if parts:
                have = pd.concat([have] + parts, ignore_index=True); parts = []
                have.to_parquet(path, index=False)
            json.dump(sorted(x.isoformat() for x in empty), open(ep, "w"))
            print(f"  {und} {c0 + len(chunk)}/{len(days)} days ({time.time() - t0:.0f}s)", flush=True)
    if bad:
        warn(f"{und}: {bad} days failed")
    note(f"{und}: {len(empty)} expiry days returned no quotes from the API")
    return have


# ------------------------------------------------------------------ backtest
def ms_of(hhmm):
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m) * 60000


def day_trades(und, day, q, s_at, close):
    """All variants for one day (vectorised over the 5-minute marks). Rows: (entry, struct, k, wing, fill, exit, ror)."""
    out = []
    q = q[(q.ask > 0) & (q.ask >= q.bid) & (q.bid >= 0) & (q.ms % 900000 == 0)]   # one consistent 15-minute grid
    if q.empty:
        return out
    times = np.array(sorted(q.ms.unique()))
    key = list(zip(np.round(q.strike.astype(float), 3), q.right))
    cols = {k_: i for i, k_ in enumerate(sorted(set(key)))}
    ti = {t: i for i, t in enumerate(times)}
    B = np.full((len(times), len(cols)), np.nan); A = B.copy()
    r_idx = np.array([ti[t] for t in q.ms]); c_idx = np.array([cols[k_] for k_ in key])
    B[r_idx, c_idx] = q.bid.values; A[r_idx, c_idx] = q.ask.values
    Mi = (A + B) / 2
    for et in ENTRIES:
        ems = ms_of(et)
        if ems not in ti or et not in s_at:
            continue
        e = ti[ems]
        S = s_at[et]; atm = round(S)
        ic, ip = cols.get((float(atm), "C")), cols.get((float(atm), "P"))
        if ic is None or ip is None:
            continue
        M = Mi[e, ic] + Mi[e, ip]
        if not M > 0:
            continue
        for wi, W in enumerate(widths(und, S)):
            for kk in KS:
                ps, cs = round(S - kk * M), round(S + kk * M)
                legs = {"put": [(ps, "P", -1), (ps - W, "P", 1)], "call": [(cs, "C", -1), (cs + W, "C", 1)]}
                legs["condor"] = legs["put"] + legs["call"]
                for st in STRUCTS:
                    L = legs[st]
                    idx = [cols.get((float(k_), r_)) for k_, r_, _ in L]
                    if any(i is None for i in idx):
                        continue
                    sg = np.array([s for _, _, s in L], dtype=float)
                    bb, aa, mm = B[:, idx], A[:, idx], Mi[:, idx]
                    if np.isnan(mm[e]).any() or (bb[e][sg < 0] <= 0).any():
                        continue
                    cr = {"mid": float(-(sg * mm[e]).sum()),
                          "natural": float(np.where(sg < 0, bb[e], -aa[e]).sum())}
                    cr["nearmid"] = cr["mid"] - 0.01 * len(L)
                    vmid = -(mm[e + 1:] * sg).sum(axis=1)                     # cost to close at mid, later marks
                    vnat = np.where(sg < 0, aa[e + 1:], -bb[e + 1:]).sum(axis=1)
                    valid = ~np.isnan(mm[e + 1:]).any(axis=1)
                    loss_T = sum(-s * (max(close - k_, 0) if r_ == "C" else max(k_ - close, 0)) for k_, r_, s in L)
                    for fill in FILLS:
                        c0 = cr[fill]
                        if c0 <= 0.02 or c0 >= 0.6 * W:
                            continue
                        risk = W - c0
                        closev = {"mid": vmid, "natural": vnat, "nearmid": vmid + 0.01 * len(L)}[fill]
                        tp = valid & (vmid <= 0.5 * c0); sl = valid & (vmid >= 3.0 * c0)
                        for ex, cond in (("hold", None), ("tp50", tp), ("sl2x", sl), ("tp50+sl2x", tp | sl)):
                            if cond is not None and cond.any():
                                pnl = c0 - float(closev[int(np.argmax(cond))])
                            else:
                                pnl = c0 - loss_T
                            out.append((et, st, kk, wi, fill, ex, max(pnl / risk, -1.0)))
    return out


def backtest(und, q):
    b = pd.read_parquet(f"{OUT}/{und.lower()}_15m_raw_2016.parquet")
    b["d"] = pd.to_datetime(b.ts.dt.date); b["tm"] = b.ts.dt.strftime("%H:%M")
    opens = b.pivot_table(index="d", columns="tm", values="open")
    closes = b.groupby("d").close.last()
    rows = []
    t0 = time.time()
    for i, (d, g) in enumerate(q.groupby("d")):
        if d not in opens.index or d not in closes.index:
            continue
        s_at = {et: float(opens.at[d, et]) for et in ENTRIES if et in opens.columns and opens.at[d, et] == opens.at[d, et]}
        r = day_trades(und, d, g, s_at, float(closes[d]))
        if r:
            df = pd.DataFrame(r, columns=["entry", "struct", "k", "wing", "fill", "exit", "ror"])
            rows.append(df.astype({"entry": CAT_E, "struct": CAT_S, "k": "float32", "wing": "int8", "fill": CAT_F,
                                   "exit": CAT_X, "ror": "float32"}).assign(d=d))
        if (i + 1) % 300 == 0:
            print(f"  backtest {und} {i + 1} days ({time.time() - t0:.0f}s)", flush=True)
    if not rows:
        return pd.DataFrame(columns=["d", "entry", "struct", "k", "wing", "fill", "exit", "ror", "und"])
    return pd.concat(rows, ignore_index=True).assign(und=und)


def score(T):
    T["era"] = np.select([T.d < "2022-01-01", T.d < "2024-01-01"], ["dev_pre22", "dev_22_23"], "holdout")
    keys = ["und", "entry", "struct", "k", "wing", "fill", "exit"]
    T = T.astype({c: str for c in ("entry", "struct", "fill", "exit")})
    g = T.groupby(keys + ["era"]).ror.agg(["size", "mean", "std"]).reset_index()
    g["t"] = g["mean"] / g["std"] * np.sqrt(g["size"])
    W_ = g.pivot_table(index=keys, columns="era", values=["size", "mean", "t"], dropna=False)
    W_.columns = [f"{a}_{b}" for a, b in W_.columns]
    dev = T[T.era != "holdout"].groupby(keys).ror.agg(["size", "mean", "std"]).reindex(W_.index)
    W_["n_dev"], W_["mean_dev"] = dev["size"], dev["mean"]
    W_["t_dev"] = dev["mean"] / dev["std"] * np.sqrt(dev["size"])
    return W_.reset_index()


def coverage():
    """Which expiry days have no quotes, and are they different days (bias check: big-move days missing?)."""
    for und in ("SPY", "QQQ"):
        path = f"{RAW}/{und.lower()}_q5.parquet"
        q = pd.read_parquet(path, columns=["d"]) if os.path.exists(path) else pd.DataFrame(columns=["d"])
        qd = set(pd.to_datetime(q.d).dt.date)
        exps = [d for d in OT.expirations(und) if START <= d <= END]
        miss = sorted(d for d in exps if d not in qd)
        b = pd.read_parquet(f"{OUT}/{und.lower()}_15m_raw_2016.parquet")
        b["d"] = b.ts.dt.date
        g = b.groupby("d").agg(o=("open", "first"), c=("close", "last"), h=("close", "max"), l=("close", "min"))
        mv = ((g.c / g.o - 1).abs() * 100).rename("absret"); rg = ((g.h - g.l) / g.o * 100).rename("range")
        by = {}
        for d in exps:
            r = by.setdefault(d.year, [0, 0]); r[0] += 1; r[1] += d in qd
        wd = pd.Series([d.weekday() for d in miss]).value_counts().sort_index().to_dict() if miss else {}
        pm = [d for d in exps if d in qd and d in mv.index]; mm = [d for d in miss if d in mv.index]
        note(f"{und} coverage: {len(qd & set(exps))}/{len(exps)} expiry days have quotes; by year "
             + ", ".join(f"{y} {b_}/{a_}" for y, (a_, b_) in sorted(by.items()))
             + f"; missing by weekday {wd}; |open-close| % present {mv[pm].mean():.2f} vs missing {mv[mm].mean() if mm else float('nan'):.2f}; "
             f"range % present {rg[pm].mean():.2f} vs missing {rg[mm].mean() if mm else float('nan'):.2f}; first missing {[str(x) for x in miss[:12]]}")
        mo = pd.Series([f"{d:%Y-%m}" for d in miss if d.year in (2021, 2022, 2025)]).value_counts().sort_index()
        tot = pd.Series([f"{d:%Y-%m}" for d in exps if d.year in (2021, 2022, 2025)]).value_counts().sort_index()
        note(f"{und} missing by month (missing/expiries): " + ", ".join(f"{m} {mo.get(m, 0)}/{t}" for m, t in tot.items()))
        apr = [d for d in exps if dt.date(2025, 4, 1) <= d <= dt.date(2025, 4, 15)]
        note(f"{und} April 1-15 2025 expiries with quotes: {[str(d) for d in apr if d in qd]} / missing {[str(d) for d in apr if d not in qd]}")
        for d in [x for x in miss if x.year == 2025][:2]:
            ds = d.strftime("%Y%m%d")
            for ep_ in ("/v2/bulk_hist/option/ohlc", "/v2/bulk_hist/option/eod"):
                try:
                    rows = OT.get_v2(ep_, root=und, exp=ds, start_date=ds, end_date=ds, ivl=QIVL, use_csv="false")
                    note(f"{und} {d} {ep_}: {len(rows) if rows else 0} rows")
                except Exception as e:
                    note(f"{und} {d} {ep_} error: {str(e)[:200]}")
        for d in miss[:0]:
            ds = d.strftime("%Y%m%d")
            try:
                rows = OT.get_v2("/v2/bulk_hist/option/quote", root=und, exp=ds, start_date=ds, end_date=ds, ivl=QIVL,
                                 start_time=35100000, end_time=57600000, use_csv="false")
                note(f"{und} {d} re-pull: {len(rows) if rows else 0} rows; sample {str(rows[:1])[:200] if rows else None}")
            except Exception as e:
                note(f"{und} {d} re-pull error: {str(e)[:200]}")


def main():
    if os.environ.get("MODE") == "probe":
        d = dt.date(2018, 2, 9); s0, _ = OT.spot("SPY")
        q = quotes_day("SPY", d, float(s0[d]))
        note(f"probe quotes {d}: rows {0 if q is None else len(q)}, contracts {0 if q is None else q.groupby(['strike','right']).ngroups}, "
             f"times {0 if q is None else q.ms.nunique()}, sample {None if q is None else q.head(3).to_dict('records')}")
        return
    if os.environ.get("MODE") == "coverage":
        coverage(); return
    Ts = []
    for und in ("SPY", "QQQ"):
        q = pull(und)
        if os.environ.get("PULL_ONLY") == "1":
            continue
        T = backtest(und, q)
        note(f"{und}: {len(T):,} variant-trades over {T.d.nunique() if len(T) else 0} days")
        exps = [d for d in OT.expirations(und) if START <= d <= END]
        qd = set(pd.to_datetime(q.d).dt.date) if len(q) else set()
        td = set(pd.to_datetime(T.d).dt.date) if len(T) else set()
        cov = {y: (sum(d.year == y for d in exps), sum(d.year == y for d in qd), sum(d.year == y for d in td))
               for y in sorted({d.year for d in exps})}
        note(f"{und} coverage by year (expiries / days with quotes / days with trades): "
             + ", ".join(f"{y} {a}/{b}/{c}" for y, (a, b, c) in cov.items()))
        Ts.append(T)
    if not Ts:
        return
    T = pd.concat(Ts, ignore_index=True)
    S = score(T)
    S.to_csv(f"{OUT}/options_full_grid.csv", index=False)                   # aggregate statistics only
    pd.set_option("display.width", 250)
    for und in ("SPY", "QQQ"):
        x = S[(S.und == und)].copy()
        ok = x[(x.mean_dev_pre22 > 0) & (x.mean_dev_22_23 > 0) & (x.n_dev >= 150)].sort_values("t_dev", ascending=False)
        top = ok.head(20)
        cols = ["entry", "struct", "k", "wing", "fill", "exit", "n_dev", "mean_dev", "t_dev", "mean_dev_pre22", "mean_dev_22_23",
                "size_holdout", "mean_holdout", "t_holdout"]
        print(f"\n{und}: {len(x)} variants, {len(ok)} positive in both dev eras. DEV TOP 20 (mean = return on risk per trade)")
        print(top[cols].round(3).to_string(index=False))
        nat = ok[ok.fill == "natural"].head(10)
        print(f"\n{und}: best NATURAL-fill variants (realistic)\n" + nat[cols].round(3).to_string(index=False))
        if len(top):
            w = top.iloc[0]
            hr = (x.mean_holdout.rank(ascending=False)[top.index[0]], len(x))
            note(f"{und} options grid: {len(ok)}/{len(x)} variants positive in both dev eras; dev winner {w.entry} {w.struct} k{w.k} "
                 f"wing{int(w.wing)} {w.fill} {w.exit}: dev {w.mean_dev * 100:+.1f}% t{w.t_dev:.1f} -> holdout {w.mean_holdout * 100:+.1f}% "
                 f"t{w.t_holdout:.1f} (holdout rank {hr[0]:.0f}/{hr[1]}); dev top-20 holdout median {top.mean_holdout.median() * 100:+.1f}%; "
                 f"best natural-fill dev {nat.iloc[0].mean_dev * 100 if len(nat) else float('nan'):+.1f}% -> holdout "
                 f"{nat.iloc[0].mean_holdout * 100 if len(nat) else float('nan'):+.1f}%")
    # marginal effects: average over the grid of each design choice (dev / holdout), natural fill
    for dim in ("entry", "exit", "struct", "k", "fill"):
        m = S.groupby(["und", dim])[["mean_dev", "mean_holdout"]].median().unstack("und") * 100
        print(f"\nMEDIAN RoR % by {dim}\n" + m.round(2).to_string())
        m.to_csv(f"{OUT}/options_full_by_{dim}.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); raise
