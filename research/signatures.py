"""
research/signatures.py -- step 2 of "event minus one": can conditions known at a
Friday close predict (a) a spike next week or (b) a run-rate quarter ahead?

Features (all known by the Friday close of week t):
  price/volume  r1w r4w r13w m12_1 vol12 dist-from-52w-high trend max-week(12)
                log price, log $ volume, volume surge vs 12-week average
  short volume  FINRA short-sale share of volume last week, change vs 12-week mean (2018-08+)
  fails         SEC fails-to-deliver $ in the last 45 days, only files already
                published (avail_date <= t), relative to $ volume
  attention     Wikipedia views last week vs 12-week median, log views
  filings       8-K count (4 wk); items 1.01 / 3.02 / 3.01|3.03|5.01 / 5.02 / 7.01|8.01 (4-8 wk);
                424B prospectus (8 wk); shelf S-3 on file (52 wk); 13D stake (8 wk);
                Form 4 count (4 wk); insider open-market purchases $ (4 wk);
                earnings filed in the last 13 weeks; earnings due next week (dates are
                normally announced in advance -- flagged as a mild look-ahead)
Targets:
  spike1   next week's return >= +25% above the equal-weight universe
  crash1   next week's return <= -25% below it
  run13    next 13 weeks >= +46.9% (the 3%/week run rate)
Model: LightGBM, trained on dev (2017-2023), scored ONCE on holdout (2024-2026).
Universe: price >= $2, >= $500k/day, >= 26 weeks listed, no funds.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
OUT = os.environ.get("OUT", "research/results")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
W, S = p.weeks, p.symbols
C, d1c, dv = p.df("close"), p.df("d1_close"), p.df("dollar_vol")
U = (f["price"] >= 2) & (f["adv"] >= 5e5) & (f["hist"] >= 26) & ~pd.Series(p.is_fund, index=S)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() < 5)
ew = r.where(U).mean(axis=1)
# memory: keep only symbols that are ever in the universe; store features as float32
# vectors over (week >= 2017, in-universe) rows as soon as each panel is built
K = U.columns[U.loc["2016-06-01":].any()]
MASK = (U & C.notna()).reindex(columns=K)
MASK.loc[MASK.index < "2017-01-06"] = False
MASK_ARR = MASK.to_numpy()
ROW_W, ROW_S = np.nonzero(MASK_ARR)


class Vec(dict):
    def __setitem__(self, k, v):
        a = v.reindex(index=W, columns=K).to_numpy(dtype="float32")[MASK_ARR]
        super().__setitem__(k, a)


def week_of(ts, cutoff_hours=8):
    """Friday label of the week an event becomes known (events after Fri 16:00 ET roll forward)."""
    sh = (pd.to_datetime(ts) + pd.Timedelta(hours=cutoff_hours)).dt.normalize()
    return sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")


def grid(df, col=None, how="count"):
    """(week, symbol[, value]) -> weeks x symbols panel."""
    d = df[df.symbol.isin(set(S)) & df.week.isin(set(W))]
    if how == "count":
        g = d.groupby(["week", "symbol"]).size()
    else:
        g = d.groupby(["week", "symbol"])[col].sum()
    return g.unstack().reindex(index=W, columns=K)


def roll(panel, n):
    return panel.fillna(0).rolling(n, min_periods=1).sum()


F = Vec()
F["r1w"], F["r4w"] = f["r1w"], f["r4w"]
F["r13w"] = C / C.shift(13) - 1
F["m12_1"], F["vol12"] = f["m12_1"], f["vol12"]
F["dist_hi52"] = 1 - f["hi52"]
F["trend"] = f["trend"]
F["max_wk12"] = r.rolling(12, min_periods=8).max()
F["log_price"] = np.log(f["price"])
F["log_adv"] = np.log(f["adv"].clip(lower=1))
F["vol_surge"] = dv / dv.rolling(12, min_periods=8).mean().shift(1)
F["ex_r1w"] = r.sub(ew, axis=0)

# short volume
sv = pd.concat(pd.read_parquet(x) for x in sorted(glob.glob(f"{P}/alt/shortvol_*.parquet")))
sv = sv[sv.symbol.isin(set(S))]
svr = (sv.assign(ratio=sv.short_volume / sv.total_volume.where(sv.total_volume > 0))
         .pivot_table(index="week", columns="symbol", values="ratio").reindex(index=W, columns=K))
F["short_ratio"] = svr
F["short_ratio_chg"] = svr - svr.rolling(12, min_periods=6).mean().shift(1)

# fails to deliver (only files already published)
ftd = pd.read_parquet(f"{P}/alt/ftd.parquet")
ftd = ftd[ftd.symbol.isin(set(S))].drop_duplicates(["settle_date", "symbol"])
ftd["usd"] = ftd.fails * ftd.price
ftd["week"] = week_of(ftd.avail_date, 0)
ftd_w = grid(ftd, "usd", "sum")
F["ftd_rel_adv"] = roll(ftd_w, 4) / (f["adv"] * 20)

# Wikipedia attention
wk = pd.concat(pd.read_parquet(x) for x in sorted(glob.glob(f"{P}/alt/wiki_*.parquet")))
wk["week"] = week_of(wk.date, 0)
wv = grid(wk, "views", "sum")
F["wiki_log"] = np.log1p(wv)
F["wiki_surge"] = wv / wv.rolling(12, min_periods=6).median().shift(1)

# filings
fi = pd.concat(pd.read_parquet(x, columns=["tickers", "form", "items", "accepted_et"])
               for x in sorted(glob.glob(f"{P}/sec/filings_*.parquet")))
fi["symbol"] = fi.tickers.str.split(",").str[0]
fi["week"] = week_of(fi.accepted_et)
k8 = fi[fi.form == "8-K"]
F["n8k_4w"] = roll(grid(k8), 4)
for name, items, n in (("i101_4w", ["1.01"], 4), ("i302_8w", ["3.02"], 8), ("distress_8w", ["3.01", "3.03", "5.01"], 8),
                       ("i502_4w", ["5.02"], 4), ("pr_4w", ["7.01", "8.01"], 4), ("i202_13w", ["2.02"], 13)):
    m = k8["items"].fillna("").apply(lambda s: any(i in s for i in items))
    F[name] = roll(grid(k8[m]), n)
F["prosp_8w"] = roll(grid(fi[fi.form.isin(["424B4", "424B5", "424B1", "424B3"])]), 8)
F["shelf_52w"] = roll(grid(fi[fi.form.isin(["S-3", "S-3ASR", "F-3"])]), 52)
F["d13_8w"] = roll(grid(fi[fi.form.str.startswith("SC 13D")]), 8)
F["form4_4w"] = roll(grid(fi[fi.form == "4"]), 4)
ins = pd.read_parquet(f"{P}/insider_buys.parquet", columns=["symbol", "filing_date", "value_usd"])
ins["week"] = week_of(ins.filing_date, 0)
F["insider_buy_4w"] = np.log1p(roll(grid(ins, "value_usd", "sum"), 4))
e = pd.read_parquet(f"{P}/earnings_events.parquet", columns=["symbol", "d0"])
e["week"] = week_of(e.d0, 0) - pd.Timedelta(days=7)        # flag sits on the week BEFORE the report
F["earn_next_wk"] = grid(e).fillna(0).clip(upper=1)

# targets
nxt = r.shift(-1).sub(ew.shift(-1), axis=0)
r13 = (C.shift(-13) / C - 1)
T = Vec()
_T = {"spike1": (nxt >= 0.25).astype(float).where(nxt.notna()),
     "crash1": (nxt <= -0.25).astype(float).where(nxt.notna()),
     "run13": (r13 >= 1.03 ** 13 - 1).astype(float).where(r13.notna())}
for k_, v_ in _T.items():
    T[k_] = v_
RET = Vec()
RET["next1_ex"] = nxt
RET["next13"] = r13
RET["next1_monclose"] = (C.shift(-1) / d1c.shift(-1) - 1).sub(ew.shift(-1), axis=0)


def long_table():
    df = pd.DataFrame({k: v for k, v in {**F, **T, **RET}.items()})
    df.insert(0, "symbol", K[ROW_S])
    df.insert(0, "week", W[ROW_W])
    return df


if __name__ == "__main__":
    df = long_table().replace([np.inf, -np.inf], np.nan)
    os.makedirs(OUT, exist_ok=True)
    cache = os.environ.get("CACHE", f"{OUT}/signatures_table.parquet")
    df.to_parquet(cache, index=False)
    print(f"rows {len(df):,}, weeks {df.week.nunique()}, features {len(F)} -> {cache}")
    print("next: python research/signatures_model.py")
