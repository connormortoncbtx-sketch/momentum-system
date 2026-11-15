"""
research/novel_stock_signals.py -- two cheap stock-signal tests on the daily bars (Nov 15, 2026; Connor: "test everything").

PRE-REGISTERED (written before results):
 #4 SAME-MONTH SEASONALITY (Heston & Sadka 2008): score = a stock's mean return in the same calendar month over all
    earlier years available (>= 3; data start 2016). Book: top decile, equal weight, monthly, 15 bps per side.
    Pass: beats the equal-weight universe AND top-minus-bottom decile spread t > 2, in BOTH 2019-23 and 2024-26.
 #5 OVERNIGHT vs INTRADAY MOMENTUM (Lou, Polk & Skouras 2019): over months t-12..t-2, cumulative overnight return
    (open / previous close) and intraday return (close / open). Books: top decile by each, and by plain 12-1
    (benchmark), monthly. Pass: overnight or intraday book beats the 12-1 book's Sharpe in BOTH 2017-23 and 2024-26.
Universe at each month end: daily close >= $5, 20-day mean $ volume >= $2M, not a fund name, >= 13 months of bars.
Delisted stocks included (daily/delisted_ohlcv_*). Monthly returns are close-to-close with the last available close
if a stock stops trading (no delisting return -- a known optimistic bias, the same for every book).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
COST = 15e-4
pd.set_option("display.width", 220)


def load():
    files = sorted(glob.glob(f"{RD}/daily/*ohlcv_*.parquet"))
    D = pd.concat((pd.read_parquet(f, columns=["symbol", "date", "open", "close", "volume"]) for f in files), ignore_index=True)
    D["date"] = pd.to_datetime(D.date)
    if D.date.dt.tz is not None:
        D["date"] = D.date.dt.tz_convert("America/New_York").dt.tz_localize(None)
    D["date"] = D.date.dt.normalize()
    D = D.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    for c in ("open", "close", "volume"):
        D[c] = D[c].astype("float32")
    return D


def stats(r, ppy=12):
    r = r.dropna()
    if len(r) < 12:
        return dict(cagr=np.nan, sharpe=np.nan, n=len(r))
    return dict(cagr=((1 + r).prod() ** (ppy / len(r)) - 1) * 100, sharpe=r.mean() / r.std() * np.sqrt(ppy), n=len(r))


def book(score, univ, fwd, q=0.9):
    """Top-decile equal-weight monthly book with turnover cost; returns monthly net returns."""
    out, prev = {}, set()
    for m in score.index:
        s = score.loc[m].where(univ.loc[m]).dropna()
        if len(s) < 50:
            continue
        pick = set(s[s.rank(pct=True) >= q].index)
        r = fwd.loc[m].reindex(list(pick)).fillna(0).mean()
        turn = len(pick ^ prev) / max(len(pick), 1)
        out[m] = r - turn * COST
        prev = pick
    return pd.Series(out)


def spread(score, univ, fwd):
    out = {}
    for m in score.index:
        s = score.loc[m].where(univ.loc[m]).dropna()
        if len(s) < 50:
            continue
        rk = s.rank(pct=True)
        f = fwd.loc[m]
        out[m] = f.reindex(rk[rk >= 0.9].index).mean() - f.reindex(rk[rk <= 0.1].index).mean()
    return pd.Series(out)


def main():
    D = load()
    print(f"daily bars {len(D):,}, {D.symbol.nunique():,} symbols", flush=True)
    D["dv"] = D.close * D.volume
    g = D.groupby("symbol", sort=False)
    D["prev_close"] = g.close.shift(1)
    D["on"] = np.log(D.open / D.prev_close)                    # overnight log return
    D["id"] = np.log(D.close / D.open)                         # intraday log return
    D["dv20"] = g.dv.transform(lambda x: x.rolling(20, min_periods=15).mean())
    D["ym"] = D.date.dt.to_period("M")
    M = D.groupby(["symbol", "ym"]).agg(close=("close", "last"), dv20=("dv20", "last"),
                                        on=("on", "sum"), id=("id", "sum"), nd=("close", "size"))
    C = M.close.unstack(0).sort_index()
    ret = C.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    names = pd.read_parquet(f"{RD}/assets.parquet").set_index("symbol")["name"].fillna("").astype(str)
    fund = pd.Series({s: B.is_fund_name(names.get(s, "")) for s in C.columns})
    univ = (C >= 5) & (M.dv20.unstack(0).reindex_like(C) >= 2e6) & (C.notna().cumsum() >= 13) & ~fund.reindex(C.columns).fillna(False)
    ew = fwd.where(univ).mean(axis=1)
    rows = []

    # ---- #4 seasonality: mean same-month return in earlier years (>= 3)
    seas = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for mo in range(1, 13):
        idx = [p for p in ret.index if p.month == mo]
        R = ret.loc[idx]
        past = R.shift(1).expanding().mean().where(R.shift(1).notna().cumsum() >= 3)   # earlier years only
        # score is formed at the end of the PREVIOUS month for the coming month `mo`
        prev_idx = [p - 1 for p in idx]
        seas.loc[[p for p in prev_idx if p in seas.index]] = past.set_axis(prev_idx).loc[[p for p in prev_idx if p in seas.index]].to_numpy()
    b4, s4 = book(seas, univ, fwd), spread(seas, univ, fwd)
    for lab, a, z in (("2019-23", "2019-01", "2023-12"), ("2024-26", "2024-01", "2026-12")):
        x, sp, e = b4.loc[a:z], s4.loc[a:z], ew.reindex(b4.index).loc[a:z]
        rows.append(dict(test="#4 seasonality top decile", period=lab, **stats(x), ew_cagr=stats(e)["cagr"], ew_sharpe=stats(e)["sharpe"],
                         spread_bp=sp.mean() * 1e4, spread_t=sp.mean() / sp.std() * np.sqrt(len(sp))))

    # ---- #5 overnight vs intraday momentum, months t-12..t-2
    ON, ID = M.on.unstack(0).reindex_like(C), M.id.unstack(0).reindex_like(C)
    on12 = ON.shift(1).rolling(11, min_periods=10).sum()
    id12 = ID.shift(1).rolling(11, min_periods=10).sum()
    m12 = C.shift(1) / C.shift(12) - 1
    corr = pd.concat([on12.stack(), id12.stack(), m12.stack()], axis=1, keys=["on", "id", "m12"]).dropna()
    print("cross-sectional pooled corr overnight/intraday/12-1:\n", corr.sample(min(len(corr), 500000), random_state=0).corr(method="spearman").round(2))
    books5 = {"12-1 total": m12, "12-1 overnight": on12, "12-1 intraday": id12}
    for name, sc in books5.items():
        bk, sp = book(sc, univ, fwd), spread(sc, univ, fwd)
        for lab, a, z in (("2017-23", "2017-02", "2023-12"), ("2024-26", "2024-01", "2026-12")):
            x, s_, e = bk.loc[a:z], sp.loc[a:z], ew.reindex(bk.index).loc[a:z]
            rows.append(dict(test=f"#5 {name} top decile", period=lab, **stats(x), ew_cagr=stats(e)["cagr"], ew_sharpe=stats(e)["sharpe"],
                             spread_bp=s_.mean() * 1e4, spread_t=s_.mean() / s_.std() * np.sqrt(len(s_))))
        if name != "12-1 total":
            bk.rename(name).to_csv(f"{RD}/results/novel_{name.replace(' ', '_').replace('-', '')}_monthly.csv")
    b4.rename("seasonality").to_csv(f"{RD}/results/novel_seasonality_monthly.csv")
    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/novel_stock_signals.csv", index=False)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
