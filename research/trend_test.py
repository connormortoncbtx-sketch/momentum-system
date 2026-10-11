"""
research/trend_test.py -- time-series momentum (trend following), cross-asset ETFs and crypto.

ETF book (2016+; dev 2016-2022, holdout 2023-2026): 41 ETFs across stocks, bonds, credit,
commodities, currencies, sectors. Monthly at month-end close: each asset long if its
lookback return > 0, else short (or cash in the long-only version); size = 10% annual vol
target per asset from 60-day realised vol, capped at 2x; book scaled to equal risk.
Crypto book (2021+; dev 2021-2023, holdout 2024-2026): 16 coins, daily at the close: long
if price > its N-day average, else cash; vol target 40%/yr per coin, equal weight.
Costs: 5 bps per unit traded (ETFs), 25 bps (crypto).
"""
import os
import numpy as np, pandas as pd
D = os.environ.get("TREND", "research_data/trend")


def stats(r, ppy):
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod(); yrs = len(r) / ppy
    return {"CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "vol%": r.std() * np.sqrt(ppy) * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(ppy), "maxDD%": (eq / eq.cummax() - 1).min() * 100}


e = pd.read_parquet(f"{D}/etf_daily.parquet")
px = e.pivot_table(index=pd.to_datetime(e.timestamp).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None),
                   columns="symbol", values="close").sort_index()
ret = px.pct_change()
vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
me = px.resample("ME").last().index
me = [px.index[px.index <= m][-1] for m in me if (px.index <= m).any()]
rows = []
for lb in (63, 126, 252):
    for longonly in (False, True):
        w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        prev = pd.Series(0.0, index=px.columns)
        cost = pd.Series(0.0, index=px.index)
        for i, d in enumerate(me[:-1]):
            sig = np.sign(px.loc[d] / px.shift(lb).loc[d] - 1)
            if longonly:
                sig = sig.clip(lower=0)
            size = (0.10 / vol.loc[d]).clip(upper=2.0)
            tgt = (sig * size).fillna(0)
            n = (tgt != 0).sum()
            tgt = tgt / max(n, 1) * 4          # equal-risk book, ~ 4 assets' worth of 10% vol each
            nxt = me[i + 1]
            w.loc[(w.index > d) & (w.index <= nxt)] = tgt.values
            cost.loc[px.index[px.index > d][0]] = (tgt - prev).abs().sum() * 5 / 1e4
            prev = tgt
        r = (w * ret).sum(axis=1) - cost
        r = r[r.index >= "2016-07-01"]
        for per, sl in (("dev 2016-22", r.index < "2023-01-01"), ("holdout 2023-26", r.index >= "2023-01-01")):
            rows.append({"book": f"ETF trend {lb}d {'long-only' if longonly else 'long/short'}", "period": per, **stats(r[sl], 252)})
spy = ret["SPY"][ret.index >= "2016-07-01"]
for per, sl in (("dev 2016-22", spy.index < "2023-01-01"), ("holdout 2023-26", spy.index >= "2023-01-01")):
    rows.append({"book": "SPY buy & hold", "period": per, **stats(spy[sl], 252)})

c = pd.read_parquet(f"{D}/crypto_daily.parquet")
cp = c.pivot_table(index=pd.to_datetime(c.timestamp).dt.normalize().dt.tz_localize(None), columns="symbol", values="close").sort_index()
cr = cp.pct_change()
cv = cr.rolling(30, min_periods=20).std() * np.sqrt(365)
for n in (20, 50, 100):
    sig = (cp > cp.rolling(n, min_periods=n).mean()).astype(float).shift(1)      # decided at yesterday's close
    size = (0.40 / cv.shift(1)).clip(upper=1.5)
    w = (sig * size).where(cp.notna())
    w = w.div(w.notna().sum(axis=1).clip(lower=1), axis=0).fillna(0)
    cost = w.diff().abs().sum(axis=1) * 25 / 1e4
    r = (w * cr).sum(axis=1) - cost
    r = r[r.index >= "2021-06-01"]
    for per, sl in (("dev 2021-23", r.index < "2024-01-01"), ("holdout 2024-26", r.index >= "2024-01-01")):
        rows.append({"book": f"crypto trend {n}d MA, long/cash", "period": per, **stats(r[sl], 365)})
btc = cr["BTC/USD"][cr.index >= "2021-06-01"]
for per, sl in (("dev 2021-23", btc.index < "2024-01-01"), ("holdout 2024-26", btc.index >= "2024-01-01")):
    rows.append({"book": "BTC buy & hold", "period": per, **stats(btc[sl], 365)})
res = pd.DataFrame(rows).round(2)
pd.set_option("display.width", 200)
print(res.pivot_table(index="book", columns="period", values=["CAGR%", "Sharpe", "maxDD%"]).round(2).to_string())
