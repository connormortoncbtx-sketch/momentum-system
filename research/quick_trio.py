"""
research/quick_trio.py -- three quick checks (Nov 15, 2026; Connor):
 1. CONDOR VARIANTS: are other entry times / structures independent of the 2pm SPY+QQQ condor (the one independent
    edge)? From results/options_series_daily.parquet (quotes source, stop at 2x credit): daily return on risk, Sharpe,
    and correlation with the 2pm condor, 2018-2023 and 2024-2026.
 2. CLASSIC TECHNICAL ANALYSIS on SPY / QQQ / IWM daily (2016-2026; close-only data): 200-day moving average filter,
    50/200 golden cross, RSI(2) < 10 buy-the-dip (Connors), MACD(12,26,9) above signal, 20-day Bollinger lower-band
    buys; each long-or-cash (cash earns 0), next-day execution, 1 bp per switch, vs buy-and-hold.
 3. THREE-SECTOR TILT: XLK + XLY + XLV equal weight, monthly rebalance, vs SPY; plus each sector alone.
Splits 2016/18-2023 and 2024-2026. Public data only.
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)


def sh(r, ppy=252):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(ppy) if len(r) > 20 and r.std() > 0 else np.nan


def cagr(r, ppy=252):
    r = r.dropna()
    return ((1 + r).prod() ** (ppy / len(r)) - 1) * 100 if len(r) > 20 else np.nan


def mdd(r):
    eq = (1 + r.dropna()).cumprod()
    return (eq / eq.cummax() - 1).min() * 100


def condors():
    o = pd.read_parquet(f"{RD}/results/options_series_daily.parquet")
    o = o[(o.src == "quotes") & (o.exit == "sl2x") & (o.wing == 0)]
    o["d"] = pd.to_datetime(o.d)
    base = o[(o.entry == "14:00") & (o.struct == "condor") & (o.k == 1.0)].groupby("d").ror.mean()
    rows = []
    for (entry, struct, k), g in o.groupby(["entry", "struct", "k"], observed=True):
        s = g.groupby("d").ror.mean()
        j = pd.concat([s, base], axis=1, keys=["v", "b"]).dropna()
        for lab, sel in (("2018-23", j.index < SPLIT), ("2024-26", j.index >= SPLIT)):
            x = j[sel]
            rows.append(dict(entry=entry, struct=struct, k=k, period=lab, days=len(x), mean_ror_pct=x.v.mean() * 100,
                             Sharpe=sh(x.v), corr_with_2pm=x.v.corr(x.b)))
    R = pd.DataFrame(rows)
    piv = R.pivot_table(index=["entry", "struct", "k"], columns="period", values=["mean_ror_pct", "Sharpe", "corr_with_2pm"]).round(2)
    print("1. CONDOR VARIANTS vs the 2pm SPY+QQQ condor (k=1):\n" + piv.to_string())
    # independent of the 2pm one AND positive in both periods
    ok = R.groupby(["entry", "struct", "k"]).apply(lambda g: (g.Sharpe > 0.5).all() and (g.corr_with_2pm < 0.5).all())
    print("\n   positive in both periods (Sharpe > 0.5) AND correlation with 2pm < 0.5 in both: "
          + (", ".join(f"{e} {s} k{k}" for (e, s, k), v in ok.items() if v) or "none"))


def ta():
    e = pd.read_parquet(f"{RD}/etf/etf_daily.parquet"); e["date"] = pd.to_datetime(e.date)
    px = e.pivot_table(index="date", columns="symbol", values="close").sort_index()
    rows = []
    for s in ("SPY", "QQQ", "IWM"):
        c = px[s].dropna(); r = c.pct_change()
        d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / 2).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 2).mean()
        rsi2 = 100 - 100 / (1 + up / dn)
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        bb_lo = c.rolling(20).mean() - 2 * c.rolling(20).std()
        rules = {
            "buy & hold": pd.Series(1.0, index=c.index),
            "above 200-day MA": (c > c.rolling(200).mean()).astype(float),
            "golden cross 50>200": (c.rolling(50).mean() > c.rolling(200).mean()).astype(float),
            "RSI(2)<10 dip, exit RSI>70": None,
            "MACD above signal": (macd > macd.ewm(span=9).mean()).astype(float),
            "below lower Bollinger, hold 5d": (c < bb_lo).astype(float).rolling(5, min_periods=1).max(),
        }
        pos = pd.Series(0.0, index=c.index); inpos = False
        for t in c.index:
            v = rsi2.get(t, np.nan)
            if not inpos and v < 10: inpos = True
            elif inpos and v > 70: inpos = False
            pos[t] = float(inpos)
        rules["RSI(2)<10 dip, exit RSI>70"] = pos
        for name, sig in rules.items():
            p = sig.shift(1).fillna(0)
            net = p * r - p.diff().abs().fillna(0) * 1e-4
            for lab, sel in (("2016-23", net.index < SPLIT), ("2024-26", net.index >= SPLIT)):
                x = net[sel]
                rows.append(dict(etf=s, rule=name, period=lab, in_market=p[sel].mean() * 100, CAGR=cagr(x), Sharpe=sh(x), maxDD=mdd(x)))
    R = pd.DataFrame(rows)
    print("\n2. CLASSIC TECHNICAL ANALYSIS (long or cash, next-day execution):")
    print(R.pivot_table(index=["etf", "rule"], columns="period", values=["CAGR", "Sharpe", "maxDD", "in_market"]).round(2).to_string())


def sectors():
    e = pd.read_parquet(f"{RD}/etf/etf_daily.parquet"); e["date"] = pd.to_datetime(e.date)
    px = e.pivot_table(index="date", columns="symbol", values="close").sort_index()[["SPY", "XLK", "XLY", "XLV"]].dropna()
    r = px.pct_change().dropna()
    m = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    books = {"SPY": m.SPY, "XLK": m.XLK, "XLY": m.XLY, "XLV": m.XLV, "XLK+XLY+XLV (equal, monthly)": m[["XLK", "XLY", "XLV"]].mean(axis=1)}
    rows = []
    for k, v in books.items():
        for lab, sel in (("2016-23", v.index < SPLIT), ("2024-26", v.index >= SPLIT)):
            x = v[sel]
            rows.append(dict(book=k, period=lab, CAGR=cagr(x, 12), Sharpe=sh(x, 12), maxDD=mdd(x)))
    R = pd.DataFrame(rows)
    print("\n3. THREE-SECTOR TILT vs SPY (monthly):")
    print(R.pivot_table(index="book", columns="period", values=["CAGR", "Sharpe", "maxDD"]).round(2).to_string())
    yr = m.groupby(m.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    yr["3-sector"] = m[["XLK", "XLY", "XLV"]].mean(axis=1).groupby(m.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    print("\n   by year %:\n" + yr.round(1).to_string())


if __name__ == "__main__":
    condors(); ta(); sectors()
