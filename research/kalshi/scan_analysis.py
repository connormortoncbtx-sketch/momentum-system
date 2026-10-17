"""
research/kalshi/scan_analysis.py -- analysis of the cached cross-market scan (kalshi/scan/s2_*.parquet), Oct 17 2026.

1. maker edge by category / fee type / series age, clustered by market
2. calibration: settled rate vs trade price by bucket
3. longshot fade: buy NO when YES trades at p, priced at the trades where the TAKER sold YES (the price a
   fader actually pays, 1 - YES bid), held to settlement, after the taker fee 0.07*p*(1-p); clustered by event
Reduces file by file (the pooled trades are ~12M rows and do not fit in memory at once).
"""
import glob, os
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SC = f"{RD}/kalshi/scan"
BINS = [0, .05, .10, .20, .35, .50, .65, .80, .90, .95, 1.0]


def reduce():
    mk, fd = [], []
    for f in glob.glob(f"{SC}/s2_*.parquet"):
        if os.path.getsize(f) < 2000:
            continue
        t = pd.read_parquet(f, columns=["ticker", "series", "window", "close_time", "first", "p", "c", "R", "edge", "taker_side"])
        if not len(t):
            continue
        t["ec"] = t.edge * t.c
        mk.append(t.groupby(["series", "window", "ticker"]).agg(ec=("ec", "sum"), c=("c", "sum"), close=("close_time", "first"),
                                                                 first=("first", "first")).reset_index())
        t["pb"] = pd.cut(t.p, BINS, labels=False, include_lowest=True)
        t["pc"], t["Rc"] = t.p * t.c, t.R * t.c
        t["fade"] = (t.p - t.R - 0.07 * t.p * (1 - t.p)) * t.c
        t["event"] = t.ticker.str.rsplit("-", n=1).str[0]
        t["yr"] = t.close_time.dt.year
        t["tno"] = t.taker_side == "no"
        fd.append(t.groupby(["series", "event", "yr", "pb", "tno"]).agg(c=("c", "sum"), pc=("pc", "sum"), Rc=("Rc", "sum"),
                                                                         fade=("fade", "sum")).reset_index())
    return pd.concat(mk, ignore_index=True), pd.concat(fd, ignore_index=True)


def clustered(g, val):
    f = g[val].sum() / g.c.sum()
    per, w = g[val] / g.c, g.c / g.c.sum()
    se = np.sqrt(((per - f) ** 2 * w ** 2).sum())
    return f * 100, (f / se if se else np.nan)


def main():
    S = pd.read_parquet(f"{SC}/series.parquet").rename(columns={"ticker": "series"})[["series", "category", "fee_type", "title"]]
    m, b = reduce()
    m = m.merge(S, on="series", how="left"); m["e"] = m.ec / m.c
    b = b.merge(S, on="series", how="left")
    pd.set_option("display.width", 250)

    def edge(g):
        E = g.ec.sum() / g.c.sum() * 100
        se = g.e.std(ddof=1) / np.sqrt(len(g)) * 100
        return pd.Series(dict(series=g.series.nunique(), markets=len(g), contracts_M=g.c.sum() / 1e6, edge_c=E, t=E / se))
    m["age"] = pd.cut((m.close - m["first"]).dt.days, [-1, 30, 90, 180, 365, 10_000], labels=["0-30d", "31-90d", "91-180d", "181-365d", ">1y"])
    for k in (["category", "window"], ["fee_type", "window"], ["age"]):
        print(f"\nMAKER EDGE BY {k}"); print(m.groupby(k, observed=True).apply(edge).round(2).to_string())
    e = b[b.tno].groupby(["event", "pb", "category", "yr"]).agg(c=("c", "sum"), pc=("pc", "sum"), Rc=("Rc", "sum"), fade=("fade", "sum")).reset_index()

    def fade(g):
        f, t = clustered(g, "fade"); p = g.pc.sum() / g.c.sum()
        return pd.Series(dict(events=len(g), contracts_M=g.c.sum() / 1e6, price=p * 100, settled=g.Rc.sum() / g.c.sum() * 100,
                              fade_c=f, t=t, ret_pct=f / (1 - p)))
    print("\nLONGSHOT FADE (taker-NO prices) BY YES BUCKET"); print(e[e.pb <= 4].groupby("pb").apply(fade).round(2).to_string())
    print("\nYES 5-35c BY CATEGORY"); print(e[e.pb.between(1, 3)].groupby("category").apply(fade).round(2).to_string())


if __name__ == "__main__":
    main()
