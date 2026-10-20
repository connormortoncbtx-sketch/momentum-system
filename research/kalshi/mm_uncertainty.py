"""
research/kalshi/mm_uncertainty.py -- do the bot's losing city-days come on uncertain-forecast days? (Oct 20, 2026)

Step 1 of "LLM forecast-confidence filter": before reading any forecaster text, test whether NUMERIC forecast
uncertainty known in advance predicts where the bot's deep fills lose.
Fills: the bot's simulated deep fills (profit_lock.fills: sweep >= 2c past the first price fills our 5-lot 2c behind,
6-48h before close, 25/market cap), held to settlement; edge in cents per contract.
Features per city-day, each used only on fills AFTER it was public:
  eve   (21:00 local the day before): |GFS MOS - NBM| disagreement
  morn  (09:00 local on the day):     |GFS MOS - NBM| disagreement, and |NBM morn - NBM eve| overnight shift
Diagnostic (hindsight, not tradeable): |actual high - NBM eve| -- the most a perfect uncertainty signal could reach.
Quintile cut-offs come from DEV (< 2025) city-days only; holdout = 2025-26. Rule tested: skip city-days in the
top dev quintile of a feature -> per-contract edge and total P&L kept, dev vs holdout.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import weather_edge as W
from profit_lock import fills
from mm_perfill import load

pd.set_option("display.width", 220)


def main():
    t = load()
    F = fills(t)
    m = W.market_intervals(pd.read_parquet(W.D / "markets.parquet"))
    ev = m.groupby("event_ticker").agg(station=("station", "first"), date=("date", "first")).reset_index()
    ev = ev.merge(W.realised(m), on="event_ticker")
    ev = W.forecasts(ev, W.mos_max(pd.read_parquet(W.D / "mos.parquet")))
    ev["high"] = np.where(np.isfinite(ev.rlo) & np.isfinite(ev.rhi), (ev.rlo + ev.rhi) / 2,
                          np.where(np.isfinite(ev.rlo), ev.rlo, ev.rhi))
    nb_eve = ev.f_eve_NBS.fillna(ev.f_eve_GFS); nb_morn = ev.f_morn_NBS.fillna(ev.f_morn_GFS)
    ev["dis_eve"] = (ev.f_eve_GFS - ev.f_eve_NBS).abs()
    ev["dis_morn"] = (ev.f_morn_GFS - ev.f_morn_NBS).abs()
    ev["shift"] = (nb_morn - nb_eve).abs()
    ev["bust"] = (ev.high - nb_eve).abs()                         # hindsight diagnostic
    F = F.merge(m[["ticker", "event_ticker"]], on="ticker").merge(
        ev[["event_ticker", "date", "t_eve", "t_morn", "dis_eve", "dis_morn", "shift", "bust"]], on="event_ticker")
    F["e"] = np.where(F.q > 0, F.R - F.px, F.px - F.R)            # per contract, held to settlement
    F["c"] = F.q.abs()
    F["period"] = np.where(F.date < W.SPLIT, "dev", "hold")
    print(f"fills {len(F):,} on {F.event_ticker.nunique():,} city-days; dev {(F.period=='dev').sum():,} hold {(F.period=='hold').sum():,}")

    rows, rule = [], []
    for feat, avail in (("dis_eve", "t_eve"), ("dis_morn", "t_morn"), ("shift", "t_morn"), ("bust", None)):
        x = F if avail is None else F[F.t >= F[avail]]
        x = x[x[feat].notna()]
        ed = ev[(ev.date < W.SPLIT) & ev[feat].notna()][feat]
        cuts = np.unique(ed.quantile([0, .2, .4, .6, .8, 1]).values)
        cuts[0], cuts[-1] = -np.inf, np.inf
        x = x.assign(qn=pd.cut(x[feat], cuts, labels=False, include_lowest=True, duplicates="drop"))
        for (per, qn), g in x.groupby(["period", "qn"]):
            pe = (g.e * g.c).groupby(g.event_ticker).sum() / g.c.groupby(g.event_ticker).sum()
            rows.append(dict(feature=feat, period=per, quintile=int(qn) + 1, city_days=g.event_ticker.nunique(),
                             contracts=g.c.sum(), edge_c=(g.e * g.c).sum() / g.c.sum() * 100,
                             per_cityday_c=pe.mean() * 100, t=pe.mean() / pe.std() * np.sqrt(len(pe)) if len(pe) > 2 else np.nan,
                             pnl_usd=(g.e * g.c).sum()))
        if avail is not None:
            top = x.qn == x.qn.max()
            for per, g in x.groupby("period"):
                keep = g[~top.loc[g.index]]
                rule.append(dict(feature=feat, period=per,
                                 edge_all_c=(g.e * g.c).sum() / g.c.sum() * 100,
                                 edge_kept_c=(keep.e * keep.c).sum() / keep.c.sum() * 100,
                                 contracts_kept_pct=keep.c.sum() / g.c.sum() * 100,
                                 pnl_kept_pct=(keep.e * keep.c).sum() / (g.e * g.c).sum() * 100))
    R = pd.DataFrame(rows)
    out = os.path.join(os.environ.get("RESEARCH_DATA", "research_data"), "results")
    os.makedirs(out, exist_ok=True)
    R.to_csv(f"{out}/kalshi_mm_uncertainty.csv", index=False)
    print("DEEP-FILL EDGE BY FORECAST-UNCERTAINTY QUINTILE (dev cut-offs; 5 = most uncertain)")
    print(R.pivot_table(index=["feature", "quintile"], columns="period",
                        values=["edge_c", "per_cityday_c", "t", "city_days"]).round(2).to_string())
    print("\nRULE: skip the most uncertain fifth of city-days")
    print(pd.DataFrame(rule).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
