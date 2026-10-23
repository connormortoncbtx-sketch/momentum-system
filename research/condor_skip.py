"""
research/condor_skip.py -- can the big losing condor days be seen coming and skipped? (Oct 23, 2026)

Same-day iron condor (k = 1 implied move; SPY $5 / QQQ 0.8% wings; $0.01 and $0.03 per leg) from
research/options_premium*.py. Max-loss days are rare (~3%) but carry the result. Candidate warnings known before the
10:00 ET entry:
  vix          VIX prior close (the 'VIX <= 25' filter was already tested: no help)
  vix_chg      VIX 1-day % change
  vix_vs20     VIX / its 20-day mean
  event_prem   today's implied move / the VIX-implied daily move -- the option market pricing a scheduled event today
  prev_ratio   yesterday's realised |move| / yesterday's implied move (volatility clustering)
  ratio5       mean of that ratio over the last 5 days
  fomc         Fed decision day
  dow          day of week
Per feature: dev = first half (to 2025-06-30) quintiles and the worst-quintile skip rule picked there; holdout =
second half. Reported: mean return on risk per quintile (dev), max-loss share per quintile, and the holdout effect
of skipping the dev-worst quintile (mean RoR kept vs all, share of total P&L kept, max-loss days avoided).
"""
import os, sys
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2025-07-01")
FOMC = pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
                       "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
                       "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
                       "2026-09-16"])


def load(und, cost):
    f = "options_premium_trades.parquet" if und == "SPY" else f"options_premium_trades_{und}.parquet"
    W = "5" if und == "SPY" else "0.8%"
    t = pd.read_parquet(f"{RD}/results/{f}")
    x = t[(t.tenor == "0DTE") & (t.struct == "iron condor") & (t.k == 1.0) & (t.W.astype(str) == W) & (t.cost == cost)]
    x = x.assign(d=pd.to_datetime(x.d)).sort_values("d").reset_index(drop=True)
    # features need every day's move / implied move: take them from the full condor rows (same days, any cost)
    x["ratio"] = x.move.abs() / x.M_pct
    x["prev_ratio"] = x.ratio.shift(1)
    x["ratio5"] = x.ratio.shift(1).rolling(5).mean()
    x["vix_chg"] = x.vix.pct_change()
    x["vix_vs20"] = x.vix / x.vix.rolling(20).mean()
    x["event_prem"] = x.M_pct / (x.vix / 100 / np.sqrt(252))
    x["fomc"] = x.d.isin(FOMC).astype(float)
    x["dow"] = x.d.dt.dayofweek.astype(float)
    x["maxloss"] = (x.ror <= -0.9).astype(float)
    x["period"] = np.where(x.d < SPLIT, "dev", "hold")
    return x


def main():
    pd.set_option("display.width", 220)
    rows, quint = [], []
    for und in ("SPY", "QQQ"):
        for cost in (0.01, 0.03):
            x = load(und, cost)
            dev = x[x.period == "dev"]
            for feat in ("vix", "vix_chg", "vix_vs20", "event_prem", "prev_ratio", "ratio5", "fomc", "dow"):
                if feat in ("fomc", "dow"):
                    x["q"] = x[feat]
                else:
                    cuts = np.unique(dev[feat].dropna().quantile([0, .2, .4, .6, .8, 1]).values)
                    cuts[0], cuts[-1] = -np.inf, np.inf
                    x["q"] = pd.cut(x[feat], cuts, labels=False, include_lowest=True, duplicates="drop")
                g = x.dropna(subset=["q"]).groupby(["period", "q"]).agg(n=("ror", "size"), ror=("ror", "mean"), maxloss=("maxloss", "mean"))
                for (per, q), r in g.iterrows():
                    quint.append(dict(und=und, cost=cost, feature=feat, period=per, bucket=q, n=r.n, ror_pct=r.ror * 100,
                                      maxloss_pct=r.maxloss * 100))
                dq = g.loc["dev"].ror
                worst = dq.idxmin()
                h = x[(x.period == "hold") & x.q.notna()]
                keep = h[h.q != worst]
                rows.append(dict(und=und, cost=cost, feature=feat, skip_bucket=worst, dev_skip_ror_pct=dq.min() * 100,
                                 dev_all_ror_pct=x[x.period == "dev"].ror.mean() * 100,
                                 hold_all_ror_pct=h.ror.mean() * 100, hold_kept_ror_pct=keep.ror.mean() * 100,
                                 hold_pnl_kept_pct=keep.ror.sum() / h.ror.sum() * 100 if h.ror.sum() else np.nan,
                                 hold_days_skipped=len(h) - len(keep), hold_maxloss_avoided=int(h.maxloss.sum() - keep.maxloss.sum()),
                                 hold_maxloss_total=int(h.maxloss.sum())))
    R, Q = pd.DataFrame(rows), pd.DataFrame(quint)
    print("SKIP THE DEV-WORST BUCKET -> HOLDOUT (second half, Jul 2025 - Oct 2026)\n" + R.round(2).to_string(index=False))
    print("\nBUCKETS (mean return on risk %, max-loss share %)\n" +
          Q.pivot_table(index=["und", "cost", "feature", "bucket"], columns="period", values=["ror_pct", "maxloss_pct", "n"]).round(1).to_string())
    out = f"{RD}/results"
    os.makedirs(out, exist_ok=True)
    R.to_csv(f"{out}/condor_skip.csv", index=False); Q.to_csv(f"{out}/condor_skip_buckets.csv", index=False)
    good = R[(R.hold_kept_ror_pct > R.hold_all_ror_pct + 0.5) & (R.hold_pnl_kept_pct > 100)]
    print("::notice::condor skip rules that helped in the holdout (kept RoR > all + 0.5pt and more total P&L): "
          + ("; ".join(f"{r.und} {r.cost} {r.feature}: {r.hold_all_ror_pct:+.1f} -> {r.hold_kept_ror_pct:+.1f}%, P&L {r.hold_pnl_kept_pct:.0f}%, "
                       f"max-loss avoided {r.hold_maxloss_avoided}/{r.hold_maxloss_total}" for r in good.itertuples()) or "none"))


if __name__ == "__main__":
    main()
