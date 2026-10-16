"""
research/rank_formula_test.py -- did the Jun 5, 2026 ranking change (stage-5 rerank fix)
end the good run? Re-ranks every logged week two ways over the same eligible names
(logged composite_rank not null = passed liquidity / EV-veto / earnings exclusions):
  OLD (live until Jun 5): 0.5 x alpha_pct_rank + 0.5 x ev_pct_rank
  NEW (live from Jun 5) : alpha_pct_rank + 0.001 x sig_momentum percentile
Top 10 -> mean forward_return_1w (Fri->Fri as logged) and Mon open -> Fri close.
"""
import numpy as np, pandas as pd
p = pd.read_csv("data/performance_log.csv", low_memory=False, parse_dates=["week_of"])
p = p[(p.week_of >= "2026-04-17") & p.composite_rank.notna() & p.alpha_pct_rank.notna()]
rows = []
for wk, g in p.groupby("week_of"):
    g = g.copy()
    if g.forward_return_1w.notna().mean() < 0.5:
        continue
    g["mo"] = (g.mon_open.where(g.mon_open > 0))
    g["moc"] = (g.fri_close / g.mo - 1).clip(-0.9, 3)
    g["f"] = g.forward_return_1w.clip(-0.9, 3)
    old = g.assign(k=0.5 * g.alpha_pct_rank + 0.5 * g.ev_pct_rank.fillna(0.5)).nlargest(10, "k")
    new = g.assign(k=g.alpha_pct_rank + 0.001 * g.sig_momentum.rank(pct=True)).nlargest(10, "k")
    act = g.nsmallest(10, "composite_rank")
    rows.append(dict(week=wk.date(), old=old.f.mean(), new=new.f.mean(), actual=act.f.mean(), eligible=g.f.mean(),
                     old_moc=old.moc.mean(), new_moc=new.moc.mean(), overlap=len(set(old.symbol) & set(new.symbol)),
                     old_vol=old.weekly_vol_predicted.median(), new_vol=new.weekly_vol_predicted.median()))
d = pd.DataFrame(rows).set_index("week")
pd.set_option("display.width", 200)
print((d[["old", "new", "actual", "eligible", "old_moc", "new_moc"]] * 100).round(1).join(d[["overlap", "old_vol", "new_vol"]].round(1)).to_string())
for lo, hi, nm in (("2026-04-17", "2026-05-29", "before the change (formed Apr 17 - May 29)"),
                   ("2026-06-05", "2026-10-02", "after the change (formed Jun 5 - Oct 2)")):
    x = d.loc[pd.Timestamp(lo).date():pd.Timestamp(hi).date()]
    print(f"\n{nm}: weeks {len(x)}")
    for c in ("old", "new", "actual", "eligible", "old_moc", "new_moc"):
        v = x[c].dropna()
        print(f"  {c:9s} mean {v.mean() * 100:+5.2f}%/wk  cum {((1 + v).prod() - 1) * 100:+6.1f}%  t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}")
