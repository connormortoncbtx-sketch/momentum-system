"""Top-10 portfolios with/without V1-style post-scoring filters (uses post_filters.py setup)."""
import os, sys
os.environ.setdefault('PANEL','research_data')
sys.argv=['x']
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'post_filters.py')).read().split('def nw_t')[0]
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
exec(src)
ret=RET['live_tue']; sc=SCORE['live_tue']; m=U&ret.notna()&sc.notna()
pool=m&(sc.where(m).rank(axis=1,ascending=False)<=50)
q=lambda fac: fac.where(pool).rank(axis=1,pct=True)
mon=q(F['monday_ret (Fri close->Mon close)']); pw=q(F['prior_week_ret']); tg=q(F['tuesday_open_gap'])
variants={
 'A plain top-10 (no filters)': pool,
 'B skip top-20% Monday runners': pool&(mon<=0.8),
 'C skip top-20% Monday AND top-20% prior week': pool&(mon<=0.8)&(pw<=0.8),
 'D skip top-20% Tuesday open gap': pool&(tg<=0.8),
 'E current rule: skip earnings in hold week': pool&(F['earnings_in_hold_week (flag)']==0),
}
for name,msk in variants.items():
    s=sc.where(msk); out={}
    for wk,row in s.iterrows():
        row=row.dropna()
        if len(row)<10: continue
        out[wk]=ret.loc[wk,row.nlargest(10).index].mean()-2*15/1e4
    r=pd.Series(out); r=r[r.index>=start]
    d,h=B.stats(r[r.index<split]),B.stats(r[r.index>=split])
    print(f"{name:48s} dev CAGR {d['CAGR%']:6.1f}% Sharpe {d['Sharpe']:5.2f} | holdout CAGR {h['CAGR%']:6.1f}% Sharpe {h['Sharpe']:5.2f}")
