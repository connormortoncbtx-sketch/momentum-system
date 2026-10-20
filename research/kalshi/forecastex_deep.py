import pandas as pd, numpy as np
def deep(df, key, ev, label):
    """proxy for the bot's deep quote: a trade printing >=2c through the previous print in the aggressor's
    direction within 10s (a sweep); a resting order 2c behind would have filled at prev +/- 2c."""
    df=df.sort_values([key,'t']).copy()
    g=df.groupby(key)
    df['prev']=g.p.shift(); df['dt']=(df.t-g.t.shift()).dt.total_seconds()
    up=(df.p>=df.prev+0.02)&(df.dt<=10); dn=(df.p<=df.prev-0.02)&(df.dt<=10)
    rows=[]
    for side,mask,fill in (('maker sells YES',up,df.prev+0.02),('maker buys YES',dn,df.prev-0.02)):
        x=df[mask].assign(fp=fill[mask]); x=x[(x.fp>0.02)&(x.fp<0.98)]
        x['pnl']=np.where(side=='maker sells YES',x.fp-x.y,x.y-x.fp); rows.append(x)
    x=pd.concat(rows); x=x.drop_duplicates([key,'t'])
    e=x.groupby(ev).pnl.mean()
    print(f'{label}: sweeps {len(x):,}, per fill {x.pnl.mean()*100:+.2f}c, per event {e.mean()*100:+.2f}c (t {e.mean()/e.std()*np.sqrt(len(e)):.1f}), events {len(e)}')
# Kalshi
K='/tmp/claude-0/-home-claude-momentum-system/b1a53494-fbd7-57cf-bb71-3265965926c9/scratchpad/rd/'
t=pd.read_parquet(K+'kalshi/trades_sample.parquet'); m=pd.read_parquet(K+'kalshi/markets.parquet')
m=m[m.result.isin(['yes','no'])]; m['y']=(m.result=='yes').astype(float)
t=t.merge(m[['ticker','event_ticker','close_time','y']],on='ticker'); t['t']=pd.to_datetime(t.t,utc=True)
t['h']=(pd.to_datetime(t.close_time,utc=True)-t.t).dt.total_seconds()/3600
t=t[(t.t>='2025-01-01')&(t.h>=6)&(t.h<48)]
deep(t,'ticker','event_ticker','KALSHI weather 6-48h')
# ForecastEx
import glob
F=K+'forecastex/'
pr=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(F+'prices_*.csv'))])
y=pr[pr.subtype=='YES'].sort_values('date').groupby('event_contract').tail(1); y=y[y.settlement_price.isin([0,1])]
P=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(F+'pairs_*.csv'))]).drop_duplicates('pair_id')
P=P[P.event_contract.isin(y.event_contract)]
P['t']=pd.to_datetime(P.pair_time.str.replace(r'(\.\d{6})\d*',r'\1',regex=True),format='ISO8601',utc=True)
P['h']=(pd.to_datetime(P.expiration_date,utc=True)-P.t).dt.total_seconds()/3600
P['y']=P.event_contract.map(y.set_index('event_contract').settlement_price); P['p']=P.yes_price
P['event']=P.event_contract.str.rsplit('_',n=1).str[0]; P['prod']=P.event_contract.str.split('_').str[0]
P=P[(P.h>=6)&(P.h<48)]
deep(P[P['prod'].str.startswith('UH')],'event_contract','event','FORECASTEX daily highs 6-48h')
deep(P[P['prod'].str.startswith('UH')&(P['prod']!='UHLAX')],'event_contract','event','FORECASTEX daily highs ex-LAX')
deep(P[~P['prod'].str.match('^(UH|UL|HRU)')],'event_contract','event','FORECASTEX non-weather 6-48h')
