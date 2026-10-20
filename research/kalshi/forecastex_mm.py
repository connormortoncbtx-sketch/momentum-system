import pandas as pd, numpy as np, glob
pr=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('prices_*.csv'))])
y=pr[pr.subtype=='YES'].sort_values('date').groupby('event_contract').tail(1)
y=y[y.settlement_price.isin([0,1])&(pd.to_datetime(y.date)>=pd.to_datetime(y.expiration_date.str[:10])-pd.Timedelta(days=1))]
Y=y.set_index('event_contract').settlement_price
P=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('pairs_*.csv'))]).drop_duplicates('pair_id')
P['t']=pd.to_datetime(P.pair_time.str.replace(r'(\.\d{6})\d*',r'\1',regex=True),format='ISO8601',utc=True)
P['exp']=pd.to_datetime(P.expiration_date,utc=True)
P['h']=(P.exp-P.t).dt.total_seconds()/3600
P=P[P.event_contract.isin(Y.index)].sort_values(['event_contract','t'])
P['y']=P.event_contract.map(Y)
P['prod']=P.event_contract.str.split('_').str[0]
P['event']=P.event_contract.str.rsplit('_',n=1).str[0]
d=P.groupby('event_contract').yes_price.diff()
s=np.sign(d).replace(0,np.nan); s=s.groupby(P.event_contract).ffill()   # +1 = YES buyer aggressed
P['s']=s
P=P[P.s.notna()&(P.yes_price>0.01)&(P.yes_price<0.99)]
P['maker']=np.where(P.s>0,P.yes_price-P.y,P.y-P.yes_price)
P['grp']=np.where(P['prod'].str.startswith('UH'),'daily high',np.where(P['prod'].str.startswith('UL'),'daily low',np.where(P['prod'].str.startswith('HRU'),'hourly temp','other')))
def stats(g):
    pc=(g.maker*g.quantity).sum()/g.quantity.sum()*100
    ev=(g.maker*g.quantity).groupby(g.event).sum()/g.quantity.groupby(g.event).sum()
    return pd.Series(dict(trades=len(g),contracts_k=g.quantity.sum()/1e3,events=g.event.nunique(),maker_c_per_ctr=pc,
        per_event_c=ev.mean()*100,t=ev.mean()/ev.std()*np.sqrt(len(ev))))
pd.set_option('display.width',250)
for w,(a,b) in {'all':(0,1e9),'6-48h':(6,48),'0-6h':(0,6),'48h+':(48,1e9)}.items():
    q=P[(P.h>=a)&(P.h<b)]
    print('==',w); print(q.groupby('grp').apply(stats).round(2).to_string())
q=P[(P.h>=6)&(P.h<48)&(P['grp']=='daily high')]
print(q.groupby('prod').apply(stats).sort_values('contracts_k',ascending=False).round(2).head(30).to_string())
print('median trade size',P.groupby('grp').quantity.median().to_dict())
