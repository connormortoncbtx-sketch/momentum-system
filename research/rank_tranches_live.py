import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
import backtest as B
P = __import__('os').environ.get('PANEL', 'research_data')
p = B.load_panel(P, '2026-10-02')
d1c, d2o, cl, nd = p.df('d1_close'), p.df('d2_open'), p.df('close'), p.df('n_days')
f = B.features(p)
# next-week returns aligned to formation week
r_mc = (cl / d1c - 1).where(nd >= 2).shift(-1)          # Mon close -> Fri close
r_to = (cl / d2o - 1).where(nd >= 2).shift(-1)          # Tue open -> Fri close (what live did)
r4 = (cl.shift(-4) / d1c.shift(-1) - 1)                 # Mon close -> Fri close 4 weeks later
lw = f['r1w']
d = pd.read_csv('data/performance_log.csv', low_memory=False,
                usecols=['week_of', 'symbol', 'composite_rank', 'alpha_rank'])
d['week_of'] = pd.to_datetime(d.week_of)
d = d[d.week_of.isin(p.weeks)]
def look(df, col):
    i = p.weeks.get_indexer(df.week_of); j = p.symbols.get_indexer(df.symbol)
    ok = j >= 0
    out = np.full(len(df), np.nan); out[ok] = df_m[col][i[ok], j[ok]]
    return out
df_m = {'mc': r_mc.to_numpy(), 'to': r_to.to_numpy(), 'r4': r4.to_numpy(), 'lw': lw.to_numpy()}
for c in df_m: d[c] = look(d, c)
d.loc[d[['mc', 'to', 'r4']].abs().gt(1.5).any(axis=1), ['mc', 'to', 'r4']] = np.nan
edges = [0, 3, 6, 10, 13, 20, 50, 100, 300, 100000]
labels = ['1-3', '4-6', '7-10', '11-13', '14-20', '21-50', '51-100', '101-300', '301+']

def table(d, rank_col, ret, since=None, until=None):
    x = d.dropna(subset=[rank_col, ret]).copy()
    if since: x = x[x.week_of >= since]
    if until: x = x[x.week_of < until]
    x['b'] = pd.cut(x[rank_col], edges, labels=labels)
    x['ex'] = x[ret] - x.groupby('week_of')[ret].transform('mean')
    wk = x.groupby(['week_of', 'b'], observed=True).agg(ex=('ex', 'mean'), raw=(ret, 'mean'), hit=(ret, lambda s: (s > 0).mean()))
    g = wk.groupby('b', observed=True)
    out = pd.DataFrame({'weeks': g.ex.count(), 'raw%': g.raw.mean() * 100, 'excess%': g.ex.mean() * 100,
                        't': g.ex.mean() / (g.ex.std() / np.sqrt(g.ex.count())), 'hit%': g.hit.mean() * 100})
    return out.round(2)

pd.set_option('display.width', 200)
for rc in ['composite_rank', 'alpha_rank']:
    for ret, lab in [('to', 'Tue open->Fri close (live timing)'), ('mc', 'Mon close->Fri close'), ('r4', '4-week hold')]:
        print(f'\n== {rc}, {lab}, all weeks with ranks'); print(table(d, rc, ret).to_string())
print('\n== composite_rank, Tue open, post Jun-5 fix'); print(table(d, 'composite_rank', 'to', since='2026-06-05').to_string())
print('\n== composite_rank, Tue open, before fix'); print(table(d, 'composite_rank', 'to', until='2026-06-05').to_string())
# exhaustion: does last-week return of top-3 differ?
x = d.dropna(subset=['composite_rank'])
x['b'] = pd.cut(x.composite_rank, edges, labels=labels)
print('\nlast-week return (formation week) by bucket, median %:'); print((x.groupby('b', observed=True).lw.median() * 100).round(2).to_string())
print('weeks with composite ranks:', x.week_of.nunique(), x.week_of.min().date(), x.week_of.max().date())
