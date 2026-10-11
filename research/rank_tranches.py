import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
import backtest as B
P = __import__('os').environ.get('PANEL', 'research_data')
p = B.load_panel(P, '2026-10-02'); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
d1c, cl = p.df('d1_close'), p.df('close')
fwd1 = B.next_week_returns(p, 'designed')
fwd4 = (cl.shift(-4) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 3)
rk = lambda x: B.xrank(x, U)
proxy = (0.35 * rk(f['rs_live']) + 0.25 * rk(f['trend']) + 0.20 * rk(f['hi52'])) / 0.80
start = pd.Timestamp('2017-01-06'); split = pd.Timestamp('2024-01-01')
edges = [0, 3, 6, 10, 13, 20, 50, 100, 250, 500, 100000]
labels = ['1-3', '4-6', '7-10', '11-13', '14-20', '21-50', '51-100', '101-250', '251-500', '501+']

def buckets(score, fwd, mask):
    s = score.where(mask & fwd.notna())
    r = s.rank(axis=1, ascending=False, method='first')
    ex = fwd.sub(fwd.where(s.notna()).mean(axis=1), axis=0)
    rows = {}
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (r > lo) & (r <= hi)
        rows[lab] = ex.where(m).mean(axis=1)
    return pd.DataFrame(rows)

def summarize(wk, k=1):
    out = {}
    for name, sl in (('DEV 2017-23', wk.index < split), ('HOLD 2024-26', wk.index >= split)):
        w = wk[sl & (wk.index >= start)]
        # non-overlapping t-stat approximation for k-week returns: use every k-th week
        wn = w.iloc[::k]
        out[(name, 'excess%')] = w.mean() * 100
        out[(name, 't')] = wn.mean() / (wn.std() / np.sqrt(wn.count()))
    return pd.DataFrame(out).round(2)

pd.set_option('display.width', 220)
sig = {'live proxy': proxy, 'rs_live (12/6/3-1 blend)': f['rs_live'], '12-1 momentum': f['m12_1']}
for name, s in sig.items():
    print(f'\n### {name}: 1-week excess vs universe (Mon close->Fri close), weekly %')
    print(summarize(buckets(s, fwd1, U)).to_string())
    print(f'### {name}: 4-week excess, %')
    print(summarize(buckets(s, fwd4, U), k=4).to_string())

# what distinguishes the very top? last-week return and volatility by proxy bucket
s = proxy.where(U & fwd1.notna()); r = s.rank(axis=1, ascending=False, method='first')
for lab, (lo, hi) in {'1-3': (0, 3), '4-13': (3, 13), '14-50': (13, 50), 'rest': (50, 1e9)}.items():
    m = (r > lo) & (r <= hi)
    print(lab, 'median last-wk ret %.2f%%' % (f['r1w'].where(m).stack().median() * 100),
          'median 12-wk vol %.1f%%' % (f['vol12'].where(m).stack().median() * 100),
          'median dist from 52w hi %.1f%%' % ((1 - f['hi52'].where(m).stack().median()) * 100))
