"""Hold-length and condition-based entry/exit grid. Dev 2017-2023 picks, holdout 2024-2026 once."""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
import backtest as B
P = __import__('os').environ.get('PANEL', 'research_data')
p = B.load_panel(P, '2026-10-02'); f = B.features(p)
U = B.universe(p, f, 5.0, 2e6)
rk = lambda x: B.xrank(x, U)
proxy = (0.35 * rk(f['rs_live']) + 0.25 * rk(f['trend']) + 0.20 * rk(f['hi52'])) / 0.80
SCORES = {'live_proxy': proxy, 'rs_live': f['rs_live'], 'm12_1': f['m12_1'],
          'm12_1+rev': rk(f['m12_1']) + rk(-f['r1w'])}
r1pct = rk(f['r1w'])
d1c = p.df('d1_close'); C = p.df('close')
WR = (d1c.shift(-2) / d1c.shift(-1) - 1); WR = WR.where(WR.abs() < 1.5)       # Mon close->Mon close, aligned to formation week
weeks = p.weeks; start = pd.Timestamp('2017-01-06'); split = pd.Timestamp('2024-01-01')
COST = 15.0

def sim(score, n=20, buffer=None, trail=None, min_hold=1, max_hold=None, entry=None, k=1, offset=0, fixed=False):
    """Weekly loop. Every week (or every k weeks for fixed holds) at Monday close using Friday data:
    exit names that fail the hold condition, fill to n with top-ranked names passing `entry`."""
    S = score.where(U).to_numpy(); Wr = WR.to_numpy(); Cl = C.to_numpy()
    E = None if entry is None else entry.to_numpy()
    sym = p.symbols
    held = {}          # col -> dict(weeks_held, peak_close)
    w = {}             # col -> weight
    out = {}
    for i, wk in enumerate(weeks):
        if wk < start - pd.Timedelta(days=7 * 60):
            continue
        row = S[i]; ok = ~np.isnan(row) & ~np.isnan(Wr[i])
        if ok.sum() < n * 3:
            continue
        rebalance = ((i + offset) % k == 0)
        cost = 0.0
        if rebalance:
            order = np.argsort(-np.where(ok, row, -np.inf))
            rank = np.empty_like(order); rank[order] = np.arange(1, len(order) + 1)
            keep = []
            for c, h in held.items():
                if not ok[c]:
                    continue
                stay = True
                if fixed:                             # fixed hold: full refresh each k weeks
                    stay = False
                if buffer is not None and rank[c] > buffer and h['wk'] >= min_hold:
                    stay = False
                if trail is not None and Cl[i, c] < h['peak'] * (1 - trail) and h['wk'] >= 1:
                    stay = False
                if max_hold is not None and h['wk'] >= max_hold:
                    stay = False
                if stay:
                    keep.append(c)
            cands = [c for c in order[: n * 6] if ok[c] and c not in keep
                     and (E is None or (E[i, c] if not np.isnan(E[i, c]) else False))]
            new = keep + cands[: n - len(keep)]
            tgt = {c: 1.0 / len(new) for c in new} if new else {}
            traded = sum(abs(tgt.get(c, 0) - w.get(c, 0)) for c in set(tgt) | set(w))
            cost = traded * COST / 1e4
            held = {c: held.get(c, {'wk': 0, 'peak': Cl[i, c]}) for c in new}
            w = tgt
        if not w:
            continue
        r = np.array([Wr[i, c] for c in w]); r = np.where(np.isnan(r), 0.0, r)
        wt = np.array(list(w.values()))
        out[wk] = float((wt * r).sum()) - cost
        nw = wt * (1 + r); nw = nw / nw.sum() if nw.sum() > 0 else nw
        w = dict(zip(w.keys(), nw))
        for c in held:
            held[c]['wk'] += 1
            if i + 1 < len(weeks) and not np.isnan(Cl[i + 1, c]):
                held[c]['peak'] = max(held[c]['peak'], Cl[i + 1, c])
    s = pd.Series(out, dtype=float)
    return s[s.index >= start]

def staggered(score, n, k, **kw):
    runs = [sim(score, n=n, k=k, offset=o, fixed=True, **kw) for o in range(k)]
    return pd.concat(runs, axis=1, sort=True).mean(axis=1)

def st(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 52
    return dict(CAGR=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, vol=r.std() * np.sqrt(52) * 100,
                Sharpe=r.mean() / r.std() * np.sqrt(52), maxDD=(eq / eq.cummax() - 1).min() * 100)

rows = []
def run(name, family, r):
    d, h = r[r.index < split], r[r.index >= split]
    rows.append(dict(family=family, design=name, **{f'dev_{a}': b for a, b in st(d).items()},
                     **{f'hold_{a}': b for a, b in st(h).items()}))

# benchmarks
spy = WR['SPY'] if 'SPY' in WR else None
ew = WR.where(U).mean(axis=1); ew = ew[ew.index >= start]
run('SPY (Mon close -> Mon close)', 'benchmark', spy[spy.index >= start].dropna())
run('Equal-weight universe', 'benchmark', ew)

no_spike = (r1pct < 0.90)          # exclude last week's top-10% movers
pullback = (f['r1w'] < 0)
for sname, sc in SCORES.items():
    for n in (10, 20):
        for k in (1, 2, 4, 8, 13):                             # (a) fixed N-week holds, staggered
            run(f'{sname} n{n} fixed {k}w', 'A fixed hold', staggered(sc, n, k))
        for mult in (2, 3, 5, 10):                             # (b) hold while rank <= buffer
            run(f'{sname} n{n} hold-while-rank<={n*mult}', 'B rank-buffer exit', sim(sc, n=n, buffer=n * mult))
        for tr in (0.10, 0.20):                               # (c) trailing stop + buffer
            run(f'{sname} n{n} buffer{n*5} trail{int(tr*100)}', 'C trailing stop', sim(sc, n=n, buffer=n * 5, trail=tr))
        for en, eo in (('no-spike', no_spike), ('pullback', pullback)):   # (d) entry conditions
            run(f'{sname} n{n} buffer{n*5} entry={en}', 'D entry condition', sim(sc, n=n, buffer=n * 5, entry=eo))
    print('done', sname, flush=True)

res = pd.DataFrame(rows)
res.to_csv('research/results/hold_conditions.csv', index=False)
print(len(res), 'designs')
