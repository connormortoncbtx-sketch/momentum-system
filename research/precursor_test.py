"""
research/precursor_test.py -- test the concrete precursors named in the winner
post-mortems (research/winner_postmortems.py, Oct 11 2026) against base rates.

The post-mortems named few precursors; the ones observable as SEC form types:
  merger_comm   Form 425 (written merger communications)        AVAV, MOVE
  solicit       DEFA14A / PREC14A / DFAN14A (proxy solicitation) SDEV
  merger_proxy  PREM14A / DEFM14A (merger proxy)
  tender        SC TO-T / SC TO-I / SC 14D9 (tender offers)
  activist      SC 13D (new 5%+ activist stake)
  late_filer    NT 10-K / NT 10-Q (notification of late filing)   -- loser precursor
  pre_vote      PRE 14A (preliminary proxy: votes on splits, share authorisations) MLGO, TJGC
For each (symbol, week of first such filing in a 4-week gap), the NEXT 1, 4 and 13
weeks' excess return vs the equal-weight universe, and the share of events that
became a run-rate winner (next 13w >= +46.9%) vs the universe base rate.
Event week = Friday after acceptance (Fri 16:00 ET cut-off); returns start the
week AFTER, so they are tradeable. Dev 2017-23, holdout 2024-26.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{P}/sec")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
C = p.df("close")
U = (f["price"] >= 2) & (f["adv"] >= 5e5) & (f["hist"] >= 26) & ~pd.Series(p.is_fund, index=p.symbols)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() < 3)
ew = r.where(U).mean(axis=1)
ex = r.sub(ew, axis=0)
ex4 = ex.shift(-1).rolling(4).sum().shift(-3)            # weeks t+1..t+4
r13 = C.shift(-13) / C - 1
ew13 = r13.where(U).mean(axis=1)
ex13 = r13.sub(ew13, axis=0)
win13 = (r13 >= 1.03 ** 13 - 1).astype(float).where(r13.notna())

GROUPS = {"merger_comm": ["425"], "solicit": ["DEFA14A", "PREC14A", "DFAN14A"],
          "merger_proxy": ["PREM14A", "DEFM14A"], "tender": ["SC TO-T", "SC TO-I", "SC 14D9"],
          "activist": ["SC 13D", "SCHEDULE 13D"], "late_filer": ["NT 10-K", "NT 10-Q"], "pre_vote": ["PRE 14A"]}
fi = pd.concat(pd.read_parquet(x, columns=["tickers", "form", "accepted_et"])
               for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
fi = fi[fi.accepted_et >= "2017-01-01"]
fi["symbol"] = fi.tickers.fillna("").str.split(",").str[0]
sh = (fi.accepted_et + pd.Timedelta(hours=8)).dt.normalize()
fi["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")

M = {k: v.to_numpy() for k, v in {"ex1": ex.shift(-1), "ex4": ex4, "ex13": ex13, "win13": win13, "U": U}.items()}
base = {}
for per, sl in (("dev", slice("2017-01-01", "2023-12-31")), ("holdout", slice("2024-01-01", "2026-10-02"))):
    m = U.loc[sl]
    base[per] = {"ex1": 0.0, "ex4": 0.0, "ex13": 0.0, "win13": win13.loc[sl].where(m).stack().mean()}


def tstat(x):
    x = x.dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 30 else np.nan


rows = []
for g, forms in GROUPS.items():
    e = fi[fi.form.isin(forms)].drop_duplicates(["symbol", "week"]).sort_values("week")
    e["gap"] = e.groupby("symbol").week.diff().dt.days
    e = e[e.gap.isna() | (e.gap > 28)]                           # first in a 4-week gap
    i = p.weeks.get_indexer(e.week); j = p.symbols.get_indexer(e.symbol)
    ok = (i >= 0) & (j >= 0)
    e, i, j = e[ok].copy(), i[ok], j[ok]
    inu = M["U"][i, j]
    e, i, j = e[inu], i[inu], j[inu]
    for k in ("ex1", "ex4", "ex13", "win13"):
        e[k] = M[k][i, j]
    e["period"] = np.where(e.week < "2024-01-01", "dev", "holdout")
    for per, gg in e.groupby("period"):
        rows.append({"precursor": g, "period": per, "n": len(gg),
                     "next1_%": gg.ex1.mean() * 100, "t1": tstat(gg.ex1),
                     "next4_%": gg.ex4.mean() * 100, "t4": tstat(gg.ex4),
                     "next13_%": gg.ex13.mean() * 100, "t13": tstat(gg.ex13),
                     "winner_rate_%": gg.win13.mean() * 100, "base_rate_%": base[per]["win13"] * 100})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("Excess return (%) after the filing week; winner = next 13w >= +46.9%\n")
print(res.round(2).to_string(index=False))
