"""
research/spike_catalog.py -- step 1 of the "event minus one" study (Oct 2026)

Catalog every big weekly move since 2017 and tie it to its catalyst using the SEC
filings index (8-K item codes, M&A forms, activist stakes, offerings) and the
earnings-event table. Then: what share is scheduled vs unscheduled, by type, and
what each type does over the following 4 weeks (continuation vs fade).

Spike  = Friday-to-Friday return >= +25% above the equal-weight universe.
Crash  = <= -25%. Universe: price >= $2, >= $500k/day, >= 26 weeks listed, no funds.
Catalyst window: filings accepted from the prior Friday 16:00 ET to this Friday 16:00 ET.
Known gap: the filings index covers issuers listed TODAY, so acquired (delisted)
companies -- most M&A targets -- are missing; M&A is under-counted.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import backtest as B

P = os.environ.get("PANEL", "research_data")
SEC = os.environ.get("SEC", f"{P}/sec")
OUT = os.environ.get("OUT", "research/results")
p = B.load_panel(P, "2026-10-02"); f = B.features(p)
C = p.df("close")
U = (f["price"] >= 2) & (f["adv"] >= 5e5) & (f["hist"] >= 26) & ~pd.Series(p.is_fund, index=p.symbols)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() < 5)
ex = r.sub(r.where(U).mean(axis=1), axis=0)
fwd4 = (C.shift(-4) / C - 1).sub((C.shift(-4) / C - 1).where(U).mean(axis=1), axis=0)

m = U.shift(1, fill_value=False) & ex.notna()          # in universe the week BEFORE the move
ev = []
for kind, sel in (("spike", ex >= 0.25), ("crash", ex <= -0.25)):
    s = (sel & m).stack()
    s = s[s]
    d = s.reset_index(); d.columns = ["week", "symbol", "_"]
    d["kind"] = kind
    ev.append(d.drop(columns="_"))
ev = pd.concat(ev)
ev = ev[ev.week >= "2017-01-06"]
i = p.weeks.get_indexer(ev.week); j = p.symbols.get_indexer(ev.symbol)
ev["ex_ret"] = ex.to_numpy()[i, j]
ev["fwd4_ex"] = fwd4.to_numpy()[i, j]
ev["price_prior"] = f["price"].to_numpy()[i - 1, j]
ev["adv_prior"] = f["adv"].to_numpy()[i - 1, j]
print(ev.groupby("kind").size(), "\nper year:\n", ev.groupby([ev.week.dt.year, "kind"]).size().unstack())

# ── catalysts ───────────────────────────────────────────────────────────────
fi = pd.concat(pd.read_parquet(x, columns=["cik", "tickers", "form", "items", "accepted_et"])
               for x in sorted(glob.glob(f"{SEC}/filings_*.parquet")))
fi = fi.assign(symbol=fi.tickers.str.split(",")).explode("symbol")
fi = fi[fi.symbol.isin(set(ev.symbol))]
# week label: Friday 16:00 cut-off (stored ET times can be up to 5h late -> conservative)
fi = fi.reset_index(drop=True)
sh = (fi.accepted_et + pd.Timedelta(hours=8)).dt.normalize()     # after Fri 16:00 rolls into next week
fi["week"] = sh + pd.to_timedelta((4 - sh.dt.weekday) % 7, unit="D")
e = pd.read_parquet(f"{P}/earnings_events.parquet", columns=["symbol", "d0"])
e["d0"] = pd.to_datetime(e.d0)
e["week"] = e.d0 + pd.to_timedelta((4 - e.d0.dt.weekday) % 7, unit="D")
earn = set(zip(e.symbol, e.week))
covered = set(fi.symbol)

MA_FORMS = {"SC TO-T", "SC TO-T/A", "SC 14D9", "SC 14D9/A", "DEFM14A", "PREM14A", "425", "S-4", "SC 13E3", "DEFM14C"}
OFFER = {"424B4", "424B5", "424B1", "424B3"}
g = fi.groupby(["symbol", "week"])
agg = g.agg(forms=("form", lambda s: set(s)), items=("items", lambda s: set(",".join(s.fillna("")).split(","))))
agg = agg.to_dict("index")


def classify(sym, wk):
    if sym not in covered:
        return "no filings data (delisted issuer)"
    a = agg.get((sym, wk), {"forms": set(), "items": set()})
    forms, items = a["forms"], {x.strip() for x in a["items"]}
    if (sym, wk) in earn or "2.02" in items:
        return "earnings (scheduled)"
    if forms & MA_FORMS or "2.01" in items:
        return "M&A"
    if "4.02" in items or "4.01" in items or "3.01" in items:
        return "accounting / listing problem"
    if forms & OFFER or "3.02" in items:
        return "offering / dilution"
    if "1.01" in items or "1.02" in items:
        return "material agreement"
    if any(x.startswith("SC 13D") for x in forms):
        return "activist / 13D stake"
    if "5.02" in items:
        return "management change"
    if "8.01" in items or "7.01" in items:
        return "press release / other 8-K"
    if forms - {"4", "4/A", "144", "SC 13G", "SC 13G/A", "3"}:
        return "other filing"
    return "no filing (news or flow)"


ev["catalyst"] = [classify(s, w) for s, w in zip(ev.symbol, ev.week)]
ev["scheduled"] = ev.catalyst.eq("earnings (scheduled)")
os.makedirs(OUT, exist_ok=True)
ev.to_parquet(f"{OUT}/spike_catalog.parquet", index=False)

pd.set_option("display.width", 200)
for kind in ("spike", "crash"):
    x = ev[ev.kind == kind]
    t = x.groupby("catalyst").agg(n=("symbol", "size"), share=("symbol", lambda s: len(s) / len(x) * 100),
                                  median_move=("ex_ret", lambda s: s.median() * 100),
                                  next4wk_mean=("fwd4_ex", lambda s: s.mean() * 100),
                                  next4wk_median=("fwd4_ex", lambda s: s.median() * 100),
                                  next4wk_up=("fwd4_ex", lambda s: (s > 0).mean() * 100))
    print(f"\n### {kind}s ({len(x):,}): catalyst mix and what happens over the next 4 weeks (excess, %)")
    print(t.sort_values("n", ascending=False).round(1).to_string())
# size profile
print("\nmedian prior price / $ volume of spikes:", ev[ev.kind == "spike"].price_prior.median().round(2),
      f"${ev[ev.kind == 'spike'].adv_prior.median() / 1e6:.1f}M/day")
