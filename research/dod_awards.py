"""
research/dod_awards.py -- a 5th independent book? DoD contract awards >= $7.5M (Oct 16, 2026)

DoD announces every award >= $7.5M at about 5pm ET on the award day (defense.gov / war.gov).
Those pages block cloud servers, so history comes from USAspending: NEW DoD contract awards
(type A/B/C/D) >= $7.5M, 2017-2026, with the base obligation date as the announcement day.
Caveats: a few awards are announced a day or two late (or not at all); USAspending's DoD
data is published with a 90-day lag, so this is a historical test, not a live feed.

Steps
  1. fetch  -> <RD>/dod/awards_<year>.parquet (cached; re-run resumes)
  2. map    recipient name -> current SEC ticker (normalised exact or prefix match, >= 8 chars)
  3. events award day 16:00 ET -> entry at the next open; size = award / market cap
  4. tests  event drift vs IWM and vs random 8-K-style draws; diversified book (all events
            >= X% of market cap, 10-session hold, 1/vol weights, 25% cap) chosen on 2017-23,
            scored on 2024-26; daily series saved for research/strategy_stack.py
"""
import glob, json, os, re, sys, time, urllib.request
import numpy as np, pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
OUT = f"{RD}/dod"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{RD}/results", exist_ok=True)
SPLIT = pd.Timestamp("2024-01-01")
FIELDS = ["Award ID", "Recipient Name", "Award Amount", "Base Obligation Date", "Start Date",
          "Recipient UEI", "Awarding Sub Agency", "Description", "Base and All Options Value"]


def post(body, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request("https://api.usaspending.gov/api/v2/search/spending_by_award/",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 422 and "Base and All Options Value" in body["fields"]:
                body["fields"] = [f for f in body["fields"] if f != "Base and All Options Value"]
                continue
            time.sleep(5 * (i + 1))
        except Exception:
            time.sleep(5 * (i + 1))
    return None


def fetch_month(y, m):
    start = pd.Timestamp(y, m, 1); end = start + pd.offsets.MonthEnd(0)
    rows, page = [], 1
    while True:
        body = {"filters": {"award_type_codes": ["A", "B", "C", "D"],
                            "agencies": [{"type": "awarding", "tier": "toptier", "name": "Department of Defense"}],
                            "time_period": [{"start_date": str(start.date()), "end_date": str(end.date()),
                                             "date_type": "new_awards_only"}],
                            "award_amounts": [{"lower_bound": 7500000}]},
                "fields": list(FIELDS), "page": page, "limit": 100, "sort": "Award Amount", "order": "desc"}
        d = post(body)
        if d is None:
            print(f"::warning::USAspending failed {y}-{m:02d} page {page}"); break
        res = d.get("results", [])
        rows += res
        if not d.get("page_metadata", {}).get("hasNext") or not res:
            break
        page += 1
    df = pd.DataFrame(rows)
    if len(df):
        df["base_date"] = pd.to_datetime(df["Base Obligation Date"], errors="coerce")
        df = df[(df.base_date >= start) & (df.base_date <= end)]        # NEW awards in this month only
    return df


def fetch():
    for y in range(2017, 2027):
        f = f"{OUT}/awards_{y}.parquet"
        if os.path.exists(f) and y < 2026:
            continue
        months = [m for m in range(1, 13) if pd.Timestamp(y, m, 1) <= pd.Timestamp("2026-09-30")]
        parts = [fetch_month(y, m) for m in months]
        df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
        if len(df):
            for c in df.columns:
                if df[c].dtype == object and c not in ("base_date",):
                    df[c] = df[c].astype(str)
            df.to_parquet(f, index=False)
        print(f"  {y}: {len(df):,} new awards >= $7.5M", flush=True)


SUFFIX = re.compile(r"\b(THE|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LLC|L L C|LTD|LIMITED|PLC|LP|L P|"
                    r"HOLDINGS?|GROUP|N A|NA|SA|NV|AG|SE)\b")


def norm(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper().replace("&", " AND "))
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", s)).strip()


def ticker_map():
    raw = urllib.request.urlopen(urllib.request.Request("https://www.sec.gov/files/company_tickers.json",
                                 headers={"User-Agent": os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"}),
                                 timeout=60).read()
    m = pd.DataFrame(json.loads(raw).values())
    m["key"] = m.title.map(norm); m["ticker"] = m.ticker.str.upper().str.replace("-", ".")
    m = m[m.key.str.len() >= 4].drop_duplicates("key")
    return m[["key", "ticker", "cik_str", "title"]]


def map_recipients(aw, tm):
    keys = dict(zip(tm.key, tm.ticker)); cik = dict(zip(tm.ticker, tm.cik_str))
    long_keys = sorted([k for k in keys if len(k) >= 8], key=len, reverse=True)
    out = {}
    for r in aw["Recipient Name"].unique():
        k = norm(r)
        if k in keys:
            out[r] = keys[k]; continue
        for lk in long_keys:                      # subsidiary naming: "NORTHROP GRUMMAN SYSTEMS" -> NORTHROP GRUMMAN
            if k.startswith(lk + " "):
                out[r] = keys[lk]; break
    aw["symbol"] = aw["Recipient Name"].map(out)
    aw["cik"] = aw.symbol.map(cik)
    return aw


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    import contract_filters as CF
    fetch()
    aw = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/awards_*.parquet")))
    aw["value"] = pd.to_numeric(aw.get("Base and All Options Value", pd.Series(np.nan, index=aw.index)), errors="coerce")
    aw["value"] = aw.value.fillna(pd.to_numeric(aw["Award Amount"], errors="coerce"))
    aw = map_recipients(aw, ticker_map())
    print(f"awards {len(aw):,}; mapped to a current ticker {aw.symbol.notna().sum():,} "
          f"({aw.symbol.notna().mean() * 100:.0f}%), {aw.symbol.nunique():,} companies")
    ev = aw[aw.symbol.notna()].copy()
    ev["t"] = ev.base_date + pd.Timedelta(hours=16)                 # announced ~5pm ET (16:00 to be safe)
    # one event per company-day, summed value
    ev = ev.groupby(["symbol", "t"], as_index=False).agg(value=("value", "sum"), n=("value", "size"),
                                                          cik=("cik", "first"))
    # market cap before the award: shares (SEC fundamentals) x prior close
    fu = pd.concat(pd.read_parquet(f, columns=["cik", "tag", "value", "accepted_et"])
                   for f in sorted(glob.glob(f"{RD}/sec/fundamentals_20*.parquet")))
    sh = fu[fu.tag.isin(["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"])].dropna(subset=["accepted_et"])
    sh = sh.sort_values("accepted_et")[["cik", "accepted_et", "value"]].rename(columns={"value": "shares"})
    ev = ev.sort_values("t")
    ev = pd.merge_asof(ev, sh.assign(cik=sh.cik.astype(float)), left_on="t", right_on="accepted_et",
                       by="cik", direction="backward")
    M = CF.Market(*CF.bars(list(ev.symbol.unique()) + ["IWM", "SPY"], pd.Timestamp("2016-09-01")))
    P = M.entry_pos(ev.t)
    prev = np.array([M.C[s].iloc[i - 1] if (s in M.C.columns and 0 < i <= len(M.days)) else np.nan for s, i in zip(ev.symbol, P)])
    ev["mcap"] = ev.shares * prev
    ev["rel"] = ev.value / ev.mcap
    # event returns vs IWM from the entry open
    O, C = M.O, M.C
    for h in (5, 20):
        out = []
        for s, i in zip(ev.symbol, P):
            if s not in C.columns or i >= len(M.days) - h or not np.isfinite(O[s].iloc[i]):
                out.append(np.nan); continue
            r = C[s].iloc[i + h] / O[s].iloc[i] - 1
            b = C["IWM"].iloc[i + h] / O["IWM"].iloc[i] - 1
            out.append(r - b)
        ev[f"ab_{h}"] = out
    ev["period"] = np.where(ev.t < SPLIT, "dev", "holdout")
    ev.to_parquet(f"{RD}/results/dod_events.parquet", index=False)
    print("\nEVENT LEVEL (IWM-adjusted from the next open, %), by award size vs market cap")
    ev["size_bin"] = pd.cut(ev.rel, [0, 0.001, 0.01, 0.05, np.inf], labels=["<0.1%", "0.1-1%", "1-5%", ">=5%"])
    for per in ("dev", "holdout"):
        for b, g in ev[ev.period == per].groupby("size_bin", observed=True):
            for h in ("ab_5", "ab_20"):
                v = g[h].dropna()
                t = v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 10 else np.nan
                print(f"  {per:8s} {b:>7s} {h}: n {len(v):5d} mean {v.mean() * 100:+6.2f} median {v.median() * 100:+6.2f} t {t:+5.2f}")
    # books (dev-chosen size threshold), same diversified construction as the contract book
    rows = []
    E = ev.join(M.features(ev)[["vol60"]])
    for thr in (0.0, 0.001, 0.01, 0.05):
        for H in (10, 20):
            X = E[E.rel >= thr] if thr > 0 else E
            r = M.book(X, H=H, cap=0.25, size="invvol")
            d, h = CF.st(r[r.index < SPLIT]), CF.st(r[r.index >= SPLIT])
            rows.append(dict(thr=thr, H=H, n=len(X), **{f"dev_{k}": v for k, v in d.items()}, **{f"hold_{k}": v for k, v in h.items()}))
    res = pd.DataFrame(rows).sort_values("dev_Sharpe", ascending=False)
    pd.set_option("display.width", 220)
    print("\nBOOKS (10/20-session hold, 1/vol weights, 25% cap; ranked on dev)"); print(res.round(2).to_string(index=False))
    best = res.iloc[0]
    X = E[E.rel >= best.thr] if best.thr > 0 else E
    rb = M.book(X, H=int(best.H), cap=0.25, size="invvol")
    pd.DataFrame({"dod": rb}).to_csv(f"{RD}/results/dod_book_daily.csv")
    print(f"::notice::DoD awards: {len(aw):,} awards, {aw.symbol.notna().mean() * 100:.0f}% mapped; best-on-dev thr {best.thr} H{int(best.H)}: "
          f"dev Sharpe {best.dev_Sharpe:.2f} ({best.dev_CAGR:.0f}%/yr), holdout Sharpe {best.hold_Sharpe:.2f} ({best.hold_CAGR:.0f}%/yr)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
