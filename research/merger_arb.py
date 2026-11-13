"""
research/merger_arb.py -- merger arbitrage, long-only (Nov 13, 2026; Connor: "in any regime there are winners -- find
them"). A mechanism, not a forecast: after a takeover is agreed, the target trades a few % below the deal price because
some deals break and holders who want out sell; whoever holds to completion collects the gap. Historically ~5-10%/yr
above cash with low market beta (Mitchell & Pulvino 2001; Baker & Savasoglu 2002).

Events (SEC filings on the TARGET's CIK, 2016+): first of PREM14A / DEFM14A (merger proxy), SC TO-T / SC TO-C (tender
offer), SC 14D9 (target's response) with no such filing in the prior 52 weeks. Entry: Monday close after the filing
week (so the announcement jump is already in the price); held while the stock trades, max 52 weeks; a stock that stops
trading is assumed taken out at its last price (0% exit return -- the deal paid). No hedge, cash and stock deals mixed.
PINNED=1 (default) keeps only events where the stock rose >= 15% over the 8 weeks to entry and its last 3 weekly
moves had a std below 2% -- the signature of a cash deal priced just under its offer (all ex ante). PINNED=0 = every event.
Tradeable at entry: price >= $1 and >= $250k/day. Costs: per-name half-spread table (universe_floor.py), both legs.
Portfolio: equal weight across all open deals each week (uninvested = 0% here; T-bill shown separately).
Survivorship caveat: completed targets are covered only through filings_delisted / the inactive part of the panel, so
the event set over-represents deals whose targets still trade (often broken deals) -> results lean pessimistic.
Dev 2017-2023 / holdout 2024-Sep 2026. Public data only.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import universe_floor as U  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
PINNED = os.environ.get("PINNED", "1") == "1"   # 1 = only targets that jumped >= 15% and then went quiet (< 2%/wk)
SPLIT = pd.Timestamp("2024-01-01")
FORMS = {"PREM14A", "DEFM14A", "SC TO-T", "SC TO-C", "SC 14D9"}
pd.set_option("display.width", 200)


def events(symbols):
    cols = ["cik", "tickers", "form", "filing_date"]
    F = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{RD}/sec/filings_20*.parquet"))]
                  + [pd.read_parquet(f"{RD}/sec/filings_delisted.parquet", columns=cols)], ignore_index=True)
    F = F[F.form.isin(FORMS)].copy()
    F["ticker"] = F.tickers.fillna("").str.split(",").str[0].str.strip().str.upper()
    F = F[F.ticker.isin(symbols)].sort_values("filing_date")
    out, last = [], {}
    for r in F.itertuples():
        prev = last.get(r.cik)
        last[r.cik] = r.filing_date
        if prev is not None and (r.filing_date - prev).days < 365:
            continue
        out.append((r.ticker, r.filing_date, r.form))
    return pd.DataFrame(out, columns=["ticker", "date", "form"]).drop_duplicates(["ticker", "date"])


def main():
    p = B.load_panel(RD); f = B.features(p)
    E = events(set(p.symbols))
    print(f"{len(E)} deal events, {E.ticker.nunique()} targets; by form: {E.form.value_counts().to_dict()}", flush=True)
    d1c = p.df("d1_close")
    wr = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 3)
    px, adv = f["price"], f["adv"]
    cost = pd.DataFrame(U.half_spread_bps(px.fillna(0).to_numpy(), adv.fillna(0).to_numpy()), index=p.weeks, columns=p.symbols)
    weeks = p.weeks
    C = p.df("close"); r1 = C / C.shift(1) - 1
    hold = pd.DataFrame(False, index=weeks, columns=p.symbols)
    deals = []
    for e in E.itertuples():
        i = weeks.searchsorted(e.date)              # formation week = first Friday on/after the filing
        if i >= len(weeks) - 2:
            continue
        t = weeks[i]
        if not (px.at[t, e.ticker] >= 1 and adv.at[t, e.ticker] >= 2.5e5):
            continue
        if PINNED:
            # a real cash-deal target: jumped on the news, then the price stopped moving (pinned below the offer)
            jump = C.iat[i, C.columns.get_loc(e.ticker)] / C.iat[max(i - 8, 0), C.columns.get_loc(e.ticker)] - 1
            vol = r1[e.ticker].iloc[max(i - 2, 0):i + 1].std()
            if not (jump >= 0.15 and vol < 0.02):
                continue
        r = wr[e.ticker].iloc[i:i + 52]
        n = int(r.notna().cumprod().sum())          # weeks until it stops trading (or 52)
        if n == 0:
            continue
        hold.iloc[i:i + n, hold.columns.get_loc(e.ticker)] = True
        tot = float((1 + r.iloc[:n]).prod() - 1)
        done = n < 52 and i + n < len(weeks) - 2
        deals.append(dict(ticker=e.ticker, entry=t, weeks=n, ret=tot, completed=done, cost_bps=float(cost.at[t, e.ticker])))
    D = pd.DataFrame(deals)
    D["part"] = np.where(D.entry < SPLIT, "dev", "hold")
    print("\nPER DEAL (return from entry to exit, before costs):")
    print(D.groupby("part").agg(deals=("ret", "size"), median_ret=("ret", "median"), mean_ret=("ret", "mean"),
                                 lost_20pct=("ret", lambda x: (x < -0.2).mean()), median_weeks=("weeks", "median"),
                                 completed=("completed", "mean")).round(3).to_string())
    # weekly equal-weight portfolio of open deals
    port, nopen = {}, {}
    prev = set()
    for i, wk in enumerate(weeks[:-2]):
        names = list(hold.columns[hold.loc[wk].to_numpy()])
        if not names:
            prev = set(); continue
        r = wr.loc[wk, names].fillna(0.0)
        cur = set(names)
        tr = (len(cur - prev) + len(prev - cur)) / max(len(cur), 1)
        c = float(cost.loc[wk, names].mean()) / 1e4 * tr
        port[wk] = float(r.mean()) - c
        nopen[wk] = len(names)
        prev = cur
    P = pd.Series(port); N = pd.Series(nopen)
    iwm = None
    try:
        e = pd.read_parquet(f"{RD}/etf/etf_daily.parquet")
        s = e[e.symbol == "IWM"].set_index("date").close
        s.index = pd.to_datetime(s.index)
        iwm = s.resample("W-FRI").last().pct_change().shift(-1).reindex(P.index)
    except Exception as ex:
        print("no IWM:", ex)
    rows = []
    for lab, x in (("dev 2017-23", P[(P.index >= "2017-01-01") & (P.index < SPLIT)]), ("holdout 2024-26", P[P.index >= SPLIT]),
                   ("all", P[P.index >= "2017-01-01"])):
        s = U.st(x)
        b = np.nan
        if iwm is not None:
            j = pd.concat([x, iwm], axis=1).dropna()
            b = np.polyfit(j.iloc[:, 1], j.iloc[:, 0], 1)[0] if len(j) > 20 else np.nan
        rows.append(dict(period=lab, weeks=len(x), deals_open_median=float(N.reindex(x.index).median()),
                         CAGR=s["CAGR"], Sharpe=s["Sharpe"], maxDD=s["maxDD"], beta_IWM=b,
                         worst_week=x.min() * 100, win_weeks=(x > 0).mean() * 100))
    R = pd.DataFrame(rows)
    print("\nPORTFOLIO (equal weight across open deals, net of spread costs):\n" + R.round(2).to_string(index=False))
    print("\nBY YEAR %:\n" + P.groupby(P.index.year).apply(lambda z: ((1 + z).prod() - 1) * 100).round(1).to_string())
    os.makedirs(f"{RD}/results", exist_ok=True)
    tag = "_pinned" if PINNED else "_all"
    D.to_csv(f"{RD}/results/merger_arb_deals{tag}.csv", index=False)
    R.to_csv(f"{RD}/results/merger_arb{tag}.csv", index=False)
    print("::notice::merger arb: " + "; ".join(f"{r.period} CAGR {r.CAGR:+.1f}% Sharpe {r.Sharpe:.2f} beta {r.beta_IWM:.2f}"
                                               for r in R.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
