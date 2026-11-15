"""
research/novel_eval.py -- evaluation of the downloaded novel signals (Nov 15, 2026; Connor: "test everything").

PRE-REGISTERED (written before any of these data were seen). Periods: dev 2018-23, holdout 2024-26. Monthly tests use
the universe and monthly returns of novel_stock_signals.py ($5, $2M/day, no funds, delisted included, 15 bps/side).
 #3 LAZY PRICES  signal = cosine similarity of the latest 10-K to the prior one (public by its acceptance time, within
                 the last 12 months). (a) quintile spread Q5 (most similar) - Q1 next month: t > 2 in both periods;
                 (b) VETO: drop Q1 'changers' from the 12-1 top-decile book: Sharpe up in both periods.
 #8 ATTENTION    abnormal Wikipedia views = log(last 5 days' views / (5 x median daily views over the prior 250 days)),
                 at each Friday; (a) decile spread top - bottom, next-week return: t > 2 in both periods, same sign;
                 (b) VETO: drop the top attention decile from the 12-1 top-decile monthly book: Sharpe up in both periods.
 #7 SUPPLY CHAIN supplier's signal = mean last-month return of its named major customers (10-Ks accepted within the
                 prior 15 months); quintile spread top - bottom, next-month supplier return: t > 2 in both periods.
 #2 FED ODDS     2pm and 3pm condor return on risk on Fed days vs other days; Fed days whose 13:00 favourite is < 0.80
                 vs >= 0.80. Reported only (expected ~40 Fed days: no pass/fail claim is possible at this size).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as B  # noqa: E402
import novel_stock_signals as N  # noqa: E402

RD = os.environ.get("RESEARCH_DATA", "research_data")
PER = (("2018-23", "2018-01", "2023-12"), ("2024-26", "2024-01", "2026-12"))
pd.set_option("display.width", 220)


def tstat(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 5 else np.nan


def quint_spread(score, univ, fwd, q=0.2):
    out = {}
    for m in score.index:
        s = score.loc[m].where(univ.loc[m]).dropna()
        if len(s) < 100:
            continue
        rk, f = s.rank(pct=True), fwd.loc[m]
        out[m] = f.reindex(rk[rk > 1 - q].index).mean() - f.reindex(rk[rk <= q].index).mean()
    return pd.Series(out)


def veto_book(m12, univ, fwd, bad):
    """12-1 top decile (as novel_stock_signals.book) with and without names flagged in `bad` (same shape, bool)."""
    return N.book(m12, univ, fwd), N.book(m12.where(~bad.reindex_like(m12).fillna(False).astype(bool)), univ, fwd)


def monthly_panel():
    D = N.load()
    D["dv"] = D.close * D.volume
    D["dv20"] = D.groupby("symbol", sort=False).dv.transform(lambda x: x.rolling(20, min_periods=15).mean())
    D["ym"] = D.date.dt.to_period("M")
    M = D.groupby(["symbol", "ym"]).agg(close=("close", "last"), dv20=("dv20", "last"))
    C = M.close.unstack(0).sort_index()
    names = pd.read_parquet(f"{RD}/assets.parquet").set_index("symbol")["name"].fillna("").astype(str)
    fund = pd.Series({s: B.is_fund_name(names.get(s, "")) for s in C.columns})
    univ = (C >= 5) & (M.dv20.unstack(0).reindex_like(C) >= 2e6) & (C.notna().cumsum() >= 13) & ~fund.reindex(C.columns).fillna(False)
    ret = C.pct_change(fill_method=None)
    return D, C, ret, ret.shift(-1), univ, C.shift(1) / C.shift(12) - 1


def sym_of(tickers, cols):
    for t in str(tickers).split(","):
        t = t.strip().upper().replace("-", ".")
        if t in cols:
            return t
    return None


def asof_monthly(ev, cols, months, col, max_age_m):
    """ev: rows with symbol, ym (period public), value col -> month x symbol table of the latest value within max_age."""
    ev = ev.dropna(subset=["symbol", col]).sort_values("ym")
    T = ev.pivot_table(index="ym", columns="symbol", values=col, aggfunc="last").reindex(months)
    return T.ffill(limit=max_age_m).reindex(columns=cols)


def main():
    rows = []
    D, C, ret, fwd, univ, m12 = monthly_panel()
    cols, months = C.columns, C.index
    # ---- #3 lazy prices
    fs = sorted(glob.glob(f"{RD}/sec/tenk_sim_*.parquet"))
    if fs:
        S = pd.concat(map(pd.read_parquet, fs), ignore_index=True)
        S["symbol"] = [sym_of(t, cols) for t in S.tickers]
        S["ym"] = pd.to_datetime(S.accepted_et.fillna(S.filing_date)).dt.to_period("M")
        print(f"#3: {len(S):,} 10-Ks, {S.sim_cos.notna().sum():,} with a prior; mapped {S.symbol.notna().mean():.0%}", flush=True)
        sim = asof_monthly(S, cols, months, "sim_cos", 11)
        rsim = asof_monthly(S, cols, months, "risk_cos", 11)
        for name, sc in (("10-K similarity", sim), ("risk-factor similarity", rsim)):
            sp = quint_spread(sc, univ, fwd)
            q1 = sc.where(univ).rank(axis=1, pct=True) <= 0.2
            base, vet = veto_book(m12, univ, fwd, q1)
            for lab, a, z in PER:
                rows.append(dict(test=f"#3 {name}: Q5-Q1 spread", period=lab, value_bp=sp.loc[a:z].mean() * 1e4, t=tstat(sp.loc[a:z]), n=len(sp.loc[a:z])))
                rows.append(dict(test=f"#3 {name}: 12-1 book Sharpe base -> veto Q1", period=lab,
                                 value_bp=np.nan, t=np.nan, n=len(base.loc[a:z]),
                                 base_sharpe=N.stats(base.loc[a:z])["sharpe"], veto_sharpe=N.stats(vet.loc[a:z])["sharpe"]))
    # ---- #8 attention
    if os.path.exists(f"{RD}/wiki/pageviews.parquet"):
        P = pd.read_parquet(f"{RD}/wiki/pageviews.parquet")
        V = P.pivot_table(index="date", columns="symbol", values="views").sort_index()
        base = V.rolling(250, min_periods=120).median().shift(5)
        last5 = V.rolling(5, min_periods=4).sum()
        ab = np.log((last5 + 1) / (5 * base + 1))
        fri = ab.index[ab.index.dayofweek == 4]
        A = ab.loc[fri]
        close = D.pivot_table(index="date", columns="symbol", values="close").reindex(columns=A.columns)
        wclose = close.reindex(close.index.union(fri)).ffill().loc[fri]
        wfwd = wclose.shift(-1) / wclose - 1
        dv = D.pivot_table(index="date", columns="symbol", values="dv20").reindex(columns=A.columns).reindex(close.index.union(fri)).ffill().loc[fri]
        wun = (wclose >= 5) & (dv >= 2e6)
        out = {}
        for d in A.index:
            s = A.loc[d].where(wun.loc[d]).dropna()
            if len(s) < 100:
                continue
            rk = s.rank(pct=True)
            out[d] = wfwd.loc[d].reindex(rk[rk >= 0.9].index).mean() - wfwd.loc[d].reindex(rk[rk <= 0.1].index).mean()
        sp = pd.Series(out)
        Am = A.groupby(A.index.to_period("M")).last().reindex(months).reindex(columns=cols)
        top = Am.where(univ).rank(axis=1, pct=True) >= 0.9
        b0, b1 = veto_book(m12, univ, fwd, top)
        print(f"#8: pageviews {P.symbol.nunique():,} symbols", flush=True)
        for lab, a, z in PER:
            x = sp[(sp.index >= a) & (sp.index <= pd.Period(z).end_time)]
            rows.append(dict(test="#8 attention: top-bottom decile, next week", period=lab, value_bp=x.mean() * 1e4, t=tstat(x), n=len(x)))
            rows.append(dict(test="#8 attention: 12-1 book Sharpe base -> veto top decile", period=lab, value_bp=np.nan, t=np.nan,
                             n=len(b0.loc[a:z]), base_sharpe=N.stats(b0.loc[a:z])["sharpe"], veto_sharpe=N.stats(b1.loc[a:z])["sharpe"]))
    # ---- #7 supply chain
    fs = sorted(glob.glob(f"{RD}/sec/customer_links_*.parquet"))
    if fs:
        L = pd.concat(map(pd.read_parquet, fs), ignore_index=True)
        L["supplier"] = [sym_of(t, cols) for t in L.tickers]
        L = L.dropna(subset=["supplier"])
        L = L[L.customer.isin(cols) & (L.supplier != L.customer)]
        L["ym"] = pd.to_datetime(L.accepted_et.fillna(L.filing_date)).dt.to_period("M")
        print(f"#7: {len(L):,} links, {L.supplier.nunique():,} suppliers, {L.customer.nunique():,} customers", flush=True)
        sig = pd.DataFrame(np.nan, index=months, columns=cols)
        for m in months:
            act = L[(L.ym <= m) & (L.ym > m - 15)]
            if act.empty:
                continue
            cr = ret.loc[m].reindex(act.customer).to_numpy()
            s = pd.Series(cr, index=act.supplier.to_numpy()).groupby(level=0).mean()
            sig.loc[m, s.index] = s.values
        sp = quint_spread(sig, univ, fwd)
        for lab, a, z in PER:
            rows.append(dict(test="#7 customer momentum: Q5-Q1 supplier spread", period=lab, value_bp=sp.loc[a:z].mean() * 1e4,
                             t=tstat(sp.loc[a:z]), n=len(sp.loc[a:z])))
    # ---- #2 Fed odds on condor days
    if os.path.exists(f"{RD}/results/kalshi_fed_events.csv"):
        K = pd.read_csv(f"{RD}/results/kalshi_fed_events.csv", parse_dates=["date"])
        O = pd.read_parquet(f"{RD}/results/options_series_daily.parquet")
        O["d"] = pd.to_datetime(O.d)
        for entry, lab in (("14:00", "2pm"), ("15:00", "3pm")):
            X = O[(O.entry == entry) & (O.struct == "condor") & (O.k.astype(float) == 1.0) & (O.wing.astype(int) == 0) & (O.exit == "hold")]
            X = X[X.d >= K.date.min()]
            fed = X.d.isin(K.date)
            unc = X.d.isin(K[K.max_prob < 0.8].date)
            rows.append(dict(test=f"#2 {lab} condor RoR: Fed days vs other", period=f"{K.date.min().date()}+",
                             value_bp=(X.ror[fed].mean() - X.ror[~fed].mean()) * 1e4, n=int(fed.sum()),
                             fed_mean=X.ror[fed].mean() * 100, other_mean=X.ror[~fed].mean() * 100,
                             uncertain_mean=X.ror[unc].mean() * 100, uncertain_n=int(unc.sum()),
                             confident_mean=X.ror[fed & ~unc].mean() * 100))
    R = pd.DataFrame(rows)
    R.to_csv(f"{RD}/results/novel_eval.csv", index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
