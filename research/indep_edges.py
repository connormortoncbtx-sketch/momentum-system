"""
research/indep_edges.py -- three candidate edges outside long-stock exposure (Nov 15, 2026; Connor: "Any other
independent edges" -> "Let's look").
 1. TREASURY AUCTION CYCLE (Lou, Yan & Zhang 2013): bond prices dip in the days before coupon auctions (dealers make
    room for supply) and recover after. Coupon auctions of 5y+ notes and bonds (results/fetch_auctions.py). Event
    study of IEF / TLT excess over SHY around auction days; long-only rules: hold TLT only in the 5 days AFTER
    auctions (else SHY); hold TLT except in the 5 days BEFORE auctions.
 2. CRYPTO TREND (the live crypto sleeve; strategy_stack.py CRYPTO, weekly).
 3. NON-STOCK TREND: the ETF sleeve's trend rule and sizing on its bond / gold / commodity / currency ETFs only.
Then each stream's Sharpe and weekly correlations with FIPV and CONDOR (results/combo_streams_weekly.csv), and the
equal-risk combination with and without the new streams, 2018-23 (crypto from mid-2021) and 2024-26. Public data only.
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SPLIT = pd.Timestamp("2024-01-01")
NONEQ = ["IEF", "TLT", "SHY", "TIP", "LQD", "HYG", "GLD", "SLV", "USO", "UNG", "DBC", "DBA", "UUP", "FXE", "FXY", "FXB"]
pd.set_option("display.width", 200)


def weekly(d):
    w = (1 + d.fillna(0)).groupby(d.index.to_period("W-FRI")).prod() - 1
    w.index = w.index.to_timestamp(how="end").normalize()
    return w


def sh(r, ppy=52):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(ppy) if len(r) > 10 and r.std() > 0 else np.nan


def main():
    e = pd.read_parquet(f"{RD}/trend/etf_daily.parquet")      # the trend sleeve's file holds the bond/commodity ETFs
    e["date"] = pd.to_datetime(e.timestamp).dt.tz_localize(None).dt.normalize() if e.timestamp.dt.tz is None else \
        pd.to_datetime(e.timestamp).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    px = e.pivot_table(index="date", columns="symbol", values="close").sort_index()
    print(f"ETF closes {px.index.min().date()}..{px.index.max().date()}, {px.shape[1]} symbols")
    ret = px.pct_change(fill_method=None)
    # ---- 1. auction cycle
    A = pd.read_csv(f"{RD}/macro/treasury_auctions.csv", parse_dates=["auction_date"])
    term_years = A.security_term.str.extract(r"(\d+)-Year")[0].astype(float)
    A = A[A.security_type.isin(["Note", "Bond"]) & (term_years >= 5)]
    days = ret.index
    adays = sorted({days[days.searchsorted(d)] for d in A.auction_date if d <= days[-1]})
    pos = days.get_indexer(adays)
    xs = {k: (ret[k] - ret["SHY"]).fillna(0) for k in ("IEF", "TLT")}
    print(f"1. AUCTION CYCLE: {len(adays)} coupon-auction days (5y+), {A.auction_date.min().date()}..{A.auction_date.max().date()}")
    rows = []
    for k, x in xs.items():
        xv = x.to_numpy()
        for lab, sel in (("2016-23", np.array([days[p] < SPLIT for p in pos])), ("2024-26", np.array([days[p] >= SPLIT for p in pos]))):
            ps = pos[sel]
            pre = [xv[p - 5:p].sum() for p in ps if p >= 5]                 # 5 days before the auction-day close
            post = [xv[p + 1:p + 6].sum() for p in ps if p + 6 < len(xv)]   # 5 days after
            day = [xv[p] for p in ps]
            se = lambda a: np.std(a) / np.sqrt(len(a))
            rows.append(dict(etf=k, period=lab, auctions=len(ps), pre5_bp=np.mean(pre) * 1e4, pre_t=np.mean(pre) / se(pre),
                             auction_day_bp=np.mean(day) * 1e4, post5_bp=np.mean(post) * 1e4, post_t=np.mean(post) / se(post)))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    inpost = pd.Series(0.0, index=days); inpre = pd.Series(0.0, index=days)
    for p in pos:
        inpost.iloc[p + 1:p + 6] = 1.0
        inpre.iloc[max(p - 4, 0):p + 1] = 1.0                              # the 5 days ending on the auction day
    rules = {"TLT buy & hold (vs SHY)": pd.Series(1.0, index=days), "TLT only 5d after auctions": inpost,
             "TLT except 5d before auctions": 1 - inpre}
    S = {}
    print("\n   long-only rules, excess over SHY (Sharpe, annual excess %):")
    for name, w in rules.items():
        r = (w.shift(1).fillna(0) * xs["TLT"] - w.diff().abs().fillna(0) * 2e-4)
        wk = weekly(r)
        S[name] = wk
        print(f"   {name:32s} 2016-23 Sharpe {sh(wk[wk.index < SPLIT]):+.2f} ({wk[wk.index < SPLIT].mean() * 5200:+.1f}%) | "
              f"2024-26 {sh(wk[wk.index >= SPLIT]):+.2f} ({wk[wk.index >= SPLIT].mean() * 5200:+.1f}%)")
    # ---- 3. non-stock trend
    sub = px[[c for c in NONEQ if c in px]]
    rr = sub.pct_change(fill_method=None)
    vol = rr.rolling(60, min_periods=40).std() * np.sqrt(252)
    me = sorted({sub.index[sub.index <= m][-1] for m in sub.resample("ME").last().index if (sub.index <= m).any()})
    w = pd.DataFrame(0.0, index=sub.index, columns=sub.columns)
    for d0, d1 in zip(me[:-1], me[1:]):
        t = ((sub.loc[d0] / sub.shift(63).loc[d0] - 1 > 0).astype(float) * (0.10 / vol.loc[d0]).clip(upper=2.0)).fillna(0)
        t = t / max((t != 0).sum(), 1) * 4
        if t.sum() > 1:
            t = t / t.sum()
        w.loc[(w.index > d0) & (w.index <= d1)] = t.values
    nonq = weekly((w * rr.fillna(0)).sum(axis=1))
    # ---- 2. crypto + combine
    C = pd.read_csv(f"{RD}/results/combo_streams_weekly.csv", index_col=0, parse_dates=True)
    try:
        cr = pd.read_csv(f"{RD}/results/stack_books_weekly_honest.csv", index_col=0, parse_dates=True)["CRYPTO"]
        cr.index = cr.index.to_period("W-FRI").to_timestamp(how="end").normalize()
    except Exception as ex:
        print("crypto series missing:", ex); cr = pd.Series(dtype=float)
    D = pd.DataFrame({"FIPV": C.FIPV, "CONDOR": C.CONDOR, "ETF_EQ": C.ETF_EQ, "CRYPTO": cr, "NONSTOCK_TREND": nonq,
                      "AUCTION_POST": S["TLT only 5d after auctions"], "AUCTION_AVOID": S["TLT except 5d before auctions"]})
    D.to_csv(f"{RD}/results/indep_edges_weekly.csv")
    for lab, a, b in (("2018-23", "2018-01-01", "2023-12-31"), ("2021-07..2023", "2021-07-01", "2023-12-31"), ("2024-26", "2024-02-01", "2026-08-31")):
        W = D.loc[a:b]
        print(f"\n=== {lab}: Sharpe per stream ===\n" + W.apply(sh).round(2).to_string())
        print("correlations with FIPV / CONDOR:\n" + W.corr()[["FIPV", "CONDOR"]].round(2).to_string())
        for cols in (["FIPV", "CONDOR"], ["FIPV", "CONDOR", "NONSTOCK_TREND"], ["FIPV", "CONDOR", "AUCTION_POST"],
                     ["FIPV", "CONDOR", "CRYPTO"], ["FIPV", "CONDOR", "NONSTOCK_TREND", "CRYPTO", "AUCTION_POST"]):
            X = W[cols].dropna()
            if len(X) < 26:
                continue
            z = (X / X.std()).mean(axis=1)
            print(f"   combo {'+'.join(cols):55s} weeks {len(X):3d} Sharpe {sh(z):.2f}")


if __name__ == "__main__":
    main()
