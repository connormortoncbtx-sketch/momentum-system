"""
research/combo_estimate.py -- first combined-Sharpe estimate of the candidate edges (Nov 14, 2026; Connor: "Do it" --
measure how independent the edges are and what stacking them could give, before the Jan 5 review).

Weekly return streams (all from saved research outputs; no new fitting):
  FIPV      FIP-steady 12-1 momentum top 30, CNN v2 collapse veto (veto bites from 2022; before that plain FIP 12-1)
            -- research/cnn_veto.py VETO_START=2017 CNN_VERSION=2 (delisted-inclusive panel)
  CONTRACT  LLM contract-win book, daily -> weekly (results/contract_book_daily.csv, 2024+)
  CONDOR    2pm SPY + QQQ iron condor (k=1, stop at 2x credit), account return = 2% of the account at risk per day,
            split across the two (results/options_series_daily.parquet)
  MERGER    merger arbitrage, cash-deal signature (results/merger_arb_weekly_pinned.csv) -- 2017-23 ONLY (2024-26
            targets are missing from the filing set)
  ETF_EQ    the ETF sleeve's 41 ETFs held at equal risk (10% vol target each, x4/N, capped 1x, month-end weights)
  ETF_TREND the live 63-day trend rule on the same ETFs (reference)
Windows: A = 2024-02 .. 2026-08 (FIPV, CONTRACT, CONDOR, ETF_EQ); B = 2018 .. 2023 (FIPV, CONDOR, MERGER, ETF_EQ).
Per window: each stream's annual return / vol / Sharpe, the correlation matrix, an equal-risk combination (every stream
scaled to the same volatility), its Sharpe; the same with every stream's mean HALVED (the usual live haircut after a
search); and the half-Kelly growth rate those Sharpes would allow (0.75 x S^2 / 2 per year, log; at volatility S / 2).
"""
import os

import numpy as np
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
SLEEVE = ['BITO', 'COPX', 'DBA', 'DBC', 'DIA', 'EEM', 'EFA', 'EWJ', 'FXB', 'FXE', 'FXI', 'FXY', 'GDX', 'GLD', 'HYG',
          'IBIT', 'IEF', 'IWM', 'LQD', 'QQQ', 'SHY', 'SLV', 'SMH', 'SPY', 'TIP', 'TLT', 'UNG', 'URA', 'USO', 'UUP',
          'VNQ', 'XBI', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
pd.set_option("display.width", 200)


def weekly(daily):
    w = (1 + daily.fillna(0)).groupby(daily.index.to_period("W-FRI")).prod().sub(1)
    w.index = w.index.to_timestamp(how="end").normalize()
    return w


def etf_books():
    e = pd.read_parquet(f"{RD}/etf/etf_daily.parquet")
    e["date"] = pd.to_datetime(e.date)
    px = e[e.symbol.isin(SLEEVE)].pivot_table(index="date", columns="symbol", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    me = sorted({px.index[px.index <= m][-1] for m in px.resample("ME").last().index if (px.index <= m).any()})
    out = {}
    for name, sig in (("ETF_EQ", lambda d: px.loc[d].notna().astype(float)),
                      ("ETF_TREND", lambda d: (px.loc[d] / px.shift(63).loc[d] - 1 > 0).astype(float))):
        w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for d0, d1 in zip(me[:-1], me[1:]):
            t = (sig(d0) * (0.10 / vol.loc[d0]).clip(upper=2.0)).fillna(0)
            t = t / max((t != 0).sum(), 1) * 4
            if t.sum() > 1:
                t = t / t.sum()
            w.loc[(w.index > d0) & (w.index <= d1)] = t.values
        out[name] = (w * ret.fillna(0)).sum(axis=1)
    return {k: weekly(v) for k, v in out.items()}


def condor():
    o = pd.read_parquet(f"{RD}/results/options_series_daily.parquet")
    x = o[(o.entry == "14:00") & (o.struct == "condor") & (o.k == 1.0) & (o.wing == 0) & (o.exit == "sl2x") & (o.src == "quotes")]
    d = x.groupby("d").ror.mean() * 0.02
    d.index = pd.to_datetime(d.index)
    return weekly(d)


def stats(r):
    return pd.Series(dict(ann_return=r.mean() * 52 * 100, ann_vol=r.std() * np.sqrt(52) * 100,
                          Sharpe=r.mean() / r.std() * np.sqrt(52), weeks=len(r)))


def combo(df, haircut=1.0):
    adj = df - df.mean() * (1 - haircut)
    z = adj / adj.std()
    c = z.mean(axis=1)
    return c.mean() / c.std() * np.sqrt(52)


def main():
    S = {}
    v = pd.read_parquet(f"{RD}/results/cnn_veto_series_v2.parquet")
    S["FIPV"] = v["FIP_M12_1|10"]
    c = pd.read_csv(f"{RD}/results/contract_book_daily.csv", parse_dates=["date"]).set_index("date").contract_book
    S["CONTRACT"] = weekly(c)
    S["CONDOR"] = condor()
    mg = pd.read_csv(f"{RD}/results/merger_arb_weekly_pinned.csv", index_col=0, parse_dates=True).merger_arb
    S["MERGER"] = mg[mg.index < "2024-01-01"]
    S.update(etf_books())
    for k in S:
        S[k].index = pd.to_datetime(S[k].index).to_period("W-FRI").to_timestamp(how="end").normalize()
    D = pd.DataFrame(S)
    windows = {"A 2024-02..2026-08": (["FIPV", "CONTRACT", "CONDOR", "ETF_EQ"], "2024-02-01", "2026-08-31"),
               "B 2018..2023": (["FIPV", "CONDOR", "MERGER", "ETF_EQ"], "2018-01-01", "2023-12-31")}
    rows = []
    for name, (cols, a, b) in windows.items():
        W = D.loc[a:b, cols + ["ETF_TREND"]].dropna(subset=cols)
        print(f"\n=== WINDOW {name}: {len(W)} weeks ===")
        print(W.apply(stats).T.round(2).to_string())
        print("\ncorrelations (weekly):\n" + W[cols].corr().round(2).to_string())
        for hc, lab in ((1.0, "backtest"), (0.5, "means halved")):
            s = combo(W[cols], hc)
            g = 0.75 * s * s / 2
            rows.append(dict(window=name, version=lab, combined_Sharpe=s, half_kelly_vol=s / 2 * 100,
                             half_kelly_growth_pct=(np.exp(g) - 1) * 100,
                             per_day_pct=(np.exp(g / 252) - 1) * 100))
    R = pd.DataFrame(rows)
    print("\nEQUAL-RISK COMBINATION and what half-Kelly sizing would allow (log growth 0.75 x S^2/2):")
    print(R.round(2).to_string(index=False))
    os.makedirs(f"{RD}/results", exist_ok=True)
    D.to_csv(f"{RD}/results/combo_streams_weekly.csv")
    R.to_csv(f"{RD}/results/combo_estimate.csv", index=False)


if __name__ == "__main__":
    main()
