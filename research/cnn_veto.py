"""
research/cnn_veto.py -- the chart-image CNN as a VETO on our best pickers (Nov 14, 2026; Connor: "pair it up with our
strongest methods of picking, prune the losers, and see the results").

The CNN's skill is almost all in its bottom decile (charts it dislikes trail the median stock by 0.5-1%/week, out of
sample 2022-23 and 2024-26). So: run each picker as before, but a stock may not be bought (or held past the next
rebalance) in a week where it sits in the CNN's bottom 10% (also tested: bottom 20%). Only 2022+ is used, where every
CNN prediction is out of sample (trained on 2017-21).
Pickers:
  OLD_PICK10   live momentum proxy (the old weekly pick-10), weekly, top 10
  PROXY_M30    the same signal, monthly, top 10 with a top-30 buffer
  M12_1        12-1 momentum (the SLOW sleeve's signal), monthly, top 30 with buffer
  FIP_M12_1    12-1 momentum within the steadier "frog in the pan" half (trend_quality.py), monthly, top 30
  CONTRACT     LLM-labelled contract-win 8-Ks (direction >= 1): event level -- abnormal return over the next 5 and 20
               sessions for events whose stock the CNN had in its bottom decile the Friday before vs the rest
Weekly books use the weekly panel (to Oct 2026), spread costs; 2022-23 and 2024-26 shown separately (2024-26 still
lacks most delisted stocks -- which would make the veto look WORSE than it is, not better). Public data only.
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
START, SPLIT = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)


def wk(x):
    return pd.to_datetime(x).dt.to_period("W-FRI").dt.end_time.dt.normalize()


def main():
    P = pd.read_parquet(f"{RD}/results/chart_cnn_preds.parquet")
    P = P[P.week >= START].copy()
    P["pct"] = P.groupby("week").cnn.rank(pct=True)
    P["wk"] = wk(P.week)
    cnn = P.pivot_table(index="wk", columns="symbol", values="pct")
    p = B.load_panel(RD); f = B.features(p)
    base = (f["hist"] >= 53) & ~pd.DataFrame(np.tile(p.is_fund, (len(p.weeks), 1)), index=p.weeks, columns=p.symbols) \
        & ~f["rsplit_52w"].fillna(False).astype(bool)
    m = base & (f["price"] >= 5) & (f["adv"] >= 2e6)
    pw = pd.Series(p.weeks, index=p.weeks).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    cn = cnn.reindex(pw.values).set_axis(p.weeks).reindex(columns=p.symbols)
    rk = lambda x: x.where(m).rank(axis=1, pct=True)
    proxy = (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80
    m12 = f["m12_1"].where(m)
    # FIP on daily bars, sampled at the panel's weeks
    D = pd.concat([pd.read_parquet(x, columns=["symbol", "date", "close"]) for x in sorted(glob.glob(f"{RD}/daily/ohlcv_*.parquet"))])
    C = D.pivot_table(index="date", columns="symbol", values="close").astype("float32"); del D
    r = C.pct_change(fill_method=None).where(lambda x: x.abs() < 0.7)
    s = np.sign(r)
    up = (s > 0).astype("float32").where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    dn = (s < 0).astype("float32").where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    momd = C.shift(21) / C.shift(252) - 1
    fip = (np.sign(momd) * (dn - up))
    fip = fip.groupby(fip.index.to_period("W-FRI")).last()
    fip.index = fip.index.end_time.normalize()
    fipw = fip.reindex(pw.values).set_axis(p.weeks).reindex(columns=p.symbols)
    top = m12.rank(axis=1, pct=True) >= 0.8
    steady = (-fipw).where(top).rank(axis=1, pct=True) >= 0.5
    fipm = m12.where(top & steady)
    small = (f["price"] < 5) | (f["adv"] < 2e6)
    cost = pd.DataFrame(U.half_spread_bps(f["price"].fillna(0).to_numpy(), f["adv"].fillna(0).to_numpy()), index=p.weeks, columns=p.symbols)
    d1c = p.df("d1_close"); wr = (d1c.shift(-2) / d1c.shift(-1) - 1).where(lambda x: x.abs() < 1.5)
    sel = p.weeks >= START
    books = {"OLD_PICK10": (proxy, 10, 1, None), "PROXY_M30": (proxy, 10, 4, 30),
             "M12_1": (m12, 30, 4, 60), "FIP_M12_1": (fipm, 30, 4, 60)}
    rows = []
    for name, (sc, n, k, buf) in books.items():
        for veto in (None, 0.1, 0.2):
            s_ = sc if veto is None else sc.where(~(cn <= veto))
            rr = U.sim(s_.loc[sel], m.loc[sel], wr.loc[sel], cost.loc[sel], small.loc[sel], n=n, k=k, buffer=buf)
            a, b = U.st(rr[rr.index < SPLIT]), U.st(rr[rr.index >= SPLIT])
            rows.append(dict(book=name, veto="none" if veto is None else f"CNN bottom {int(veto * 100)}%",
                             CAGR_2022_23=a["CAGR"], Sharpe_2022_23=a["Sharpe"], DD_2022_23=a["maxDD"],
                             CAGR_2024_26=b["CAGR"], Sharpe_2024_26=b["Sharpe"], DD_2024_26=b["maxDD"]))
        print(f"  {name}", flush=True)
    R = pd.DataFrame(rows)
    # how often the veto bites on each book's would-be picks
    cov = cn.loc[sel].notna().mean(axis=1).mean()
    print(f"\nCNN coverage of panel stocks in 2022+: {cov:.0%} of stock-weeks have a prediction")
    print("\nBOOKS with and without the CNN veto (CAGR %, Sharpe, max drawdown %):")
    print(R.round(2).to_string(index=False))
    # contract wins, event level
    J, E = [], []
    for d in [f"{RD}/llm"] + sorted(glob.glob(f"{RD}/llm_full/part*")):
        if os.path.exists(f"{d}/judgements.parquet") and os.path.exists(f"{d}/events.parquet"):
            J.append(pd.read_parquet(f"{d}/judgements.parquet")); E.append(pd.read_parquet(f"{d}/events.parquet"))
    ev = pd.concat(E).drop_duplicates("accession").merge(pd.concat(J).drop_duplicates("accession"), on="accession")
    ev = ev[ev.ok.fillna(False) & (ev.event_type == "contract_win") & (ev.direction >= 1)].copy()
    ev["t"] = pd.to_datetime(ev.accepted_true)
    ev["wk_prev"] = (ev.t - pd.Timedelta(days=3)).dt.to_period("W-FRI").dt.end_time.dt.normalize()   # last completed week
    ev["cnn_pct"] = [cnn.at[w, s] if (w in cnn.index and s in cnn.columns) else np.nan for w, s in zip(ev.wk_prev, ev.symbol)]
    ev["period"] = np.where(ev.t < SPLIT, "2022-23", "2024-26")
    ev = ev[ev.t >= START]
    print(f"\nCONTRACT WINS: {len(ev)} events since 2022, {ev.cnn_pct.notna().sum()} with a CNN prediction "
          f"(the CNN covers $5+ / $2M+ stocks; many contract filers are smaller)")
    g = ev[ev.cnn_pct.notna()].assign(grp=lambda x: np.where(x.cnn_pct <= 0.1, "CNN bottom 10%", np.where(x.cnn_pct <= 0.2, "CNN 10-20%", "rest")))
    print(g.groupby(["period", "grp"]).agg(events=("ab_5", "size"), abn_5d=("ab_5", "mean"), abn_20d=("ab_20", "mean"),
                                           win_20d=("ab_20", lambda x: (x > 0).mean())).round(3).to_string())
    os.makedirs(f"{RD}/results", exist_ok=True)
    R.to_csv(f"{RD}/results/cnn_veto.csv", index=False)
    v = R[R.veto != "CNN bottom 20%"].pivot(index="book", columns="veto", values=["Sharpe_2022_23", "Sharpe_2024_26"])
    print("::notice::CNN veto (Sharpe none -> bottom-10% veto, 2022-23 | 2024-26): " + "; ".join(
        f"{b} {v.loc[b, ('Sharpe_2022_23', 'none')]:.2f}->{v.loc[b, ('Sharpe_2022_23', 'CNN bottom 10%')]:.2f} | "
        f"{v.loc[b, ('Sharpe_2024_26', 'none')]:.2f}->{v.loc[b, ('Sharpe_2024_26', 'CNN bottom 10%')]:.2f}" for b in v.index))


if __name__ == "__main__":
    main()
