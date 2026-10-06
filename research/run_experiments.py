"""
research/run_experiments.py
===========================
Pre-registered experiment set for the Oct 2026 strategy review. Writes
research/results/report.md plus CSVs. Run:

    python research/run_experiments.py --data <research-data dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest as B  # noqa: E402

COST_BPS = 15.0
N = 10


def hold_k(p, score, mask, k, n=N, cost_bps=COST_BPS):
    """Rebalance every k weeks: buy Monday close of week t+1, sell Friday close of week t+k."""
    d1c, cl = p.df("d1_close"), p.df("close")
    out = {}
    weeks = score.index
    for j in range(0, len(weeks), k):
        wk = weeks[j]
        i = p.weeks.get_loc(wk)          # position in the FULL calendar
        if i + k >= len(p.weeks):
            break
        row = score.loc[wk].where(mask.loc[wk]).dropna()
        entry = d1c.iloc[i + 1]
        exit_ = cl.iloc[i + k]
        r = (exit_ / entry - 1).reindex(row.index)
        r = r.where(r.abs() < 3).dropna()
        row = row.reindex(r.index)
        if len(row) < n:
            continue
        out[wk] = r[row.nlargest(n).index].mean() - 2 * cost_bps / 1e4
    return pd.Series(out, dtype=float)


def stats_k(r, k):
    """Stats for k-week holding returns (annualise with 52/k periods)."""
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) * k / 52
    dd = eq / eq.cummax() - 1
    return {"periods": len(r), "CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
            "vol%": r.std() * np.sqrt(52 / k) * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(52 / k), "maxDD%": dd.min() * 100,
            "win%": (r > 0).mean() * 100}


def fmt(df, digits=2):
    return df.round(digits).to_markdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--last-week", default="2026-10-02")
    ap.add_argument("--start", default="2017-01-06", help="first formation week evaluated")
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    p = B.load_panel(a.data, a.last_week)
    f = B.features(p)
    U = B.universe(p, f, 5.0, 2e6)          # main universe
    U_wide = B.universe(p, f, 1.0, 5e5)     # closer to live filters (price >= $1)
    fwd = B.next_week_returns(p, "designed")
    fwd_broken = B.next_week_returns(p, "broken")
    fwd_stop7 = B.next_week_returns(p, "designed", stop=0.07)
    keep = p.weeks >= pd.Timestamp(a.start)

    def R(df):
        return df.loc[keep]

    # live momentum proxy: same blend weights as pipeline/signals/momentum.py
    # (rs .35, trend .25, breakout .20, vol surge .12, atr .08 -> no vol-surge/atr history,
    #  so renormalise the three we can reproduce)
    rk = lambda x: B.xrank(x, U)
    live_proxy = (0.35 * rk(f["rs_live"]) + 0.25 * rk(f["trend"]) + 0.20 * rk(f["hi52"])) / 0.80
    signals = {
        "r1w (last week winners)": f["r1w"],
        "r4w (last month winners)": f["r4w"],
        "reversal_1w (last week losers)": -f["r1w"],
        "m12_1": f["m12_1"],
        "m6_1": f["m6_1"],
        "m3_1": f["m3_1"],
        "rs_live (40/35/25 blend)": f["rs_live"],
        "hi52 (near 52w high)": f["hi52"],
        "live_momentum_proxy": live_proxy,
        "low_vol (-vol12)": -f["vol12"],
        "m12_1 + reversal": rk(f["m12_1"]) + rk(-f["r1w"]),
    }

    lines = ["# Backtest results\n",
             f"Data: Alpaca SIP adjusted daily bars condensed weekly, formation weeks "
             f"{a.start} → {a.last_week}. Universe: price ≥ $5, 4-wk avg daily $vol ≥ $2M, "
             f"≥ 53 wks history, funds/ETFs excluded, delisted names included while trading. "
             f"Median universe size: {int(R(U).sum(axis=1).median()):,}. "
             f"Costs: {COST_BPS:.0f} bps per side. Top-{N} equal weight.\n"]

    # ── A. signal diagnostics ────────────────────────────────────────────────
    diag = {}
    dec_tbl = {}
    for name, s in signals.items():
        ic = R(B.weekly_ic(s, fwd, U))
        dec = R(B.decile_spread(s, fwd, U))
        diag[name] = {"mean_IC": ic.mean(), "IC_t": ic.mean() / ic.std() * np.sqrt(ic.count()),
                      "IC>0 %": (ic > 0).mean() * 100,
                      "D10 wk%": dec[10].mean() * 100, "D1 wk%": dec[1].mean() * 100,
                      "D10-D1 wk%": (dec[10] - dec[1]).mean() * 100}
        dec_tbl[name] = dec.mean() * 100
    diag = pd.DataFrame(diag).T.sort_values("IC_t", ascending=False)
    diag.to_csv(out / "signal_diagnostics.csv")
    pd.DataFrame(dec_tbl).T.to_csv(out / "deciles.csv")
    lines += ["## A. Does each signal sort next week's returns? (designed timing, gross)\n",
              fmt(diag, 3), "\nD10 = top decile by signal, D1 = bottom. Weekly %, gross of costs.\n"]

    # ── B. top-10 portfolios ─────────────────────────────────────────────────
    spy = R(fwd["SPY"]) if "SPY" in fwd.columns else None
    ew = R(fwd.where(U).mean(axis=1))
    port = {}
    for name, s in signals.items():
        port[name] = B.top_n(R(s), R(fwd), R(U), N, COST_BPS)
    if spy is not None:
        port["SPY (benchmark)"] = spy
    port["Equal-weight universe"] = ew
    tbl = pd.DataFrame({k: B.stats(v) for k, v in port.items()}).T.sort_values("Sharpe", ascending=False)
    tbl.to_csv(out / "portfolios_top10.csv")
    lines += ["## B. Top-10 weekly portfolios, net of costs (designed timing)\n", fmt(tbl)]
    yr = pd.DataFrame({k: B.by_year(v) for k, v in port.items()})
    yr.to_csv(out / "portfolios_by_year.csv")
    lines += ["\n### Calendar-year returns %\n", fmt(yr.T, 1)]

    # ── C. execution variants on the live proxy ──────────────────────────────
    lp = R(live_proxy)
    ex = {
        "designed (Mon close → Fri close)": B.top_n(lp, R(fwd), R(U), N, COST_BPS),
        "broken (Tue open → next Mon open)": B.top_n(lp, R(fwd_broken), R(U), N, COST_BPS),
        "designed + 7% hard stop": B.top_n(lp, R(fwd_stop7), R(U), N, COST_BPS),
        "designed, wide universe ($1/$0.5M)": B.top_n(R(B.xrank(f['rs_live'], U_wide) * .35 / .8
                                                         + B.xrank(f['trend'], U_wide) * .25 / .8
                                                         + B.xrank(f['hi52'], U_wide) * .20 / .8),
                                                       R(fwd), R(U_wide), N, COST_BPS),
        "designed, top-30": B.top_n(lp, R(fwd), R(U), 30, COST_BPS),
    }
    t = pd.DataFrame({k: B.stats(v) for k, v in ex.items()}).T
    t.to_csv(out / "execution_variants.csv")
    lines += ["\n## C. Live-momentum proxy: execution variants\n", fmt(t)]

    # ── D. holding period + regime filter ────────────────────────────────────
    hp = {}
    for name in ["live_momentum_proxy", "m12_1", "rs_live (40/35/25 blend)", "hi52 (near 52w high)"]:
        for k in (1, 4, 13):
            hp[(name, k)] = stats_k(hold_k(p, signals[name].loc[keep], U.loc[keep], k), k)
    t = pd.DataFrame(hp).T
    t.index.names = ["signal", "hold_weeks"]
    t.to_csv(out / "holding_period.csv")
    lines += ["\n## D. Holding period (rebalance every k weeks)\n", fmt(t)]

    if "SPY" in p.symbols:
        C = p.df("close")["SPY"]
        risk_on = (C > C.rolling(40).mean()).reindex(p.weeks)
        base = ex["designed (Mon close → Fri close)"]
        filt = base.where(risk_on.reindex(base.index).fillna(False), 0.0)
        t = pd.DataFrame({"no filter": B.stats(base), "cash when SPY < 40w SMA": B.stats(filt)}).T
        lines += ["\n## E. Market regime filter (live proxy, weekly)\n", fmt(t)]

    # ── F. recent window ─────────────────────────────────────────────────────
    recent = {k: v.loc["2025-10-01":] for k, v in port.items()}
    t = pd.DataFrame({k: B.stats(v) for k, v in recent.items()}).T.sort_values("Sharpe", ascending=False)
    lines += ["\n## F. Last 12 months only (Oct 2025 → Sep 2026)\n", fmt(t)]

    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
