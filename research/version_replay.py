"""
research/version_replay.py -- would reverting to an earlier version of the live scoring
stack have mattered? (Oct 16, 2026)

For each snapshot (git commit) of the live system: its LightGBM model pickle, its
config/weights.json and its SIGNAL_FEATURES list. Every week logged in
data/performance_log.csv is re-scored with that snapshot:
  * sub-signals as logged that week; *_adj = signal x weight x regime multiplier using
    the SNAPSHOT's weights and that week's logged regime
  * coverage gate (>= 10 of 14 sub-signals non-neutral), price >= $1, EV veto (bottom
    20% of logged ev_pct_rank), rank by alpha with sig_momentum as tie-break
  * top 10 -> mean forward_return_1w (as logged, same return for every snapshot)
Only weeks AFTER a snapshot's date are out of sample for it (the model was trained on
earlier weeks). Benchmarks: the picks actually made (logged composite_rank <= 10) and the
mean of all logged names that week.
"""
import ast, io, json, pickle, subprocess, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
SNAPS = {"Apr 20 (live start)": "2026-04-20T00:00", "May 08": "2026-05-08T23:59", "May 22 (peak)": "2026-05-22T23:59",
         "May 29": "2026-05-29T00:00", "Jun 13": "2026-06-13T23:59", "Aug 01": "2026-08-01T23:59", "current": None}
SUBS = ["sig_momentum_rs", "sig_momentum_trend", "sig_momentum_vol_surge", "sig_momentum_breakout",
        "sig_catalyst_earnings", "sig_catalyst_insider", "sig_catalyst_analyst", "sig_fund_growth",
        "sig_fund_quality", "sig_fund_profitability", "sig_fund_value", "sig_sentiment_news",
        "sig_sentiment_analyst", "sig_sentiment_short"]


def git(c, path, binary=False):
    out = subprocess.run(["git", "show", f"{c}:{path}"], capture_output=True).stdout
    return out if binary else out.decode()


def snapshot(before):
    c = subprocess.run(["git", "rev-list", "-1", f"--before={before}", "main"] if before else
                       ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    src = git(c, "pipeline/04_model.py")
    i = src.index("SIGNAL_FEATURES = [")
    feats = ast.literal_eval(src[i + len("SIGNAL_FEATURES = "): src.index("]", i) + 1])
    return c, feats, json.loads(git(c, "config/weights.json")), pickle.load(io.BytesIO(git(c, "models/lgbm_model.pkl", True)))


def score_week(w, feats, weights, model):
    w = w.copy()
    base, regime = weights["signal_weights"], w.regime.iloc[0]
    rm = weights["regime_multipliers"].get(regime, {})
    for s in ("momentum", "catalyst", "fundamentals", "sentiment"):
        w[f"sig_{s}_adj"] = w[f"sig_{s}"] * base.get(s, 0.0) * rm.get(s, 1.0)
    w["sig_technicals_adj"] = w["sig_momentum_trend"] * base.get("technicals", 0.15) * rm.get("technicals", 1.0)
    n = getattr(model, "n_features_in_", len(feats))
    use = ["sig_momentum", "sig_catalyst", "sig_fundamentals", "sig_sentiment"] if n <= 4 else [f for f in feats if f in w.columns]
    if len(use) != n:
        return None
    w["alpha"] = model.predict_proba(w[use].fillna(0.5).values)[:, 1]
    real = (w[SUBS].notna() & (w[SUBS] != 0.5)).sum(axis=1)
    ok = (real >= 10) & (w.prior_close.fillna(0) >= 1)
    if w.ev_pct_rank.notna().any():
        ok &= ~(w.ev_pct_rank < 0.20)
    c = w[ok].sort_values(["alpha", "sig_momentum"], ascending=False)
    return c.head(10)


def main():
    p = pd.read_csv("data/performance_log.csv", low_memory=False, parse_dates=["week_of"])
    p = p[p.sig_momentum_rs.notna() & p.forward_return_1w.notna() & p.regime.notna()]
    weeks = sorted(w for w, g in p.groupby("week_of") if len(g) > 500 and g.forward_return_1w.notna().mean() > 0.5)
    print("weeks:", [str(w.date()) for w in weeks])
    res = {}
    for name, before in SNAPS.items():
        c, feats, weights, model = snapshot(before)
        out = {}
        for wk in weeks:
            top = score_week(p[p.week_of == wk], feats, weights, model)
            if top is not None:
                out[wk] = top.forward_return_1w.clip(-0.9, 3).mean()
        res[name] = pd.Series(out)
        print(f"{name:20s} {c[:7]} features {len(feats)} weights {weights['signal_weights']}")
    res = pd.DataFrame(res)
    act = p[p.composite_rank <= 10].groupby("week_of").forward_return_1w.mean()
    allm = p.groupby("week_of").forward_return_1w.apply(lambda x: x.clip(-0.9, 3).mean())
    res["actual picks"] = act.reindex(res.index); res["all names"] = allm.reindex(res.index)
    pd.set_option("display.width", 220)
    print("\nWEEKLY top-10 forward return (%), each snapshot replayed on every logged week")
    print((res * 100).round(1).to_string())
    print("\nAFTER EACH SNAPSHOT'S DATE ONLY (out of sample): mean weekly % / cumulative % / weeks")
    for name, before in SNAPS.items():
        d = pd.Timestamp(before[:10]) if before else pd.Timestamp("2026-10-10")
        r = res.loc[res.index >= d, name].dropna()
        b = res.loc[r.index, "all names"]
        if len(r):
            print(f"  {name:20s} mean {r.mean() * 100:5.2f}  cum {((1 + r).prod() - 1) * 100:6.1f}  vs all-names mean "
                  f"{b.mean() * 100:5.2f}  weeks {len(r)}")
    print("\nWHOLE PERIOD (in + out of sample): mean weekly %")
    print((res.mean() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
