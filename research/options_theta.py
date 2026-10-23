"""
research/options_theta.py -- the same-day condor on REAL 2016-2024 option prices from ThetaData (Oct 23, 2026)

The modelled test (condor_synth.py) said the condor LOST in 2016-23 (SPY -2.0%/trade, QQQ -4.4%) while the real-price
backtest (options_premium.py, Alpaca, Feb 2024+) made +3.1% / +5.3%. This settles it with real prices, same rules:
  entry 10:00 ET at each leg's 10:00-10:15 one-minute VWAP, implied move M = ATM call + ATM put, condor k = 1 (shorts
  round(S0 -/+ M)), wings SPY $5 / QQQ 0.8% (min $1); an untraded insurance wing is priced 1c (as the shadow does), an
  untraded short leg skips the day; credit >= 60% of width skips (bad data); $0.01 and $0.03 per leg; settle at the
  4pm close (spot from the Alpaca 15-minute bars in results/<und>_15m_raw_2016.parquet).
Only days that had a same-day expiry (SPY: Mon/Wed/Fri from ~2016, every day from 2022; QQQ: later).
Data: Theta Terminal v3 (local, started by .github/workflows/theta_run.yml), one request per day per underlying for the
whole chain's 1-minute bars 10:00-10:15; per-contract VWAP / volume cached in research_data/theta/<und>_chain.parquet
(resumable: re-runs skip cached days). MODE=probe only tests the API formats and stops.
Outputs: results/options_theta_trades_<und>.parquet, results/options_theta_summary.csv, annotations.
"""
import datetime as dt, io, json, os, sys, time
import numpy as np, pandas as pd
import requests

RD = os.environ.get("RESEARCH_DATA", "research_data")
BASE = os.environ.get("THETA_URL", "http://127.0.0.1:25503/v3")
API = os.environ.get("THETA_API", "v3")                         # the "v3" jar served today runs the v2 REST API
CACHE_DIR = f"{RD}/theta"; OUT = f"{RD}/results"
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
START, END = dt.date(2016, 1, 1), dt.date(2024, 3, 31)        # overlaps Alpaca's Feb-Mar 2024 for validation
note = lambda m: print(f"::notice::{m}", flush=True)
warn = lambda m: print(f"::warning::{m}", flush=True)


def get(path, **q):
    q.setdefault("format", "json")
    for i in range(5):
        try:
            r = requests.get(f"{BASE}{path}", params=q, timeout=120, headers={"Connection": "close"})
        except requests.RequestException as e:
            time.sleep(3 * (i + 1)); last = str(e); continue
        if r.status_code == 472:                                  # Theta: no data for this request
            return []
        if r.status_code >= 500:
            time.sleep(3 * (i + 1)); last = f"{r.status_code} {r.text[:200]}"; continue
        if r.status_code >= 400:
            raise RuntimeError(f"{path} {q} -> {r.status_code}: {r.text[:300]}")
        try:
            return flatten(r.json())
        except ValueError:
            return pd.read_csv(io.StringIO(r.text)).to_dict("records")
    raise RuntimeError(f"{path} {q} -> {last}")


def get_v2(path, **q):
    """v2 REST: {"header": {"format": [...]} , "response": [...]}; bulk responses are [{"ticks": [...], "contract": {...}}]."""
    url, out = f"{BASE}{path}", []
    while url:
        for i in range(5):
            try:
                r = requests.get(url, params=q, timeout=180, headers={"Connection": "close"})
            except requests.RequestException as e:
                time.sleep(3 * (i + 1)); last = str(e); continue
            if r.status_code == 472:
                return out
            if r.status_code >= 500 or r.status_code == 429:
                time.sleep(3 * (i + 1)); last = f"{r.status_code} {r.text[:200]}"; continue
            if r.status_code >= 400:
                raise RuntimeError(f"{path} {q} -> {r.status_code}: {r.text[:300]}")
            break
        else:
            raise RuntimeError(f"{path} {q} -> {last}")
        j = r.json()
        h = j.get("header", {})
        if h.get("error_type") and h.get("error_type") not in ("null", None, "NO_DATA"):
            if "NO_DATA" in str(h.get("error_type")) or "no data" in str(h.get("error_msg", "")).lower():
                return out
            raise RuntimeError(f"{path} {q} -> {h.get('error_type')}: {h.get('error_msg')}")
        fmt = h.get("format")
        for item in j.get("response", []):
            if isinstance(item, dict) and "ticks" in item:
                c = item.get("contract", {})
                f = item.get("format") or fmt
                out += [{**c, **dict(zip(f, t))} for t in item["ticks"]]
            elif fmt and isinstance(item, list):
                out.append(dict(zip(fmt, item)))
            else:
                out.append(item)
        nxt = h.get("next_page")
        url, q = (nxt, {}) if nxt and nxt not in ("null", "None") else (None, q)
    return out


def flatten(j):
    """Theta v3 JSON comes as a list, {"response": [...]}, or response rows of {"contract": {...}, "data": [...]}."""
    rows = j.get("response", j) if isinstance(j, dict) else j
    out = []
    for r in rows if isinstance(rows, list) else [rows]:
        if isinstance(r, dict) and "data" in r and isinstance(r["data"], list):
            c = r.get("contract", {})
            out += [{**c, **x} for x in r["data"]]
        else:
            out.append(r)
    return out


def chain_bars(und, d):
    """1-minute bars 10:00-10:15 for every contract expiring today (strike='*', right='both')."""
    ds = d.strftime("%Y%m%d")
    if API == "v2":
        return get_v2("/v2/bulk_hist/option/ohlc", root=und, exp=ds, start_date=ds, end_date=ds, ivl=60000,
                      start_time=36000000, end_time=36900000, use_csv="false")
    return get("/option/history/ohlc", symbol=und, expiration=ds, strike="*", right="both", date=ds,
               interval="1m", start_time="10:00:00", end_time="10:15:00")


def norm_rows(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    low = {c.lower(): c for c in df.columns}
    pick = lambda *names: next((low[n] for n in names if n in low), None)
    k, r_, cl, vol, vw = pick("strike"), pick("right"), pick("close"), pick("volume"), pick("vwap")
    if k is None or cl is None:
        raise RuntimeError(f"unexpected columns {list(df.columns)[:15]}")
    out = pd.DataFrame({"strike": pd.to_numeric(df[k], errors="coerce"), "right": df[r_].astype(str).str.upper().str[0],
                        "close": pd.to_numeric(df[cl], errors="coerce"), "volume": pd.to_numeric(df[vol], errors="coerce").fillna(0)})
    if out.strike.median() > 5000:                                 # v2-style 1/1000-dollar strikes
        out["strike"] /= 1000
    out["vwap"] = pd.to_numeric(df[vw], errors="coerce") if vw else np.nan
    return out


def agg(b):
    """per contract: volume-weighted price over 10:00-10:15 (bar vwap if given, else close), volume."""
    b = b[b.close > 0]
    px = b.vwap.where(b.vwap > 0, b.close)
    g = b.assign(pv=px * b.volume.clip(lower=0), px=px).groupby(["strike", "right"])
    return pd.DataFrame({"vwap": g.pv.sum() / g.volume.sum().replace(0, np.nan), "mean_px": g.px.mean(),
                         "volume": g.volume.sum()}).reset_index().assign(price=lambda x: x.vwap.fillna(x.mean_px))


def probe():
    e = sorted(expirations("SPY"))
    note(f"probe ({API}): SPY expirations {len(e)}, first {e[:2]}, 2018 count {sum(1 for x in e if x.year == 2018)}")
    d = dt.date(2018, 2, 9)
    rows = chain_bars("SPY", d)
    note(f"probe: chain {d} rows {len(rows)}, sample {json.dumps(rows[:2], default=str)[:600]}")
    if rows:
        a = agg(norm_rows(rows))
        note(f"probe: contracts {len(a)}, strikes {a.strike.min()}-{a.strike.max()}, vol {a.volume.sum():.0f}")


def expirations(und):
    if API == "v2":
        r = get_v2("/v2/list/expirations", root=und)
        vals = [x if not isinstance(x, dict) else next(iter(x.values())) for x in r]
        return set(pd.to_datetime(pd.Series(vals).astype(str), format="%Y%m%d", errors="coerce").dropna().dt.date)
    e = pd.DataFrame(get("/option/list/expirations", symbol=und))
    col = next(c for c in e.columns if "exp" in c.lower())
    if "symbol" in e.columns:
        e = e[e.symbol.astype(str).str.upper() == und]
    s = pd.to_datetime(e[col].astype(str).str.replace("-", ""), format="%Y%m%d", errors="coerce").dropna().dt.date
    return set(s)


def spot(und):
    b = pd.read_parquet(f"{OUT}/{und.lower()}_15m_raw_2016.parquet")
    b["d"] = b.ts.dt.date; b["tm"] = b.ts.dt.strftime("%H:%M")
    return b[b.tm == "10:00"].set_index("d").open, b.groupby("d").close.last()


def pull(und):
    path = f"{CACHE_DIR}/{und.lower()}_chain.parquet"
    have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["d"])
    done = set(pd.to_datetime(have.d).dt.date) if len(have) else set()
    exps = expirations(und)
    s0, _ = spot(und)
    days = sorted(d for d in exps if START <= d <= END and d in s0.index and d not in done)
    if os.environ.get("LIMIT"):
        days = days[:int(os.environ["LIMIT"])]
    note(f"{und}: {len([d for d in exps if START <= d <= END])} same-day expiries in range, {len(done)} cached, {len(days)} to pull")
    new, t0 = [], time.time()
    for i, d in enumerate(days):
        try:
            a = agg(norm_rows(chain_bars(und, d)))
        except Exception as ex:
            warn(f"{und} {d}: {str(ex)[:200]}"); continue
        lo, hi = s0[d] * 0.9, s0[d] * 1.1
        new.append(a[(a.strike >= lo) & (a.strike <= hi)].assign(d=pd.Timestamp(d)))
        if (i + 1) % 100 == 0 or i == len(days) - 1:
            have = pd.concat([have] + new, ignore_index=True); new = []
            have.to_parquet(path, index=False)
            print(f"  {und} {i + 1}/{len(days)} days ({time.time() - t0:.0f}s)", flush=True)
    return have


def backtest(und, ch):
    s0, sT = spot(und)
    rows = []
    for d, g in ch.groupby(ch.d.dt.date):
        if d not in s0.index or d not in sT.index:
            continue
        S, ST = float(s0[d]), float(sT[d])
        px = {(round(k, 3), r): p for k, r, p in zip(g.strike, g.right, g.price) if p == p and p > 0}
        atm = round(S)
        c, p = px.get((atm, "C")), px.get((atm, "P"))
        if c is None or p is None:
            continue
        M = c + p
        W = 5 if und == "SPY" else max(1, round(0.008 * S))
        ps, cs = round(S - M), round(S + M)
        sp, sc = px.get((ps, "P")), px.get((cs, "C"))
        if sp is None or sc is None:
            continue
        lp, lc = px.get((ps - W, "P"), 0.01), px.get((cs + W, "C"), 0.01)
        loss = min(max(ps - ST, 0), W) + min(max(ST - cs, 0), W)
        for cost in (0.01, 0.03):
            cr = sp + sc - lp - lc - 4 * cost
            if cr <= 0 or cr >= 0.6 * W:
                continue
            rows.append(dict(d=pd.Timestamp(d), und=und, cost=cost, spot=S, close=ST, M_pct=M / S, W=W, credit=cr,
                             loss=loss, ror=(cr - loss) / (W - cr), maxloss=loss >= W - 1e-9))
    return pd.DataFrame(rows)


def summarise(t):
    out = []
    for (und, cost), g in t.groupby(["und", "cost"]):
        groups = [(str(y), x) for y, x in g.groupby(g.d.dt.year)] + [("2016-23", g[g.d.dt.year <= 2023]), ("ALL", g)]
        for lab, x in groups:
            if len(x) < 15:
                continue
            nav = (1 + 0.02 * x.ror).cumprod()
            out.append(dict(und=und, cost=cost, period=lab, trades=len(x), ror_mean_pct=x.ror.mean() * 100,
                            t=x.ror.mean() / x.ror.std() * np.sqrt(len(x)), win_pct=(x.ror > 0).mean() * 100,
                            maxloss_pct=x.maxloss.mean() * 100, book_2pct_ret_pct=(nav.iloc[-1] - 1) * 100,
                            book_2pct_dd_pct=(nav / nav.cummax() - 1).min() * 100))
    return pd.DataFrame(out)


def validate(t):
    """Feb-Mar 2024: these same days from the Alpaca-price backtest."""
    msgs = []
    for und, f in (("SPY", "options_premium_trades.parquet"), ("QQQ", "options_premium_trades_QQQ.parquet")):
        if not os.path.exists(f"{OUT}/{f}"):
            continue
        a = pd.read_parquet(f"{OUT}/{f}")
        Wl = "5" if und == "SPY" else "0.8%"
        a = a[(a.tenor == "0DTE") & (a.struct == "iron condor") & (a.k == 1.0) & (a.W.astype(str) == Wl) & (a.cost == 0.01)]
        a = a.assign(d=pd.to_datetime(a.d)).set_index("d")
        x = t[(t.und == und) & (t.cost == 0.01)].set_index("d")
        j = x[["ror", "M_pct"]].join(a[["ror", "M_pct"]], rsuffix="_alp", how="inner")
        if len(j):
            msgs.append(f"{und} {len(j)} overlap days: RoR corr {j.ror.corr(j.ror_alp):.2f}, M corr {j.M_pct.corr(j.M_pct_alp):.3f}, "
                        f"mean RoR theta {j.ror.mean() * 100:+.1f} vs alpaca {j.ror_alp.mean() * 100:+.1f}")
    return msgs


def main():
    if os.environ.get("MODE") == "probe":
        probe(); return
    T = []
    for und in ("SPY", "QQQ"):
        try:
            ch = pull(und)
            t = backtest(und, ch)
            t.to_parquet(f"{OUT}/options_theta_trades_{und}.parquet", index=False)
            T.append(t)
        except Exception as ex:
            warn(f"{und}: {str(ex)[:300]}")
    t = pd.concat(T, ignore_index=True)
    S = summarise(t)
    S.to_csv(f"{OUT}/options_theta_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(2).to_string(index=False))
    for m in validate(t):
        note("theta validation: " + m)
    k = S[S.period.isin(["2016-23", "2018", "2020", "2022", "2023"])]
    note("theta real-price condor (RoR mean %, t, 2%/day book %, DD %): " + "; ".join(
        f"{r.und} {r.cost} {r.period}: {r.ror_mean_pct:+.1f} t{r.t:.1f} {r.book_2pct_ret_pct:+.0f}% {r.book_2pct_dd_pct:.0f}% n{r.trades}"
        for r in k.itertuples()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc(); print(tb); print("::error::" + tb[-900:].replace("\n", " | ")); raise
