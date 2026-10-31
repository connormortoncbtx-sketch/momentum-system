"""
kalshi_mm/dashboard.py -- read-only live dashboard for the Kalshi bot.

Serves one auto-refreshing page on 127.0.0.1:8787 (localhost only). Reach it from your phone/laptop through Tailscale
('tailscale serve' publishes it as an https URL that only devices on your tailnet can open) -- nothing is exposed to the
internet. Shows: bottom line vs deposits (sell-now and expected at settlement), bot / lottery status, open positions by
market date with odds and colour (decided winners / losers), the settled scorecard and per-city lines (report.py), and
the weekly roll. Places and cancels nothing. Kalshi data is fetched at most every CACHE_S seconds.
"""
import datetime as dt
import html
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot as B          # noqa: E402
import positions as P    # noqa: E402
import report as R       # noqa: E402
import history as HI     # noqa: E402


def svg_line(pts, band=None, w=640, h=170, fmt=lambda v: f"{v:+.0f}", zero=True):
    """pts: [(label, y)]; band: [(lo, hi)] per point (optional). Inline SVG, theme colours via CSS variables."""
    if len(pts) < 2:
        return "<div class='mut'>Not enough data yet.</div>"
    ys = [y for _, y in pts] + ([b for lo_hi in (band or []) for b in lo_hi if b == b] if band else []) + ([0] if zero else [])
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    pl, pr, pt, pb = 74, 12, 14, 34
    X = lambda i: pl + i * (w - pl - pr) / (len(pts) - 1)
    Y = lambda v: pt + (hi - v) * (h - pt - pb) / (hi - lo)
    out = [f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' style='display:block'>"]
    if band:
        up = [f"{X(i):.1f},{Y(b[1]):.1f}" for i, b in enumerate(band) if b[1] == b[1]]
        dn = [f"{X(i):.1f},{Y(b[0]):.1f}" for i, b in reversed(list(enumerate(band))) if b[0] == b[0]]
        if up:
            out.append(f"<polygon points='{' '.join(up + dn)}' fill='var(--mut)' opacity='.18'/>")
    if zero and lo < 0 < hi:
        out.append(f"<line x1='{pl}' x2='{w - pr}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--mut)' stroke-dasharray='4 4' stroke-width='1'/>")
    last = pts[-1][1]
    col = "var(--pos)" if last >= 0 else "var(--neg)"
    out.append(f"<polyline fill='none' stroke='{col}' stroke-width='2.2' points='{' '.join(f'{X(i):.1f},{Y(y):.1f}' for i, (_, y) in enumerate(pts))}'/>")
    for v in (hi, lo):
        out.append(f"<text x='{pl - 6}' y='{Y(v) + 4:.1f}' text-anchor='end' font-size='20' fill='var(--mut)'>{fmt(v)}</text>")
    for i in (0, len(pts) - 1):
        out.append(f"<text x='{X(i):.1f}' y='{h - 8}' text-anchor='{'start' if i == 0 else 'end'}' font-size='20' fill='var(--mut)'>{html.escape(str(pts[i][0]))}</text>")
    out.append("</svg>")
    return "".join(out)


def svg_bars(pts, w=640, h=150, fmt=lambda v: f"{v:+.0f}"):
    if not pts:
        return "<div class='mut'>No settled days yet.</div>"
    ys = [y for _, y in pts] + [0]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi += 1
    pl, pr, pt, pb = 74, 12, 14, 34
    bw = (w - pl - pr) / len(pts)
    Y = lambda v: pt + (hi - v) * (h - pt - pb) / (hi - lo)
    out = [f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' style='display:block'>",
           f"<line x1='{pl}' x2='{w - pr}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--mut)' stroke-width='1'/>"]
    for i, (lab, y) in enumerate(pts):
        x = pl + i * bw + bw * .15
        top, bot = (Y(y), Y(0)) if y >= 0 else (Y(0), Y(y))
        out.append(f"<rect x='{x:.1f}' y='{top:.1f}' width='{bw * .7:.1f}' height='{max(bot - top, 1):.1f}' rx='2' "
                   f"fill='var(--{'pos' if y >= 0 else 'neg'})'><title>{html.escape(str(lab))}: {fmt(y)}</title></rect>")
    for v in (hi, lo):
        out.append(f"<text x='{pl - 6}' y='{Y(v) + 4:.1f}' text-anchor='end' font-size='20' fill='var(--mut)'>{fmt(v)}</text>")
    for i in (0, len(pts) - 1):
        out.append(f"<text x='{pl + i * bw + bw / 2:.1f}' y='{h - 8}' text-anchor='middle' font-size='20' fill='var(--mut)'>{html.escape(str(pts[i][0]))}</text>")
    out.append("</svg>")
    return "".join(out)


def meter(frac, warn=.8):
    f = max(0.0, min(frac, 1.0))
    col = "var(--neg)" if frac >= 1 else ("#e0a030" if frac >= warn else "var(--pos)")
    return (f"<div style='background:var(--line);border-radius:4px;height:8px;min-width:80px'>"
            f"<div style='width:{f * 100:.0f}%;height:8px;border-radius:4px;background:{col}'></div></div>")


def limits_card():
    """Budget, per-city-day cap and per-market cap: where the bot stands now and how often each one bound."""
    try:
        c = HI.caps(B.EVENT_CAP, B.INV_CAP, B.SIZE)
    except Exception as e:
        return f"<div class='card'><h2>Limits</h2><div class='neg'>failed: {html.escape(str(e))}</div></div>"
    if not c:
        return ""
    n = c["now"]
    blk_ev, blk_mk = B.EVENT_CAP - B.SIZE, B.INV_CAP - B.SIZE
    out = [f"<div class='card'><h2>Limits — budget, city-day cap ({B.EVENT_CAP:.0f}), market cap ({B.INV_CAP}). "
           f"A city-day stops adding above {blk_ev:.0f} contracts (one more {B.SIZE}-lot would breach)</h2>"]
    frac = n["used"] / n["budget"] if n["budget"] else 0
    wk = c["budget_days"][-7:]
    out.append(f"<div class='grid'><div><div class='mut'>Budget in use now</div><div class='big' style='font-size:22px'>"
               f"{money(n['used'])} / {money(n['budget'])}</div>{meter(frac)}</div>"
               f"<div><div class='mut'>Last 7 days: average / peak</div><div class='big' style='font-size:22px'>"
               f"{sum(a for _, a, _ in wk) / len(wk):.0f}% / {max(m for _, _, m in wk):.0f}%</div></div></div>")
    if c["open_events"]:
        rows_ = []
        for ev, held, mk in c["open_events"]:
            top_t, top_v = max(mk.items(), key=lambda kv: kv[1])
            city = ev.split("-")[0].replace("KXHIGHT", "").replace("KXHIGH", "").replace("KXLOWT", "low ")
            rows_.append(f"<tr><td>{html.escape(city)}</td><td>{html.escape(ev.split('-')[1])}</td>"
                         f"<td>{held:.0f} / {B.EVENT_CAP:.0f}{' blocked' if held > blk_ev else ''}</td><td>{meter(held / B.EVENT_CAP)}</td>"
                         f"<td>{html.escape(top_t.rsplit('-', 1)[1])} {top_v:.0f} / {B.INV_CAP}</td></tr>")
        out.append("<h2 style='margin-top:12px'>Open city-days now</h2><div class='scroll'><table><tr><th>City</th><th>Day</th>"
                   "<th>Held / cap</th><th></th><th>Biggest market / cap</th></tr>" + "".join(rows_) + "</table></div>")
    md = [r for r in c["market_days"] if r[0]][-14:]
    bd = {d: m for d, _, m in c["budget_days"]}
    if md:
        rows_ = "".join(f"<tr><td>{d:%a %b %d}</td><td>{ne}</td><td class='{'neg' if nb else ''}'>{nb} ({100 * nb / ne:.0f}%)</td>"
                        f"<td>{pk:.0f}</td><td>{nm}</td><td>{bd.get(d, float('nan')):.0f}%</td></tr>" for d, ne, nb, pk, nm in reversed(md))
        tb = sum(r[2] for r in md); te = sum(r[1] for r in md)
        out.append(f"<h2 style='margin-top:12px'>History by market day · city-days blocked by the cap: {tb} of {te} ({100 * tb / te:.0f}%)</h2>"
                   "<div class='scroll'><table><tr><th>Market day</th><th>City-days</th><th>Blocked</th><th>Peak held</th>"
                   "<th>Markets blocked</th><th>Peak budget</th></tr>" + rows_ + "</table></div>"
                   "<div class='mut'>Peak budget = highest share of the budget in use on that calendar day. "
                   "If 'Blocked' stays high once favourite-only is proven, the city-day cap is what's limiting size.</div>")
    out.append("</div>")
    return "".join(out)


_edge = {"t": 0.0, "data": None, "err": None}
EDGE_S = 3600                                      # the edge card is recomputed hourly (one API call per unsettled market)


def edge_refresher():
    import edge_now as EN
    while True:
        try:
            d = EN.compute()
            _edge.update(t=time.time(), data=d, err=None)
        except Exception as e:
            _edge.update(err=str(e)[:200])
        time.sleep(EDGE_S)


def edge_card():
    """kmm edge-now on the page (Oct 30: no copyable terminal on the phone)."""
    d = _edge.get("data")
    if not d:
        msg = f"failed: {html.escape(_edge['err'])}" if _edge.get("err") else "computing (first run takes a minute)..."
        return f"<div class='card'><h2>Edge now</h2><div class='mut'>{msg}</div></div>"
    import edge_now as EN
    body = []
    for lab, o in d["rows"]:
        if not o:
            body.append(f"<tr><td>{html.escape(lab)}</td><td colspan=6 class='mut'>too few city-days</td></tr>"); continue
        (m1, _, _), (_, b1d, D), (m2, _, _), (_, b2d, _) = o["cd"], o["cd_day"], o["ct"], o["ct_day"]
        sm, sb, wt = o["shrunk"]
        sig = lambda m, b: "pos" if m - b > 0 else ("neg" if m + b < 0 else "")
        body.append(f"<tr><td>{html.escape(lab)}</td><td>{o['G']}</td><td>{D}</td>"
                    f"<td class='{sig(m1, b1d)}'>{m1:+.1f}c ±{b1d:.1f}</td><td class='{sig(m2, b2d)}'>{m2:+.1f}c ±{b2d:.1f}</td>"
                    f"<td>{sm:+.1f}c ±{sb:.1f} <span class='mut'>({wt:.0%} live)</span></td></tr>")
    age = (time.time() - _edge["t"]) / 60
    return (f"<div class='card'><h2>Edge now -- includes {d['prov']} decided-but-unpaid markets ({d['pdays']} city-days); "
            f"updated {age:.0f} min ago</h2><div class='scroll'><table><tr><th>View</th><th>City-days</th><th>Days</th>"
            f"<th>Per city-day</th><th>Per contract</th><th>Shrunk per contract</th></tr>{''.join(body)}</table></div>"
            f"<div class='mut'>± = 95% band clustered by MARKET DAY (cities on one date move together). Green/red = the band "
            f"clears zero. Shrunk = research prior {EN.PRIOR_C:+.1f}c blended with live data -- the cap-ladder number "
            f"(step up at >= +2c).</div></div>")


def history_cards():
    out = []
    try:
        acc = HI.account_series()
        pts = [(t.strftime("%b %d %-I%p"), v) for t, v in acc[-24 * 21:]]
        note = "" if os.environ.get("DEPOSIT_HISTORY") else " · set DEPOSIT_HISTORY so early hours aren't shown against later deposits"
        out.append(f"<div class='card'><h2>Account vs money deposited, hourly{note}</h2>"
                   f"{svg_line(pts, fmt=lambda v: f'{v:+.0f}')}</div>")
        dep = HI.deployment_series()[-24 * 21:]
        if dep:
            avg = sum(r[1] for r in dep) / len(dep)
            bud = next((r[2] for r in reversed(dep) if r[2]), None)
            out.append(f"<div class='card'><h2>Capital deployed, hourly -- % of the bot's account value in positions (at cost; "
                       f"resting orders not included)</h2>"
                       f"{svg_line([(t.strftime('%b %d %-I%p'), v) for t, v, _ in dep], fmt=lambda v: f'{v:.0f}%')}"
                       f"<div class='mut'>Now {dep[-1][1]:.0f}% · average {avg:.0f}% over the chart · peak {max(r[1] for r in dep):.0f}%"
                       + (f" · budget {bud:.0f}% of value" if bud else "") + "</div></div>")
        days = HI.settled_days()
        if days:
            cum, run = [], 0.0
            for d, usd, cpc, ev in days:
                run += usd; cum.append((d.strftime("%b %d"), run))
            rows_ = "".join(f"<tr><td>{d:%a %b %d}</td><td>{ev}</td><td class='{'pos' if usd >= 0 else 'neg'}'>{usd:+.2f}</td>"
                            f"<td class='{'pos' if cpc >= 0 else 'neg'}'>{cpc:+.1f}c</td></tr>" for d, usd, cpc, ev in reversed(days))
            out.append(f"<div class='card'><h2>Settled P&amp;L by market day ($) · running total {run:+.2f}</h2>"
                       f"{svg_bars([(d.strftime('%b %d'), usd) for d, usd, _, _ in days], fmt=lambda v: f'{v:+.0f}')}"
                       f"{svg_line(cum, h=120, fmt=lambda v: f'{v:+.0f}')}"
                       f"<div class='scroll'><table><tr><th>Market day</th><th>City-days</th><th>$</th><th>per contract</th></tr>{rows_}</table></div></div>")
        since = dt.date.fromisoformat(os.environ.get("EVAL_FROM", "2026-10-24"))
        for lab, cv in ((f"CLEAN TEST: market days from {since:%b %d} (favourite-only + fast quoting + :45-:59 quiet window)", HI.convergence(since)),
                        (f"CLEAN TEST from {since:%b %d}, weighted by contracts (what the account feels)", HI.convergence(since, "contract")),
                        ("ALL HISTORY, current-logic trades only: favourite fills outside :45-:59, old logic included "
                         "(more data; slower quoting before Oct 22, and the filter was chosen after seeing the data)",
                         HI.convergence(filt=HI.current_logic)),
                        ("ALL HISTORY, current-logic trades only, WEIGHTED BY CONTRACTS (compare with kmm diagnose's +10.3c)",
                         HI.convergence(weight="contract", filt=HI.current_logic)),
                        ("all market days since launch (includes the old logic)", HI.convergence()),
                        ("all market days since launch, WEIGHTED BY CONTRACTS (compare with the current-logic contract-weighted chart)",
                         HI.convergence(weight="contract"))):
            if not cv:
                out.append(f"<div class='card'><h2>Is it working? {lab}</h2><div class='mut'>No settled city-days yet.</div></div>")
                continue
            n, m, b = cv[-1]
            verdict = ("too early to tell" if n < 30 or b != b else
                       ("edge is positive (band above 0)" if m - b > 0 else ("edge is negative (band below 0)" if m + b < 0 else "band still straddles 0")))
            out.append(f"<div class='card'><h2>Is it working? {lab} -- edge per contract, running average over settled city-days "
                       f"(shaded = ±2 standard errors; research target +2 to +3c; decision at ~50)</h2>"
                       f"{svg_line([(f'{i} city-days', mm) for i, mm, _ in cv], band=[(mm - bb, mm + bb) for _, mm, bb in cv], fmt=lambda v: f'{v:+.0f}c')}"
                       f"<div>Now: <b class='{'pos' if m >= 0 else 'neg'}'>{m:+.1f}c</b> ± {b if b == b else float('nan'):.1f}c over {n} city-days — {verdict}</div></div>")
        fd = HI.fills_per_day()
        if fd:
            out.append(f"<div class='card'><h2>Contracts filled per day</h2>"
                       f"{svg_bars([(d.strftime('%b %d'), v) for d, v in fd[-30:]], h=120, fmt=lambda v: f'{v:.0f}')}</div>")
    except Exception as e:
        out.append(f"<div class='card'><h2>History</h2><div class='neg'>history failed: {html.escape(str(e))}</div></div>")
    return "".join(out)

PORT = int(os.environ.get("DASH_PORT", "8787"))
CACHE_S = 300                                      # rebuild every 5 min: analysis is days/weeks, not seconds
# The dashboard shares the account's Kalshi rate limit with the bot. It always yields: this process makes at most 2
# requests/s (the bot's own limiter is separate), so a refresh is slow but can never crowd out execution.
B.READ_LIMIT = B.RateLimit(2)
B.WRITE_LIMIT = B.RateLimit(1)
CT = ZoneInfo("America/Chicago")
_cache = {"t": 0.0, "page": "", "lock": threading.Lock()}
_last_good = {}


def svc(name):
    try:
        return subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "?"


def money(x, sign=False):
    return (f"{'+' if x >= 0 else '−'}${abs(x):,.2f}" if sign else f"${x:,.2f}")


def build_page():
    now = dt.datetime.now(CT)
    dep = B.num(os.environ.get("DEPOSITS_USD"))
    err = ""
    try:
        rows, cash = P.collect()
        _last_good.update(rows=rows, cash=cash, t=now)
    except Exception as e:
        if _last_good.get("rows") is not None:          # e.g. a rate-limit blip: show the last good copy, flagged
            rows, cash = _last_good["rows"], _last_good["cash"]
            err = f"Kalshi busy ({html.escape(str(e))[:80]}) -- positions as of {_last_good['t']:%-I:%M %p}"
        else:
            rows, cash, err = [], 0.0, f"Kalshi fetch failed: {html.escape(str(e))[:200]}"
    lrows = [r for r in rows if r.get("lottery")]     # the weekly roll: wholly separate, shown in its own card
    rows = [r for r in rows if not r.get("lottery")]
    net = P.collect.netted                      # sure payouts on mutually exclusive buckets, already in cash
    sell = cash - net + sum(r["sell"] for r in rows + lrows)          # whole account (deposits are the whole account)
    exp = cash - net + sum(r["ev"] for r in rows + lrows)
    try:
        rep = R.build()
    except Exception as e:
        rep = f"report failed: {e}"
    lot = {}
    try:
        lot = json.load(open(os.path.join(B.LOG_DIR, "lottery.json")))
    except Exception:
        pass
    lot_cash = B.LF.cash_in_bal(lot, {r["ticker"]: r["n"] for r in lrows}) if lot else 0.0
    lot_sell = lot_cash + sum(r["sell"] for r in lrows)
    lot_exp = lot_cash + sum(r["ev"] for r in lrows)
    killed = os.path.exists(B.KILL_FILE)
    status = svc("kmm")
    ok = status == "active" and not killed

    def cls(v):
        return "pos" if v >= 0 else "neg"

    h = [f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="120"><title>Kalshi bot</title><style>
:root{{--bg:#0f1115;--card:#171a21;--ink:#e8eaed;--mut:#9aa0a6;--pos:#3ecf8e;--neg:#ff6b6b;--line:#262a33}}
@media (prefers-color-scheme: light){{:root{{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--mut:#5f6368;--pos:#0a8a57;--neg:#c62828;--line:#e3e5e8}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,system-ui,Segoe UI,Roboto,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}}
h1{{font-size:18px;margin:4px 0}} h2{{font-size:15px;margin:0 0 8px;color:var(--mut);font-weight:600}}
.big{{font-size:30px;font-weight:700}} .pos{{color:var(--pos)}} .neg{{color:var(--neg)}} .mut{{color:var(--mut)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td,th{{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{color:var(--mut);font-weight:600}} td:nth-child(-n+3),th:nth-child(-n+3){{text-align:left}} .scroll{{overflow-x:auto}}
tr.win td{{background:color-mix(in srgb,var(--pos) 10%,transparent)}} tr.lose td{{background:color-mix(in srgb,var(--neg) 10%,transparent)}}
pre{{white-space:pre-wrap;font-size:12.5px;margin:0}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:var(--{'pos' if ok else 'neg'})}}
</style></head><body><main>
<h1><span class="dot"></span>Kalshi bot — {'running' if ok else ('KILL SWITCH ON' if killed else 'service ' + html.escape(status))}</h1>
<div class="mut">{now:%a %b %d, %-I:%M %p} CT · data refreshes every 5 min{(' · <span class=neg>' + err + '</span>') if err else ''}</div>"""]
    if dep:
        h.append(f"""<div class="card"><h2>Bottom line vs ${dep:,.2f} deposited</h2><div class="grid">
<div><div class="mut">If sold now</div><div class="big {cls(sell - dep)}">{money(sell - dep, True)}</div><div class="mut">{money(sell)}</div></div>
<div><div class="mut">Expected at settlement</div><div class="big {cls(exp - dep)}">{money(exp - dep, True)}</div><div class="mut">{money(exp)}</div></div>
<div><div class="mut">Cash</div><div class="big">{money(cash)}</div><div class="mut">{len(rows)} open positions</div></div></div>
<div class="mut">Of that: market-making bot {money(sell - lot_sell)} sold now / {money(exp - lot_exp)} at settlement · weekly roll {money(lot_sell)} / {money(lot_exp)}</div>
<div class="mut">Positions are net of {money(net)} of sure payouts Kalshi already returned to cash (NO on several buckets of one city-day: at most one can lose). Per-day tables below show them gross.</div></div>""")
    try:
        kb, kt = P.kalshi_truth()
        h.append("<div class='card'><h2>Kalshi's own numbers (to reconcile with the app and with ours)</h2>"
                 + "<div class='grid'>" + "".join(f"<div><div class='mut'>{html.escape(k)}</div><div class='big' style='font-size:20px'>"
                                                 f"{money(v)}</div></div>" for k, v in kb.items())
                 + f"<div><div class='mut'>realized P&amp;L, {kt['markets']} markets</div><div class='big {cls(kt['realized_pnl'])}' "
                   f"style='font-size:20px'>{money(kt['realized_pnl'], True)}</div></div>"
                 + f"<div><div class='mut'>fees paid</div><div class='big' style='font-size:20px'>{money(kt['fees_paid'])}</div></div>"
                 + "</div><div class='mut'>Straight from Kalshi's API. Our bottom line = cash + open positions valued by us; if the app "
                   "and this card disagree with it, these are the numbers to trust.</div></div>")
    except Exception as e:
        h.append(f"<div class='card'><h2>Kalshi's own numbers</h2><div class='neg'>unavailable: {html.escape(str(e))[:200]}</div></div>")
    h.append(edge_card())
    h.append(limits_card())
    h.append(history_cards())
    for day in sorted({r["day"] for r in rows}):
        g = [r for r in rows if r["day"] == day]
        tc, te = sum(r["cost"] for r in g), sum(r["ev"] for r in g)
        body = "".join(
            f"<tr class='{'win' if r['p'] > .97 else ('lose' if r['p'] < .03 else '')}'><td>{html.escape(r['city'])}</td>"
            f"<td>{html.escape(r['bucket'])}</td><td>{r['side']}</td><td>{r['n']:.1f}</td><td>{money(r['cost'])}</td>"
            f"<td>{r['p'] * 100:.0f}%{'*' if r['src'] == 'last' else ''}</td><td>{money(r['sell'])}</td><td>{money(r['ev'])}</td>"
            f"<td class='{cls(r['ev'] - r['cost'])}'>{money(r['ev'] - r['cost'], True)}</td></tr>" for r in g)
        h.append(f"""<div class="card"><h2>{html.escape(day)} — {len(g)} positions · cost {money(tc)} · expected {money(te)}
<span class="{cls(te - tc)}">({money(te - tc, True)})</span> · decided: {sum(r['p'] > .97 for r in g)} won, {sum(r['p'] < .03 for r in g)} lost</h2>
<div class="scroll"><table><tr><th>City</th><th>Bucket</th><th>Side</th><th>Qty</th><th>Cost</th><th>Chance</th><th>Sell now</th><th>Expected</th><th>vs cost</th></tr>
{body}</table></div></div>""")
    try:
        for day, rs, tot in HI.recent_settled(3):
            won = sum(1 for r in rs if r[3] > 0.005); lost = sum(1 for r in rs if r[3] < -0.005)
            body = "".join(
                f"<tr class='{'win' if r[3] > 0.005 else ('lose' if r[3] < -0.005 else '')}'><td>{html.escape(r[0].split('-')[0].replace('KXHIGHT', '').replace('KXHIGH', ''))}</td>"
                f"<td>{html.escape(r[0].rsplit('-', 1)[1])}</td>"
                f"<td>{('YES ' + format(r[1], '.0f')) if r[1] > 0.5 else (('NO ' + format(-r[1], '.0f')) if r[1] < -0.5 else 'flat')}</td>"
                f"<td>{r[2]:.0f}</td><td>{'YES' if r[4] >= 0.5 else 'NO'}</td><td class='{cls(r[3])}'>{money(r[3], True)}</td></tr>" for r in rs)
            h.append(f"""<div class="card"><h2>SETTLED {day:%a %b %d} -- {len(rs)} markets · <span class="{cls(tot)}">{money(tot, True)}</span>
· {won} won, {lost} lost</h2><div class="scroll"><table><tr><th>City</th><th>Bucket</th><th>Held at close</th><th>Contracts filled</th>
<th>Result</th><th>P&amp;L</th></tr>{body}</table></div><div class="mut">Every fill counted (round trips too), so it matches the
settled P&amp;L table. Lottery excluded.</div></div>""")
    except Exception as e:
        h.append(f"<div class='card'><h2>Recently settled</h2><div class='neg'>failed: {html.escape(str(e))[:200]}</div></div>")
    if lot:
        h.append(f"""<div class="card"><h2>Weekly roll</h2>{html.escape(str(lot.get('status')))} · week {html.escape(str(lot.get('week')))}
· leg {lot.get('leg')} / {os.environ.get('LOT_LEGS', '7')} · worth {money(lot_sell)} sold now, {money(lot_exp)} expected
{('· ' + ', '.join(f"{html.escape(r['city'])} {html.escape(r['bucket'])} {r['side']} {r['n']:.0f} @ {r['p'] * 100:.0f}%" for r in lrows)) if lrows else ''}
<div class="mut">Wholly separate from the bot: not in its caps, budget, loss stop or any of the "is it working" numbers above.</div></div>""")
    h.append(f"""<div class="card"><h2>Scorecard (settled) and per city</h2><pre>{html.escape(rep)}</pre></div>
<div class="mut">Chance = market odds that the side held pays $1 (* = thin book, last trade). Expected = qty × chance.
Read-only page; it never trades.</div></main></body></html>""")
    return "".join(h)


def refresher():
    """Rebuild the page in the background every CACHE_S seconds; requests always get the latest finished copy
    instantly (building takes 5-10 s: one Kalshi book lookup per open position)."""
    while True:
        try:
            page = build_page()
        except Exception as e:
            page = f"<pre>dashboard error: {html.escape(str(e))}</pre>"
        with _cache["lock"]:
            _cache["page"], _cache["t"] = page, time.time()
        time.sleep(CACHE_S)


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404); self.end_headers(); return
        with _cache["lock"]:
            page = (_cache["page"] or "<meta http-equiv='refresh' content='5'><p style='font-family:sans-serif'>"
                    "Dashboard starting -- first load takes ~10 seconds...</p>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=refresher, daemon=True).start()
    threading.Thread(target=edge_refresher, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
