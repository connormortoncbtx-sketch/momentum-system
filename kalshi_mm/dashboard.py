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

PORT = int(os.environ.get("DASH_PORT", "8787"))
CACHE_S = 45
CT = ZoneInfo("America/Chicago")
_cache = {"t": 0.0, "page": "", "lock": threading.Lock()}


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
    except Exception as e:
        rows, cash, err = [], 0.0, f"Kalshi fetch failed: {html.escape(str(e))[:200]}"
    sell = cash + sum(r["sell"] for r in rows)
    exp = cash + sum(r["ev"] for r in rows)
    try:
        rep = R.build()
    except Exception as e:
        rep = f"report failed: {e}"
    lot = {}
    try:
        lot = json.load(open(os.path.join(B.LOG_DIR, "lottery.json")))
    except Exception:
        pass
    killed = os.path.exists(B.KILL_FILE)
    status = svc("kmm")
    ok = status == "active" and not killed

    def cls(v):
        return "pos" if v >= 0 else "neg"

    h = [f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Kalshi bot</title><style>
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
<div class="mut">{now:%a %b %d, %-I:%M %p} CT · refreshes every minute{(' · <span class=neg>' + err + '</span>') if err else ''}</div>"""]
    if dep:
        h.append(f"""<div class="card"><h2>Bottom line vs ${dep:,.2f} deposited</h2><div class="grid">
<div><div class="mut">If sold now</div><div class="big {cls(sell - dep)}">{money(sell - dep, True)}</div><div class="mut">{money(sell)}</div></div>
<div><div class="mut">Expected at settlement</div><div class="big {cls(exp - dep)}">{money(exp - dep, True)}</div><div class="mut">{money(exp)}</div></div>
<div><div class="mut">Cash</div><div class="big">{money(cash)}</div><div class="mut">{len(rows)} open positions</div></div></div></div>""")
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
    if lot:
        h.append(f"""<div class="card"><h2>Weekly roll</h2>{html.escape(str(lot.get('status')))} · week {html.escape(str(lot.get('week')))}
· leg {lot.get('leg')} / {os.environ.get('LOT_LEGS', '7')} · bankroll {money(B.num(lot.get('cash')) + B.num(lot.get('held')) * B.num(lot.get('price')))}
{('· holding ' + html.escape(str(lot.get('ticker')))) if lot.get('ticker') else ''}</div>""")
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
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
