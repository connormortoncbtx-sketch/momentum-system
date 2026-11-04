"""
kalshi_mm/daily_report.py -- the daily "is anything broken?" report (Nov 4, 2026; Connor: "run the study daily and
review it as well to make sure nothing else is broken").

Runs on the droplet once a day (kmm-report.timer, ~10:15 Central, after the morning settlements). Writes one markdown
report and pushes it to Connor's PRIVATE GitHub repo (REPORT_REPO, token in /etc/kmm/report_token -- never in the
public repo), where Claude's scheduled daily review reads it. If any health check fails, a short phone alert goes out
too, so a problem never waits for the review.

Sections:
  HEALTH (last 24h, rule-based)  services up; bot restarts; loss-stop / cancel-all events; order/market errors; age of
       the last snapshot; city-days over the cap; fills inside the quiet window (should be ~0 since QUIET_FULL);
       markets past their day with no result (stuck settlements); weekly roll state
  EDGE  kmm edge-now numbers (clean test / current-logic / all), market-day bands and the shrunk estimate, with the
       frozen ladder rule's verdict
  STUDY  the full `kmm study` output for the clean test (every cut, incl. DEPTH, BURST, minute windows, markouts)
Read-only: places and cancels nothing.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot as B          # noqa: E402
import report as R       # noqa: E402
import history as H      # noqa: E402

B.READ_LIMIT = B.RateLimit(2)
LOG = B.LOG_DIR
REPO_DIR = os.environ.get("REPORT_DIR", "/home/kmm/reports")
TOKEN_FILE = os.environ.get("REPORT_TOKEN_FILE", "/etc/kmm/report_token")
EVAL_FROM = os.environ.get("EVAL_FROM", "2026-10-27")
NOW = dt.datetime.now(dt.timezone.utc)
DAY_AGO = NOW - dt.timedelta(hours=24)


def ts(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def recent(name, field="ts"):
    out = []
    for r in R.rows(name, include_lottery=True):
        x = r.get("fill") or r
        t = ts(x.get("created_time") if name == "fills" else r.get(field))
        if t and t >= DAY_AGO:
            out.append(r)
    return out


def svc(name):
    try:
        return subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def health():
    """[(ok, line)]"""
    out = []
    for s in ("kmm", "kmm-dash", "kmm-sniper", "kmm-lottery.timer"):
        st = svc(s)
        out.append((st == "active", f"service {s}: {st}"))
    ev = recent("events")
    starts = [e for e in ev if e.get("event") == "start"]
    out.append((len(starts) <= 2, f"bot (re)starts in 24h: {len(starts)}" + (" -- more than updates explain" if len(starts) > 2 else "")))
    ca = [e for e in ev if e.get("event") == "cancel_all"]
    stops = [e for e in ca if "daily loss" in str(e.get("reason"))]
    out.append((not stops, f"daily-loss stops in 24h: {len(stops)}" + (f" ({stops[-1].get('reason')})" if stops else "")))
    errs = [e for e in ev if e.get("event") == "error"]
    out.append((len(errs) < 5, f"bot loop errors in 24h: {len(errs)}" + (f"; last: {str(errs[-1].get('error'))[:120]}" if errs else "")))
    er = recent("errors")
    by = {}
    for e in er:
        by[e.get("where")] = by.get(e.get("where"), 0) + 1
    out.append((sum(by.values()) < 50, f"order/market errors in 24h: {sum(by.values())} {by if by else ''}"))
    snaps = R.rows("snapshots")
    last = ts(snaps[-1].get("ts")) if snaps else None
    age = (NOW - last).total_seconds() / 60 if last else 1e9
    out.append((age < 15, f"last snapshot {age:.0f} min ago"))
    over, worst = [], 0.0
    for s in snaps[-3000:]:
        t = ts(s.get("ts"))
        if not t or t < DAY_AGO:
            continue
        evs = {}
        for tk, p in (s.get("positions") or {}).items():
            evs[tk.rsplit("-", 1)[0]] = evs.get(tk.rsplit("-", 1)[0], 0.0) + abs(float(p))
        for e, held in evs.items():
            cap = B.EVENT_CAP_NEW if e.split("-")[0] in B.NEW_SERIES else B.EVENT_CAP
            worst = max(worst, held - cap)
            if held > cap + 0.5:
                over.append(e)
    out.append((not over, f"city-days over their cap in 24h: {len(set(over))}" + (f" (worst +{worst:.0f})" if over else "")))
    fills = [f.get("fill") or f for f in recent("fills")]
    lot = B.LF.tickers(LOG)
    fills = [x for x in fills if x.get("ticker") not in lot]
    qf = [x for x in fills if B.QUIET and B.QUIET[0] <= (ts(x.get("created_time")) or NOW).minute <= B.QUIET[1]]
    qc = sum(R.fill_yes(x)[2] for x in qf)
    nc = sum(R.fill_yes(x)[2] for x in fills)
    out.append((qc <= 10, f"fills in the quiet window in 24h: {len(qf)} ({qc:.0f} of {nc:.0f} contracts)"
                + (" -- quotes are not being pulled at :45" if qc > 10 else "")))
    res = R.market_results()
    stuck = sorted({x.get("ticker") for x in (f.get("fill") or f for f in R.rows("fills"))
                    if x.get("ticker") and x.get("ticker") not in res and x.get("ticker") not in lot
                    and (H.market_day(x.get("ticker")) or dt.date.max) <= (NOW.date() - dt.timedelta(days=3))})
    out.append((not stuck, f"markets 3+ days past their day with no result: {len(stuck)}" + (f" e.g. {stuck[:3]}" if stuck else "")))
    try:
        lj = json.load(open(os.path.join(LOG, "lottery.json")))
        out.append((True, f"weekly roll: {lj.get('status')} week {lj.get('week')} leg {lj.get('leg')}"))
    except Exception:
        out.append((True, "weekly roll: no state"))
    return out


def edge_section():
    import edge_now as EN
    c = EN.compute()
    lines = [f"Includes {c['prov']} decided-but-unpaid markets ({c['pdays']} city-days).", "",
             "| view | city-days | days | per city-day (±by day) | per contract (±by day) | shrunk per contract |",
             "|---|---|---|---|---|---|"]
    verdict = None
    for lab, o in c["rows"]:
        if not o:
            lines.append(f"| {lab} | too few | | | | |"); continue
        (m1, _, _), (_, b1d, D), (m2, _, _), (_, b2d, _) = o["cd"], o["cd_day"], o["ct"], o["ct_day"]
        sm, sb, wt = o["shrunk"]
        lines.append(f"| {lab} | {o['G']} | {D} | {m1:+.1f}c ±{b1d:.1f} | {m2:+.1f}c ±{b2d:.1f} | {sm:+.1f}c ±{sb:.1f} ({wt:.0%} live) |")
        if verdict is None and lab.startswith("CLEAN"):
            pause = m1 + b1d < 0
            verdict = ("PAUSE AND REVIEW (per-city-day band entirely below zero)" if pause else
                       "STEP UP one rung" if sm >= 2 else "HOLD" if sm >= 1 else "STEP BACK one rung")
    lines += ["", f"Frozen ladder rule on the clean test: **{verdict}** (applied at the weekly review, Wednesdays)."]
    return "\n".join(lines), verdict


def study_section():
    try:
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "study.py"), EVAL_FROM],
                           capture_output=True, text=True, timeout=3600)
        return p.stdout[-60000:] + (("\nSTDERR:\n" + p.stderr[-3000:]) if p.returncode else "")
    except Exception as e:
        return f"study failed: {e}"


def push(path, day):
    if not os.path.exists(TOKEN_FILE):
        print(f"no {TOKEN_FILE}: report written to {path}, not pushed"); return False
    import base64
    tok = open(TOKEN_FILE).read().strip()
    hdr = "Authorization: Basic " + base64.b64encode(f"x-access-token:{tok}".encode()).decode()
    g = ["git", "-C", REPO_DIR, "-c", f"http.extraHeader={hdr}", "-c", "credential.helper="]
    for cmd in (["add", "-A"], ["-c", "user.name=kmm-bot", "-c", "user.email=kmm-bot@localhost", "commit", "-qm", f"report {day}"],
                ["pull", "-q", "--rebase"], ["push", "-q"]):
        r = subprocess.run(g + cmd, capture_output=True, text=True)
        if r.returncode and cmd[0] != "commit":
            print(f"git {cmd[0]} failed: {r.stderr.replace(tok, '***')[:300]}"); return False
    return True


def main():
    from zoneinfo import ZoneInfo
    day = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    t0 = time.time()
    hl = health()
    bad = [l for ok, l in hl if not ok]
    try:
        edge_md, verdict = edge_section()
    except Exception as e:
        edge_md, verdict = f"edge failed: {e}", None
    st = study_section()
    md = [f"# Kalshi bot daily report -- {day}", "",
          f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC in {time.time() - t0:.0f}s. "
          f"{'**' + str(len(bad)) + ' health check(s) FAILED.**' if bad else 'All health checks passed.'}", "",
          "## Health (last 24h)", ""] + [f"- {'OK  ' if ok else 'FAIL'} {l}" for ok, l in hl] + \
         ["", "## Edge (kmm edge-now)", "", edge_md, "", f"## Study (clean test from {EVAL_FROM})", "", "```", st, "```", ""]
    os.makedirs(os.path.join(REPO_DIR, "reports"), exist_ok=True)
    path = os.path.join(REPO_DIR, "reports", f"{day}.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    pushed = push(path, day)
    print(f"report {path} ({'pushed' if pushed else 'local only'}); {len(bad)} failed checks; ladder: {verdict}")
    if bad:
        B.notify("Daily check FAILED:\n" + "\n".join(bad)[:900], title="Kalshi bot daily check", prio="high")


if __name__ == "__main__":
    main()
