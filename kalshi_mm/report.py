"""
kalshi_mm/report.py -- daily summary of the bot's logs, sent to the phone via ntfy (NTFY_CHANNEL).

The edge measurement: realized P&L per contract from settlements (Kalshi settlement records:
revenue minus the cost of the yes/no contracts held), compared with the backtest's ~1.3-1.9c for
deep fills. Also: fills today, balance, open positions, errors. Logs stay on the server (the code
repo is public, so account data is not pushed there).
"""
import datetime as dt, json, os, urllib.request

LOG = os.environ.get("LOG_DIR", "/home/kmm/data")


def rows(name):
    p = f"{LOG}/{name}.jsonl"
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def cents(d, key):
    """Kalshi fields come as integer cents or as *_dollars strings."""
    if d.get(f"{key}_dollars") is not None:
        return num(d[f"{key}_dollars"]) * 100
    return num(d.get(key))


def main():
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    fills = rows("fills")
    f_today = [f for f in fills if str((f.get("fill") or f).get("created_time", "")).startswith(today)]
    contracts_today = sum(num((f.get("fill") or f).get("count_fp", (f.get("fill") or f).get("count"))) for f in f_today)
    st = [s.get("settlement") or s for s in rows("settlements")]
    pnl_c, n_contracts = 0.0, 0.0
    for s in st:
        cost = cents(s, "yes_total_cost") + cents(s, "no_total_cost")
        pnl_c += cents(s, "revenue") - cost
        n_contracts += num(s.get("yes_count_fp", s.get("yes_count"))) + num(s.get("no_count_fp", s.get("no_count")))
    snaps = rows("snapshots")
    last = snaps[-1] if snaps else {}
    errs = [e for e in rows("errors") if str(e.get("ts", "")).startswith(today)]
    per = pnl_c / n_contracts if n_contracts else float("nan")
    msg = (f"{today} | mode {last.get('mode', '?')} | balance ${last.get('balance', 0):.2f}, value ${last.get('value', 0):.2f}\n"
           f"fills today: {len(f_today)} ({contracts_today:.0f} contracts)\n"
           f"settled to date: {len(st)} markets, {n_contracts:.0f} contracts, P&L ${pnl_c / 100:+.2f} "
           f"= {per:+.2f}c/contract (backtest deep fills ~1.3-1.9c)\n"
           f"open positions: {len(last.get('positions', {}))} | resting orders: {last.get('n_resting', 0)} | errors today: {len(errs)}")
    print(msg)
    ch = os.environ.get("NTFY_CHANNEL")
    if ch:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{ch}", data=msg.encode(),
                                                      headers={"Title": "Kalshi bot daily"}), timeout=20)


if __name__ == "__main__":
    main()
