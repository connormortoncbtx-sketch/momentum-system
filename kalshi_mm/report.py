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


def dollars_field(d, key):
    """Price/fee in CENTS from whichever form Kalshi sent: <key>_dollars / <key>_fixed (dollars), or <key>
    (integer cents, or a dollar string like '0.0175' in newer responses)."""
    for k in (f"{key}_dollars", f"{key}_fixed"):
        if d.get(k) is not None:
            return num(d[k]) * 100
    v = d.get(key)
    if isinstance(v, str) and "." in v:
        return num(v) * 100
    return num(v)


def market_results(fills=None, st=None):
    """{ticker: 1.0/0.0} for every market the bot ever filled in that has a result. Settlement records only exist
    where a position was still held at settlement; markets fully traded in and out never get one, so their result is
    looked up directly (public market endpoint) and cached in LOG_DIR/results_cache.json. Unknown/open -> absent."""
    fills = rows("fills") if fills is None else fills
    st = [s.get("settlement") or s for s in rows("settlements")] if st is None else st
    out = {}
    for s in st:
        r = str(s.get("market_result", "")).lower()
        if r in ("yes", "no"):
            out[s.get("ticker")] = 1.0 if r == "yes" else 0.0
    path = os.path.join(LOG, "results_cache.json")
    try:
        cache = json.load(open(path))
    except Exception:
        cache = {}
    out.update({k: v for k, v in cache.items() if v is not None})
    need = {(f.get("fill") or f).get("ticker") for f in fills} - set(out)
    today = dt.datetime.now(dt.timezone.utc).date()
    try:
        import bot as B
        k = B.Kalshi("dry")                                     # public endpoint, no keys needed
    except Exception:
        k = None
    changed = False
    for tk in sorted(t for t in need if t):
        try:                                                    # only markets whose date has passed
            d = tk.split("-")[1]
            mday = dt.date(2000 + int(d[:2]), ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                               "NOV", "DEC"].index(d[2:5]) + 1, int(d[5:7]))
            if mday >= today or k is None:
                continue
            m = k.req("GET", f"/markets/{tk}").get("market", {})
        except Exception:
            continue
        r = str(m.get("result", "")).lower()
        if r in ("yes", "no"):
            out[tk] = cache[tk] = 1.0 if r == "yes" else 0.0; changed = True
    if changed:
        json.dump(cache, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return out


def build():
    """The report text -- shared by the daily ntfy report and the dashboard."""
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    fills = rows("fills")
    f_today = [f for f in fills if str((f.get("fill") or f).get("created_time", "")).startswith(today)]
    contracts_today = sum(num((f.get("fill") or f).get("count_fp", (f.get("fill") or f).get("count"))) for f in f_today)
    st = [s.get("settlement") or s for s in rows("settlements")]
    pnl_c, n_contracts = 0.0, 0.0
    for s in st:
        cost = dollars_field(s, "yes_total_cost") + dollars_field(s, "no_total_cost")
        pnl_c += dollars_field(s, "revenue") - cost
        n_contracts += num(s.get("yes_count_fp", s.get("yes_count"))) + num(s.get("no_count_fp", s.get("no_count")))
    # The research's measure: EVERY fill in a settled market, valued at the result. Settlement records alone only
    # show the net position left at settlement; round trips (bought and later sold) net out before that, so the
    # settlement line on its own leans toward the inventory that went against the bot.
    result = market_results(fills, st)
    f_pnl, f_n, f_cnt, f_fee = 0.0, 0, 0.0, 0.0
    city = {}                                                   # series -> city-day -> [pnl cents, contracts]
    for f in fills:
        x = f.get("fill") or f
        R = result.get(x.get("ticker"))
        if R is None:
            continue
        c = num(x.get("count_fp", x.get("count")))
        side, act = str(x.get("side", "")).lower(), str(x.get("action", "")).lower()
        p = dollars_field(x, "yes_price" if side == "yes" else "no_price") / 100
        win = R if side == "yes" else 1 - R                     # payout of the side traded
        fee = dollars_field(x, "fee_cost")
        pl = (win - p if act == "buy" else p - win) * c * 100
        f_pnl += pl; f_fee += fee
        f_n += 1; f_cnt += c
        tk = str(x.get("ticker", ""))
        cd = city.setdefault(tk.split("-")[0], {}).setdefault(tk.rsplit("-", 1)[0], [0.0, 0.0])
        cd[0] += pl - fee; cd[1] += c
    snaps = rows("snapshots")
    last = snaps[-1] if snaps else {}
    errs = [e for e in rows("errors") if str(e.get("ts", "")).startswith(today)
            and "post only cross" not in str(e.get("error", ""))]      # older logs: book-moved refusals aren't errors
    pocs = sum(1 for e in rows("events") if e.get("event") == "post_only_cross" and str(e.get("ts", "")).startswith(today))
    per = pnl_c / n_contracts if n_contracts else float("nan")
    dep = num(os.environ.get("DEPOSITS_USD"))                  # net money put in (deposits - withdrawals), set in /etc/kmm/env
    # conservative: open positions at what they'd sell for NOW (the bid); the mid flatters dying buckets late in the day
    v, how = (last["value_at_bid"], "open positions at the bid, i.e. sold now") if "value_at_bid" in last else \
             (last.get("value", 0), "open positions at mid -- optimistic until the bot logs value_at_bid")
    score = (f"BOTTOM LINE: {'+' if v >= dep else '-'}${abs(v - dep):.2f} overall "
             f"(${v:.2f} vs ${dep:.2f} deposited; {how})\n") if dep else \
            "BOTTOM LINE: set DEPOSITS_USD in /etc/kmm/env to see profit vs money put in\n"
    msg = (score + f"{today} | mode {last.get('mode', '?')} | balance ${last.get('balance', 0):.2f}, value ${last.get('value', 0):.2f}\n"
           f"fills today: {len(f_today)} ({contracts_today:.0f} contracts)\n"
           f"ALL fills in finished markets: {f_n} fills, {f_cnt:.0f} contracts, P&L ${(f_pnl - f_fee) / 100:+.2f} "
           f"= {(f_pnl - f_fee) / f_cnt if f_cnt else float('nan'):+.2f}c/contract after ${f_fee / 100:.2f} fees "
           f"(target from per-fill study: +2-3c)\n"
           f"  of which still held at settlement: {len(st)} markets, {n_contracts:.0f} contracts, ${pnl_c / 100:+.2f} "
           f"= {per:+.2f}c/contract (the leftover inventory, biased toward losers)\n"
           f"budget now: ${last.get('budget', 0):.2f} | open positions: {len(last.get('positions', {}))} | resting orders: {last.get('n_resting', 0)} | errors today: {len(errs)}"
           f" | post-only refusals (book moved, not errors): {pocs}")
    # per city: edge per contract and its uncertainty from city-day to city-day. Watch only -- a city is a CUT
    # candidate only after >= 60 city-days with the edge clearly below zero (research: city differences are noise)
    lines = []
    for ser, days in sorted(city.items()):
        per_day = [v[0] / v[1] for v in days.values() if v[1] > 0]
        tot_p, tot_c = sum(v[0] for v in days.values()), sum(v[1] for v in days.values())
        n = len(per_day)
        mean = sum(per_day) / n if n else float("nan")
        se = (sum((x - mean) ** 2 for x in per_day) / (n - 1)) ** 0.5 / n ** 0.5 if n > 2 else float("nan")
        flag = " <- CUT candidate" if n >= 60 and mean + 2 * se < 0 else ""
        lines.append(f"  {ser.replace('KXHIGHT', '').replace('KXHIGH', '')}: {n} city-days, {tot_c:.0f} contracts, "
                     f"{tot_p / tot_c if tot_c else float('nan'):+.1f}c/contract"
                     + (f" (city-day avg {mean:+.1f}c +/- {se:.1f})" if n > 2 else "") + flag)
    if lines:
        msg += "\nby city (settled; watch only until ~60 city-days each):\n" + "\n".join(lines)
    return msg


def main():
    msg = build()
    print(msg)
    ch = os.environ.get("NTFY_CHANNEL")
    if ch:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{ch}", data=msg.encode(),
                                                      headers={"Title": "Kalshi bot daily"}), timeout=20)


if __name__ == "__main__":
    main()
