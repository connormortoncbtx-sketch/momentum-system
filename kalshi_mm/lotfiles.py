"""
kalshi_mm/lotfiles.py -- what belongs to the weekly roll (lottery.py), read from its own files (Oct 24, 2026).

The roll shares the Kalshi account with the market maker but is meant to be wholly separate from it. Kalshi
subaccounts would do that natively but need the Advanced API tier, so the split is done here instead:
  tickers()       every market the roll ever ordered in (lottery.jsonl order events + lottery.json), so the bot's
                  caps / budget / loss stop and every report can leave them out
  cash_in_bal()   how much of Kalshi's cash balance is the roll's money right now (its unspent bankroll, or a payout
                  that has landed but the roll hasn't booked yet), so the bot's account value excludes it
Verified Oct 24 from the logs: a resting order does NOT lower Kalshi's cash balance (231 order changes without a
fill: balance moved in 7, corr 0.1), so cash is only spent on fills. Stdlib only (imported by bot.py and report.py).
"""
import json
import os

LOG = os.environ.get("LOG_DIR", "/home/kmm/data")
_cache = {}


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def state(log=None):
    """lottery.json as a dict ({} if absent or unreadable)."""
    p = os.path.join(log or LOG, "lottery.json")
    key = ("state", p, _mtime(p))
    if key not in _cache:
        try:
            _cache[key] = json.load(open(p))
        except Exception:
            _cache[key] = {}
    return _cache[key]


def tickers(log=None):
    """Every ticker the roll has ordered in, ever."""
    log = log or LOG
    pj, ps = os.path.join(log, "lottery.jsonl"), os.path.join(log, "lottery.json")
    key = ("tickers", log, _mtime(pj), _mtime(ps))
    if key not in _cache:
        out = set()
        if os.path.exists(pj):
            for line in open(pj):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("ticker"):
                    out.add(r["ticker"])
        st = state(log)
        if st.get("ticker"):
            out.add(st["ticker"])
        out.update(h.get("ticker") for h in st.get("history", []) if h.get("ticker"))
        _cache[key] = frozenset(out)
    return _cache[key]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def cash_in_bal(st, live_pos, result=None):
    """The roll's money sitting in Kalshi's cash balance right now.
    st: lottery.json; live_pos: {ticker: contracts} from the account; result: YES value (1/0) of the roll's market
    once decided (used only when its position has already been paid out). lottery.json is updated every 15 min, so
    fills and payouts since its last pass are reconstructed from the live position, not taken from the file."""
    if st.get("status") not in ("buying", "holding"):
        return 0.0                                  # idle / done: whatever is left is back in the common pool
    cash, held, px, tk = num(st.get("cash")), num(st.get("held")), num(st.get("price")), st.get("ticker")
    if not tk:
        return max(0.0, cash)
    p = num(live_pos.get(tk, 0.0))
    if p <= 0 and held > 0:                         # position gone = settled; the payout is already in cash
        return max(0.0, cash + held * (px if result is None else result))
    return max(0.0, cash - (p - held) * px)         # fills since its last pass were paid at its order price
