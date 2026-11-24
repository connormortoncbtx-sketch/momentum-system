"""Live-account options readiness probe (Nov 24, 2026). Prints capabilities only -- no dollar amounts or positions
(public repo). Checks: options levels, margin vs cash, PDT flag, whether one $1-wide condor's max loss is coverable,
and whether SPY / XSP same-day option contracts are listed and tradable at Alpaca."""
import datetime as dt, os, requests
H = {"APCA-API-KEY-ID": os.environ["KL"], "APCA-API-SECRET-KEY": os.environ["SL"]}
B = "https://api.alpaca.markets"
a = requests.get(f"{B}/v2/account", headers=H, timeout=30).json()
ob = float(a.get("options_buying_power") or 0)
print(f"::notice::status {a.get('status')}; options approved level {a.get('options_approved_level')}, trading level "
      f"{a.get('options_trading_level')}; multiplier {a.get('multiplier')} ({'margin' if float(a.get('multiplier') or 1) >= 2 else 'cash'} account); "
      f"PDT flag {a.get('pattern_day_trader')}; day trades in last 5 days {a.get('daytrade_count')}")
print(f"::notice::options buying power covers a $1-wide condor (~$100): {ob >= 100}; covers two: {ob >= 200}; covers a $5-wide (~$500): {ob >= 500}")
today = dt.date.today()
for und in ("SPY", "XSP"):
    r = requests.get(f"{B}/v2/options/contracts", headers=H, timeout=30,
                     params={"underlying_symbols": und, "expiration_date_gte": str(today),
                             "expiration_date_lte": str(today + dt.timedelta(days=7)), "limit": 1000})
    js = r.json() if r.ok else {}
    cs = js.get("option_contracts") or []
    exps = sorted({c["expiration_date"] for c in cs})
    trad = sum(1 for c in cs if c.get("tradable"))
    styles = sorted({c.get("style", "?") for c in cs})
    print(f"::notice::{und}: HTTP {r.status_code}; contracts listed {len(cs)} (tradable {trad}); expiries next 7d {exps[:6]}; style {styles}"
          + ("" if r.ok else f"; error {str(js)[:160]}"))
