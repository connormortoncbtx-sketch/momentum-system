"""Probe: what option data can this Alpaca account get? Chain snapshots (IV/greeks),
expired contract lists, and historical bars for expired contracts (2024-25)."""
import os, datetime as dt
K, S = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
def note(msg): print(f"::notice::{msg}"[:3000])
try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest, OptionBarsRequest, OptionTradesRequest
    from alpaca.data.timeframe import TimeFrame
    oc = OptionHistoricalDataClient(K, S)
    ch = oc.get_option_chain(OptionChainRequest(underlying_symbol="SOFI"))
    keys = list(ch)[:3]
    s0 = ch[keys[0]] if keys else None
    note(f"chain SOFI: {len(ch)} contracts; sample {keys[0] if keys else None}: IV={getattr(s0,'implied_volatility',None)} "
         f"greeks={getattr(s0,'greeks',None)} quote={getattr(s0,'latest_quote',None)}")
except Exception as e:
    note(f"chain error: {type(e).__name__}: {e}")
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import AssetStatus
    tc = TradingClient(K, S, paper=True)
    r = tc.get_option_contracts(GetOptionContractsRequest(underlying_symbols=["SOFI"], status=AssetStatus.INACTIVE,
                                expiration_date_gte=dt.date(2024, 3, 1), expiration_date_lte=dt.date(2024, 4, 30), limit=100))
    cs = r.option_contracts or []
    note(f"expired SOFI contracts Mar-Apr 2024: {len(cs)}; sample {[c.symbol for c in cs[:5]]}; next_page {bool(r.next_page_token)}")
    sym = cs[len(cs)//2].symbol if cs else None
except Exception as e:
    note(f"contracts error: {type(e).__name__}: {e}"); sym = None
for s in [sym, "AAPL240621C00200000", "SOFI250117C00010000"]:
    if not s:
        continue
    try:
        b = oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day,
                               start=dt.datetime(2024, 1, 1), end=dt.datetime(2025, 2, 1))).df
        note(f"bars {s}: {len(b)} daily bars; first {b.index[0] if len(b) else None}; cols {list(b.columns)}; "
             f"tail {b.tail(2).to_dict('records') if len(b) else None}")
    except Exception as e:
        note(f"bars {s} error: {type(e).__name__}: {e}")
    try:
        t = oc.get_option_trades(OptionTradesRequest(symbol_or_symbols=s, start=dt.datetime(2024, 3, 1),
                                 end=dt.datetime(2024, 3, 8), limit=5)).df
        note(f"trades {s}: {len(t)} rows sample {t.head(2).to_dict('records') if len(t) else None}")
    except Exception as e:
        note(f"trades {s} error: {type(e).__name__}: {e}")
