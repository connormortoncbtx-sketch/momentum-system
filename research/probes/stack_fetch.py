"""research/probes/stack_fetch.py -- dry run of shadow #7's live data path (no writes)."""
import datetime as dt, sys, os
sys.path.insert(0, os.getcwd())
import automation.shadow_stack as SS
start = dt.date.today() - dt.timedelta(days=420)
px = SS.etf_daily(start); cp = SS.crypto_daily(start)
e, c = SS.etf_book(px), SS.crypto_book(cp)
print(f"::notice::ETF bars {px.shape} last {px.index[-1].date()} missing {sorted(set(SS.ETFS) - set(px.columns))}; "
      f"crypto bars {cp.shape} last {cp.index[-1].date()} missing {sorted(set(SS.COINS) - set(cp.columns))}")
print(f"::notice::last-week returns: ETF book {SS.window_ret(e, '2026-10-05', '2026-10-12') * 100:+.2f}%, "
      f"crypto book {SS.window_ret(c, '2026-10-05', '2026-10-12') * 100:+.2f}%, "
      f"1y Sharpe ETF {e.tail(252).mean() / e.tail(252).std() * 252 ** 0.5:.2f} crypto {c.tail(365).mean() / c.tail(365).std() * 365 ** 0.5:.2f}")
