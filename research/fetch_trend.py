"""research/fetch_trend.py -- daily bars for cross-asset ETFs (2016+) and crypto majors
(as far back as Alpaca has them) for time-series momentum / trend-following tests.
Output: research_data/trend/etf_daily.parquet, research_data/trend/crypto_daily.parquet"""
import datetime as dt, os
from pathlib import Path
import pandas as pd
OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "trend"
K, S = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
ETFS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "FXI", "TLT", "IEF", "SHY", "TIP", "LQD", "HYG",
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "UUP", "FXE", "FXY", "FXB", "VNQ", "XLE", "XLK", "XLF",
        "XLV", "XLU", "XLI", "XLY", "XLP", "XLB", "SMH", "XBI", "GDX", "COPX", "URA", "IBIT", "BITO"]
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "BCH/USD", "LINK/USD", "AVAX/USD", "DOGE/USD",
          "XRP/USD", "DOT/USD", "UNI/USD", "AAVE/USD", "SHIB/USD", "MKR/USD", "XTZ/USD", "GRT/USD"]


def main():
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    OUT.mkdir(parents=True, exist_ok=True)
    sc = StockHistoricalDataClient(K, S)
    e = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=ETFS, timeframe=TimeFrame.Day,
                          start=dt.datetime(2016, 1, 1), adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df.reset_index()
    e.to_parquet(OUT / "etf_daily.parquet", index=False)
    cc = CryptoHistoricalDataClient(K, S)
    c = cc.get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=CRYPTO, timeframe=TimeFrame.Day,
                           start=dt.datetime(2015, 1, 1))).df.reset_index()
    c.to_parquet(OUT / "crypto_daily.parquet", index=False)
    print(f"::notice::trend data: ETFs {e.symbol.nunique()} ({e.timestamp.min():%Y-%m}..), "
          f"crypto {c.symbol.nunique()} ({c.timestamp.min():%Y-%m}..), rows {len(e):,}/{len(c):,}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-10:]:
            print(f"::error::{line}")
        raise
