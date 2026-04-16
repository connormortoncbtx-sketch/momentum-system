"""
Stage 1 — Universe Builder
===========================
First run:  full bootstrap — fetches all NYSE+NASDAQ tickers, applies
            liquidity gate, writes data/universe.csv

Subsequent runs: upsert mode
    - Updates price/vol/mcap for existing symbols (fast)
    - Checks for new symbols not yet in universe, inserts passing ones
    - Flags symbols failing the liquidity gate consecutively
    - Removes symbols that have failed 3 weeks in a row

Ticker sources (in priority order):
    1. nasdaqtrader.com NASDAQ listing
    2. nasdaqtrader.com NYSE/other listing (with retry + timeout)
    3. Wikipedia S&P 500 + Russell 1000 (fallback if FTP fails)

Run:  python pipeline/01_universe.py
      python pipeline/01_universe.py --force   # full rebuild
"""

import os
import time
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

MIN_PRICE         = 5.0
MIN_AVG_VOL       = 500_000
MIN_MARKET_CAP    = 300_000_000
MIN_HISTORY_DAYS  = 60
BATCH_SIZE        = 100
BATCH_SLEEP       = 1.0
MAX_GATE_FAILURES = 3
CACHE_MAX_AGE_DAYS = 7

DATA_DIR   = Path("data")
OUTPUT_RAW = DATA_DIR / "universe_raw.csv"
OUTPUT     = DATA_DIR / "universe.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def fetch_nasdaq_ftp():
    urls = [
        "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv",
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    ]
    try:
        df = pd.read_csv(urls[0])
        df = df.rename(columns={"Symbol":"symbol","Company Name":"name"})
        df["exchange"]="NASDAQ"; df["sector"]=""; df["industry"]=""
        log.info(f"  NASDAQ (GitHub): {len(df):,}")
        return df[["symbol","name","exchange","sector","industry"]]
    except Exception:
        pass
    try:
        import urllib.request
        with urllib.request.urlopen(urls[1], timeout=20) as r:
            raw = r.read().decode("utf-8")
        rows=[]
        for line in raw.strip().split("\n")[1:]:
            parts=line.split("|")
            if len(parts)<3 or parts[0]=="File Creation Time": continue
            rows.append({"symbol":parts[0].strip(),"name":parts[1].strip(),"exchange":"NASDAQ","sector":"","industry":""})
        log.info(f"  NASDAQ (FTP): {len(rows):,}")
        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"  NASDAQ failed: {e}")
        return pd.DataFrame(columns=["symbol","name","exchange","sector","industry"])

def fetch_nyse_ftp(retries=3, timeout=25):
    import urllib.request

    # Primary: GitHub-hosted NYSE listings CSV (same pattern as NASDAQ fetch)
    # This avoids the unreliable NASDAQ FTP server for NYSE data
    github_urls = [
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full_tickers.json",
        "https://raw.githubusercontent.com/datasets/nyse-listings/master/data/nyse-listed-symbols.csv",
    ]

    # Try GitHub CSV first
    for github_url in github_urls:
        try:
            req = urllib.request.Request(
                github_url,
                headers={"User-Agent": "momentum-system/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode("utf-8")

            if github_url.endswith(".json"):
                import json
                data = json.loads(raw)
                # rreichel3 format: list of {ticker, name, exchange}
                rows = []
                items = data if isinstance(data, list) else list(data.values())
                for item in items:
                    if isinstance(item, dict):
                        sym = item.get("ticker") or item.get("symbol") or ""
                    else:
                        sym = str(item)
                    if sym:
                        rows.append({
                            "symbol": sym.strip(),
                            "name": item.get("name", "") if isinstance(item, dict) else "",
                            "exchange": "NYSE",
                            "sector": "",
                            "industry": "",
                        })
            else:
                # CSV format
                import io
                df = pd.read_csv(io.StringIO(raw))
                # Detect symbol column
                sym_col = next((c for c in df.columns if c.lower() in
                               ["symbol", "ticker", "act symbol"]), df.columns[0])
                name_col = next((c for c in df.columns if "name" in c.lower()), None)
                rows = []
                for _, row in df.iterrows():
                    sym = str(row[sym_col]).strip()
                    name = str(row[name_col]).strip() if name_col else ""
                    if sym and sym != "nan":
                        rows.append({
                            "symbol": sym,
                            "name": name,
                            "exchange": "NYSE",
                            "sector": "",
                            "industry": "",
                        })

            if rows:
                log.info(f"  NYSE (GitHub): {len(rows):,}")
                return pd.DataFrame(rows)
        except Exception as e:
            log.debug(f"  NYSE GitHub source failed: {e}")

    # Fallback: NASDAQ FTP (unreliable but worth trying)
    url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    for attempt in range(1, retries+1):
        try:
            log.info(f"  NYSE FTP attempt {attempt}/{retries}...")
            with urllib.request.urlopen(url, timeout=timeout) as r:
                raw = r.read().decode("utf-8")
            rows = []
            for line in raw.strip().split("\n")[1:]:
                parts = line.split("|")
                if len(parts) < 3 or parts[0] == "File Creation Time":
                    continue
                rows.append({
                    "symbol":   parts[0].strip(),
                    "name":     parts[1].strip(),
                    "exchange": parts[2].strip(),
                    "sector":   "",
                    "industry": "",
                })
            log.info(f"  NYSE/other (FTP): {len(rows):,}")
            return pd.DataFrame(rows)
        except Exception as e:
            log.warning(f"  NYSE attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(3)
    log.warning("  NYSE FTP failed — using Wikipedia fallback")
    return fetch_wikipedia_fallback()

def fetch_wikipedia_fallback():
    rows=[]
    sources=[
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",0,"NYSE"),
        ("https://en.wikipedia.org/wiki/Russell_1000_Index",2,"NYSE"),
    ]
    for url,table_idx,exchange in sources:
        try:
            tables=pd.read_html(url)
            df=tables[table_idx]
            sym_col=next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),None)
            name_col=next((c for c in df.columns if "compan" in c.lower() or "security" in c.lower() or "name" in c.lower()),None)
            if sym_col:
                for _,row in df.iterrows():
                    sym=str(row[sym_col]).strip().replace(".","- ")
                    name=str(row[name_col]).strip() if name_col else ""
                    if sym and sym!="nan":
                        rows.append({"symbol":sym,"name":name,"exchange":exchange,"sector":"","industry":""})
            log.info(f"  Wikipedia {url[-30:]}: {len(df)} rows")
        except Exception as e:
            log.warning(f"  Wikipedia failed {url}: {e}")
    log.info(f"  Wikipedia total: {len(rows)}")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol","name","exchange","sector","industry"])

def combine_and_clean(nasdaq, nyse):
    df=pd.concat([nasdaq,nyse],ignore_index=True)
    df["symbol"]=df["symbol"].str.strip().str.upper()
    df=df[df["symbol"].str.len()>0]
    bad=(
        df["symbol"].str.endswith("W") |   # warrants
        df["symbol"].str.endswith("U") |   # units
        df["symbol"].str.endswith("R") |   # rights
        df["symbol"].str.contains(r"\.", regex=True) |  # BRK.B style
        df["symbol"].str.contains("-",  regex=False) |  # preferred shares
        df["symbol"].str.startswith("ZZ") |
        df["symbol"].str.contains("FILE", regex=False) |
        # Debt instruments and baby bonds — typically end in L, Z, or are
        # 5+ char symbols ending in specific patterns
        (df["symbol"].str.len() >= 5) & (
            df["symbol"].str.endswith("L") |   # notes/preferreds (RILYN, RILYL etc)
            df["symbol"].str.endswith("Z") |   # notes (RILYZ, ATLCZ etc)
            df["symbol"].str.endswith("I") |   # notes (TRINI, INBKZ etc)
            df["symbol"].str.endswith("G") |   # notes (RWAYZ, DCOMG etc)
            df["symbol"].str.endswith("H") |   # notes
            df["symbol"].str.endswith("M") |   # notes (BPOPM etc)
            df["symbol"].str.endswith("P")     # preferred (BANFP etc)
        )
    )
    df=df[~bad].drop_duplicates(subset="symbol",keep="first").reset_index(drop=True)
    log.info(f"  Combined: {len(df):,} after dedup+filter")
    return df

def gate_batch(symbols):
    end=datetime.date.today()
    start=end-datetime.timedelta(days=90)
    try:
        raw=yf.download(symbols,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=True)
    except Exception as e:
        log.warning(f"Batch error: {e}"); return []
    passed=[]
    if len(symbols)==1:
        sym=symbols[0]
        try:
            close=raw["Close"].dropna(); vol=raw["Volume"].dropna()
            if len(close)<MIN_HISTORY_DAYS: return []
            p=float(close.iloc[-1]); v=float(vol.tail(20).mean())
            if p>=MIN_PRICE and v>=MIN_AVG_VOL:
                passed.append({"symbol":sym,"last_price":round(p,2),"avg_vol_20d":int(v),"history_days":len(close)})
        except Exception: pass
        return passed
    for sym in symbols:
        try:
            close=raw["Close"][sym].dropna(); vol=raw["Volume"][sym].dropna()
            if len(close)<MIN_HISTORY_DAYS: continue
            p=float(close.iloc[-1]); v=float(vol.tail(20).mean())
            if p>=MIN_PRICE and v>=MIN_AVG_VOL:
                passed.append({"symbol":sym,"last_price":round(p,2),"avg_vol_20d":int(v),"history_days":len(close)})
        except Exception: continue
    return passed

def run_liquidity_gate(symbols):
    passed=[]; batches=[symbols[i:i+BATCH_SIZE] for i in range(0,len(symbols),BATCH_SIZE)]
    log.info(f"  Gate: {len(symbols):,} symbols, {len(batches)} batches...")
    for i,batch in enumerate(batches,1):
        if i%10==0: log.info(f"    Batch {i}/{len(batches)}")
        passed.extend(gate_batch(batch))
        if i<len(batches): time.sleep(BATCH_SLEEP)
    return pd.DataFrame(passed)

def fetch_market_caps(symbols):
    """
    Fetch market caps using a reliable fallback chain:
    1. fast_info.market_cap (fast but unreliable)
    2. fast_info.shares * fast_info.last_price (calculated, more reliable)
    3. info["marketCap"] (slow but most reliable)
    """
    caps={}
    need_fallback=[]

    # Tier 1: fast_info.market_cap
    for i,sym in enumerate(symbols):
        if i%100==0 and i>0:
            time.sleep(0.5)
        try:
            fi=yf.Ticker(sym).fast_info
            cap=getattr(fi,"market_cap",None)
            if cap and cap > 1_000_000:   # valid if > $1M
                caps[sym]=int(cap)
            else:
                # Try calculated: shares * price
                shares=getattr(fi,"shares",None)
                price=getattr(fi,"last_price",None) or getattr(fi,"previous_close",None)
                if shares and price and shares > 0 and price > 0:
                    caps[sym]=int(shares*price)
                else:
                    need_fallback.append(sym)
        except Exception:
            need_fallback.append(sym)

    # Tier 2: full info dict for anything that failed tier 1
    if need_fallback:
        log.info(f"    Falling back to full info for {len(need_fallback)} symbols...")
        for i,sym in enumerate(need_fallback):
            if i%50==0 and i>0:
                time.sleep(1.0)
            try:
                info=yf.Ticker(sym).info
                cap=info.get("marketCap",0) or 0
                caps[sym]=int(cap) if cap > 0 else 0
            except Exception:
                caps[sym]=0

    return caps

def enrich_sectors(symbols):
    out={}
    for sym in symbols:
        try:
            info=yf.Ticker(sym).info
            out[sym]={"sector":info.get("sector",""),"industry":info.get("industry","")}
        except Exception: out[sym]={"sector":"","industry":""}
    return out

def update_existing(master):
    symbols=master["symbol"].tolist()
    log.info(f"  Updating {len(symbols):,} existing symbols...")
    batches=[symbols[i:i+BATCH_SIZE] for i in range(0,len(symbols),BATCH_SIZE)]
    updates={}
    end=datetime.date.today(); start=end-datetime.timedelta(days=30)
    for i,batch in enumerate(batches,1):
        if i%10==0: log.info(f"    Batch {i}/{len(batches)}")
        try:
            raw=yf.download(batch,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=True)
            for sym in batch:
                try:
                    close=raw["Close"][sym].dropna() if len(batch)>1 else raw["Close"].dropna()
                    vol=raw["Volume"][sym].dropna() if len(batch)>1 else raw["Volume"].dropna()
                    if len(close)>0:
                        updates[sym]={"last_price":round(float(close.iloc[-1]),2),"avg_vol_20d":int(vol.tail(20).mean())}
                except Exception: continue
        except Exception as e: log.warning(f"  Update batch error: {e}")
        time.sleep(BATCH_SLEEP*0.5)
    for sym,vals in updates.items():
        idx=master[master["symbol"]==sym].index
        if len(idx):
            master.loc[idx,"last_price"]=vals["last_price"]
            master.loc[idx,"avg_vol_20d"]=vals["avg_vol_20d"]

    # Refresh missing or zero market caps
    # Coerce to numeric first — injected symbols may have NaN stored as string
    master["market_cap"] = pd.to_numeric(master["market_cap"], errors="coerce")
    missing_cap=master[master["market_cap"].isna()|(master["market_cap"]==0)]["symbol"].tolist()
    if missing_cap:
        log.info(f"  Refreshing {len(missing_cap):,} missing market caps...")
        # Process in chunks to avoid rate limiting
        chunk_size = 500
        for chunk_start in range(0, len(missing_cap), chunk_size):
            chunk = missing_cap[chunk_start:chunk_start+chunk_size]
            caps=fetch_market_caps(chunk)
            for sym,cap in caps.items():
                idx=master[master["symbol"]==sym].index
                if len(idx): master.loc[idx,"market_cap"]=cap
            if chunk_start + chunk_size < len(missing_cap):
                log.info(f"  Market cap chunk {chunk_start//chunk_size+1} done, continuing...")
                time.sleep(2.0)

    # Backfill missing sector data — fetch in small batches to avoid rate limits
    missing_sector=master[
        master["sector"].isna() | (master["sector"]=="") | (master["sector"]=="Unknown")
    ]["symbol"].tolist()
    if missing_sector:
        log.info(f"  Backfilling sectors for {len(missing_sector):,} symbols...")
        # Process in chunks of 200 to avoid timing out Stage 1
        chunk_size = 200
        chunk = missing_sector[:chunk_size]
        if len(missing_sector) > chunk_size:
            log.info(f"  (processing first {chunk_size} this run, remainder next week)")
        sectors=enrich_sectors(chunk)
        for sym,data in sectors.items():
            if data.get("sector"):
                idx=master[master["symbol"]==sym].index
                if len(idx):
                    master.loc[idx,"sector"]=data["sector"]
                    master.loc[idx,"industry"]=data.get("industry","")

    return master

def check_gate_failures(master):
    if "gate_failures" not in master.columns: master["gate_failures"]=0
    failing=(master["last_price"]<MIN_PRICE)|(master["avg_vol_20d"]<MIN_AVG_VOL)|(master["market_cap"]<MIN_MARKET_CAP)
    master.loc[failing,"gate_failures"]+=1
    master.loc[~failing,"gate_failures"]=0
    before=len(master)
    master=master[master["gate_failures"]<MAX_GATE_FAILURES].copy()
    removed=before-len(master)
    if removed: log.info(f"  Removed {removed} symbols after {MAX_GATE_FAILURES} gate failures")
    return master

def find_new_symbols(raw_df, master):
    existing=set(master["symbol"].tolist())
    new=[s for s in raw_df["symbol"].tolist() if s not in existing]
    log.info(f"  New symbols to evaluate: {len(new):,}")
    return new

def insert_new_symbols(new_syms, raw_df, master):
    if not new_syms: return master
    gate_df=run_liquidity_gate(new_syms)
    if gate_df.empty: return master
    caps=fetch_market_caps(gate_df["symbol"].tolist())
    gate_df["market_cap"]=gate_df["symbol"].map(caps).fillna(0)
    gate_df=gate_df[gate_df["market_cap"]>=MIN_MARKET_CAP].copy()
    if gate_df.empty: log.info("  No new symbols passed gate"); return master
    meta=raw_df[["symbol","name","exchange"]].drop_duplicates("symbol")
    gate_df=gate_df.merge(meta,on="symbol",how="left")
    sectors=enrich_sectors(gate_df["symbol"].tolist())
    gate_df["sector"]=gate_df["symbol"].map(lambda s: sectors.get(s,{}).get("sector",""))
    gate_df["industry"]=gate_df["symbol"].map(lambda s: sectors.get(s,{}).get("industry",""))
    gate_df["gate_failures"]=0
    gate_df["as_of"]=str(datetime.date.today())
    log.info(f"  Inserting {len(gate_df)} new symbols")
    return pd.concat([master,gate_df],ignore_index=True)

def full_bootstrap(raw_df):
    gate_df=run_liquidity_gate(raw_df["symbol"].tolist())
    if gate_df.empty: log.error("Empty after gate"); return pd.DataFrame()
    merged=raw_df.merge(gate_df,on="symbol",how="inner")
    log.info(f"  Gate passed: {len(merged):,}")
    log.info("  Fetching market caps...")
    caps=fetch_market_caps(merged["symbol"].tolist())
    merged["market_cap"]=merged["symbol"].map(caps).fillna(0)
    merged=merged[merged["market_cap"]>=MIN_MARKET_CAP].copy()
    log.info(f"  Mcap filter passed: {len(merged):,}")
    try:
        log.info("  Enriching sectors...")
        sectors=enrich_sectors(merged["symbol"].tolist())
        merged["sector"]=merged["symbol"].map(lambda s: sectors.get(s,{}).get("sector",""))
        merged["industry"]=merged["symbol"].map(lambda s: sectors.get(s,{}).get("industry",""))
    except Exception as e: log.warning(f"  Sector enrichment failed: {e}")
    merged["gate_failures"]=0
    return merged

def run(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0=time.time()
    log.info("="*60)
    log.info("STAGE 1 — Universe Builder")
    log.info("="*60)
    today=datetime.date.today()

    upsert_mode=False
    if not force and OUTPUT.exists():
        master=pd.read_csv(OUTPUT)
        if "as_of" in master.columns:
            last_run=pd.to_datetime(master["as_of"].max()).date()
            age=(today-last_run).days
            if age<CACHE_MAX_AGE_DAYS:
                log.info(f"Universe {age}d old — using cache ({len(master):,} symbols)")
                return master
        upsert_mode=True
        log.info(f"Existing universe ({len(master):,}) — upsert mode")
    else:
        master=pd.DataFrame()
        log.info("No universe found — full bootstrap")

    log.info("Fetching ticker lists...")
    nasdaq=fetch_nasdaq_ftp()
    nyse=fetch_nyse_ftp()
    raw_df=combine_and_clean(nasdaq,nyse)
    raw_df.to_csv(OUTPUT_RAW,index=False)

    if upsert_mode:
        log.info("Updating existing symbols...")
        master=update_existing(master)
        log.info("Checking gate failures...")
        master=check_gate_failures(master)
        log.info("Checking for new symbols...")
        new_syms=find_new_symbols(raw_df,master)
        if new_syms: master=insert_new_symbols(new_syms,raw_df,master)
    else:
        master=full_bootstrap(raw_df)
        if master.empty: log.error("Bootstrap failed"); return

    master=master.sort_values("avg_vol_20d",ascending=False).reset_index(drop=True)
    master["as_of"]=str(today)
    cols=["symbol","name","exchange","sector","industry","last_price","avg_vol_20d","market_cap","history_days","gate_failures","as_of"]
    master=master[[c for c in cols if c in master.columns]]
    master.to_csv(OUTPUT,index=False)

    elapsed=time.time()-t0
    log.info("="*60)
    log.info(f"Done in {elapsed/60:.1f} min — {len(master):,} symbols → {OUTPUT}")
    log.info("="*60)
    if "sector" in master.columns:
        for sector,count in master["sector"].value_counts().head(8).items():
            log.info(f"  {str(sector):<35} {count:>5}")
    return master

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true",help="Force full rebuild")
    args=parser.parse_args()
    run(force=args.force)
