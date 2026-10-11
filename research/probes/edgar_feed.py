"""Probe: does the EDGAR latest-8-K atom feed parse in Actions?"""
import sys, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_filings as L
from automation import shadow_filings as S
xml = L.sec_get(S.FEED.format(start=0)) or ""
print(f"::notice::feed bytes {len(xml)}; first 300: {xml[:300].replace(chr(10), ' ')}")
since = dt.datetime.now(S.ET) - dt.timedelta(days=4)
e = S.feed_entries(since)
print(f"::notice::entries since {since:%Y-%m-%d %H:%M}: {len(e)}; with study items: {sum(bool(set(x['items'].split(',')) & S.ITEMS) for x in e)}; sample {e[:2]}")
