"""
research/fetch_wiki.py -- daily English-Wikipedia page views per listed company (attention), Nov 15, 2026.
1. Wikidata SPARQL (public API): items with a ticker symbol (P249, directly or as a stock-exchange qualifier) and an
   English Wikipedia article -> ticker -> article title; kept only for symbols in our weekly panel.
2. Wikimedia REST pageviews API (public): daily user views per article, 2016-01-01 -> yesterday.
Output: research_data/wiki/ticker_articles.csv, research_data/wiki/pageviews.parquet (symbol, date, views).
"""
import concurrent.futures as cf
import datetime as dt
import glob
import json
import os
import threading
import time
import urllib.parse
import urllib.request

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
UA = "MomentumAlphaResearch/1.0 (momentum-bot@users.noreply.github.com)"
LIMIT = int(os.environ.get("LIMIT") or 0)
Q = """
SELECT DISTINCT ?ticker ?article WHERE {
  { ?item p:P414 ?st . ?st pq:P249 ?ticker . } UNION { ?item wdt:P249 ?ticker . }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
}"""


def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if "404" in str(e):
                return None
            time.sleep(3 * (k + 1))
    return None


def main():
    raw = get("https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(Q), timeout=300)
    B = json.loads(raw)["results"]["bindings"]
    M = pd.DataFrame({"symbol": [b["ticker"]["value"].upper().strip() for b in B],
                      "article": [urllib.parse.unquote(b["article"]["value"].rsplit("/", 1)[1]) for b in B]})
    syms = set(pd.concat((pd.read_parquet(f, columns=["symbol"]) for f in glob.glob(f"{RD}/weekly_*.parquet"))).symbol)
    M = M[M.symbol.isin(syms)].drop_duplicates()
    # one article per symbol: ambiguous tickers (several articles) are dropped
    M = M[~M.symbol.duplicated(keep=False)]
    os.makedirs(f"{RD}/wiki", exist_ok=True)
    M.to_csv(f"{RD}/wiki/ticker_articles.csv", index=False)
    print(f"wikidata: {len(B):,} ticker-article pairs; {len(M):,} unambiguous panel symbols", flush=True)
    if LIMIT:
        M = M.head(LIMIT)
    end = (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
    lock, out = threading.Lock(), []

    def one(r):
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
               f"{urllib.parse.quote(r.article, safe='')}/daily/20160101/{end}")
        raw = get(url)
        if not raw:
            return
        items = json.loads(raw).get("items", [])
        df = pd.DataFrame({"symbol": r.symbol, "date": pd.to_datetime([i["timestamp"][:8] for i in items]),
                           "views": [i["views"] for i in items]})
        with lock:
            out.append(df)

    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(one, M.itertuples()))
    P = pd.concat(out, ignore_index=True)
    P["views"] = P.views.astype("int32")
    P.to_parquet(f"{RD}/wiki/pageviews.parquet", index=False)
    print(f"::notice::wiki pageviews: {P.symbol.nunique():,} symbols, {len(P):,} rows, {P.date.min().date()}..{P.date.max().date()}")


if __name__ == "__main__":
    main()
