"""
research/fetch_auctions.py -- US Treasury auction dates 2015 -> today from the Treasury's public Fiscal Data API
(auctions_query), for the auction-cycle test (Lou, Yan & Zhang 2013). Nov 15, 2026.
Output: research_data/macro/treasury_auctions.csv (auction_date, issue_date, security_type, security_term, offering_amt).
"""
import json
import os
import urllib.request

import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"


def main():
    rows, page = [], 1
    while True:
        url = (f"{BASE}?fields=auction_date,issue_date,security_type,security_term,offering_amt,cusip"
               f"&filter=auction_date:gte:2015-01-01&sort=auction_date&page[size]=10000&page[number]={page}")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "momentum-research"}), timeout=120) as r:
            j = json.loads(r.read())
        rows += j["data"]
        if page >= j["meta"]["total-pages"]:
            break
        page += 1
    A = pd.DataFrame(rows)
    os.makedirs(f"{RD}/macro", exist_ok=True)
    A.to_csv(f"{RD}/macro/treasury_auctions.csv", index=False)
    print(A.security_type.value_counts().to_dict())
    print(f"::notice::treasury auctions: {len(A):,} rows {A.auction_date.min()}..{A.auction_date.max()}")


if __name__ == "__main__":
    main()
