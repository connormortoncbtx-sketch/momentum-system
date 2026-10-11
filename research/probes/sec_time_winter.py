"""Probe: SEC raw acceptanceDateTime vs index-page Accepted time for winter (EST) filings."""
import json, os, re, time, urllib.request
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
def get(u):
    time.sleep(0.2)
    return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA})).read().decode()
for cik in (320193, 1045810, 789019):
    r = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))["filings"]["recent"]
    n = 0
    for i in range(len(r["form"])):
        mo = r["filingDate"][i][5:7]
        if r["form"][i] in ("8-K", "10-Q", "10-K") and mo in ("01", "02", "12") and n < 3:
            acc = r["accessionNumber"][i]
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm")
            m = re.search(r"Accepted</div>\s*<div class=\"info\">([^<]+)<", idx)
            print(f"::notice::{cik} {r['form'][i]} filingDate {r['filingDate'][i]} raw {r['acceptanceDateTime'][i]} | index {m.group(1) if m else '?'}")
            n += 1
