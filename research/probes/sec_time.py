"""Probe: raw acceptanceDateTime values from the SEC submissions API vs the
filing index page's 'Accepted' time (which EDGAR shows in Eastern time)."""
import json, os, re, time, urllib.request
UA = os.environ.get("SEC_USER_AGENT") or "MomentumAlpha research momentum-bot@users.noreply.github.com"
def get(u):
    time.sleep(0.2)
    return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA})).read().decode()
for cik in (320193, 1045810):
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    r = sub["filings"]["recent"]
    shown = 0
    for i in range(len(r["form"])):
        if r["form"][i] in ("8-K", "10-K", "10-Q") and shown < 4:
            acc = r["accessionNumber"][i]
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm")
            m = re.search(r"Accepted</div>\s*<div class=\"info\">([^<]+)<", idx)
            print(f"::notice::cik {cik} {r['form'][i]} {acc} filingDate {r['filingDate'][i]} "
                  f"raw acceptanceDateTime {r['acceptanceDateTime'][i]} | index page Accepted {m.group(1) if m else '?'}")
            shown += 1
    files = sub["filings"].get("files", [])
    if files:
        old = json.loads(get(f"https://data.sec.gov/submissions/{files[-1]['name']}"))
        for i in range(min(2, len(old["form"]))):
            acc = old["accessionNumber"][i]
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm")
            m = re.search(r"Accepted</div>\s*<div class=\"info\">([^<]+)<", idx)
            print(f"::notice::OLD cik {cik} {old['form'][i]} {acc} filingDate {old['filingDate'][i]} "
                  f"raw {old['acceptanceDateTime'][i]} | index {m.group(1) if m else '?'}")
