"""
research/probes/dod_contracts.py -- routes to DoD daily contract announcements (defense.gov
blocks plain requests with 403): (1) defense.gov RSS, (2) Wayback Machine CDX index of the
daily contract pages + one archived page parsed. Read-only.
"""
import json, re, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}


def get(url, t=60):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", None) or str(e)[:80], ""


st, x = get("https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400&Site=945&max=10")
print(f"::notice::RSS status {st}, items {x.count('<item>')}, first title {re.findall(r'<title>([^<]+)', x)[1:3]}")
st, x = get("https://www.defense.gov/News/Contracts/")
print(f"::notice::listing w/ browser UA status {st}")
cdx = ("http://web.archive.org/cdx/search/cdx?url=defense.gov/News/Contracts/Contract/Article/*"
       "&output=json&fl=timestamp,original,statuscode&filter=statuscode:200&collapse=urlkey&limit=20000")
st, x = get(cdx, 180)
rows = json.loads(x)[1:] if st == 200 and x.strip().startswith("[") else []
yrs = {}
for ts, u, _ in rows:
    yrs[ts[:4]] = yrs.get(ts[:4], 0) + 1
print(f"::notice::Wayback CDX status {st}, unique article URLs {len(rows)}, captures by year {dict(sorted(yrs.items()))}")
if rows:
    ts, u, _ = rows[len(rows) // 2]
    st2, a = get(f"http://web.archive.org/web/{ts}id_/{u}", 90)
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", a))
    i = txt.find("awarded")
    title = re.findall(r"Contracts [Ff]or ([A-Z][a-z]+\.? \d{1,2}, \d{4})", txt)
    print(f"::notice::archived page {ts} status {st2}, date {title[:1]}, 'awarded' x{txt.count('awarded')}: {txt[max(0, i - 220): i + 120]}")
