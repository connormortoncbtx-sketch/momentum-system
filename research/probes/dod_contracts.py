"""
research/probes/dod_contracts.py -- can we read the full daily announcements from war.gov
(RSS links) and page back through older listings? Read-only.
"""
import re, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9"}


def get(url, t=60):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", None) or str(e)[:80], ""


st, x = get("https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400&Site=945&max=500", 120)
links = re.findall(r"<link>\s*(https?://[^<\s]+)\s*</link>", x)[1:]
print(f"::notice::RSS {st}: {len(links)} links, first {links[:1]}, last {links[-1:]}")
for u in (links[:1] + links[-1:]):
    s2, a = get(u)
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", a))
    i = txt.find("awarded")
    print(f"::notice::article {u[-60:]} status {s2}, {len(txt)} chars, 'awarded' x{txt.count('awarded')}: {txt[max(0, i - 200): i + 150]}")
    time.sleep(1)
for host in ("https://www.war.gov", "https://www.defense.gov"):
    for page in (1, 300):
        s3, h = get(f"{host}/News/Contracts/?Page={page}")
        arts = set(re.findall(r'/News/Contracts/Contract/Article/(\d+)/', h))
        ds = re.findall(r"Contracts [Ff]or ([A-Z][a-z]+\.? \d{1,2}, \d{4})", h)
        print(f"::notice::{host} listing page {page}: status {s3}, {len(arts)} articles, dates {ds[:1]} .. {ds[-1:]}")
        time.sleep(1)
