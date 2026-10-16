"""
research/probes/dod_contracts.py -- how much history does the defense.gov contracts RSS
serve, and does each item carry the full announcement text? Read-only.
"""
import re, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
for mx in (100, 1000, 5000):
    url = f"https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400&Site=945&max={mx}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
            x = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"::notice::max={mx}: {e}"); continue
    items = re.findall(r"<item>(.*?)</item>", x, flags=re.S)
    dates = [re.sub(r"\s+", " ", t).strip() for t in re.findall(r"<title>(.*?)</title>", x, flags=re.S)[1:]]
    pub = re.findall(r"<pubDate>(.*?)</pubDate>", x)
    first = items[0] if items else ""
    desc = re.findall(r"<description>(.*?)</description>", first, flags=re.S)
    body = re.sub(r"&lt;[^&]*&gt;|<[^>]+>", " ", desc[0]) if desc else ""
    body = re.sub(r"\s+", " ", body)
    print(f"::notice::max={mx}: {len(items)} items, {len(x) / 1e6:.1f} MB, titles {dates[:1]} .. {dates[-1:]}, "
          f"pubDate {pub[:1]} .. {pub[-1:]}, first item text {len(body)} chars, 'awarded' x{body.count('awarded')}")
    if mx == 100:
        i = body.find("awarded")
        print(f"::notice::sample: {body[max(0, i - 250): i + 200]}")
