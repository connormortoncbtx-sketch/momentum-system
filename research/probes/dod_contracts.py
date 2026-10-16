"""
research/probes/dod_contracts.py -- can a GitHub runner read defense.gov's daily contract
announcements (published ~5pm ET each business day, awards >= $7.5M)? Reports HTTP status,
how far back the listing goes, and a parsed sample. Read-only.
"""
import re, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research; momentum-bot@users.noreply.github.com)",
      "Accept": "text/html,application/xhtml+xml"}


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", None) or str(e)[:80], ""


for page in (1, 50, 200, 400):
    st, html = get(f"https://www.defense.gov/News/Contracts/?Page={page}")
    links = sorted(set(re.findall(r'href="(https://www\.defense\.gov/News/Contracts/Contract/Article/\d+/)"', html)))
    dates = re.findall(r"Contracts [Ff]or ([A-Z][a-z]+\.? \d{1,2}, \d{4})", html)
    print(f"::notice::listing page {page}: status {st}, {len(links)} article links, dates {dates[:2]} .. {dates[-1:] if dates else ''}")
    time.sleep(1)
    if page == 1 and links:
        st2, art = get(links[0])
        txt = re.sub(r"<[^>]+>", " ", art); txt = re.sub(r"\s+", " ", txt)
        m = re.search(r"((?:[A-Z][\w&.,'\- ]+?), ([A-Z][\w .'-]+), ([A-Z][a-z]+(?: [A-Z][a-z]+)?),? (?:is|was|has been) (?:being )?awarded[^.]*?\$([\d,]+)", txt)
        n = len(re.findall(r"awarded", txt))
        print(f"::notice::article status {st2}, chars {len(txt)}, 'awarded' x{n}; sample: {txt[txt.find('awarded') - 200: txt.find('awarded') + 150] if n else txt[:300]}")
