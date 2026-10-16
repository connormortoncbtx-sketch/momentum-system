"""
research/probes/usaspending.py -- can a runner query USAspending for DoD contract awards
>= $7.5M with dates and recipients? Read-only.
"""
import json, urllib.request

body = {"filters": {"award_type_codes": ["A", "B", "C", "D"],
                    "agencies": [{"type": "awarding", "tier": "toptier", "name": "Department of Defense"}],
                    "time_period": [{"start_date": "2019-03-01", "end_date": "2019-03-31", "date_type": "action_date"}],
                    "award_amounts": [{"lower_bound": 7500000}]},
        "fields": ["Award ID", "Recipient Name", "Award Amount", "Base Obligation Date", "Start Date",
                   "Recipient UEI", "recipient_id", "Awarding Sub Agency", "Description"],
        "page": 1, "limit": 100, "sort": "Award Amount", "order": "desc"}
req = urllib.request.Request("https://api.usaspending.gov/api/v2/search/spending_by_award/",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    res = d.get("results", [])
    print(f"::notice::USAspending OK: {len(res)} results, hasNext {d.get('page_metadata', {}).get('hasNext')}")
    for x in res[:4]:
        print(f"::notice::  {x.get('Base Obligation Date')} | {x.get('Recipient Name')} | ${x.get('Award Amount'):,.0f} | {str(x.get('Description'))[:60]}")
except Exception as e:
    print(f"::notice::USAspending failed: {getattr(e, 'code', '')} {str(e)[:200]}")
