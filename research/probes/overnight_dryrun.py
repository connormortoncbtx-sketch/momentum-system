"""Dry run of automation/shadow_overnight.py on the week of Oct 12-16, 2026 (no state written)."""
import datetime as dt, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import automation.shadow_overnight as S
tmp = Path(tempfile.mkdtemp())
S.STATE, S.LEDGER = tmp / "state.json", tmp / "ledger.jsonl"
S.START = dt.date(2026, 10, 12)
st = {"picks": {}, "last_open_date": None, "nav": {"raw_1bp": 100.0, "raw_2bp": 100.0, "hedged_1bp": 100.0, "day_only": 100.0, "iwm_night": 100.0}}
picks = S.picks_for(dt.date(2026, 10, 9), st)
print(f"::notice::formation Oct 9 top 10: {picks}")
S.main()
rows = [json.loads(l) for l in open(S.LEDGER)] if S.LEDGER.exists() else []
for r in rows:
    print(f"::notice::{r['entry_close']}->{r['open_date']} form {r['formation']} night {r['basket_night_bp']}bp median {r['median_name_bp']} IWM {r['iwm_night_bp']} nav {r['nav']}")
