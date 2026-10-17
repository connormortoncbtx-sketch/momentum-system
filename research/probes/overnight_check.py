import datetime as dt, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import automation.shadow_overnight as S
names = ['MRNA', 'HPE', 'DELL', 'PAYS', 'TEAM', 'OKTA', 'ECO', 'LITE', 'HZO', 'ASX', 'IWM', 'SPY']
d = S.bars_recent(names, dt.date(2026, 10, 13))
for s, g in d.groupby("symbol"):
    g = g.set_index("date")
    print(f"::notice::{s}: " + " | ".join(f"{k}: o {r.open:.2f} c {r.close:.2f}" for k, r in g.iterrows()))
