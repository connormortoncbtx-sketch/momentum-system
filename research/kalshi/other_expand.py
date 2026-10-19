"""research/kalshi/other_expand.py -- weather_expand's per-fill deep-fill test on the non-weather candidate list
(expand_candidates.txt: recurring daily/weekly/monthly series outside sports/politics/elections, active since July 2026,
>= 50k contracts in the scan sample; the 8 current weather highs are appended as the control)."""
import os, sys
os.environ.setdefault("SERIES_FILE", "expand_candidates.txt")
os.environ.setdefault("N_PER", "40")
sys.path.insert(0, os.path.dirname(__file__))
import weather_expand
if __name__ == "__main__":
    weather_expand.main()
