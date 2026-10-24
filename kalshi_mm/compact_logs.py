"""
kalshi_mm/compact_logs.py -- shrink snapshots.jsonl / cycles.jsonl to at most one line per minute (Oct 23, 2026).

With the live order-book stream the bot cycled every ~1s and logged a snapshot (with every position) each time, so the
logs grew by tens of MB a day and the dashboard ran out of its memory allowance reading them. The bot now logs once a
minute; this one-off pass thins the lines written before that. Keeps the last line of each minute (and every cycle
line that placed or cancelled orders). Safe to re-run.
"""
import json, os, sys

LOG = os.environ.get("LOG_DIR", "/home/kmm/data")


def compact(name, keep_active=False):
    p = f"{LOG}/{name}.jsonl"
    if not os.path.exists(p):
        return
    before = os.path.getsize(p)
    if before < 5e6:
        return
    keep, last_min = [], {}
    with open(p) as f:
        for i, line in enumerate(f):
            try:
                ts = json.loads(line).get("ts", "")[:16]
            except Exception:
                continue
            if keep_active and ('"placed": 0' not in line or '"cancelled": 0' not in line):
                keep.append(line); continue
            if ts in last_min:
                keep[last_min[ts]] = None
            last_min[ts] = len(keep); keep.append(line)
    with open(p + ".tmp", "w") as f:
        f.writelines(l for l in keep if l is not None)
    os.replace(p + ".tmp", p)
    print(f"compacted {name}: {before / 1e6:.1f} MB -> {os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    compact("snapshots"); compact("cycles", keep_active=True)
