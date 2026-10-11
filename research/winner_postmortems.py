"""
research/winner_postmortems.py -- an AI post-mortem of each big winner and loser.

Input: research/samples/winner_sample.parquet (quarterly top/bottom 1%, 2024-2026,
with their 8-Ks from 2 weeks before to the end of the 13-week window).
For each stock: up to 6 filings (anonymised, ~2,000 characters each, dated relative
to the window start) + the realised move -> Claude Haiku (Batches) returns the most
likely driver, when it happened, whether its date was known in advance, whether
anything visible BEFORE the window hinted at it, and the observable precursor if any.
Output: research_data/llm/postmortems.parquet. Runs via research_run.yml.
"""
import json, os, re, sys, time, types
from pathlib import Path
import concurrent.futures as cf
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_filings as L

OUT = Path(os.environ.get("RESEARCH_DATA", "research_data")) / "llm"
SYSTEM = """You are a forensic equity analyst. You get the SEC 8-K filings (company name hidden)
of a small US stock around a 13-week window in which its price moved a lot, with each filing's
day offset relative to the window start (negative = before the window began).
Explain the most likely driver of the move. Return ONLY JSON:
{"driver": one of ["clinical_or_regulatory","earnings_or_guidance","contract_or_customer","acquired_or_merger",
  "financing_or_balance_sheet","dilution_or_offering","sector_or_theme_rally","squeeze_or_speculation",
  "restructuring_or_turnaround","reverse_merger_or_spac","delisting_or_distress","unclear"],
 "catalyst_day": integer day offset of the main catalyst, or null,
 "scheduled": true if the catalyst's date was publicly known in advance (earnings date, PDUFA date,
   announced trial readout, shareholder vote), else false,
 "hinted_before": "yes", "partly" or "no" -- did filings dated BEFORE day 0 hint at what came,
 "precursor": short text naming the observable precursor that could have flagged it in advance, or "none",
 "summary": max 25 words}"""


def texts_for(row):
    accs = [a for a in (row.accessions or "").split("|") if a][:6]
    dates = (row.accepted or "").split("|")[:6]
    parts = []
    for acc, d in zip(accs, dates):
        t = L.fetch_one(types.SimpleNamespace(accession=acc, cik=row.cik, symbol=row.symbol))
        if t and t["text"]:
            off = (pd.Timestamp(d) - pd.Timestamp(row.week)).days
            parts.append(f"--- filing at day {off:+d} ---\n{t['text'][:2000]}")
    return "\n\n".join(parts)


def main():
    import anthropic
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet("research/samples/winner_sample.parquet")
    s = s[s.n_filings > 0].reset_index(drop=True)
    with cf.ThreadPoolExecutor(4) as ex:
        s["text"] = list(ex.map(texts_for, s.itertuples(index=False)))
    s = s[s.text.str.len() > 200]
    client = anthropic.Anthropic()
    reqs = []
    for i, r in s.iterrows():
        mv = f"rose {r.next13 * 100:.0f}%" if r.next13 > 0 else f"fell {abs(r.next13) * 100:.0f}%"
        reqs.append({"custom_id": f"r{i}", "params": {"model": L.MODEL, "max_tokens": 350, "system": SYSTEM,
                     "messages": [{"role": "user", "content": f"Over the 13-week window the stock {mv}.\n\n{r.text}"}]}})
    b = client.messages.batches.create(requests=reqs)
    print(f"batch {b.id}: {len(reqs)} requests", flush=True)
    while client.messages.batches.retrieve(b.id).processing_status != "ended":
        time.sleep(60)
    res = {}
    tok = [0, 0]
    for x in client.messages.batches.results(b.id):
        if x.result.type != "succeeded":
            continue
        m = re.search(r"\{.*\}", x.result.message.content[0].text, flags=re.S)
        try:
            res[int(x.custom_id[1:])] = json.loads(m.group(0)) if m else {}
        except Exception:
            res[int(x.custom_id[1:])] = {}
        tok[0] += x.result.message.usage.input_tokens; tok[1] += x.result.message.usage.output_tokens
    j = pd.DataFrame.from_dict(res, orient="index")
    out = s.drop(columns=["text"]).join(j)
    out.to_parquet(OUT / "postmortems.parquet", index=False)
    print(f"::notice::postmortems: {len(out)} judged; tokens in {tok[0]:,} out {tok[1]:,}; "
          + " | ".join(f"{g}: " + ", ".join(f"{k} {v}" for k, v in d.driver.value_counts().head(4).items())
                       for g, d in out.groupby("group")))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
