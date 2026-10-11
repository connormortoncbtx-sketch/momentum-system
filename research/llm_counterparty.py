"""
research/llm_counterparty.py -- second read of every contract-win 8-K (direction >= 1)
across llm/, llm_full/part*, llm_hist/part*: who is the counterparty and how firm
is the deal. Claude Haiku via the Message Batches API (~1-2k filings, < $3).
-> <RESEARCH_DATA>/llm_counterparty.parquet
"""
import glob, json, os, re, time
import pandas as pd

RD = os.environ.get("RESEARCH_DATA", "research_data")
MODEL = os.environ.get("LLM_MODEL") or "claude-haiku-4-5-20251001"
SYSTEM = """You read an SEC 8-K (company name hidden) that announces a contract, order or award.
Return ONLY a JSON object:
{"counterparty_type": one of ["us_federal_government","state_local_government","foreign_government",
   "large_public_company","small_or_private_company","related_party","unknown"],
 "counterparty_named": true/false (is the customer identified by name?),
 "firmness": one of ["firm_order_or_award","framework_or_master_agreement","loi_mou_or_nonbinding","option_or_extension"],
 "revenue_within_12m": true/false/null (does the text say revenue/deliveries start within 12 months?),
 "contract_value_usd": number or null,
 "duration_years": number or null,
 "repeat_customer": true/false/null (expansion of an existing relationship?)}"""


def main():
    J, T = [], []
    for d in [f"{RD}/llm"] + sorted(glob.glob(f"{RD}/llm_full/part*")) + sorted(glob.glob(f"{RD}/llm_hist/part*")):
        if os.path.exists(f"{d}/judgements.parquet"):
            J.append(pd.read_parquet(f"{d}/judgements.parquet"))
            T.append(pd.read_parquet(f"{d}/texts.parquet", columns=["accession", "text"]))
    j = pd.concat(J).drop_duplicates("accession"); t = pd.concat(T).drop_duplicates("accession")
    cw = j[(j.event_type == "contract_win") & (j.direction >= 1)].merge(t, on="accession")
    out_f = f"{RD}/llm_counterparty.parquet"
    done = set(pd.read_parquet(out_f).accession) if os.path.exists(out_f) else set()
    cw = cw[~cw.accession.isin(done)]
    print(f"contract wins to read: {len(cw):,}")
    if cw.empty:
        return
    import anthropic
    client = anthropic.Anthropic()
    reqs = [{"custom_id": a.replace("-", "_"),
             "params": {"model": MODEL, "max_tokens": 200, "system": SYSTEM,
                        "messages": [{"role": "user", "content": txt[:9000]}]}}
            for a, txt in zip(cw.accession, cw.text)]
    b = client.messages.batches.create(requests=reqs)
    t0 = time.time()
    while client.messages.batches.retrieve(b.id).processing_status != "ended":
        if time.time() - t0 > 4 * 3600:
            raise RuntimeError("batch not finished in 4h")
        time.sleep(60)
    rows = []
    for res in client.messages.batches.results(b.id):
        acc = res.custom_id.replace("_", "-")
        if res.result.type != "succeeded":
            continue
        m = re.search(r"\{.*\}", res.result.message.content[0].text, flags=re.S)
        try:
            x = json.loads(m.group(0)) if m else {}
        except Exception:
            x = {}
        rows.append({"accession": acc, **{k: x.get(k) for k in ("counterparty_type", "counterparty_named", "firmness",
                                                                    "revenue_within_12m", "duration_years", "repeat_customer")},
                     "cp_value_usd": pd.to_numeric(x.get("contract_value_usd"), errors="coerce")})
    new = pd.DataFrame(rows)
    for c in ("counterparty_named", "revenue_within_12m", "repeat_customer"):
        new[c] = new[c].astype("string")
    new["duration_years"] = pd.to_numeric(new.duration_years, errors="coerce")
    if done:
        new = pd.concat([pd.read_parquet(out_f), new])
    new.to_parquet(out_f, index=False)
    print(f"::notice::counterparty reads: {len(rows):,} new, {len(new):,} total; "
          + ", ".join(f"{k} {v}" for k, v in new.counterparty_type.value_counts().items()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"::error::{line}")
        raise
