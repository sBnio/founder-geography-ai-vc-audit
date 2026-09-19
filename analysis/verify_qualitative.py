#!/usr/bin/env python3
"""Post-collection read-only verifier for the final qualitative archive.

This convenience script was added for reproducibility packaging after data
collection. It is not part of the preregistered/frozen study design and does
not modify data or perform new inferential tests.
"""
from __future__ import annotations
import csv, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "data" / "qualitative"
DIMS = ["geography_present","evidence_status","direction","decision_relevance","coder_confidence"]

def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def nominal_alpha(a,b):
    n=len(a)
    do=sum(x!=y for x,y in zip(a,b))/n
    pooled=[*a,*b]
    counts=Counter(pooled)
    N=len(pooled)
    ae=sum(c*(c-1) for c in counts.values())/(N*(N-1))
    de=1-ae
    return math.nan if math.isclose(de,0.0,abs_tol=1e-15) else 1-do/de

researcher={r["coding_id"]:r for r in read_csv(Q/"researcher_coding_pre_adjudication.csv")}
second={r["coding_id"]:r for r in read_csv(Q/"second_coder_sample_pre_adjudication.csv")}
key={r["coding_id"]:r for r in read_csv(Q/"coding_key_poststudy.csv")}
final={r["coding_id"]:r for r in read_csv(Q/"qualitative_codes_final_360.csv")}
adjud={r["coding_id"]:r for r in read_csv(Q/"adjudication_evidence_status.csv")}

assert len(researcher)==360 and len(second)==72 and len(key)==360 and len(final)==360
assert set(second).issubset(researcher)
assert set(final)==set(researcher)==set(key)

for cid,rr in researcher.items():
    fr=final[cid]
    for dim in DIMS:
        expected=rr[dim]
        if dim=="evidence_status" and cid in adjud:
            expected=adjud[cid]["final_adjudicated_code"]
        assert fr[dim]==expected, (cid,dim,expected,fr[dim])

print("Pre-adjudication agreement (n=72)")
for dim in DIMS:
    a=[researcher[c][dim] for c in second]
    b=[second[c][dim] for c in second]
    agree=sum(x==y for x,y in zip(a,b))
    print(f"{dim}: {agree}/72 ({agree/72:.2%}), alpha={nominal_alpha(a,b):.6f}")

unsupported=[r for r in final.values() if r["unsupported_geography"].strip().lower()=="yes"]
print("\nFinal qualitative results")
print(f"unsupported geography: {len(unsupported)}/360")
print("direction:", dict(Counter(r["direction"] for r in unsupported)))
print("condition/country:", dict(Counter((key[r["coding_id"]]["condition"],key[r["coding_id"]]["country"]) for r in unsupported)))
print("model:", dict(Counter(key[r["coding_id"]]["model_id"] for r in unsupported)))
print("strength:", dict(Counter(key[r["coding_id"]]["strength"] for r in unsupported)))
