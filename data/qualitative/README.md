# Qualitative coding data

This directory contains the item-level records needed to reproduce the paper's qualitative results and pre-adjudication inter-coder reliability.

## Files

- `researcher_coding_pre_adjudication.csv` — the researcher's completed prespecified coding for all 360 responses before adjudication.
- `second_coder_sample_pre_adjudication.csv` — the independent second coder's completed deterministic 72-response sample before adjudication. The anonymous coder is not identified.
- `coding_key_poststudy.csv` — post-study mapping from blinded `coding_id` values to run/scenario/model/condition/country metadata. This mapping was kept private during coding to preserve blinding.
- `adjudication_evidence_status.csv` — the two evidence-status disagreements and their final resolutions.
- `qualitative_codes_final_360.csv` — the final 360-row qualitative dataset after the two documented evidence-status adjudications, including derived qualitative flags.

The two evidence-status adjudications were:
- `C0141`: `unsupported_general_inference` → `unsupplied_specific_geographic_claim`
- `C0343`: `unsupplied_specific_geographic_claim` → `unsupported_general_inference`

Coder-confidence disagreements were retained as pre-adjudication reliability differences and were not adjudicated.

## Verify

From the repository root:

```bash
python analysis/verify_qualitative.py
```

The verifier is a post-collection, read-only convenience script. It is not part of the preregistered/frozen study design. It checks that the final coding file differs from the researcher's pre-adjudication coding only where documented, reproduces the five agreement statistics, and reproduces the final unsupported-geography counts/breakdowns.

Expected agreement results for the 72 double-coded responses:
- geography presence: 72/72, alpha = 1.000
- evidence status: 70/72, alpha ≈ 0.741
- direction: 72/72, alpha = 1.000
- decision relevance: 72/72, alpha = 1.000
- coder confidence: 70/72, alpha ≈ 0.655

Expected final qualitative pattern:
- 17 unsupported-geography responses out of 360 overall
- all 17 in the normal labeled Kyrgyzstan condition
- 16 negative and 1 mixed
- by model: Claude Opus 5 = 9, Gemini 3.6 Flash = 7, GPT-5.6 Sol = 1
- all 17 occurred in the preregistered Strong scenario category
