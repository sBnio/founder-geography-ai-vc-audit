# Validation report

This report summarizes checks performed on the clean submission/reproducibility archive.

## Qualitative coding integrity

The completed researcher workbook (360 responses), independent second-coder workbook (72 responses), final adjudication record, and final 360-row qualitative CSV were cross-checked before packaging.

- Final coding rows: 360
- Researcher pre-adjudication rows: 360
- Independent second-coder rows: 72
- Documented evidence-status adjudications: 2 (`C0141`, `C0343`)
- Mismatches between final codes and researcher codes after applying those two adjudications: **0**

Pre-adjudication agreement:
- geography presence: 72/72 (100.00%), nominal Krippendorff alpha = 1.000
- evidence status: 70/72 (97.22%), alpha = 0.740942
- direction: 72/72 (100.00%), alpha = 1.000
- decision relevance: 72/72 (100.00%), alpha = 1.000
- coder confidence: 70/72 (97.22%), alpha = 0.654589

Final qualitative result:
- unsupported geographic reasoning: 17/360 overall
- normal labeled Kyrgyzstan: 17/72
- matched labeled United States: 0/72
- anonymous: 0/72
- mitigation Kyrgyzstan: 0/72
- mitigation United States: 0/72
- direction among unsupported cases: 16 negative, 1 mixed
- model counts: Claude Opus 5 = 9; Gemini 3.6 Flash = 7; GPT-5.6 Sol = 1
- all 17 unsupported cases occurred in Strong scenarios

## Repository hygiene

The public archive excludes `.env` and API credentials, the prior Git history, `.venv`, caches, raw attempt logs and duplicate raw backups, excluded pilot/synthetic development data, private signature/approval documents, and personally identifying information about the anonymous second coder.

`SHA256SUMS.txt` records hashes for the final archived files.

## Final CER package audit

The final public repository package was additionally checked for Columbia Economic Review submission completeness.

- automated test suite: **28/28 passed**
- qualitative verification helper reproduces all reported pre-adjudication agreement statistics and final unsupported-geography counts
- separate main-paper figures present: **3/3**
- formatted main-paper table file present: **Tables 1–4**
- machine-readable main-paper tables present: **4/4**
- supplementary-material document present
- standalone supplementary figures present: **2/2**
- API credentials / `.env`: **not included**
- prior Git history: **not included**
- `.pytest_cache`, `__pycache__`, `.pyc`: **not included in distributed archive**
- manuscript is intentionally submitted separately to CER rather than duplicated here

The append-only raw technical-attempt log is retained privately. The public 360-row processed dataset includes raw response text and parsed outcomes for every retained successful evaluation and is sufficient to reproduce the manuscript analyses. The raw attempt log can be supplied privately to editors if specifically requested after a credentials/security review.
