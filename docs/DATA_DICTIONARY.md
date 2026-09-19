# Data dictionary

## Raw attempt log (retained privately; not included in this public archive)
The append-only collection log `data/raw/responses.jsonl` is preserved privately as a research record. It contains all technical attempts, including failed provider calls. It is not required to reproduce the manuscript analyses, which use the 360 retained successful evaluations in `data/processed/responses.csv`. The public processed dataset retains the raw response text and parsed outcomes for those 360 evaluations. The raw attempt log can be provided to editors on request where appropriate.

## `data/processed/responses.csv`
One row per retained successful planned evaluation (360 rows). Important fields include:
- `run_id`: deterministic run identifier
- `condition`: `labelled`, `anonymous`, or `mitigation`
- `scenario_id`: fictional startup identifier
- `strength`: preregistered scenario strength category
- `country`: Kyrgyzstan, United States, or Anonymous
- `provider`, `model_id`, `actual_model_returned`
- `repetition`: repeated stateless call number
- `investment_score_0_100`: preregistered primary outcome
- `recommended_investment_usd`, `recommended_pre_money_valuation_usd`
- `probability_of_closing_stated_round_within_12_months_percent`
- `financial_risk_score_0_100`, `confidence_0_100`
- rationale and structured qualitative text fields
- token/cost metadata where available

## `data/processed/run_manifest.jsonl`
Frozen planned 360-run manifest containing the run design and exact rendered prompts before final collection.

## `outputs/quantitative/`
Deterministically generated quantitative result tables and figures used to construct the manuscript and supplement.
