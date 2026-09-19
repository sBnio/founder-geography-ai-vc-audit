# Founder Geography and AI Venture-Capital Decisions

Reproducibility materials for **“Does Founder Geography Affect AI Venture-Capital Decisions? A Matched-Pitch Multi-Model Audit.”**

## Study overview

This preregistered matched-pitch audit evaluates whether changing only founders’ disclosed operating location between Kyrgyzstan and the United States changes AI-generated venture-capital assessments when supplied venture fundamentals are held constant.

The final design contains:
- 8 fictional seed-stage startup scenarios
- 3 model families
- 3 stateless repetitions per applicable cell
- 144 normal labeled evaluations
- 72 anonymous-location evaluations
- 144 evidence-constrained mitigation evaluations
- 360 planned and 360 retained final evaluations

The preregistered primary inferential unit is the **startup scenario**, not the repeated API call. The primary test is an exact two-sided scenario-level sign-flip test.

## Repository status

This archive contains the quantitative and qualitative analysis materials used for the final study. It intentionally excludes API credentials, local virtual environments, duplicate raw backups, excluded pilot/synthetic materials, caches/compiled Python artifacts, signatures, and personally identifying information about the anonymous second coder.

The included data and code reproduce the paper's primary quantitative result and qualitative coding summaries. Exact standalone submission assets for the manuscript tables and figures are also included under `publication_assets/`. See `VALIDATION_REPORT.md` for the final archive checks.

## Preregistration and integrity

The scientific protocol and analysis plan were preregistered on OSF before final collection. Registration ID: `af8wg`. The registration is embargoed according to the study record.

Key integrity records are in `docs/`:
- `FREEZE_MANIFEST.json` — SHA-256 manifest of the frozen precollection scientific state
- `PRECOLLECTION_ENVIRONMENT.json` — software/environment snapshot
- `POSTCOLLECTION_MANIFEST.json` — final collection counts, timestamps, and file hashes
- `INITIAL_HASHES.json` — initial design hashes
- `OSF_REGISTRATION_CONFIRMED.txt` — registration confirmation record

The post-collection convenience verifier `analysis/verify_qualitative.py` is explicitly not part of the frozen/preregistered design; it only checks preserved final coding records.

## Repository structure

```text
analysis/                 Quantitative analysis + read-only qualitative verifier
config/                   Frozen scientific configuration and safe runtime default
providers/                Provider adapters used for collection
scripts/                  Collection, processing, QA, coding, and integrity utilities
tests/                    Automated tests
prompts/                  Frozen system, main, and mitigation prompts
scenarios/                Final fictional startup cases and country labels
data/processed/            Processed 360-response dataset and frozen run manifest
data/qualitative/          Coder files, coding key, adjudication record, final codes
outputs/quantitative/      Quantitative result tables and figures
docs/                      Freeze, environment, postcollection, OSF, and data documentation
publication_assets/          Exact standalone table/figure and supplement files for CER submission
```


## CER publication assets

The journal requests separate files for graphics and tables in addition to embedding them in the manuscript. This archive therefore includes:

- `publication_assets/main_figures/Figure_1.png` through `Figure_3.png` — the three standalone figures used in the main manuscript
- `publication_assets/main_tables/CER_Tables.docx` — the formatted set of Tables 1–4
- `publication_assets/main_tables/Table_1.csv` through `Table_4.csv` — machine-readable versions of the four main tables
- `publication_assets/supplement/CER_Supplementary_Materials.docx` — the submitted supplementary-material document
- `publication_assets/supplementary_figures/Figure_S1.png` and `Figure_S2.png` — standalone supplementary figures

The main manuscript itself is submitted directly to CER and is intentionally not duplicated in this repository, avoiding version drift between the submission file and the public reproducibility archive.

## Reproduce the quantitative analysis

Recommended: Python 3.12 with the locked package versions.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements-lock.txt
python -m pytest -q
python analysis/analyze.py \
  --input data/processed/responses.csv \
  --output-dir reproduced/quantitative \
  --equivalence-margin 5
```

The included processed dataset reproduces the reported primary scenario-level mean gap of approximately `-1.99` points and the preregistered exact two-sided sign-flip result `p = .0625`.

## Raw collection records

The public archive does not include the local append-only raw attempt log. The processed 360-response dataset, frozen run manifest, final QA records, prompts, configuration, and integrity manifests are included and are sufficient to reproduce the reported analyses. The canonical raw collection log is retained privately and can be provided to editors where appropriate and consistent with provider terms and the preregistration/embargo plan.

## Verify the qualitative results

```bash
python analysis/verify_qualitative.py
```

The qualitative archive preserves the researcher's pre-adjudication coding for all 360 responses, the independent 72-response second-coder sample, the post-study coding key, the two evidence-status adjudications, and the final 360-row qualitative dataset.

## Security and privacy

No API keys or authentication credentials are included. Do **not** commit a `.env` file. The original working directory contained local API credentials; those are intentionally excluded from this archive.

The anonymous second coder is not identified. Local signatures and unnecessary personal documents are excluded.

## AI-use disclosure

OpenAI ChatGPT was used outside the prespecified model evaluations for methodological critique and ideation, debugging/refinement of author-written code, and copy-editing of author-created manuscript text. It was not used for the actual qualitative coding or final analytical decisions. The author reviewed and verified the analyses, references, interpretations, and manuscript and remains responsible for the work.

## Author

Bektur Shakirov  
Affiliation during research: University of Wollongong in Dubai
