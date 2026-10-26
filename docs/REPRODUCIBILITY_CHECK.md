# Reproducibility check performed during repository cleanup

The supplied working repository was audited before this clean archive was assembled.

## Automated tests

Command:

```bash
python -m pytest -q
```

Result: **28 passed**.

## Quantitative re-run

The primary analysis was re-run from `data/processed/responses.csv` into a fresh output directory. Key generated tables matched the archived final outputs. The primary JSON differed only by a floating-point representation at approximately 4e-16 in one confidence-interval endpoint across the execution environment; substantive values and conclusions were unchanged.

Key reproduced results include:
- scenario-level mean Kyrgyzstan-minus-United States score gap: approximately `-1.9861`
- exact two-sided scenario-level sign-flip `p = .0625`
- prespecified ±5-point TOST supported equivalence (`max one-sided p ≈ .0042`)
- model-specific and secondary quantitative tables reproduced from the included processed dataset

## Security audit of the supplied working repository

The repository's own security audit passed. Additional inspection found live API values in the local `.env` file, but `.env` was ignored, was not tracked by Git, and no likely live API keys were found in reachable tracked Git history. The clean archive excludes `.env` entirely.

## Qualitative coding records

The final archive includes the researcher's completed coding for all 360 responses, the second coder's independent 72-response sample, the adjudication records, and the final qualitative coding dataset.

These files are stored in data/qualitative/ and can be checked using:

python analysis/verify_qualitative.py

The archived qualitative verification reproduces the reported coding counts and pre-adjudication agreement statistics.

## Scope of freeze verification

The original precollection freeze manifest is preserved unchanged.

The public repository omits eight original development files and includes post-study documentation and verification files. As a result, the original full-freeze verification script does not pass against this reduced public archive.

All 41 original frozen files retained in this release were checked against the original manifest, and their hashes matched. Complete verification of the original frozen file set requires the original working archive.