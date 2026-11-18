# Cantill.ai Procurement Spend Classifier

A local Python pipeline for cleaning purchase records, excluding non-spend, deduplicating supplier/item identities, converting currencies, and classifying purchase lines into a runtime-supplied taxonomy.

## Important confidentiality rule
The `data/` directory contains the confidential assessment kit and is intentionally ignored by Git. **Never force-add or commit it.** The public repository should contain source code, configs, documentation and permitted output files only.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
From the project root:

```bash
python src/main.py --company all
```

Outputs are written to `outputs/geetesh__A.csv` and `outputs/geetesh__B.csv`.

For a blind-test Company C package, place the fresh invoice file and taxonomy file under `data/`, create `config/company_C.json` using the supplied schema, then run:

```bash
python src/main.py --company C
```

The tool is designed to run unattended. No paid API key is required.

## Design notes
- Taxonomy values are loaded at runtime; category names are not embedded in source code.
- A uses ERP as the primary record where a corresponding invoice-tool line is matched; tool-side duplicates are excluded.
- A ERP labels are used as supplied for A's labelled training/build output. For a strict zero-label experiment, hide `labels_A.csv` and use a tree-only nearest-neighbour fallback (see limitation below).
- B uses the 3,000 seed labels as nearest-neighbour training examples and converts currencies to USD using the line rate or monthly FX table.
- Similarity-based classification uses TF-IDF cosine similarity and returns a confidence proxy, not a calibrated probability.
- Supplier/item keys are deterministic normalized-text or source-ID hashes. They are reproducible but do not replace a reviewed entity-resolution master.

## Limitations to review before submission
This is a strong runnable baseline, not a guaranteed full-score solution. A's cross-system matching uses PO and supplier/amount similarity heuristics; B's supplier IDs are treated as canonical vendor identities. For a high-stakes spend cube, validate ambiguous exclusions, fuzzy matches, FX gaps and taxonomy predictions against the task's hidden test criteria. Improve and test before the 10:00 AM freeze.

## Optional zero-label check for Company A

```bash
python src/main.py --company A --zero-label --output-dir outputs_zero_label
```

This run deliberately ignores A's supplied labels for category prediction and uses the taxonomy-only fallback. The default run uses the supplied A labels for the build dataset.
