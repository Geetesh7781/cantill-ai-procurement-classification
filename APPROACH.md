# Approach

## Pipeline

The pipeline loads source CSV files, normalizes text fields, applies procurement exclusion rules, calculates net spend, generates deterministic supplier and item keys, and classifies included line items against the runtime taxonomy.

The output follows the required submission schema.

## Matching and De-duplication

- **Company A:** ERP goods receipts are treated as the preferred counted records. Invoice-tool rows are matched using normalized PO references, supplier-name token similarity, and net/gross amount proximity. Matched invoice-tool records are excluded as duplicates. Repeated invoice-tool rows are also deduplicated.
- **Company B:** Vendor IDs are used to generate supplier keys. Item keys are generated using material numbers and descriptions. Distinct vendor IDs are not automatically assumed to represent the same legal supplier.

## Classification

- **Company A:** Supplied ERP labels are used in the labelled build. The zero-label mode performs taxonomy-based classification without labelled examples for category prediction. Other processing may still use the supplied label file. TF-IDF text similarity is used to rank taxonomy paths.
- **Company B:** Seed-labelled examples are used as nearest neighbours. Classification uses TF-IDF cosine similarity over line descriptions, material numbers, and GL descriptions.
- **Company C:** The configurable runner supports classification against a runtime taxonomy without labelled examples. Taxonomy columns are read from configuration, supporting up to four category levels. Local execution could not be validated because the Company C input and taxonomy CSV files were not present in the supplied local data folder.

## Tools and Dependencies

Python, pandas, NumPy, scikit-learn TF-IDF, RapidFuzz, and deterministic hashing.

No external or paid LLM API is used. The pipeline has no API-key dependency.

## Evaluation

A group-based holdout evaluation was performed:

| Company | Holdout records | Exact match accuracy | L1 accuracy |
|---|---:|---:|---:|
| A | 7,520 | 51.02% | 79.31% |
| B | 885 | 66.78% | 76.05% |

These results are from the local labelled holdout evaluation and should not be interpreted as hidden-test performance.

Confidence scores are similarity-based proxies and are not calibrated probabilities.

## Limitations

- Supplier matching is primarily based on normalized identifiers and available source fields.
- Similarity-based classification may be less reliable for short or ambiguous descriptions.
- Company C requires its actual input and taxonomy files for local validation.
- Hidden-test performance may differ from local holdout results.