# Decisions and assumptions

1. All source files are confidential. They remain under `data/`, which is excluded by `.gitignore`.
2. Company A ERP `value_zar` is already net of tax. Company A invoice-tool `gross_amount` is divided by `1 + VAT`; blank VAT defaults to 15%, as instructed.
3. For Company A, ERP is the primary record when a tool invoice is matched. The invoice-tool record is excluded as `DUPLICATE`.
4. Company A matching starts with normalized PO digits, then compares supplier names and net amount. Missing or reformatted PO references can remain unmatched; this is a known risk.
5. Company A invoice statuses: cancelled -> `CANCELLED`; anything other than approved/paid -> `NOT_APPROVED`.
6. Company B amounts are converted to USD. The line `currency_rate` is interpreted as currency units per USD, so local-currency amount is divided by the rate. Missing rates fall back to the monthly FX table.
7. Generic exclusion keywords cover taxes, intercompany, royalties, financing, revenue and partner billing. These are heuristic rules and should be reviewed against source-specific account codes.
8. Unparseable dates are not silently used for FX month selection; invoice date is the fallback when posting date is unavailable. If no month/rate exists, converted spend is set to zero and should be flagged/reviewed in a production implementation.
9. Supplier/item keys are deterministic hashes of normalized supplier or item identity. Company B vendor IDs are treated as canonical supplier IDs.
10. Empty or generic descriptions reduce classification confidence. The system still assigns the closest available category because the brief requests classification for every included line.
11. TF-IDF confidence is a similarity proxy, not a calibrated probability. Human review is recommended for low-confidence cases.
12. Company C config is expected to map its columns and taxonomy at runtime; no category names are hard-coded.
