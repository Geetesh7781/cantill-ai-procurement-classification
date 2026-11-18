
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Load zero-label predictions
pred = pd.read_csv(
    ROOT / "outputs_zero_label" / "geetesh__A.csv"
)

# Load original labels for evaluation only
labels = pd.read_csv(
    DATA / "company_A" / "labels_A.csv"
)

# Keep ERP records only
pred = pred[
    pred["record_id"].astype(str).str.startswith("ERP:")
].copy()

# Merge predictions with original labels
merged = pred.merge(
    labels,
    on="record_id",
    how="inner"
)

# Evaluate included records only
merged = merged[merged["include"] == 1].copy()

# Check category accuracy
for level in ["l1", "l2", "l3"]:

    actual_col = f"category_{level}"
    pred_col = level

    actual = (
        merged[actual_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    predicted = (
        merged[pred_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    accuracy = (actual == predicted).mean() * 100

    print(f"{level.upper()} Accuracy: {accuracy:.2f}%")

# Calculate exact three-level accuracy
actual_categories = merged[
    ["category_l1", "category_l2", "category_l3"]
].fillna("").astype(str)

predicted_categories = merged[
    ["l1", "l2", "l3"]
].fillna("").astype(str)

exact = (
    actual_categories.to_numpy()
    == predicted_categories.to_numpy()
).all(axis=1).mean() * 100

print("\n" + "=" * 40)
print("ZERO-LABEL EVALUATION")
print("=" * 40)

print("Compared records:", len(merged))
print(f"Exact category accuracy: {exact:.2f}%")

print("\nEvaluation completed successfully!")
