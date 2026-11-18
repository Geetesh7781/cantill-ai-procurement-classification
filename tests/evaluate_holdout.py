
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def evaluate(company, df, labels, id_col, feature_cols, label_cols):

    # Match source records with labelled examples
    merged = df.merge(
        labels,
        on=id_col,
        how="inner",
        suffixes=("", "_label")
    )

    if "include" in merged.columns:
        merged = merged[merged["include"] == 1].copy()

    merged = merged.dropna(
        subset=label_cols,
        how="all"
    )

    merged = merged.reset_index(drop=True)

    # Prepare text features
    texts = (
        merged[feature_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    actual = (
        merged[label_cols]
        .fillna("")
        .astype(str)
        .agg(" > ".join, axis=1)
    )

    # GROUP-BASED TRAIN / TEST SPLIT
    # Same material number will not appear in both sets

    groups = merged["material_number"].fillna("").astype(str).str.strip()

    fallback = (
        merged[feature_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.strip()
    )

    groups = groups.where(
        (groups != "") & (groups.str.lower() != "nan"),
        fallback
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(merged, groups=groups)
    )

    # TF-IDF model
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=120000,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(
        texts.iloc[train_idx]
    )

    X_test = vectorizer.transform(
        texts.iloc[test_idx]
    )

    # Similarity matching
    similarities = cosine_similarity(
        X_test,
        X_train
    )

    best = similarities.argmax(axis=1)

    predicted = (
        actual.iloc[train_idx]
        .iloc[best]
        .to_numpy()
    )

    expected = actual.iloc[test_idx].to_numpy()

    # Exact category accuracy
    exact_accuracy = (
        np.mean(predicted == expected) * 100
    )

    # Level 1 accuracy
    actual_l1 = (
        merged.iloc[test_idx][label_cols[0]]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    predicted_l1 = np.array([
        merged.iloc[train_idx].iloc[i][label_cols[0]]
        for i in best
    ]).astype(str)

    l1_accuracy = (
        np.mean(predicted_l1 == actual_l1) * 100
    )

    # Results
    print("\n" + "=" * 45)
    print(f"COMPANY {company} GROUP HOLDOUT RESULTS")
    print("=" * 45)

    print(f"Total labelled records  : {len(merged)}")
    print(f"Training records       : {len(train_idx)}")
    print(f"Testing records        : {len(test_idx)}")
    print(f"Training groups        : {groups.iloc[train_idx].nunique()}")
    print(f"Testing groups         : {groups.iloc[test_idx].nunique()}")

    print(f"Exact category accuracy: {exact_accuracy:.2f}%")
    print(f"Level 1 accuracy       : {l1_accuracy:.2f}%")

    return exact_accuracy, l1_accuracy


# ==========================================
# COMPANY A
# ==========================================

a = pd.read_csv(
    DATA / "company_A" / "erp_goods_receipts.csv",
    low_memory=False
)

a_labels = pd.read_csv(
    DATA / "company_A" / "labels_A.csv"
)

a["record_id"] = "ERP:" + a["line_id"].astype(str)

evaluate(
    "A",
    a,
    a_labels,
    "record_id",
    [
        "material_short_text",
        "service_short_text",
        "material_number"
    ],
    [
        "category_l1",
        "category_l2",
        "category_l3"
    ]
)


# ==========================================
# COMPANY B
# ==========================================

b = pd.read_csv(
    DATA / "company_B" / "invoice_lines.csv",
    low_memory=False
)

b_labels = pd.read_csv(
    DATA / "company_B" / "seed_labels.csv"
)

evaluate(
    "B",
    b,
    b_labels,
    "line_id",
    [
        "line_description",
        "material_number",
        "gl_account_description"
    ],
    [
        "category_l1",
        "category_l2",
        "category_l3"
    ]
)

print("\nGROUP-BASED EVALUATION COMPLETED!")
