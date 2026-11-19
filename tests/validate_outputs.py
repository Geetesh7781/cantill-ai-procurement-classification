
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = [
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "l1",
    "l2",
    "l3",
    "l4",
    "confidence"
]

for name in [
    "geetesh__A.csv",
    "geetesh__B.csv",
    "geetesh__C.csv"
]:
    p = ROOT / "outputs_blind" / name

    if not p.exists():
        print(name, "FILE NOT FOUND")
        continue

    df = pd.read_csv(p)

    assert list(df.columns) == EXPECTED, (
        name, list(df.columns)
    )

    assert df["include"].isin([0, 1]).all()
    assert df["confidence"].between(0, 1).all()

    assert (
        df.loc[df["include"] == 0, "spend"] == 0
    ).all()

    print(name, len(df), "rows OK")

print("Validation completed.")
