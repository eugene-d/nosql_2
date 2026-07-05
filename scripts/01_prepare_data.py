import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_PATH = PROJECT_DIR / "data" / "arxiv-metadata-oai-snapshot.json"
OUT_PATH = PROJECT_DIR / "data" / "arxiv_subset.parquet"
MAX_RECORDS = 8000

CS_PREFIXES = ("cs.",)


def parse_year(rec):
    try:
        return int(rec["update_date"][:4])
    except Exception:
        return None


rows = []
skipped = 0

print(f"Читаю {RAW_PATH.name} ...")

with open(RAW_PATH, "r") as f:
    for line in tqdm(f, desc="parsing", unit=" lines"):
        if len(rows) >= MAX_RECORDS:
            break
        rec = json.loads(line)
        cats = rec.get("categories", "")

        if not any(cats.startswith(p) or f" {p}" in cats for p in CS_PREFIXES):
            skipped += 1
            continue

        yr = parse_year(rec)
        if yr is None:
            skipped += 1
            continue

        main_cat = cats.split()[0]

        abstract = rec.get("abstract", "").strip()
        if len(abstract) < 50:
            skipped += 1
            continue

        rows.append(
            {
                "arxiv_id": rec["id"],
                "title": rec["title"].replace("\n", " ").strip(),
                "abstract": abstract.replace("\n", " "),
                "authors": rec.get("authors", ""),
                "year": yr,
                "category": main_cat,
            }
        )

df = pd.DataFrame(rows)
print(f"\nЗібрано {len(df)} записів, пропущено {skipped}")

print("\n--- Розподіл за категоріями (топ-15) ---")
cat_counts = df["category"].value_counts().head(15)
for cat, cnt in cat_counts.items():
    print(f"  {cat:20s}  {cnt}")

print("\n--- Розподіл за роками ---")
yr_counts = df["year"].value_counts().sort_index()
for y, cnt in yr_counts.items():
    print(f"  {y}  {cnt}")

df.to_parquet(OUT_PATH, index=False)
print(f"\nЗбережено: {OUT_PATH}  ({len(df)} рядків)")
