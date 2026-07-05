import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "arxiv-papers"
DIM = 768
pc = Pinecone(api_key=API_KEY)

existing = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("Index ready")
else:
    print(f"Index '{INDEX_NAME}' already exists")

idx = pc.Index(INDEX_NAME)

df = pd.read_parquet(ROOT / "data" / "arxiv_subset.parquet")
emb = np.load(ROOT / "embeddings" / "embeddings.npy")

print(f"Uploading {len(df)} vectors ...")

for start in tqdm(range(0, len(df), 200)):
    end = min(start + 200, len(df))
    batch = []
    for i in range(start, end):
        row = df.iloc[i]
        meta = {
            "arxiv_id": str(row["arxiv_id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"])[:500],
            "authors": str(row["authors"])[:200],
            "year": int(row["year"]),
            "category": str(row["category"]),
        }
        batch.append(
            {
                "id": f"paper_{i}",
                "values": emb[i].tolist(),
                "metadata": meta,
            }
        )
    idx.upsert(vectors=batch)

time.sleep(3)
stats = idx.describe_index_stats()
print(f"\nTotal vectors in index: {stats.total_vector_count}")
