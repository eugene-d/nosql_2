import os, time, re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer("allenai/specter2_base")

df = pd.read_parquet(ROOT / "data" / "arxiv_subset.parquet")

df["abs_len"] = df["abstract"].str.len()
long30 = df.nlargest(30, "abs_len").reset_index(drop=True)
print(f"Selected 30 papers, abstract lengths {long30['abs_len'].min()}-{long30['abs_len'].max()} chars")


def chunk_fixed(text, chunk_words=50, overlap=10):
    words = text.split()
    chunks = []
    pos = 0
    while pos < len(words):
        c = " ".join(words[pos:pos + chunk_words])
        chunks.append(c)
        pos += chunk_words - overlap
    return chunks


def chunk_semantic(text, max_words=80):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf, buf_len = [], [], 0
    for s in sents:
        wc = len(s.split())
        if buf_len + wc > max_words and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += wc
    if buf:
        chunks.append(" ".join(buf))
    return chunks


DIM = 768
BATCH_SZ = 100


def create_and_load(index_name, all_chunks_data):
    """embed + upload chunks into a fresh pinecone index"""
    existing = [ix.name for ix in pc.list_indexes()]
    if index_name in existing:
        pc.delete_index(index_name)
        time.sleep(5)

    pc.create_index(
        name=index_name,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)
    print(f"Index '{index_name}' ready")

    ix = pc.Index(index_name)

    texts = [d["text"] for d in all_chunks_data]
    print(f"  Encoding {len(texts)} chunks ...")
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    for start in tqdm(range(0, len(all_chunks_data), BATCH_SZ), desc="uploading"):
        end = min(start + BATCH_SZ, len(all_chunks_data))
        batch = []
        for j in range(start, end):
            d = all_chunks_data[j]
            batch.append({
                "id": d["id"],
                "values": vecs[j].tolist(),
                "metadata": d["meta"],
            })
        ix.upsert(vectors=batch)

    time.sleep(3)
    st = ix.describe_index_stats()
    print(f"  Vectors in '{index_name}': {st.total_vector_count}")
    return ix


fixed_data = []
sem_data = []

for i, row in long30.iterrows():
    txt = row["title"] + " [SEP] " + row["abstract"]

    fc = chunk_fixed(txt)
    for ci, c in enumerate(fc):
        fixed_data.append({
            "id": f"fc_{i}_{ci}",
            "text": c,
            "meta": {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"][:200],
                "chunk_text": c[:500],
                "chunk_num": ci,
                "year": int(row["year"]),
                "category": row["category"],
            },
        })

    sc = chunk_semantic(txt)
    for ci, c in enumerate(sc):
        sem_data.append({
            "id": f"sc_{i}_{ci}",
            "text": c,
            "meta": {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"][:200],
                "chunk_text": c[:500],
                "chunk_num": ci,
                "year": int(row["year"]),
                "category": row["category"],
            },
        })

print(f"\nFixed chunks: {len(fixed_data)},  Semantic chunks: {len(sem_data)}")

ix_fixed = create_and_load("arxiv-chunks-fixed", fixed_data)
ix_sem = create_and_load("arxiv-chunks-semantic", sem_data)


# --- search test ---
test_queries = [
    "attention mechanism in neural networks",
    "privacy preserving machine learning on distributed data",
    "how to detect adversarial examples",
]


def search_chunks(ix, q, label):
    qv = model.encode(q, normalize_embeddings=True).tolist()
    res = ix.query(vector=qv, top_k=5, include_metadata=True)
    print(f"\n  [{label}] query: '{q}'")
    for m in res["matches"]:
        md = m["metadata"]
        chunk_preview = md.get("chunk_text", "")[:100]
        print(f"    {m['score']:.4f} | {md['title'][:60]} | chunk#{md['chunk_num']}")
        print(f"           {chunk_preview}")


print(f"\n{'='*60}")
print("  Chunk search comparison")
print(f"{'='*60}")

for q in test_queries:
    search_chunks(ix_fixed, q, "FIXED")
    search_chunks(ix_sem, q, "SEMANTIC")
    print()
