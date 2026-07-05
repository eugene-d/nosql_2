import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
idx = pc.Index("arxiv-papers")
model = SentenceTransformer("allenai/specter2_base")

df = pd.read_parquet(ROOT / "data" / "arxiv_subset.parquet")

# --- BM25 index ---
print("Building BM25 index ...")
corpus = []
for _, row in df.iterrows():
    doc = f"{row['title']} {row['abstract']}"
    corpus.append(doc.lower().split())

bm25 = BM25Okapi(corpus)
print(f"BM25 index built over {len(corpus)} docs")


def bm25_search(query, k=20):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:k]
    res = []
    for i in top_idx:
        if scores[i] > 0:
            res.append((f"paper_{i}", scores[i], df.iloc[i]))
    return res


def vector_search(query, k=20):
    qv = model.encode(query, normalize_embeddings=True).tolist()
    res = idx.query(vector=qv, top_k=k, include_metadata=True)
    out = []
    for m in res["matches"]:
        paper_i = int(m["id"].split("_")[1])
        out.append((m["id"], m["score"], df.iloc[paper_i]))
    return out


def rrf(ranked_lists, k=60):
    """merge ranked lists with 1/(k+rank) weighting"""
    scores = defaultdict(float)
    meta = {}
    for rlist in ranked_lists:
        for rank, (pid, _, row) in enumerate(rlist, 1):
            scores[pid] += 1.0 / (k + rank)
            meta[pid] = row
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(pid, scores[pid], meta[pid]) for pid in sorted_ids]


def hybrid_search(query, top_k=5, rrf_k=60):
    bm = bm25_search(query)
    vec = vector_search(query)
    fused = rrf([bm, vec], k=rrf_k)
    return bm[:top_k], vec[:top_k], fused[:top_k]


def print_results(results, method):
    print(f"\n  --- {method} ---")
    for rank, (pid, score, row) in enumerate(results, 1):
        t = row["title"][:80]
        print(f"  {rank}. [{score:.4f}] {t}")
        print(f"     cat={row['category']}  year={row['year']}")


# test queries from the task
queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"  Query: '{q}'")
    print(f"{'='*60}")

    bm_res, vec_res, hyb_res = hybrid_search(q)
    print_results(bm_res, "BM25")
    print_results(vec_res, "Vector (Pinecone)")
    print_results(hyb_res, "Hybrid (RRF)")

    # overlap analysis
    bm_ids = {r[0] for r in bm_res}
    vec_ids = {r[0] for r in vec_res}
    hyb_ids = {r[0] for r in hyb_res}
    only_hyb = hyb_ids - bm_ids - vec_ids
    overlap = bm_ids & vec_ids
    print(f"\n  BM25-only: {bm_ids - vec_ids - hyb_ids}")
    print(f"  Vector-only: {vec_ids - bm_ids - hyb_ids}")
    print(f"  Both BM25 & Vector: {overlap}")
    print(f"  In hybrid but not in individual top-5: {only_hyb}")
