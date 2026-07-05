import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
idx = pc.Index("arxiv-papers")

model = SentenceTransformer("allenai/specter2_base")


def embed_query(text):
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def show_results(res, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for m in res["matches"]:
        md = m["metadata"]
        print(f"  [{m['score']:.4f}] {md['title'][:90]}")
        print(f"           cat={md['category']}  year={md['year']}")


# semantic search
q1 = "teaching machines to recognize objects in pictures"
r1 = idx.query(vector=embed_query(q1), top_k=5, include_metadata=True)
show_results(r1, f"Semantic: '{q1}'")

# A) reinforcement learning, last 5 years, cs.LG
q2a = "reinforcement learning"
r2a = idx.query(
    vector=embed_query(q2a),
    top_k=5,
    include_metadata=True,
    filter={
        "year": {"$gte": 2021},
        "category": "cs.LG",
    },
)
show_results(r2a, "Filtered: RL + cs.LG + year>=2021")

# B) any category, before 2015
q2b = "reinforcement learning"
r2b = idx.query(
    vector=embed_query(q2b),
    top_k=5,
    include_metadata=True,
    filter={"year": {"$lt": 2015}},
)
show_results(r2b, "Filtered: RL + year<2015")


# compare cos / dot / L2 locally
print(f"\n{'='*60}")
print("  Local metric comparison")
print(f"{'='*60}")

df = pd.read_parquet(ROOT / "data" / "arxiv_subset.parquet")
emb_all = np.load(ROOT / "embeddings" / "embeddings.npy")

qv = model.encode(q1, normalize_embeddings=True)

cos_scores = emb_all @ qv
top5_cos = np.argsort(cos_scores)[::-1][:5]

dot_scores = emb_all @ qv  # same op for normalized vecs
top5_dot = np.argsort(dot_scores)[::-1][:5]

# L2 distance (lower = closer)
diff = emb_all - qv
l2_dist = np.linalg.norm(diff, axis=1)
top5_l2 = np.argsort(l2_dist)[:5]

for name, idxs, scores in [
    ("Cosine", top5_cos, cos_scores),
    ("Dot product", top5_dot, dot_scores),
    ("L2 distance", top5_l2, l2_dist),
]:
    print(f"\n  --- {name} ---")
    for rank, i in enumerate(idxs, 1):
        title = df.iloc[i]["title"][:80]
        sc = scores[i]
        print(f"  {rank}. [{sc:.4f}] {title}")

same = list(top5_cos) == list(top5_dot)
print(f"\nCosine top-5 == Dot product top-5: {same}")
print(f"L2 top-5 == Cosine top-5: {list(top5_l2) == list(top5_cos)}")
