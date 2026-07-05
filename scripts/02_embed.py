from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DATA_FILE = ROOT / "data" / "arxiv_subset.parquet"
EMB_DIR = ROOT / "embeddings"
EMB_FILE = EMB_DIR / "embeddings.npy"

MODEL_NAME = "allenai/specter2_base"
BATCH = 64

EMB_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(DATA_FILE)
print(f"Loaded {len(df)} papers from parquet")

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

print(f"Loading model {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding ...")
embeddings = model.encode(
    texts,
    batch_size=BATCH,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(f"Texts encoded: {len(embeddings)}")
print(f"Dimension: {embeddings.shape[1]}")
print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")

np.save(EMB_FILE, embeddings)
print(f"Saved to {EMB_FILE}")
