"""Task 2 — Stratified sampling, chunking, embedding, and vector indexing.

Pipeline:
  1. Load the cleaned dataset (data/filtered_complaints.csv, ~454k rows).
  2. Draw a stratified sample (~12k complaints) proportional to each of the
     four product categories.
  3. Split each narrative with LangChain's RecursiveCharacterTextSplitter
     (chunk_size=500, chunk_overlap=50 — matches the provided pre-built store).
  4. Embed every chunk with sentence-transformers/all-MiniLM-L6-v2 (384-d).
  5. Persist a ChromaDB collection with per-chunk metadata so retrieved chunks
     trace back to their source complaint.

Run:  python src/build_vector_store.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILTERED_CSV = ROOT / "data" / "filtered_complaints.csv"
VECTOR_DIR = ROOT / "vector_store"
COLLECTION = "cfpb_complaints"

# ---- Tunables -------------------------------------------------------------- #
SAMPLE_SIZE = 12_000          # target stratified sample (within 10k-15k band)
CHUNK_SIZE = 500              # characters
CHUNK_OVERLAP = 50            # characters
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 256
RANDOM_SEED = 42
# Drop near-empty narratives that would yield a single trivial chunk.
MIN_NARRATIVE_CHARS = 30


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation across product_category (largest-remainder)."""
    counts = df["product_category"].value_counts()
    total = len(df)
    # proportional target per category, with largest-remainder rounding
    raw = {cat: n * c / total for cat, c in counts.items()}
    alloc = {cat: int(math.floor(v)) for cat, v in raw.items()}
    remainder = n - sum(alloc.values())
    for cat in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:remainder]:
        alloc[cat] += 1

    parts = []
    for cat, k in alloc.items():
        sub = df[df["product_category"] == cat]
        k = min(k, len(sub))  # never request more than available
        parts.append(sub.sample(n=k, random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print("Stratified sample allocation:")
    for cat in counts.index:
        print(f"  {cat:<16} pop={counts[cat]:>7}  ({counts[cat]/total:5.1%})"
              f"  -> sampled {alloc[cat]:>5}")
    print(f"  TOTAL sampled: {len(out)}")
    return out


def main() -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"Loading {FILTERED_CSV} ...")
    df = pd.read_csv(FILTERED_CSV, dtype=str)
    df = df[df["narrative"].fillna("").str.len() >= MIN_NARRATIVE_CHARS]
    print(f"  {len(df):,} complaints with usable narratives")

    sample = stratified_sample(df, SAMPLE_SIZE, RANDOM_SEED)

    # ---- chunk ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    ids, docs, metas = [], [], []
    for row in sample.itertuples(index=False):
        chunks = splitter.split_text(row.narrative)
        for j, ch in enumerate(chunks):
            ids.append(f"{row.complaint_id}_{j}")
            docs.append(ch)
            metas.append({
                "complaint_id": str(row.complaint_id),
                "product_category": str(row.product_category),
                "product": str(row.product),
                "issue": str(row.issue),
                "company": str(row.company),
                "state": str(row.state),
                "date_received": str(row.date_received),
                "chunk_index": j,
                "total_chunks": len(chunks),
            })
    print(f"Produced {len(docs):,} chunks from {len(sample):,} complaints "
          f"({len(docs)/len(sample):.2f} chunks/complaint)")

    # ---- embed ----
    print(f"Embedding with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        docs, batch_size=EMBED_BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    print(f"  embeddings shape: {embeddings.shape}")

    # ---- index (ChromaDB, persisted) ----
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    B = 5_000
    for i in range(0, len(ids), B):
        collection.add(
            ids=ids[i:i + B],
            documents=docs[i:i + B],
            embeddings=embeddings[i:i + B].tolist(),
            metadatas=metas[i:i + B],
        )
        print(f"  indexed {min(i + B, len(ids)):,}/{len(ids):,}")

    print(f"\nDone. ChromaDB collection '{COLLECTION}' "
          f"({collection.count():,} chunks) -> {VECTOR_DIR}")


if __name__ == "__main__":
    main()
