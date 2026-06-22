# Task 2 — Chunking, Embedding & Vector Store (Report)

Script: [`src/build_vector_store.py`](../src/build_vector_store.py) ·
Output: persisted ChromaDB collection `cfpb_complaints` in [`vector_store/`](../vector_store/)

> Note: the brief mentions "five product categories," but this dataset variant
> defines **four** target products (Credit Card, Personal Loan, Savings Account,
> Money Transfer). Sampling is stratified across those four.

## Sampling strategy

From the 454k cleaned complaints, I drew a **stratified sample of 12,000**
complaints (mid-point of the 10k–15k band) using **proportional allocation**:
each category receives a share of the 12k equal to its share of the cleaned
population, with largest-remainder rounding so the parts sum exactly to 12,000.
This preserves the real-world class balance rather than forcing equal sizes —
the retriever should see the same product mix users will actually ask about.

| Category | Population | Share | Sampled |
|---|---:|---:|---:|
| Credit Card | 189,159 | 41.7% | 5,000 |
| Savings Account | 140,147 | 30.9% | 3,704 |
| Money Transfer | 98,615 | 21.7% | 2,606 |
| Personal Loan | 26,105 | 5.7% | 690 |
| **Total** | **454,026** | 100% | **12,000** |

Sampling uses a fixed seed (`RANDOM_SEED=42`) so the build is reproducible.
Personal Loan is the smallest stratum (690); if downstream retrieval quality for
that product is weak, the proportional scheme can be swapped for a floored
minimum-per-class allocation — the function is parameterized to make this easy.

## Chunking approach

I used LangChain's **`RecursiveCharacterTextSplitter`** with
**`chunk_size=500`, `chunk_overlap=50`** (characters), and a separator hierarchy
of paragraph → line → sentence → word → character. This matches the spec of the
provided pre-built store, so the sample-built index and the full store stay
comparable.

Rationale for the size/overlap:
- **500 chars (~80–90 words)** keeps each chunk to roughly a single
  issue/event. The median cleaned narrative is ~131 words, so most complaints
  split into 1–3 focused chunks — embedding a whole multi-paragraph complaint as
  one vector would blur distinct issues and hurt retrieval precision.
- **50-char overlap (10%)** preserves continuity across boundaries so a sentence
  split mid-thought still appears intact in one chunk.
- Result: **33,573 chunks** from 12,000 complaints (**2.80 chunks/complaint**).

The recursive splitter is preferred over a naive fixed-width cut because it
breaks on natural boundaries first, avoiding mid-word/mid-sentence fragments.

## Embedding model

**`sentence-transformers/all-MiniLM-L6-v2`** (384-dim, ~80 MB). Chosen because:
- It is the model the pre-built store uses, keeping the sample index consistent.
- It is small and CPU-friendly — essential here, since this is an Intel Mac with
  no GPU (full encode of 33,573 chunks took ~30 min).
- It is a strong, well-benchmarked general-purpose sentence encoder whose
  short-passage semantic-similarity quality is excellent for its size — a good
  speed/quality trade-off for retrieval over short complaint chunks.

Embeddings are **L2-normalized** and the ChromaDB collection uses
**cosine** space (`hnsw:space=cosine`), so similarity is normalized dot product.

## Vector store & metadata

ChromaDB `PersistentClient` writes to [`vector_store/`](../vector_store/)
(`chroma.sqlite3` + HNSW index files). Each chunk is stored with its embedding,
the chunk text, a stable id `"{complaint_id}_{chunk_index}"`, and metadata:
`complaint_id, product_category, product, issue, company, state,
date_received, chunk_index, total_chunks`. This lets every retrieved chunk be
traced back to its source complaint and filtered by product/issue at query time.

**Verification:** `collection.count() == 33,573`; a sample query
("unauthorized charges on my credit card") returns on-topic Credit Card chunks
with cosine distances ~0.25 and intact source metadata.
