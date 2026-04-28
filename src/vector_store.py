"""
vector_store.py
───────────────
FAISS-backed vector store for review embeddings.

Why FAISS (not Pinecone/Weaviate):
  - Local: no API keys, no setup, no cost
  - Fast enough for 100–200 reviews (sub-millisecond retrieval)
  - Dead simple to save/load as a file

Index type: IndexFlatIP (Inner Product = cosine similarity on L2-normalised vectors)
"""

import numpy as np
import faiss
import pickle
from pathlib import Path


INDEX_PATH  = "outputs/faiss.index"
CHUNKS_PATH = "outputs/chunks.pkl"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)   # avoid div-by-zero
    return matrix / norms


def build_index(embeddings: np.ndarray, chunks: list[dict]) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat inner-product index from embeddings.
    Saves index + chunk metadata to disk.

    Args:
        embeddings: np.ndarray shape (N, D)
        chunks:     list of chunk dicts matching embeddings row-for-row

    Returns:
        FAISS index (also saved to disk)
    """
    dim = embeddings.shape[1]
    normalised = _normalise(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(normalised)

    print(f"[vector_store] Built index: {index.ntotal} vectors, dim={dim}")

    # Persist
    Path("outputs").mkdir(exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[vector_store] Saved to {INDEX_PATH} and {CHUNKS_PATH}")
    return index


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a previously saved FAISS index + chunk metadata from disk."""
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run build_index() first."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"[vector_store] Loaded index with {index.ntotal} vectors")
    return index, chunks


def retrieve(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    query_vector: np.ndarray,
    top_k: int = 10,
    similarity_threshold: float = 0.0
) -> list[dict]:
    """
    Retrieve top-k most similar chunks for a query vector.

    Args:
        index:                FAISS index
        chunks:               chunk metadata list (aligned with index)
        query_vector:         shape (D,) float32
        top_k:                number of results to retrieve
        similarity_threshold: min cosine similarity to include (0.0 = no filter)

    Returns:
        List of chunk dicts, each augmented with a 'score' key (cosine similarity)
        Sorted descending by score.
    """
    # Normalise query
    q = query_vector.reshape(1, -1).astype(np.float32)
    q = _normalise(q)

    scores, indices = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < similarity_threshold:
            continue
        result = {**chunks[idx], "score": float(score)}
        results.append(result)

    print(f"[vector_store] Retrieved {len(results)} chunks (top_k={top_k})")
    return results
