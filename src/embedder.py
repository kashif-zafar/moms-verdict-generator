"""
embedder.py
───────────
Generate embeddings using BGE-M3 via local Ollama API.
Falls back gracefully with clear error messages if Ollama is unavailable.

BGE-M3 is multilingual — handles Arabic + English natively.
Output: 1024-dimensional float32 vectors.
"""

import numpy as np
import requests
import time
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "bge-m3"
EMBED_DIM       = 1024   # BGE-M3 output dimension


def _embed_single(text: str, retries: int = 3) -> Optional[np.ndarray]:
    """
    Call Ollama embedding endpoint for one text.
    Returns numpy float32 array of shape (EMBED_DIM,).
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            vector = response.json()["embedding"]
            return np.array(vector, dtype=np.float32)

        except requests.exceptions.ConnectionError:
            print(
                f"[embedder] Ollama not reachable. "
                f"Make sure it's running: `ollama serve`"
            )
            return None

        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(f"[embedder] Timeout on attempt {attempt+1}. Retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"[embedder] Unexpected error: {e}")
            return None

    return None


def embed_chunks(chunks: list[dict], batch_size: int = 16) -> tuple[np.ndarray, list[dict]]:
    """
    Embed all chunks using BGE-M3.

    Args:
        chunks:     List of chunk dicts (each has 'text', 'id', 'rating', etc.)
        batch_size: Process this many at a time (purely for progress logging)

    Returns:
        embeddings: np.ndarray of shape (N, EMBED_DIM)
        valid_chunks: chunks that were successfully embedded (same order)
    """
    embeddings = []
    valid_chunks = []
    total = len(chunks)

    print(f"[embedder] Embedding {total} chunks with {EMBED_MODEL}...")

    for i, chunk in enumerate(chunks):
        if i % batch_size == 0:
            print(f"[embedder] Progress: {i}/{total}")

        vector = _embed_single(chunk["text"])
        if vector is not None:
            embeddings.append(vector)
            valid_chunks.append(chunk)
        else:
            print(f"[embedder] Skipping chunk {chunk.get('chunk_id', i)} (embedding failed)")

    if not embeddings:
        raise RuntimeError(
            "[embedder] No embeddings generated. "
            "Check that Ollama is running and bge-m3 is pulled:\n"
            "  ollama pull bge-m3"
        )

    matrix = np.vstack(embeddings)
    print(f"[embedder] Done. Matrix shape: {matrix.shape}")
    return matrix, valid_chunks


def embed_query(query: str) -> Optional[np.ndarray]:
    """
    Embed a single query string for retrieval.
    Returns shape (EMBED_DIM,) float32 array.
    """
    print(f"[embedder] Embedding query: '{query[:60]}...'")
    return _embed_single(query)
