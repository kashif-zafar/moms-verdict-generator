"""
loader.py
─────────
Load and clean product reviews from JSON.
Handles:
  - removing duplicates
  - stripping noise (HTML tags, excess whitespace)
  - chunking long reviews into overlapping segments
  - preserving metadata (rating, language, id)
"""

import json
import re
from pathlib import Path


def load_reviews(filepath: str) -> list[dict]:
    """Load reviews from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Reviews file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    print(f"[loader] Loaded {len(reviews)} raw reviews from {filepath}")
    return reviews


def clean_text(text: str) -> str:
    """
    Clean a single review text.
    - Strip HTML tags
    - Collapse whitespace
    - Remove repeated punctuation
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)
    # Remove repeated punctuation (!!!!! → !)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    return text.strip()


def deduplicate(reviews: list[dict]) -> list[dict]:
    """Remove exact-duplicate review texts."""
    seen = set()
    unique = []
    for r in reviews:
        key = r["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    removed = len(reviews) - len(unique)
    if removed:
        print(f"[loader] Removed {removed} duplicate review(s)")
    return unique


def chunk_review(review: dict, max_chars: int = 400, overlap: int = 50) -> list[dict]:
    """
    Split long reviews into overlapping chunks.
    Short reviews (< max_chars) are returned as-is.
    Each chunk inherits the parent review's metadata.
    """
    text = review["text"]
    if len(text) <= max_chars:
        return [{**review, "chunk_id": f"{review['id']}_0"}]

    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + max_chars
        chunk_text = text[start:end]
        chunks.append({
            **review,
            "text": chunk_text,
            "chunk_id": f"{review['id']}_{chunk_idx}"
        })
        start += max_chars - overlap
        chunk_idx += 1

    return chunks


def prepare_chunks(filepath: str, max_chars: int = 400) -> list[dict]:
    """
    Full pipeline: load → clean → deduplicate → chunk.
    Returns list of chunk dicts ready for embedding.
    """
    reviews = load_reviews(filepath)
    reviews = deduplicate(reviews)

    all_chunks = []
    for review in reviews:
        review["text"] = clean_text(review["text"])
        chunks = chunk_review(review, max_chars=max_chars)
        all_chunks.extend(chunks)

    langs = set(r.get("lang", "en") for r in all_chunks)
    print(f"[loader] Prepared {len(all_chunks)} chunks | Languages: {langs}")
    return all_chunks
