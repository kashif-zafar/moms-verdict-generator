"""
confidence.py
─────────────
Compute confidence score from retrieval signals — not from the LLM.

Three signals:
  1. retrieval_coverage    — how relevant are the retrieved chunks?
  2. sentiment_consistency — do reviewers agree with each other?
  3. review_volume_score   — how many total reviews back this verdict?

Weights: 40% / 40% / 20%
These match the @field_validator in schema.py — do not change independently.
"""


SIMILARITY_THRESHOLD = 0.30   # chunks below this score are "not relevant"
VOLUME_SOFT_CAP      = 100    # 100+ reviews → max volume score


def compute_confidence(
    retrieved_chunks: list[dict],
    total_reviews: int,
) -> dict:
    """
    Compute all three signals and return a dict ready to merge into the
    LLM's JSON output before Pydantic validation.

    Args:
        retrieved_chunks: chunks returned by vector_store.retrieve()
                          (each must have 'score' and 'rating' keys)
        total_reviews:    total number of reviews in the dataset (before retrieval)

    Returns:
        {
          "confidence_score": float,
          "confidence_breakdown": {
              "retrieval_coverage": float,
              "sentiment_consistency": float,
              "review_volume_score": float
          }
        }
    """
    if not retrieved_chunks:
        return {
            "confidence_score": 0.0,
            "confidence_breakdown": {
                "retrieval_coverage": 0.0,
                "sentiment_consistency": 0.0,
                "review_volume_score": 0.0,
            }
        }

    # ── Signal 1: Retrieval coverage ────────────────────────────────────────
    # What fraction of retrieved chunks crossed the similarity threshold?
    relevant = [c for c in retrieved_chunks if c.get("score", 0) >= SIMILARITY_THRESHOLD]
    retrieval_coverage = len(relevant) / len(retrieved_chunks)

    # ── Signal 2: Sentiment consistency ─────────────────────────────────────
    # Low variance in ratings → reviewers agree → high confidence.
    # High variance (1-star + 5-star mix) → conflicting → lower confidence.
    ratings = [c["rating"] for c in retrieved_chunks if "rating" in c]
    if ratings:
        avg = sum(ratings) / len(ratings)
        variance = sum((r - avg) ** 2 for r in ratings) / len(ratings)
        # On a 1–5 scale, theoretical max variance ≈ 4.0. Normalise and invert.
        sentiment_consistency = max(0.0, 1.0 - (variance / 4.0))
    else:
        sentiment_consistency = 0.5   # unknown → neutral

    # ── Signal 3: Review volume score ───────────────────────────────────────
    # More reviews → more reliable signal. Soft cap at VOLUME_SOFT_CAP.
    review_volume_score = min(total_reviews / VOLUME_SOFT_CAP, 1.0)

    # ── Weighted composite ──────────────────────────────────────────────────
    confidence_score = (
        retrieval_coverage    * 0.4 +
        sentiment_consistency * 0.4 +
        review_volume_score   * 0.2
    )

    breakdown = {
        "retrieval_coverage":    round(retrieval_coverage,    2),
        "sentiment_consistency": round(sentiment_consistency, 2),
        "review_volume_score":   round(review_volume_score,   2),
    }

    print(
        f"[confidence] score={confidence_score:.2f} | "
        f"retrieval={retrieval_coverage:.2f} | "
        f"sentiment={sentiment_consistency:.2f} | "
        f"volume={review_volume_score:.2f}"
    )

    return {
        "confidence_score":    round(confidence_score, 2),
        "confidence_breakdown": breakdown,
    }
