"""
schema.py
─────────
Pydantic models for Moms Verdict Generator structured output.
Validates every field so bad LLM output is caught before it reaches the caller.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class ConfidenceBreakdown(BaseModel):
    """Three independent signals that compose the final confidence score."""
    retrieval_coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of top-k chunks above similarity threshold"
    )
    sentiment_consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agreement between reviews; low = polar opinions"
    )
    review_volume_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised review count; soft cap at 100 reviews"
    )


class ProductVerdict(BaseModel):
    """
    Final structured output for a single product.
    All claims must be grounded in retrieved review chunks.
    """
    product_name: str = Field(..., description="Name of the reviewed product")

    pros: List[str] = Field(
        ..., min_length=1, max_length=5,
        description="3–5 positive points extracted from reviews"
    )
    cons: List[str] = Field(
        ..., min_length=1, max_length=5,
        description="3–5 negative points extracted from reviews"
    )
    common_complaints: List[str] = Field(
        default_factory=list,
        description="Issues mentioned by 3+ reviewers (subset of cons)"
    )
    best_use_cases: List[str] = Field(
        ..., min_length=1,
        description="Scenarios product excels at, e.g. 'travel', 'newborn'"
    )

    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted composite: retrieval×0.4 + sentiment×0.4 + volume×0.2"
    )
    confidence_breakdown: ConfidenceBreakdown

    review_count_used: int = Field(
        ..., ge=1,
        description="Number of review chunks passed to the LLM"
    )
    languages_detected: List[str] = Field(
        default_factory=lambda: ["en"],
        description="ISO 639-1 language codes found in retrieved chunks"
    )

    @field_validator("confidence_score")
    @classmethod
    def score_matches_breakdown(cls, v, info):
        """Ensure top-level score is consistent with breakdown weights."""
        bd = info.data.get("confidence_breakdown")
        if bd:
            computed = (
                bd.retrieval_coverage    * 0.4 +
                bd.sentiment_consistency * 0.4 +
                bd.review_volume_score   * 0.2
            )
            if abs(v - computed) > 0.05:
                raise ValueError(
                    f"confidence_score {v:.2f} deviates from "
                    f"computed breakdown value {computed:.2f}"
                )
        return round(v, 2)
