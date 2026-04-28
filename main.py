"""
main.py
───────
Moms Verdict Generator — main pipeline.

Execution order:
  1. Load + clean + chunk reviews       (loader.py)
  2. Embed chunks with BGE-M3           (embedder.py)
  3. Store in FAISS                     (vector_store.py)
  4. Embed query + retrieve top-k       (vector_store.py)
  5. Compute confidence from signals    (confidence.py)
  6. Call LLM via OpenRouter            (llm.py)
  7. Merge confidence into LLM output
  8. Validate with Pydantic             (schema.py)
  9. Save verdict JSON to outputs/

Usage:
  python main.py --product "Chicco KeyFit 30" --reviews data/reviews.json
  python main.py --product "Chicco KeyFit 30" --reviews data/reviews.json --rebuild
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loader       import prepare_chunks
from embedder     import embed_chunks, embed_query
from vector_store import build_index, load_index, retrieve
from confidence   import compute_confidence
from llm          import call_llm
from schema       import ProductVerdict


TOP_K              = 10     # number of review chunks to retrieve
SIMILARITY_THRESH  = 0.0    # set > 0 to filter low-quality retrievals
OUTPUT_DIR         = Path("outputs")


def run_pipeline(
    product_name: str,
    reviews_path: str,
    api_key: str = None,
    rebuild_index: bool = False,
) -> ProductVerdict:
    """
    Full end-to-end pipeline. Returns a validated ProductVerdict.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    index_exists = (OUTPUT_DIR / "faiss.index").exists()

    # ── Step 1–4: Build or load FAISS index ─────────────────────────────────
    if rebuild_index or not index_exists:
        print("\n── Step 1: Loading and cleaning reviews ───────────────────")
        chunks = prepare_chunks(reviews_path)

        print("\n── Step 2: Generating embeddings (BGE-M3 via Ollama) ──────")
        embeddings, chunks = embed_chunks(chunks)

        print("\n── Step 3: Building FAISS index ────────────────────────────")
        index = build_index(embeddings, chunks)

    else:
        print("\n[main] Existing index found. Loading from disk (use --rebuild to re-embed)")
        index, chunks = load_index()

    total_reviews = len(set(c["id"] for c in chunks))

    # ── Step 5: Embed query and retrieve ────────────────────────────────────
    print("\n── Step 4: Retrieving relevant review chunks ───────────────")
    # Generic query — extracts the full spectrum of opinions
    query = f"What do people think about {product_name}? pros cons complaints"
    query_vector = embed_query(query)

    if query_vector is None:
        raise RuntimeError("Failed to embed query. Check Ollama is running.")

    retrieved = retrieve(
        index, chunks, query_vector,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESH
    )

    if not retrieved:
        raise RuntimeError("No chunks retrieved. Check your data and index.")

    # ── Step 6: Compute confidence ──────────────────────────────────────────
    print("\n── Step 5: Computing confidence score ──────────────────────")
    confidence_data = compute_confidence(retrieved, total_reviews)

    # ── Step 7: Call LLM ────────────────────────────────────────────────────
    print("\n── Step 6: Calling LLM via OpenRouter ──────────────────────")
    llm_output = call_llm(
        product_name=product_name,
        retrieved_chunks=retrieved,
        total_reviews=total_reviews,
        api_key=api_key,
    )

    # ── Step 8: Merge confidence (override LLM's self-reported confidence) ──
    # We trust our computed signals over the LLM's self-assessment
    llm_output["confidence_score"]    = confidence_data["confidence_score"]
    llm_output["confidence_breakdown"] = confidence_data["confidence_breakdown"]
    llm_output["review_count_used"]   = len(retrieved)

    # Detect languages from retrieved chunks
    langs = sorted(set(c.get("lang", "en") for c in retrieved))
    llm_output["languages_detected"] = langs

    # ── Step 9: Validate with Pydantic ──────────────────────────────────────
    print("\n── Step 7: Validating output with Pydantic ─────────────────")
    try:
        verdict = ProductVerdict(**llm_output)
    except Exception as e:
        print(f"\n[main] Validation failed: {e}")
        print("[main] Raw LLM output:")
        print(json.dumps(llm_output, indent=2, ensure_ascii=False))
        raise

    # ── Step 10: Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = product_name.replace(" ", "_").lower()
    out_path = OUTPUT_DIR / f"verdict_{safe_name}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdict.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"\n── Done ────────────────────────────────────────────────────")
    print(f"[main] Verdict saved to {out_path}")
    print(f"[main] Confidence score: {verdict.confidence_score}")
    print(f"[main] Pros: {len(verdict.pros)} | Cons: {len(verdict.cons)}")
    print(f"[main] Common complaints: {verdict.common_complaints}")

    return verdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Moms Verdict Generator — Reviews → Structured Insight"
    )
    parser.add_argument(
        "--product", required=True,
        help='Product name, e.g. "Chicco KeyFit 30 Infant Car Seat"'
    )
    parser.add_argument(
        "--reviews", default="data/reviews.json",
        help="Path to JSON reviews file (default: data/reviews.json)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild FAISS index even if one exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    verdict = run_pipeline(
        product_name=args.product,
        reviews_path=args.reviews,
        api_key=args.api_key,
        rebuild_index=args.rebuild,
    )

    print("\n── Final Verdict ───────────────────────────────────────────")
    print(json.dumps(verdict.model_dump(), indent=2, ensure_ascii=False))
