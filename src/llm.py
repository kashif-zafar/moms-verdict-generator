"""
llm.py
──────
Call OpenRouter API to generate the structured verdict JSON.

Model choice: Qwen/qwen2.5-72b-instruct (strong multilingual, free-tier available)
Fallback:     meta-llama/llama-3-8b-instruct

The system prompt enforces grounding:
  - LLM must only use the provided review chunks
  - Must return valid JSON matching ProductVerdict schema
  - No external knowledge, no hallucination
"""

import json
import os
import requests


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary model: Qwen 2.5 72B (excellent multilingual, handles Arabic well)
PRIMARY_MODEL  = "qwen/qwen-2.5-72b-instruct"
FALLBACK_MODEL = "meta-llama/llama-3-8b-instruct"


SYSTEM_PROMPT = """You are a product review analyst. Your job is to read \
a set of customer reviews and produce a structured, grounded verdict — \
like a trusted mom who has read every review so you don't have to.

RULES (non-negotiable):
1. Only use information present in the provided reviews. Do NOT add external knowledge.
2. Every claim in pros/cons/complaints must be traceable to at least one review chunk.
3. If fewer than 5 reviews were provided, set confidence_score below 0.4.
4. Do NOT invent use cases. Extract them from review text only.
5. For multilingual reviews (Arabic + English): treat both equally.
6. common_complaints must only list issues mentioned by 3 or more reviewers.
7. Return ONLY valid JSON. No markdown fences, no explanation, no preamble.
8. Limit pros to 3-5 items, cons to 3-5 items.

OUTPUT SCHEMA (return exactly this structure):
{
  "product_name": "string",
  "pros": ["string"],
  "cons": ["string"],
  "common_complaints": ["string"],
  "best_use_cases": ["string"],
  "confidence_score": 0.0,
  "confidence_breakdown": {
    "retrieval_coverage": 0.0,
    "sentiment_consistency": 0.0,
    "review_volume_score": 0.0
  },
  "review_count_used": 0,
  "languages_detected": ["en"]
}"""


USER_PROMPT_TEMPLATE = """Product: {product_name}
Reviews retrieved ({k} of {total} total in dataset):

{review_text}

---
Analyze the above reviews and return the JSON verdict.
Focus on patterns across multiple reviews, not single outliers.
If a complaint appears in 3+ reviews, include it in both cons AND common_complaints."""


def _build_review_text(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered review text for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lang = chunk.get("lang", "?")
        rating = chunk.get("rating", "?")
        text = chunk.get("text", "")
        lines.append(f"[{i}] (lang={lang}, rating={rating}/5)\n{text}")
    return "\n\n".join(lines)


def call_llm(
    product_name: str,
    retrieved_chunks: list[dict],
    total_reviews: int,
    api_key: str = None,
    model: str = None,
) -> dict:
    """
    Call OpenRouter to generate the structured verdict.

    Args:
        product_name:     name of the product being reviewed
        retrieved_chunks: top-k chunks from FAISS retrieval
        total_reviews:    total dataset size (for context in prompt)
        api_key:          OpenRouter API key (or set OPENROUTER_API_KEY env var)
        model:            override model (defaults to PRIMARY_MODEL)

    Returns:
        Parsed JSON dict matching ProductVerdict schema fields.
        Raises ValueError if the response cannot be parsed as JSON.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OpenRouter API key not found. "
            "Set OPENROUTER_API_KEY environment variable or pass api_key="
        )

    model = model or PRIMARY_MODEL
    review_text = _build_review_text(retrieved_chunks)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        product_name=product_name,
        k=len(retrieved_chunks),
        total=total_reviews,
        review_text=review_text,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/moms-verdict",   # optional, for OpenRouter stats
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.1,      # low temp → deterministic, grounded output
        "max_tokens": 800,
    }

    print(f"[llm] Calling {model} via OpenRouter...")

    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()

    except requests.exceptions.HTTPError as e:
        # If primary model fails (quota, unavailable), retry with fallback
        if response.status_code in (429, 503) and model != FALLBACK_MODEL:
            print(f"[llm] Primary model unavailable. Retrying with {FALLBACK_MODEL}...")
            return call_llm(
                product_name, retrieved_chunks, total_reviews,
                api_key=api_key, model=FALLBACK_MODEL
            )
        raise RuntimeError(f"OpenRouter API error: {e}")
    except ValueError as e:
        raise ValueError(
            f"[llm] Failed to parse OpenRouter JSON response.\n"
            f"Response text:\n{response.text}\n"
            f"Parse error: {e}"
        )

    if isinstance(response_json, dict) and "error" in response_json:
        error_details = response_json["error"]
        error_message = (
            error_details.get("message")
            if isinstance(error_details, dict) else str(error_details)
        )
        if model != FALLBACK_MODEL:
            print(f"[llm] OpenRouter returned an error: {error_message}. Retrying with {FALLBACK_MODEL}...")
            return call_llm(
                product_name, retrieved_chunks, total_reviews,
                api_key=api_key, model=FALLBACK_MODEL
            )
        raise RuntimeError(
            f"[llm] OpenRouter returned error: {error_message}\n"
            f"Response JSON:\n{json.dumps(response_json, indent=2, ensure_ascii=False)}"
        )

    def _extract_content(resp_json: dict) -> str | None:
        if "choices" in resp_json:
            choices = resp_json["choices"]
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        return message.get("content")
        if "output" in resp_json:
            output = resp_json["output"]
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, list) and content:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            return first_content.get("text")
        return None

    raw_content = _extract_content(response_json)
    if raw_content is None:
        raise RuntimeError(
            "[llm] OpenRouter response did not contain a valid completion payload."
            f"\nResponse JSON:\n{json.dumps(response_json, indent=2, ensure_ascii=False)}"
        )

    raw_content = raw_content.strip()

    # Strip accidental markdown fences if LLM adds them
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[llm] LLM returned non-JSON output.\n"
            f"Raw response:\n{raw_content}\n"
            f"Parse error: {e}"
        )

    print(f"[llm] Parsed JSON response successfully")
    return result
