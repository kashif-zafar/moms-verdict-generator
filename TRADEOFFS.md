# TRADEOFFS.md — Moms Verdict Generator

## Why this problem

The brief listed several example problems. I chose **Moms Verdict** over the others
for three reasons:

**It's the most defensible AI engineering problem in the list.**
Most of the other examples (PDP generation, email triage, gift finder) are
primarily prompt engineering problems — one well-crafted prompt + an LLM call
gets you 80% of the way there. Moms Verdict requires real retrieval architecture:
you have 200 messy reviews, not a single clean input. You can't just stuff them
all in a context window and prompt your way to a good result at scale.

**It maps directly to Mumzworld's core value.**
Mumzworld's trust signal to mothers is "other moms already vetted this."
A structured verdict that surfaces what real buyers said — grounded, not
hallucinated — is the natural AI form of that trust signal. It's not a
demo feature; it's a product feature.

**It's honestly evaluable.**
I can write objective evals with known ground truths: the stiff buckle complaint
appears in 4 of 20 reviews — it must surface. Travel is mentioned — it must appear
in use cases. This is better than problems where "good output" requires human
judgment on every run.

---

## What I rejected and why

| Problem | Why I didn't pick it |
|---|---|
| Voice memo → shopping list | Requires audio infrastructure (Whisper), adds setup complexity without adding RAG depth |
| Product image → PDP content | Multimodal, but primarily a prompt problem — single image in, text out, no retrieval |
| Return reason classifier | Clean, well-scoped, but too simple — one classification call, no retrieval, no real data challenge |
| Customer service email triage | Good problem, but the "messy data" angle is weaker — emails are already structured |
| Duplicate product detection | Interesting embeddings problem, but requires a real product catalog I'd have to fabricate extensively |

---

## Architecture decisions

### Embeddings: BGE-M3 via Ollama (not OpenAI ada-002)

**Why BGE-M3:**
- Natively multilingual — handles Arabic and English in the same embedding space
- No API key required — runs locally via Ollama, zero cost, zero latency variance
- I already had Ollama set up from prior RAG work, which saved setup time

**Tradeoff accepted:**
BGE-M3 via Ollama returns cosine similarity scores in the 0.3–0.5 range for
good matches, not the 0.8+ range you'd see from OpenAI embeddings. This caused
the first eval failure (retrieval_coverage = 0.0) when the threshold was set to 0.65.
Recalibrating to 0.30 fixed it. The lesson: always calibrate thresholds empirically
per model, don't port defaults between embedding models.

### Vector store: FAISS (not Pinecone, Weaviate, or Chroma)

**Why FAISS:**
- Local — no API keys, no network calls, no account setup
- For 100–200 reviews, retrieval takes < 1ms. Managed DBs add latency and complexity
  with zero benefit at this scale
- The index saves to disk (`faiss.index` + `chunks.pkl`), so re-embedding only
  happens when reviews change (`--rebuild` flag)

**When FAISS would be wrong:**
Production Mumzworld with millions of reviews across thousands of products would
need a managed store (Weaviate or Qdrant) with filtering by `product_id`. That's
a real engineering problem — but it's not this assignment.

### LLM: OpenRouter + Qwen 2.5 72B

**Why OpenRouter:**
- The brief explicitly mentioned it as the recommended gateway
- Free-tier access to Qwen 2.5 72B, which handles Arabic natively
- Single API, multiple model fallbacks — if Qwen hits quota, the code
  automatically retries with LLaMA 3 8B

**Why Qwen 2.5 72B over LLaMA:**
- Stronger multilingual performance, especially Arabic
- The output schema requires the model to count (complaint frequency) and
  reason structurally — larger models do this more reliably

**Tradeoff accepted:**
Free-tier rate limits mean the pipeline can't run at high throughput. For a
5-hour assignment generating ~10 test verdicts, this was fine. Production
would use a paid tier or a self-hosted model.

### Structured output: Pydantic v2 (not raw JSON parsing)

**Why Pydantic:**
- The brief explicitly required schema validation
- Pydantic v2 catches type errors, range violations (`confidence_score` must be 0–1),
  and cross-field inconsistencies (the `@field_validator` checks that the top-level
  score matches the breakdown weights within ±0.05)
- Failures are explicit exceptions, not silent wrong values

**Design decision — override LLM confidence with computed signals:**
The LLM generates a `confidence_score` in its JSON. We discard it and replace it
with our three-signal computed score before Pydantic validation. Reason: LLMs
are systematically overconfident. Our signals are based on actual data properties
(retrieval similarity, rating variance, review count) that the LLM cannot observe.

### Confidence scoring: 3-signal heuristic (not LLM self-assessment)

Weights: retrieval_coverage × 0.4 + sentiment_consistency × 0.4 + review_volume × 0.2

**Why these weights:**
- Retrieval quality and sentiment agreement are the two strongest signals that
  the output is grounded in real data — equal weight at 40% each
- Volume is a weaker signal (20 reviews can still produce a great verdict) — 20%

**Known weakness:**
The volume signal soft-caps at 100 reviews. At 20 reviews it contributes 0.04
to the final score, which feels right. But at exactly 100 reviews vs. 101 reviews,
the difference is negligible — the soft cap is a crude approximation.

---

## What I cut (and why)

| Cut | Reason | Would reconsider if... |
|---|---|---|
| Web UI / API layer | Assignment brief explicitly said don't build UI | Submitting to production, not assessment |
| Arabic output fields | LLM output is English-only; Arabic chunks are input only | Had a second pass to translate verdict fields |
| Post-processing complaint filter | Would enforce 3+ threshold in Python, not just prompt | Common_complaints accuracy was borderline failing |
| Caching layer | No repeated queries in test scenario | High-traffic production use case |
| Per-product index | One FAISS index for all products | Multi-product dataset |

---

## What I'd build next

1. **Arabic output fields** — The verdict JSON is English-only. A second LLM call
   could translate `pros`, `cons`, and `common_complaints` into native Arabic (not
   translated Arabic — prompted to write as a native Arabic speaker would).

2. **Post-processing complaint counter** — Count keyword frequency across retrieved
   chunks in Python and enforce the 3+ threshold before passing to the LLM.
   This removes the LLM's counting unreliability entirely.

3. **Per-product FAISS index** — Partition the index by `product_id` so the same
   pipeline handles a full catalog without cross-product retrieval contamination.

4. **Loom-style confidence explanation** — Add a `confidence_explanation` field
   to the schema: a one-sentence plain-English summary of why the score is what it is.
   E.g., "Based on 10 reviews with mixed sentiment and strong retrieval coverage."

5. **Adversarial eval expansion** — Currently 2 adversarial cases. A real eval
   suite would have 20+, including injected fake reviews, copy-pasted review spam,
   and reviews in languages outside EN/AR.

6. **Context window management** — At 20 reviews with top-k=10, the full prompt
   sits around ~1,000 tokens — well within Qwen's 128k limit. At scale (10,000+
   reviews), a map-reduce pattern would be needed: summarise each retrieved chunk
   before passing to the LLM, or batch reviews into groups and merge verdicts.
   Not built here because it would be over-engineering for the stated dataset size.
---

## Time log

| Phase | Time |
|---|---|
| Problem selection + reading brief | ~20 min |
| Architecture planning (stack decisions) | ~30 min |
| Schema + prompt design | ~45 min |
| Core pipeline code (loader, embedder, FAISS, LLM) | ~90 min |
| Debugging (similarity threshold calibration) | ~30 min |
| Eval suite + test cases | ~40 min |
| Documentation (README, EVALS.md, TRADEOFFS.md) | ~45 min |
| **Total** | **~5 hours** |
