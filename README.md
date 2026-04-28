# Moms Verdict Generator

### Track A — AI Engineering Intern | Mumzworld Take-Home

> Takes 100–200 messy product reviews (English + Arabic) and converts them into
> a clean, structured verdict: pros, cons, common complaints, best use cases,
> and a confidence score — grounded entirely in the input reviews.

---

## One-paragraph summary

Moms Verdict Generator is a RAG pipeline that synthesizes noisy e-commerce reviews
into a validated, structured JSON verdict. It uses BGE-M3 embeddings (multilingual,
local via Ollama) to retrieve the most relevant review chunks into a FAISS index,
computes a three-signal confidence score from retrieval quality, sentiment consistency,
and review volume, then calls Qwen 2.5 72B via OpenRouter to generate structured
output that is validated against a Pydantic schema. The system handles English and
Arabic reviews natively, expresses uncertainty explicitly when data is thin, and
includes an objective eval suite that caught a real calibration bug during development.

---

## Setup — clone to first output in under 5 minutes

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- [OpenRouter](https://openrouter.ai) free API key

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull BGE-M3

```bash
ollama serve          # keep this terminal open
ollama pull bge-m3    # ~1.2GB, one-time download
```

### 3. Set your OpenRouter API key

```bash
# Linux / Mac
export OPENROUTER_API_KEY="sk-or-your-key-here"

# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-your-key-here"

# Windows CMD
set OPENROUTER_API_KEY=sk-or-your-key-here
```

### 4. Run the pipeline

```bash
python main.py --product "Chicco KeyFit 30 Infant Car Seat" --reviews data/reviews.json
```

### 5. Run evals

```bash
python eval.py --verdict outputs/verdict_chicco_*.json --reviews data/reviews.json
```

---

## Architecture

```
Reviews (JSON — EN + AR)
        │
        ▼
  [loader.py]         Clean + deduplicate + chunk
        │
        ▼
  [embedder.py]       BGE-M3 via Ollama → 1024-dim vectors
        │
        ▼
  [vector_store.py]   FAISS IndexFlatIP — build + save to disk
        │
        ▼  ← query embedded, top-k retrieved
  [confidence.py]     3-signal confidence score (not LLM-generated)
        │
        ▼
  [llm.py]            OpenRouter → Qwen 2.5 72B → raw JSON
        │
        ▼
  [schema.py]         Pydantic v2 — schema + cross-field validation
        │
        ▼
  verdict JSON   →   [eval.py] — 5 objective checks
```

---

## Stack

| Component    | Choice            | Why                                               |
| ------------ | ----------------- | ------------------------------------------------- |
| Embeddings   | BGE-M3 via Ollama | Multilingual (EN + AR), local, free, no API key   |
| Vector store | FAISS (local)     | Zero config, < 1ms retrieval for 100–200 reviews  |
| LLM          | OpenRouter        | Free tier, multi-model, explicit in brief         |
| Model        | Qwen 2.5 72B      | Best multilingual performance on free tier        |
| Fallback     | LLaMA 3 8B        | Auto-retry if Qwen hits quota                     |
| Validation   | Pydantic v2       | Schema + cross-field constraints, explicit errors |

---

## Confidence scoring

Confidence is computed from three independent signals **before** the LLM is called.
The LLM's self-reported confidence is discarded and replaced with this computed score.

| Signal                  | Weight | Description                                         |
| ----------------------- | ------ | --------------------------------------------------- |
| `retrieval_coverage`    | 40%    | Fraction of top-k chunks above similarity threshold |
| `sentiment_consistency` | 40%    | Low rating variance = reviewers agree               |
| `review_volume_score`   | 20%    | Normalised review count, soft cap at 100            |

A Pydantic `@field_validator` cross-checks the top-level `confidence_score`
against the breakdown weights and raises an explicit error if they diverge by > 0.05.

---

## Uncertainty handling

The system expresses uncertainty explicitly — it does not hide it:

- **Thin data** (< 5 reviews): `confidence_score` falls below 0.4, signalling low trust
- **Low retrieval coverage**: if chunks score below the similarity threshold,
  `retrieval_coverage` = 0.0, pulling confidence down significantly
- **Polar reviews** (mixed 1-star + 5-star): `sentiment_consistency` is low,
  and both sides are represented in pros and cons — the signal is not flattened
- **Out-of-scope input** (wrong product reviews): confidence collapses to ~0.2
  and no product-specific claims are invented
- **Pydantic validation failure**: if the LLM returns malformed JSON or violates
  schema constraints, the error is raised explicitly — no silent bad output

---

## Project structure

```
moms_verdict/
├── main.py               ← pipeline orchestrator (run this)
├── eval.py               ← objective evaluation suite
├── requirements.txt
├── README.md
├── EVALS.md              ← eval rubric, 10 test cases, results, known failures
├── TRADEOFFS.md          ← architecture decisions, what was cut, what's next
├── data/
│   └── reviews.json      ← 20 synthetic reviews (15 EN + 5 AR) — LLM-generated
├── src/
│   ├── schema.py         ← Pydantic models + cross-field validator
│   ├── loader.py         ← load, clean, deduplicate, chunk
│   ├── embedder.py       ← BGE-M3 via Ollama
│   ├── vector_store.py   ← FAISS build + retrieval
│   ├── confidence.py     ← 3-signal confidence scoring
│   └── llm.py            ← OpenRouter API + fallback logic
└── outputs/
    ├── faiss.index        ← persisted vector index
    ├── chunks.pkl         ← persisted chunk metadata
    └── verdict_*.json     ← generated verdicts
```

---

## Tooling

| Tool                            | Role                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------- |
| **ChatGPT (GPT-4o)**            | Architecture planning — back-and-forth to pressure-test stack choices before writing any code |
| **Claude (Sonnet)**             | Primary coding assistant — all module code, eval suite, and documentation                     |
| **Qwen 2.5 72B via OpenRouter** | Runtime LLM — generates structured verdict JSON                                               |
| **BGE-M3 via Ollama**           | Runtime embeddings — local, multilingual, no API key                                          |

**How they were used:**
ChatGPT was used in the planning phase only — no code was generated there. I described
the problem, proposed a stack, and used the conversation to validate choices (FAISS vs.
managed DB, BGE-M3 vs. ada-002, prompt structure). Claude generated all module code
based on those decisions. Each module was reviewed before use.

**Where I stepped in:**
The default similarity threshold (0.65) was wrong for BGE-M3's score range — good matches
were scoring 0.3–0.5, so `retrieval_coverage` was 0.0. The eval suite caught this on the
first run. I identified the root cause and fixed it manually (changed threshold to 0.30
in `confidence.py`). The AI did not catch this — the eval did.

The key system prompt and user prompt template are committed in full in `src/llm.py`.

---

## Evals summary

See `EVALS.md` for full rubric, 10 test cases, and known failure modes.

| Run   | Threshold | Score | Result                   |
| ----- | --------- | ----- | ------------------------ |
| Run 1 | 0.65      | 0.34  | 4/5 — calibration failed |
| Run 2 | 0.30      | 0.74  | **5/5 ✓**                |

---

## Tradeoffs summary

See `TRADEOFFS.md` for full architecture decisions and what was cut.

**Why FAISS over Pinecone:** overkill for 100–200 reviews — signals poor judgment at this scale.
**Why computed confidence over LLM confidence:** LLMs are overconfident; data signals are not.
**What was cut:** UI, API layer, Arabic output fields, per-product index partitioning.

---

## AI usage note

ChatGPT (GPT-4o) for architecture planning. Claude (Sonnet) as primary coding assistant
for all modules and documentation. Qwen 2.5 72B via OpenRouter as the runtime LLM.
BGE-M3 via Ollama for embeddings. The similarity threshold bug was caught by evals.

## Time log

| Phase                             | Time         |
| --------------------------------- | ------------ |
| Problem selection + brief reading | ~20 min      |
| Architecture planning (ChatGPT)   | ~30 min      |
| Schema + prompt design            | ~45 min      |
| Pipeline code + review            | ~90 min      |
| Debugging (threshold calibration) | ~30 min      |
| Eval suite + test cases           | ~40 min      |
| Documentation                     | ~45 min      |
| **Total**                         | **~5 hours** |
