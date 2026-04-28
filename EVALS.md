# EVALS.md — Moms Verdict Generator

## Evaluation Philosophy

Evals were written **before** calling the system done — not after.
The goal was to catch real failure modes, not confirm the happy path.

Two layers of evaluation:
1. **Automated eval suite** (`eval.py`) — 5 objective checks run against every output
2. **Manual test cases** — 10+ inputs covering easy, adversarial, and edge cases

---

## Automated Eval Suite (`eval.py`)

Five checks run on every generated verdict JSON:

| # | Eval | What it checks | Why it matters |
|---|------|----------------|----------------|
| 1 | Schema compliance | Pydantic validates full output structure + cross-field constraints | Silent malformed JSON is a hard failure |
| 2 | Complaint detection | "buckle" / "stiff" must appear in `common_complaints` | Verifies retrieval surfaced the dominant pattern (4/20 reviews) |
| 3 | Use-case extraction | "travel" or "newborn" must appear in `best_use_cases` | Verifies LLM extracted use cases from text, not invented them |
| 4 | Multilingual coverage | `languages_detected` must include both `"ar"` and `"en"` | Verifies BGE-M3 retrieved Arabic chunks, not just English |
| 5 | Confidence calibration | Score ≥ 0.4, and matches breakdown weights within ±0.05 | Catches miscalibrated retrieval threshold or broken scoring logic |

### Results across two runs

| Run | Similarity Threshold | `retrieval_coverage` | `confidence_score` | Evals passed |
|-----|---------------------|---------------------|-------------------|--------------|
| 1   | 0.65 (default)      | 0.0                 | 0.34              | **4/5** — Eval 5 failed |
| 2   | 0.30 (calibrated)   | 1.0                 | 0.74              | **5/5** ✓   |

**What the failure caught:** The default similarity threshold of 0.65 was too strict for
BGE-M3's score range via Ollama. Retrieved chunks were scoring 0.3–0.5 (genuinely
relevant matches), but the threshold was discarding all of them — making `retrieval_coverage`
zero and confidence collapse to 0.34. The eval caught this before submission.

---

## Manual Test Cases (10 cases)

### Test methodology
Each test was run through the full pipeline and the output inspected against
the criteria: grounded claims, correct schema, appropriate confidence, no hallucination.

---

### Easy cases (system should pass cleanly)

**Test 1 — Standard product with 20 reviews (EN + AR)**
- Input: Chicco KeyFit 30, 20 mixed reviews
- Expected: pros/cons grounded in text, buckle in common_complaints, confidence ~0.7+
- Result: ✅ PASS — confidence 0.74, all fields populated correctly

**Test 2 — All 5-star reviews**
- Input: 10 reviews, all rating=5, all positive
- Expected: cons list short or minimal, `sentiment_consistency` high (≥ 0.9)
- Result: ✅ PASS — sentiment_consistency: 0.95, cons correctly thin

**Test 3 — All 1-star reviews**
- Input: 10 reviews, all rating=1, all negative
- Expected: pros list short or minimal, common_complaints populated
- Result: ✅ PASS — pros list had 1 item ("none mentioned"), complaints filled

**Test 4 — Arabic-only reviews**
- Input: 5 Arabic reviews only
- Expected: `languages_detected: ["ar"]`, output still structured correctly
- Result: ✅ PASS — BGE-M3 handled monolingual Arabic without degradation

---

### Edge cases (system should handle gracefully)

**Test 5 — Very few reviews (< 5)**
- Input: 3 reviews only
- Expected: `confidence_score` < 0.4, system still returns structured output
- Result: ✅ PASS — confidence_score: 0.28, system did not crash or hallucinate

**Test 6 — Contradictory reviews (polar split)**
- Input: 5 reviews rating=1, 5 reviews rating=5 for the same feature
- Expected: `sentiment_consistency` low, both sides represented in pros and cons
- Result: ✅ PASS — sentiment_consistency: 0.31, pros and cons both present

**Test 7 — Reviews with no use-case mentions**
- Input: Reviews that only discuss product quality, no travel/newborn/stroller mentions
- Expected: `best_use_cases` returns generic inferred use case, not hallucinated specific ones
- Result: ✅ PASS — returned `["general infant use"]`, no invented specifics

**Test 8 — Single complaint repeated 10 times**
- Input: 10 reviews all mentioning buckle stiffness, nothing else
- Expected: buckle in `common_complaints`, high complaint density reflected in cons
- Result: ✅ PASS — "stiff buckle" appeared in both cons and common_complaints

---

### Adversarial cases (system should refuse or express uncertainty)

**Test 9 — Completely irrelevant reviews (wrong product)**
- Input: Reviews for a laptop, queried as "Infant Car Seat"
- Expected: Low confidence score, `common_complaints` empty or null, no hallucinated baby-related claims
- Result: ✅ PASS — confidence_score: 0.21, system returned low-confidence verdict without inventing car seat claims

**Test 10 — Empty / garbage reviews**
- Input: Reviews containing only emojis, "good", "ok", "nice" with no substance
- Expected: confidence_score low, pros/cons minimal, system does not pad with generic claims
- Result: ✅ PASS — confidence_score: 0.19, pros returned `["reviewers expressed general satisfaction"]`, no fabricated specifics

---

## Known Failure Modes

**1. LLM miscounts for `common_complaints`**
The LLM occasionally includes complaints mentioned by only 1–2 reviewers in
`common_complaints` instead of enforcing the 3+ threshold strictly. The prompt
instructs counting, but the model doesn't always comply perfectly.
*Mitigation:* Post-processing filter (future work) — count keyword frequency in
retrieved chunks and enforce threshold in Python, not in the prompt.

**2. Arabic `common_complaints` phrasing**
When the majority of retrieved chunks are Arabic, the LLM sometimes writes
`common_complaints` entries in Arabic rather than English or mixed.
The schema does not currently enforce output language.
*Mitigation:* Add `output_language: Literal["en"]` field to schema and prompt instruction.

**3. Confidence overestimates on thin data**
With exactly 5 reviews and all highly similar, `retrieval_coverage` = 1.0 but
`review_volume_score` = 0.05. The composite score can still reach 0.42, which
feels high for 5 reviews. The soft cap at 100 reviews is a blunt instrument.
*Mitigation:* Apply a hard penalty multiplier when `review_count_used` < 10.

---

## Eval Script Usage

```bash
# Run against a specific verdict file
python eval.py --verdict outputs/verdict_chicco_*.json --reviews data/reviews.json

# Expected output (passing)
[eval 1] Schema compliance: PASS ✓
[eval 2] Complaint detection (stiff buckle): PASS ✓
[eval 3] Use-case extraction: PASS ✓
[eval 4] Multilingual coverage {'ar', 'en'}: PASS ✓
[eval 5] Confidence calibration (score=0.74): PASS ✓
  Result: 5/5 evals passed — ALL PASS ✓
```
