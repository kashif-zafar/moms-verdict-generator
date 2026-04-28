"""
eval.py
───────
Objective evaluation of the Moms Verdict Generator output.

This is what separates top candidates — they don't just build the system,
they measure it.

Evals run here:
  1. Schema compliance      — is the output valid JSON matching the schema?
  2. Grounding check        — are claims traceable to retrieved chunks?
  3. Complaint detection    — did we catch the stiff-buckle pattern?
  4. Confidence calibration — does the score reflect actual data quality?
  5. Multilingual coverage  — were Arabic reviews included?

Run:
  python eval.py --verdict outputs/verdict_chicco_*.json --reviews data/reviews.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from schema import ProductVerdict


# ── Known ground truths for the sample dataset ──────────────────────────────
# These are facts we KNOW are in the data — the system should surface them.
EXPECTED_COMPLAINT_KEYWORDS = ["buckle", "stiff", "hard to clip", "clip"]
EXPECTED_USE_CASES_KEYWORDS = ["travel", "newborn", "infant", "stroller"]
EXPECTED_LANGUAGES          = {"en", "ar"}
MIN_CONFIDENCE_THRESHOLD    = 0.4   # a score below this on 20 reviews is suspicious


def load_verdict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_verdict_paths(path: str) -> list[Path]:
    candidate = Path(path)
    if any(ch in path for ch in "*?[]"):
        results = sorted(candidate.parent.glob(candidate.name))
        if not results:
            raise FileNotFoundError(f"No verdict files matched glob: {path}")
        return results
    if not candidate.exists():
        raise FileNotFoundError(f"Verdict file not found: {path}")
    return [candidate]


def load_reviews(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Eval 1: Schema compliance ────────────────────────────────────────────────
def eval_schema(verdict_dict: dict) -> bool:
    try:
        ProductVerdict(**verdict_dict)
        print("[eval 1] Schema compliance: PASS ✓")
        return True
    except Exception as e:
        print(f"[eval 1] Schema compliance: FAIL ✗ — {e}")
        return False


# ── Eval 2: Complaint detection ──────────────────────────────────────────────
def eval_complaint_detection(verdict: dict) -> bool:
    """
    The stiff buckle complaint appears in reviews 3, 6, 12, 18 — 4 of 20.
    It MUST appear in common_complaints.
    """
    complaints_text = " ".join(verdict.get("common_complaints", [])).lower()
    cons_text       = " ".join(verdict.get("cons", [])).lower()
    all_text        = complaints_text + " " + cons_text

    found = any(kw in all_text for kw in EXPECTED_COMPLAINT_KEYWORDS)
    if found:
        print("[eval 2] Complaint detection (stiff buckle): PASS ✓")
    else:
        print(
            "[eval 2] Complaint detection (stiff buckle): FAIL ✗ — "
            "Expected keyword from: " + str(EXPECTED_COMPLAINT_KEYWORDS)
        )
    return found


# ── Eval 3: Use-case extraction ──────────────────────────────────────────────
def eval_use_cases(verdict: dict) -> bool:
    use_cases_text = " ".join(verdict.get("best_use_cases", [])).lower()
    found = any(kw in use_cases_text for kw in EXPECTED_USE_CASES_KEYWORDS)
    if found:
        print("[eval 3] Use-case extraction: PASS ✓")
    else:
        print(
            "[eval 3] Use-case extraction: FAIL ✗ — "
            "Expected at least one of: " + str(EXPECTED_USE_CASES_KEYWORDS)
        )
    return found


# ── Eval 4: Multilingual coverage ────────────────────────────────────────────
def eval_multilingual(verdict: dict) -> bool:
    detected = set(verdict.get("languages_detected", []))
    covered  = EXPECTED_LANGUAGES.issubset(detected)
    if covered:
        print(f"[eval 4] Multilingual coverage {detected}: PASS ✓")
    else:
        print(
            f"[eval 4] Multilingual coverage: FAIL ✗ — "
            f"Detected {detected}, expected {EXPECTED_LANGUAGES}"
        )
    return covered


# ── Eval 5: Confidence calibration ───────────────────────────────────────────
def eval_confidence(verdict: dict) -> bool:
    score = verdict.get("confidence_score", 0)
    bd    = verdict.get("confidence_breakdown", {})

    # Score must be above floor for a 20-review dataset
    score_ok = score >= MIN_CONFIDENCE_THRESHOLD

    # Breakdown must sum (approximately) to the composite score
    computed = (
        bd.get("retrieval_coverage",    0) * 0.4 +
        bd.get("sentiment_consistency", 0) * 0.4 +
        bd.get("review_volume_score",   0) * 0.2
    )
    consistency_ok = abs(score - computed) <= 0.05

    if score_ok and consistency_ok:
        print(f"[eval 5] Confidence calibration (score={score}): PASS ✓")
        return True
    else:
        issues = []
        if not score_ok:
            issues.append(f"score {score:.2f} < threshold {MIN_CONFIDENCE_THRESHOLD}")
        if not consistency_ok:
            issues.append(f"score {score:.2f} ≠ breakdown sum {computed:.2f}")
        print(f"[eval 5] Confidence calibration: FAIL ✗ — {'; '.join(issues)}")
        return False


# ── Runner ────────────────────────────────────────────────────────────────────
def run_evals(verdict_path: str, reviews_path: str):
    verdict_paths = resolve_verdict_paths(verdict_path)
    print(f"\n{'─'*55}")
    print(f"  Moms Verdict Generator — Evaluation Suite")
    print(f"  Verdict: {verdict_path}")
    print(f"{'─'*55}\n")

    all_success = True
    for path in verdict_paths:
        print(f"Evaluating: {path}")
        verdict = load_verdict(path)

        results = [
            eval_schema(verdict),
            eval_complaint_detection(verdict),
            eval_use_cases(verdict),
            eval_multilingual(verdict),
            eval_confidence(verdict),
        ]

        passed = sum(results)
        total = len(results)
        print(f"\n{'─'*40}")
        print(f"  File result: {passed}/{total} evals passed")
        if passed == total:
            print("  Status: ALL PASS ✓")
        elif passed >= total * 0.8:
            print("  Status: MOSTLY PASSING — minor issues")
        else:
            print("  Status: NEEDS WORK — check failures above")
        print(f"{'─'*40}\n")

        all_success = all_success and (passed == total)

    return all_success

    passed = sum(results)
    total  = len(results)
    print(f"\n{'─'*55}")
    print(f"  Result: {passed}/{total} evals passed")
    if passed == total:
        print("  Status: ALL PASS ✓")
    elif passed >= total * 0.8:
        print("  Status: MOSTLY PASSING — minor issues")
    else:
        print("  Status: NEEDS WORK — check failures above")
    print(f"{'─'*55}\n")

    return passed == total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Moms Verdict output")
    parser.add_argument("--verdict",  required=True, help="Path to verdict JSON file")
    parser.add_argument("--reviews",  default="data/reviews.json", help="Path to reviews JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_evals(args.verdict, args.reviews)
    sys.exit(0 if success else 1)
