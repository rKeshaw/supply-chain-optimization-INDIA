"""
Extraction accuracy evaluation against data/crisis_timeline.json's own labeled
expected_extraction data. Signal-detection accuracy is the first line of the
problem statement's evaluation focus, and it needs a test despite the labels
already existing in the repo.

Unlike every other test in this suite, this one calls the REAL LLM (all other
extraction tests mock agents.llm_client.call_llm). That's a deliberate,
different kind of test — an accuracy eval, not a unit test of parsing logic —
so it's skipped by default: the main suite (pytest tests/ -q) must stay fast,
free, and independent of network/API availability. Opt in with
RUN_EXTRACTION_EVAL=1 (and a configured GROQ_API_KEYS/GROQ_API_KEY).

For a clean, deck-ready accuracy table without pytest's output formatting,
run extraction_eval_report.py at the repo root instead — it reuses
evaluate_extraction() below directly.
"""
import json
import os
from pathlib import Path

import pytest

from agents.extraction_agent import parse

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_timeline() -> list[dict]:
    return json.loads((DATA_DIR / "crisis_timeline.json").read_text(encoding="utf-8"))


def evaluate_extraction() -> list[dict]:
    """Run every curated event through the real extractor (agents.extraction_agent.parse,
    not event_from_curated_timeline's replay bypass) and compare against its own
    expected_extraction label. Returns one result dict per event, in timeline order."""
    results = []
    for item in _load_timeline():
        raw_text = f"{item['headline']}\n\n{item['body_excerpt']}"
        expected = item["expected_extraction"]
        event = parse(raw_text, source_override=item.get("source"))

        if event is None:
            results.append({
                "id": item["id"],
                "headline": item["headline"],
                "extracted": False,
                "expected_event_type": expected.get("event_type"),
                "expected_element": expected.get("affected_graph_element"),
                "event_type_correct": False,
                "element_correct": False,
                "severity_in_range": False,
                "confidence_in_range": False,
            })
            continue

        sev_lo, sev_hi = expected.get("severity_range", [0.0, 1.0])
        conf_lo, conf_hi = expected.get("confidence_range", [0.0, 1.0])

        results.append({
            "id": item["id"],
            "headline": item["headline"],
            "extracted": True,
            "expected_event_type": expected.get("event_type"),
            "actual_event_type": event.event_type,
            "event_type_correct": event.event_type == expected.get("event_type"),
            "expected_element": expected.get("affected_graph_element"),
            "actual_element": event.affected_graph_element,
            "element_correct": event.affected_graph_element == expected.get("affected_graph_element"),
            "actual_severity": event.severity,
            "severity_in_range": sev_lo <= event.severity <= sev_hi,
            "actual_confidence": event.confidence,
            "confidence_in_range": conf_lo <= event.confidence <= conf_hi,
        })
    return results


def test_extraction_accuracy_meets_baseline():
    """Real-LLM accuracy check against the curated timeline's own labels.

    Thresholds are set from two actual runs against this repo's real LLM
    (Groq llama-3.3-70b-versatile, temperature=0.0), not an assumed-safe
    guess: event_type accuracy measured 92% then 75% run-to-run on the same
    12 headlines (genuine LLM variance, not flakiness in this test — several
    of these headlines are legitimately ambiguous between e.g. "closure" and
    "capacity_reduction"), while affected_graph_element measured 100% both
    times — a materially more reliable signal, since naming which corridor
    is usually explicit in the text where judging an event's exact severity
    tier is a closer call. The 65%/90% thresholds below sit comfortably
    under both observed runs for each metric respectively, so this catches a
    genuine regression (extraction broadly breaking) without being flaky
    over the ordinary run-to-run drift already measured.
    """
    if not os.environ.get("RUN_EXTRACTION_EVAL"):
        pytest.skip(
            "Calls the real LLM — skipped by default so the main suite stays "
            "fast/deterministic. Opt in with RUN_EXTRACTION_EVAL=1."
        )

    results = evaluate_extraction()
    n = len(results)
    event_type_acc = sum(r["event_type_correct"] for r in results) / n
    element_acc = sum(r["element_correct"] for r in results) / n

    for r in results:
        status = "OK" if r["event_type_correct"] and r["element_correct"] else "MISS"
        print(
            f"[{status}] {r['id']}: event_type={r.get('actual_event_type')!r} "
            f"(expected {r['expected_event_type']!r}), "
            f"element={r.get('actual_element')!r} (expected {r['expected_element']!r})"
        )

    assert event_type_acc >= 0.65, f"event_type accuracy {event_type_acc:.0%} below 65% baseline"
    assert element_acc >= 0.90, f"affected_graph_element accuracy {element_acc:.0%} below 90% baseline"
