"""
Scenario agent: proactively surfaces rising-risk hypotheses from the
current risk score state, before any single event justifies treating
a corridor as definitively disrupted.

This is the anticipatory layer — distinct from the reactive extraction agent.
"""

import json
import logging
from typing import Optional

from agents.llm_client import call_llm

logger = logging.getLogger(__name__)


_SYSTEM_INSTRUCTION = """You are a geopolitical energy risk analyst.

Given the current risk signals in an energy supply chain, you identify corridors
that are accumulating risk faster than the signals alone justify acting on,
and propose plausible near-term disruption hypotheses.

These hypotheses are ADVISORY ONLY. They help pre-warm scenario analysis
but never directly update graph state — only confirmed events (from the
extraction agent) do that.

Return a JSON array of hypothesis objects:
[
  {
    "corridor_id": "graph element ID (e.g. chk_hormuz)",
    "hypothesis": "plain-language hypothesis description",
    "evidence_basis": "which current risk signals support this",
    "escalation_probability": "float 0-1, analyst's estimate of probability within 14 days",
    "recommended_precompute": ["list of scenario IDs to pre-warm"],
    "urgency": "LOW | MEDIUM | HIGH"
  }
]

Return an empty array [] if no corridors are accumulating concerning risk patterns.
Escalation probability must be an honest estimate, not alarmist.
"""


def generate_hypotheses(
    graph_state: dict,
    recent_events: list,
    params: dict,
) -> list[dict]:
    """
    Analyze current risk scores and recent events to surface proactive hypotheses.

    Triggers only when at least one corridor has risk_score > 0.2 — below that
    threshold there is nothing worth surfacing.

    Args:
        graph_state: Current graph state dict (from get_graph_state_json).
        recent_events: List of recent Event dicts processed in the last 7 days.
        params: Parameters dict.

    Returns:
        List of hypothesis dicts. Empty list if no notable risk accumulation.
    """
    # Filter to nodes with meaningful risk scores
    elevated_nodes = [
        n for n in graph_state.get("nodes", [])
        if n.get("risk_score", 0) > 0.2
        and n.get("type") in ("chokepoint", "source")
    ]

    if not elevated_nodes:
        logger.info("Scenario agent: no corridors with risk_score > 0.2. No hypotheses.")
        return []

    # Build prompt context
    elevated_summary = json.dumps(
        [
            {
                "id": n["id"],
                "name": n.get("name"),
                "type": n.get("type"),
                "risk_score": round(n["risk_score"], 3),
                "openness": round(n.get("openness", 1.0), 3),
            }
            for n in elevated_nodes
        ],
        indent=2,
    )

    recent_summary = json.dumps(
        [
            {
                "entity": e.get("entity"),
                "event_type": e.get("event_type"),
                "severity": e.get("severity"),
                "confidence": e.get("confidence"),
                "timestamp": e.get("timestamp"),
            }
            for e in (recent_events or [])[-10:]  # last 10 events max
        ],
        indent=2,
    )

    prompt = f"""Current supply chain elements with elevated risk:
{elevated_summary}

Recent events processed (last 10):
{recent_summary}

Identify any corridors accumulating risk that warrant pre-emptive scenario analysis.
Return your assessment as a JSON array of hypothesis objects."""

    raw = call_llm(
        prompt=prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.1,  # slight temperature for creative hypothesis generation
        expect_json=True,
    )

    if raw is None:
        logger.error("Scenario agent: LLM returned None.")
        return []

    try:
        hypotheses = json.loads(raw)
        if not isinstance(hypotheses, list):
            hypotheses = [hypotheses]
    except json.JSONDecodeError as e:
        logger.error(f"Scenario agent: JSON parse failed: {e}")
        return []

    # Validate and filter: only return hypotheses about known graph elements
    valid_ids = {n["id"] for n in graph_state.get("nodes", [])}
    filtered = []
    for h in hypotheses:
        corridor = h.get("corridor_id")
        if corridor and corridor not in valid_ids:
            logger.debug(f"Scenario agent: unknown corridor_id '{corridor}' — skipping.")
            continue
        # Clamp probability to [0, 1]
        prob = h.get("escalation_probability", 0)
        try:
            h["escalation_probability"] = max(0.0, min(1.0, float(prob)))
        except (TypeError, ValueError):
            h["escalation_probability"] = 0.0
        filtered.append(h)

    logger.info(f"Scenario agent: {len(filtered)} hypothesis(es) generated.")
    return filtered
