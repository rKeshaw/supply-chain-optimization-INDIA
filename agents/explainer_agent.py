"""
Explainer agent: generates the decision-ready plain-language brief.

Strict constraint: it may only restate values explicitly given to it in the prompt.
It is forbidden from inventing any number. The prompt is structured so that every
number the agent might cite is handed to it verbatim.
"""

import json
import logging
from typing import Optional

from agents.llm_client import call_llm

logger = logging.getLogger(__name__)


_SYSTEM_INSTRUCTION = """You are an expert energy policy analyst writing a decision brief for India's
Ministry of Petroleum and Natural Gas.

Your brief will be read by officials who need to act within hours. It must be:
- Factual: you may ONLY cite numbers explicitly given in the data below. NEVER invent figures.
- Concise: 3-4 short paragraphs maximum.
- Action-oriented: lead with the recommended action, then justify with data.
- Transparent: state which numbers are model estimates vs. observed facts.

Return a JSON object with this structure:
{
  "headline": "one-line alert summary (under 100 chars)",
  "situation": "paragraph 1: what happened, which corridor, severity",
  "impact": "paragraph 2: supply loss in bbl/day, shortfall %, price impact, GDP drag",
  "recommendation": "paragraph 3: specific routing action, volume, corridors, SPR draw",
  "caveats": "paragraph 4: confidence levels, model assumptions, what could change",
  "data_sources_cited": ["list of data fields you referenced from the provided context"],
  "numbers_used": {"field_name": value}
}

If confidence is below 0.5, the headline must start with [UNCONFIRMED].
Never start a sentence with 'I'. Write as a system output, not a first-person analyst.
"""


def summarize(
    event,
    validated_routing: dict,
    economic_impact: dict,
    spr_status: dict,
    critic_result: dict,
    graph_state: dict,
) -> dict:
    """
    Generate the decision brief from all upstream module outputs.

    Every number the agent cites is explicitly passed in this prompt — it cannot
    reach outside its context window to invent figures.

    Args:
        event: Triggering Event object (or None for replay mode).
        validated_routing: Output of policy_critic_agent.verify.
        economic_impact: Output of economic_model.compute_cascade.
        spr_status: Output of reserve_optimizer.get_spr_status_summary.
        critic_result: Full critic verification result.
        graph_state: Snapshot of current graph state.

    Returns:
        Dict with headline, situation, impact, recommendation, caveats,
        data_sources_cited, numbers_used. Returns a fallback dict on LLM failure.
    """
    # Build the data context handed to the explainer — this is all it is allowed to use
    event_context = {}
    if event:
        event_context = {
            "entity": event.entity,
            "event_type": event.event_type,
            "severity": event.severity,
            "confidence": event.confidence,
            "justification": event.justification,
            "affected_graph_element": event.affected_graph_element,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
        }

    routing_summary = {}
    pareto = validated_routing.get("pareto_routes", {})
    if pareto:
        comp = pareto.get("pareto_comparison", {})
        routing_summary = {
            "recommended_route": "cost_optimal",
            "volume_delivered_bbl_day": comp.get("cost_optimal", {}).get("volume_delivered"),
            "avg_cost_per_bbl": comp.get("cost_optimal", {}).get("avg_cost_per_bbl"),
            "avg_transit_days": comp.get("cost_optimal", {}).get("avg_transit_days"),
            "fastest_option_transit_days": comp.get("time_optimal", {}).get("avg_transit_days"),
            "routing_feasible": pareto.get("cost_optimal", {}).get("feasible"),
        }

    economic_context = {
        "gap_bbl_day": economic_impact.get("gap_bbl_day"),
        "shortfall_pct": economic_impact.get("shortfall_pct"),
        "shortfall_scope": economic_impact.get("shortfall_scope"),
        "national_shortfall_lower_bound_pct": economic_impact.get("national_shortfall_lower_bound_pct"),
        "crude_price_change_pct": economic_impact.get("crude_price_change_pct"),
        "crude_price_change_range_pct": economic_impact.get("crude_price_change_range_pct"),
        "crude_price_change_scope": economic_impact.get("crude_price_change_scope"),
        "retail_price_change_pct": economic_impact.get("retail_price_change_pct"),
        "retail_price_delayed": economic_impact.get("retail_price_delayed"),
        "gdp_drag_pct": economic_impact.get("gdp_drag_pct"),
        "power_sector_stress": economic_impact.get("power_sector_stress"),
        "import_bill_increase_usd_bn": economic_impact.get("import_bill_increase_usd_bn"),
    }

    spr_context = {
        "total_days_remaining": spr_status.get("total_days_remaining"),
        "status": spr_status.get("status"),
        "total_inventory_bbl": spr_status.get("total_inventory_bbl"),
        "fill_pct": spr_status.get("fill_pct"),
    }

    policy_context = {
        "all_clear": critic_result.get("all_clear"),
        "violations": [v["rule_id"] for v in critic_result.get("violations", [])],
        "re_solve_required": critic_result.get("re_solve_required"),
    }

    full_context = json.dumps({
        "triggering_event": event_context,
        "routing": routing_summary,
        "economic_impact": economic_context,
        "spr_reserve": spr_context,
        "policy_check": policy_context,
    }, indent=2)

    prompt = f"""Generate the decision brief based ONLY on this data:

{full_context}

Do not invent any numbers. Reference only the values above.
Return the full JSON brief object."""

    raw = call_llm(
        prompt=prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.0,
        expect_json=True,
    )

    if raw is None:
        logger.error("Explainer agent: LLM returned None. Returning fallback brief.")
        return _fallback_brief(event, economic_impact, spr_status)

    try:
        brief = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Explainer agent: JSON parse failed: {e}")
        return _fallback_brief(event, economic_impact, spr_status)

    # Audit: log which numbers were cited to verify no hallucination
    numbers_used = brief.get("numbers_used", {})
    if numbers_used:
        logger.info(f"Explainer cited {len(numbers_used)} data points: {list(numbers_used.keys())}")

    return brief


def _fallback_brief(
    event,
    economic_impact: dict,
    spr_status: dict,
) -> dict:
    """
    Fallback brief when LLM is unavailable — constructed from structured data only.
    All values are read directly from upstream module outputs, no LLM involved.
    """
    entity = event.entity if event else "Unknown"
    severity = event.severity if event else 0.0
    confidence = event.confidence if event else 0.0
    shortfall = economic_impact.get("shortfall_pct", 0)
    price_change = economic_impact.get("crude_price_change_pct", 0)
    spr_days = spr_status.get("total_days_remaining", 0)
    spr_fill = spr_status.get("status", "UNKNOWN")

    conf_prefix = "[UNCONFIRMED] " if confidence < 0.5 else ""

    return {
        "headline": f"{conf_prefix}Supply disruption: {entity} — {shortfall:.1f}% shortfall",
        "situation": (
            f"A {event.event_type if event else 'disruption'} event affecting {entity} "
            f"has been detected with severity {severity:.2f} and confidence {confidence:.2f}. "
            "Full LLM brief unavailable — this is an automated structured summary."
        ),
        "impact": (
            f"Estimated supply gap: {economic_impact.get('gap_bbl_day', 0):,.0f} bbl/day "
            f"({shortfall:.1f}% of baseline). "
            f"Crude price impact: +{price_change:.1f}%. "
            f"GDP drag: {economic_impact.get('gdp_drag_pct', 0):.2f}pp (if sustained >14 days)."
        ),
        "recommendation": (
            f"Activate rerouting via available corridors. "
            f"SPR status: {spr_fill} ({spr_days:.1f} days cover). "
            "Draw from SPR if refinery inventory drops below 3-day safety floor."
        ),
        "caveats": "LLM explainer unavailable. All numbers are from model outputs, not verified observations.",
        "data_sources_cited": ["economic_impact", "spr_status", "event"],
        "numbers_used": {
            "shortfall_pct": shortfall,
            "crude_price_change_pct": price_change,
            "spr_days_remaining": spr_days,
        },
        "_fallback": True,
    }
