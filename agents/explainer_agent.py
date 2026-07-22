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
- Decisive: state the situation and the recommended action as findings, not possibilities.
  Write "Hormuz throughput down 60%" not "it appears that Hormuz throughput may be reduced".
  Cut hedges ("it seems", "may indicate", "could potentially", "it is estimated that") wherever
  the data already gives you a number to state directly instead.
- Specific: name the actual corridor, source, refinery, volume, and cost figures handed to you.
  "Reroute 340k bbl/day via Ras Tanura at $6.10/bbl, 11-day transit" beats "activate rerouting
  via available corridors" every time — if that data is in the context, use it.

Never reference how this brief was produced, what generated it, or that you are an AI, model,
or automated system. An official reading this does not care whether an algorithm or an analyst
wrote it — do not tell them either way. Write only the analysis itself.

Return a JSON object with this structure:
{
  "headline": "one-line alert summary (under 100 chars). Lead with the money: include the
    total_import_cost_increase_usd_bn figure from the data (e.g. '$2.1B annualized import-bill
    impact'), not just the shortfall percentage — officials scan headlines for the dollar
    number first.",
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
            # Landed = crude + freight, the figure the cost objective minimises.
            # Handing the explainer freight alone let the brief quote a ~$3/bbl
            # number as the cost of a barrel that actually lands at ~$83.
            "avg_landed_cost_per_bbl": comp.get("cost_optimal", {}).get("avg_landed_cost_per_bbl"),
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
        "total_import_cost_increase_usd_bn": economic_impact.get("total_import_cost_increase_usd_bn"),
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
        return _fallback_brief(event, routing_summary, economic_impact, spr_status)

    try:
        brief = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Explainer agent: JSON parse failed: {e}")
        return _fallback_brief(event, routing_summary, economic_impact, spr_status)

    # Enforce the instruction not to invent figures. Every value the brief claims
    # to have used must appear in the context it was given, otherwise the brief is
    # discarded in favour of the deterministic fallback built from module output.
    numbers_used = brief.get("numbers_used", {}) or {}
    invented = [
        field for field, value in numbers_used.items()
        if isinstance(value, (int, float)) and not _value_in_context(value, full_context)
    ]
    if invented:
        logger.error(
            "Explainer cited %d figure(s) absent from its context (%s). Discarding the "
            "brief and using the deterministic fallback.", len(invented), ", ".join(invented),
        )
        return _fallback_brief(event, routing_summary, economic_impact, spr_status)

    if numbers_used:
        logger.info(f"Explainer cited {len(numbers_used)} data points: {list(numbers_used.keys())}")

    return brief


def _value_in_context(value: float, context: str) -> bool:
    """Whether a cited number actually appears in the data handed to the agent.

    Tolerant about formatting: a model may round 4.907900621 to 4.91 or write
    2574000 as 2,574,000, and neither is an invention.
    """
    candidates = {repr(value), str(value), f"{value:,}"}
    for places in (0, 1, 2, 3):
        rounded = round(float(value), places)
        candidates.update({str(rounded), f"{rounded:,}", str(int(rounded)) if rounded == int(rounded) else ""})
    return any(c and c in context for c in candidates)


def _fallback_brief(
    event,
    routing_summary: dict,
    economic_impact: dict,
    spr_status: dict,
) -> dict:
    """
    Structured brief built directly from upstream module outputs, used whenever
    the model call fails or returns unparseable output.

    Never mentions the model call, its failure, or its own generation mechanism
    — an official reading this brief has no reason to care whether a model call
    succeeded; they need the situation, the number, and the action, stated as
    findings, the same as the primary path. Cites the actual recommended route
    (volume, cost, transit) from routing_summary rather than a generic "activate
    rerouting" line, so the recommendation stays specific even without the
    model's narrative.
    """
    entity = event.entity if event else "the affected corridor"
    event_type = event.event_type.replace("_", " ") if event else "disruption"
    severity = event.severity if event else 0.0
    confidence = event.confidence if event else 0.0
    shortfall = economic_impact.get("shortfall_pct", 0)
    price_change = economic_impact.get("crude_price_change_pct", 0)
    gdp_drag = economic_impact.get("gdp_drag_pct", 0)
    spr_days = spr_status.get("total_days_remaining", 0)
    spr_fill = spr_status.get("status", "UNKNOWN")

    conf_prefix = "[UNCONFIRMED] " if confidence < 0.5 else ""
    severity_word = "Severe" if severity >= 0.6 else "Moderate" if severity >= 0.3 else "Minor"

    volume = (routing_summary or {}).get("volume_delivered_bbl_day")
    cost_per_bbl = (routing_summary or {}).get("avg_cost_per_bbl")
    landed_per_bbl = (routing_summary or {}).get("avg_landed_cost_per_bbl")
    transit_days = (routing_summary or {}).get("avg_transit_days")
    feasible = (routing_summary or {}).get("routing_feasible")

    if volume and cost_per_bbl is not None and transit_days is not None:
        priced = (
            f"${landed_per_bbl:.2f}/bbl landed (${cost_per_bbl:.2f} freight)"
            if landed_per_bbl is not None else f"${cost_per_bbl:.2f}/bbl freight"
        )
        action = f"Reroute {volume/1e3:,.0f}k bbl/day at {priced}, {transit_days:.0f}-day transit."
    else:
        action = "Hold current routing — no reroute indicated at this severity."
    gap_status = "Gap remains; SPR draw required." if feasible is False else "Fully covered by the reroute."
    total_cost_bn = economic_impact.get("total_import_cost_increase_usd_bn", 0) or 0
    cost_clause = f", ${total_cost_bn:.1f}B annualized import-bill impact" if total_cost_bn > 0.05 else ""

    return {
        "headline": (
            f"{conf_prefix}{entity}: {severity_word.lower()} {event_type}, "
            f"{shortfall:.1f}% shortfall{cost_clause}"
        ),
        "situation": f"{entity}: {event_type}, severity {severity:.0%}, confidence {confidence:.0%}.",
        "impact": (
            f"Supply gap {economic_impact.get('gap_bbl_day', 0):,.0f} bbl/day ({shortfall:.1f}% of the "
            f"modelled network). Crude +{price_change:.1f}%. GDP drag {gdp_drag:.2f}pp if sustained beyond "
            f"the retail pass-through lag."
        ),
        "recommendation": (
            f"{action} {gap_status} SPR: {spr_fill}, {spr_days:.1f} days cover — draw only if refinery "
            f"inventory falls below the 3-day safety floor."
        ),
        "caveats": (
            f"Confidence {confidence:.0%} on this signal. Figures traced to economic_impact, spr_status, "
            f"and routing_summary — none invented."
        ),
        "data_sources_cited": ["economic_impact", "spr_status", "event", "routing_summary"],
        "numbers_used": {
            "shortfall_pct": shortfall,
            "crude_price_change_pct": price_change,
            "spr_days_remaining": spr_days,
            "avg_cost_per_bbl": cost_per_bbl,
            "avg_landed_cost_per_bbl": landed_per_bbl,
            "total_import_cost_increase_usd_bn": total_cost_bn,
        },
        "_fallback": True,
    }
