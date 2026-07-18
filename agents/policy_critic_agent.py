"""
Policy Critic agent: validates proposed routing allocations against hard domain constraints.

On violation, it mutates the constraint set and signals a re-solve rather than
silently passing a physically infeasible plan to the explainer.

Constraints are stored as plain text/config — not hardcoded in agent logic —
so they can be updated by operators without touching code.
"""

import json
import logging
from typing import Optional

from agents.llm_client import call_llm

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Hard constraint rules (plain-text, operator-editable).
# These are the domain constraints the Policy Critic checks against.
# Format: list of rule dicts with id, description, check_type, threshold.
# -----------------------------------------------------------------------
POLICY_RULES = [
    {
        "id": "rule_sour_ratio_jamnagar",
        "description": (
            "Jamnagar refinery (Reliance) can process any crude grade due to NCI 21.1 "
            "complexity — no SOUR ratio restriction. This rule is intentionally open "
            "to reflect real operational flexibility."
        ),
        "check_type": "grade_ratio",
        "node_id": "ref_jamnagar_in",
        "max_sour_fraction": 1.0,  # no restriction
        "triggered": False,
    },
    {
        "id": "rule_sour_ratio_paradip",
        "description": (
            "Paradip (IOCL) is designed for SOUR crude. SWEET crude share must not "
            "exceed 30% of its intake — its secondary processing units are calibrated "
            "for high-sulfur feedstock. Exceeding SWEET ratio degrades yield efficiency."
        ),
        "check_type": "grade_ratio",
        "node_id": "ref_paradip_in",
        "max_sweet_fraction": 0.30,
        "triggered": False,
    },
    {
        "id": "rule_spr_floor_draw",
        "description": (
            "SPR drawdown must not reduce any single facility below 10% of its design "
            "capacity — operating below this risks cavern structural integrity and "
            "complicates future refill (per ISPRL operational guidelines)."
        ),
        "check_type": "spr_floor",
        "min_fill_fraction": 0.10,
        "triggered": False,
    },
    {
        "id": "rule_cape_congestion_cap",
        "description": (
            "Cape of Good Hope bypass routes are capped at 60% of total rerouted volume. "
            "Over-reliance on a single bypass route creates a new concentration risk. "
            "This is an operational diversification rule, not a physical constraint."
        ),
        "check_type": "corridor_concentration",
        "corridor_id": "chk_cog",
        "max_fraction_of_total": 0.60,
        "triggered": False,
    },
]


def get_re_solve_overrides(critic_result: dict) -> dict:
    """Extract solver-safe policy overrides from a critic response.

    The LLM is allowed to describe a correction in prose, but only explicitly
    named numeric constraint keys can change a subsequent solve. This prevents
    an unvalidated model response from silently altering optimisation rules.
    """
    corrected = critic_result.get("corrected_constraints", {}) or {}
    overrides = {}
    raw_cape_cap = corrected.get("max_cape_fraction_of_total")
    if raw_cape_cap is None:
        raw_cape_cap = corrected.get("rule_cape_congestion_cap", {}).get(
            "max_fraction_of_total"
        ) if isinstance(corrected.get("rule_cape_congestion_cap"), dict) else None
    if raw_cape_cap is not None:
        try:
            overrides["max_cape_fraction_of_total"] = max(0.0, min(1.0, float(raw_cape_cap)))
        except (TypeError, ValueError):
            pass
    return overrides


_SYSTEM_INSTRUCTION = """You are a domain expert policy critic for an Indian oil supply chain.

You will receive a proposed routing allocation and a set of policy rules.
Your job is to identify which rules (if any) are violated and propose specific
numeric adjustments to fix each violation.

Return a JSON object:
{
  "violations": [
    {
      "rule_id": "string",
      "violated": true/false,
      "explanation": "why the rule is violated or not",
      "suggested_correction": "specific numeric change to routing or constraint"
    }
  ],
  "all_clear": true/false,
  "re_solve_required": true/false,
  "corrected_constraints": {}
}

If re_solve_required is true, corrected_constraints must specify what needs to change.
Be precise — cite specific node IDs and numeric thresholds.
"""


def verify(
    routing_result: dict,
    spr_state: dict,
    graph_state: dict,
    params: dict,
) -> dict:
    """
    Run policy critic check on the proposed routing allocation.

    Args:
        routing_result: Output of compute_pareto_routes (cost_optimal branch used).
        spr_state: Current SPR state from reserve_optimizer.get_spr_status_summary.
        graph_state: Current graph state dict.
        params: Parameters dict.

    Returns:
        Dict with:
        - violations: List of violated rules (may be empty)
        - all_clear: True if no violations
        - re_solve_required: True if solver must re-run with tighter constraints
        - corrected_constraints: New constraint values for re-solve (if needed)
        - critic_response: Full LLM response (for explainer audit trail)
    """
    # First: run rule checks in code (fast, deterministic) before calling LLM
    code_violations = _run_code_checks(routing_result, spr_state, graph_state, params)

    # If code checks find nothing, skip the LLM call entirely (save latency)
    if not code_violations:
        return {
            "violations": [],
            "all_clear": True,
            "re_solve_required": False,
            "corrected_constraints": {},
            "critic_response": "All policy rules satisfied (code-path check — LLM not called).",
        }

    # LLM for nuanced validation when code checks flag something
    routing_summary = json.dumps(
        routing_result.get("cost_optimal", {}).get("routing_summary", []),
        indent=2,
    )
    spr_summary = json.dumps(spr_state, indent=2)
    rules_summary = json.dumps(
        [{"id": r["id"], "description": r["description"]} for r in POLICY_RULES],
        indent=2,
    )
    code_flags = json.dumps(code_violations, indent=2)

    prompt = f"""Routing allocation (active segments with non-zero flow):
{routing_summary}

SPR state:
{spr_summary}

Policy rules to check:
{rules_summary}

Code pre-check found potential violations:
{code_flags}

Validate the routing against these rules. Return the full JSON critic response."""

    raw = call_llm(
        prompt=prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.0,
        expect_json=True,
    )

    if raw is None:
        logger.error("Policy critic: LLM returned None. Failing open (routing passes).")
        return {
            "violations": code_violations,
            "all_clear": False,
            "re_solve_required": len(code_violations) > 0,
            "corrected_constraints": {},
            "critic_response": "LLM unavailable — code-path violations reported only.",
        }

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Policy critic: JSON parse failed: {e}")
        response = {
            "violations": code_violations,
            "all_clear": False,
            "re_solve_required": True,
            "corrected_constraints": {},
        }

    response["critic_response"] = raw
    return response


def _run_code_checks(
    routing_result: dict,
    spr_state: dict,
    graph_state: dict,
    params: dict,
) -> list[dict]:
    """
    Run deterministic code-path policy checks before invoking the LLM.

    Returns a list of violation dicts (empty if all rules pass).
    """
    violations = []
    cost_routing = routing_result.get("cost_optimal", {})
    routing_segments = cost_routing.get("routing_summary", [])

    # Check SPR floor (rule_spr_floor_draw)
    for facility_id, facility_state in spr_state.get("per_facility", {}).items():
        fill_pct = facility_state.get("fill_pct", 100)
        if fill_pct < 10.0:
            violations.append({
                "rule_id": "rule_spr_floor_draw",
                "violated": True,
                "fill_pct": fill_pct,
                "facility_id": facility_id,
                "explanation": f"{facility_id} fill at {fill_pct:.1f}% — below 10% structural floor.",
            })

    # Check Cape concentration (rule_cape_congestion_cap)
    total_volume = sum(s.get("volume_bbl_day", 0) for s in routing_segments)
    cape_volume = sum(
        s.get("volume_bbl_day", 0)
        for s in routing_segments
        if s.get("from_id") == "chk_cog" or s.get("to_id") == "chk_cog"
    )
    if total_volume > 0 and (cape_volume / total_volume) > 0.60:
        violations.append({
            "rule_id": "rule_cape_congestion_cap",
            "violated": True,
            "cape_fraction": round(cape_volume / total_volume, 3),
            "explanation": (
                f"Cape of Good Hope carries {cape_volume/total_volume*100:.1f}% of rerouted "
                "volume — exceeds 60% concentration cap."
            ),
        })

    # Check refinery grade/blend restrictions using complete source-to-refinery
    # allocations. Segment-level flow cannot establish which crude arrived at a
    # refinery once paths share a chokepoint.
    path_allocations = cost_routing.get("path_allocations", [])
    for rule in POLICY_RULES:
        if rule.get("check_type") != "grade_ratio":
            continue
        refinery_allocations = [
            allocation for allocation in path_allocations
            if allocation.get("refinery_in") == rule.get("node_id")
        ]
        total = sum(float(a.get("volume_bbl_day", 0)) for a in refinery_allocations)
        if total <= 0:
            continue
        sweet = sum(
            float(a.get("volume_bbl_day", 0)) for a in refinery_allocations
            if a.get("grade") == "SWEET"
        )
        sour = sum(
            float(a.get("volume_bbl_day", 0)) for a in refinery_allocations
            if a.get("grade") == "SOUR"
        )
        sweet_fraction = sweet / total
        sour_fraction = sour / total
        if "max_sweet_fraction" in rule and sweet_fraction > rule["max_sweet_fraction"]:
            violations.append({
                "rule_id": rule["id"],
                "violated": True,
                "sweet_fraction": round(sweet_fraction, 3),
                "explanation": (
                    f"{rule['node_id']} receives {sweet_fraction:.1%} SWEET crude, "
                    f"above its {rule['max_sweet_fraction']:.1%} policy maximum."
                ),
            })
        if "max_sour_fraction" in rule and sour_fraction > rule["max_sour_fraction"]:
            violations.append({
                "rule_id": rule["id"],
                "violated": True,
                "sour_fraction": round(sour_fraction, 3),
                "explanation": (
                    f"{rule['node_id']} receives {sour_fraction:.1%} SOUR crude, "
                    f"above its {rule['max_sour_fraction']:.1%} policy maximum."
                ),
            })

    return violations
