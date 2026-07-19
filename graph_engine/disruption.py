"""
Disruption and degradation model for the Energy Supply Chain Resilience system.

Applies scenarios to the graph and recomputes flow — without ever mutating the baseline.
A scenario is a dict mapping graph element IDs to openness multipliers in [0, 1].
"""

import copy
from typing import Optional

import networkx as nx


# Default named scenario library — as specified in plan.md Section 4.
# These are one-click demo scenarios plus the basis for backtesting.
DEFAULT_SCENARIOS: dict[str, dict] = {
    "hormuz_partial": {
        "name": "Hormuz Partial Closure",
        "description": (
            "Strait of Hormuz capacity reduced to 40% — models a major maritime incident, "
            "IRGC vessel interdictions, or partial blockade. Reflects the most-cited risk "
            "scenario from the 2025 US-Iran standoff."
        ),
        "scenario_dict": {"chk_hormuz": 0.4},
        "affected_element": "chk_hormuz",
    },
    "hormuz_full": {
        "name": "Hormuz Full Closure",
        "description": (
            "Strait of Hormuz capacity reduced to 0% — complete closure. Worst-case scenario "
            "that would immediately cut off ~40-45% of India's crude imports."
        ),
        "scenario_dict": {"chk_hormuz": 0.0},
        "affected_element": "chk_hormuz",
    },
    "red_sea_suspension": {
        "name": "Red Sea / Bab-el-Mandeb Suspension",
        "description": (
            "Bab-el-Mandeb fully closed — all traffic must reroute via Cape of Good Hope. "
            "Models an escalation of Houthi attacks to complete maritime exclusion. "
            "Note: in 2024, global Red Sea flows already dropped from 9.3 to 4.1 Mb/d "
            "due to Houthi actions — this is partially real today."
        ),
        "scenario_dict": {"chk_bab": 0.0},
        "affected_element": "chk_bab",
    },
    "opec_cut": {
        "name": "OPEC+ Emergency Production Cut",
        "description": (
            "Source-side capacity reduction: 15% cut applied to all OPEC+ member source nodes "
            "(Iraq, Saudi Arabia, UAE, Kuwait). Models an emergency OPEC+ meeting response to "
            "a geopolitical crisis or demand shock."
        ),
        "scenario_dict": {
            "src_iraq": 0.85,
            "src_saudi": 0.85,
            "src_uae": 0.85,
            "src_kuwait": 0.85,
        },
        "affected_element": "src_saudi",  # representative element for display
    },
    "correlated_gulf_crisis": {
        "name": "Correlated Gulf Crisis (Hormuz + Red Sea)",
        "description": (
            "Both the Strait of Hormuz and Bab-el-Mandeb fully closed simultaneously — models "
            "a compounding regional crisis rather than an isolated single-corridor incident. "
            "The mechanism already exists via the custom-scenario API (verified this session: "
            "combining both closures correctly triggers non-linear SPR drawdown once the "
            "network's remaining slack — the Cape of Good Hope bypass and diversified sources — "
            "runs out); this is that same combination as a one-click named scenario instead of "
            "only reachable through the raw custom-scenario request."
        ),
        "scenario_dict": {"chk_hormuz": 0.0, "chk_bab": 0.0},
        "affected_element": "chk_hormuz",  # representative element for display
    },
}


def apply_scenario(
    G: nx.DiGraph,
    scenario_dict: dict[str, float],
) -> nx.DiGraph:
    """
    Apply a disruption scenario to the graph and return a new graph.
    NEVER mutates the original graph.

    Args:
        G: The baseline (or current) graph.
        scenario_dict: Dict mapping graph_element_id -> openness_multiplier in [0, 1].
                       Can reference node IDs (adjusts all incident edges) or
                       edge IDs via their 'edge_id' attribute.

    Returns:
        A deep copy of G with adjusted edge capacities reflecting the disruption.
        The original G is unchanged.
    """
    G_disrupted = copy.deepcopy(G)

    for element_id, openness_multiplier in scenario_dict.items():
        openness_multiplier = max(0.0, min(1.0, float(openness_multiplier)))

        if element_id in G_disrupted.nodes:
            # Node disruption: update node openness, cascade to all incident edges
            G_disrupted.nodes[element_id]["openness"] = openness_multiplier
            G_disrupted.nodes[element_id]["risk_score"] = 1.0 - openness_multiplier

            for u, v, data in G_disrupted.edges(element_id, data=True):
                base_cap = data.get("base_capacity_bbl_day", 0)
                other_openness = G_disrupted.nodes[v].get("openness", 1.0)
                data["capacity"] = base_cap * openness_multiplier * other_openness

            for u, v, data in G_disrupted.in_edges(element_id, data=True):
                base_cap = data.get("base_capacity_bbl_day", 0)
                other_openness = G_disrupted.nodes[u].get("openness", 1.0)
                data["capacity"] = base_cap * openness_multiplier * other_openness

        else:
            # Edge disruption: find the edge by its edge_id attribute
            found = False
            for u, v, data in G_disrupted.edges(data=True):
                if data.get("edge_id") == element_id:
                    base_cap = data.get("base_capacity_bbl_day", 0)
                    data["capacity"] = base_cap * openness_multiplier
                    data["openness"] = openness_multiplier
                    found = True
                    break
            if not found:
                # element_id not found in nodes or edge IDs — log and skip
                import warnings
                warnings.warn(
                    f"apply_scenario: element_id '{element_id}' not found in graph. "
                    "Check node/edge IDs in scenario_dict.",
                    stacklevel=2,
                )

    return G_disrupted


# NOTE: An iterative max-flow congestion penalty (effective_cost = base·(1+γ·util²))
# was removed: it was never called, and its purpose — stopping the solver from
# dumping all rerouted volume onto one backup corridor — is already enforced by the
# routing LP's hard Cape-of-Good-Hope corridor cap.


def get_flow_delta(
    baseline: dict,
    disrupted: dict,
) -> dict:
    """
    Compute the difference between baseline and disrupted flow states.

    Args:
        baseline: Output of compute_baseline on the undisrupted graph.
        disrupted: Output of compute_baseline on the disrupted graph.

    Returns:
        Dict with flow_loss, flow_loss_pct, per_refinery_delta.
    """
    flow_loss = baseline["flow_value"] - disrupted["flow_value"]
    flow_loss_pct = flow_loss / max(baseline["flow_value"], 1) * 100

    per_refinery_delta = {}
    for ref_id in baseline.get("flow_per_refinery", {}):
        base_flow = baseline["flow_per_refinery"].get(ref_id, 0)
        dis_flow = disrupted.get("flow_per_refinery", {}).get(ref_id, 0)
        per_refinery_delta[ref_id] = {
            "base_flow": base_flow,
            "disrupted_flow": dis_flow,
            "loss": base_flow - dis_flow,
            "loss_pct": (base_flow - dis_flow) / max(base_flow, 1) * 100,
        }

    return {
        "flow_loss_bbl_day": flow_loss,
        "flow_loss_pct": flow_loss_pct,
        "per_refinery_delta": per_refinery_delta,
    }
