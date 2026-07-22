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
            "Combining both closures exhausts the network's remaining slack, the Cape of Good "
            "Hope bypass and the diversified sources, and drives the reserve drawdown "
            "non-linearly."
        ),
        "scenario_dict": {"chk_hormuz": 0.0, "chk_bab": 0.0},
        "affected_element": "chk_hormuz",  # representative element for display
    },
}


def apply_scenario(
    G: nx.DiGraph,
    scenario_dict: dict[str, float],
    mode: str = "layer",
) -> nx.DiGraph:
    """
    Apply a disruption scenario to the graph and return a new graph.
    NEVER mutates the original graph.

    Args:
        G: The baseline (or current) graph.
        scenario_dict: Dict mapping graph_element_id -> target openness in [0, 1].
                       Can reference node IDs or edge IDs via 'edge_id'.
        mode: "layer" (default) applies the target as a ceiling, so stacking a
              second disruption can never make an element MORE open than it
              already is — applying "Hormuz partial" after "Hormuz full" must not
              reopen the strait. "set" assigns the value outright, which is what
              a manual openness slider means.

    Returns:
        A deep copy of G with adjusted capacities. IDs that matched nothing are
        recorded on the returned graph as ``graph["unresolved_scenario_elements"]``
        so a caller can reject the request rather than silently disrupting less
        than it asked for. The original G is unchanged.
    """
    G_disrupted = copy.deepcopy(G)
    unresolved: list[str] = []

    for element_id, target in scenario_dict.items():
        target = max(0.0, min(1.0, float(target)))

        if element_id in G_disrupted.nodes:
            # A scenario is a known state, not a rumour: it sets structural
            # openness and is left alone by risk decay.
            current = float(G_disrupted.nodes[element_id].get("structural_openness", 1.0))
            G_disrupted.nodes[element_id]["structural_openness"] = (
                min(current, target) if mode == "layer" else target
            )

        else:
            # Edge disruption: find the edge by its edge_id attribute
            found = False
            for u, v, data in G_disrupted.edges(data=True):
                if data.get("edge_id") == element_id:
                    current = float(data.get("structural_openness", 1.0))
                    data["structural_openness"] = min(current, target) if mode == "layer" else target
                    found = True
                    break
            if not found:
                unresolved.append(element_id)

    from graph_engine.build_graph import refresh_openness
    refresh_openness(G_disrupted)
    G_disrupted.graph["unresolved_scenario_elements"] = unresolved
    return G_disrupted


# NOTE: An iterative max-flow congestion penalty (effective_cost = base·(1+γ·util²))
# was removed: it was never called, and its purpose — stopping the solver from
# dumping all rerouted volume onto one backup corridor — is already enforced by the
# routing LP's hard Cape-of-Good-Hope corridor cap.

