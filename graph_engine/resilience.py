"""
Resilience analytics: N-1/N-2 vulnerability ranking, HHI diversification index,
and Gomory-Hu tree precomputation for instant click-to-fail bottleneck lookup.
"""

import copy
from typing import Optional

import networkx as nx

from graph_engine.build_graph import compute_baseline
from graph_engine.disruption import apply_scenario


def _deliverable_volume(G: nx.DiGraph, params: Optional[dict]) -> float:
    """Grade-aware deliverable volume via the routing solver.

    Uses the same constraint-aware allocator as the recommendation engine, so the
    vulnerability ranking measures the crude that can actually be *delivered to a
    compatible refinery* — not the grade-blind max-flow. This is what keeps Hormuz
    (which every Middle-East SOUR/SWEET barrel to India transits) correctly at the
    top of the ranking instead of a grade-blind chokepoint like Malacca.
    """
    from graph_engine.routing import compute_pareto_routes
    demand = {
        nid: data.get("consumption_rate_bbl_day", 0)
        for nid, data in G.nodes(data=True)
        if data.get("type") == "refinery_out"
    }
    route = compute_pareto_routes(G, demand, params or {}).get("cost_optimal", {})
    return float(route.get("total_volume", 0.0))


def compute_n1_vulnerability(
    G: nx.DiGraph,
    baseline_flow: float,
    degradation_level: float = 0.2,
    params: Optional[dict] = None,
    grade_aware: bool = True,
) -> list[dict]:
    """
    For every chokepoint and source node, apply an 80% degradation and record the
    loss in *deliverable* volume. Returns a ranked list by vulnerability index (desc).

    Vulnerability index = (baseline - disrupted) / baseline.

    By default (``grade_aware=True`` and ``params`` supplied) the loss is measured
    with the grade-aware routing solver, consistent with the recommendation engine.
    Passing ``grade_aware=False`` (or no params) falls back to the grade-blind
    max-flow measure for a fast, dependency-light estimate.

    Args:
        G: Baseline graph (not mutated).
        baseline_flow: Undisrupted deliverable volume (bbl/day) — used only for the
            max-flow fallback; the grade-aware path recomputes its own baseline.
        degradation_level: Openness multiplier applied (0.2 = 80% degradation).
        params: Parameters dict (required for grade-aware measurement).
        grade_aware: Use the routing solver when True and params are available.

    Returns:
        List of dicts sorted by vulnerability_index descending.
    """
    use_grade_aware = grade_aware and params is not None
    base_volume = _deliverable_volume(G, params) if use_grade_aware else baseline_flow

    candidate_nodes = [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("type") in ("chokepoint", "source")
    ]

    ranking = []
    for nid, data in candidate_nodes:
        G_degraded = apply_scenario(G, {nid: degradation_level})
        if use_grade_aware:
            disrupted_volume = _deliverable_volume(G_degraded, params)
            disrupted_cut_set = None
        else:
            disrupted = compute_baseline(G_degraded)
            disrupted_volume = disrupted["flow_value"]
            disrupted_cut_set = disrupted["cut_set"]
        flow_loss = base_volume - disrupted_volume
        vuln_index = flow_loss / max(base_volume, 1)

        ranking.append({
            "node_id": nid,
            "node_name": data.get("name", nid),
            "node_type": data.get("type"),
            "flow_loss_bbl_day": flow_loss,
            "vulnerability_index": vuln_index,
            "disrupted_flow_bbl_day": disrupted_volume,
            "disrupted_cut_set": disrupted_cut_set,
            "measure": "grade_aware_routing" if use_grade_aware else "max_flow",
        })

    ranking.sort(key=lambda x: x["vulnerability_index"], reverse=True)
    return ranking


def compute_n2_vulnerability(
    G: nx.DiGraph,
    baseline_flow: float,
    top_n: int = 3,
    degradation_level: float = 0.2,
    params: Optional[dict] = None,
) -> list[dict]:
    """
    Run paired (N-2) degradations on the top_n most critical elements.
    Finds two-element combinations whose simultaneous failure causes
    disproportionate flow loss (i.e., worse than the sum of individual losses).

    Uses the same grade-aware measure as N-1 when ``params`` is supplied, so the
    two contingency analyses are consistent.

    Args:
        G: Baseline graph.
        baseline_flow: Undisrupted total flow (grade-aware baseline when params given).
        top_n: Consider the top N most vulnerable nodes for pairing.
        degradation_level: Openness multiplier for each degraded node.
        params: Parameters dict (enables grade-aware measurement).

    Returns:
        List of dicts with pair, combined_flow_loss, synergy_factor
        (combined_loss / sum_of_individual_losses), sorted by synergy_factor desc.
    """
    use_grade_aware = params is not None
    base_volume = _deliverable_volume(G, params) if use_grade_aware else baseline_flow

    # First get N-1 ranking to find candidates (same measure as the pairing below)
    n1 = compute_n1_vulnerability(G, base_volume, degradation_level, params=params)
    candidates = [r["node_id"] for r in n1[:top_n]]

    results = []
    seen_pairs = set()
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            pair_key = tuple(sorted([a, b]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            G_degraded = apply_scenario(G, {a: degradation_level, b: degradation_level})
            disrupted_volume = (
                _deliverable_volume(G_degraded, params) if use_grade_aware
                else compute_baseline(G_degraded)["flow_value"]
            )
            combined_loss = base_volume - disrupted_volume

            # Individual losses
            a_loss = next(r["flow_loss_bbl_day"] for r in n1 if r["node_id"] == a)
            b_loss = next(r["flow_loss_bbl_day"] for r in n1 if r["node_id"] == b)
            sum_individual = a_loss + b_loss

            synergy = combined_loss / max(sum_individual, 1)

            results.append({
                "pair": [a, b],
                "combined_flow_loss_bbl_day": combined_loss,
                "sum_individual_losses_bbl_day": sum_individual,
                "synergy_factor": synergy,
                "_note": "synergy > 1.0 means the combination is worse than the sum of parts",
            })

    results.sort(key=lambda x: x["synergy_factor"], reverse=True)
    return results


def compute_hhi(G: nx.DiGraph, flow_dict: dict) -> dict:
    """
    Compute the Herfindahl-Hirschman Index over source-node flow shares.

    HHI = sum of squared market shares (each source as fraction of total flow).
    Range [0, 1]: higher = more concentrated, lower = more diversified.
    HHI > 0.25 is considered highly concentrated (analogous to DoJ market power standard).

    Args:
        G: Graph (used to identify source nodes).
        flow_dict: Flow assignment from max-flow solution.

    Returns:
        Dict with hhi_value, source_shares, interpretation.
    """
    source_flows: dict[str, float] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") == "source":
            outflow = sum(flow_dict.get(nid, {}).values())
            source_flows[nid] = outflow

    total_flow = sum(source_flows.values())
    if total_flow == 0:
        return {
            "hhi_value": 1.0,
            "source_shares": {},
            "interpretation": "No flow — HHI undefined, set to maximum (1.0)",
        }

    source_shares = {nid: flow / total_flow for nid, flow in source_flows.items()}
    hhi = sum(share ** 2 for share in source_shares.values())

    if hhi > 0.25:
        interpretation = "HIGHLY_CONCENTRATED — single source dominates, high vulnerability"
    elif hhi > 0.15:
        interpretation = "MODERATELY_CONCENTRATED — meaningful but not extreme dependency"
    else:
        interpretation = "DIVERSIFIED — no single source dominates"

    return {
        "hhi_value": round(hhi, 4),
        "source_shares": {nid: round(s, 4) for nid, s in source_shares.items()},
        "interpretation": interpretation,
    }


def compute_composite_resilience_score(
    baseline_flow: float,
    disrupted_flow: float,
    days_to_recovery: Optional[float],
    n_alternative_corridors: int,
) -> dict:
    """
    Compute a single composite resilience score for a given disruption.

    Components:
    - Flow retention: fraction of baseline flow maintained
    - Recovery speed: penalizes longer recovery times
    - Corridor diversity: rewards availability of multiple viable corridors

    Args:
        baseline_flow: Undisrupted total flow (bbl/day).
        disrupted_flow: Flow under disruption (bbl/day).
        days_to_recovery: Estimated days to restore full flow (None if unknown).
        n_alternative_corridors: Number of viable alternative routes remaining.

    Returns:
        Dict with score [0,1], component breakdowns, and interpretation.
    """
    flow_retention = disrupted_flow / max(baseline_flow, 1)

    if days_to_recovery is None:
        recovery_score = 0.5  # unknown — neutral
    else:
        recovery_score = max(0.0, 1.0 - days_to_recovery / 30.0)  # 30-day horizon

    corridor_score = min(1.0, n_alternative_corridors / 3.0)  # 3+ corridors = max score

    # Weighted composite: flow retention is most important
    composite = 0.5 * flow_retention + 0.3 * recovery_score + 0.2 * corridor_score

    return {
        "composite_score": round(composite, 3),
        "flow_retention": round(flow_retention, 3),
        "recovery_score": round(recovery_score, 3),
        "corridor_diversity_score": round(corridor_score, 3),
        "n_alternative_corridors": n_alternative_corridors,
        "days_to_recovery": days_to_recovery,
    }


# NOTE: A Gomory-Hu "instant bottleneck lookup" was previously precomputed here.
# It was removed deliberately: it is built on the grade-blind undirected capacity
# graph, so its answers disagree with the authoritative grade/transit-aware solver
# (deliverable_state); it optimizes a non-problem (a full solve on this ~28-node
# graph is ~30 ms); and it is stale the moment a disruption changes a capacity.
# The structural-bottleneck insight it was meant to provide is delivered correctly
# by compute_n1_vulnerability above.
