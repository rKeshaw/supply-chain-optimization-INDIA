"""
Resilience analytics: N-1 contingency ranking and the HHI source-concentration index.
"""

import copy
from typing import Optional

import networkx as nx

from graph_engine.build_graph import compute_baseline
from graph_engine.disruption import apply_scenario
from graph_engine.routing import deliverable_state


def _deliverable_volume(G: nx.DiGraph, params: Optional[dict]) -> float:
    """Grade-aware deliverable volume via the routing solver.

    Uses the same constraint-aware allocator as the recommendation engine, so the
    vulnerability ranking measures the crude that can actually be *delivered to a
    compatible refinery* — not the grade-blind max-flow. This is what keeps Hormuz
    (which every Middle-East SOUR/SWEET barrel to India transits) correctly at the
    top of the ranking instead of a grade-blind chokepoint like Malacca.
    """
    from graph_engine.routing import deliverable_state
    return float(deliverable_state(G, params or {}).get("flow_value", 0.0))


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

    # Bypass routes and pipeline-head ports carry real volume and can fail, so
    # they belong in a contingency ranking alongside straits and sources.
    candidate_nodes = [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("type") in ("chokepoint", "bypass", "port", "source")
    ]
    baseline_state = deliverable_state(G, params or {}) if use_grade_aware else None
    flow_share = {}
    if baseline_state:
        total = max(baseline_state["flow_value"], 1.0)
        for bucket in ("transit_flow", "per_source", "per_refinery"):
            for nid, vol in baseline_state.get(bucket, {}).items():
                flow_share[nid] = max(flow_share.get(nid, 0.0), vol / total)

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
            # How much of baseline flow moves through this node. A node whose
            # loss the network can absorb still ranks by how much depends on it,
            # which is what separates the many candidates that lose nothing.
            "baseline_flow_share": round(flow_share.get(nid, 0.0), 4),
        })

    ranking.sort(key=lambda x: (x["vulnerability_index"], x["baseline_flow_share"]), reverse=True)
    return ranking


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


# Structural bottleneck analysis lives in compute_n1_vulnerability above. A
# precomputed Gomory-Hu tree would be faster but is built on the grade-blind
# undirected capacity graph, so it disagrees with the grade and transit aware
# solver, and it goes stale the moment a disruption changes a capacity. A full
# solve on a graph this size takes about thirty milliseconds.
