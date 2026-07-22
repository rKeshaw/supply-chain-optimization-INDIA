"""Phase 1 — one-solver-of-truth invariants.

These lock in the correctness keystone: the number the UI reports as "supply
shortfall" must come from the same grade/transit-aware solver as the routing, and
chokepoint *node* capacities must actually bind (the old max-flow path ignored them).
"""

import json
from pathlib import Path

import pytest

from graph_engine.build_graph import load_graph
from graph_engine.disruption import apply_scenario, DEFAULT_SCENARIOS
from graph_engine.routing import deliverable_state, compute_pareto_routes

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def graph_and_params():
    G, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    return G, params


def test_baseline_deliverable_matches_capacity(graph_and_params):
    G, params = graph_and_params
    d = deliverable_state(G, params)
    expected = sum(data.get("consumption_rate_bbl_day") or 0
                   for _, data in G.nodes(data=True) if data.get("type") == "refinery_out")
    assert d["flow_value"] == expected
    assert d["gap_bbl_day"] == 0


@pytest.mark.parametrize("scenario_id", ["hormuz_full", "red_sea_suspension", "opec_cut"])
def test_shortfall_equals_routing_gap(graph_and_params, scenario_id):
    """The canonical deliverable gap and the cost-route gap must agree exactly —
    no more grade-blind max-flow saying one thing and the LP another."""
    G, params = graph_and_params
    G_dis = apply_scenario(G, DEFAULT_SCENARIOS[scenario_id]["scenario_dict"])

    d = deliverable_state(G_dis, params)
    route = compute_pareto_routes(G_dis, {
        nid: data.get("consumption_rate_bbl_day", 0)
        for nid, data in G_dis.nodes(data=True) if data.get("type") == "refinery_out"
    }, params)["cost_optimal"]

    assert d["flow_value"] == pytest.approx(route["total_volume"], rel=1e-6)


def test_chokepoint_capacity_is_enforced(graph_and_params):
    """Every chokepoint's routed throughput must respect its declared capacity ×
    openness — the property NetworkX max-flow silently violated."""
    G, params = graph_and_params
    d = deliverable_state(G, params)
    for cp, flow in d["transit_flow"].items():
        cap = (G.nodes[cp].get("capacity_bbl_day") or 0) * G.nodes[cp].get("openness", 1.0)
        assert flow <= cap + 1.0, f"{cp} routes {flow:.0f} > capacity {cap:.0f}"


def test_partial_hormuz_binds_at_capacity(graph_and_params):
    """Under a partial Hormuz closure the transit flow must be capped at the reduced
    capacity — direct proof the node limit now bites."""
    G, params = graph_and_params
    G_dis = apply_scenario(G, {"chk_hormuz": 0.4})
    d = deliverable_state(G_dis, params)
    hormuz_cap = (G.nodes["chk_hormuz"]["capacity_bbl_day"]) * 0.4
    assert d["transit_flow"].get("chk_hormuz", 0) <= hormuz_cap + 1.0
