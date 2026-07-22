import json
import pytest
from pathlib import Path
from datetime import datetime, timezone
import networkx as nx

from graph_engine.build_graph import load_graph, compute_baseline, apply_event_to_graph
from graph_engine.disruption import apply_scenario
from graph_engine.resilience import compute_n1_vulnerability, compute_hhi
from graph_engine.economic_model import compute_cascade
from graph_engine.routing import compute_pareto_routes, solve_min_cost_flow
from agents.schema import Event

DATA_DIR = Path(__file__).parent.parent / "data"

def test_load_graph_and_baseline():
    """Verify that graph loads and baseline flow matches expected refinery capacity."""
    G, nodes, edges = load_graph(DATA_DIR)
    
    # Every node in the data file must survive schema validation and reach the graph.
    raw = json.loads((DATA_DIR / "nodes.json").read_text(encoding="utf-8"))
    assert len(nodes) == len(raw)
    assert set(nodes) == {n["id"] for n in raw}
    
    # Compute baseline flow
    baseline = compute_baseline(G)
    flow_val = baseline["flow_value"]
    
    # Refining throughput is the binding constraint at baseline: supply exceeds it,
    # so max-flow should land exactly on total modelled demand.
    expected = sum(n.consumption_rate_bbl_day or 0
                   for n in nodes.values() if n.type == "refinery_out")
    assert flow_val == expected

    # And every refinery must be fully supplied, not just the total.
    assert all(v == pytest.approx(1.0) for v in baseline["fulfillment"].values())

def test_hormuz_in_min_cut():
    """
    Sanity check: Strait of Hormuz must be in the minimum cut set because
    Middle Eastern imports (Iraq, Saudi, UAE, Kuwait) constitute ~2.55 Mb/d flow
    passing through Hormuz.
    """
    G, nodes, edges = load_graph(DATA_DIR)
    assert flow_has_hormuz_dependence(G)

def flow_has_hormuz_dependence(G):
    """Closing the strait India routes most of its crude through must cost it
    real volume. Measured against this graph's own baseline, so the check stays
    valid as the network changes."""
    baseline_flow = compute_baseline(G)["flow_value"]
    disrupted_flow = compute_baseline(apply_scenario(G, {"chk_hormuz": 0.0}))["flow_value"]
    return disrupted_flow < baseline_flow * 0.9

def test_red_sea_bypass_model():
    """
    Verify Red Sea/Bab-el-Mandeb closure forces traffic to bypass via Cape of Good Hope,
    and that flow is maintained if Cape edge has capacity.
    """
    G, nodes, edges = load_graph(DATA_DIR)
    
    # Disruption: Red Sea closed
    G_disrupted = apply_scenario(G, {"chk_bab": 0.0})
    res = compute_baseline(G_disrupted)
    
    # If Bab is closed, westbound Russia Urals (750k) and Nigeria (200k) routes must reroute via Cape of Good Hope.
    # Total Cape capacity is 1,000,000 bbl/day. So flow should route through Cape.
    # Let's confirm that flow is maintained (close to baseline) but costs are higher in routing
    # (cost checking is in routing tests).
    assert res["flow_value"] > 2000000

def test_national_supply_gap_does_not_move_the_crude_benchmark():
    """An Indian supply gap alone must NOT be converted into a Brent move.

    An earlier form of this assertion expected the opposite: a 10% national gap producing
    +33.33% crude, from national_gap_pct / 0.30. That was a misuse of the cited
    elasticity and it drove the headline number roughly 20x too high — a Hormuz
    closure reported "+54.1% crude, petrol Rs 141.92/L", and a correlated crisis
    pinned at the +75% guardrail.

    The error is one of kind, not just scale. A transit disruption strands
    India's cargoes; it does not destroy barrels. The crude still reaches the
    world market, so the benchmark barely moves. What moves is what India pays
    to land a replacement cargo — captured exactly by the landed-cost channel.
    """
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    cascade = compute_cascade(500_000, 5_000_000, 15, params)

    # The physical shortfall is still reported in full...
    assert cascade["shortfall_pct"] == 10.0
    assert cascade["national_shortfall_lower_bound_pct"] == 10.0
    # ...it is simply not laundered into a global price move it cannot cause.
    assert cascade["crude_price_change_pct"] == 0.0
    assert cascade["national_gap_price_treatment"].startswith("not_converted")


def test_global_supply_loss_reproduces_eia_worked_example():
    """The elasticity must be applied the way its source applies it.

    EIA's July 2023 STEO Perspectives article (the cited origin of the
    -0.24/-0.36 range, centred -0.30) works a hurricane scenario: ~1.5 Mb/d of
    Gulf of Mexico production lost — about 1.5% of world supply — moving Brent
    roughly $4.00/bbl, i.e. ~5% on an ~$80 benchmark. Feeding that same loss
    through this model must land in the same place.
    """
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    cascade = compute_cascade(
        0.0, 2_574_000, 0, params, market_supply_loss_bbl_day=1_500_000
    )

    assert cascade["market_supply_loss_pct_global"] == pytest.approx(1.47, abs=0.01)
    assert cascade["crude_price_change_pct"] == pytest.approx(4.9, abs=0.1)
    assert cascade["crude_price_driver"] == "global_market_loss"


def test_hormuz_closure_strands_global_supply_net_of_bypass_pipelines():
    """Hormuz is an EGRESS chokepoint and must move the benchmark.

    Removing the (wrong) national-gap channel is only half the fix: a full
    Hormuz closure then reported a 0.0% benchmark move, which inverts the error.
    There is no alternative sea route out of the Persian Gulf, so closing it
    genuinely removes supply from the world market — net of the pipelines that
    reach water outside the strait.
    """
    from graph_engine.economic_model import global_supply_loss_bbl_day

    G, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    bypass = params["hormuz_bypass_pipeline_capacity_bbl_day"]["value"]
    transit = G.nodes["chk_hormuz"]["global_transit_bbl_day"]

    stranded = global_supply_loss_bbl_day(apply_scenario(G, {"chk_hormuz": 0.0}), params)
    assert stranded == pytest.approx(transit - bypass)

    # Transit-only chokepoints reroute cargo; they strand nothing.
    assert global_supply_loss_bbl_day(apply_scenario(G, {"chk_bab": 0.0}), params) == 0.0
    assert global_supply_loss_bbl_day(apply_scenario(G, {"chk_malacca": 0.0}), params) == 0.0


def test_economic_model_does_not_extrapolate_five_refineries_to_india():
    """A local refinery gap must not be presented as a national supply forecast."""
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    cascade = compute_cascade(257_400, 2_574_000, 0, params)

    assert cascade["shortfall_pct"] == 10.0
    assert cascade["national_shortfall_lower_bound_pct"] == 5.15
    assert cascade["modelled_network_share_of_national_consumption"] == 0.5148
    assert cascade["shortfall_scope"] == "modelled five-refinery network"


def test_apply_event_does_not_mutate_baseline_graph():
    """Applying a signal must produce a new state without corrupting baseline."""
    G, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))

    baseline_openness = G.nodes["chk_hormuz"]["openness"]
    baseline_risk = G.nodes["chk_hormuz"]["risk_score"]
    baseline_timestamp = G.nodes["chk_hormuz"]["last_updated"]
    baseline_edge_capacity = G["src_iraq"]["chk_hormuz"]["capacity"]

    event = Event(
        id="evt-immutability",
        source="test",
        timestamp=datetime(2026, 7, 16, tzinfo=timezone.utc),
        entity="Strait of Hormuz",
        event_type="closure",
        severity=0.8,
        confidence=0.9,
        affected_graph_element="chk_hormuz",
        justification="Regression test event",
    )

    G_updated = apply_event_to_graph(G, event, params)

    assert G_updated is not G
    assert G_updated.nodes["chk_hormuz"]["openness"] < baseline_openness
    assert G_updated["src_iraq"]["chk_hormuz"]["capacity"] < baseline_edge_capacity

    assert G.nodes["chk_hormuz"]["openness"] == baseline_openness
    assert G.nodes["chk_hormuz"]["risk_score"] == baseline_risk
    assert G.nodes["chk_hormuz"]["last_updated"] == baseline_timestamp
    assert G["src_iraq"]["chk_hormuz"]["capacity"] == baseline_edge_capacity


def test_risk_transitions_support_decay_reopening_and_edge_events():
    """Closure raises risk, calendar time decays it, and reopening lowers it."""
    G, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    initial_timestamp = datetime(2026, 7, 16, tzinfo=timezone.utc)

    closure = Event(
        id="evt-closure",
        source="test",
        timestamp=initial_timestamp,
        entity="Strait of Hormuz",
        event_type="closure",
        severity=0.8,
        confidence=0.9,
        affected_graph_element="chk_hormuz",
        justification="Confirmed closure",
    )
    closed = apply_event_to_graph(G, closure, params)
    # A CONFIRMED closure is a known operating state, so it lands on structural
    # openness rather than on decaying risk.
    assert closed.nodes["chk_hormuz"]["openness"] == pytest.approx(0.28)
    assert closed.nodes["chk_hormuz"]["structural_openness"] == pytest.approx(0.28)

    # ...and it must still be closed weeks later without any reinforcing signal.
    quiet = apply_event_to_graph(
        closed,
        Event(id="evt-elsewhere", source="test",
              timestamp=datetime(2026, 8, 20, tzinfo=timezone.utc),
              entity="Nigeria", event_type="capacity_reduction", severity=0.2,
              confidence=0.5, affected_graph_element="src_nigeria",
              justification="unrelated minor signal"),
        params,
    )
    assert quiet.nodes["chk_hormuz"]["openness"] == pytest.approx(0.28), (
        "a confirmed closure must not fade just because the news cycle moved on"
    )

    # An UNCONFIRMED report of the same thing is what risk exists for, and it
    # does decay.
    rumour = apply_event_to_graph(
        G,
        Event(id="evt-rumour", source="test", timestamp=initial_timestamp,
              entity="Strait of Hormuz", event_type="closure", severity=0.8,
              confidence=0.4, affected_graph_element="chk_hormuz",
              justification="unverified report"),
        params,
    )
    assert rumour.nodes["chk_hormuz"]["risk_score"] > 0
    assert rumour.nodes["chk_hormuz"]["structural_openness"] == 1.0
    faded = apply_event_to_graph(
        rumour,
        Event(id="evt-elsewhere-2", source="test",
              timestamp=datetime(2026, 8, 20, tzinfo=timezone.utc),
              entity="Nigeria", event_type="capacity_reduction", severity=0.2,
              confidence=0.5, affected_graph_element="src_nigeria",
              justification="unrelated minor signal"),
        params,
    )
    assert faded.nodes["chk_hormuz"]["openness"] > rumour.nodes["chk_hormuz"]["openness"]

    reopening = Event(
        id="evt-reopening",
        source="test",
        timestamp=datetime(2026, 7, 18, tzinfo=timezone.utc),
        entity="Strait of Hormuz",
        event_type="reopening",
        severity=0.9,
        confidence=0.9,
        affected_graph_element="chk_hormuz",
        justification="Verified reopening",
    )
    reopened = apply_event_to_graph(closed, reopening, params)
    # Reopening restores capacity: 0.28 structural + 0.81 of confirmed reopening,
    # capped at fully open.
    assert reopened.nodes["chk_hormuz"]["openness"] > closed.nodes["chk_hormuz"]["openness"]
    assert reopened.nodes["chk_hormuz"]["structural_openness"] == pytest.approx(1.0)

    edge_event = Event(
        id="evt-edge",
        source="test",
        timestamp=initial_timestamp,
        entity="Iraq export lane",
        event_type="closure",
        severity=1.0,
        confidence=1.0,
        affected_graph_element="e_iraq_hormuz",
        justification="Edge-specific outage",
    )
    edge_closed = apply_event_to_graph(G, edge_event, params)
    assert edge_closed["src_iraq"]["chk_hormuz"]["openness"] == 0.0
    assert edge_closed["src_iraq"]["chk_hormuz"]["capacity"] == 0.0


def test_constraint_aware_routing_enforces_grade_and_shared_capacity():
    """Every recommendation must be a feasible, complete procurement path."""
    G, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    demand = {
        node_id: data["consumption_rate_bbl_day"]
        for node_id, data in G.nodes(data=True)
        if data.get("type") == "refinery_out"
    }

    result = compute_pareto_routes(G, demand, params)["cost_optimal"]
    allocations = result["path_allocations"]

    assert result["feasible"] is True
    assert result["fulfillment"]["ref_paradip_out"] == 1.0
    assert allocations
    assert all(
        allocation["grade"] in G.nodes[allocation["refinery_in"]]["grade_compatibility"]
        for allocation in allocations
    )
    assert all(
        allocation["path"][0] == allocation["source_id"]
        and allocation["path"][-1] == allocation["refinery_out"]
        for allocation in allocations
    )

    by_source = {}
    by_chokepoint = {}
    for allocation in allocations:
        volume = allocation["volume_bbl_day"]
        by_source[allocation["source_id"]] = by_source.get(allocation["source_id"], 0) + volume
        for chokepoint in allocation["chokepoints"]:
            by_chokepoint[chokepoint] = by_chokepoint.get(chokepoint, 0) + volume

    assert all(
        volume <= G.nodes[source_id]["capacity_bbl_day"]
        for source_id, volume in by_source.items()
    )
    assert all(
        volume <= G.nodes[chokepoint]["capacity_bbl_day"]
        for chokepoint, volume in by_chokepoint.items()
    )


def _cape_volume(result):
    """Cape is a bypass (not type=="chokepoint"), so detect it by path membership."""
    return sum(
        allocation["volume_bbl_day"]
        for allocation in result["path_allocations"]
        if "chk_cog" in allocation["path"]
    )


def test_cape_policy_cap_binds_when_the_network_has_alternatives():
    """The Cape concentration rule holds whenever respecting it is possible.

    A diversification ceiling is priced, not absolute, so the guarantee is not
    "never exceeded" — it is "never exceeded to save money". With the network
    otherwise intact the solver has alternatives, so the cap must bind exactly.
    """
    G, _, _ = load_graph(DATA_DIR)
    demand = {
        node_id: data["consumption_rate_bbl_day"]
        for node_id, data in G.nodes(data=True)
        if data.get("type") == "refinery_out"
    }
    cap_fraction = 0.05
    result = solve_min_cost_flow(
        G, demand, policy_constraints={"max_cape_fraction_of_total": cap_fraction},
    )

    assert _cape_volume(result) <= sum(demand.values()) * cap_fraction + 1.0
    assert not result["policy_breaches"], (
        "the Cape ceiling was exceeded even though the network could respect it"
    )


def test_cape_policy_cap_is_exceeded_only_to_avoid_a_shortfall_and_is_reported():
    """When the only way to respect the ceiling is to leave refineries short,
    the solver exceeds it — and says by how much.

    A hard ceiling would withhold deliverable crude to honour a self-imposed
    guideline, and the resulting gap would reach the economic model as a physical
    shortfall driving the growth drag and the power-stress flag. A policy choice
    must not be priced as a barrel that does not exist, so the breach is made and
    surfaced.
    """
    G, _, _ = load_graph(DATA_DIR)
    demand = {
        node_id: data["consumption_rate_bbl_day"]
        for node_id, data in G.nodes(data=True)
        if data.get("type") == "refinery_out"
    }
    G_bab_closed = apply_scenario(G, {"chk_bab": 0.0})
    cap_fraction = 0.01
    result = solve_min_cost_flow(
        G_bab_closed, demand,
        policy_constraints={"max_cape_fraction_of_total": cap_fraction},
    )

    ceiling = sum(demand.values()) * cap_fraction
    cape_volume = _cape_volume(result)
    assert cape_volume > ceiling, "expected the ceiling to be breached in this scenario"

    breach = result["policy_breaches"].get("cape_share:chk_cog")
    assert breach is not None, "the Cape ceiling was breached without being reported"
    # The reported breach must be the actual excess, not a flag.
    assert breach == pytest.approx(cape_volume - ceiling, rel=1e-3)
