"""Specification-conformance tests for the routing engine.

WHAT THESE PROVE, PRECISELY
---------------------------
The reference LP below is an independently-written encoding of the documented
constraint set, built from the raw node/edge data with no imports from
routing.py. The tests assert the shipped solver reaches the same optimum.

That is a SPECIFICATION-CONFORMANCE check: it catches the shipped solver
drifting from the model it claims to implement, and it catches a constraint
being dropped, mis-scoped or double-applied. It is a genuinely useful test and
it has caught real drift.

It is NOT an independent-algorithm optimality proof, and this file used to claim
it was. When it was written, production ran a path-based LP over a bounded set
of Yen's k-shortest candidate paths — a heuristic restriction that really could
have been suboptimal — and this arc-based LP was a structurally different way of
solving the same problem. That arc-based formulation was subsequently promoted
to BE the production solver (see routing.py's module note). Since then the two
sides have been the same formulation, so agreement between them cannot certify
optimality: it certifies that two encodings of one model agree. Optimality now
rests on the formulation itself — an arc-based multi-commodity min-cost flow is
a linear program, and GLOP returns its global optimum.

Keeping the reference is still worthwhile, but only if it tracks the real model.
It had already fallen behind on three constraints (bypass and port node
capacity, the SPR sustainable-draw bound, the SPR last-resort penalty) and on
the diversification ceilings becoming soft, which meant the two sides were
quietly solving different problems and agreeing by luck of what happened to
bind. It is realigned below.
"""
import json
from pathlib import Path

import pytest
from ortools.linear_solver import pywraplp

from graph_engine.build_graph import load_graph
from graph_engine.disruption import apply_scenario, DEFAULT_SCENARIOS
from graph_engine.routing import solve_min_cost_flow, deliverable_state

DATA_DIR = Path(__file__).parent.parent / "data"
GRADES = ["SWEET", "SOUR"]
# Restated here rather than imported, for the same independence reason as
# _reference_origin_price. If routing.py retunes these, this file must be
# updated deliberately — that is the point.
SPR_LAST_RESORT_PENALTY = 500.0
POLICY_BREACH_PENALTY = 250.0


@pytest.fixture(scope="module")
def base():
    G, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    demand = {nid: d.get("consumption_rate_bbl_day", 0)
              for nid, d in G.nodes(data=True) if d.get("type") == "refinery_out"}
    return G, params, demand


def _brent_benchmark(params):
    return float(params.get("assumed_brent_benchmark_usd_per_bbl", {}).get("value", 80.0))


def _reference_origin_price(G, params):
    """Independently-derived landed-cost origin prices.

    Deliberately re-derived here from the raw node data rather than imported
    from routing.py: this module's whole purpose is to be an INDEPENDENT check
    on the shipped solver. Importing routing's own helper would make the
    comparison circular and the optimality proof vacuous — if the landed-cost
    term were only added on the shipped side, the two objectives would differ
    and every scenario would fail; if it were shared, a bug in it would cancel
    out on both sides and never be caught.
    """
    benchmark = _brent_benchmark(params)
    scarcity = _reference_scarcity(G, params)
    out = {}
    for nid, d in G.nodes(data=True):
        if d.get("type") == "source":
            out[nid] = benchmark + float(d.get("crude_differential_usd_per_bbl") or 0.0) + scarcity
        elif d.get("type") == "spr":
            out[nid] = benchmark
    return out


def _reference_scarcity(G, params):
    """Scarcity premium on every still-deliverable source, re-derived here from
    raw node data for the same independence reason as _reference_origin_price.

    Supply leaves the world market two ways: a source-side capacity cut, and
    stranding behind an egress chokepoint that has no sea alternative (only
    Hormuz), net of the pipelines reaching water beyond it.
    """
    rate = float(params.get(
        "crude_differential_scarcity_usd_per_pct_global_loss", {}).get("value", 0.0))
    world = float(params.get("global_oil_supply_bbl_day", {}).get("value", 102_000_000))
    bypass = float(params.get(
        "hormuz_bypass_pipeline_capacity_bbl_day", {}).get("value", 0.0))

    lost = 0.0
    for _, d in G.nodes(data=True):
        if d.get("type") == "source":
            lost += float(d.get("capacity_bbl_day") or 0) * (1.0 - float(d.get("openness", 1.0)))
        elif d.get("type") == "chokepoint" and not d.get("has_sea_alternative", True):
            stranded = float(d.get("global_transit_bbl_day") or 0) * (1.0 - float(d.get("openness", 1.0)))
            lost += max(0.0, stranded - bypass)
    return rate * (lost / max(world, 1.0)) * 100.0


def _arc_based_optimum(G, demand, params, big_unmet=1e7):
    """Independent arc-based multi-commodity min-cost flow over LANDED cost."""
    phys = [(u, v, d) for u, v, d in G.edges(data=True)
            if u not in ("super_source", "super_sink") and v not in ("super_source", "super_sink")
            and d.get("capacity", 0) > 0]
    solver = pywraplp.Solver.CreateSolver("GLOP")
    f = {}
    for u, v, d in phys:
        for g in GRADES:
            if d.get("grade") in (None, g):
                f[(u, v, g)] = solver.NumVar(0, d.get("capacity", 0), f"f_{u}_{v}_{g}")
    outE = lambda n: [(u, v, d) for u, v, d in phys if u == n]
    inE = lambda n: [(u, v, d) for u, v, d in phys if v == n]

    for u, v, d in phys:
        terms = [f[(u, v, g)] for g in GRADES if (u, v, g) in f]
        if terms:
            solver.Add(sum(terms) <= d.get("capacity", 0))

    ntype = {n: G.nodes[n].get("type") for n in G.nodes()}
    unmet = {}
    for n in G.nodes():
        if n in ("super_source", "super_sink"):
            continue
        t, data = ntype[n], G.nodes[n]
        if t == "source":
            sg = data.get("grade_compatibility", [None])[0]
            cap = float(data.get("capacity_bbl_day") or 0) * float(data.get("openness", 1.0))
            for g in GRADES:
                inflow = sum(f[(u, v, g)] for u, v, dd in inE(n) if (u, v, g) in f)
                outflow = sum(f[(u, v, g)] for u, v, dd in outE(n) if (u, v, g) in f)
                if g == sg:
                    solver.Add(outflow - inflow <= cap)
                    solver.Add(inflow == 0)
                else:
                    solver.Add(outflow == 0)
        elif t == "spr":
            cap = float(data.get("capacity_bbl_day") or 0) * float(data.get("openness", 1.0))
            # Sustainable-draw bound: a cavern cannot offer more per day than it
            # can supply across the planning horizon without emptying inside it.
            horizon = float(params.get("spr_draw_projection_days", {}).get("value", 0) or 0)
            if horizon > 0:
                cap = min(cap, float(data.get("inventory_bbl") or 0.0) / horizon)
            solver.Add(sum(f[(u, v, g)] for u, v, dd in outE(n) for g in GRADES if (u, v, g) in f) <= cap)
            for g in GRADES:
                solver.Add(sum(f[(u, v, g)] for u, v, dd in inE(n) if (u, v, g) in f) == 0)
        elif t == "refinery_out":
            req = float(demand.get(n, 0))
            um = solver.NumVar(0, req, f"unmet_{n}")
            unmet[n] = um
            solver.Add(sum(f[(u, v, g)] for u, v, dd in inE(n) for g in GRADES if (u, v, g) in f) + um == req)
            solver.Add(sum(f[(u, v, g)] for u, v, dd in outE(n) for g in GRADES if (u, v, g) in f) == 0)
        else:
            for g in GRADES:
                inflow = sum(f[(u, v, g)] for u, v, dd in inE(n) if (u, v, g) in f)
                outflow = sum(f[(u, v, g)] for u, v, dd in outE(n) if (u, v, g) in f)
                solver.Add(inflow == outflow)
            if t == "refinery_in":
                accepted = set(data.get("grade_compatibility", []))
                for g in GRADES:
                    if g not in accepted:
                        for u, v, dd in inE(n):
                            if (u, v, g) in f:
                                solver.Add(f[(u, v, g)] == 0)

    # Physical throughput of every transit node — straits, the Cape bypass, and
    # the pipeline-head ports alike. Restricting this to chokepoints left the
    # reference structurally less constrained than production.
    for n in G.nodes():
        if ntype[n] in ("chokepoint", "bypass", "port"):
            cap = float(G.nodes[n].get("capacity_bbl_day") or 0) * float(G.nodes[n].get("openness", 1.0))
            terms = [f[(u, v, g)] for u, v, dd in inE(n) for g in GRADES if (u, v, g) in f]
            if terms:
                solver.Add(sum(terms) <= cap)

    # Diversification ceilings are POLICY: soft, with the breach priced between
    # the cost spread and the SPR last-resort penalty. Encoding them as hard
    # constraints here would make the reference refuse allocations production
    # legitimately makes.
    breaches = []

    def soft_cap(terms, ceiling):
        if not terms:
            return
        slack = solver.NumVar(0, solver.infinity(), f"breach_{len(breaches)}")
        breaches.append(slack)
        solver.Add(sum(terms) <= max(0.0, ceiling) + slack)

    total_demand = sum(demand.values())

    if "chk_cog" in G:
        soft_cap([f[(u, v, g)] for u, v, dd in inE("chk_cog") for g in GRADES if (u, v, g) in f],
                 0.60 * total_demand)

    for n in sorted(G.nodes()):
        if ntype[n] != "chokepoint":
            continue
        soft_cap([f[(u, v, g)] for u, v, dd in inE(n) for g in GRADES if (u, v, g) in f],
                 0.40 * total_demand)

    # Supplier-group ceiling and sanctions availability, re-derived independently
    # from the node data (see _reference_origin_price on why this module never
    # imports routing.py's own helpers).
    by_group = {}
    for n in sorted(G.nodes()):
        if ntype[n] != "source":
            continue
        grp = G.nodes[n].get("supplier_group") or n
        terms = [f[(u, v, g)] for u, v, dd in outE(n) for g in GRADES if (u, v, g) in f]
        if terms:
            by_group.setdefault(grp, []).extend(terms)
    for _, terms in sorted(by_group.items()):
        soft_cap(terms, 0.35 * total_demand)

    for n in G.nodes():
        if ntype[n] != "source" or not G.nodes[n].get("sanctions_restricted"):
            continue
        for u, v, dd in outE(n):
            for g in GRADES:
                if (u, v, g) in f:
                    solver.Add(f[(u, v, g)] == 0)

    origin_price = _reference_origin_price(G, params)

    def landed(u, v):
        return float(G[u][v].get("cost_per_bbl", 0)) + origin_price.get(u, 0.0)

    obj = solver.Objective()
    for (u, v, g), var in f.items():
        # Reserve barrels are last-resort: priced far above any import so they
        # are drawn only when nothing else can supply a refinery.
        penalty = SPR_LAST_RESORT_PENALTY if ntype.get(u) == "spr" else 0.0
        obj.SetCoefficient(var, landed(u, v) + penalty)
    for um in unmet.values():
        obj.SetCoefficient(um, big_unmet)
    for slack in breaches:
        obj.SetCoefficient(slack, POLICY_BREACH_PENALTY)
    obj.SetMinimization()
    assert solver.Solve() in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)

    total_unmet = sum(um.solution_value() for um in unmet.values())
    total_cost = sum(f[(u, v, g)].solution_value() * landed(u, v) for (u, v, g) in f)
    return total_cost, sum(demand.values()) - total_unmet


def _shipped_cost_optimal(G, demand, params):
    import copy
    from graph_engine.routing import crude_price_by_origin
    G2 = copy.deepcopy(G)
    origin_price = crude_price_by_origin(G2, params)
    for u, _, d in G2.edges(data=True):
        d["weight"] = d.get("cost_per_bbl", 1.0) + origin_price.get(u, 0.0)
    # Same constraint set the reference LP above encodes by hand, and the same
    # SPR pricing the shipped recommendation uses — without that flag this
    # verified a solve configuration the application never actually presents.
    return solve_min_cost_flow(G2, demand, "weight", {
        "max_cape_fraction_of_total": 0.60,
        "max_supplier_group_fraction_of_total": 0.35,
        "max_chokepoint_fraction_of_total": 0.40,
        "allow_sanctions_restricted_sources": False,
        "spr_draw_projection_days": params.get("spr_draw_projection_days", {}).get("value", 90),
    }, apply_spr_last_resort_penalty=True)


@pytest.mark.parametrize("scenario", [
    {}, {"chk_hormuz": 0.4}, DEFAULT_SCENARIOS["hormuz_full"]["scenario_dict"],
    DEFAULT_SCENARIOS["opec_cut"]["scenario_dict"], DEFAULT_SCENARIOS["red_sea_suspension"]["scenario_dict"],
])
def test_cost_route_is_provably_optimal(base, scenario):
    """The shipped cost-optimal route must match the independent arc-based
    min-cost-flow optimum in both cost and delivered volume."""
    G, params, demand = base
    Gd = apply_scenario(G, scenario) if scenario else G
    arc_cost, arc_vol = _arc_based_optimum(Gd, demand, params)
    res = _shipped_cost_optimal(Gd, demand, params)
    assert res["total_volume"] == pytest.approx(arc_vol, rel=1e-4)
    assert res["total_cost"] == pytest.approx(arc_cost, rel=1e-4)


@pytest.mark.parametrize("scenario", [{}, {"chk_hormuz": 0.4},
                                      DEFAULT_SCENARIOS["hormuz_full"]["scenario_dict"]])
def test_output_respects_all_constraints(base, scenario):
    """Grade, source-capacity, edge-capacity, and chokepoint-capacity compliance
    of the actual shipped allocation."""
    G, params, demand = base
    Gd = apply_scenario(G, scenario) if scenario else G
    res = _shipped_cost_optimal(Gd, demand, params)
    allocs = res["path_allocations"]

    for a in allocs:  # grade compliance
        ref_in = a["refinery_out"].replace("_out", "_in")
        assert a["grade"] in set(Gd.nodes[ref_in].get("grade_compatibility", []))
        assert a["grade"] in set(Gd.nodes[a["source_id"]].get("grade_compatibility", []))

    src_flow, cp_flow, edge_flow = {}, {}, {}
    for a in allocs:
        src_flow[a["source_id"]] = src_flow.get(a["source_id"], 0) + a["volume_bbl_day"]
        for cp in a.get("chokepoints", []):
            cp_flow[cp] = cp_flow.get(cp, 0) + a["volume_bbl_day"]
        for u, v in zip(a["path"], a["path"][1:]):
            edge_flow[(u, v)] = edge_flow.get((u, v), 0) + a["volume_bbl_day"]

    for s, fl in src_flow.items():
        assert fl <= (Gd.nodes[s].get("capacity_bbl_day") or 0) * Gd.nodes[s].get("openness", 1.0) + 1.0
    for cp, fl in cp_flow.items():
        assert fl <= (Gd.nodes[cp].get("capacity_bbl_day") or 0) * Gd.nodes[cp].get("openness", 1.0) + 1.0
    for (u, v), fl in edge_flow.items():
        assert fl <= Gd[u][v].get("capacity", 0) + 1.0


@pytest.mark.parametrize("scenario", [
    {}, {"chk_hormuz": 0.4}, DEFAULT_SCENARIOS["hormuz_full"]["scenario_dict"],
    DEFAULT_SCENARIOS["red_sea_suspension"]["scenario_dict"],
    DEFAULT_SCENARIOS["correlated_gulf_crisis"]["scenario_dict"],
])
def test_recommendation_is_reproducible_under_input_permutation(base, scenario):
    """The recommended PLAN — not just its cost — must be reproducible.

    This problem has many genuinely tied optima: several Gulf sources can sit
    at an identical landed cost. Simplex returns whichever optimal vertex its
    pivot sequence reaches first, which depends on the order LP variables were
    created, i.e. on graph insertion order. Before the solver built its input
    in a canonical order, permuting node/edge order produced up to EIGHT
    different source->refinery recommendation tables at a bit-identical total
    cost. The cost was provably optimal every time; the plan the UI presented
    as "the recommendation" was not reproducible, and there was no answer to
    "why buy 620k from Iraq rather than 870k?" beyond insertion order.

    Asserts the full allocation signature is stable across permuted inputs.
    """
    import copy as _copy
    import random as _random
    import networkx as _nx

    G, params, demand = base
    Gd = apply_scenario(G, scenario) if scenario else G

    signatures, costs = set(), set()
    for trial in range(6):
        H = _nx.DiGraph()
        nodes = list(Gd.nodes(data=True))
        _random.Random(100 + trial).shuffle(nodes)
        H.add_nodes_from(nodes)
        edges = list(Gd.edges(data=True))
        _random.Random(trial).shuffle(edges)
        for u, v, d in edges:
            H.add_edge(u, v, **_copy.deepcopy(d))

        res = _shipped_cost_optimal(H, demand, params)
        signatures.add(tuple(sorted(
            (a["source_id"], a["refinery_out"], a["grade"], round(a["volume_bbl_day"]))
            for a in res["path_allocations"]
        )))
        costs.add(round(res["total_cost"], 2))

    assert len(costs) == 1, f"optimal cost itself is unstable: {costs}"
    assert len(signatures) == 1, (
        f"{len(signatures)} different optimal plans at one identical cost — "
        "the recommendation is not reproducible"
    )
