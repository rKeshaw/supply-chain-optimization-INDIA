"""
FastAPI application: REST API for the Energy Supply Chain Resilience system.

Endpoints:
  GET  /api/health                 — liveness check
  GET  /api/graph/state            — current graph state (nodes + edges with openness/risk)
  GET  /api/graph/baseline         — baseline max-flow and min-cut results
  GET  /api/graph/vulnerability     — N-1 vulnerability ranking
  POST /api/scenario/apply         — apply a disruption scenario
  POST /api/signal                 — process raw news signal through full pipeline
  POST /api/replay/run             — run full crisis timeline replay
  GET  /api/replay/status          — current replay position and state
  GET  /api/spr/status             — SPR inventory levels
  GET  /api/economic/cascade       — economic impact for current disruption
  GET  /api/scenarios/list         — list available named scenarios
  POST /api/twin/simulate          — run digital twin for N days
  GET  /api/live/status            — live-sensing loop status + recent live signal log
  POST /api/live/poll-now          — manually trigger one live-sensing poll cycle
  WS   /ws/live                    — push channel: broadcasts a summary after every
                                     processed signal (replay step, custom signal,
                                     scenario apply, or live-sensed event), so the
                                     UI can react without polling.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph_engine.build_graph import load_graph, compute_baseline, get_graph_state_json
from graph_engine.routing import (
    compute_pareto_routes,
    deliverable_state,
    refinery_demand,
    reroute_premium_vs_baseline,
)
from graph_engine.disruption import apply_scenario, DEFAULT_SCENARIOS
from graph_engine.resilience import compute_n1_vulnerability, compute_hhi
from graph_engine.reserve_optimizer import get_spr_status_summary, planned_draw_from_allocations
from graph_engine.digital_twin import build_initial_state, run_digital_twin
from graph_engine.economic_model import (
    compute_cascade,
    compute_backtest_cascade,
    global_supply_loss_bbl_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global application state (loaded once at startup)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
APP_STATE: dict = {}


def _remember_event(event: Optional[dict]) -> None:
    """Append a processed event to the rolling recent_events window (see
    lifespan's APP_STATE.update), capped at 10 — scenario_agent's own prompt
    already keeps only the last 10, so the cap here just bounds memory."""
    if not event:
        return
    history = APP_STATE.setdefault("recent_events", [])
    history.append(event)
    APP_STATE["recent_events"] = history[-10:]


async def broadcast_update(payload: dict) -> None:
    """Push a compact event summary to every connected /ws/live client.

    Carries a summary rather than the full pipeline result. Clients that want
    routing, cascade or brief detail have REST endpoints for them, so this
    channel only needs to say that something changed. Any client that errors has
    already disconnected and is dropped.
    """
    dead = []
    for client in APP_STATE.get("ws_clients", []):
        try:
            await client.send_json(payload)
        except Exception:
            dead.append(client)
    for client in dead:
        APP_STATE.get("ws_clients", []).remove(client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load graph, compute the canonical baseline state, and (if enabled) start
    the background live-sensing loop at startup."""
    logger.info("Loading graph from data/...")
    G, nodes, edges = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    baseline = compute_baseline(G)  # max-flow utility (structural sanity only)
    sim_state = build_initial_state(G)

    # Canonical undisrupted state — the ONE authority (grade + transit + SPR aware LP)
    # that every reported number diffs against.
    from graph_engine.routing import avg_cost_per_bbl as _avg_cost_per_bbl
    baseline_deliverable = deliverable_state(G, params)
    baseline_cost_route = {
        "path_allocations": baseline_deliverable["path_allocations"],
        "total_volume": baseline_deliverable["flow_value"],
    }
    baseline_avg_cost = _avg_cost_per_bbl(baseline_cost_route)
    # Expose to orchestration/twin via the shared params dict (params convention).
    params["_baseline_routing_avg_cost_per_bbl"] = {"value": baseline_avg_cost}
    # The full baseline allocation, not just its average cost: the cost channel
    # has to re-price the baseline mix at CURRENT crude prices to isolate the
    # routing premium from the benchmark move, and that needs the allocations.
    params["_baseline_cost_route"] = {"value": baseline_cost_route}

    # Structural criticality: the share of baseline deliverable flow that depends
    # on each node. Computed once against the pristine baseline topology and
    # attached as a node attribute, so the deep copy inside apply_scenario
    # carries it into every disrupted graph. Both the risk objective in
    # routing.py and the interface read it.
    #
    # Flow share answers how much of the network currently depends on a node,
    # which is the question the risk weighting needs. A measure based on volume
    # lost when a node degrades answers something else and ranks Hormuz low,
    # because the Cape of Good Hope bypass can absorb most of its volume at a
    # cost and time penalty. Hormuz carries roughly 90% of baseline flow against
    # about 3.5% for a typical minor source.
    total_baseline_flow = max(baseline_deliverable["flow_value"], 1.0)
    flow_share_by_node: dict[str, float] = {}
    for cp, flow in baseline_deliverable["transit_flow"].items():
        flow_share_by_node[cp] = flow / total_baseline_flow
    for src, flow in baseline_deliverable["per_source"].items():
        flow_share_by_node[src] = flow / total_baseline_flow
    for ref, flow in baseline_deliverable["per_refinery"].items():
        flow_share_by_node[ref] = flow / total_baseline_flow

    # Covered types carry their real flow share, which may be zero for a source
    # sitting under sanctions. Every other type defaults to 1.0 and takes no
    # discount, because this measure was never meant to cover them: the reserve
    # carries no flow under normal operation, and the Cape bypass volume is not
    # counted as chokepoint transit.
    FLOW_CRITICALITY_COVERED_TYPES = {"source", "chokepoint", "refinery_out"}
    for nid, node_data in G.nodes(data=True):
        if node_data.get("type") in FLOW_CRITICALITY_COVERED_TYPES:
            node_data["flow_criticality"] = min(1.0, flow_share_by_node.get(nid, 0.0))
        else:
            node_data["flow_criticality"] = 1.0

    # Check the solved network against the reference figures in parameters.json,
    # so a drift between the model and its own documented assumptions surfaces at
    # startup rather than in a demo.
    modelled_capacity = sum(
        data.get("consumption_rate_bbl_day") or 0
        for _, data in G.nodes(data=True) if data.get("type") == "refinery_out"
    )
    declared_capacity = params.get("modeled_refinery_capacity_bbl_day", {}).get("value")
    if declared_capacity and abs(modelled_capacity - declared_capacity) > 1:
        logger.warning(
            "Modelled refining capacity is %s bbl/day but parameters.json declares %s. "
            "Update modeled_refinery_capacity_bbl_day.", f"{modelled_capacity:,.0f}", f"{declared_capacity:,.0f}",
        )
    hormuz_share = baseline_deliverable["transit_flow"].get("chk_hormuz", 0.0) / max(baseline_deliverable["flow_value"], 1.0)
    declared_share = params.get("hormuz_india_exposure_pct", {}).get("value")
    if declared_share:
        logger.info(
            "Hormuz share of baseline supply: %.1f%% (reference figure %.1f%%).",
            hormuz_share * 100, declared_share * 100,
        )
        if abs(hormuz_share - declared_share) > 0.15:
            logger.warning(
                "Hormuz exposure differs from the reference by more than 15 points. "
                "Either the routing policy or hormuz_india_exposure_pct needs revisiting."
            )

    APP_STATE.update({
        "G_baseline": G,
        "G_current": G,
        "nodes": nodes,
        "edges": edges,
        "params": params,
        "baseline": baseline,
        "baseline_deliverable": baseline_deliverable,
        "baseline_cost_route": baseline_cost_route,
        "baseline_avg_cost_per_bbl": baseline_avg_cost,
        "sim_state": sim_state,
        "replay_index": 0,
        "replay_log": [],
        # Rolling window of recent processed events (oldest first, capped at
        # 10), fed to scenario_agent so it can reason about a developing
        # pattern rather than only ever the single latest signal.
        "recent_events": [],
        "live_signal_log": [],
        "live_last_poll_at": None,
        "current_scenario": None,
        "scenario_result": None,
        "last_brief": None,
        "n1_ranking": None,
        "ws_clients": [],
        "broadcast_fn": broadcast_update,
        "live_task": None,
        "live_stop_event": None,
        "live_toggle_lock": asyncio.Lock(),
        # Guards concurrent writes to G_current, scenario_result, and n1_ranking.
        "state_lock": asyncio.Lock(),
    })

    logger.info(
        f"Graph loaded: {len(nodes)} nodes, {len(edges)} edges. "
        f"Baseline flow: {baseline['flow_value']:,.0f} bbl/day."
    )

    from agents.live_sensing_scheduler import LIVE_INGESTION_ENABLED, start_live_loop, stop_live_loop
    if LIVE_INGESTION_ENABLED:
        await start_live_loop(APP_STATE)
        logger.info("Live ingestion ENABLED at startup — background sensing loop started.")
    else:
        logger.info("Live ingestion disabled at startup — replay-only mode. Toggle via POST /api/live/enable.")

    yield

    await stop_live_loop(APP_STATE)


app = FastAPI(
    title="Energy Supply Chain Resilience API",
    description="AI-driven crude oil supply chain resilience system for India",
    version="1.0.0",
    lifespan=lifespan,
)

# The interface is served from this origin. The wildcard allows the API to be
# exercised from a separate dev server or from /docs on another port.
ALLOWED_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    scenario_id: Optional[str] = None  # use named scenario if provided
    custom: Optional[dict[str, float]] = None  # custom {node_id: openness} dict


class SignalRequest(BaseModel):
    text: str
    source: Optional[str] = None
    timestamp: Optional[str] = None  # ISO 8601
    replay_mode: bool = False


class SimulateRequest(BaseModel):
    scenario_id: Optional[str] = None
    custom_scenario: Optional[dict[str, float]] = None
    use_current_graph: bool = False
    horizon_days: int = 60
    enable_live_weather: bool = False  # opt-in live marine-weather overlay
    compare_no_reroute: bool = False  # also run the "no adaptive rerouting" counterfactual


class NLOpsQuery(BaseModel):
    query: str


def _scenario_from_current_graph() -> dict[str, float]:
    """Express the live graph state as a twin-compatible scenario.

    The twin intentionally begins from the immutable baseline so it can seed
    pre-disruption cargoes correctly.  This adapter carries the live node and
    explicitly disrupted-edge availability into that baseline simulation.
    Node effects are represented once at the node level; their incident edge
    capacity changes are therefore not copied a second time.
    """
    G_baseline = APP_STATE["G_baseline"]
    G_current = APP_STATE.get("G_current", G_baseline)
    scenario: dict[str, float] = {}

    for node_id, current_data in G_current.nodes(data=True):
        baseline_openness = G_baseline.nodes[node_id].get("openness", 1.0)
        current_openness = current_data.get("openness", 1.0)
        if abs(current_openness - baseline_openness) > 1e-6:
            scenario[node_id] = current_openness

    for u, v, current_data in G_current.edges(data=True):
        edge_id = current_data.get("edge_id")
        if not edge_id or not G_baseline.has_edge(u, v):
            continue
        baseline_openness = G_baseline[u][v].get("openness", 1.0)
        current_openness = current_data.get("openness", 1.0)
        if abs(current_openness - baseline_openness) > 1e-6:
            scenario[edge_id] = current_openness

    return scenario


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. API is running at /docs"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "graph_loaded": "G_baseline" in APP_STATE,
        "baseline_flow_bbl_day": APP_STATE.get("baseline", {}).get("flow_value"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Live push channel — the Command Center connects here instead of only
    polling on user action. See broadcast_update() for the payload shape."""
    await websocket.accept()
    APP_STATE.setdefault("ws_clients", []).append(websocket)
    try:
        while True:
            # Nothing expected from the client; just keep the connection open
            # and detect disconnects via the exception this raises.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients = APP_STATE.get("ws_clients", [])
        if websocket in clients:
            clients.remove(websocket)


@app.get("/api/provider/health")
async def provider_health():
    """Redacted provider health status for the LLM balancer."""
    try:
        from api.settings import get_provider_health
        return get_provider_health()
    except ImportError:
        raise HTTPException(status_code=500, detail="Settings module not found")


@app.get("/api/graph/state")
async def graph_state():
    """Current graph state with openness and risk scores for all nodes and edges.

    Edge flows come from the grade and transit aware solve, so the map shows the
    same allocation as the recommendations.
    """
    G = APP_STATE.get("G_current", APP_STATE.get("G_baseline"))
    flow = deliverable_state(G, APP_STATE.get("params", {}))["flow_dict"]
    return get_graph_state_json(G, flow)


@app.get("/api/graph/baseline")
async def graph_baseline():
    """Undisrupted deliverable flow and per-refinery fulfilment.

    ``flow_value_bbl_day`` is the figure every other number is measured against.
    ``structural_min_cut`` comes from the max-flow computation and serves as a
    topology check.
    """
    d = APP_STATE["baseline_deliverable"]
    mf = APP_STATE["baseline"]
    return {
        "flow_value_bbl_day": d["flow_value"],
        "fulfillment": d["fulfillment"],
        "flow_per_refinery": d["per_refinery"],
        "per_source": d["per_source"],
        "transit_flow": d["transit_flow"],
        "structural_min_cut": mf["cut_set"],
        # Per-edge baseline flow, keyed by edge id. The map styles routes against
        # normal operations, and that reference comes from the server so it
        # survives a page reload.
        "edge_flows": {
            data.get("edge_id"): d["flow_dict"].get(u, {}).get(v, 0.0)
            for u, v, data in APP_STATE["G_baseline"].edges(data=True)
            if data.get("edge_id")
        },
    }


@app.get("/api/graph/vulnerability")
async def graph_vulnerability():
    """N-1 vulnerability ranking of all chokepoints and sources.

    Solved against the live disrupted graph (G_current) rather than the
    pristine baseline, so the ranking answers the operationally relevant
    question: given what is already closed, what else is critical? The result
    is cached and cleared whenever G_current changes.
    """
    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    baseline_flow = APP_STATE["baseline_deliverable"]["flow_value"]
    ranking = APP_STATE.get("n1_ranking")
    if ranking is None:
        ranking = compute_n1_vulnerability(G_current, baseline_flow, params=APP_STATE["params"])
        APP_STATE["n1_ranking"] = ranking

    hhi = compute_hhi(G_current, deliverable_state(G_current, APP_STATE["params"])["flow_dict"])

    return {
        "n1_ranking": ranking,
        "hhi": hhi,
        "baseline_flow_bbl_day": baseline_flow,
    }


@app.get("/api/routes/baseline")
async def routes_baseline():
    """The undisrupted procurement plan, which is the reference every disruption
    is measured against.

    Returns the three Pareto routes, cheapest, fastest and lowest risk, with
    per-corridor allocations and plain-language reasoning covering why the cost
    route looks the way it does and what the alternatives cost.
    """
    G = APP_STATE["G_baseline"]
    params = APP_STATE["params"]
    demand = refinery_demand(G)
    pareto = compute_pareto_routes(G, demand, params)

    def summarize(route):
        allocs = route.get("path_allocations", [])
        tv = sum(a["volume_bbl_day"] for a in allocs) or 1
        return {
            "total_volume_bbl_day": round(sum(a["volume_bbl_day"] for a in allocs)),
            "avg_cost_per_bbl": round(sum(a["volume_bbl_day"] * a.get("cost_per_bbl", 0) for a in allocs) / tv, 2),
            # Landed = crude at origin + freight, i.e. what the cost objective
            # actually minimises. Ranking the options on freight alone made the
            # "cheapest" plan read as dearer than the "fastest" one, because the
            # cheap plan buys a discounted grade and pays a little more to ship it.
            "avg_landed_cost_per_bbl": round(
                sum(a["volume_bbl_day"] * a.get("landed_cost_per_bbl", 0) for a in allocs) / tv, 2
            ),
            "avg_transit_days": round(sum(a["volume_bbl_day"] * a.get("transit_time_days", 0) for a in allocs) / tv, 1),
        }

    cost_route = pareto["cost_optimal"]
    # Per-corridor allocation rows for the recommended plan. The inlet and outlet
    # suffixes come from refinery node splitting and are stripped for display.
    import re as _re
    def clean_name(data):
        name = data.get("name", "")
        return _re.sub(r"\s*[—–-]\s*(Inlet|Outlet)\s*$", "", name).strip() or name
    node_name = {nid: clean_name(data) for nid, data in G.nodes(data=True)}
    def transit_names(path):
        return [node_name[p] for p in path if G.nodes[p].get("type") in ("chokepoint", "bypass")]
    rows = sorted(
        [{
            "source": node_name[a["source_id"]],
            "refinery": node_name[a["refinery_out"]],
            "grade": a["grade"],
            "volume_bbl_day": round(a["volume_bbl_day"]),
            "transits": transit_names(a["path"]) or ["direct"],
            "cost_per_bbl": round(a.get("cost_per_bbl", 0), 2),
            "transit_time_days": round(a.get("transit_time_days", 0)),
        } for a in cost_route.get("path_allocations", [])],
        key=lambda r: -r["volume_bbl_day"],
    )

    # Source concentration (top contributors) for the reasoning text.
    by_source = {}
    for a in cost_route.get("path_allocations", []):
        by_source[node_name[a["source_id"]]] = by_source.get(node_name[a["source_id"]], 0) + a["volume_bbl_day"]
    total = sum(by_source.values()) or 1
    top_sources = sorted(by_source.items(), key=lambda kv: -kv[1])[:3]

    cost_s, time_s, risk_s = summarize(cost_route), summarize(pareto["time_optimal"]), summarize(pareto["risk_optimal"])
    hhi = compute_hhi(G, APP_STATE["baseline_deliverable"]["flow_dict"])

    return {
        "recommended": "cost_optimal",
        "allocations": rows,
        # Raw allocations too: the Route Transformation diff needs source/corridor
        # IDs and volumes to aggregate against, which the display rows above
        # (names, formatted transits) cannot provide.
        "path_allocations": cost_route.get("path_allocations", []),
        "summary": {"cost_optimal": cost_s, "time_optimal": time_s, "risk_optimal": risk_s},
        "reasoning": {
            # The optimality guarantee rests on the formulation: an arc-based
            # multi-commodity min-cost flow is a linear program, and its optimum
            # is global for the stated model.
            "why_optimal": (
                "Minimum landed cost (crude at origin plus freight) for supplying every refinery, "
                "subject to crude-grade compatibility (sweet/sour), source export ceilings, per-lane "
                "and transit-node capacities, and the sustainable draw limit on the strategic reserve. "
                "Solved as an arc-based multi-commodity min-cost-flow linear program, so the optimum "
                "is global for the stated model rather than the best of a sampled set of routes. "
                "Diversification ceilings (supplier, chokepoint, Cape) are policy rather than physics "
                "and are priced, not enforced absolutely — the solver will exceed one and report it "
                "rather than leave a refinery short or drain the reserve."
            ),
            "top_sources": [
                {"source": s, "volume_bbl_day": round(v), "share_pct": round(v / total * 100, 1)}
                for s, v in top_sources
            ],
            "concentration_hhi": hhi["hhi_value"],
            "concentration_note": hhi["interpretation"],
            # Stated in LANDED cost, the quantity being optimised. Freight is
            # given alongside it because the two move in opposite directions
            # here and quoting only one of them is what made this panel appear
            # to contradict itself.
            "options": (
                f"Cheapest: ${cost_s['avg_landed_cost_per_bbl']}/bbl landed "
                f"(${cost_s['avg_cost_per_bbl']} of that freight) over {cost_s['avg_transit_days']}d. "
                f"Fastest trims transit to {time_s['avg_transit_days']}d at "
                f"${time_s['avg_landed_cost_per_bbl']}/bbl landed — cheaper freight "
                f"(${time_s['avg_cost_per_bbl']}) but a dearer barrel, which is why it is not the cost pick. "
                f"Lowest-risk lands at ${risk_s['avg_landed_cost_per_bbl']}/bbl, identical to cheapest at "
                f"baseline because no corridor is under threat yet."
            ),
        },
    }


@app.get("/api/routes/current")
async def routes_current():
    """Pareto routing for the graph's current state, which is the baseline on a
    fresh load and the live disrupted state once a signal, scenario or replay
    step has run.

    Shares the {pareto_routes, pareto_comparison} shape with the signal
    pipeline, so the routes panel can populate on page load and after a refresh
    taken part way through a disruption.
    """
    G = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    params = APP_STATE["params"]
    demand = refinery_demand(G)
    pareto_routes = compute_pareto_routes(G, demand, params)
    return {
        "pareto_routes": {
            k: {
                "label": v.get("label"),
                "feasible": v.get("feasible"),
                "total_volume": v.get("total_volume"),
                "fulfillment": v.get("fulfillment"),
                "routing_summary": v.get("routing_summary", []),
                "path_allocations": v.get("path_allocations", []),
                "policy_breaches": v.get("policy_breaches", {}),
            }
            for k, v in pareto_routes.items()
            if k != "pareto_comparison"
        },
        "pareto_comparison": pareto_routes.get("pareto_comparison", {}),
    }


@app.get("/api/scenarios/list")
async def list_scenarios():
    """List all available named disruption scenarios."""
    return {
        "scenarios": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "affected_element": v.get("affected_element"),
            }
            for k, v in DEFAULT_SCENARIOS.items()
        ]
    }


@app.post("/api/scenario/apply")
async def apply_scenario_endpoint(req: ScenarioRequest):
    """
    Apply a disruption scenario to the graph and return the flow impact.

    Named scenarios come from DEFAULT_SCENARIOS; a custom request supplies its
    own element-to-openness mapping.
    """
    if req.scenario_id:
        if req.scenario_id not in DEFAULT_SCENARIOS:
            raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found.")
        scenario_dict = DEFAULT_SCENARIOS[req.scenario_id]["scenario_dict"]
        scenario_name = DEFAULT_SCENARIOS[req.scenario_id]["name"]
    elif req.custom:
        scenario_dict = req.custom
        scenario_name = "Custom Scenario"
    else:
        raise HTTPException(status_code=400, detail="Provide scenario_id or custom dict.")

    G_baseline = APP_STATE["G_baseline"]
    # Scenarios layer onto the live state rather than the pristine baseline, so
    # disrupting a second element leaves the first one disrupted. Replay steps
    # and custom signals accumulate the same way. Reset Twin clears everything
    # through /api/replay/reset.
    G_current = APP_STATE.get("G_current", G_baseline)
    # A named scenario layers onto whatever is already disrupted; a hand-set
    # openness value from the inspector is an outright assignment.
    G_disrupted = apply_scenario(
        G_current, scenario_dict, mode="layer" if req.scenario_id else "set"
    )
    unresolved = G_disrupted.graph.get("unresolved_scenario_elements") or []
    if unresolved:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown graph element(s): {', '.join(sorted(unresolved))}",
        )
    params = APP_STATE["params"]

    # Canonical disrupted state, the three Pareto routes, the policy-critic
    # review and the AI decision brief all come from ONE call so the flow-loss
    # shown, the routing panel and the brief can never disagree — and so a
    # scenario applied from the dropdown or the map gets the same agent review
    # a news signal does (see agents/orchestration.py:evaluate_disruption).
    from agents.orchestration import evaluate_disruption
    evaluation = await asyncio.to_thread(
        evaluate_disruption, G_disrupted, APP_STATE["sim_state"], params,
        label=scenario_name, scenario_dict=scenario_dict,
    )
    pareto_routes = evaluation["pareto_routes"]
    disrupted_cost_route = pareto_routes.get("cost_optimal", {})
    disrupted_flow = disrupted_cost_route.get("total_volume", 0.0)
    cascade = evaluation["cascade"]

    per_ref_disrupted: dict[str, float] = {}
    for a in disrupted_cost_route.get("path_allocations", []):
        per_ref_disrupted[a["refinery_out"]] = per_ref_disrupted.get(a["refinery_out"], 0.0) + a["volume_bbl_day"]

    baseline_d = APP_STATE["baseline_deliverable"]
    baseline_flow = baseline_d["flow_value"]
    flow_loss = baseline_flow - disrupted_flow
    flow_loss_pct = flow_loss / max(baseline_flow, 1) * 100

    per_refinery_delta = {}
    for ref_id, base_flow in baseline_d["per_refinery"].items():
        dis_flow = per_ref_disrupted.get(ref_id, 0.0)
        per_refinery_delta[ref_id] = {
            "base_flow": base_flow,
            "disrupted_flow": dis_flow,
            "loss": base_flow - dis_flow,
            "loss_pct": (base_flow - dis_flow) / max(base_flow, 1) * 100,
        }

    # Persist the disrupted graph state and invalidate the N-1 cache so
    # the vulnerability ranking reflects the updated network topology.
    async with APP_STATE["state_lock"]:
        APP_STATE["G_current"] = G_disrupted
        APP_STATE["current_scenario"] = scenario_dict
        APP_STATE["scenario_result"] = {
            "scenario_name": scenario_name,
            "scenario_dict": scenario_dict,
            "disrupted_flow": disrupted_flow,
            "cascade": cascade,
        }
        APP_STATE["n1_ranking"] = None

    await broadcast_update({
        "kind": "scenario_apply", "origin": "manual",
        "label": scenario_name, "flow_loss_pct": flow_loss_pct,
    })

    payload = {
        "scenario_name": scenario_name,
        "scenario_dict": scenario_dict,
        "baseline_flow_bbl_day": baseline_flow,
        "disrupted_flow_bbl_day": disrupted_flow,
        "flow_loss_bbl_day": flow_loss,
        "flow_loss_pct": flow_loss_pct,
        "per_refinery_delta": per_refinery_delta,
        "economic_cascade": cascade,
        "routing": {
            "pareto_routes": {
                key: {
                    "label": value.get("label"),
                    "feasible": value.get("feasible"),
                    "total_volume": value.get("total_volume"),
                    "fulfillment": value.get("fulfillment"),
                    "path_allocations": value.get("path_allocations", []),
                    "policy_breaches": value.get("policy_breaches", {}),
                }
                for key, value in pareto_routes.items()
                if key != "pareto_comparison"
            },
            "pareto_comparison": pareto_routes.get("pareto_comparison", {}),
        },
        "graph_state": evaluation["graph_state"],
        "spr": evaluation["spr"],
        "policy_check": evaluation["policy_check"],
        "brief": evaluation["brief"],
        "scenario_hypotheses": evaluation["scenario_hypotheses"],
    }
    APP_STATE["last_brief"] = {"kind": "disruption", "res": payload}
    return payload


@app.post("/api/signal")
async def process_signal_endpoint(req: SignalRequest):
    """
    Process raw news text through the full pipeline:
    extraction → graph update → routing → SPR → cascade → critic → brief.
    """
    from agents.orchestration import process_signal

    timestamp_dt = None
    if req.timestamp:
        try:
            timestamp_dt = datetime.fromisoformat(req.timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp: {req.timestamp}")

    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    sim_state = APP_STATE["sim_state"]
    params = APP_STATE["params"]

    result = await asyncio.to_thread(
        process_signal,
        raw_text=req.text,
        G_current=G_current,
        sim_state=sim_state,
        params=params,
        source_override=req.source,
        timestamp_override=timestamp_dt,
        recent_events=APP_STATE.get("recent_events"),
    )

    updated_graph = result.pop("_updated_graph", None)
    if result.get("recompute_triggered") and updated_graph is not None:
        async with APP_STATE["state_lock"]:
            APP_STATE["G_current"] = updated_graph
            APP_STATE["n1_ranking"] = None
    _remember_event(result.get("event"))

    await broadcast_update({
        "kind": "custom_signal", "origin": "custom",
        "label": req.text[:120], "recompute_triggered": result.get("recompute_triggered"),
    })

    if result.get("recompute_triggered"):
        APP_STATE["last_brief"] = {"kind": "pipeline", "res": result}
    return result


@app.post("/api/replay/run")
async def run_replay():
    """
    Advance the replay by one step (one crisis_timeline event).
    Call repeatedly to walk through the 2025-2026 crisis timeline.
    """
    from agents.orchestration import process_signal

    timeline = json.loads((DATA_DIR / "crisis_timeline.json").read_text(encoding="utf-8"))
    idx = APP_STATE.get("replay_index", 0)

    if idx >= len(timeline):
        return {
            "status": "complete",
            "message": "All events in the timeline have been replayed.",
            "total_replayed": len(timeline),
        }

    event_data = timeline[idx]
    APP_STATE["replay_index"] = idx + 1

    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    sim_state = APP_STATE["sim_state"]
    params = APP_STATE["params"]

    timestamp_dt = datetime.fromisoformat(event_data["original_timestamp"])

    # The replay ships the headline and body text through the SAME live
    # extraction the custom-signal endpoint uses — no event_override. The
    # timeline file supplies the article text and its real publication date;
    # severity, confidence, event type and the affected corridor are all
    # decided by the model at replay time.
    #
    # The expected_extraction block still in crisis_timeline.json is no longer
    # an input. It is retained only as the labelled ground truth that
    # tests/test_extraction_accuracy.py scores the model against.
    result = await asyncio.to_thread(
        process_signal,
        raw_text=f"{event_data['headline']}\n\n{event_data['body_excerpt']}",
        G_current=G_current,
        sim_state=sim_state,
        params=params,
        source_override=event_data.get("source"),
        timestamp_override=timestamp_dt,
        recent_events=APP_STATE.get("recent_events"),
    )

    updated_graph = result.pop("_updated_graph", None)
    _remember_event(result.get("event"))

    APP_STATE["replay_log"].append({
        "replay_step": idx + 1,
        "event_id": event_data["id"],
        "headline": event_data["headline"],
        "event_timestamp": event_data["original_timestamp"],
        "result_summary": {
            "recompute_triggered": result.get("recompute_triggered"),
            "reason": result.get("reason", "relevant" if result.get("recompute_triggered") else "below_threshold"),
            "latency_ms": result.get("latency_ms") or result.get("latency", {}).get("total_pipeline_ms"),
            # Archived article text, live model extraction. The timeline
            # supplies only the headline, body and publication date now.
            "ingestion_mode": "live_extraction_replay",
        },
    })

    if result.get("recompute_triggered") and updated_graph is not None:
        async with APP_STATE["state_lock"]:
            APP_STATE["G_current"] = updated_graph
            APP_STATE["n1_ranking"] = None

    await broadcast_update({
        "kind": "replay_step", "origin": "replay",
        "label": event_data["headline"], "recompute_triggered": result.get("recompute_triggered"),
    })

    if result.get("recompute_triggered"):
        APP_STATE["last_brief"] = {"kind": "pipeline", "res": result}

    return {
        "replay_step": idx + 1,
        "total_events": len(timeline),
        "event_processed": {
            "id": event_data["id"],
            "headline": event_data["headline"],
            "original_timestamp": event_data["original_timestamp"],
            "known_market_impact": event_data.get("known_market_impact"),
        },
        "pipeline_result": result,
    }


@app.get("/api/replay/status")
async def replay_status():
    """Current replay position and log."""
    timeline = json.loads((DATA_DIR / "crisis_timeline.json").read_text(encoding="utf-8"))
    return {
        "current_index": APP_STATE.get("replay_index", 0),
        "total_events": len(timeline),
        "replay_log": APP_STATE.get("replay_log", []),
        "complete": APP_STATE.get("replay_index", 0) >= len(timeline),
    }


@app.post("/api/replay/reset")
async def reset_replay():
    """Reset graph state and replay to baseline."""
    async with APP_STATE["state_lock"]:
        APP_STATE["G_current"] = APP_STATE["G_baseline"]
        APP_STATE["n1_ranking"] = None
    APP_STATE["replay_index"] = 0
    APP_STATE["replay_log"] = []
    APP_STATE["recent_events"] = []
    APP_STATE["live_signal_log"] = []
    APP_STATE["current_scenario"] = None
    APP_STATE["scenario_result"] = None
    APP_STATE["last_brief"] = None
    APP_STATE["sim_state"] = build_initial_state(APP_STATE["G_baseline"])
    return {"status": "reset", "message": "Graph and replay reset to baseline."}


@app.get("/api/live/status")
async def live_status():
    """Whether the background live-sensing loop is running, plus its recent log.

    ``enabled`` reflects the loop's runtime state, which the enable and disable
    endpoints can change at any point in a session, rather than the startup
    default.

    Live signals from sanctions, news and weather stay distinguishable from
    curated replay events and manually submitted ones, so the interface can
    label each by where it came from.
    """
    from agents.live_sensing_scheduler import LIVE_POLL_INTERVAL_S, is_running
    return {
        "enabled": is_running(APP_STATE),
        "poll_interval_s": LIVE_POLL_INTERVAL_S,
        "last_poll_at": APP_STATE.get("live_last_poll_at"),
        "live_signal_log": APP_STATE.get("live_signal_log", []),
    }


@app.post("/api/live/enable")
async def enable_live_ingestion():
    """Start the background live-sensing loop (sanctions/news/weather) if it
    is not already running. Idempotent — safe to call repeatedly."""
    from agents.live_sensing_scheduler import start_live_loop
    async with APP_STATE["live_toggle_lock"]:
        await start_live_loop(APP_STATE)
    await broadcast_update({"kind": "live_toggle", "enabled": True})
    return await live_status()


@app.post("/api/live/disable")
async def disable_live_ingestion():
    """Stop the background live-sensing loop if it is running. Idempotent.

    Signals already logged remain in live_signal_log; only new polling stops.
    """
    from agents.live_sensing_scheduler import stop_live_loop
    async with APP_STATE["live_toggle_lock"]:
        await stop_live_loop(APP_STATE)
    await broadcast_update({"kind": "live_toggle", "enabled": False})
    return await live_status()


@app.post("/api/live/poll-now")
async def live_poll_now():
    """Manually trigger one live-sensing poll cycle immediately.

    Available even when the background loop is disabled — useful for a demo
    ("watch it check right now") or for verifying the adapters without waiting
    for the configured interval.
    """
    from agents.live_sensing_scheduler import run_poll_cycle
    summary = await run_poll_cycle(APP_STATE)
    return {
        "summary": summary,
        "live_signal_log": APP_STATE.get("live_signal_log", []),
    }


@app.get("/api/brief/current")
async def brief_current():
    """The most recently produced decision brief, or null if nothing has been
    processed yet. Held server side so a page reload keeps the analysis."""
    return APP_STATE.get("last_brief")


@app.get("/api/spr/status")
async def spr_status():
    """Reserve inventory, the draw the live plan commits to, and how long the
    caverns last at that rate.

    The planned draw comes from the same solve that produces the
    recommendations, so the reserve panel tracks whatever is currently
    disrupted rather than reporting a static stock figure.
    """
    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    d = deliverable_state(G_current, APP_STATE["params"])
    return get_spr_status_summary(
        APP_STATE["sim_state"],
        APP_STATE["params"],
        planned_draw=planned_draw_from_allocations(d["path_allocations"]),
    )


@app.get("/api/economic/cascade")
async def economic_cascade(days_elapsed: int = 0):
    """Economic cascade for the current disruption state."""
    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    params = APP_STATE["params"]

    d = deliverable_state(G_current, params)
    total_demand = d["total_demand"]
    gap = d["gap_bbl_day"]

    # Cost + market channels so a volume-neutral cost shock still registers here too.
    premium = reroute_premium_vs_baseline(
        G_current, params,
        {"path_allocations": d["path_allocations"], "total_volume": d["flow_value"]},
    )
    market_supply_loss = global_supply_loss_bbl_day(G_current, params)
    return compute_cascade(
        gap, total_demand, days_elapsed, params,
        reroute_cost_premium_usd_per_bbl=premium,
        delivered_volume_bbl_day=d["flow_value"],
        market_supply_loss_bbl_day=market_supply_loss,
    )


@app.get("/api/backtest/april2025")
async def backtest_april2025():
    """
    Decompose the April 2025 US-Iran standoff into the part the physical model
    explains and the implied risk premium.

    Brent rose 8% in a single session. The standoff was an escalation rather
    than a supply loss: no production was shut and no egress closed, so no
    barrels left the world market. The benchmark channel therefore prices it at
    zero and the whole observed move resolves as risk premium, which is the
    result this endpoint exists to show.
    """
    params = APP_STATE["params"]

    return compute_backtest_cascade(
        # No barrels left the world market during the standoff.
        market_supply_loss_bbl_day=0.0,
        # India-relevant exposure at the time, carried for shortfall context. It
        # never reaches the benchmark channel.
        actual_gap_bbl_day=308_000,
        baseline_brent_price_usd=78.0,  # approximate Brent before the event
        actual_brent_change_pct=8.0,    # actual observed +8% spike per PS
        days_elapsed=1,
        params=params,
    )


@app.post("/api/twin/simulate")
async def simulate_twin(req: SimulateRequest):
    """
    Run the digital twin for N days under the specified scenario.
    Returns daily snapshots of inventory, flow, SPR, and economic cascade.
    """
    G_baseline = APP_STATE["G_baseline"]
    params = APP_STATE["params"]

    if req.use_current_graph:
        scenario_dict = _scenario_from_current_graph()
    elif req.scenario_id:
        if req.scenario_id not in DEFAULT_SCENARIOS:
            raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found.")
        scenario_dict = DEFAULT_SCENARIOS[req.scenario_id]["scenario_dict"]
    elif req.custom_scenario:
        scenario_dict = req.custom_scenario
    else:
        scenario_dict = {}  # baseline simulation

    horizon = min(req.horizon_days, 90)
    snapshots = run_digital_twin(
        G_baseline=G_baseline,
        scenario_dict=scenario_dict,
        params=params,
        horizon_days=horizon,
        enable_live_weather=req.enable_live_weather,
    )

    response = {
        "horizon_days": horizon,
        "scenario": scenario_dict,
        "snapshots": snapshots,
        "summary": {
            "min_fulfillment_pct": min(s["fulfillment_pct_overall"] for s in snapshots),
            "days_with_gap": sum(1 for s in snapshots if s["gap_bbl_day"] > 0),
            "total_spr_drawn": sum(s["spr_draw_bbl_day"] for s in snapshots),
        },
    }

    if req.compare_no_reroute:
        no_reroute_snapshots = run_digital_twin(
            G_baseline=G_baseline,
            scenario_dict=scenario_dict,
            params=params,
            horizon_days=horizon,
            enable_live_weather=req.enable_live_weather,
            disable_rerouting=True,
        )
        # The ceiling this disrupted network can actually reach, from the same
        # solver that produces the plan — the yardstick for "stabilised".
        from graph_engine.disruption import apply_scenario as _apply
        ceiling_state = deliverable_state(_apply(G_baseline, scenario_dict), params)
        ceiling_pct = ceiling_state["flow_value"] / max(ceiling_state["total_demand"], 1.0) * 100.0
        landed = params.get("assumed_landed_crude_cost_usd_per_bbl", {}).get("value", 86.0)

        response["no_reroute_snapshots"] = no_reroute_snapshots
        response["summary"]["feasible_ceiling_pct"] = round(ceiling_pct, 1)
        response["summary"]["days_to_stabilize_with_reroute"] = _days_to_stabilize(snapshots, ceiling_pct)
        response["summary"]["days_to_stabilize_without_reroute"] = _days_to_stabilize(no_reroute_snapshots, ceiling_pct)
        response["summary"]["total_cost_with_reroute_usd"] = _total_reroute_cost(snapshots, landed)
        response["summary"]["total_cost_without_reroute_usd"] = _total_reroute_cost(no_reroute_snapshots, landed)

    return response


def _days_to_stabilize(snapshots: list[dict], ceiling_pct: float) -> Optional[int]:
    """First day from which supply holds at the network's feasible ceiling for
    the rest of the horizon.

    The ceiling is what the disrupted network can reach, so arriving there marks
    the end of the decline rather than a return to normal, and the interface
    words it that way. A closure that permanently removes a third of supply sits
    near 67%, where a fixed target such as 99% could never be met.

    The condition has to hold to the end of the horizon. Accepting the first day
    that merely touches the ceiling would report a transient dip as the settling
    point.
    """
    if not snapshots:
        return None
    target = max(0.0, ceiling_pct - 1.0)
    settled_from = None
    for s in reversed(snapshots):
        if s["fulfillment_pct_overall"] >= target:
            settled_from = s["day"]
        else:
            break
    return settled_from


def _total_reroute_cost(snapshots: list[dict], landed_cost_per_bbl: float) -> float:
    """Total economic cost of a scenario, combining what rerouting costs with
    what going short costs.

    Pricing the reroute premium alone would make inaction look cheap, since the
    arm that never reroutes pays almost no premium precisely because it fails to
    deliver. Unserved barrels are charged at landed replacement cost.
    """
    premium = sum(s["cascade"].get("daily_reroute_cost_premium_usd", 0.0) for s in snapshots)
    unserved = sum(s.get("gap_bbl_day", 0.0) for s in snapshots) * landed_cost_per_bbl
    return premium + unserved


@app.post("/api/nl-ops")
async def nl_ops_endpoint(req: NLOpsQuery):
    """
    Process a natural language command into a digital twin simulation.
    """
    from agents.nl_ops_agent import parse_nl_command
    
    available_nodes = [
        {"id": nid, "name": data.get("name", nid)} 
        for nid, data in APP_STATE["G_baseline"].nodes(data=True)
    ]
    
    parsed = parse_nl_command(req.query, DEFAULT_SCENARIOS, available_nodes)
    if not parsed:
        raise HTTPException(status_code=400, detail="Failed to parse the natural language query.")
        
    sim_req = SimulateRequest(
        scenario_id=parsed.scenario_id,
        custom_scenario=parsed.custom_scenario,
        horizon_days=parsed.horizon_days,
        use_current_graph=False
    )
    
    result = await simulate_twin(sim_req)
    
    return {
        "parsed_command": parsed.model_dump(),
        "simulation_result": result
    }
