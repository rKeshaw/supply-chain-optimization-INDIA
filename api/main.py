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
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph_engine.build_graph import load_graph, compute_baseline, get_graph_state_json
from graph_engine.routing import compute_pareto_routes, deliverable_state, reroute_cost_premium
from graph_engine.disruption import apply_scenario, DEFAULT_SCENARIOS
from graph_engine.resilience import compute_n1_vulnerability, compute_hhi
from graph_engine.reserve_optimizer import get_spr_status_summary
from graph_engine.digital_twin import build_initial_state, run_digital_twin
from graph_engine.economic_model import compute_cascade, compute_backtest_cascade

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


async def broadcast_update(payload: dict) -> None:
    """Push a compact event summary to every connected /ws/live client.

    Deliberately sends a SUMMARY, not the full pipeline result (routing,
    cascade, brief) — clients that want the details already have the matching
    REST endpoints; this channel's job is just "something changed, go refetch
    or update your feed", not duplicating the whole response payload over the
    socket. Silently drops any client that errors (already disconnected).
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

    # Structural criticality — what fraction of baseline deliverable flow
    # actually depends on this node — computed ONCE against the pristine
    # baseline topology, not recomputed per request. Attached as a node
    # attribute so apply_scenario's deepcopy carries it through to every future
    # disrupted graph automatically; see routing.py's risk-optimal weighting and
    # the frontend's pathRiskScore(), which both blend risk_score with this.
    #
    # An earlier version used resilience.py's N-1 vulnerability_index (flow LOST
    # if this node is degraded) instead. That measures something different and
    # gave the wrong answer for the flagship case: this network has a Cape of
    # Good Hope bypass that can absorb almost all of Hormuz's normal volume at
    # a cost/time penalty, so degrading Hormuz barely reduces DELIVERABLE
    # volume — its vulnerability_index came out at 0.048, near the BOTTOM of
    # the ranking (chk_malacca ranked above it), which halved Hormuz's
    # effective risk instead of amplifying it. Baseline FLOW SHARE avoids that:
    # it directly answers "how much of the network currently depends on this
    # node", which is the question that actually matters here — Hormuz carries
    # ~90% of baseline flow vs. ~3.5% for a typical minor source, exactly
    # matching the real-world intuition (and the PS's own framing) that a
    # single-corridor closure should read as far riskier than an equally-"58%
    # open" but easily-substituted source.
    total_baseline_flow = max(baseline_deliverable["flow_value"], 1.0)
    flow_share_by_node: dict[str, float] = {}
    for cp, flow in baseline_deliverable["transit_flow"].items():
        flow_share_by_node[cp] = flow / total_baseline_flow
    for src, flow in baseline_deliverable["per_source"].items():
        flow_share_by_node[src] = flow / total_baseline_flow
    for ref, flow in baseline_deliverable["per_refinery"].items():
        flow_share_by_node[ref] = flow / total_baseline_flow

    # Covered types get their real (possibly zero, e.g. an unused source under
    # sanctions) flow share. Everything else — SPR (carries zero *normal* flow
    # by design, which is not the same as "not critical"), refinery_in, bypass
    # (chk_cog's reroute volume isn't tracked as "chokepoint" transit flow),
    # super_source/sink — defaults to 1.0 (no discount) so their risk is never
    # silently understated for a type this measure was never meant to cover.
    FLOW_CRITICALITY_COVERED_TYPES = {"source", "chokepoint", "refinery_out"}
    for nid, node_data in G.nodes(data=True):
        if node_data.get("type") in FLOW_CRITICALITY_COVERED_TYPES:
            node_data["flow_criticality"] = min(1.0, flow_share_by_node.get(nid, 0.0))
        else:
            node_data["flow_criticality"] = 1.0

    APP_STATE.update({
        "G_baseline": G,
        "G_current": G,  # mutable current state (replaced on each signal/scenario)
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
        "replay_narrative_clock": None,
        "live_signal_log": [],
        "live_last_poll_at": None,
        "current_scenario": None,
        "scenario_result": None,
        "ws_clients": [],
        "broadcast_fn": broadcast_update,
        "live_task": None,
        "live_stop_event": None,
        "live_toggle_lock": asyncio.Lock(),
    })

    logger.info(
        f"Graph loaded: {len(nodes)} nodes, {len(edges)} edges. "
        f"Baseline flow: {baseline['flow_value']:,.0f} bbl/day."
    )

    # Background live-sensing loop (sanctions registry, news, weather). The env
    # var picks the default at boot — off by default, so the demo stays
    # deterministic — but it can be flipped on or off at runtime via
    # POST /api/live/enable and /api/live/disable (the UI header toggle uses
    # exactly these). See live_sensing_scheduler.start_live_loop/stop_live_loop.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    horizon_days: int = 30
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

    Edge flows shown on the map come from the canonical grade/transit-aware solve,
    not the grade-blind max-flow, so the picture matches the recommendations.
    """
    G = APP_STATE.get("G_current", APP_STATE.get("G_baseline"))
    flow = deliverable_state(G, APP_STATE.get("params", {}))["flow_dict"]
    return get_graph_state_json(G, flow)


@app.get("/api/graph/baseline")
async def graph_baseline():
    """Baseline (undisrupted) deliverable flow and per-refinery fulfillment.

    ``flow_value_bbl_day`` is the canonical LP figure (the single source of truth);
    ``structural_min_cut`` is the max-flow cut set, kept only as a topology sanity aid.
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
    }


@app.get("/api/graph/vulnerability")
async def graph_vulnerability():
    """N-1 vulnerability ranking of all chokepoints and sources."""
    G = APP_STATE["G_baseline"]
    baseline_flow = APP_STATE["baseline_deliverable"]["flow_value"]
    ranking = compute_n1_vulnerability(G, baseline_flow, params=APP_STATE["params"])

    # HHI over the canonical (grade-aware) source allocation, not the max-flow one.
    hhi = compute_hhi(G, APP_STATE["baseline_deliverable"]["flow_dict"])

    return {
        "n1_ranking": ranking,
        "hhi": hhi,
        "baseline_flow_bbl_day": baseline_flow,
    }


@app.get("/api/routes/baseline")
async def routes_baseline():
    """The initial, undisrupted optimal procurement plan — the reference every
    disruption is measured against. Returns the three Pareto routes (cheapest /
    fastest / lowest-risk) with per-corridor allocations and plain-language
    reasoning about why the cost route is what it is and what the alternatives cost.
    """
    G = APP_STATE["G_baseline"]
    params = APP_STATE["params"]
    demand = {nid: data.get("consumption_rate_bbl_day", 0)
              for nid, data in G.nodes(data=True) if data.get("type") == "refinery_out"}
    pareto = compute_pareto_routes(G, demand, params)

    def summarize(route):
        allocs = route.get("path_allocations", [])
        tv = sum(a["volume_bbl_day"] for a in allocs) or 1
        return {
            "total_volume_bbl_day": round(sum(a["volume_bbl_day"] for a in allocs)),
            "avg_cost_per_bbl": round(sum(a["volume_bbl_day"] * a.get("cost_per_bbl", 0) for a in allocs) / tv, 2),
            "avg_transit_days": round(sum(a["volume_bbl_day"] * a.get("transit_time_days", 0) for a in allocs) / tv, 1),
        }

    cost_route = pareto["cost_optimal"]
    # Per-corridor allocation rows for the cost-optimal (recommended) plan.
    # Strip the "<dash> Inlet/Outlet" split-node suffix for clean display (robust to
    # em-dash / en-dash / hyphen variants in the data).
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
        "summary": {"cost_optimal": cost_s, "time_optimal": time_s, "risk_optimal": risk_s},
        "reasoning": {
            "why_optimal": (
                "This is the provably minimum-cost plan that fully supplies all refineries while "
                "respecting every constraint — crude-grade compatibility (sweet/sour), source export "
                "volumes, per-lane and strait (transit) capacities, and the Cape-of-Good-Hope "
                "diversification cap. It is verified to match, to the dollar, an independent "
                "arc-based min-cost-flow optimum."
            ),
            "top_sources": [
                {"source": s, "volume_bbl_day": round(v), "share_pct": round(v / total * 100, 1)}
                for s, v in top_sources
            ],
            "concentration_hhi": hhi["hhi_value"],
            "concentration_note": hhi["interpretation"],
            "options": (
                f"Cheapest: ${cost_s['avg_cost_per_bbl']}/bbl over {cost_s['avg_transit_days']}d. "
                f"Fastest trims transit to {time_s['avg_transit_days']}d but costs ${time_s['avg_cost_per_bbl']}/bbl. "
                f"Lowest-risk costs ${risk_s['avg_cost_per_bbl']}/bbl — identical to cheapest at baseline "
                f"because no corridor is under threat yet; the routes diverge only once a disruption raises risk."
            ),
        },
    }


@app.get("/api/routes/current")
async def routes_current():
    """Pareto routing for whatever the graph's CURRENT state is — baseline on a
    fresh load, or the live disrupted state if a signal/scenario/replay step has
    already run. Same {pareto_routes, pareto_comparison} shape as process_signal's
    "routing" key, so the frontend can populate the Routes tab immediately on page
    load (or after a browser refresh mid-disruption) instead of showing nothing
    until the next event arrives.
    """
    G = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    params = APP_STATE["params"]
    demand = {nid: data.get("consumption_rate_bbl_day", 0)
              for nid, data in G.nodes(data=True) if data.get("type") == "refinery_out"}
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
    Apply a disruption scenario to the graph and return flow impact.
    Named scenarios use DEFAULT_SCENARIOS; custom uses the provided dict.
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
    # Layer the new scenario on top of whatever is already disrupted (G_current),
    # not the pristine baseline — otherwise disrupting a second node silently
    # reverts every previously-disrupted node/edge back to fully open, since each
    # call would rebuild from scratch. G_current already accumulates correctly
    # across replay steps and custom signals; scenario application now matches
    # that same "layer onto the current state" model. Explicit reset remains
    # available via /api/replay/reset ("Reset Twin").
    G_current = APP_STATE.get("G_current", G_baseline)
    G_disrupted = apply_scenario(G_current, scenario_dict)
    params = APP_STATE["params"]

    demand = {
        node_id: data.get("consumption_rate_bbl_day", 0)
        for node_id, data in G_disrupted.nodes(data=True)
        if data.get("type") == "refinery_out"
    }
    total_demand = sum(demand.values())

    # Canonical disrupted state + the three Pareto routes come from the SAME LP solve
    # family, so the flow-loss shown and the routing panel can never disagree.
    pareto_routes = compute_pareto_routes(G_disrupted, demand, params)
    disrupted_cost_route = pareto_routes.get("cost_optimal", {})
    disrupted_flow = disrupted_cost_route.get("total_volume", 0.0)

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

    # Cost channel: how much more the disrupted routing costs vs. baseline routing.
    premium = reroute_cost_premium(APP_STATE.get("baseline_cost_route", {}), disrupted_cost_route)

    # Market channel: source barrels removed from the global market by this scenario
    # (e.g. OPEC+ cut) — moves the crude benchmark even if India reroutes to cover.
    market_supply_loss = 0.0
    for node_id, base_data in G_baseline.nodes(data=True):
        if base_data.get("type") != "source":
            continue
        base_cap = base_data.get("capacity_bbl_day") or 0
        dis_open = G_disrupted.nodes[node_id].get("openness", 1.0)
        market_supply_loss += base_cap * (1.0 - dis_open)

    gap = max(0.0, total_demand - disrupted_flow)
    cascade = compute_cascade(
        gap, total_demand, 0, params,
        reroute_cost_premium_usd_per_bbl=premium,
        delivered_volume_bbl_day=disrupted_flow,
        market_supply_loss_bbl_day=market_supply_loss,
    )

    # Update current graph state
    APP_STATE["G_current"] = G_disrupted
    APP_STATE["current_scenario"] = scenario_dict
    APP_STATE["scenario_result"] = {
        "scenario_name": scenario_name,
        "scenario_dict": scenario_dict,
        "disrupted_flow": disrupted_flow,
        "cascade": cascade,
    }

    await broadcast_update({
        "kind": "scenario_apply", "origin": "manual",
        "label": scenario_name, "flow_loss_pct": flow_loss_pct,
    })

    return {
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
                }
                for key, value in pareto_routes.items()
                if key != "pareto_comparison"
            },
            "pareto_comparison": pareto_routes.get("pareto_comparison", {}),
        },
        "graph_state": get_graph_state_json(
            G_disrupted, pareto_routes.get("cost_optimal", {}).get("flow_dict")
        ),
    }


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

    result = process_signal(
        raw_text=req.text,
        G_current=G_current,
        sim_state=sim_state,
        params=params,
        source_override=req.source,
        timestamp_override=timestamp_dt,
    )

    # Persist the exact state produced by orchestration; rebuilding it from the
    # event would discard risk decay and event-type-specific transitions.
    updated_graph = result.pop("_updated_graph", None)
    if result.get("recompute_triggered") and updated_graph is not None:
        APP_STATE["G_current"] = updated_graph

    await broadcast_update({
        "kind": "custom_signal", "origin": "custom",
        "label": req.text[:120], "recompute_triggered": result.get("recompute_triggered"),
    })

    return result


@app.post("/api/replay/run")
async def run_replay():
    """
    Advance the replay by one step (one crisis_timeline event).
    Call repeatedly to walk through the 2025-2026 crisis timeline.
    """
    from agents.orchestration import process_signal
    from agents.extraction_agent import event_from_curated_timeline

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

    # Process this event through the full pipeline
    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    sim_state = APP_STATE["sim_state"]
    params = APP_STATE["params"]

    timestamp_dt = datetime.fromisoformat(event_data["original_timestamp"])
    curated_event = event_from_curated_timeline(event_data)

    # Compressed "narrative clock" for risk decay ONLY (see apply_event_to_graph's
    # decay_as_of docstring) — the real headline date (timestamp_dt) is still
    # used for display/audit everywhere else. The 12 curated events span real
    # calendar gaps up to 224 days; replaying those verbatim decays each
    # corridor's risk almost to zero between events, making one continuous,
    # still-live crisis look like a series of disconnected incidents that keep
    # "resolving" themselves before the next one starts, instead of building on
    # each other the way the PS's own narrative (and the curated data) intends.
    # REPLAY_NARRATIVE_STEP_DAYS keeps the whole 12-step arc within
    # schema.py's decay cap (30 days), so even the last event still carries the
    # accumulated weight of everything before it.
    REPLAY_NARRATIVE_STEP_DAYS = 2
    if idx == 0 or "replay_narrative_clock" not in APP_STATE or APP_STATE["replay_narrative_clock"] is None:
        narrative_clock = timestamp_dt
    else:
        narrative_clock = APP_STATE["replay_narrative_clock"] + timedelta(days=REPLAY_NARRATIVE_STEP_DAYS)
    APP_STATE["replay_narrative_clock"] = narrative_clock

    result = process_signal(
        raw_text=f"{event_data['headline']}\n\n{event_data['body_excerpt']}",
        G_current=G_current,
        sim_state=sim_state,
        params=params,
        source_override=event_data.get("source"),
        timestamp_override=timestamp_dt,
        event_override=curated_event,
        decay_as_of=narrative_clock,
    )

    updated_graph = result.pop("_updated_graph", None)

    # Log replay step
    APP_STATE["replay_log"].append({
        "replay_step": idx + 1,
        "event_id": event_data["id"],
        "headline": event_data["headline"],
        "result_summary": {
            "recompute_triggered": result.get("recompute_triggered"),
            # "unrelated" vs "below_threshold" vs "relevant" — the curated
            # timeline deliberately includes two unrelated test cases (a
            # cricket headline, a chip-maker announcement) to demonstrate the
            # extraction agent correctly ignores non-energy news; this lets
            # the UI label those as filtered noise instead of listing them
            # identically alongside genuine supply chain signals.
            "reason": result.get("reason", "relevant" if result.get("recompute_triggered") else "below_threshold"),
            "latency_ms": result.get("latency_ms") or result.get("latency", {}).get("total_pipeline_ms"),
            "ingestion_mode": "curated_replay",
        },
    })

    if result.get("recompute_triggered") and updated_graph is not None:
        APP_STATE["G_current"] = updated_graph

    await broadcast_update({
        "kind": "replay_step", "origin": "replay",
        "label": event_data["headline"], "recompute_triggered": result.get("recompute_triggered"),
    })

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
    APP_STATE["G_current"] = APP_STATE["G_baseline"]
    APP_STATE["replay_index"] = 0
    APP_STATE["replay_log"] = []
    APP_STATE["replay_narrative_clock"] = None
    APP_STATE["live_signal_log"] = []
    APP_STATE["current_scenario"] = None
    APP_STATE["scenario_result"] = None
    APP_STATE["sim_state"] = build_initial_state(APP_STATE["G_baseline"])
    return {"status": "reset", "message": "Graph and replay reset to baseline."}


@app.get("/api/live/status")
async def live_status():
    """Whether the background live-sensing loop is currently running, and its
    recent log.

    ``enabled`` reflects the loop's actual runtime state (it can be flipped by
    POST /api/live/enable or /api/live/disable at any point in the session),
    not just the LIVE_INGESTION_ENABLED startup default.

    Distinguishes LIVE-detected signals (sanctions/news/weather) from the
    curated replay and manually-submitted custom signals, so the UI can label
    each one honestly instead of implying everything came from a live feed.
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


@app.get("/api/spr/status")
async def spr_status():
    """Current SPR inventory levels and draw status."""
    return get_spr_status_summary(APP_STATE["sim_state"], APP_STATE["params"])


@app.get("/api/economic/cascade")
async def economic_cascade(days_elapsed: int = 0):
    """Economic cascade for the current disruption state."""
    G_current = APP_STATE.get("G_current", APP_STATE["G_baseline"])
    params = APP_STATE["params"]

    d = deliverable_state(G_current, params)
    total_demand = d["total_demand"]
    gap = d["gap_bbl_day"]

    # Cost + market channels so a volume-neutral cost shock still registers here too.
    premium = reroute_cost_premium(
        APP_STATE.get("baseline_cost_route", {}),
        {"path_allocations": d["path_allocations"], "total_volume": d["flow_value"]},
    )
    market_supply_loss = sum(
        (data.get("capacity_bbl_day") or 0) * (1.0 - data.get("openness", 1.0))
        for _, data in G_current.nodes(data=True)
        if data.get("type") == "source"
    )
    return compute_cascade(
        gap, total_demand, days_elapsed, params,
        reroute_cost_premium_usd_per_bbl=premium,
        delivered_volume_bbl_day=d["flow_value"],
        market_supply_loss_bbl_day=market_supply_loss,
    )


@app.get("/api/backtest/april2025")
async def backtest_april2025():
    """
    Backtest the economic model against the April 14, 2025 US-Iran standoff.
    Brent rose +8% in a single session. This endpoint shows what the model predicts
    vs. what actually happened, with error metrics.
    """
    params = APP_STATE["params"]
    # Estimated India-relevant supply gap from the April 2025 threat event:
    # ~40% of India's 2.574 Mb/d modeled imports × 30% risk probability = ~308 kb/d implied gap
    estimated_gap = 308_000  # bbl/day implied gap from threat pricing

    return compute_backtest_cascade(
        actual_gap_bbl_day=estimated_gap,
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

    horizon = min(req.horizon_days, 60)  # cap at 60 days
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
        response["no_reroute_snapshots"] = no_reroute_snapshots
        response["summary"]["days_to_stabilize_with_reroute"] = _days_to_stabilize(snapshots)
        response["summary"]["days_to_stabilize_without_reroute"] = _days_to_stabilize(no_reroute_snapshots)
        response["summary"]["total_cost_with_reroute_usd"] = _total_reroute_cost(snapshots)
        response["summary"]["total_cost_without_reroute_usd"] = _total_reroute_cost(no_reroute_snapshots)

    return response


def _days_to_stabilize(snapshots: list[dict]) -> Optional[int]:
    """First day fulfillment recovers to >= 99% *after* the shock's trough.

    Pipeline inertia (cargoes already in transit when a scenario starts) keeps
    fulfillment near 100% for the first few days regardless of the disruption
    — scanning from day 0 would misreport that pre-shock grace period as
    "already stabilized." Instead find the day of minimum fulfillment (the
    trough), then look forward from there. Returns None if recovery to 99%
    never happens within the simulated horizon (reported as "not stabilized"
    by the caller, not a misleading number).
    """
    if not snapshots:
        return None
    trough_idx = min(range(len(snapshots)), key=lambda i: snapshots[i]["fulfillment_pct_overall"])
    for s in snapshots[trough_idx:]:
        if s["fulfillment_pct_overall"] >= 99.0:
            return s["day"]
    return None


def _total_reroute_cost(snapshots: list[dict]) -> float:
    """Sum of each day's reroute cost premium over undisrupted baseline routing."""
    return sum(s["cascade"].get("daily_reroute_cost_premium_usd", 0.0) for s in snapshots)


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
