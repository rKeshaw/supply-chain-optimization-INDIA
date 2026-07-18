"""
Main orchestration pipeline: process_signal.

Rewritten using LangGraph to implement a true state-based agent workflow
with feedback loops (Policy Critic <-> Optimizer).
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any, TypedDict, Dict

import networkx as nx
from langgraph.graph import StateGraph, END

from agents.schema import Event
from agents import extraction_agent
from agents import scenario_agent
from agents import policy_critic_agent
from agents import explainer_agent
from graph_engine.build_graph import apply_event_to_graph, get_graph_state_json
from graph_engine.routing import compute_pareto_routes
from graph_engine.reserve_optimizer import get_spr_status_summary
from graph_engine.economic_model import compute_cascade

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """The shared state dictionary for the LangGraph workflow."""
    raw_text: str
    G_current: nx.DiGraph
    sim_state: dict
    params: dict
    source_override: Optional[str]
    timestamp_override: Optional[datetime]
    significance_threshold: Optional[float]
    event_override: Optional[Event]
    
    # Timestamps for latency tracking
    t_start: datetime
    t_graph_updated: Optional[datetime]
    t_brief_emitted: Optional[datetime]
    
    # Pipeline outputs
    event: Optional[Dict[str, Any]]
    signal_strength: float
    recompute_triggered: bool
    reason: str
    
    G_updated: Optional[nx.DiGraph]
    graph_state: Optional[dict]
    
    pareto_routes: Optional[dict]
    spr_status: Optional[dict]
    cascade: Optional[dict]
    
    critic_result: Optional[dict]
    policy_overrides: Optional[dict]
    
    brief: Optional[str]
    scenario_hypotheses: list
    
    latency_ms: float


def _elapsed_ms(t_start: datetime) -> float:
    return (datetime.now(timezone.utc) - t_start).total_seconds() * 1000


# --- Node Functions ---

def extract_signal_node(state: PipelineState) -> dict:
    """Extract event from raw text and check significance threshold."""
    t_start = state["t_start"]
    
    event = state.get("event_override") or extraction_agent.parse(
        state["raw_text"],
        source_override=state.get("source_override"),
        timestamp_override=state.get("timestamp_override"),
    )
    
    if event is None:
        logger.warning("process_signal: extraction returned None. Skipping pipeline.")
        return {
            "event": None,
            "recompute_triggered": False,
            "reason": "extraction_failed",
            "latency_ms": _elapsed_ms(t_start),
            "signal_strength": 0.0,
        }
        
    threshold = state.get("significance_threshold")
    if threshold is None:
        threshold = state["params"].get("recompute_significance_threshold", {}).get("value", 0.12)
        
    signal_strength = event.severity * event.confidence
    if event.event_type == "unrelated" or signal_strength < threshold:
        logger.info(
            f"process_signal: event '{event.id}' below threshold "
            f"({signal_strength:.3f} < {threshold}). Logged, not recomputed."
        )
        return {
            # mode="json" so timestamp serializes to an ISO string, not a raw
            # datetime object — scenario_agent_node later json.dumps() this
            # dict directly, which otherwise raises (caught, but silently
            # drops the scenario-hypothesis step every single time).
            "event": event.model_dump(mode="json"),
            "signal_strength": signal_strength,
            "recompute_triggered": False,
            "reason": "below_threshold",
            "latency_ms": _elapsed_ms(t_start),
        }
        
    return {
        "event": event.model_dump(mode="json"),
        "signal_strength": signal_strength,
        "recompute_triggered": True,
        "reason": "above_threshold",
    }


def update_graph_node(state: PipelineState) -> dict:
    """Apply event to graph (risk score update + openness cascade).

    Reconstructs the Event from state["event"] (a declared TypedDict field)
    rather than carrying the raw Pydantic object through an undeclared state
    key — the same pattern explainer_node already uses below. An earlier
    version threaded the object through an undeclared "_event_obj" key; with
    this LangGraph version, a node's returned key that isn't declared in the
    TypedDict schema has no channel to propagate through, so update_graph_node
    silently received None here and apply_event_to_graph's `if event is None:
    return G` no-opped the entire graph update — recompute_triggered still
    reported True (decided earlier in this function), but the graph never
    actually changed. Confirmed and fixed; see the exact same reconstruction
    at explainer_node below.
    """
    event_obj = Event(**state["event"])

    G_updated = apply_event_to_graph(state["G_current"], event_obj, state["params"])
    t_graph_updated = datetime.now(timezone.utc)
    
    return {
        "G_updated": G_updated,
        "t_graph_updated": t_graph_updated,
    }


def optimize_routing_node(state: PipelineState) -> dict:
    """Solve routing (Pareto routes) and compute economic cascade."""
    G_updated = state["G_updated"]
    params = state["params"]
    
    demand = {
        nid: data.get("consumption_rate_bbl_day", 0)
        for nid, data in G_updated.nodes(data=True)
        if data.get("type") == "refinery_out"
    }
    
    # If policy critic generated overrides in a previous iteration, use them
    policy_overrides = state.get("policy_overrides")
    
    pareto_routes = compute_pareto_routes(
        G_updated, demand, params, policy_overrides=policy_overrides
    )

    spr_status = get_spr_status_summary(state["sim_state"], params)

    total_demand = sum(demand.values())
    cost_optimal_route = pareto_routes.get("cost_optimal", {})
    cost_optimal_volume = cost_optimal_route.get("total_volume", 0)
    gap = max(0.0, total_demand - cost_optimal_volume)

    # Cost channel: reroute premium vs. the undisrupted baseline landed cost.
    from graph_engine.routing import avg_cost_per_bbl
    baseline_avg = params.get("_baseline_routing_avg_cost_per_bbl", {}).get("value", 0.0)
    premium = max(0.0, avg_cost_per_bbl(cost_optimal_route) - baseline_avg)

    # Market channel: source barrels removed from the global market (sources start
    # fully open in the baseline, so the openness deficit is the barrels lost).
    market_supply_loss = sum(
        (data.get("capacity_bbl_day") or 0) * (1.0 - data.get("openness", 1.0))
        for _, data in G_updated.nodes(data=True)
        if data.get("type") == "source"
    )

    cascade = compute_cascade(
        gap, total_demand, 0, params,
        reroute_cost_premium_usd_per_bbl=premium,
        delivered_volume_bbl_day=cost_optimal_volume,
        market_supply_loss_bbl_day=market_supply_loss,
    )
    
    return {
        "pareto_routes": pareto_routes,
        "spr_status": spr_status,
        "cascade": cascade,
    }


def policy_critic_node(state: PipelineState) -> dict:
    """Policy critic evaluates the flow."""
    G_updated = state["G_updated"]
    critic_result = policy_critic_agent.verify(
        routing_result=state["pareto_routes"],
        spr_state=state["spr_status"],
        graph_state=get_graph_state_json(G_updated),
        params=state["params"],
    )
    
    updates = {"critic_result": critic_result}
    
    if critic_result.get("re_solve_required", False) and not state.get("policy_overrides"):
        # We only allow one re-solve to prevent infinite loops
        logger.info("process_signal: critic requested re-solve. Re-running routing...")
        overrides = policy_critic_agent.get_re_solve_overrides(critic_result)
        updates["policy_overrides"] = overrides
        
    return updates


def explainer_node(state: PipelineState) -> dict:
    """Generate the executive brief."""
    G_updated = state["G_updated"]
    graph_with_flow = get_graph_state_json(
        G_updated, state["pareto_routes"].get("cost_optimal", {}).get("flow_dict")
    )
    
    # Reconstruct event object for explainer
    event_obj = Event(**state["event"])
    
    brief = explainer_agent.summarize(
        event=event_obj,
        validated_routing={"pareto_routes": state["pareto_routes"]},
        economic_impact=state["cascade"],
        spr_status=state["spr_status"],
        critic_result=state["critic_result"],
        graph_state=graph_with_flow,
    )
    
    t_brief_emitted = datetime.now(timezone.utc)
    return {
        "brief": brief,
        "t_brief_emitted": t_brief_emitted,
        "graph_state": graph_with_flow,
    }


def scenario_agent_node(state: PipelineState) -> dict:
    """Scenario agent generates hypothetical secondary shocks."""
    hypotheses = []
    try:
        hypotheses = scenario_agent.generate_hypotheses(
            graph_state=state["graph_state"],
            recent_events=[state["event"]],
            params=state["params"],
        )
    except Exception as e:
        logger.warning(f"Scenario agent failed (non-critical): {e}")
        
    return {
        "scenario_hypotheses": hypotheses,
        "latency_ms": _elapsed_ms(state["t_start"]),
    }


# --- Edges ---

def route_after_extraction(state: PipelineState) -> str:
    """Conditional edge after extraction."""
    if state.get("recompute_triggered"):
        return "update_graph"
    return END

def route_after_critic(state: PipelineState) -> str:
    """Conditional edge after policy critic."""
    # If a re-solve is required AND we haven't already applied overrides, loop back.
    if state["critic_result"].get("re_solve_required") and state.get("policy_overrides") is not None and not state["critic_result"].get("re_solve_ran"):
        state["critic_result"]["re_solve_ran"] = True
        return "optimize_routing"
    return "explainer"


# --- Build LangGraph ---

workflow = StateGraph(PipelineState)

workflow.add_node("extract_signal", extract_signal_node)
workflow.add_node("update_graph", update_graph_node)
workflow.add_node("optimize_routing", optimize_routing_node)
workflow.add_node("policy_critic", policy_critic_node)
workflow.add_node("explainer", explainer_node)
workflow.add_node("scenario_agent", scenario_agent_node)

workflow.set_entry_point("extract_signal")

workflow.add_conditional_edges(
    "extract_signal",
    route_after_extraction,
    {"update_graph": "update_graph", END: END}
)
workflow.add_edge("update_graph", "optimize_routing")
workflow.add_edge("optimize_routing", "policy_critic")
workflow.add_conditional_edges(
    "policy_critic",
    route_after_critic,
    {"optimize_routing": "optimize_routing", "explainer": "explainer"}
)
workflow.add_edge("explainer", "scenario_agent")
workflow.add_edge("scenario_agent", END)

app_graph = workflow.compile()


def process_signal(
    raw_text: str,
    G_current: nx.DiGraph,
    sim_state: dict,
    params: dict,
    source_override: Optional[str] = None,
    timestamp_override: Optional[datetime] = None,
    significance_threshold: Optional[float] = None,
    event_override: Optional[Event] = None,
) -> dict:
    """
    Entry point for the API to invoke the LangGraph pipeline.
    """
    initial_state = {
        "raw_text": raw_text,
        "G_current": G_current,
        "sim_state": sim_state,
        "params": params,
        "source_override": source_override,
        "timestamp_override": timestamp_override,
        "significance_threshold": significance_threshold,
        "event_override": event_override,
        "t_start": datetime.now(timezone.utc),
        "policy_overrides": None,
    }
    
    # Run the graph
    final_state = app_graph.invoke(initial_state)
    
    # Format the output to match the expected API contract
    if not final_state.get("recompute_triggered"):
        return {
            "event": final_state.get("event"),
            "signal_strength": final_state.get("signal_strength", 0.0),
            "recompute_triggered": False,
            "reason": final_state.get("reason"),
            "latency_ms": final_state.get("latency_ms"),
        }
        
    pareto_routes = final_state.get("pareto_routes", {})
    event = final_state["event"]
    
    total_latency_ms = final_state["latency_ms"]
    
    logger.info(
        f"[LATENCY_FULL] "
        f"signal_ts={event['timestamp']} "
        f"graph_update_ts={final_state['t_graph_updated'].isoformat()} "
        f"brief_ts={final_state['t_brief_emitted'].isoformat()} "
        f"total_pipeline_ms={total_latency_ms:.0f}"
    )
    
    return {
        "event": event,
        "_updated_graph": final_state["G_updated"],
        "signal_strength": final_state["signal_strength"],
        "recompute_triggered": True,
        "graph_state": final_state["graph_state"],
        "routing": {
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
                if k not in ("pareto_comparison",)
            },
            "pareto_comparison": pareto_routes.get("pareto_comparison", {}),
        },
        "spr": final_state["spr_status"],
        "cascade": final_state["cascade"],
        "policy_check": final_state["critic_result"],
        "brief": final_state["brief"],
        "scenario_hypotheses": final_state.get("scenario_hypotheses", []),
        "latency": {
            "signal_ts": event["timestamp"],
            "graph_update_ts": final_state["t_graph_updated"].isoformat(),
            "brief_emitted_ts": final_state["t_brief_emitted"].isoformat(),
            "total_pipeline_ms": total_latency_ms,
        },
    }
