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
from graph_engine.routing import compute_pareto_routes, refinery_demand
from graph_engine.reserve_optimizer import get_spr_status_summary, planned_draw_from_allocations
from graph_engine.economic_model import compute_cascade, global_supply_loss_bbl_day

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
    decay_as_of: Optional[datetime]
    
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
    # Must live in the state schema: policy_critic_node replaces critic_result
    # wholesale each pass, so a guard stored there is discarded every iteration.
    re_solve_count: int
    
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
        reason = "unrelated" if event.event_type == "unrelated" else "below_threshold"
        logger.info(
            f"process_signal: event '{event.id}' {reason} "
            f"({signal_strength:.3f} < {threshold}). Logged, not recomputed."
        )
        return {
            "event": event.model_dump(mode="json"),
            "signal_strength": signal_strength,
            "recompute_triggered": False,
            "reason": reason,
            "latency_ms": _elapsed_ms(t_start),
        }
        
    return {
        "event": event.model_dump(mode="json"),
        "signal_strength": signal_strength,
        "recompute_triggered": True,
        "reason": "above_threshold",
    }


def update_graph_node(state: PipelineState) -> dict:
    """Apply an event to the graph, updating risk scores and openness.

    The Event is rebuilt from ``state["event"]``, a declared field on the state
    schema. Only declared fields have a channel to propagate through, so passing
    the Pydantic object directly would arrive as None here and silently skip the
    whole graph update.
    """
    event_obj = Event(**state["event"])

    G_updated = apply_event_to_graph(
        state["G_current"], event_obj, state["params"],
        decay_as_of=state.get("decay_as_of"),
    )
    t_graph_updated = datetime.now(timezone.utc)
    
    return {
        "G_updated": G_updated,
        "t_graph_updated": t_graph_updated,
    }


def optimize_routing_node(state: PipelineState) -> dict:
    """Solve routing (Pareto routes) and compute economic cascade."""
    G_updated = state["G_updated"]
    params = state["params"]
    
    demand = refinery_demand(G_updated)
    policy_overrides = state.get("policy_overrides")
    
    pareto_routes = compute_pareto_routes(
        G_updated, demand, params, policy_overrides=policy_overrides
    )

    cost_optimal_route = pareto_routes.get("cost_optimal", {})
    spr_status = get_spr_status_summary(
        state["sim_state"], params,
        planned_draw=planned_draw_from_allocations(cost_optimal_route.get("path_allocations", [])),
    )

    total_demand = sum(demand.values())
    cost_optimal_volume = cost_optimal_route.get("total_volume", 0)
    gap = max(0.0, total_demand - cost_optimal_volume)

    from graph_engine.routing import reroute_premium_vs_baseline
    premium = reroute_premium_vs_baseline(G_updated, params, cost_optimal_route)
    market_supply_loss = global_supply_loss_bbl_day(G_updated, params)

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
        "re_solve_count": state.get("re_solve_count", 0) + (1 if policy_overrides else 0),
    }


def policy_critic_node(state: PipelineState) -> dict:
    """Policy critic evaluates the routing plan and flags violations.

    A re-solve is requested at most once per pipeline invocation, guarded by
    the counter check that precedes evaluation.
    """
    G_updated = state["G_updated"]
    critic_result = policy_critic_agent.verify(
        routing_result=state["pareto_routes"],
        spr_state=state["spr_status"],
        graph_state=get_graph_state_json(G_updated),
        params=state["params"],
    )
    
    updates = {"critic_result": critic_result}

    if state.get("re_solve_count", 0) == 0 and critic_result.get("re_solve_required", False):
        overrides = policy_critic_agent.get_re_solve_overrides(critic_result)
        if overrides:
            logger.info("process_signal: critic requested re-solve. Re-running routing...")
            updates["policy_overrides"] = overrides
        else:
            logger.info(
                "process_signal: critic flagged %s but proposed no actionable constraint "
                "change; reporting the violation without re-solving.",
                [v.get("rule_id") for v in critic_result.get("violations", [])],
            )

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
    """Conditional edge after the policy critic.

    Re-solves only when (a) the cap has not been reached, (b) the critic
    requested it, and (c) the critic supplied actionable constraint overrides.
    The counter is checked first so a malformed critic response cannot loop.
    """
    if (
        state.get("re_solve_count", 0) < 1
        and state["critic_result"].get("re_solve_required")
        and state.get("policy_overrides")
    ):
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
    decay_as_of: Optional[datetime] = None,
) -> dict:
    """
    Entry point for the API to invoke the LangGraph pipeline.

    decay_as_of: see apply_event_to_graph's docstring. Only the curated replay
    passes this (a compressed narrative clock); live/custom signals leave it
    None and decay uses the event's own real timestamp.
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
        "decay_as_of": decay_as_of,
        "t_start": datetime.now(timezone.utc),
        "policy_overrides": None,
        "re_solve_count": 0,
    }
    
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
                    "policy_breaches": v.get("policy_breaches", {}),
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
