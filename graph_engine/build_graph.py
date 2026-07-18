"""
Graph builder for the Energy Supply Chain Resilience system.

Loads nodes and edges from JSON data files, constructs a directed graph
ready for max-flow and min-cost-flow computation, and computes the
baseline (undisrupted) flow state.

Design contract:
- The baseline graph is NEVER mutated after initial build.
- All disruption scenarios operate on deep copies.
- Effective edge capacity = base_capacity × edge.openness × min(from_node.openness, to_node.openness)
"""

import json
import copy
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import networkx as nx

from agents.schema import Node, Edge


def load_graph(data_dir: Path) -> tuple[nx.DiGraph, dict[str, Node], list[Edge]]:
    """
    Load nodes and edges from JSON files and construct the directed supply chain graph.

    Node-splitting is applied to refinery nodes: each refinery has a `_in` and `_out`
    node joined by one internal edge whose capacity = the refinery's processing throughput.
    This means node failures and edge failures share one code path downstream.

    Args:
        data_dir: Path to the directory containing nodes.json and edges.json.

    Returns:
        Tuple of (graph, nodes_dict, edges_list).
        - graph: NetworkX DiGraph with all nodes and edges loaded.
        - nodes_dict: Dict mapping node ID to validated Node object.
        - edges_list: List of validated Edge objects.

    Raises:
        FileNotFoundError: If nodes.json or edges.json are missing.
        ValueError: If any refinery_in node is missing its corresponding internal edge.
        ValidationError: If node or edge data fails schema validation.
    """
    nodes_path = data_dir / "nodes.json"
    edges_path = data_dir / "edges.json"

    nodes_raw = json.loads(nodes_path.read_text(encoding="utf-8"))
    edges_raw = json.loads(edges_path.read_text(encoding="utf-8"))

    # Remove comment-only entries from edges (entries with only _comment key)
    edges_raw = [e for e in edges_raw if "id" in e]

    # Validate and build node dict
    nodes: dict[str, Node] = {}
    for n in nodes_raw:
        node = Node(**n)
        nodes[node.id] = node

    # Validate edges
    edges: list[Edge] = []
    for e in edges_raw:
        edge = Edge(**e)
        edges.append(edge)

    # Validate: every refinery_in must have an internal edge to its refinery_out
    refinery_in_ids = {nid for nid, n in nodes.items() if n.type == "refinery_in"}
    edge_pairs = {(e.from_id, e.to_id) for e in edges if e.mode == "internal"}
    missing_internal = []
    for in_id in refinery_in_ids:
        out_id = in_id.replace("_in", "_out")
        if (in_id, out_id) not in edge_pairs:
            missing_internal.append(f"{in_id} -> {out_id}")
    if missing_internal:
        raise ValueError(
            f"Missing internal refinery edges: {missing_internal}. "
            "Check edges.json — every refinery_in must connect to its refinery_out."
        )

    # Build the graph
    G = nx.DiGraph()

    # Add nodes with all attributes
    for nid, node in nodes.items():
        G.add_node(nid, **node.model_dump())

    # Add edges, computing effective capacity from openness of both endpoints
    for edge in edges:
        from_openness = nodes[edge.from_id].openness if edge.from_id in nodes else 1.0
        to_openness = nodes[edge.to_id].openness if edge.to_id in nodes else 1.0
        node_openness_min = min(from_openness, to_openness)
        effective_openness = edge.openness * node_openness_min
        effective_capacity = edge.base_capacity_bbl_day * effective_openness

        G.add_edge(
            edge.from_id,
            edge.to_id,
            # Flow computation attributes
            capacity=effective_capacity,
            weight=edge.cost_per_bbl,  # used by min_cost_flow
            # Original attributes for disruption/routing modules
            edge_id=edge.id,
            base_capacity_bbl_day=edge.base_capacity_bbl_day,
            cost_per_bbl=edge.cost_per_bbl,
            transit_time_days=edge.transit_time_days,
            openness=edge.openness,
            risk_multiplier=edge.risk_multiplier,
            grade=edge.grade,
            mode=edge.mode,
            path=edge.path,
        )

    return G, nodes, edges


def compute_baseline(G: nx.DiGraph) -> dict:
    """
    Compute undisrupted network max-flow and min-cut between super_source and super_sink.

    Returns a dict with:
    - flow_value: Total max-flow (bbl/day)
    - flow_dict: Per-edge flow assignment {from_node: {to_node: flow_value}}
    - cut_set: List of (from_id, to_id) edge tuples in the minimum cut
    - cut_value: Min-cut capacity (should equal flow_value)
    - flow_per_refinery: Dict mapping each refinery_out node to its allocated inflow

    The min-cut set should include Hormuz-related edges if the data is correct —
    this is the primary sanity check for the graph data (see test_graph_engine.py).
    """
    if "super_source" not in G or "super_sink" not in G:
        raise ValueError("Graph must contain 'super_source' and 'super_sink' nodes.")

    flow_value, flow_dict = nx.maximum_flow(G, "super_source", "super_sink")

    # Min-cut partition
    cut_value, partition = nx.minimum_cut(G, "super_source", "super_sink")
    reachable, non_reachable = partition
    cut_set = [
        (u, v) for u, v in G.edges()
        if u in reachable and v in non_reachable
    ]

    # Flow delivered to each refinery_out node
    flow_per_refinery: dict[str, float] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") == "refinery_out":
            inflow = sum(
                flow_dict.get(pred, {}).get(nid, 0)
                for pred in G.predecessors(nid)
            )
            flow_per_refinery[nid] = inflow

    # Fulfilment per refinery (fraction of consumption_rate_bbl_day met)
    fulfillment: dict[str, float] = {}
    for nid, flow in flow_per_refinery.items():
        consumption = G.nodes[nid].get("consumption_rate_bbl_day") or 0
        fulfillment[nid] = (flow / consumption) if consumption > 0 else 1.0

    return {
        "flow_value": flow_value,
        "flow_dict": flow_dict,
        "cut_set": cut_set,
        "cut_value": cut_value,
        "flow_per_refinery": flow_per_refinery,
        "fulfillment": fulfillment,
    }


def apply_event_to_graph(
    G: nx.DiGraph,
    event,  # agents.schema.Event — avoid circular import
    params: dict,
) -> nx.DiGraph:
    """
    Apply a validated Event to the graph's risk scores and return an updated graph.

    NEVER mutates the input graph — returns a shallow copy with updated attributes.
    Only confirmed Events (not unrelated or None) update graph state.

    Args:
        G: Current graph state.
        event: Validated Event object (from extraction_agent.parse).
        params: Parameters dict (for risk_decay_factor).

    Returns:
        Updated graph with risk_score and openness adjusted for the affected element.
    """
    from agents.schema import decay_risk_score, update_risk_score  # local import to avoid circular

    if event is None:
        return G
    decay = params.get("risk_decay_factor", {}).get("value", 0.92)
    G_updated = copy.deepcopy(G)

    # Decay every tracked risk value to the timestamp of the newly processed
    # signal. This keeps replay and live event streams on the same time basis.
    _decay_graph_risk_to_timestamp(G_updated, event.timestamp, decay, decay_risk_score)

    if event.event_type == "unrelated" or event.affected_graph_element is None:
        return G_updated

    target_id = event.affected_graph_element

    if target_id in G_updated.nodes:
        node_data = G_updated.nodes[target_id]
        current_risk = node_data.get("risk_score", 0.0)

        if event.event_type == "reopening":
            # Reopening is evidence that the operating risk has reduced.
            risk_reduction = event.severity * event.confidence
            new_risk = current_risk * (1.0 - risk_reduction)
        else:
            new_risk = update_risk_score(
                current_risk, event.severity, event.confidence, decay
            )

        node_data["risk_score"] = new_risk
        node_data["openness"] = 1.0 - new_risk
        node_data["last_updated"] = event.timestamp.isoformat()
        _refresh_effective_capacities(G_updated)
        return G_updated

    # Events may target a canonical edge ID as specified by the Event schema.
    for _, _, edge_data in G_updated.edges(data=True):
        if edge_data.get("edge_id") != target_id:
            continue

        current_risk = edge_data.get("risk_score", 0.0)
        if event.event_type == "reopening":
            new_risk = current_risk * (1.0 - event.severity * event.confidence)
        else:
            new_risk = update_risk_score(
                current_risk, event.severity, event.confidence, decay
            )

        edge_data["risk_score"] = new_risk
        edge_data["openness"] = 1.0 - new_risk
        edge_data["last_updated"] = event.timestamp.isoformat()
        _refresh_effective_capacities(G_updated)
        return G_updated

    # Keep the time-decayed snapshot even when entity resolution was stale.
    return G_updated


def _decay_graph_risk_to_timestamp(
    G: nx.DiGraph,
    as_of: datetime,
    decay_factor_per_day: float,
    decay_risk_score,
) -> None:
    """Advance all node/edge risk values to ``as_of`` in place."""
    for _, data in G.nodes(data=True):
        last_updated = _as_utc_datetime(data.get("last_updated"))
        elapsed_days = (as_of - last_updated).total_seconds() / 86_400 if last_updated else 0.0
        data["risk_score"] = decay_risk_score(
            data.get("risk_score", 0.0), decay_factor_per_day, elapsed_days
        )
        data["openness"] = 1.0 - data["risk_score"]

    for _, _, data in G.edges(data=True):
        last_updated = _as_utc_datetime(data.get("last_updated"))
        elapsed_days = (as_of - last_updated).total_seconds() / 86_400 if last_updated else 0.0
        data["risk_score"] = decay_risk_score(
            data.get("risk_score", 0.0), decay_factor_per_day, elapsed_days
        )
        data["openness"] = 1.0 - data["risk_score"]

    _refresh_effective_capacities(G)


def _as_utc_datetime(value) -> Optional[datetime]:
    """Normalize datetime values loaded from Pydantic or JSON data."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _refresh_effective_capacities(G: nx.DiGraph) -> None:
    """Recompute every capacity from immutable base capacity and openness."""
    for u, v, edge_data in G.edges(data=True):
        node_openness = min(
            G.nodes[u].get("openness", 1.0),
            G.nodes[v].get("openness", 1.0),
        )
        edge_data["capacity"] = (
            edge_data.get("base_capacity_bbl_day", 0.0)
            * edge_data.get("openness", 1.0)
            * node_openness
        )


def get_graph_state_json(G: nx.DiGraph, flow_dict: Optional[dict] = None) -> dict:
    """
    Serialize the current graph state to a JSON-serializable dict for the frontend.

    Returns:
        Dict with 'nodes' and 'edges' arrays, each containing display attributes.
    """
    nodes_out = []
    for nid, data in G.nodes(data=True):
        nodes_out.append({
            "id": nid,
            "type": data.get("type"),
            "name": data.get("name"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "openness": data.get("openness", 1.0),
            "risk_score": data.get("risk_score", 0.0),
            "capacity_bbl_day": data.get("capacity_bbl_day"),
            "inventory_bbl": data.get("inventory_bbl"),
            "consumption_rate_bbl_day": data.get("consumption_rate_bbl_day"),
            "last_updated": data.get("last_updated"),
        })

    edges_out = []
    for u, v, data in G.edges(data=True):
        edges_out.append({
            "id": data.get("edge_id"),
            "from_id": u,
            "to_id": v,
            "capacity": data.get("capacity"),
            "base_capacity_bbl_day": data.get("base_capacity_bbl_day"),
            "cost_per_bbl": data.get("cost_per_bbl"),
            "transit_time_days": data.get("transit_time_days"),
            "openness": data.get("openness", 1.0),
            "effective_openness": data.get("capacity", 0) / max(data.get("base_capacity_bbl_day", 1), 1),
            "flow_bbl_day": (flow_dict or {}).get(u, {}).get(v, 0.0),
            "mode": data.get("mode"),
            "grade": data.get("grade"),
            "path": data.get("path"),
        })

    return {"nodes": nodes_out, "edges": edges_out}
