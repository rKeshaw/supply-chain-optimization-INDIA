"""Path-level daily supply-chain digital twin.

Each shipment is an entire source-to-refinery cargo.  It can arrive only after
the cumulative transit time of its selected path; intermediate graph edges are
never treated as independent cargoes.
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from graph_engine.disruption import apply_scenario
from graph_engine.routing import compute_pareto_routes
from graph_engine.build_graph import apply_event_to_graph
from agents.weather_adapter import check_weather_disruptions
import logging

logger = logging.getLogger(__name__)



@dataclass
class ShipmentInTransit:
    """One complete crude cargo moving from a source to a refinery."""
    source_id: str
    refinery_out: str
    volume_bbl: float
    grade: str
    days_remaining: float
    cost_per_bbl: float
    path: list[str]


@dataclass
class RefineryState:
    node_id: str
    inventory_bbl: float
    consumption_rate_bbl_day: float

    @property
    def days_of_cover(self) -> float:
        return self.inventory_bbl / self.consumption_rate_bbl_day if self.consumption_rate_bbl_day > 0 else float("inf")

    @property
    def is_critical(self) -> bool:
        return self.days_of_cover < 3.0


def build_initial_state(G: nx.DiGraph) -> dict:
    """Build mutable refinery and physically-capacitated SPR state from graph data."""
    refineries = {
        nid: RefineryState(
            node_id=nid,
            inventory_bbl=data.get("inventory_bbl") or 0.0,
            consumption_rate_bbl_day=data.get("consumption_rate_bbl_day") or 0.0,
        )
        for nid, data in G.nodes(data=True)
        if data.get("type") == "refinery_out"
    }
    spr = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") != "spr":
            continue
        connected_refineries = [
            target.replace("_in", "_out")
            for _, target in G.out_edges(nid)
            if G.nodes[target].get("type") == "refinery_in"
        ]
        spr[nid] = {
            "node_id": nid,
            "inventory_bbl": data.get("inventory_bbl") or 0.0,
            "storage_capacity_bbl": data.get("storage_capacity_bbl") or 0.0,
            "max_discharge_bbl_day": data.get("capacity_bbl_day") or 0.0,
            "connected_refineries": connected_refineries,
        }
    return {"refineries": refineries, "spr": spr, "shipments_in_transit": [], "day": 0}


def run_digital_twin(
    G_baseline: nx.DiGraph,
    scenario_dict: dict[str, float],
    params: dict,
    horizon_days: int = 30,
    initial_state: Optional[dict] = None,
    enable_live_weather: bool = False,
) -> list[dict]:
    """Simulate inventory, path-level arrivals, dispatches, and SPR support daily.

    ``enable_live_weather`` is opt-in: the default demo/backtest path stays fully
    deterministic and never blocks on the external marine API. Turn it on only
    when a live-telemetry overlay is genuinely wanted.
    """
    from graph_engine.economic_model import compute_cascade
    from graph_engine.reserve_optimizer import optimize_spr_draw
    from graph_engine.routing import avg_cost_per_bbl

    G_disrupted = apply_scenario(G_baseline, scenario_dict)

    # Optionally overlay live marine-weather disruptions (opt-in — see docstring).
    if enable_live_weather:
        try:
            weather_events = check_weather_disruptions()
            for event in weather_events:
                logger.info(f"Digital Twin applying weather event: {event.justification}")
                G_disrupted = apply_event_to_graph(G_disrupted, event, params)
        except Exception as e:
            logger.warning(f"Digital Twin failed to apply weather events: {e}")

    state = copy.deepcopy(initial_state) if initial_state is not None else build_initial_state(G_baseline)
    demand = {nid: ref.consumption_rate_bbl_day for nid, ref in state["refineries"].items()}
    total_demand = sum(demand.values())

    # Undisrupted baseline routing cost — reference for the daily reroute premium.
    baseline_avg_cost = params.get("_baseline_routing_avg_cost_per_bbl", {}).get("value")
    if baseline_avg_cost is None:
        baseline_avg_cost = avg_cost_per_bbl(
            compute_pareto_routes(G_baseline, demand, params).get("cost_optimal", {})
        )

    # At the start of a scenario, cargoes already ordered under normal routes
    # remain in the pipeline. Seed one daily arrival for every day of each
    # baseline path's transit time, then dispatches under the scenario take over.
    if initial_state is None:
        baseline_routing = compute_pareto_routes(G_baseline, demand, params)["cost_optimal"]
        state["shipments_in_transit"] = _seed_steady_state_pipeline(baseline_routing)

    results = []
    for day in range(horizon_days):
        arrivals = _advance_shipments(state)
        for refinery_out, volume in arrivals.items():
            state["refineries"][refinery_out].inventory_bbl += volume

        # Decide today’s future dispatches from the disrupted network.
        try:
            routing = compute_pareto_routes(G_disrupted, demand, params)["cost_optimal"]
            new_shipments = _shipments_from_path_allocations(routing.get("path_allocations", []))
            state["shipments_in_transit"].extend(new_shipments)
        except Exception:
            routing = {"total_volume": 0.0, "fulfillment": {}, "feasible": False, "path_allocations": []}

        # SPR is assessed against projected end-of-day inventory, then released
        # before consumption so an available same-day pipeline draw prevents a
        # false stockout.
        projected = copy.deepcopy(state)
        for refinery in projected["refineries"].values():
            refinery.inventory_bbl = max(0.0, refinery.inventory_bbl - refinery.consumption_rate_bbl_day)
        spr_draws = optimize_spr_draw(projected, params)
        spr_allocations = _allocate_spr_to_connected_refineries(state, spr_draws, params)
        for spr_id, draw in spr_draws.items():
            state["spr"][spr_id]["inventory_bbl"] = max(0.0, state["spr"][spr_id]["inventory_bbl"] - draw)
        for refinery_out, volume in spr_allocations.items():
            state["refineries"][refinery_out].inventory_bbl += volume

        unmet = _consume_refineries(state)
        gap_bbl_day = sum(unmet.values())
        delivered_today = total_demand - gap_bbl_day
        # Cost channel: today's reroute premium vs. undisrupted baseline routing.
        day_premium = max(0.0, avg_cost_per_bbl(routing) - baseline_avg_cost)
        cascade = compute_cascade(
            gap_bbl_day, total_demand, day, params,
            reroute_cost_premium_usd_per_bbl=day_premium,
            delivered_volume_bbl_day=delivered_today,
        )

        results.append({
            "day": day,
            "total_demand_bbl_day": total_demand,
            "total_routed_bbl_day": routing["total_volume"],
            "arrivals_bbl_day": sum(arrivals.values()),
            "spr_draw_bbl_day": sum(spr_draws.values()),
            "gap_bbl_day": gap_bbl_day,
            "fulfillment_pct_overall": delivered_today / max(total_demand, 1.0) * 100,
            "refineries": {
                nid: {
                    "inventory_bbl": ref.inventory_bbl,
                    "days_of_cover": ref.days_of_cover,
                    "consumption_rate": ref.consumption_rate_bbl_day,
                    "is_critical": ref.is_critical,
                    "arrivals_bbl": arrivals.get(nid, 0.0),
                    "spr_draw_bbl": spr_allocations.get(nid, 0.0),
                    "unmet_bbl": unmet.get(nid, 0.0),
                }
                for nid, ref in state["refineries"].items()
            },
            "spr": {
                nid: {
                    "inventory_bbl": facility["inventory_bbl"],
                    "storage_capacity_bbl": facility["storage_capacity_bbl"],
                    "fill_pct": facility["inventory_bbl"] / max(facility["storage_capacity_bbl"], 1.0) * 100,
                }
                for nid, facility in state["spr"].items()
            },
            "cascade": cascade,
            "routing_feasible": routing.get("feasible", False),
            "shipments_in_transit_count": len(state["shipments_in_transit"]),
        })
    return results


def _seed_steady_state_pipeline(routing: dict) -> list[ShipmentInTransit]:
    """Seed cargoes already at different points along each baseline path."""
    shipments = []
    for allocation in routing.get("path_allocations", []):
        transit_days = max(1, math.ceil(allocation["transit_time_days"]))
        for offset in range(transit_days):
            shipment = _shipment_from_allocation(allocation)
            shipment.days_remaining = float(offset + 1)
            shipments.append(shipment)
    return shipments


def _shipments_from_path_allocations(allocations: list[dict]) -> list[ShipmentInTransit]:
    return [_shipment_from_allocation(allocation) for allocation in allocations if allocation.get("volume_bbl_day", 0) > 0]


def _shipment_from_allocation(allocation: dict) -> ShipmentInTransit:
    return ShipmentInTransit(
        source_id=allocation["source_id"],
        refinery_out=allocation["refinery_out"],
        volume_bbl=float(allocation["volume_bbl_day"]),
        grade=allocation["grade"],
        days_remaining=max(1.0, float(allocation["transit_time_days"])),
        cost_per_bbl=float(allocation["cost_per_bbl"]),
        path=list(allocation["path"]),
    )


def _advance_shipments(state: dict) -> dict[str, float]:
    """Advance complete cargoes by one day and return refinery arrivals."""
    arrivals = {nid: 0.0 for nid in state["refineries"]}
    in_transit = []
    for shipment in state["shipments_in_transit"]:
        shipment.days_remaining -= 1.0
        if shipment.days_remaining <= 0:
            arrivals[shipment.refinery_out] += shipment.volume_bbl
        else:
            in_transit.append(shipment)
    state["shipments_in_transit"] = in_transit
    return arrivals


def _allocate_spr_to_connected_refineries(state: dict, draws: dict[str, float], params: dict) -> dict[str, float]:
    """Assign each facility's draw only to refineries with a physical link."""
    floor_days = params.get("spr_safety_floor_days", {}).get("value", 3.0)
    allocations = {nid: 0.0 for nid in state["refineries"]}
    for spr_id, draw in draws.items():
        if draw <= 0:
            continue
        connected = state["spr"][spr_id].get("connected_refineries", [])
        needs = {
            ref_id: max(
                0.0,
                (floor_days + 1.0) * state["refineries"][ref_id].consumption_rate_bbl_day
                - state["refineries"][ref_id].inventory_bbl,
            )
            for ref_id in connected
        }
        total_need = sum(needs.values())
        if total_need <= 0:
            continue
        for ref_id, need in needs.items():
            allocations[ref_id] += draw * need / total_need
    return allocations


def _consume_refineries(state: dict) -> dict[str, float]:
    """Consume available crude and retain any unmet refinery demand explicitly."""
    unmet = {}
    for ref_id, refinery in state["refineries"].items():
        fulfilled = min(refinery.inventory_bbl, refinery.consumption_rate_bbl_day)
        refinery.inventory_bbl -= fulfilled
        unmet[ref_id] = refinery.consumption_rate_bbl_day - fulfilled
    return unmet
