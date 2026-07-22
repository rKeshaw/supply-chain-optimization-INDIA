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
    disable_rerouting: bool = False,
) -> list[dict]:
    """Simulate inventory, path-level arrivals, dispatches, and SPR support daily.

    ``enable_live_weather`` is opt-in: the default demo/backtest path stays fully
    deterministic and never blocks on the external marine API. Turn it on only
    when a live-telemetry overlay is genuinely wanted.

    ``disable_rerouting`` is the "no adaptive rerouting" counterfactual: instead
    of re-solving ``compute_pareto_routes`` against the disrupted network every
    day, it keeps shipping via the exact same paths the optimizer chose before
    the disruption, capped by whatever capacity those specific paths still have
    — volume through a closed/degraded corridor is simply lost, with no search
    for an alternative, ever, within the horizon. SPR draw still runs as normal
    (it's a domestic reserve mechanism, not an import-routing decision).
    """
    from graph_engine.economic_model import compute_cascade
    from graph_engine.routing import reroute_premium_vs_baseline

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

    # Reference plan for the daily reroute premium. The application solves this
    # once at startup; solving here covers standalone use from tests or scripts.
    if not (params or {}).get("_baseline_cost_route", {}).get("value"):
        params = dict(params or {})
        params["_baseline_cost_route"] = {
            "value": compute_pareto_routes(G_baseline, demand, params)["cost_optimal"]
        }

    # At the start of a scenario, cargoes already ordered under normal routes
    # remain in the pipeline. Seed one daily arrival for every day of each
    # baseline path's transit time, then dispatches under the scenario take over.
    # Also the reference point for the no-rerouting counterfactual below.
    baseline_routing = None
    if initial_state is None or disable_rerouting:
        baseline_routing = compute_pareto_routes(G_baseline, demand, params)["cost_optimal"]
    # Seed the steady-state pipeline from the baseline plan (arrivals that were
    # already en route when the disruption struck).
    if initial_state is None:
        state["shipments_in_transit"] = _seed_steady_state_pipeline(baseline_routing)

    no_reroute_allocations = None
    if disable_rerouting:
        no_reroute_allocations = _no_reroute_allocations(
            baseline_routing.get("path_allocations", []), G_disrupted
        )

    # The disrupted-network routing plan is re-solved on demand inside the daily
    # loop whenever SPR inventory changes materially (see _spr_capacity_changed).
    steady_routing = None
    last_spr_inventory: dict[str, float] = {}

    results = []
    for day in range(horizon_days):
        arrivals = _advance_shipments(state)
        for refinery_out, volume in arrivals.items():
            state["refineries"][refinery_out].inventory_bbl += volume

        if disable_rerouting:
            total_volume = sum(a["volume_bbl_day"] for a in no_reroute_allocations)
            routing = {
                "total_volume": total_volume,
                "feasible": total_volume > 0,
                "path_allocations": no_reroute_allocations,
            }
        else:
            # Re-solve whenever SPR inventory has shifted by more than 1% of
            # daily discharge. The LP's SPR capacity ceiling is inventory-
            # dependent, so a plan valid on day 0 may over-commit a partially-
            # drained cavern on day N.
            spr_changed = _spr_capacity_changed(
                state["spr"], last_spr_inventory, threshold_fraction=0.01
            )
            if steady_routing is None or spr_changed:
                G_day = _apply_current_spr_capacity(G_disrupted, state["spr"])
                try:
                    steady_routing = compute_pareto_routes(G_day, demand, params)["cost_optimal"]
                except Exception:
                    logger.exception(
                        "Digital twin: routing re-solve failed on day %d; "
                        "continuing with previous plan.", day
                    )
            last_spr_inventory = {sid: s["inventory_bbl"] for sid, s in state["spr"].items()}
            routing = steady_routing

        allocations = routing.get("path_allocations", [])

        state["shipments_in_transit"].extend(
            _shipments_from_path_allocations([a for a in allocations if not a.get("is_spr")])
        )
        spr_draws, spr_allocations = _execute_spr_draw(state, allocations)

        unmet = _consume_refineries(state)
        gap_bbl_day = sum(unmet.values())
        delivered_today = total_demand - gap_bbl_day
        day_premium = reroute_premium_vs_baseline(G_disrupted, params, routing)
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


def _spr_capacity_changed(
    spr_state: dict,
    prev_inventory: dict[str, float],
    threshold_fraction: float = 0.01,
) -> bool:
    """Return True when any SPR facility's inventory has changed by more than
    threshold_fraction of its maximum daily discharge rate since the last solve.

    The comparison is against max_discharge_bbl_day rather than total storage
    so that a tiny absolute change on a large cavern does not force a redundant
    LP solve when the routing plan cannot meaningfully improve.
    """
    for sid, facility in spr_state.items():
        prev = prev_inventory.get(sid, float("inf"))
        current = facility["inventory_bbl"]
        max_discharge = facility.get("max_discharge_bbl_day", 1.0) or 1.0
        if abs(current - prev) > threshold_fraction * max_discharge:
            return True
    return False


def _apply_current_spr_capacity(G: nx.DiGraph, spr_state: dict) -> nx.DiGraph:
    """Return a shallow copy of G with each SPR node's capacity_bbl_day updated
    to reflect the remaining inventory in the simulation state.

    The LP uses capacity_bbl_day as the ceiling for daily discharge. Using the
    static graph value after several days of drawdown would allow the solver to
    commit more SPR volume than the cavern actually holds.
    """
    G_day = G.copy()
    for sid, facility in spr_state.items():
        if sid not in G_day:
            continue
        max_discharge = facility.get("max_discharge_bbl_day", 0.0) or 0.0
        remaining = facility["inventory_bbl"]
        # Cap daily supply at the lesser of the physical discharge rate and
        # remaining inventory — the exact figure the LP's horizon constraint approximates.
        G_day.nodes[sid]["capacity_bbl_day"] = min(max_discharge, remaining)
    return G_day


def _seed_steady_state_pipeline(routing: dict) -> list[ShipmentInTransit]:
    """Seed cargoes already at different points along each baseline path.

    Reserve allocations are excluded. A barrel sitting in an Indian cavern is
    not a cargo at sea, and seeding it as one would credit the simulation with
    stock it has not drawn.
    """
    shipments = []
    for allocation in routing.get("path_allocations", []):
        if allocation.get("is_spr"):
            continue
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


def _no_reroute_allocations(baseline_allocations: list[dict], G_disrupted: nx.DiGraph) -> list[dict]:
    """The "no adaptive rerouting" counterfactual: keep shipping via the exact
    same paths the optimizer chose before the disruption, and lose whatever
    volume those specific paths can no longer carry — never search for an
    alternative. Two passes:

    1. Per-path bottleneck: cap each path's volume by the tightest edge
       capacity along that same path in the disrupted graph (a corridor
       that's now closed or degraded carries less, or nothing).
    2. Aggregate rebalance: several original paths can share one source,
       chokepoint, or SPR facility even though each looks fine on its own
       edge — mirrors the same aggregate capacity limit the real solver
       enforces per node (see _solve_arc_based_flow) by proportionally
       scaling down paths that collectively overrun a shared facility's
       true capacity.
    """
    node_type = {n: data.get("type") for n, data in G_disrupted.nodes(data=True)}

    capped = []
    for alloc in baseline_allocations:
        path = alloc["path"]
        bottleneck = alloc["volume_bbl_day"]
        broken = False
        for u, v in zip(path, path[1:]):
            if not G_disrupted.has_edge(u, v):
                broken = True
                break
            bottleneck = min(bottleneck, float(G_disrupted[u][v].get("capacity", 0.0)))
        if broken or bottleneck <= 1e-6:
            continue
        new_alloc = dict(alloc)
        new_alloc["volume_bbl_day"] = bottleneck
        capped.append(new_alloc)

    def _aggregate_cap(node_id: str) -> Optional[float]:
        if node_type.get(node_id) not in ("source", "spr", "chokepoint"):
            return None
        data = G_disrupted.nodes[node_id]
        return float(data.get("capacity_bbl_day") or 0.0) * float(data.get("openness", 1.0))

    for _ in range(5):  # a handful of passes converges quickly at realistic path-sharing depth
        usage: dict[str, float] = {}
        for a in capped:
            for node_id in {a["source_id"]} | set(a.get("chokepoints", [])):
                usage[node_id] = usage.get(node_id, 0.0) + a["volume_bbl_day"]

        over_subscribed = False
        for node_id, used in usage.items():
            cap = _aggregate_cap(node_id)
            if cap is None or cap <= 0 or used <= cap * 1.0001:
                continue
            over_subscribed = True
            ratio = cap / used
            for a in capped:
                if node_id == a["source_id"] or node_id in a.get("chokepoints", []):
                    a["volume_bbl_day"] *= ratio
        if not over_subscribed:
            break

    return [a for a in capped if a["volume_bbl_day"] > 1e-6]


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


def _execute_spr_draw(state: dict, allocations: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """Carry out the routing plan's reserve draw and deplete the caverns.

    The routing solve owns the drawdown decision. It is aware of grade, of
    discharge capacity and of each cavern's sustainable rate, and it is the plan
    already shown in the routing panel, so the simulation executes it rather
    than deciding a second time and risking two different accounts of the same
    reserve.

    Recapped against remaining stock every day, so a facility that empties part
    way through the horizon stops delivering instead of going negative.

    Returns draw per facility and delivery per refinery, both in barrels a day.
    """
    draws = {spr_id: 0.0 for spr_id in state["spr"]}
    deliveries = {ref_id: 0.0 for ref_id in state["refineries"]}

    for allocation in allocations:
        if not allocation.get("is_spr"):
            continue
        facility = state["spr"].get(allocation["source_id"])
        target = allocation.get("refinery_out")
        if facility is None or target not in state["refineries"]:
            continue
        volume = min(float(allocation.get("volume_bbl_day", 0.0)), facility["inventory_bbl"])
        if volume <= 0:
            continue
        facility["inventory_bbl"] -= volume
        draws[allocation["source_id"]] += volume
        deliveries[target] += volume
        state["refineries"][target].inventory_bbl += volume

    return draws, deliveries


def _consume_refineries(state: dict) -> dict[str, float]:
    """Consume available crude and retain any unmet refinery demand explicitly."""
    unmet = {}
    for ref_id, refinery in state["refineries"].items():
        fulfilled = min(refinery.inventory_bbl, refinery.consumption_rate_bbl_day)
        refinery.inventory_bbl -= fulfilled
        unmet[ref_id] = refinery.consumption_rate_bbl_day - fulfilled
    return unmet
