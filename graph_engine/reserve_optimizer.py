"""
SPR (Strategic Petroleum Reserve) optimizer.

Decides when and how much to draw down from ISPRL facilities to maintain
refinery safety floors, and when to trigger refill once disruption eases.

Policy contract (from plan.md):
- Draw only as much per day as needed to keep every refinery's projected
  days-of-cover above the safety floor.
- Never draw more than the SPR's maximum daily discharge rate.
- Refill begins when the disruption risk score drops below a threshold.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_refill_allocation(
    state: dict,
    params: dict,
    surplus_bbl_day: float,
    current_corridor_risk: float,
) -> dict[str, float]:
    """
    Compute SPR refill allocation when conditions allow (disruption easing).

    Refill is triggered when:
    1. The affected corridor's risk score drops below spr_refill_threshold_risk
    2. There is a rerouted supply surplus (supply > demand)

    Refill rate is capped at 30% of the surplus (to avoid over-committing supply
    back to SPR when refineries may still need buffer stock).

    Args:
        state: Current simulation state.
        params: Parameters dict.
        surplus_bbl_day: Daily supply surplus available for reallocation.
        current_corridor_risk: Current risk score of the disrupted corridor.

    Returns:
        Dict mapping spr node IDs to refill amounts (bbl/day).
        All zeros if refill conditions are not met.
    """
    refill_threshold = params.get("spr_refill_threshold_risk", {}).get("value", 0.25)
    target_days = params.get("spr_target_days", {}).get("value", 9.5)
    national_consumption = params.get("national_consumption_bbl_day", {}).get("value", 5_000_000)

    if current_corridor_risk > refill_threshold:
        return {spr_id: 0.0 for spr_id in state["spr"]}

    if surplus_bbl_day <= 0:
        return {spr_id: 0.0 for spr_id in state["spr"]}

    physical_capacity = sum(s.get("storage_capacity_bbl", 0.0) for s in state["spr"].values())
    target_total_bbl = min(target_days * national_consumption, physical_capacity)
    current_total_bbl = sum(s["inventory_bbl"] for s in state["spr"].values())
    total_deficit = max(0.0, target_total_bbl - current_total_bbl)

    if total_deficit <= 0:
        return {spr_id: 0.0 for spr_id in state["spr"]}

    max_refill_today = min(surplus_bbl_day * 0.3, total_deficit)

    # Distribute proportionally by deficit at each facility
    refills: dict[str, float] = {}
    facility_deficits = {}
    for spr_id, s in state["spr"].items():
        cap_share = s.get("storage_capacity_bbl", 0.0) / max(physical_capacity, 1)
        target_facility = target_total_bbl * cap_share
        facility_deficits[spr_id] = max(0.0, target_facility - s["inventory_bbl"])

    total_facility_deficit = sum(facility_deficits.values())
    for spr_id in state["spr"]:
        share = facility_deficits[spr_id] / max(total_facility_deficit, 1)
        refills[spr_id] = min(
            max_refill_today * share,
            facility_deficits[spr_id],
        )

    return refills


def planned_draw_from_allocations(allocations: list[dict]) -> dict[str, float]:
    """Per-facility daily reserve draw implied by a solved routing plan."""
    draw: dict[str, float] = {}
    for allocation in allocations or []:
        if allocation.get("is_spr"):
            facility = allocation["source_id"]
            draw[facility] = draw.get(facility, 0.0) + float(allocation.get("volume_bbl_day", 0.0))
    return draw


def get_spr_status_summary(
    state: dict,
    params: dict,
    planned_draw: Optional[dict[str, float]] = None,
) -> dict:
    """
    Generate a human-readable SPR status summary for the UI reserve panel.

    ``planned_draw`` is the per-facility daily drawdown the CURRENT routing plan
    proposes (see planned_draw_from_allocations). Without it this summary is a
    function of stored inventory alone, which never changes until the reserve is
    physically drawn — so the dashboard's reserve tile sat at one constant value
    no matter what was disrupted, which is precisely the moment it needs to move.
    What actually changes the instant a corridor closes is the RATE the reserve
    is committed to, and therefore how long it lasts. That is what days_to_floor
    reports, and what escalates the status flag.

    Returns:
        Dict with total_inventory_bbl, total_days_remaining, per_facility details,
        the planned draw, days_to_floor, and status (HEALTHY / WARNING / CRITICAL).
    """
    national_consumption = params.get("national_consumption_bbl_day", {}).get("value", 5_000_000)
    target_days = params.get("spr_target_days", {}).get("value", 9.5)
    floor_fraction = params.get("spr_structural_floor_fraction", {}).get("value", 0.10)
    projection_days = params.get("spr_draw_projection_days", {}).get("value", 90)
    planned_draw = planned_draw or {}

    total_inventory = sum(s["inventory_bbl"] for s in state["spr"].values())
    total_capacity = sum(s.get("storage_capacity_bbl", 0.0) for s in state["spr"].values())
    target_inventory = min(target_days * national_consumption, total_capacity)
    total_days = total_inventory / max(national_consumption, 1)

    per_facility = {}
    facility_days_to_floor = []
    for spr_id, s in state["spr"].items():
        storage = s.get("storage_capacity_bbl", 0.0)
        draw = float(planned_draw.get(spr_id, 0.0))
        # Barrels available above the structural floor, at the committed rate.
        headroom = max(0.0, s["inventory_bbl"] - storage * floor_fraction)
        days_to_floor = (headroom / draw) if draw > 0 else None
        if days_to_floor is not None:
            facility_days_to_floor.append(days_to_floor)
        per_facility[spr_id] = {
            "inventory_bbl": s["inventory_bbl"],
            "days_remaining": s["inventory_bbl"] / max(national_consumption, 1),
            "storage_capacity_bbl": storage,
            "fill_pct": s["inventory_bbl"] / max(storage, 1) * 100,
            "planned_draw_bbl_day": round(draw),
            "days_to_floor": round(days_to_floor, 1) if days_to_floor is not None else None,
        }

    days_to_floor = min(facility_days_to_floor) if facility_days_to_floor else None

    if days_to_floor is not None and days_to_floor < projection_days / 3:
        # The committed rate empties a cavern well inside the planning horizon;
        # that outranks a comfortable fill level.
        status = "CRITICAL"
    elif days_to_floor is not None and days_to_floor < projection_days:
        status = "WARNING"
    elif total_inventory >= target_inventory * 0.8:
        status = "HEALTHY"
    elif total_inventory >= target_inventory * 0.4:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return {
        "total_inventory_bbl": total_inventory,
        "total_days_remaining": round(total_days, 2),
        "target_days_requested": target_days,
        "target_days_effective": target_inventory / max(national_consumption, 1),
        "total_storage_capacity_bbl": total_capacity,
        "fill_pct": total_inventory / max(total_capacity, 1) * 100,
        "planned_draw_bbl_day": round(sum(planned_draw.values())),
        "days_to_floor": round(days_to_floor, 1) if days_to_floor is not None else None,
        "planning_horizon_days": projection_days,
        "status": status,
        "per_facility": per_facility,
    }
