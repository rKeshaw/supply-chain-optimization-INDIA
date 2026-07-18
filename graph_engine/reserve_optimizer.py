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


def optimize_spr_draw(
    state: dict,
    params: dict,
) -> dict[str, float]:
    """
    Compute SPR drawdown amounts for the current day across all SPR facilities.

    Policy: draw only as much as needed to maintain safety floor across all refineries.
    Draws are sourced from SPR facilities in order of proximity to under-served refineries.
    Total draw per facility is capped at its max_discharge_bbl_day.

    Args:
        state: Current simulation state (refineries, spr dicts).
        params: Parameters dict with spr_safety_floor_days and spr_max_discharge_rate.

    Returns:
        Dict mapping spr node IDs to draw amounts (bbl/day) for today.
        Returns {spr_id: 0} for all if no draw is needed.
    """
    floor_days = params.get("spr_safety_floor_days", {}).get("value", 3.0)
    draws: dict[str, float] = {spr_id: 0.0 for spr_id in state["spr"]}

    # State is expected to represent projected end-of-day inventory. Allocate
    # only to physically connected refineries and never draw more than their
    # remaining safety-floor deficit.
    remaining_need = {
        ref_id: max(0.0, (floor_days - ref.days_of_cover) * ref.consumption_rate_bbl_day)
        for ref_id, ref in state["refineries"].items()
        if ref.consumption_rate_bbl_day > 0
    }

    for spr_id, facility in state["spr"].items():
        facility_max = min(
            facility["inventory_bbl"], facility["max_discharge_bbl_day"]
        )
        connected = facility.get("connected_refineries", [])
        connected_need = sum(remaining_need.get(ref_id, 0.0) for ref_id in connected)
        draw = min(facility_max, connected_need)
        draws[spr_id] = draw

        if draw <= 0 or connected_need <= 0:
            continue
        for ref_id in connected:
            need = remaining_need.get(ref_id, 0.0)
            remaining_need[ref_id] = max(0.0, need - draw * need / connected_need)

    logger.debug(
        f"SPR draw: total={sum(draws.values()):,.0f} bbl/day across "
        f"{len([d for d in draws.values() if d > 0])} facilities. "
        f"Remaining connected safety-floor need is {sum(remaining_need.values()):,.0f} bbl/day."
    )

    return draws


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


def get_spr_status_summary(state: dict, params: dict) -> dict:
    """
    Generate a human-readable SPR status summary for the UI reserve panel.

    Returns:
        Dict with total_inventory_bbl, total_days_remaining, per_facility details,
        and status flag (HEALTHY / WARNING / CRITICAL).
    """
    national_consumption = params.get("national_consumption_bbl_day", {}).get("value", 5_000_000)
    target_days = params.get("spr_target_days", {}).get("value", 9.5)

    total_inventory = sum(s["inventory_bbl"] for s in state["spr"].values())
    total_capacity = sum(s.get("storage_capacity_bbl", 0.0) for s in state["spr"].values())
    target_inventory = min(target_days * national_consumption, total_capacity)
    total_days = total_inventory / max(national_consumption, 1)

    per_facility = {}
    for spr_id, s in state["spr"].items():
        per_facility[spr_id] = {
            "inventory_bbl": s["inventory_bbl"],
            "days_remaining": s["inventory_bbl"] / max(national_consumption, 1),
            "storage_capacity_bbl": s.get("storage_capacity_bbl", 0.0),
            "fill_pct": s["inventory_bbl"] / max(s.get("storage_capacity_bbl", 0.0), 1) * 100,
        }

    if total_inventory >= target_inventory * 0.8:
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
        "status": status,
        "per_facility": per_facility,
    }
