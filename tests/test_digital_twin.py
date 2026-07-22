import copy
import json
from pathlib import Path

import pytest

from graph_engine.build_graph import load_graph
from graph_engine.digital_twin import build_initial_state, run_digital_twin
from graph_engine.reserve_optimizer import (
    compute_refill_allocation,
    get_spr_status_summary,
)


DATA_DIR = Path(__file__).parent.parent / "data"


def _graph_and_params():
    graph, _, _ = load_graph(DATA_DIR)
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    return graph, params


def test_baseline_twin_keeps_inventory_and_fulfillment_stable():
    """Steady-state cargoes must prevent a fictional baseline stockout."""
    graph, params = _graph_and_params()
    snapshots = run_digital_twin(graph, {}, params, horizon_days=10)

    assert all(snapshot["gap_bbl_day"] == 0 for snapshot in snapshots)
    assert all(snapshot["fulfillment_pct_overall"] == 100.0 for snapshot in snapshots)
    assert all(snapshot["spr_draw_bbl_day"] == 0 for snapshot in snapshots)
    # Cover must hold at whatever the data says it starts at: steady-state arrivals
    # replace exactly what is consumed. Pinning a literal day-count would only
    # re-assert the inventory constant, not the conservation property.
    initial_cover = (
        graph.nodes["ref_jamnagar_out"]["inventory_bbl"]
        / graph.nodes["ref_jamnagar_out"]["consumption_rate_bbl_day"]
    )
    assert all(
        snapshot["refineries"]["ref_jamnagar_out"]["days_of_cover"] == pytest.approx(initial_cover)
        for snapshot in snapshots
    )


def test_disruption_uses_pipeline_arrivals_before_degrading_fulfillment():
    """A closure affects new dispatches, not cargoes already in transit.

    The property is the ORDER of events: supply holds while in-transit cargoes and
    stored crude last, then degrades. When it degrades depends on how many days of
    cover the data gives each refinery, so the test derives that rather than
    asserting a fixed day index.
    """
    graph, params = _graph_and_params()
    min_cover = min(
        (data["inventory_bbl"] or 0) / (data["consumption_rate_bbl_day"] or 1)
        for _, data in graph.nodes(data=True)
        if data.get("type") == "refinery_out"
    )
    horizon = int(min_cover) + 30
    snapshots = run_digital_twin(graph, {"chk_hormuz": 0.0}, params, horizon_days=horizon)

    assert snapshots[0]["fulfillment_pct_overall"] == 100.0
    assert snapshots[0]["arrivals_bbl_day"] == snapshots[0]["total_demand_bbl_day"]

    first_gap = next((s["day"] for s in snapshots if s["gap_bbl_day"] > 0), None)
    assert first_gap is not None, "a full Hormuz closure must eventually bite"
    assert first_gap >= min_cover, (
        f"supply degraded on day {first_gap} despite {min_cover:.1f} days of stored cover — "
        "the buffer is being ignored"
    )
    assert snapshots[-1]["fulfillment_pct_overall"] < 100.0


def test_spr_status_uses_explicit_physical_capacity_and_draw_limits():
    """Facility fill and draw rate must be bounded by real modelled limits."""
    graph, params = _graph_and_params()
    state = build_initial_state(graph)
    status = get_spr_status_summary(state, params)

    assert status["total_storage_capacity_bbl"] == 39_200_000
    assert 72.0 < status["fill_pct"] < 74.0
    assert status["per_facility"]["spr_vizag"]["fill_pct"] < 100.0


def test_reserve_draw_depletes_the_caverns_and_stops_at_empty():
    """Reserve barrels leave storage when the plan ships them.

    The routing solve owns the drawdown decision and the simulation executes it,
    so barrels shipped from a cavern and barrels removed from its stock have to
    agree. Two independent views of the reserve would let a sustained closure
    ship crude that never leaves storage.
    """
    graph, params = _graph_and_params()
    snapshots = run_digital_twin(graph, {"chk_hormuz": 0.0}, params, horizon_days=60)

    stock = [sum(f["inventory_bbl"] for f in s["spr"].values()) for s in snapshots]
    drawn = sum(s["spr_draw_bbl_day"] for s in snapshots)

    assert drawn > 0, "a full Hormuz closure must call on the reserve"
    assert stock[-1] < stock[0], "reserve stock must fall while it is being drawn"
    # Each snapshot records stock AFTER that day's draw, so day 0's own draw has
    # already left storage by the time the first reading is taken.
    opening = stock[0] + snapshots[0]["spr_draw_bbl_day"]
    assert opening - stock[-1] == pytest.approx(drawn, rel=1e-6), (
        "barrels shipped from the reserve must equal barrels removed from storage"
    )
    assert all(a >= b for a, b in zip(stock, stock[1:])), "stock must never rise mid-drawdown"
    for snapshot in snapshots:
        for facility in snapshot["spr"].values():
            assert facility["inventory_bbl"] >= 0.0


def test_refill_cannot_exceed_physical_capacity():
    """Refill targets are capped by aggregate and facility storage capacity."""
    graph, params = _graph_and_params()
    state = build_initial_state(graph)
    for facility in state["spr"].values():
        facility["inventory_bbl"] = facility["storage_capacity_bbl"] - 100.0

    refills = compute_refill_allocation(
        state, params, surplus_bbl_day=1_000_000, current_corridor_risk=0.0
    )
    assert all(refill <= 100.0 for refill in refills.values())
