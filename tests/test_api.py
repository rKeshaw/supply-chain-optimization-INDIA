"""API endpoint tests without the deprecated synchronous TestClient lifecycle."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agents.schema import Event
from api import main
from graph_engine.build_graph import compute_baseline, load_graph
from graph_engine.digital_twin import build_initial_state
from graph_engine.routing import deliverable_state, avg_cost_per_bbl, avg_landed_cost_per_bbl

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(autouse=True)
def initialized_app_state():
    """Populate the same state as application lifespan for direct endpoint tests.

    Mirrors api/main.py's lifespan() APP_STATE.update(...) shape exactly —
    including the locks and caches added after this fixture was first written
    (state_lock, n1_ranking, ws_clients, live_*) — so a real endpoint call
    doesn't KeyError on a key only the live app's startup used to set.
    """
    previous = dict(main.APP_STATE)
    data_dir = Path(__file__).parent.parent / "data"
    graph, nodes, edges = load_graph(data_dir)
    params = json.loads((data_dir / "parameters.json").read_text(encoding="utf-8"))
    _d = deliverable_state(graph, params)
    main.APP_STATE.clear()
    main.APP_STATE.update({
        "G_baseline": graph,
        "G_current": graph,
        "nodes": nodes,
        "edges": edges,
        "params": params,
        "baseline": compute_baseline(graph),
        "baseline_deliverable": _d,
        "baseline_cost_route": {"path_allocations": _d["path_allocations"], "total_volume": _d["flow_value"]},
        "baseline_avg_cost_per_bbl": avg_cost_per_bbl(
            {"path_allocations": _d["path_allocations"], "total_volume": _d["flow_value"]}
        ),
        "sim_state": build_initial_state(graph),
        "replay_index": 0,
        "replay_log": [],
        "recent_events": [],
        "live_signal_log": [],
        "live_last_poll_at": None,
        "current_scenario": None,
        "scenario_result": None,
        "last_brief": None,
        "n1_ranking": None,
        "ws_clients": [],
        "broadcast_fn": main.broadcast_update,
        "live_task": None,
        "live_stop_event": None,
        "live_toggle_lock": asyncio.Lock(),
        "state_lock": asyncio.Lock(),
    })
    yield
    main.APP_STATE.clear()
    main.APP_STATE.update(previous)


def _modelled_demand():
    """Total refinery throughput the network has to supply, read from the data."""
    nodes = json.loads((DATA_DIR / "nodes.json").read_text(encoding="utf-8"))
    return sum(n.get("consumption_rate_bbl_day") or 0
               for n in nodes if n.get("type") == "refinery_out")


def test_api_health():
    """Baseline must supply every modelled refinery in full, whatever the network size."""
    data = asyncio.run(main.health())
    assert data["status"] == "ok"
    assert data["baseline_flow_bbl_day"] == _modelled_demand()


def test_api_graph_state_includes_actual_flow():
    """Every node in the data file must reach the client, and flow must be real."""
    nodes = json.loads((DATA_DIR / "nodes.json").read_text(encoding="utf-8"))
    data = asyncio.run(main.graph_state())
    assert len(data["nodes"]) == len(nodes)
    assert any(edge["flow_bbl_day"] > 0 for edge in data["edges"])


def test_api_baseline_and_vulnerability():
    baseline = asyncio.run(main.graph_baseline())
    vulnerability = asyncio.run(main.graph_vulnerability())
    assert baseline["flow_value_bbl_day"] == _modelled_demand()
    assert vulnerability["hhi"]["hhi_value"] > 0.0
    # The contingency ranking must put a genuine single point of failure on top,
    # not an element the network can absorb.
    top = vulnerability["n1_ranking"][0]
    assert top["vulnerability_index"] > 0.0
    assert top["node_id"] == "chk_hormuz"


def test_api_scenario_apply_returns_routing_and_flow_state():
    # Full closure still causes a genuine volume shortfall — Hormuz-transiting
    # supply (~60%+ of Gulf sources) cannot be fully replaced by non-Hormuz
    # alternatives (Oman, Cape-route suppliers) alone.
    data = asyncio.run(main.apply_scenario_endpoint(main.ScenarioRequest(scenario_id="hormuz_full")))
    assert data["disrupted_flow_bbl_day"] < data["baseline_flow_bbl_day"]
    assert data["flow_loss_pct"] > 10.0, "closing the strait India depends on most must bite"
    assert data["routing"]["pareto_routes"]["cost_optimal"]["path_allocations"]
    assert any(edge["flow_bbl_day"] > 0 for edge in data["graph_state"]["edges"])


def test_hormuz_partial_mostly_absorbed_by_diversified_sources_but_costs_more():
    """Diversification (Oman bypass, Cape-route suppliers, Russian streams)
    absorbs the BULK of a -60% Hormuz cut, but neither fully nor for free.

    Deliberately does NOT assert an exact delivered volume. That figure is a
    direct function of how much spare source capacity the model is given, which
    is a modelling assumption rather than a property of the system: this
    assertion had to be rewritten twice while that assumption was being
    corrected (fully absorbed at 2.33x supply slack, a ~3% gap at 1.20x, fully
    absorbed again once capacity was restored to export availability). An
    assertion that flips whenever a parameter under review changes is testing
    the parameter, not the behaviour.

    What IS stable, and worth locking down, is the economics: rerouting around a
    degraded Hormuz must cost materially more per barrel. That holds at any
    slack level, because the alternatives are genuinely further away.
    """
    baseline = asyncio.run(main.apply_scenario_endpoint(main.ScenarioRequest(custom={"chk_hormuz": 1.0})))
    disrupted = asyncio.run(main.apply_scenario_endpoint(main.ScenarioRequest(scenario_id="hormuz_partial")))

    delivered = disrupted["disrupted_flow_bbl_day"]
    assert delivered > 2574000 * 0.90, "diversification must absorb the bulk of a partial cut"
    base_route = baseline["routing"]["pareto_routes"]["cost_optimal"]
    dis_route = disrupted["routing"]["pareto_routes"]["cost_optimal"]
    # Priced on LANDED cost, not freight. Freight alone misses the larger half of
    # what a reroute costs: switching to a pricier grade, and the scarcity premium
    # a disruption puts on every barrel still deliverable.
    assert avg_cost_per_bbl(dis_route) > avg_cost_per_bbl(base_route), "freight must rise"
    landed_premium = avg_landed_cost_per_bbl(dis_route) - avg_landed_cost_per_bbl(base_route)
    assert landed_premium > 1.0, f"landed cost must rise materially, got ${landed_premium:.2f}/bbl"


def test_api_twin_simulate_baseline_has_no_gap():
    data = asyncio.run(main.simulate_twin(main.SimulateRequest(horizon_days=5)))
    assert len(data["snapshots"]) == 5
    assert data["summary"]["days_with_gap"] == 0


def test_api_twin_uses_current_graph_when_requested():
    asyncio.run(main.apply_scenario_endpoint(main.ScenarioRequest(scenario_id="hormuz_partial")))
    data = asyncio.run(main.simulate_twin(main.SimulateRequest(use_current_graph=True, horizon_days=5)))
    assert data["scenario"]["chk_hormuz"] == 0.4


def test_replay_runs_headlines_through_live_extraction(monkeypatch):
    """Replay must send the article text through the real extraction agent.

    It previously passed a pre-built Event from the timeline's
    expected_extraction block, which short-circuited extraction_agent.parse
    entirely (orchestration.py's `event_override or parse(...)`), so the
    severity, confidence and affected corridor a viewer saw were read from a
    file rather than decided by the model. This asserts the bypass is gone:
    parse() is called, and its output is what reaches the graph.

    The LLM itself is stubbed so the suite stays hermetic and offline —
    tests/test_extraction_accuracy.py is where real model output is scored
    against the timeline's labels.
    """
    captured = {}

    def fake_parse(raw_text, source_override=None, timestamp_override=None):
        captured["raw_text"] = raw_text
        return Event(
            id="evt-live-parse",
            source=source_override or "TEST",
            timestamp=timestamp_override or datetime.now(timezone.utc),
            entity="Strait of Hormuz",
            event_type="closure",
            severity=0.9,
            confidence=0.9,
            affected_graph_element="chk_hormuz",
            justification="stubbed extraction",
        )

    monkeypatch.setattr("agents.extraction_agent.parse", fake_parse)

    result = asyncio.run(main.run_replay())

    assert result["replay_step"] == 1
    # The event carries the model's id, not the timeline record's "evt_001".
    assert result["pipeline_result"]["event"]["id"] == "evt-live-parse"
    # The real headline and body text were what got parsed.
    assert "airstrikes on Iran" in captured["raw_text"]
    assert main.APP_STATE["replay_log"][0]["result_summary"]["ingestion_mode"] == "live_extraction_replay"
