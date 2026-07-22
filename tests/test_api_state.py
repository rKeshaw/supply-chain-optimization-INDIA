"""Focused API state tests that do not require an ASGI test-server lifecycle."""

import asyncio
import copy
from pathlib import Path

from api import main
from graph_engine.build_graph import load_graph


def test_signal_endpoint_persists_orchestration_graph_exactly(monkeypatch):
    """The API must not reconstruct a different state from event severity."""
    baseline, _, _ = load_graph(Path(__file__).parent.parent / "data")
    updated = copy.deepcopy(baseline)
    updated.nodes["chk_hormuz"]["openness"] = 0.28
    updated.nodes["chk_hormuz"]["risk_score"] = 0.72

    prior_state = dict(main.APP_STATE)
    main.APP_STATE.clear()
    main.APP_STATE.update({
        "G_baseline": baseline,
        "G_current": baseline,
        "sim_state": {},
        "params": {},
        "n1_ranking": None,
        "state_lock": asyncio.Lock(),
    })

    def fake_process_signal(**_kwargs):
        return {
            "recompute_triggered": True,
            "event": {"affected_graph_element": "chk_hormuz"},
            "graph_state": {"nodes": [], "edges": []},
            "_updated_graph": updated,
        }

    monkeypatch.setattr("agents.orchestration.process_signal", fake_process_signal)

    try:
        response = asyncio.run(main.process_signal_endpoint(main.SignalRequest(text="test")))
        assert "_updated_graph" not in response
        assert main.APP_STATE["G_current"] is updated
        assert main.APP_STATE["G_current"].nodes["chk_hormuz"]["openness"] == 0.28
    finally:
        main.APP_STATE.clear()
        main.APP_STATE.update(prior_state)
