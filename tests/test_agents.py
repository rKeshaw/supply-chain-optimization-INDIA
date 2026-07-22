import json
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime, timezone

from agents.schema import Event, resolve_entity
from agents.extraction_agent import parse
from agents.policy_critic_agent import verify, get_re_solve_overrides
from agents.explainer_agent import summarize

@patch("agents.extraction_agent.call_llm")
def test_extraction_agent(mock_call):
    """Test that extraction agent parses valid JSON, resolves aliases, and handles retries."""
    # Mock successful parse on first attempt
    mock_call.return_value = json.dumps({
        "id": "evt-test-1",
        "source": "Reuters",
        "timestamp": "2025-04-14T08:30:00Z",
        "entity": "Strait of Hormuz",
        "event_type": "capacity_reduction",
        "severity": 0.8,
        "confidence": 0.9,
        "affected_graph_element": "Strait of Hormuz",
        "justification": "US-Iran standoff threat near Gulf."
    })
    
    event = parse("Iran warns US military vessels in Gulf near Strait of Hormuz.")
    
    assert event is not None
    assert event.id == "evt-test-1"
    assert event.source == "Reuters"
    # Alias lookup should resolve "Strait of Hormuz" -> "chk_hormuz"
    assert event.affected_graph_element == "chk_hormuz"
    assert event.severity == 0.8
    assert event.confidence == 0.9

@patch("agents.extraction_agent.call_llm")
def test_extraction_agent_retry(mock_call):
    """Test that extraction agent retries once if first attempt fails validation."""
    # First attempt: invalid JSON
    # Second attempt: valid JSON
    mock_call.side_effects = [
        "invalid-json-garbage",
        json.dumps({
            "id": "evt-test-2",
            "source": "Bloomberg",
            "timestamp": "2025-01-18T06:30:00Z",
            "entity": "Red Sea",
            "event_type": "capacity_reduction",
            "severity": 0.5,
            "confidence": 0.8,
            "affected_graph_element": "chk_bab",
            "justification": "Houthi resumed attacks."
        })
    ]
    
    # Configure mock side effect
    mock_call.side_effect = mock_call.side_effects
    
    event = parse("Houthi forces announce resumed attacks on Red Sea shipping.")
    
    assert event is not None
    assert event.id == "evt-test-2"
    assert event.affected_graph_element == "chk_bab"
    assert mock_call.call_count == 2

def test_policy_critic():
    """Verify that Policy Critic detects SPR floor and corridor concentration violations."""
    # Mock data violating SPR safety floor (fill at 8% < 10%)
    # A reserve is breached by sustaining a draw, not by its fill on any one
    # morning, so the violation is expressed as a drawdown the plan proposes:
    # 1.0 Mb held above a 0.5 Mb floor, drawn at 85 kb/day, reaches the floor in
    # under 6 days — well inside the planning horizon.
    routing_result = {
        "cost_optimal": {
            "feasible": True,
            "total_volume": 2574000,
            "routing_summary": [
                {"from": "src_saudi", "to": "chk_hormuz", "volume_bbl_day": 800000},
                {"from": "chk_cog", "to": "ref_jamnagar_in", "volume_bbl_day": 10000},
            ],
            "path_allocations": [{
                "source_id": "spr_vizag",
                "is_spr": True,
                "refinery_in": "ref_vizag_in",
                "refinery_out": "ref_vizag_out",
                "grade": "SOUR",
                "path": ["spr_vizag", "ref_vizag_in", "ref_vizag_out"],
                "volume_bbl_day": 85000,
            }],
        }
    }

    spr_state = {
        "total_days_remaining": 7.8,
        "status": "WARNING",
        "per_facility": {
            # planned_draw_bbl_day and days_to_floor mirror what
            # reserve_optimizer.get_spr_status_summary actually computes for
            # this scenario (500k bbl headroom above the 10% floor, drawn at
            # 85k bbl/day = 5.9 days to floor) — the policy critic now reads
            # these precomputed fields directly rather than re-deriving them.
            "spr_vizag": {
                "inventory_bbl": 1_000_000,
                "storage_capacity_bbl": 5_000_000,
                "planned_draw_bbl_day": 85000,
                "days_to_floor": 5.9,
            },
        },
    }

    graph_state = {}
    params = {}
    
    with patch("agents.policy_critic_agent.call_llm") as mock_llm:
        mock_llm.return_value = json.dumps({
            "violations": [
                {
                    "rule_id": "rule_spr_floor_draw",
                    "violated": True,
                    "explanation": "spr_vizag fill at 8.5% is below structural safety floor of 10%.",
                    "suggested_correction": "Reduce SPR vizag release rate."
                }
            ],
            "all_clear": False,
            "re_solve_required": True,
            "corrected_constraints": {}
        })
        
        result = verify(routing_result, spr_state, graph_state, params)
        
        assert result["all_clear"] is False
        assert result["re_solve_required"] is True
        assert len(result["violations"]) == 1
        assert result["violations"][0]["rule_id"] == "rule_spr_floor_draw"


def test_policy_critic_detects_invalid_refinery_grade_mix():
    """The critic must reject a SWEET allocation to SOUR-only Paradip."""
    routing_result = {
        "cost_optimal": {
            "routing_summary": [],
            "path_allocations": [{
                "source_id": "src_nigeria",
                "refinery_in": "ref_paradip_in",
                "refinery_out": "ref_paradip_out",
                "grade": "SWEET",
                "volume_bbl_day": 300000,
            }],
        }
    }
    spr_state = {"per_facility": {}}

    with patch("agents.policy_critic_agent.call_llm") as mock_llm:
        mock_llm.return_value = json.dumps({
            "violations": [{"rule_id": "rule_sour_ratio_paradip", "violated": True}],
            "all_clear": False,
            "re_solve_required": True,
            "corrected_constraints": {"ref_paradip_in": {"max_sweet_fraction": 0.3}},
        })
        result = verify(routing_result, spr_state, {}, {})

    assert result["re_solve_required"] is True
    assert result["violations"][0]["rule_id"] == "rule_sour_ratio_paradip"


def test_policy_critic_exposes_only_validated_solver_overrides():
    """Only a named numeric correction may affect a routing re-solve."""
    overrides = get_re_solve_overrides({
        "corrected_constraints": {
            "rule_cape_congestion_cap": {"max_fraction_of_total": "0.45"},
            "free_text": "close all routes",
        }
    })
    assert overrides == {"max_cape_fraction_of_total": 0.45}

@patch("agents.scenario_agent.call_llm")
@patch("agents.policy_critic_agent.call_llm")
@patch("agents.explainer_agent.call_llm")
def test_process_signal_actually_mutates_graph_state(mock_explainer_llm, mock_critic_llm, mock_scenario_llm):
    """process_signal must actually mutate graph state when it reports doing so.

    Reporting recompute_triggered while leaving the graph untouched is a silent
    failure, and it is only visible to a test that runs the real pipeline rather
    than a mock of it.
    """
    from pathlib import Path
    from graph_engine.build_graph import load_graph
    from graph_engine.digital_twin import build_initial_state
    from agents.orchestration import process_signal

    mock_explainer_llm.return_value = None  # exercises each agent's real fallback path
    mock_critic_llm.return_value = None
    mock_scenario_llm.return_value = None

    G, _, _ = load_graph(Path(__file__).parent.parent / "data")
    params = json.loads((Path(__file__).parent.parent / "data" / "parameters.json").read_text())
    sim_state = build_initial_state(G)

    event = Event(
        id="evt-regression-1", source="TEST", timestamp=datetime.now(timezone.utc),
        entity="Iran", location=None, event_type="sanction",
        severity=0.5, confidence=0.95, affected_graph_element="src_iran",
        justification="regression test event",
    )

    result = process_signal(
        raw_text=event.justification, G_current=G, sim_state=sim_state, params=params,
        event_override=event,
    )

    assert result["recompute_triggered"] is True
    updated_graph = result["_updated_graph"]
    assert updated_graph is not G  # must be a distinct object, not an untouched pass-through
    assert updated_graph.nodes["src_iran"]["risk_score"] > 0.0
    assert updated_graph.nodes["src_iran"]["openness"] < 1.0
    assert G.nodes["src_iran"]["risk_score"] == 0.0  # original graph must stay untouched


@patch("agents.explainer_agent.call_llm")
def test_explainer_no_hallucination(mock_llm):
    """Test that Explainer parses successfully and falls back to structured brief on LLM failure."""
    # LLM failure case
    mock_llm.return_value = None
    
    event = Event(
        id="evt-1",
        source="Reuters",
        timestamp=datetime.now(timezone.utc),
        entity="Strait of Hormuz",
        event_type="capacity_reduction",
        severity=0.8,
        confidence=0.9,
        affected_graph_element="chk_hormuz",
        justification="closure threat"
    )
    
    economic_impact = {
        "shortfall_pct": 12.5,
        "crude_price_change_pct": 250.0,
        "retail_price_change_pct": 180.0,
        "gdp_drag_pct": 10.0,
        "power_sector_stress": "ELEVATED",
        "gap_bbl_day": 321750
    }
    
    spr_status = {
        "total_days_remaining": 6.2,
        "status": "WARNING",
        "total_inventory_bbl": 31000000,
        "fill_pct": 58.0
    }
    
    # Run explainer summarization
    brief = summarize(
        event=event,
        validated_routing={},
        economic_impact=economic_impact,
        spr_status=spr_status,
        critic_result={},
        graph_state={}
    )
    
    # Should fall back gracefully to a structured summary containing exact values
    assert brief is not None
    assert brief["_fallback"] is True
    assert brief["numbers_used"]["shortfall_pct"] == 12.5
    assert brief["numbers_used"]["crude_price_change_pct"] == 250.0
    assert brief["numbers_used"]["spr_days_remaining"] == 6.2


# ---------------------------------------------------------------------------
# Policy Critic — the re-solve loop and the rules that drive it.
#
# Both halves regressed together and must be tested together: the four code
# rules were all structurally unreachable, so verify() always short-circuited
# to all_clear and the LLM critic never ran. That masked an unbounded loop in
# the LangGraph edge — meaning fixing the rules ALONE would have converted a
# dead feature into a hanging one.
# ---------------------------------------------------------------------------

def _critic_inputs(scenario):
    """Real solver output for a scenario, shaped as the critic consumes it."""
    from pathlib import Path
    from graph_engine.build_graph import load_graph, get_graph_state_json
    from graph_engine.routing import compute_pareto_routes
    from graph_engine.disruption import apply_scenario
    from graph_engine.digital_twin import build_initial_state
    from graph_engine.reserve_optimizer import get_spr_status_summary

    data_dir = Path(__file__).parent.parent / "data"
    G, _, _ = load_graph(data_dir)
    params = json.loads((data_dir / "parameters.json").read_text(encoding="utf-8"))
    demand = {n: d.get("consumption_rate_bbl_day", 0)
              for n, d in G.nodes(data=True) if d.get("type") == "refinery_out"}
    Gd = apply_scenario(G, scenario) if scenario else G
    routing = compute_pareto_routes(Gd, demand, params)
    spr = get_spr_status_summary(build_initial_state(G), params)
    return routing, spr, get_graph_state_json(Gd), params


def test_diversification_breach_reported_by_solver_becomes_a_violation():
    """A ceiling the solver chose to exceed must reach the critic as a violation.

    The critic reads the breach the solver reports rather than recomputing a
    share against its own denominator, so there is a single account of what a
    plan did.
    """
    from agents.policy_critic_agent import _run_code_checks

    routing, spr, graph_state, params = _critic_inputs({"src_russia_urals": 0.0})
    breaches = routing["cost_optimal"].get("policy_breaches") or {}
    assert breaches, "expected this scenario to force a diversification breach"

    fired = _run_code_checks(routing, spr, graph_state, params)
    reported = {v.get("constraint") for v in fired}
    assert set(breaches) <= reported, "a solver breach went unreported by the critic"
    for violation in fired:
        if violation.get("constraint") in breaches:
            assert violation["excess_bbl_day"] == round(breaches[violation["constraint"]])


def test_baseline_breach_is_marginal_and_prefers_the_ceiling_over_the_reserve():
    """Undisrupted, the plan sits fractionally over one supplier ceiling.

    Vizag cannot be filled inside every ceiling at once, so something has to
    give. Exceeding a diversification guideline by half a percentage point is
    the cheapest of the three available concessions, ahead of drawing the
    national reserve daily and far ahead of leaving a refinery short. The
    breach is small, it is confined to one ceiling, and it is reported.
    """
    from agents.policy_critic_agent import _run_code_checks

    routing, spr, graph_state, params = _critic_inputs({})
    cost = routing["cost_optimal"]
    breaches = cost.get("policy_breaches") or {}

    assert set(breaches) == {"supplier_group:russia"}
    national_demand = sum(
        n.get("consumption_rate_bbl_day") or 0
        for n in graph_state["nodes"] if n.get("type") == "refinery_out"
    )
    assert breaches["supplier_group:russia"] / national_demand < 0.01
    assert not any(a.get("is_spr") for a in cost["path_allocations"]), (
        "the reserve must not be drawn during normal operations"
    )
    assert [v["constraint"] for v in _run_code_checks(routing, spr, graph_state, params)
            if v.get("constraint")] == ["supplier_group:russia"]


def test_non_binding_rule_is_skipped():
    """max_sour_fraction of 1.0 cannot be exceeded; it must not be evaluated."""
    from agents.policy_critic_agent import POLICY_RULES

    jamnagar = next(r for r in POLICY_RULES if r["id"] == "rule_sour_ratio_jamnagar")
    assert jamnagar["binding"] is False


@pytest.mark.parametrize("corrected,expected_solves", [
    ({}, 0),                                          # nothing actionable -> report, don't spin
    ({"max_cape_fraction_of_total": 0.45}, 1),        # actionable -> exactly one re-solve
])
def test_policy_critic_resolve_loop_terminates(corrected, expected_solves):
    """A critic that keeps reporting a violation must not loop forever.

    The guard lives on the pipeline state rather than on critic_result, which
    policy_critic_node replaces wholesale on each pass and would therefore
    discard, leaving the pipeline to run to its recursion limit. Here the critic
    reports the same violation on every call, which is the worst case, and the
    pipeline must still finish.
    """
    from pathlib import Path
    from graph_engine.build_graph import load_graph
    from graph_engine.digital_twin import build_initial_state
    import agents.policy_critic_agent as pc
    import agents.explainer_agent as ex
    import agents.scenario_agent as sa
    import agents.orchestration as orch

    data_dir = Path(__file__).parent.parent / "data"
    G, _, _ = load_graph(data_dir)
    params = json.loads((data_dir / "parameters.json").read_text(encoding="utf-8"))

    calls = {"n": 0}

    def always_violating(**_kwargs):
        calls["n"] += 1
        return {
            "violations": [{"rule_id": "rule_cape_congestion_cap"}],
            "all_clear": False,
            "re_solve_required": True,
            "corrected_constraints": corrected,
        }

    event = Event(
        id="loop-test", source="TEST", timestamp=datetime.now(timezone.utc),
        entity="Strait of Hormuz", event_type="closure", severity=0.9,
        confidence=0.95, affected_graph_element="chk_hormuz", justification="test",
    )

    with patch.object(pc, "verify", always_violating), \
         patch.object(ex, "summarize", lambda **k: {"headline": "stub"}), \
         patch.object(sa, "generate_hypotheses", lambda **k: []):
        result = orch.process_signal(
            "test", G, build_initial_state(G), params, event_override=event
        )

    assert result["recompute_triggered"] is True
    # critic runs once per routing solve: the initial one, plus each re-solve
    assert calls["n"] == 1 + expected_solves
    assert result["policy_check"]["violations"], "violations must still reach the brief"
