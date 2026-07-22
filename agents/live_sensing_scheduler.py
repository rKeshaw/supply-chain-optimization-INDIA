"""Background live-sensing loop — the "continuously, not weekly" piece.

Polls all live adapters (sanctions registry, news, marine weather) on an
interval and feeds whatever they find through the same process_signal()
pipeline that /api/signal and /api/replay/run use — a live-detected event is
handled by the exact same orchestration, not a separate code path.

Off by default. The project's own design principle (plan.md, and the curated
crisis_timeline.json replay it argues for) is that a live feed is an optional
enhancement layered on top of a reliable, deterministic demo default — not a
replacement for it. Enable explicitly via LIVE_INGESTION_ENABLED=true.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

LIVE_INGESTION_ENABLED = os.environ.get("LIVE_INGESTION_ENABLED", "false").strip().lower() == "true"
LIVE_POLL_INTERVAL_S = float(os.environ.get("LIVE_POLL_INTERVAL_S", "1800"))  # 30 min default


async def _poll_sanctions(app_state: dict) -> list[dict]:
    from agents.sanctions_adapter import check_sanctions_updates

    try:
        events = await asyncio.to_thread(check_sanctions_updates)
    except Exception as e:
        logger.error(f"Live sensing: sanctions adapter failed: {e}")
        return []

    results = []
    for event in events:
        results.append(await _run_event_through_pipeline(app_state, event, origin="live_sanctions"))
    return results


async def _poll_news(app_state: dict) -> list[dict]:
    from agents.news_adapter import check_news_signals
    from agents.orchestration import process_signal

    try:
        signals = await asyncio.to_thread(check_news_signals)
    except Exception as e:
        logger.error(f"Live sensing: news adapter failed: {e}")
        return []

    results = []
    for sig in signals:
        try:
            # Off the event loop: process_signal is a blocking call (LLM
            # network round trips, a retry sleep in extraction_agent), unlike
            # every REST handler's asyncio.to_thread(process_signal, ...) —
            # called directly here it would freeze /ws/live and every
            # concurrent request for the length of the pipeline.
            result = await asyncio.to_thread(
                process_signal,
                raw_text=sig["raw_text"],
                G_current=app_state.get("G_current", app_state["G_baseline"]),
                sim_state=app_state["sim_state"],
                params=app_state["params"],
                source_override=sig.get("source"),
                timestamp_override=sig.get("timestamp"),
            )
            await _apply_pipeline_result(app_state, result, origin="live_news", label=sig["raw_text"])
            results.append(result)
        except Exception as e:
            logger.error(f"Live sensing: process_signal failed for news item: {e}")
    return results


async def _poll_weather(app_state: dict) -> list[dict]:
    from agents.weather_adapter import check_weather_disruptions

    try:
        events = await asyncio.to_thread(check_weather_disruptions)
    except Exception as e:
        logger.error(f"Live sensing: weather adapter failed: {e}")
        return []

    results = []
    for event in events:
        results.append(await _run_event_through_pipeline(app_state, event, origin="live_weather"))
    return results


async def _run_event_through_pipeline(app_state: dict, event, origin: str) -> dict:
    from agents.orchestration import process_signal

    result = await asyncio.to_thread(
        process_signal,
        raw_text=event.justification,
        G_current=app_state.get("G_current", app_state["G_baseline"]),
        sim_state=app_state["sim_state"],
        params=app_state["params"],
        event_override=event,
    )
    await _apply_pipeline_result(app_state, result, origin=origin, label=event.justification)
    return result


async def _apply_pipeline_result(app_state: dict, result: dict, origin: str, label: str) -> None:
    updated_graph = result.pop("_updated_graph", None)
    if result.get("recompute_triggered") and updated_graph is not None:
        # Same lock and cache-invalidation every REST write path takes (see
        # api/main.py's apply_scenario_endpoint / process_signal_endpoint /
        # run_replay) — without it, a live-sensed update could race a
        # concurrent request's read-modify-write of G_current, and the N-1
        # vulnerability ranking would keep serving the pre-event topology
        # indefinitely, since it is only recomputed when the cache is cleared.
        state_lock = app_state.get("state_lock")
        if state_lock is not None:
            async with state_lock:
                app_state["G_current"] = updated_graph
                app_state["n1_ranking"] = None
        else:
            app_state["G_current"] = updated_graph
            app_state["n1_ranking"] = None

    app_state.setdefault("live_signal_log", []).append({
        "origin": origin,
        "label": label,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "recompute_triggered": result.get("recompute_triggered"),
        # "unrelated" (genuinely irrelevant, e.g. a keyword-search false
        # positive) vs "below_threshold" (a real but minor signal) vs
        # "relevant" (triggered a full recompute) — lets the UI distinguish
        # noise from signal instead of listing everything identically.
        "reason": result.get("reason", "relevant" if result.get("recompute_triggered") else "below_threshold"),
        "latency_ms": result.get("latency_ms") or result.get("latency", {}).get("total_pipeline_ms"),
    })
    # Cap the in-memory log so a long-running process doesn't grow unbounded.
    app_state["live_signal_log"] = app_state["live_signal_log"][-200:]

    # Push to any connected /ws/live clients — best-effort; a missing or
    # failing broadcast function must never break signal processing itself.
    broadcast_fn = app_state.get("broadcast_fn")
    if broadcast_fn is not None:
        try:
            await broadcast_fn({
                "kind": "live_signal", "origin": origin, "label": label,
                "recompute_triggered": result.get("recompute_triggered"),
            })
        except Exception as e:
            logger.warning(f"Live sensing: broadcast failed (non-critical): {e}")


async def run_poll_cycle(app_state: dict) -> dict:
    """Run one full poll of every live adapter. Exposed standalone so both the
    background loop and a manual /api/live/poll-now endpoint share one path."""
    sanctions_results = await _poll_sanctions(app_state)
    news_results = await _poll_news(app_state)
    weather_results = await _poll_weather(app_state)
    app_state["live_last_poll_at"] = datetime.now(timezone.utc).isoformat()
    return {
        "sanctions_events": len(sanctions_results),
        "news_signals": len(news_results),
        "weather_events": len(weather_results),
    }


async def live_sensing_loop(app_state: dict, stop_event: asyncio.Event) -> None:
    """The background task itself. Runs until stop_event is set (toggle-off or app shutdown)."""
    logger.info(f"Live sensing loop started (interval={LIVE_POLL_INTERVAL_S}s).")
    while not stop_event.is_set():
        try:
            summary = await run_poll_cycle(app_state)
            logger.info(f"Live sensing poll complete: {summary}")
        except Exception as e:
            # A single bad cycle must never kill the loop.
            logger.error(f"Live sensing loop: unexpected error in poll cycle: {e}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=LIVE_POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            pass  # normal: interval elapsed, loop again
    logger.info("Live sensing loop stopped.")


# ---------------------------------------------------------------------------
# Runtime start/stop — lets an operator (or the UI toggle) flip live sensing
# on or off mid-session instead of only at process startup via the env var.
# The env var still picks the default at boot; these functions are what
# POST /api/live/enable and /api/live/disable in api/main.py call.
# ---------------------------------------------------------------------------

def is_running(app_state: dict) -> bool:
    task = app_state.get("live_task")
    return task is not None and not task.done()


async def start_live_loop(app_state: dict) -> bool:
    """Start the background loop if it isn't already running. Idempotent."""
    if is_running(app_state):
        return True
    stop_event = asyncio.Event()
    app_state["live_stop_event"] = stop_event
    app_state["live_task"] = asyncio.create_task(live_sensing_loop(app_state, stop_event))
    return True


async def stop_live_loop(app_state: dict) -> bool:
    """Stop the background loop if it is running, and wait for it to exit. Idempotent."""
    stop_event = app_state.get("live_stop_event")
    task = app_state.get("live_task")
    if stop_event is None or task is None:
        return True
    stop_event.set()
    await task
    app_state["live_task"] = None
    app_state["live_stop_event"] = None
    return True
