"""
Extraction agent: parses raw news text into schema-valid Event objects.

Single-purpose: given text + schema, return JSON. Nothing else.
Schema validation happens before the event is passed to the graph engine.
On schema validation failure, retries once. If retry fails, returns None.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from agents.schema import Event, resolve_entity, known_element_ids, render_known_elements_prompt_block
from agents.llm_client import call_llm

logger = logging.getLogger(__name__)


_SYSTEM_INSTRUCTION = """You are an energy supply chain signal extraction engine.

Your only job is to extract a structured event from the given news text.
You must return a single JSON object matching the schema below, and nothing else.
Do not add commentary, explanation, or markdown.

EVENT SCHEMA:
{
  "id": "string — a unique identifier you generate",
  "source": "string — the news source or domain if identifiable, else 'UNKNOWN'",
  "timestamp": "ISO 8601 UTC datetime — use the article date if given, else current time",
  "entity": "string — NEVER null, even for unrelated events. The geographic entity, country, or corridor affected (exact name from article) when relevant; for an unrelated story, use its main subject instead (e.g. 'India cricket team', 'Nvidia') — always a string.",
  "location": "string or null — more specific location if mentioned",
  "event_type": "one of: capacity_reduction | closure | reopening | price_shock | sanction | unrelated",
  "severity": "float in [0, 1] — 1.0 = complete closure of a major global corridor; 0 = no impact",
  "confidence": "float in [0, 1] — 1.0 = confirmed official announcement; 0.5 = unverified rumor; 0 = speculation",
  "affected_graph_element": "string or null — the canonical ID of the affected supply chain element, or null if unrelated/unresolvable",
  "justification": "string — one sentence explaining the extraction"
}

SEVERITY GUIDANCE:
- 0.0: No supply impact (unrelated event)
- 0.2: Minor: price signal, rumor, small cut
- 0.4: Moderate: confirmed partial closure, significant production cut
- 0.6: Significant: major corridor at elevated risk, armed conflict threatening a route
- 0.8: Severe: documented closure, confirmed attack on major tanker
- 1.0: Maximum: complete verified closure of a primary corridor (Hormuz, Bab)

CONFIDENCE GUIDANCE:
- 0.9-1.0: Official government/ministry announcement or verified by multiple major outlets
- 0.7-0.9: Single major news outlet (Reuters, Bloomberg, FT)
- 0.5-0.7: Unverified report or single regional source
- 0.3-0.5: Rumor, social media, single anonymous source
- 0.0-0.3: Speculation or clearly tangential

AFFECTED_GRAPH_ELEMENT: Use the exact canonical IDs below. If the entity does not match any:
__KNOWN_ELEMENTS_BLOCK__
(a generic "Russia" mention resolves to src_russia_espo, the larger of its two export streams)
If unrelated (cricket, tech, sports, general economy), set event_type to "unrelated" and affected_graph_element to null.
"""
# The prompt's element list is generated from the live node set (agents/schema.py),
# not hand-maintained — using str.replace (not .format) since the JSON schema block
# above contains literal braces that .format would otherwise misinterpret.
_SYSTEM_INSTRUCTION = _SYSTEM_INSTRUCTION.replace(
    "__KNOWN_ELEMENTS_BLOCK__", render_known_elements_prompt_block()
)


def parse(
    raw_text: str,
    source_override: Optional[str] = None,
    timestamp_override: Optional[datetime] = None,
) -> Optional[Event]:
    """
    Parse raw news text into a schema-valid Event object.

    Validates the LLM output against the Event Pydantic schema before returning.
    On validation failure, retries once with an error correction prompt.
    Returns None if both attempts fail.

    Also records three timestamps for latency measurement (plan.md Module 14):
    1. Signal timestamp (from article or timestamp_override)
    2. Parse completion timestamp (logged here)
    3. The caller logs the recommendation emission timestamp separately.

    Args:
        raw_text: The raw news article text or headline to parse.
        source_override: Manually specify the source (overrides LLM extraction).
        timestamp_override: Override the event timestamp (for replayed events).

    Returns:
        Validated Event object, or None if extraction fails.
    """
    t_parse_start = datetime.now(timezone.utc)

    prompt = f"""Extract the supply chain event from the following news text:

---
{raw_text}
---

Return only the JSON event object. Nothing else."""

    raw_json = call_llm(
        prompt=prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.0,
        expect_json=True,
    )

    if raw_json is None:
        logger.error("Extraction agent: LLM returned None. Skipping event.")
        return None

    event = _parse_and_validate(raw_json, source_override, timestamp_override)

    if event is None:
        # One retry with error correction
        logger.warning("Extraction agent: validation failed on first attempt. Retrying...")
        time.sleep(1.0)
        retry_prompt = f"""The JSON you returned was invalid. Fix it and return only the corrected JSON.

Original text:
{raw_text}

Required fields: id, source, timestamp, entity, location, event_type (must be one of the listed values),
severity (float 0-1), confidence (float 0-1), affected_graph_element (string or null), justification.

Return only the corrected JSON object."""

        raw_json_2 = call_llm(
            prompt=retry_prompt,
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.0,
            expect_json=True,
        )
        if raw_json_2:
            event = _parse_and_validate(raw_json_2, source_override, timestamp_override)

    t_parse_end = datetime.now(timezone.utc)

    if event:
        latency_ms = (t_parse_end - t_parse_start).total_seconds() * 1000
        logger.info(
            f"[LATENCY] signal_ts={event.timestamp.isoformat()} "
            f"parse_complete={t_parse_end.isoformat()} "
            f"parse_latency_ms={latency_ms:.0f}"
        )

    return event


def event_from_curated_timeline(item: dict) -> Event:
    """Build a deterministic Event from a reviewed replay-timeline record.

    Replay data is curated before the demo, so asking an external LLM to
    re-extract it only adds failure modes and makes latency unreproducible.
    The resulting event is still validated against the same Event contract as
    live extraction.
    """
    expected = item.get("expected_extraction", {})
    severity_range = expected.get("severity_range", [0.0, 0.0])
    confidence_range = expected.get("confidence_range", [0.0, 0.0])
    severity = sum(severity_range) / max(len(severity_range), 1)
    confidence = sum(confidence_range) / max(len(confidence_range), 1)
    timestamp = datetime.fromisoformat(item["original_timestamp"].replace("Z", "+00:00"))

    return Event(
        id=item["id"],
        source=item.get("source", "CURATED_REPLAY"),
        timestamp=timestamp,
        entity=expected.get("entity", "Unknown"),
        location=None,
        event_type=expected.get("event_type", "unrelated"),
        severity=severity,
        confidence=confidence,
        affected_graph_element=expected.get("affected_graph_element"),
        justification=f"Curated replay signal: {item.get('headline', 'no headline')}",
    )


def _parse_and_validate(
    raw_json: str,
    source_override: Optional[str],
    timestamp_override: Optional[datetime],
) -> Optional[Event]:
    """
    Attempt to parse and validate raw JSON string as an Event.

    Args:
        raw_json: Raw JSON string from LLM.
        source_override: Optional source string to override.
        timestamp_override: Optional datetime to override.

    Returns:
        Validated Event or None.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error(f"Extraction agent: JSON decode failed: {e}. Raw: {raw_json[:200]}")
        return None

    # Inject guaranteed unique ID if missing or empty
    if not data.get("id"):
        data["id"] = f"evt-{uuid.uuid4().hex[:8]}"

    # Apply overrides
    if source_override:
        data["source"] = source_override
    if timestamp_override:
        data["timestamp"] = timestamp_override.isoformat()
    elif "timestamp" not in data or not data["timestamp"]:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Attempt to resolve affected_graph_element via alias table if LLM used free text
    if data.get("affected_graph_element") and data["affected_graph_element"] not in known_element_ids():
        resolved = resolve_entity(data["affected_graph_element"])
        if resolved:
            logger.debug(
                f"Alias resolved: '{data['affected_graph_element']}' → '{resolved}'"
            )
            data["affected_graph_element"] = resolved
        else:
            logger.debug(
                f"Alias unresolvable: '{data['affected_graph_element']}'. Setting to None."
            )
            data["affected_graph_element"] = None

    try:
        event = Event(**data)
    except ValidationError as e:
        logger.error(f"Extraction agent: Pydantic validation failed: {e}")
        return None

    return event
