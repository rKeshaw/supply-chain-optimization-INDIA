"""Live news sensing adapter — GDELT DOC 2.0 API.

Free, unauthenticated, genuinely live global news monitoring (GDELT indexes
news roughly every 15 minutes). This is the "news feeds" source the problem
statement names as a suggested technology; nothing in the codebase queried a
real news source before this module — the demo default was, and remains, the
curated crisis_timeline.json replay.

Verification note (kept here deliberately, not swept under the rug): this
adapter was built and its request/response shape verified against GDELT's real
API from the development sandbox. That sandbox's shared outbound IP was
rate-limited by GDELT (HTTP 429, "limit requests to one every 5 seconds")
across every attempt, including after long waits — almost certainly from other
traffic sharing the same cloud IP range, not from this adapter's own request
rate. Full live end-to-end verification (an actual matched article flowing
through the pipeline) was NOT possible from that sandbox. The endpoint URL,
query shape, and the 429 handling below are all built against GDELT's real,
documented behavior and a real (if rate-limited) response payload observed
during development — but this should be re-verified once from a normal,
non-shared network before relying on it in a live demo.

Design mirrors agents/sanctions_adapter.py and agents/weather_adapter.py:
timeout-guarded, never raises, returns [] on any failure.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional

from agents.schema import known_element_ids, resolve_entity, ALIAS_TABLE

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
FETCH_TIMEOUT_S = 10.0
# GDELT's own published constraint (confirmed live: 429 with this exact message
# when violated). A module-level timestamp enforces it across calls regardless
# of caller cadence.
MIN_REQUEST_INTERVAL_S = 5.0
_last_request_ts = 0.0

# Query one rotating keyword per call rather than the whole node set at once —
# a good-citizen choice given the real rate limit above. Rotates through the
# corridor/chokepoint names, which is where a genuine physical disruption signal
# is most likely to show up first (matches the sensing layer's own emphasis).
_KEYWORD_ROTATION_STATE = {"index": 0}


def _rotating_keywords() -> list[str]:
    """Chokepoint and bypass names — the highest-value, lowest-ambiguity search
    terms (a source-country name alone is too noisy a news query on its own)."""
    from agents.schema import _load_raw_nodes  # local import: internal helper, deliberately not public API
    return [n["name"] for n in _load_raw_nodes() if n.get("type") in ("chokepoint", "bypass")]


def fetch_recent_articles(query: str, max_records: int = 5) -> Optional[list[dict]]:
    """Query GDELT for recent articles matching `query`. Returns None on failure."""
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "sort": "datedesc",
    }
    url = f"{GDELT_DOC_API}?{urllib.parse.urlencode(params)}"
    _last_request_ts = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EnergyResilience/1.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as response:
            body = response.read().decode("utf-8", errors="replace")
            if not body.strip():
                # GDELT's DOC API returns an empty body (not {"articles": []})
                # for a zero-result query — a normal, expected outcome, not a
                # fetch failure. json.loads("") raises "Expecting value: line 1
                # column 1 (char 0)", which previously got logged identically
                # to a genuine failure (a 429, a timeout) and read as "this is
                # broken" when it just meant "nothing matched this keyword".
                logger.debug(f"GDELT returned no results for query {query!r} (empty body).")
                return []
            data = json.loads(body)
            return data.get("articles", [])
    except Exception as e:
        logger.warning(f"GDELT fetch failed for query {query!r}: {e}")
        return None


def check_news_signals(lookback_minutes: int = 30) -> list[dict]:
    """Poll one rotating keyword for recent articles that resolve to a known
    graph element and fall within the lookback window.

    Returns a list of {"raw_text", "source", "timestamp", "url"} dicts — raw
    material for agents.extraction_agent.parse(), not pre-built Events, since
    a news headline (unlike an official sanctions listing) needs the LLM's
    severity/confidence judgement, not a fixed default.
    """
    keywords = _rotating_keywords()
    if not keywords:
        return []
    idx = _KEYWORD_ROTATION_STATE["index"] % len(keywords)
    _KEYWORD_ROTATION_STATE["index"] = idx + 1
    keyword = keywords[idx]

    articles = fetch_recent_articles(keyword)
    if not articles:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    signals = []
    for a in articles:
        seendate = a.get("seendate")  # GDELT format: YYYYMMDDTHHMMSSZ
        try:
            ts = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if ts < cutoff:
            continue
        title = a.get("title", "")
        if not resolve_entity(keyword):
            continue  # keyword itself must map to a known element; guards against drift
        signals.append({
            "raw_text": title,
            "source": a.get("domain", "GDELT"),
            "timestamp": ts,
            "url": a.get("url"),
            "matched_keyword": keyword,
        })
    return signals
