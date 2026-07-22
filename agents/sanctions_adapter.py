"""Sanctions-registry sensing adapter — OFAC SDN list.

This is the "sanctions registries" source the problem statement names as a
suggested technology; the rest of the sensing layer (news replay, weather) had
no sanctions-registry ingestion at all before this module.

Design (same shape as agents/weather_adapter.py — timeout-guarded, never raises,
returns [] on any failure so a network hiccup can't take down the pipeline):

1. Download the US Treasury OFAC Specially Designated Nationals (SDN) list —
   free, unauthenticated, no signup, ~19k rows.
2. Diff against a local snapshot of previously-seen entry IDs. On the very
   first run (no snapshot yet), the whole list is the baseline — it does NOT
   emit ~19k "new sanction" events; only entries added AFTER that baseline are
   ever reported. This mirrors how a real monitoring service bootstraps.
3. For each genuinely NEW entry, scan its name/program/remarks text for any of
   our known entity aliases (reusing agents.schema's data-driven ALIAS_TABLE —
   adding a node to nodes.json makes it scannable here too, no code change).
4. Emit a schema-valid Event for each match, at official-government-source
   confidence, for the extraction pipeline / process_signal to consume exactly
   like any other event.
"""

import csv
import io
import json
import logging
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.schema import Event, ALIAS_TABLE

logger = logging.getLogger(__name__)

SDN_CSV_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"
FETCH_TIMEOUT_S = 20.0  # the list is several MB; a short timeout would always fail
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / ".ofac_snapshot.json"

# A new SDN listing is a confirmed official government action — the highest
# confidence tier per the extraction agent's own confidence guidance (0.9-1.0).
SANCTION_EVENT_CONFIDENCE = 0.95
# Severity default for a newly sanctioned entity. Moderate-high: a real signal,
# but a single new SDN entry is not automatically a full corridor closure the
# way a confirmed physical event is — matches the extraction agent's "0.4-0.6"
# guidance band for "confirmed but not yet a documented major disruption".
SANCTION_EVENT_SEVERITY = 0.5


def fetch_sdn_csv() -> Optional[str]:
    """Download the current OFAC SDN list as raw CSV text."""
    try:
        req = urllib.request.Request(SDN_CSV_URL, headers={"User-Agent": "EnergyResilience/1.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to fetch OFAC SDN list: {e}")
        return None


def parse_sdn_entries(csv_text: str) -> list[dict]:
    """Parse the SDN CSV into entries with the fields we can match against.

    The file has no header row. Columns (per OFAC's published SDN.CSV layout):
    ent_num, SDN_Name, SDN_Type, Program, Title, Call_Sign, Vess_type,
    Tonnage, GRT, Vess_flag, Vess_owner, Remarks.
    """
    entries = []
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if len(row) < 12:
            continue
        entries.append({
            "ent_num": row[0].strip(),
            "name": row[1].strip(),
            "sdn_type": row[2].strip(),
            "program": row[3].strip(),
            "remarks": row[11].strip(),
        })
    return entries


def _load_seen_ids() -> Optional[set[str]]:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return set(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))
    except Exception as e:
        logger.warning(f"OFAC snapshot unreadable, treating as first run: {e}")
        return None


def _save_seen_ids(ids: set[str]) -> None:
    try:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(sorted(ids)), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to persist OFAC snapshot: {e}")


def _compile_alias_patterns() -> list[tuple[re.Pattern, str]]:
    """Word-boundary regex for every alias, longest phrase first.

    Raw substring containment produces
    false positives at an unusable rate — e.g. the 4-letter alias "oman"
    matching inside "Romania", or "iraq"-style short aliases matching inside
    an unrelated word in a long free-text remarks field. Word-boundary regex
    only matches a whole word/phrase, not a fragment of a longer one.
    """
    patterns = []
    for alias in sorted(ALIAS_TABLE.keys(), key=len, reverse=True):
        if len(alias) < 4:
            continue  # too short to be a meaningful signal even as a whole word
        patterns.append((re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE), ALIAS_TABLE[alias]))
    return patterns


_ALIAS_PATTERNS = _compile_alias_patterns()


def _scan_for_known_element(entry: dict) -> Optional[str]:
    """Match an SDN entry against the data-driven alias table.

    Deliberately scans only the structured, low-noise fields — ``program``
    (clean bracketed tags like "[IRAN] [SDGT]") and ``name`` — NOT the free-text
    ``remarks`` field, which contains addresses, dates of birth, and nationality
    prose where a short alias frequently appears as an incidental fragment
    rather than a genuine reference to the modeled entity. Program is checked
    first: it is the highest-precision signal OFAC itself assigns.
    """
    for field in ("program", "name"):
        text = entry.get(field, "")
        for pattern, node_id in _ALIAS_PATTERNS:
            if pattern.search(text):
                return node_id
    return None


def check_sanctions_updates() -> list[Event]:
    """Fetch the SDN list, diff against the last snapshot, return new-entry Events.

    Returns [] on any failure (network, parse, or first-run bootstrap) — the
    caller should treat that as "nothing to report right now", not an error.
    """
    csv_text = fetch_sdn_csv()
    if csv_text is None:
        return []

    entries = parse_sdn_entries(csv_text)
    if not entries:
        logger.warning("OFAC SDN list fetched but produced no parseable entries.")
        return []

    current_ids = {e["ent_num"] for e in entries}
    seen_ids = _load_seen_ids()

    if seen_ids is None:
        logger.info(f"OFAC sensing: establishing baseline of {len(current_ids)} entries. No events on first run.")
        _save_seen_ids(current_ids)
        return []

    new_entries = [e for e in entries if e["ent_num"] not in seen_ids]
    _save_seen_ids(current_ids)

    if not new_entries:
        return []

    timestamp = datetime.now(timezone.utc)
    events = []
    for entry in new_entries:
        matched_id = _scan_for_known_element(entry)
        if not matched_id:
            continue
        events.append(Event(
            id=f"ofac_{entry['ent_num']}_{int(timestamp.timestamp())}",
            source="OFAC SDN List (US Treasury)",
            timestamp=timestamp,
            entity=entry["name"],
            location=None,
            event_type="sanction",
            severity=SANCTION_EVENT_SEVERITY,
            confidence=SANCTION_EVENT_CONFIDENCE,
            affected_graph_element=matched_id,
            justification=f"New OFAC SDN listing: '{entry['name']}' (program: {entry['program']}).",
        ))

    logger.info(f"OFAC sensing: {len(new_entries)} new SDN entries, {len(events)} matched a modeled entity.")
    return events
