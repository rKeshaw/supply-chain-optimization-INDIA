"""
Schema definitions and alias table for the Energy Supply Chain Resilience system.

The Event, Node, and Edge schemas form the contract between the sensing layer
and the graph engine. The alias table maps free-text entity mentions to canonical
graph element IDs, enabling the extraction agent to resolve geography without
needing a full NER system.
"""

import json
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Optional
from datetime import datetime


class Event(BaseModel):
    """
    Structured event extracted from raw news/signal text by the extraction agent.
    This is the only object that the sensing layer passes to the graph engine.
    All fields must be schema-valid before the event is applied to the graph.
    """
    id: str
    source: str
    timestamp: datetime
    entity: str
    location: Optional[str] = None
    event_type: Literal[
        "capacity_reduction", "closure", "reopening",
        "price_shock", "sanction", "weather_disruption", "unrelated"
    ]
    severity: float  # [0, 1] — 1.0 = complete closure of a major corridor
    confidence: float  # [0, 1] — 1.0 = confirmed official announcement
    affected_graph_element: Optional[str] = None  # None if unrelated or unresolvable
    justification: str

    @field_validator("severity", "confidence")
    @classmethod
    def clamp_01(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def unrelated_has_no_element(self) -> "Event":
        if self.event_type == "unrelated":
            self.affected_graph_element = None
        return self


class Node(BaseModel):
    """
    A node in the supply chain graph. Refineries are split into _in and _out
    nodes joined by an internal edge whose capacity = the refinery's throughput limit.
    """
    id: str
    type: Literal[
        # "port": marine crude terminal feeding inland refineries by pipeline.
        # A coastal refinery's own berth stays folded into its refinery_in node.
        "source", "chokepoint", "bypass", "port", "refinery_in", "refinery_out",
        "spr", "super_source", "super_sink"
    ]
    name: str
    lat: float
    lon: float
    capacity_bbl_day: Optional[float] = None
    storage_capacity_bbl: Optional[float] = None  # SPR physical storage limit
    grade_compatibility: list[str] = []
    inventory_bbl: Optional[float] = None
    consumption_rate_bbl_day: Optional[float] = None
    # Effective availability, DERIVED: structural_openness * (1 - risk_score).
    # Everything downstream reads this one field.
    openness: float = 1.0
    # Known physical or policy state (a closed strait, a quota, an outage). Set
    # by scenarios, persists until explicitly changed.
    structural_openness: float = 1.0
    # Decaying news-driven risk. Fades without reinforcement; a structural state
    # does not.
    risk_score: float = 0.0
    # Procurement eligibility, not physical: the barrels still count toward
    # global supply, but the routing LP will not allocate them unless
    # routing_policy.allow_sanctions_restricted_sources is set.
    sanctions_restricted: bool = False
    confidence: float = 1.0
    last_updated: datetime

    @field_validator("openness", "risk_score", "confidence")
    @classmethod
    def clamp_01(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    class Config:
        extra = "allow"  # allow _source, _note fields in JSON without errors


class Edge(BaseModel):
    """
    A directed edge in the supply chain graph. Modes:
      - "sea":      an ocean shipping lane (the bulk of the network).
      - "pipeline": a physical crude pipeline. Currently the SPR emergency-release
                    links (reserve cavern -> co-located refinery); this is also the
                    mode a port->inland-refinery crude pipeline would use if an
                    inland refinery is added.
      - "internal": a virtual/structural link — the refinery in->out processing
                    edge (capacity = throughput limit) and the super_source/sink
                    connectors.
    """
    id: str
    from_id: str
    to_id: str
    mode: Literal["sea", "pipeline", "internal"]
    base_capacity_bbl_day: float
    cost_per_bbl: float
    transit_time_days: float
    openness: float = 1.0
    grade: Optional[str] = None  # crude grade this edge carries (None = grade-agnostic)
    path: Optional[list[list[float]]] = None  # pre-calculated geographical path
    @field_validator("openness")
    @classmethod
    def clamp_01(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    class Config:
        extra = "allow"  # allow _source, _note fields in JSON


# ---------------------------------------------------------------------------
# Alias Table — DATA-DRIVEN.
#
# Adding a node to data/nodes.json is enough to make it resolvable by the sensing
# layer. Its name is segmented into aliases automatically and any "aliases" list
# in its JSON entry is merged in, so no code file needs hand-editing and the
# table cannot drift away from the node set it describes.
#
# A small SUPPLEMENTAL_ALIASES layer remains in code ONLY for terms that are
# genuinely not one node's identity: regional umbrella terms that span multiple
# real chokepoints/sources in the real world ("Persian Gulf", "OPEC"), or a
# deliberate default among ambiguous siblings (a bare "Russia" mention resolving
# to the larger ESPO stream rather than Urals). Each entry documents why it can't
# just live on one node.
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ALIAS_ELIGIBLE_TYPES = {"source", "chokepoint", "bypass", "refinery_in", "spr"}
# Segments too generic to trust as a unique alias for any single node — every
# refinery has an "Inlet", every reserve facility is "ISPRL something".
_GENERIC_SEGMENT_BLACKLIST = {"inlet", "outlet", "terminal", "port", "island", "isprl", "virtual"}
_STOPWORDS = {"the", "of", "a", "an", "and", "in", "at"}

# Regional/umbrella terms and deliberate tie-break defaults — see module docstring.
SUPPLEMENTAL_ALIASES: dict[str, str] = {
    "persian gulf": "chk_hormuz",   # the Gulf itself is not a node; Hormuz is its only exit chokepoint
    "gulf": "chk_hormuz",
    "arabian gulf": "chk_hormuz",
    "gulf of aden": "chk_bab",      # feeds into Bab-el-Mandeb; no separate node modeled
    "russia": "src_russia_espo",    # bare "Russia" defaults to the larger (ESPO/Pacific) stream
    "russian": "src_russia_espo",
    "russia federation": "src_russia_espo",
    "opec": "src_saudi",            # OPEC/OPEC+ affects many source nodes at once; the Event schema
    "opec+": "src_saudi",           # is single-target, so this is a documented proxy, not a full model
                                     # of multi-node events (see DEFAULT_SCENARIOS["opec_cut"] for the
                                     # multi-node version used by scenario disruptions).
    "arabian sea": "ref_jamnagar_in",   # west-coast SPM anchorage zone, not a distinct node
    "gulf of kutch": "ref_jamnagar_in",
    "bay of bengal": "ref_paradip_in",  # east-coast SPM anchorage zone, not a distinct node
}


def _load_raw_nodes() -> list[dict]:
    path = _DATA_DIR / "nodes.json"
    if not path.exists():
        return []
    return [n for n in json.loads(path.read_text(encoding="utf-8")) if "id" in n]


def _derive_segments(name: str) -> list[str]:
    """Split a node's display name into precise alias phrases.

    Splits on structural punctuation only (dashes, parens, slashes, commas) —
    deliberately NOT a full word-by-word tokenization, which would produce
    generic single-word collisions (e.g. every refinery's name would yield
    "inlet"). Each resulting phrase is kept whole, filtered against the
    generic-segment blacklist and a minimum length.
    """
    normalized = name.lower()
    for ch in "—()/,":
        normalized = normalized.replace(ch, "-")
    segments = [s.strip() for s in normalized.split("-")]
    kept = [
        s for s in segments
        if len(s) > 2 and s not in _GENERIC_SEGMENT_BLACKLIST
    ]

    # Keep the whole name flattened to one phrase as well. Splitting on hyphens
    # suits a structural name such as "Russia - Urals (Novorossiysk)" but shreds
    # a hyphenated proper noun like "Bab-el-Mandeb", leaving the strait without
    # its own name among the aliases.
    flattened = " ".join(normalized.replace("-", " ").split())
    if len(flattened) > 2 and flattened not in kept:
        kept.append(flattened)
    return kept


def _build_alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    explicit: dict[str, str] = {}
    for node in _load_raw_nodes():
        if node.get("type") not in _ALIAS_ELIGIBLE_TYPES:
            continue
        node_id = node["id"]
        for phrase in _derive_segments(node.get("name", "")):
            table.setdefault(phrase, node_id)  # derived: lowest priority, first-writer wins
        for phrase in node.get("aliases", []):
            explicit[phrase.lower().strip()] = node_id  # author-curated: overrides derived
    table.update(explicit)
    table.update(SUPPLEMENTAL_ALIASES)  # most deliberately curated: overrides everything
    return table


def _tokenize(text: str) -> frozenset[str]:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return frozenset(w for w in cleaned.split() if w not in _STOPWORDS)


ALIAS_TABLE: dict[str, str] = _build_alias_table()
# Precomputed token sets for the fallback matcher, built once alongside the table.
_ALIAS_TOKENS: dict[str, frozenset[str]] = {alias: _tokenize(alias) for alias in ALIAS_TABLE}


def resolve_entity(mention: str) -> Optional[str]:
    """
    Resolve a free-text entity mention to a canonical graph element ID.

    Two passes:
    1. Exact match against ALIAS_TABLE (fast path — covers the common short forms).
    2. Token-subset fallback: normalizes the mention to a stopword-stripped token
       set and matches against any alias whose own token set contains it, or is
       contained by it (e.g. "Hormuz Strait" and "the Strait of Hormuz" both
       token-match "strait of hormuz" without needing every literal word-order
       variant enumerated by hand). Containment must be exact (a proper subset/
       superset), not partial overlap, to keep false positives rare; ties prefer
       the longer (more specific) alias.

       The two directions carry different weights of evidence. A mention
       contained by an alias is safe, since a short mention matching a longer
       canonical name is a genuine abbreviation. An alias contained by a longer
       mention is where a one-word alias becomes a liability, because it fires on
       any sentence containing that word: a bare "gulf" would otherwise resolve
       "Gulf of Mexico" and "the gulf war" to the Strait of Hormuz. A one-token
       alias therefore matches only by exact match or by the safe direction.

    Args:
        mention: Raw entity string from the extraction agent output.

    Returns:
        Canonical graph element ID string, or None if no match found.
        Caller should treat None as 'unresolvable' — the event may still be
        logged but should not trigger a graph state update.
    """
    if not mention:
        return None
    key = mention.lower().strip()
    if key in ALIAS_TABLE:
        return ALIAS_TABLE[key]

    mention_tokens = _tokenize(mention)
    if not mention_tokens:
        return None
    best_alias, best_len = None, -1
    for alias, tokens in _ALIAS_TOKENS.items():
        if not tokens:
            continue
        # Safe direction: the mention is an abbreviation of a longer alias.
        matched = mention_tokens <= tokens
        # Risky direction: the alias appears inside a longer mention. Requires at
        # least two tokens of evidence — see this function's docstring.
        if not matched and len(tokens) >= 2:
            matched = tokens <= mention_tokens
        if matched and len(tokens) > best_len:
            best_alias, best_len = alias, len(tokens)
    return ALIAS_TABLE[best_alias] if best_alias else None


def known_element_ids() -> set[str]:
    """All canonical graph element IDs the extraction agent may target.

    Derived from data/nodes.json — replaces a hand-maintained duplicate list
    that had to be edited every time a node was added.
    """
    return {n["id"] for n in _load_raw_nodes() if n.get("type") in _ALIAS_ELIGIBLE_TYPES}


def render_known_elements_prompt_block() -> str:
    """Render the current node set as prompt text, grouped by type.

    Builds the extraction agent's system instruction from the live node set, so
    the model is always told about the nodes that actually exist.
    """
    groups: dict[str, list[str]] = {}
    for node in _load_raw_nodes():
        t = node.get("type")
        if t not in _ALIAS_ELIGIBLE_TYPES:
            continue
        groups.setdefault(t, []).append(node["id"])
    labels = {
        "chokepoint": "Known chokepoints", "bypass": "Known bypass routes",
        "source": "Source nodes", "refinery_in": "Refineries", "spr": "Strategic reserves",
    }
    lines = [f"- {labels[t]}: {', '.join(sorted(ids))}" for t, ids in groups.items() if ids]
    return "\n".join(lines)


def update_risk_score(
    current_risk: float,
    severity: float,
    confidence: float,
    decay: float
) -> float:
    """
    Update a node or edge's risk score with a confidence-weighted harmful
    signal. Time decay is applied separately through ``decay_risk_score``.

    Formula: risk' = 1 - (1 - current_risk) × (1 - severity × confidence)

    The formula is a bounded noisy-OR update: independent credible harmful
    signals accumulate without allowing risk to exceed one. A confirmed severe
    event therefore changes operating state immediately instead of being diluted
    by the elapsed-time decay coefficient.

    Args:
        current_risk: Current risk score for the element [0, 1].
        severity: Event severity [0, 1].
        confidence: Event confidence [0, 1].
        decay: Retained for backwards-compatible call sites. Decay is applied
            by ``decay_risk_score`` before this function is called.

    Returns:
        Updated risk score, clamped to [0, 1].
    """
    del decay
    new_signal = severity * confidence
    new_risk = 1.0 - (1.0 - current_risk) * (1.0 - new_signal)
    return max(0.0, min(1.0, new_risk))


_DECAY_ELAPSED_DAYS_CAP = 30.0


def decay_risk_score(
    current_risk: float,
    decay_factor_per_day: float,
    elapsed_days: float,
) -> float:
    """Decay risk toward normal over elapsed calendar time.

    ``decay_factor_per_day`` is a retention factor: 0.92 means 92% of the
    prior risk remains after one day without a reinforcing signal.

    Elapsed time is capped at ``_DECAY_ELAPSED_DAYS_CAP`` (30 days) before
    exponentiating. This parameter's own documentation already treats ~28 days
    (log(0.1)/log(0.92)) as "days_to_baseline" — the point by which risk has
    faded to its natural resting floor. Without a cap, that same exponential
    keeps compounding for however long the gap to the NEXT reinforcing signal
    happens to be: replaying the curated crisis timeline (real gaps up to 224
    days between events on different corridors) drove risk to ~1e-9, i.e.
    numerically zero — reading as "the prior escalation was completely
    forgotten/restored" rather than "faded to a low resting background level
    and stayed there absent reinforcement", which is what the PS's own framing
    describes ("the supply threat live" through an extended period, not reset
    by every quiet news cycle). Capping means a node settles to a persistent
    ~8% floor of whatever risk it last held, rather than continuing to decay
    toward a meaningless near-zero the longer real time happens to elapse.
    """
    elapsed_days = max(0.0, float(elapsed_days))
    elapsed_days = min(elapsed_days, _DECAY_ELAPSED_DAYS_CAP)
    decay_factor_per_day = max(0.0, min(1.0, float(decay_factor_per_day)))
    return max(0.0, min(1.0, current_risk * (decay_factor_per_day ** elapsed_days)))
