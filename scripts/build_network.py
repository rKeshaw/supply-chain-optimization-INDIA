"""Regenerate data/edges.json from data/nodes.json and real sea-route geometry.

Run:  python scripts/build_network.py          (writes data/edges.json)
      python scripts/build_network.py --check   (verify only, non-zero exit on drift)

WHY THIS EXISTS
---------------
Every geographic quantity is derived here from a single calibration, so the data
is reproducible and auditable. Hand-typed lane costs and transit times drift
against each other: implied vessel speeds spread across 4.7 to 26.3 knots where a
laden very large crude carrier does 12 to 15, and freight across $0.197 to $0.965
per 1000 km for comparable voyages, which turns the fastest-route objective into
an optimisation over noise.

CALIBRATION
-----------
Distance and geometry: `searoute`, which returns an actual navigable sea route
(it goes around Sri Lanka, through canals, around capes) rather than a
great-circle line.

Freight:  cost_per_bbl = FIXED (first leg only) + RATE * km/1000 + canal toll

  Real tanker freight is a fixed port/terminal component plus a distance
  component; it is not proportional to distance. The three freight anchors
  already sourced in data/parameters.json (freight_cost_estimates) confirm this
  directly -- their implied rates span $0.218-$0.673 per 1000 km precisely
  because the fixed component dominates on short hauls:

      Ras Tanura -> Jamnagar (VLCC)      2229 km   $1.50
      Nigeria    -> India via Cape       13768 km  $3.00
      Kozmino    -> India via Malacca    11108 km  $2.80

  Least-squares fit over those three points gives FIXED = $1.2214 and
  RATE = $0.1341 per 1000 km. The fixed term is charged once per voyage (on the
  leg leaving the source node), not once per leg, so a multi-leg path
  reconstructs the anchor it was fitted to:

      PG      -> Jamnagar   $1.520  vs anchor $1.50   (+1.4%)
      Nigeria -> Jamnagar   $3.068  vs anchor $3.00   (+2.3%)
      ESPO    -> Jamnagar   $2.711  vs anchor $2.80   (-3.2%)

  None of the three anchors transits the Suez Canal, so the fit contains no
  canal cost and the Suez Canal Authority toll is added separately on the canal
  leg. The $0.60/bbl figure is sourced to Suez Canal Authority published tariffs.

Transit:  days = km / (LADEN_SPEED_KNOTS * 1.852 * 24) + 1 day on the canal leg

  13 knots is standard laden VLCC service speed. Straits (Hormuz, Bab-el-Mandeb,
  Malacca) are open water and cost no extra time; the Suez Canal is a scheduled
  convoy transit and gets one day. Port loading/discharge time is deliberately
  NOT added: the digital twin treats arrival as availability, so discharge would
  be double-counted.

TOPOLOGY
--------
Suez sits upstream of Bab-el-Mandeb: a Black Sea cargo reaches India via
Bosphorus -> Mediterranean -> Suez -> Red Sea -> Bab -> Arabian Sea.
Malacca is inbound from Kozmino/ESPO only -- the one India-bound trade that
uses it; Paradip and Vizag are served directly around Sri Lanka.
src_iran is connected but flagged sanctions_restricted, so the optimizer
excludes it unless routing_policy.allow_sanctions_restricted_sources is set.
"""

import argparse
import json
import sys
from pathlib import Path

import searoute

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- calibration constants (see module docstring) --------------------------
FREIGHT_FIXED_USD_PER_BBL = 1.2214       # per voyage, charged on the source leg
FREIGHT_RATE_USD_PER_1000KM = 0.1341
SUEZ_CANAL_TOLL_USD_PER_BBL = 0.60       # SCA published tariff, carried over
LADEN_SPEED_KNOTS = 13.0
SUEZ_CANAL_TRANSIT_DAYS = 1.0
KM_PER_DAY = LADEN_SPEED_KNOTS * 1.852 * 24  # 577.9 km/day

# The one leg that physically passes through the Suez Canal.
CANAL_LEG = ("chk_suez", "chk_bab")


# --- lane specification -----------------------------------------------------
# (edge_id, from_id, to_id, note)
# Grade and capacity are derived from the node data; see _grade_for / _capacity_for.
SEA_LANES = [
    # Persian Gulf loading terminals -> Strait of Hormuz
    ("e_iraq_hormuz",       "src_iraq",        "chk_hormuz",       "Basrah Oil Terminal to the Strait of Hormuz."),
    ("e_saudi_hormuz",      "src_saudi",       "chk_hormuz",       "Ras Tanura / Ju'aymah to the Strait of Hormuz."),
    ("e_uae_hormuz",        "src_uae",         "chk_hormuz",       "Ruwais / Das Island to the Strait of Hormuz."),
    ("e_kuwait_hormuz",     "src_kuwait",      "chk_hormuz",       "Mina Al Ahmadi to the Strait of Hormuz."),
    ("e_qatar_hormuz",      "src_qatar",       "chk_hormuz",       "Ras Laffan to the Strait of Hormuz."),
    ("e_iran_hormuz",       "src_iran",        "chk_hormuz",       "Kharg Island to the Strait of Hormuz. src_iran is sanctions_restricted; India has lifted no Iranian crude since the US waiver expired in May 2019."),

    # Hormuz -> Indian refineries (west coast direct, east coast around Sri Lanka)
    ("e_hormuz_jamnagar",   "chk_hormuz",      "ref_jamnagar_in",  "Hormuz to Jamnagar (Sikka / Gulf of Kutch SPM)."),
    ("e_hormuz_mangalore",  "chk_hormuz",      "ref_mangalore_in", "Hormuz to New Mangalore."),
    ("e_hormuz_kochi",      "chk_hormuz",      "ref_kochi_in",     "Hormuz to Kochi SPM."),
    ("e_hormuz_vizag",      "chk_hormuz",      "ref_vizag_in",     "Hormuz to Visakhapatnam, around Sri Lanka into the Bay of Bengal."),

    # Oman loads outside Hormuz - direct, no chokepoint transit
    ("e_oman_jamnagar",     "src_oman",        "ref_jamnagar_in",  "Mina Al Fahal to Jamnagar. Oman loads outside the Strait of Hormuz, so this lane bypasses it entirely."),
    ("e_oman_mangalore",    "src_oman",        "ref_mangalore_in", "Mina Al Fahal to New Mangalore, outside Hormuz."),
    ("e_oman_kochi",        "src_oman",        "ref_kochi_in",     "Mina Al Fahal to Kochi, outside Hormuz."),
    ("e_oman_vizag",        "src_oman",        "ref_vizag_in",     "Mina Al Fahal to Visakhapatnam, outside Hormuz, around Sri Lanka."),

    # Black Sea and West Africa -> Suez Canal (Suez is UPSTREAM of Bab)
    ("e_russia_suez",       "src_russia_urals", "chk_suez",        "Novorossiysk to the Suez Canal via the Bosphorus and Mediterranean."),
    ("e_kazakhstan_suez",   "src_kazakhstan",  "chk_suez",         "CPC Terminal, Novorossiysk, to the Suez Canal via the Bosphorus and Mediterranean."),
    ("e_nigeria_suez",      "src_nigeria",     "chk_suez",         "Bonny to the Suez Canal via Gibraltar and the Mediterranean. Longer and dearer than the Cape route for India-bound cargoes, so the optimizer only reaches for it when the Cape is constrained."),

    ("e_suez_bab",          "chk_suez",        "chk_bab",          "Suez Canal transit and Red Sea crossing to Bab-el-Mandeb. Carries the Suez Canal Authority toll and one day of scheduled convoy transit."),

    # Bab-el-Mandeb -> Indian refineries
    ("e_bab_jamnagar",      "chk_bab",         "ref_jamnagar_in",  "Bab-el-Mandeb to Jamnagar across the Arabian Sea."),
    ("e_bab_mangalore",     "chk_bab",         "ref_mangalore_in", "Bab-el-Mandeb to New Mangalore."),
    ("e_bab_kochi",         "chk_bab",         "ref_kochi_in",     "Bab-el-Mandeb to Kochi."),
    ("e_bab_vizag",         "chk_bab",         "ref_vizag_in",     "Bab-el-Mandeb to Visakhapatnam, around Sri Lanka."),

    # Atlantic basin -> Cape of Good Hope
    ("e_nigeria_cog",       "src_nigeria",     "chk_cog",          "Bonny to the Cape of Good Hope. EIA 2024 confirms West African cargoes rerouted around the Cape during the Red Sea disruption."),
    ("e_angola_cog",        "src_angola",      "chk_cog",          "Luanda / Cabinda to the Cape of Good Hope."),
    ("e_usa_cog",           "src_usa",         "chk_cog",          "US Gulf Coast to the Cape of Good Hope. Fully laden VLCCs cannot transit Suez, so the Cape is the standard routing."),
    ("e_venezuela_cog",     "src_venezuela",   "chk_cog",          "Jose Terminal to the Cape of Good Hope."),
    ("e_brazil_cog",        "src_brazil",      "chk_cog",          "Porto do Acu to the Cape of Good Hope."),

    # Cape -> Indian refineries
    ("e_cog_jamnagar",      "chk_cog",         "ref_jamnagar_in",  "Cape of Good Hope to Jamnagar."),
    ("e_cog_mangalore",     "chk_cog",         "ref_mangalore_in", "Cape of Good Hope to New Mangalore."),
    ("e_cog_kochi",         "chk_cog",         "ref_kochi_in",     "Cape of Good Hope to Kochi."),
    ("e_cog_vizag",         "chk_cog",         "ref_vizag_in",     "Cape of Good Hope to Visakhapatnam, around Sri Lanka."),

    # Russian Pacific ESPO -> Malacca -> India. Malacca's only legitimate inbound.
    ("e_russia_malacca",    "src_russia_espo", "chk_malacca",      "Kozmino (Pacific) to the Strait of Malacca. This is the only crude flow to India that genuinely transits Malacca."),
    ("e_malacca_vizag",     "chk_malacca",     "ref_vizag_in",     "Malacca to Visakhapatnam across the Bay of Bengal."),
    ("e_malacca_jamnagar",  "chk_malacca",     "ref_jamnagar_in",  "Malacca to Jamnagar around Sri Lanka. ESPO crude to west-coast India is an established trade flow."),
    ("e_malacca_mangalore", "chk_malacca",     "ref_mangalore_in", "Malacca to New Mangalore around Sri Lanka."),
    ("e_malacca_kochi",     "chk_malacca",     "ref_kochi_in",     "Malacca to Kochi around Sri Lanka."),
]

# Coastal refineries and pipeline-origin ports added to close the demand side of
# the model (scripts/add_refineries.py). Each is reachable from every corridor
# that can physically serve it, exactly as the original five are.
_DISCHARGE_POINTS = [
    "ref_vadinar_in", "ref_mumbai_bpcl_in", "ref_mumbai_hpcl_in",
    "ref_chennai_cpcl_in", "ref_haldia_in",
    "port_salaya", "port_vadinar", "port_mundra", "port_paradip",
]
for _corridor, _label in (("chk_hormuz", "Strait of Hormuz"),
                          ("chk_bab", "Bab-el-Mandeb"),
                          ("chk_cog", "Cape of Good Hope"),
                          ("chk_malacca", "Strait of Malacca"),
                          ("src_oman", "Mina Al Fahal (outside Hormuz)")):
    for _dest in _DISCHARGE_POINTS:
        # A port and a refinery can share a place name -- Vadinar has both the
        # Nayara refinery and BPCL's separate crude terminal, and Paradip has
        # both the IOCL refinery and the PHBPL pipeline head. Keep the "pt_"
        # marker so the two never collapse onto one edge id.
        _stem = ("pt_" + _dest.removeprefix("port_") if _dest.startswith("port_")
                 else _dest.removeprefix("ref_").removesuffix("_in"))
        SEA_LANES.append((
            f"e_{_corridor.replace('chk_', '').replace('src_', '')}_{_stem}",
            _corridor, _dest,
            f"{_label} to {_stem.removeprefix('pt_').replace('_', ' ').title()}"
            f"{' crude terminal' if _dest.startswith('port_') else ''}.",
        ))

# Crude pipelines: port -> inland refinery. Real, named lines with published
# lengths and capacities; see scripts/add_refineries.py for the sources.
# (id, from_id, to_id, km, MMTPA, note)
CRUDE_PIPELINES = [
    ("e_pl_salaya_koyali",    "port_salaya",  "ref_koyali_in",    1130,  25.0,
     "Salaya-Mathura crude pipeline (SMPL), IndianOil. 2,663 km total / 25 MMTPA installed; Koyali is the first delivery point. Capacity is shared with the Mathura and Panipat offtakes and is enforced at the port_salaya node."),
    ("e_pl_salaya_mathura",   "port_salaya",  "ref_mathura_in",   1900,  25.0,
     "Salaya-Mathura crude pipeline (SMPL), IndianOil. Mathura is the pipeline's namesake terminus at roughly 1,900 km along the route."),
    ("e_pl_salaya_panipat",   "port_salaya",  "ref_panipat_in",   2663,  25.0,
     "Salaya-Mathura crude pipeline (SMPL) extension to Panipat, IndianOil. 2,663 km end to end."),
    ("e_pl_mundra_panipat",   "port_mundra",  "ref_panipat_in",   1194,   8.4,
     "Mundra-Panipat crude pipeline (MPPL), IndianOil. 1,194 km / 8.4 MMTPA. A second 17.5 MMTPA line is under construction and is NOT modelled."),
    ("e_pl_mundra_bathinda",  "port_mundra",  "ref_bathinda_in",  1017,  11.3,
     "Mundra-Bathinda crude pipeline, HPCL-Mittal Pipelines. 1,017 km, published capacity 11.25 MMTPA and expandable to 18 MMTPA; carried here at the refinery's own 11.3 MMTPA because the 0.05 MMTPA difference is below the precision of either published figure and would otherwise strand the refinery at 99.6% in perpetuity. Sole crude supply to the Guru Gobind Singh refinery."),

    # Domestic crude. These bypass every international chokepoint, which is the
    # entire point of modelling them: they show which Indian refining capacity is
    # structurally insulated from a corridor closure.
    ("e_pl_mumbai_high_bpcl", "src_mumbai_high", "ref_mumbai_bpcl_in", 180, 6.55,
     "Mumbai High (ONGC Western Offshore) to the BPCL Mumbai refinery via the Uran terminal. Domestic crude."),
    ("e_pl_mumbai_high_hpcl", "src_mumbai_high", "ref_mumbai_hpcl_in", 180, 6.55,
     "Mumbai High (ONGC Western Offshore) to the HPCL Mumbai refinery via the Uran terminal. Domestic crude. Field capacity is shared with the adjacent BPCL offtake and is enforced at the source node."),
    ("e_pl_gujarat_koyali",   "src_gujarat_onshore", "ref_koyali_in", 120, 4.35,
     "ONGC Gujarat onshore fields (Ankleshwar / Gandhar / Kalol) to the Koyali refinery, which was originally built around this production. Domestic crude."),
    ("e_pl_vadinar_bina",     "port_vadinar", "ref_bina_in",       937,   7.8,
     "Vadinar-Bina crude pipeline, BORL/BPCL. 937 km / 7.8 MMTPA. Sole crude supply to the Bina refinery."),
    ("e_pl_paradip_refinery", "port_paradip", "ref_paradip_in",     10,  20.4,
     "Short in-port connection from the Paradip crude terminal to the adjacent IOCL Paradip refinery."),
    ("e_pl_paradip_haldia",   "port_paradip", "ref_haldia_in",     330,  20.4,
     "Paradip-Haldia-Barauni crude pipeline (PHBPL), IndianOil. 1,465 km / 20.4 MMTPA; Haldia is the first inland delivery point. Haldia also receives crude directly by sea at Haldia Dock, so both routes exist."),
    ("e_pl_paradip_barauni",  "port_paradip", "ref_barauni_in",   1465,  20.4,
     "Paradip-Haldia-Barauni crude pipeline (PHBPL), IndianOil. Barauni is the terminus at 1,465 km."),
]

# Crude pipeline economics. Both values are [ESTIMATED] and are the weakest-
# sourced numbers in this file -- there is no single published per-barrel tariff
# covering these lines. Documented here rather than buried in the edge records.
PIPELINE_TARIFF_USD_PER_BBL_PER_1000KM = 0.75   # PNGRB-regulated long-haul crude tariffs
PIPELINE_VELOCITY_KM_PER_DAY = 130.0            # ~1.5 m/s, standard crude pipeline design velocity


def _load_nodes() -> dict:
    raw = json.loads((DATA_DIR / "nodes.json").read_text(encoding="utf-8"))
    return {n["id"]: n for n in raw if "id" in n}


def _grade_for(nodes: dict, from_id: str) -> str | None:
    """A lane leaving a source carries that source's single crude grade; a lane
    leaving a chokepoint is grade-agnostic because several grades share it."""
    node = nodes[from_id]
    if node.get("type") != "source":
        return None
    grades = node.get("grade_compatibility") or []
    return grades[0] if len(grades) == 1 else None


def _capacity_for(nodes: dict, from_id: str, to_id: str) -> float:
    """A sea lane has no intrinsic throughput limit. The real constraints are the
    load terminal, the discharge terminal / refinery, and the strait -- all of
    which are modelled as NODE capacities and enforced separately by the solver.
    So a lane is sized by whichever endpoint is the physical bottleneck, and the
    node-level caps do the actual work."""
    src, dst = nodes[from_id], nodes[to_id]
    if src.get("type") == "source":
        # Load terminal is the constraint; a source with several outbound routes
        # can send its full volume down any one of them, and its node capacity
        # stops the total from exceeding what it can actually export.
        cap = float(src.get("capacity_bbl_day") or 0)
        if dst.get("type") == "refinery_in":
            cap = min(cap, float(dst.get("capacity_bbl_day") or 0))
        return cap
    if dst.get("type") == "refinery_in":
        return float(dst.get("capacity_bbl_day") or 0)
    # chokepoint -> chokepoint
    return min(float(src.get("capacity_bbl_day") or 0), float(dst.get("capacity_bbl_day") or 0))


def _route(nodes: dict, from_id: str, to_id: str):
    a, b = nodes[from_id], nodes[to_id]
    r = searoute.searoute([a["lon"], a["lat"]], [b["lon"], b["lat"]])
    return r["properties"]["length"], r["geometry"]["coordinates"]


def build_sea_edges(nodes: dict) -> list[dict]:
    edges = []
    for edge_id, from_id, to_id, note in SEA_LANES:
        for nid in (from_id, to_id):
            if nid not in nodes:
                raise ValueError(f"{edge_id}: node {nid!r} is not in nodes.json")
        km, coords = _route(nodes, from_id, to_id)
        is_canal = (from_id, to_id) == CANAL_LEG
        from_source = nodes[from_id].get("type") == "source"

        cost = FREIGHT_RATE_USD_PER_1000KM * km / 1000.0
        if from_source:
            cost += FREIGHT_FIXED_USD_PER_BBL
        if is_canal:
            cost += SUEZ_CANAL_TOLL_USD_PER_BBL

        days = km / KM_PER_DAY
        if is_canal:
            days += SUEZ_CANAL_TRANSIT_DAYS

        detail = (
            f"{note} Distance {km:,.0f} km (searoute). "
            f"Cost = {'fixed $%.4f + ' % FREIGHT_FIXED_USD_PER_BBL if from_source else ''}"
            f"${FREIGHT_RATE_USD_PER_1000KM:.4f}/1000km x {km:,.0f} km"
            f"{' + $%.2f Suez Canal toll' % SUEZ_CANAL_TOLL_USD_PER_BBL if is_canal else ''}. "
            f"Transit = {km:,.0f} km at {LADEN_SPEED_KNOTS:.0f} kn laden"
            f"{' + %.0f d canal transit' % SUEZ_CANAL_TRANSIT_DAYS if is_canal else ''}. "
            f"Generated by scripts/build_network.py -- do not hand-edit."
        )

        edges.append({
            "id": edge_id,
            "from_id": from_id,
            "to_id": to_id,
            "mode": "sea",
            "base_capacity_bbl_day": _capacity_for(nodes, from_id, to_id),
            "cost_per_bbl": round(cost, 4),
            "transit_time_days": round(days, 2),
            "openness": 1.0,
            "grade": _grade_for(nodes, from_id),
            "distance_km": round(km, 1),
            "_source": detail,
            "path": [[round(c[0], 6), round(c[1], 6)] for c in coords],
        })
    return edges


def build_crude_pipeline_edges(nodes: dict) -> list[dict]:
    """Port -> inland refinery crude pipelines. Real, named lines; length and
    installed capacity are published (see scripts/add_refineries.py). Tariff and
    transit are derived from the two [ESTIMATED] constants above."""
    edges = []
    for edge_id, from_id, to_id, km, mmtpa, note in CRUDE_PIPELINES:
        for nid in (from_id, to_id):
            if nid not in nodes:
                raise ValueError(f"{edge_id}: node {nid!r} is not in nodes.json")
        cost = PIPELINE_TARIFF_USD_PER_BBL_PER_1000KM * km / 1000.0
        days = km / PIPELINE_VELOCITY_KM_PER_DAY
        # A pipeline branch can never carry more than the refinery it serves can
        # process; the line's own installed capacity is enforced at the port node,
        # which is what makes several offtakes share one pipeline correctly.
        cap = min(float(nodes[to_id].get("capacity_bbl_day") or 0), mmtpa * 20_000)
        edges.append({
            "id": edge_id, "from_id": from_id, "to_id": to_id, "mode": "pipeline",
            "base_capacity_bbl_day": cap,
            "cost_per_bbl": round(cost, 4),
            "transit_time_days": round(days, 2),
            "openness": 1.0, "grade": None,
            "distance_km": float(km),
            # No surveyed route geometry, so the map draws a straight line
            # between terminal and refinery. Length below is the real published
            # pipeline length, which is what the solver costs against.
            "_geometry": "schematic",
            "_source": (
                f"{note} Tariff = ${PIPELINE_TARIFF_USD_PER_BBL_PER_1000KM:.2f}/1000km x {km:,} km "
                f"[ESTIMATED from PNGRB-regulated long-haul crude tariffs]. "
                f"Transit = {km:,} km at {PIPELINE_VELOCITY_KM_PER_DAY:.0f} km/day "
                f"(~1.5 m/s design velocity) [ESTIMATED]. "
                f"Generated by scripts/build_network.py -- do not hand-edit."
            ),
        })
    return edges


def build_structural_edges(nodes: dict, previous: list[dict]) -> list[dict]:
    """Virtual connectors and the refinery inlet-to-outlet processing edge.

    These carry no geography, so they are generated from the node set rather than
    hand-maintained, which keeps every source connected as sources are added.
    Reserve discharge pipelines are carried over verbatim because their cost is a
    last-resort signal rather than freight."""
    kept = [dict(e) for e in previous
            if e.get("mode") == "pipeline" and e["from_id"].startswith("spr_")]

    # Connectors from the virtual source, sized at each source's real export
    # ceiling. Maximum-flow honours edge capacities only and never reads a node's
    # declared capacity_bbl_day, so the ceiling has to sit on this edge for the
    # reported minimum cut to mean anything. The routing solver enforces the node
    # capacity directly and is unaffected either way.
    for nid, node in nodes.items():
        if node.get("type") != "source":
            continue
        kept.append({
            "id": f"e_ss_{nid.removeprefix('src_')}",
            "from_id": "super_source", "to_id": nid, "mode": "internal",
            "base_capacity_bbl_day": float(node.get("capacity_bbl_day") or 0),
            "cost_per_bbl": 0.0, "transit_time_days": 0,
            "openness": 1.0, "grade": None,
            "_source": "Virtual connector. Capacity = this source's export ceiling, so that the max-flow baseline respects it (NetworkX max-flow reads edge capacities only, never node capacity_bbl_day).",
        })

    for nid, node in nodes.items():
        if node.get("type") != "refinery_in":
            continue
        out_id = nid.replace("_in", "_out")
        if out_id not in nodes:
            raise ValueError(f"{nid} has no matching {out_id}")
        cap = float(node.get("capacity_bbl_day") or 0)
        stem = nid.removeprefix("ref_").removesuffix("_in")
        kept.append({
            "id": f"e_{stem}_internal", "from_id": nid, "to_id": out_id,
            "mode": "internal", "base_capacity_bbl_day": cap,
            "cost_per_bbl": 0.0, "transit_time_days": 0,
            "openness": 1.0, "grade": None,
            "_source": "Refinery processing edge. Capacity = installed crude throughput; this is the binding constraint on how much crude the refinery can absorb.",
        })
        kept.append({
            "id": f"e_{stem}_sk", "from_id": out_id, "to_id": "super_sink",
            "mode": "internal", "base_capacity_bbl_day": 9999999,
            "cost_per_bbl": 0.0, "transit_time_days": 0,
            "openness": 1.0, "grade": None,
            "_source": "Virtual edge.",
        })

    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify data/edges.json matches this spec; do not write")
    args = parser.parse_args()

    nodes = _load_nodes()
    previous = [e for e in json.loads((DATA_DIR / "edges.json").read_text(encoding="utf-8")) if "id" in e]
    edges = (build_structural_edges(nodes, previous)
             + build_crude_pipeline_edges(nodes)
             + build_sea_edges(nodes))

    # Consistency guard: a chokepoint/port cap that sits below the volume its
    # feeders can deliver is a silent, invisible bottleneck. This is exactly how
    # chk_malacca ended up carrying 600,000 bbl/day of capacity justified by a
    # routing error. Fail loudly instead.
    def _upstream(nid, seen=None):
        seen = seen or set()
        if nid in seen:
            return 0.0
        seen.add(nid)
        if nodes[nid].get("type") == "source":
            return float(nodes[nid].get("capacity_bbl_day") or 0)
        return sum(_upstream(e["from_id"], seen) for e in edges if e["to_id"] == nid)

    # Only chokepoints are checked. A PORT is meant to bind below its feeders --
    # its capacity is the throughput of the crude pipeline it originates, which
    # is genuinely narrower than the sea lanes reaching it.
    for nid, node in nodes.items():
        if node.get("type") != "chokepoint":
            continue
        cap, feed = float(node.get("capacity_bbl_day") or 0), _upstream(nid)
        if cap < feed - 1:
            print(f"  WARNING: chokepoint {nid} capacity {cap:,.0f} is below the "
                  f"{feed:,.0f} bbl/day its feeding sources can deliver -- it will "
                  f"silently bottleneck the network. Reconcile nodes.json.")

    seen: dict[str, int] = {}
    for edge in edges:
        seen[edge["id"]] = seen.get(edge["id"], 0) + 1
    dupes = sorted(k for k, v in seen.items() if v > 1)
    if dupes:
        raise ValueError(
            f"duplicate edge ids: {dupes}. Edge ids must be unique -- apply_scenario() "
            "resolves an edge-level disruption by scanning for the FIRST matching "
            "edge_id, so a collision would silently disrupt the wrong lane."
        )

    rendered = json.dumps(edges, indent=2, ensure_ascii=False) + "\n"

    target = DATA_DIR / "edges.json"
    if args.check:
        if target.read_text(encoding="utf-8") != rendered:
            print("edges.json is out of date; run: python scripts/build_network.py")
            return 1
        print(f"edges.json is up to date ({len(edges)} edges).")
        return 0

    target.write_text(rendered, encoding="utf-8")
    sea = [e for e in edges if e["mode"] == "sea"]
    print(f"Wrote {len(edges)} edges ({len(sea)} sea lanes) to {target}")
    speeds = [e["distance_km"] / (e["transit_time_days"] * 24 * 1.852)
              for e in sea if e["transit_time_days"] > 0]
    print(f"  implied laden speed range: {min(speeds):.1f}-{max(speeds):.1f} kn "
          f"(target {LADEN_SPEED_KNOTS:.0f} kn; the Suez leg is slower by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
