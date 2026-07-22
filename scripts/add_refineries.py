"""One-shot: extend data/nodes.json to cover India's import-fed refining capacity.

Run once:  python scripts/add_refineries.py

The model covers 16 import-fed refineries. An earlier 5-refinery cut (128.7 MMTPA, 2.574 Mb/d) left every
source node was scoped to that supplier's volume into ALL of India (~5.5 Mb/d).
Supply therefore exceeded demand by 2.14x purely by construction, so once the
routing geography was corrected even a full Hormuz closure showed 0% loss. This
closes the demand side instead of shrinking the supply side.

WHAT IS INCLUDED, AND WHY
-------------------------
Every refinery that runs on IMPORTED crude, whether it receives it at its own
berth/SPM or through a crude pipeline from a coastal terminal. Capacities are
MMTPA converted at 20,000 bbl/day per MMTPA -- the exact convention the existing
data already uses (Jamnagar 68.2 -> 1,364,000; Paradip 15.0 -> 300,000).

WHAT IS EXCLUDED, AND WHY  (18.0 MMTPA / 360,000 bbl/day)
---------------------------------------------------------
Refineries that do not draw on the seaborne import corridors this model exists
to analyse. Counting them as import demand would overstate it:

  Pachpadra / Barmer (HRRL)   9.00 MMTPA  Rajasthan Mangala domestic crude
  Numaligarh (NRL)            3.00 MMTPA  Assam domestic crude, inland pipeline
  Bongaigaon (IOCL)           2.35 MMTPA  Assam domestic crude
  Guwahati (IOCL)             1.00 MMTPA  Assam domestic crude
  Tatipaka (ONGC)             1.00 MMTPA  KG-basin domestic crude
  Nagapattinam (CPCL)         1.00 MMTPA  shut; being rebuilt to 9 MMTPA
  Digboi (IOCL)               0.65 MMTPA  Assam domestic crude

PORTS
-----
A crude pipeline has to start at a marine terminal, so the terminals that feed
inland refineries are modelled explicitly as `port` nodes rather than folded
invisibly into a sea lane. Only those four are added. Coastal refineries keep
the existing convention where the refinery_in node doubles as its own receipt
terminal (Jamnagar's inlet is the Sikka SPM, Kochi's is the Kochi SPM, and so
on) -- the smaller dedicated berths are deliberately not modelled as separate
nodes, and that simplification is noted here rather than hidden.

SOURCES
-------
PPAC / MoPNG installed refinery capacity; company pages for HMEL, BORL, CPCL.
Pipeline lengths and capacities: IndianOil crude pipelines (SMPL 2,663 km /
25 MMTPA; PHBPL 1,465 km / 20.4 MMTPA; MPPL 1,194 km / 8.4 MMTPA), BPCL
(Vadinar-Bina 937 km / 7.8 MMTPA), HMEL (Mundra-Bathinda 1,017 km / 11.25 MMTPA).
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BBL_PER_MMTPA = 20_000
STAMP = "2026-07-15T00:00:00Z"

# (id_stem, display name, lat, lon, MMTPA, receipt, source note)
REFINERIES = [
    ("vadinar", "Vadinar (Nayara Energy)", 22.29, 69.72, 20.0, "coastal",
     "PPAC/MoPNG installed capacity 20.0 MMTPA. India's second-largest single-site refinery. Receives crude at its own Vadinar SPM in the Gulf of Kutch. Majority Rosneft-linked ownership; the principal processor of Russian Urals in India, and itself the subject of EU sanctions listed in July 2025 -- the most geopolitically exposed refinery in the country and the most conspicuous omission from the previous 5-refinery model."),
    ("mumbai_bpcl", "Mumbai (BPCL)", 19.00, 72.90, 12.0, "coastal",
     "PPAC/MoPNG installed capacity 12.0 MMTPA. Mahul, Maharashtra. Receives crude via the Jawahar Dweep offshore terminal in Mumbai harbour."),
    ("chennai_cpcl", "Manali / Chennai (CPCL)", 13.16, 80.26, 10.5, "coastal",
     "PPAC/MoPNG installed capacity 10.5 MMTPA. Chennai Petroleum Corporation, an IOCL subsidiary. Receives crude at Chennai port. This is India's only other east-coast import refinery besides Paradip and Visakhapatnam."),
    ("mumbai_hpcl", "Mumbai (HPCL)", 19.01, 72.89, 9.5, "coastal",
     "PPAC/MoPNG installed capacity 9.5 MMTPA. Mahul, Maharashtra. Shares the Jawahar Dweep receipt terminal with the adjacent BPCL refinery."),
    ("haldia", "Haldia (IOCL)", 22.03, 88.10, 8.0, "coastal",
     "PPAC/MoPNG installed capacity 8.0 MMTPA. West Bengal. Receives crude at Haldia Dock and is also a delivery point on the Paradip-Haldia-Barauni crude pipeline; both routes are modelled."),
    ("panipat", "Panipat (IOCL)", 29.33, 76.97, 15.0, "inland",
     "PPAC/MoPNG installed capacity 15.0 MMTPA (expansion to 25 MMTPA under way). Haryana. Landlocked: fed by the Salaya-Mathura pipeline and by the Mundra-Panipat pipeline."),
    ("koyali", "Koyali / Vadodara (IOCL)", 22.40, 73.13, 13.7, "inland",
     "PPAC/MoPNG installed capacity 13.7 MMTPA (expansion to 18 MMTPA under way). Gujarat. Landlocked: fed by the Salaya-Mathura crude pipeline from the Gulf of Kutch."),
    ("bathinda", "Guru Gobind Singh, Bathinda (HMEL)", 30.03, 74.95, 11.3, "inland",
     "PPAC/MoPNG installed capacity 11.3 MMTPA. HPCL-Mittal Energy, Punjab. Landlocked: fed exclusively by the 1,017 km Mundra-Bathinda crude pipeline. The single largest refinery missing from the previous model."),
    ("mathura", "Mathura (IOCL)", 27.47, 77.72, 8.0, "inland",
     "PPAC/MoPNG installed capacity 8.0 MMTPA. Uttar Pradesh. Landlocked: fed by the Salaya-Mathura crude pipeline."),
    ("bina", "Bina (BORL / BPCL)", 24.18, 78.23, 7.8, "inland",
     "PPAC/MoPNG installed capacity 7.8 MMTPA. Bharat Oman Refineries, Madhya Pradesh. Landlocked: fed exclusively by the 937 km Vadinar-Bina crude pipeline."),
    ("barauni", "Barauni (IOCL)", 25.47, 85.97, 6.0, "inland",
     "PPAC/MoPNG installed capacity 6.0 MMTPA (expansion to 9 MMTPA under way). Bihar. Landlocked: fed by the Paradip-Haldia-Barauni crude pipeline."),
]

# (id, display name, lat, lon, capacity MMTPA, source note)
PORTS = [
    ("port_salaya", "Salaya Crude Terminal (Gulf of Kutch)", 22.32, 69.60, 25.0,
     "Origin of IndianOil's 2,663 km Salaya-Mathura crude pipeline (SMPL), installed capacity 25 MMTPA, which supplies the Koyali, Mathura and Panipat refineries. Modelled as its own node because a crude pipeline must originate at a marine terminal; folding it into a sea lane would hide both the terminal and the pipeline."),
    ("port_vadinar", "Vadinar Crude Terminal (BPCL)", 22.28, 69.70, 7.8,
     "Origin of the 937 km Vadinar-Bina crude pipeline, installed capacity 7.8 MMTPA, the sole crude supply to the BORL Bina refinery. Physically adjacent to the Nayara Vadinar refinery but a separate BPCL installation, so it is modelled separately rather than merged."),
    ("port_mundra", "Mundra Crude Terminal (Gulf of Kutch)", 22.74, 69.70, 19.65,
     "Origin of two crude pipelines: IndianOil's 1,194 km Mundra-Panipat line (8.4 MMTPA) and HMEL's 1,017 km Mundra-Bathinda line (11.25 MMTPA). Capacity 19.65 MMTPA is the sum of the two."),
    ("port_paradip", "Paradip Crude Terminal (Odisha)", 20.27, 86.68, 20.4,
     "Origin of IndianOil's 1,465 km Paradip-Haldia-Barauni crude pipeline (PHBPL), installed capacity 20.4 MMTPA. Seaborne crude for the Paradip refinery now lands here and moves the short distance to the refinery inlet, so that the same terminal can also feed Haldia and Barauni inland -- which is how it works in reality."),
]


def main() -> None:
    path = DATA / "nodes.json"
    nodes = json.loads(path.read_text(encoding="utf-8"))
    have = {n["id"] for n in nodes}
    added = []

    for pid, name, lat, lon, mmtpa, note in PORTS:
        if pid in have:
            continue
        nodes.append({
            "id": pid, "type": "port", "name": name, "lat": lat, "lon": lon,
            "capacity_bbl_day": round(mmtpa * BBL_PER_MMTPA),
            "grade_compatibility": ["SWEET", "SOUR"],
            "inventory_bbl": None, "consumption_rate_bbl_day": None,
            "openness": 1.0, "risk_score": 0.0, "confidence": 1.0,
            "last_updated": STAMP, "_source": note,
        })
        added.append((pid, mmtpa))

    for stem, name, lat, lon, mmtpa, receipt, note in REFINERIES:
        bbl = round(mmtpa * BBL_PER_MMTPA)
        for suffix, ntype in (("_in", "refinery_in"), ("_out", "refinery_out")):
            nid = f"ref_{stem}{suffix}"
            if nid in have:
                continue
            entry = {
                "id": nid, "type": ntype, "name": f"{name} — {'Inlet' if suffix == '_in' else 'Outlet'}",
                "lat": lat, "lon": lon, "capacity_bbl_day": bbl,
                # Every Indian refinery in this set runs a mixed sweet/sour slate.
                # Per-refinery crude-slate limits are a real constraint but there
                # is no public per-refinery specification to base them on, so
                # none is invented here; the only grade restriction in the model
                # remains Paradip's SOUR design, which was already authored.
                "grade_compatibility": ["SWEET", "SOUR"],
                # Three days of throughput, matching the ratio the five existing
                # refineries already use (Jamnagar 4,092,000 / 1,364,000 = 3.0).
                "inventory_bbl": bbl * 3 if ntype == "refinery_out" else None,
                "consumption_rate_bbl_day": bbl if ntype == "refinery_out" else None,
                "openness": 1.0, "risk_score": 0.0, "confidence": 1.0,
                "last_updated": STAMP,
                "_source": note,
                "_note": f"Crude receipt: {receipt}.",
            }
            nodes.append(entry)
        added.append((f"ref_{stem}", mmtpa))

    path.write_text(json.dumps(nodes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    refin = sum(n.get("consumption_rate_bbl_day") or 0 for n in nodes if n["type"] == "refinery_out")
    print(f"added {len(added)} entities")
    print(f"modelled import-fed refining demand: {refin:,.0f} bbl/day "
          f"({refin / BBL_PER_MMTPA:.1f} MMTPA)")


if __name__ == "__main__":
    main()
