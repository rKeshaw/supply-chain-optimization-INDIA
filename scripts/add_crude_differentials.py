"""One-shot: add crude_differential_usd_per_bbl and supplier_group to source nodes.

Run once: python scripts/add_crude_differentials.py

Differentials are quoted against Brent, in USD/bbl, for AUGUST 2026 loading --
the month a procurement desk is buying as of 2026-07-21. They are a SNAPSHOT,
not a stable property: Saudi Arab Light's Asia OSP ran +$19.50 (May) -> +$15.50
(June) -> +$9.50 (July) -> -$1.50 (August) as the Hormuz risk premium unwound,
and Urals traded at a PREMIUM from March to June before flipping back to a $10
discount. Refresh before any operational use.

Middle East OSPs are published against the Oman/Dubai average, not Brent. They
are converted here by subtracting the Brent-Dubai EFS, taken as $1.30/bbl (it
ran $0.70-$1.45 through 2026). That conversion is applied consistently and is
noted on each affected node.

CONFIDENCE TIERS -- each node records which applies:
  VERIFIED   a published assessment or OSP was found for this grade
  STRUCTURAL derived from a benchmark identity, not estimated
  ESTIMATED  no public assessment found at this granularity; positioned by
             grade quality relative to the verified points. These are the
             weakest inputs in the model and should not be relied on for a
             procurement ranking without a real price feed.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
EFS = 1.30  # Brent - Dubai, USD/bbl

# node_id: (differential_vs_brent, supplier_group, tier, note)
DIFFERENTIALS = {
    "src_russia_urals": (-10.00, "russia", "VERIFIED",
        "Urals into Indian ports traded at discounts of $10/bbl or more to Dated Brent for August "
        "delivery (Reuters, 7 July 2026), close to the widest since 2022. Urals had been at a PREMIUM "
        "from March to June while Middle East supply was disrupted, then flipped back as Gulf and "
        "Iranian exports were restored."),
    "src_russia_espo": (-8.75, "russia", "VERIFIED",
        "ESPO assessed at ICE Brent minus $8.50-$9.00/bbl (February 2026), widened from $7.50-$8.00. "
        "Midpoint used. Older than the other assessments here."),
    "src_saudi": (-1.50 - EFS, None, "VERIFIED",
        "Aramco set August Arab Light OSP for Asia at $1.50/bbl BELOW the Oman/Dubai average - an $11 "
        f"cut from July, the largest in over two decades and a six-year low. Converted to a Brent "
        f"basis by subtracting the ${EFS:.2f} Brent-Dubai EFS."),
    "src_iraq": (1.35 - EFS, None, "VERIFIED",
        "SOMO set August Basrah Medium OSP for Asia at $1.35/bbl over the Oman/Dubai average, up from "
        f"$0.30 in July. Converted to a Brent basis by subtracting the ${EFS:.2f} Brent-Dubai EFS."),
    "src_kuwait": (-6.00 - EFS, None, "VERIFIED",
        "KPC set Kuwait Export Crude at a $6.00/bbl discount to the Oman/Dubai average. This is the "
        f"JUNE 2026 figure - no August assessment was found - so it is staler than the Saudi and Iraqi "
        f"values. Converted to a Brent basis by subtracting the ${EFS:.2f} Brent-Dubai EFS."),
    "src_nigeria": (1.70, None, "VERIFIED",
        "Bonny Light offered at Dated Brent plus $1.70. NNPC publishes Nigerian OSPs as Dated Brent "
        "differentials, so no basis conversion is needed. The figure is a 2025 market offering rather "
        "than an August 2026 OSP."),
    "src_venezuela": (-9.00, None, "VERIFIED",
        "Merey-16 and Hamaca traded at $8.50-$9.50/bbl under Brent futures in late January 2026. "
        "Highly unstable: offers ranged from -$13 to -$22/bbl earlier that month depending on cargo "
        "and destination. Midpoint of the traded range used."),
    "src_oman": (-EFS, None, "STRUCTURAL",
        "Oman is one of the two grades in the Oman/Dubai marker basket, so its differential to Brent "
        f"is the negative of the Brent-Dubai EFS by identity, not an estimate. EFS taken as ${EFS:.2f}."),

    # No public assessment found at this granularity. Positioned by grade quality
    # (API gravity and sulphur, both recorded in each node's own _source text)
    # relative to the verified points above.
    "src_uae": (0.50, None, "ESTIMATED",
        "Murban, 40.5 API / 0.6% S - light sweet, better quality than Arab Light. ADNOC prices Murban "
        "outright through IFAD futures rather than as a differential (August OSP $80.01/bbl, down 21.2% "
        "from July's $101.48), so no directly comparable differential is published."),
    "src_qatar": (-2.50, None, "ESTIMATED",
        "Qatar Marine, ~33 API / 1.9% S - medium sour, positioned near Arab Light."),
    "src_angola": (0.80, None, "ESTIMATED",
        "Cabinda, ~32 API / 0.15% S - medium sweet, positioned below Bonny Light on gravity."),
    "src_usa": (0.30, None, "ESTIMATED",
        "US Gulf Coast light sweet delivered to Asia, ~38 API / 0.4% S."),
    "src_brazil": (-0.50, None, "ESTIMATED",
        "Tupi/Lula, ~29 API / 0.35% S - medium sweet, low sulphur but heavier."),
    "src_kazakhstan": (-1.50, None, "ESTIMATED",
        "CPC Blend, ~45 API / 0.55% S - light sweet, but freight-disadvantaged into Asia. Asian buying "
        "rose in 2026 as European demand eased, widening CPC discounts by roughly $1/bbl."),
    "src_iran": (-12.00, None, "ESTIMATED",
        "Iranian Heavy, ~30 API / 1.8% S. Sanctioned barrels clear at a deep discount well beyond "
        "quality; no transparent assessment exists. Only reachable at all when "
        "routing_policy.allow_sanctions_restricted_sources is enabled."),
    "src_mumbai_high": (1.00, None, "ESTIMATED",
        "Mumbai High, ~39 API / 0.15% S - light sweet. Domestic crude is sold to Indian refiners on "
        "an import-parity formula rather than a published Brent differential."),
    "src_gujarat_onshore": (0.50, None, "ESTIMATED",
        "Gujarat onshore, waxy light low-sulphur. Domestic crude, import-parity priced."),
}


def main() -> None:
    path = DATA / "nodes.json"
    nodes = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for n in nodes:
        spec = DIFFERENTIALS.get(n["id"])
        if spec is None:
            continue
        diff, group, tier, note = spec
        n["crude_differential_usd_per_bbl"] = round(diff, 2)
        if group:
            n["supplier_group"] = group
        n["_differential_source"] = f"[{tier}] {note}"
        seen.add(n["id"])

    missing = [nid for nid, n in ((x["id"], x) for x in nodes)
               if n.get("type") == "source" and nid not in seen]
    if missing:
        raise ValueError(f"source nodes without a differential: {missing}")

    path.write_text(json.dumps(nodes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'source':<22}{'$/bbl':>8}  tier        group")
    for n in sorted((x for x in nodes if x.get("type") == "source"),
                    key=lambda x: x["crude_differential_usd_per_bbl"]):
        tier = n["_differential_source"].split("]")[0].strip("[")
        print(f"  {n['id']:<20}{n['crude_differential_usd_per_bbl']:>8.2f}  {tier:<11} "
              f"{n.get('supplier_group', '-')}")


if __name__ == "__main__":
    main()
