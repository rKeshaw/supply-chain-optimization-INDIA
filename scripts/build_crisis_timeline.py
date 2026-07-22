"""Regenerate data/crisis_timeline.json.

Run: python scripts/build_crisis_timeline.py

The replay walks the 2026 Strait of Hormuz crisis, which is live as of
2026-07-21: the strait was closed on 11 July after the 8 July collapse of the
June interim agreement, and the Houthis declared a blockade of Saudi shipping
on 20 July. Two deliberately irrelevant headlines are included so the
extraction agent's "unrelated" classification stays demonstrable, and one
genuine but minor energy headline sits below the significance threshold.

severity/confidence ranges are midpointed by
agents.extraction_agent.event_from_curated_timeline, so a range of [0.9, 1.0]
becomes 0.95. signal strength = severity x confidence, and
recompute_significance_threshold is 0.12.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# (date, source, url, headline, body, market_impact, entity, event_type, element, sev, conf)
EVENTS = [
    ("2026-02-28T05:00:00Z", "Reuters", "https://www.reuters.com/world/middle-east/",
     "US and Israel launch coordinated airstrikes on Iran; Tehran warns Gulf shipping",
     "The United States and Israel carried out coordinated airstrikes on Iranian military and nuclear "
     "sites early Saturday. Iran's Supreme National Security Council warned that it would treat the "
     "Strait of Hormuz as a zone of active conflict and that vessels linked to the attacking states "
     "transited at their own risk. Tanker owners immediately began pausing Gulf fixtures.",
     "Brent opened sharply higher; war risk insurance for Gulf transits repriced within hours.",
     "Strait of Hormuz", "capacity_reduction", "chk_hormuz", [0.5, 0.8], [0.85, 1.0]),

    ("2026-03-02T11:00:00Z", "Bloomberg", "https://www.bloomberg.com/energy",
     "Iran declares Strait of Hormuz closed; tanker traffic collapses",
     "Iran formally declared the Strait of Hormuz closed to commercial traffic on Monday, days after "
     "US and Israeli strikes. Tanker transits through the strait fell to almost nothing within 48 "
     "hours. Roughly a fifth of global oil supply normally passes through the waterway.",
     "Largest single-session move in Brent since 2022.",
     "Strait of Hormuz", "closure", "chk_hormuz", [0.85, 1.0], [0.8, 0.95]),

    ("2026-03-08T16:00:00Z", "Financial Times", "https://www.ft.com/commodities",
     "Brent crude passes $100 a barrel as Hormuz closure holds",
     "Brent crude passed $100 a barrel on Sunday, the first time since 2022, as the closure of the "
     "Strait of Hormuz entered its second week with no sign of resolution. Analysts warned that a "
     "sustained closure would push prices substantially higher.",
     "Brent above $100; March 2026 went on to record the largest-ever monthly increase in oil prices, "
     "peaking at $126.",
     "Strait of Hormuz", "price_shock", "chk_hormuz", [0.5, 0.8], [0.95, 1.0]),

    ("2026-03-14T07:00:00Z", "Press Information Bureau", "https://pib.gov.in/",
     "India launches Operation Sankalp to escort Indian-flagged vessels out of the Gulf",
     "The Indian Navy has begun escorting Indian-flagged merchant vessels out of the Persian Gulf "
     "under Operation Sankalp. Five India-flagged LPG carriers are being evacuated. The Ministry of "
     "Petroleum and Natural Gas said refiners had been directed to accelerate non-Gulf procurement.",
     "Indian refiners shifted heavily to Atlantic-basin and Russian barrels.",
     "Strait of Hormuz", "capacity_reduction", "chk_hormuz", [0.6, 0.85], [0.9, 1.0]),

    ("2026-03-27T09:30:00Z", "Reuters", "https://www.reuters.com/markets/commodities/",
     "Iran bars vessels of US, Israel and allied states from Strait of Hormuz",
     "Iran announced that the Strait of Hormuz would remain closed specifically to vessels flagged or "
     "operated by the United States, Israel and their allies, while signalling that other traffic "
     "might be permitted case by case. Shipowners described the guidance as unworkable in practice.",
     "Freight rates for Gulf loadings reached multi-year highs.",
     "Strait of Hormuz", "closure", "chk_hormuz", [0.75, 0.95], [0.85, 1.0]),

    ("2026-04-13T06:00:00Z", "Associated Press", "https://apnews.com/",
     "US Navy begins blockade of Iranian oil export terminals",
     "The United States began a naval blockade of Iranian oil export terminals, including Kharg "
     "Island, in an effort to halt Iranian crude exports entirely. Iran condemned the action as an "
     "act of war.",
     "Iranian exports, already under sanctions, fell close to zero.",
     "Iran", "sanction", "src_iran", [0.7, 0.95], [0.9, 1.0]),

    ("2026-05-04T08:00:00Z", "Lloyd's List", "https://www.lloydslist.com/",
     "Operation Project Freedom begins escorting merchant ships through Hormuz",
     "A multinational naval escort operation began convoying merchant vessels through the Strait of "
     "Hormuz. Transit volumes recovered partially, though well below pre-crisis levels, and only "
     "convoyed vessels were moving.",
     "Partial restoration of Gulf liftings; freight remained elevated.",
     "Strait of Hormuz", "reopening", "chk_hormuz", [0.25, 0.5], [0.8, 0.95]),

    ("2026-05-29T12:00:00Z", "Reuters", "https://www.reuters.com/world/middle-east/",
     "US lifts naval blockade of Iranian ports as talks progress",
     "The United States ended its naval blockade of Iranian oil terminals as indirect talks with "
     "Tehran progressed. Iranian officials described the move as a precondition for any wider "
     "de-escalation.",
     "Iranian export capacity nominally restored, though sanctions remain in force.",
     "Iran", "reopening", "src_iran", [0.5, 0.8], [0.85, 1.0]),

    ("2026-06-08T14:00:00Z", "Reuters", "https://www.reuters.com/business/aerospace-defense/",
     "Houthi missiles strike two commercial vessels in Gulf of Aden",
     "Houthi forces struck the M/V Tavvishi and the M/V Norderney with missiles in the Gulf of Aden, "
     "saying both operators had called at Israeli ports. The group threatened to resume systematic "
     "targeting of shipping linked to Israel in the Red Sea.",
     "Red Sea transits fell again; more tonnage diverted around the Cape of Good Hope.",
     "Bab-el-Mandeb", "capacity_reduction", "chk_bab", [0.55, 0.8], [0.85, 1.0]),

    ("2026-06-17T10:00:00Z", "Associated Press", "https://apnews.com/",
     "US and Iran sign memorandum to end blockades and reopen Hormuz",
     "The United States and Iran signed a memorandum committing both sides to end blockades and "
     "restore normal transit through the Strait of Hormuz. Shipping bodies welcomed the agreement "
     "while cautioning that implementation would determine whether traffic genuinely returned.",
     "Brent fell back sharply from its March peak.",
     "Strait of Hormuz", "reopening", "chk_hormuz", [0.55, 0.85], [0.8, 0.95]),

    ("2026-06-19T09:00:00Z", "US Maritime Administration", "https://www.maritime.dot.gov/msci/",
     "MARAD issues advisory 2026-006 for Red Sea, Bab el-Mandeb and Gulf of Aden",
     "The US Maritime Administration issued advisory 2026-006 covering the Red Sea, Bab el-Mandeb "
     "Strait, Gulf of Aden, Arabian Sea and Somali Basin, superseding 2025-012 and running to "
     "22 September 2026. US-flagged vessels were advised that transmitting AIS increases targeting "
     "risk and were strongly advised to switch transponders off.",
     "Insurance rates for Red Sea transits rose; AIS coverage of the corridor degraded.",
     "Bab-el-Mandeb", "capacity_reduction", "chk_bab", [0.4, 0.65], [0.9, 1.0]),

    ("2026-06-25T11:00:00Z", "Times of India", "https://timesofindia.indiatimes.com/",
     "India beat England by seven wickets to take the T20 series at Eden Gardens",
     "India sealed the T20 series with a seven-wicket win over England at Eden Gardens on Thursday, "
     "chasing down 189 with eleven balls to spare. The captain praised the middle order's composure.",
     "None - unrelated to energy markets.",
     "India cricket team", "unrelated", None, [0.0, 0.05], [0.0, 0.1]),

    ("2026-07-02T13:00:00Z", "Press Information Bureau", "https://pib.gov.in/",
     "Petroleum Minister says India holds 60 days of crude stocks",
     "Union Petroleum and Natural Gas Minister Hardeep Singh Puri said India held crude oil stocks "
     "sufficient for 60 days, LNG inventories for 60 days and LPG inventories for 45 days. He added "
     "that about 70 per cent of India's crude imports were now routed outside the Strait of Hormuz, "
     "against roughly 55 per cent earlier.",
     "Reassurance statement; no supply disruption implied.",
     "India", "capacity_reduction", "chk_hormuz", [0.05, 0.15], [0.9, 1.0]),

    ("2026-07-08T18:00:00Z", "Reuters", "https://www.reuters.com/world/middle-east/",
     "Interim US-Iran agreement collapses after fresh Iranian attacks",
     "The interim agreement reached in June collapsed after fresh Iranian attacks on shipping. Both "
     "sides accused the other of violating the memorandum. Naval escort operations were suspended "
     "pending review.",
     "Brent reversed most of its June decline within two sessions.",
     "Strait of Hormuz", "capacity_reduction", "chk_hormuz", [0.6, 0.85], [0.85, 1.0]),

    ("2026-07-11T04:00:00Z", "Bloomberg", "https://www.bloomberg.com/energy",
     "Iran closes Strait of Hormuz again after attack on transiting vessel",
     "Iran closed the Strait of Hormuz for a second time on Saturday after attacking a vessel it said "
     "had transited an unauthorised route. The US and Iran exchanged strikes overnight. Tanker "
     "owners again suspended Gulf fixtures.",
     "India's refiners were reported to have secured crude supplies through August.",
     "Strait of Hormuz", "closure", "chk_hormuz", [0.9, 1.0], [0.9, 1.0]),

    ("2026-07-14T15:00:00Z", "Financial Times", "https://www.ft.com/commodities",
     "Brent breaches $84 as US-Iran tensions escalate again",
     "Brent crude rose above $84 a barrel on Tuesday, its highest in nearly a month, as renewed "
     "US-Iran hostilities raised fears of sustained supply disruption. Analysts put the risk premium "
     "at $10 to $15 a barrel.",
     "Brent above $84; risk premium estimated at $10-15/bbl.",
     "Strait of Hormuz", "price_shock", "chk_hormuz", [0.45, 0.7], [0.95, 1.0]),

    ("2026-07-18T10:00:00Z", "Economic Times", "https://economictimes.indiatimes.com/tech",
     "Indian IT services firms report stronger-than-expected June quarter earnings",
     "India's largest IT services companies reported stronger-than-expected earnings for the June "
     "quarter, citing a recovery in discretionary technology spending among North American clients. "
     "Shares rose in early trade.",
     "None - unrelated to energy markets.",
     "Indian IT services", "unrelated", None, [0.0, 0.05], [0.0, 0.1]),

    ("2026-07-20T08:00:00Z", "Reuters", "https://www.reuters.com/business/energy/",
     "Houthis declare maritime blockade of Saudi shipping at Bab el-Mandeb",
     "Yemen's Houthis declared an immediate maritime blockade of Saudi-linked shipping at the Bab "
     "el-Mandeb gateway to the Red Sea. The declaration widens the group's targeting beyond "
     "Israeli-linked vessels and raises the prospect of simultaneous disruption at both of the "
     "region's chokepoints.",
     "Compounding risk: Hormuz closed and Bab el-Mandeb under blockade at the same time.",
     "Bab-el-Mandeb", "closure", "chk_bab", [0.7, 0.95], [0.85, 1.0]),
]


def main() -> None:
    out = []
    for i, (ts, src, url, headline, body, impact, entity, etype, elem, sev, conf) in enumerate(EVENTS, 1):
        out.append({
            "id": f"evt_{i:03d}",
            "source": src,
            "url": url,
            "original_timestamp": ts,
            "replay_order": i,
            "headline": headline,
            "body_excerpt": body,
            "known_market_impact": impact,
            "expected_extraction": {
                "entity": entity,
                "event_type": etype,
                "affected_graph_element": elem,
                "severity_range": list(sev),
                "confidence_range": list(conf),
            },
        })
    (DATA / "crisis_timeline.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {len(out)} events, {out[0]['original_timestamp'][:10]} to {out[-1]['original_timestamp'][:10]}")
    for e in out:
        x = e["expected_extraction"]
        strength = (sum(x["severity_range"]) / 2) * (sum(x["confidence_range"]) / 2)
        verdict = "unrelated" if x["event_type"] == "unrelated" else (
            "ACTS" if strength >= 0.12 else "below threshold")
        print(f"  {e['original_timestamp'][:10]}  {str(x['affected_graph_element'] or '-'):<12} "
              f"{strength:5.3f}  {verdict}")


if __name__ == "__main__":
    main()
