# AI-Driven Energy Supply Chain Resilience

An anticipatory decision-support system for India's crude oil supply chain. It watches geopolitical and logistics risk signals, models disruption scenarios and their downstream economic impact, and turns that into procurement and rerouting recommendations a desk can act on — instead of a reactive scramble once the disruption has already landed.

India imports roughly 88% of its crude, with 40–45% of it transiting the Strait of Hormuz. Strategic Petroleum Reserves cover about 9.5 days of consumption. The point of this system is to shrink the time between a disruption signal appearing and a usable response being on someone's desk.

---

## What it does

- **Senses risk** from news, sanctions registries, and marine weather, converting free-text signals into schema-validated events.
- **Models disruptions** — a Hormuz closure, a Red Sea suspension, an OPEC+ cut, or a correlated multi-corridor shock — and works out the cascade through refinery run rates, crude and retail prices, GDP, and power-sector stress.
- **Reroutes procurement** by solving a multi-commodity min-cost-flow problem, not a heuristic search over a shortlist of candidate paths. Grade compatibility, source and chokepoint capacity, and diversification limits are all hard constraints. The result is presented as cheapest, fastest, and lowest-risk options, not one collapsed answer.
- **Optimizes the strategic reserve**, scheduling SPR drawdown against refinery safety floors and modelling replenishment as a disruption eases.
- **Runs a digital twin** — a daily, cargo-level simulation over a configurable horizon — and can run it a second way with adaptive rerouting switched off, so the value of the optimizer shows up as a number (days and dollars) instead of an assertion.
- **Reads the same brief three ways.** A Decision Brief can be viewed from a procurement, refinery-operations, or policy angle without re-running anything — it's a reshaping of the same underlying numbers, not a separate report.

The UI has two views of the same network state side by side: a geospatial map for where things physically sit, and a force-directed graph for when the topology matters more than the geography.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
   Signal sources       │              Agent layer                │
  (news / sanctions /   │                                         │
   weather / replay) ──▶│  Extraction → Graph update → Routing →  │
                        │  SPR → Economic cascade → Policy Critic │
                        │            ↕ (re-solve loop) ↓          │
                        │              Explainer brief            │
                        └─────────────────┬───────────────────────┘
                                          │
                        ┌─────────────────▼───────────────────────┐
                        │           Graph engine                  │
                        │  Arc-based min-cost-flow (OR-Tools) with│
                        │  flow decomposition for path attribution│
                        │  Digital twin · SPR optimizer · N-1 /   │
                        │  HHI resilience · economic model        │
                        └─────────────────┬───────────────────────┘
                                          │
                     FastAPI (REST + WebSocket)  ──▶  Web command center
                                                      (MapLibre + deck.gl + D3)
```

The agent pipeline is orchestrated with **LangGraph** as a stateful workflow, including a feedback loop where the Policy Critic can send an infeasible plan back to the optimizer with tightened constraints before any brief goes out.

### Layout

| Path | Responsibility |
|---|---|
| `agents/` | Signal extraction, orchestration, policy critic, explainer, scenario, and live-sensing adapters |
| `graph_engine/` | Graph construction, the routing solver, digital twin, SPR optimizer, economic model, resilience analytics |
| `api/` | FastAPI application (REST + `/ws/live` push channel) and provider settings |
| `frontend/` | Single-page command center — geospatial map and an alternative network-graph view of the same state |
| `data/` | Network model (`nodes.json`, `edges.json`), `parameters.json`, and the curated `crisis_timeline.json` replay |

---

## Getting started

**Requirements:** Python 3.11+.

```bash
python -m venv venv
venv/Scripts/activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

**Configure the LLM provider.** The agents call the Groq API. Copy the template and add your key(s):

```bash
cp .env.example .env
```

```
GROQ_API_KEYS=key1,key2   # comma-separated keys enable load balancing
```

**Run:**

```bash
uvicorn api.main:app --reload
```

Open **http://localhost:8000** for the command center; the interactive API reference is at **/docs**.

The demo runs from the curated crisis timeline by default and needs no external feeds. Live sensing is opt-in (see below).

---

## Using it

- **Replay** — `POST /api/replay/run` steps through the 2025–2026 crisis timeline event by event.
- **Scenarios** — apply a named disruption (`hormuz_full`, `red_sea_suspension`, `opec_cut`, `correlated_gulf_crisis`, …) via `POST /api/scenario/apply`, or click any node or corridor in the UI to disrupt it directly.
- **Custom signal** — submit raw headline text to `POST /api/signal` and watch it move through extraction, rerouting, and the brief.
- **Digital twin** — `POST /api/twin/simulate` projects inventory, fulfillment, and SPR draw over the horizon; pass `compare_no_reroute` to get the no-adaptive-rerouting counterfactual alongside it.
- **Natural language** — `POST /api/nl-ops` takes something like "model a 60-day Hormuz closure" and turns it into the matching scenario simulation. Not yet wired into the UI, but reachable directly or via `/docs`.

### Live sensing (optional)

A background loop can poll real, free, unauthenticated sources — the OFAC SDN sanctions list, GDELT news, and marine weather — and feed anything it finds through the same pipeline. It's disabled by default so the demo stays deterministic:

```
LIVE_INGESTION_ENABLED=true
```

`POST /api/live/poll-now` triggers a single poll cycle on demand, whether the loop is on or off.

---

## Design principles

- **One source of truth.** Every reported figure — baseline flow, scenario loss, vulnerability ranking, the economic gap — comes out of the same grade- and capacity-aware solve, so the map and the recommendations can never disagree with each other.
- **Provable routing, not a heuristic with good vibes.** The solver is an arc-based multi-commodity min-cost-flow formulation — one variable per edge and grade, flow conservation at every node — rather than an enumerated shortlist of candidate paths. Path-level detail (source, corridor, transit time) is recovered afterward by decomposing the solved flow, which is exact and polynomial, not a re-solve.
- **Explicit, testable assumptions.** Every economic parameter in `data/parameters.json` is named, sourced, and status-tagged (`VERIFIED` / `ESTIMATED` / `DESIGN_DEFAULT`). The explainer agent may only restate numbers the model actually produced — it can't invent a figure.
- **Honest scope.** The model covers five named refineries (roughly 51% of national capacity). National-level figures are reported as a lower-bound exposure, never an extrapolated forecast, and market risk premia are kept separate from modelled physical shortfall.

---

## Data note

The network model, crisis timeline, and parameters are curated for demonstration and calibrated against public reporting. Sources and confidence status are documented inline in `data/parameters.json`. Refresh benchmark values before relying on this for anything operational.
