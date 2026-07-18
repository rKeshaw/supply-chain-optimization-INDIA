# AI-Driven Energy Supply Chain Resilience

An anticipatory decision-support system for India's crude oil supply chain. It monitors geopolitical and logistics risk signals, models disruption scenarios and their downstream economic impact, and generates executable procurement-rerouting recommendations — turning a reactive crisis response into a managed process.

India imports ~88% of its crude, with 40–45% transiting the Strait of Hormuz. Its Strategic Petroleum Reserves cover roughly 9.5 days of consumption. This system exists to compress the time between a disruption signal and an actionable response.

---

## What it does

- **Senses risk** from news, sanctions registries, and marine weather, and converts free-text signals into schema-validated events.
- **Models disruptions** — Hormuz closure, Red Sea suspension, OPEC+ cut — and computes the cascade through refinery run rates, crude and retail prices, GDP, and power-sector stress.
- **Reroutes procurement** with a constraint-aware optimizer that respects crude-grade compatibility, source and chokepoint capacity, and diversification limits — presenting cheapest, fastest, and lowest-risk options.
- **Optimizes the strategic reserve**, scheduling SPR drawdown against refinery safety floors and modelling replenishment as disruption eases.
- **Runs a digital twin** — a daily, cargo-level simulation over a configurable horizon — behind a geospatial command-center UI.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
   Signal sources       │              Agent layer                │
  (news / sanctions /   │                                         │
   weather / replay) ──▶│  Extraction → Graph update → Routing →   │
                        │  SPR → Economic cascade → Policy Critic  │
                        │            ↕ (re-solve loop) ↓           │
                        │              Explainer brief             │
                        └─────────────────┬───────────────────────┘
                                          │
                        ┌─────────────────▼───────────────────────┐
                        │           Graph engine                   │
                        │  Constraint-aware min-cost-flow (OR-Tools)│
                        │  Digital twin · SPR optimizer · N-1 /    │
                        │  HHI resilience · economic model         │
                        └─────────────────┬───────────────────────┘
                                          │
                     FastAPI (REST + WebSocket)  ──▶  Web command center
                                                      (MapLibre + deck.gl)
```

The agent pipeline is orchestrated with **LangGraph** as a stateful workflow, including a feedback loop in which the Policy Critic can send an infeasible plan back to the optimizer with tightened constraints before any brief is emitted.

### Layout

| Path | Responsibility |
|---|---|
| `agents/` | Signal extraction, orchestration, policy critic, explainer, scenario, and live-sensing adapters |
| `graph_engine/` | Graph construction, routing LP, digital twin, SPR optimizer, economic model, resilience analytics |
| `api/` | FastAPI application (REST + `/ws/live` push channel) and provider settings |
| `frontend/` | Single-page geospatial command center |
| `data/` | Network model (`nodes.json`, `edges.json`), `parameters.json`, and the curated `crisis_timeline.json` replay |
| `tests/` | Correctness and consistency suite, including independent optimality verification |

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

The demo runs from the curated crisis timeline by default and requires no external feeds. Live sensing is opt-in (see below).

---

## Using it

- **Replay** — `POST /api/replay/run` steps through the 2025 crisis timeline event by event.
- **Scenarios** — apply a named disruption (`hormuz_full`, `red_sea_suspension`, `opec_cut`, …) via `POST /api/scenario/apply`, or click any node or corridor in the UI to disrupt it directly.
- **Custom signal** — submit raw headline text to `POST /api/signal` and watch it flow through extraction, rerouting, and the brief.
- **Digital twin** — `POST /api/twin/simulate` projects inventory, fulfillment, and SPR draw over the horizon.

### Live sensing (optional)

A background loop can poll real, free, unauthenticated sources — the OFAC SDN sanctions list, GDELT news, and marine weather — and feed anything it finds through the same pipeline. It is disabled by default so the demo stays deterministic:

```
LIVE_INGESTION_ENABLED=true
```

`POST /api/live/poll-now` triggers a single poll cycle on demand, with the loop on or off.

---

## Design principles

- **One source of truth.** Every reported figure — baseline flow, scenario loss, vulnerability ranking, the economic gap — comes from the same grade- and capacity-aware solve, so the map and the recommendations can never disagree.
- **Provable routing.** The recommended plan is verified against an independent, separately-implemented arc-based min-cost-flow optimum. See `tests/test_routing_optimality.py`.
- **Explicit, testable assumptions.** Every economic parameter in `data/parameters.json` is named, sourced, and status-tagged (`VERIFIED` / `ESTIMATED` / `DESIGN_DEFAULT`). The explainer agent may only restate numbers produced by the model — it cannot invent figures.
- **Honest scope.** The model covers five named refineries (~51% of national capacity). National figures are reported as lower-bound exposure, never extrapolated forecasts, and market risk premia are kept distinct from modelled physical shortfall.

---

## Tests

```bash
pytest -q
```

The suite covers the graph engine, routing optimality, flow consistency, digital twin, agents, and the API surface.

---

## Data note

The network model, crisis timeline, and parameters are curated for demonstration and calibrated against public reporting. Sources and confidence status are documented inline in `data/parameters.json`. Refresh benchmark values before any operational use.
