# India Energy Supply Chain Resilience

A decision support system for India's crude oil imports. It watches for
disruption signals, works out what a disruption does to the network, and
produces a procurement plan that can be acted on.

India imports roughly 88% of the crude it refines, and a large share of that
still moves through the Strait of Hormuz. The question this addresses is narrow
and practical. When a corridor closes, where do the barrels come from instead,
what does that cost, and how long before supply settles.

## What it does

Four things, in the order they happen.

**Reads signals.** News headlines, the OFAC (Office of Foreign Assets Control)
sanctions list and marine weather are converted into structured events carrying
a severity and a confidence. A confirmed closure is handled differently from an
unconfirmed report of the same event. One is a known operating state, the other
is a risk that should fade if nothing reinforces it.

**Re-solves the network.** Any event that clears the significance threshold
triggers a fresh solve of the entire import network. It is a multi-commodity
minimum cost flow across 63 nodes and 146 edges, using OR-Tools. Crude grade,
source export ceilings, strait capacity, pipeline throughput, refinery
compatibility and the sustainable draw limit on the strategic reserve are all
hard constraints. A solve takes about 40 milliseconds.

**Prices it.** The cost objective is landed cost, meaning the crude price at
origin plus the freight to bring it here. This is the difference between a model
that assumes India buys Gulf crude because it is nearby and one that recognises
India buys Russian crude because it is ten dollars a barrel cheaper.

**Simulates forward.** A daily digital twin tracks cargoes in transit, refinery
inventory and reserve drawdown across a 60 day horizon. It can run the same
scenario a second time with adaptive rerouting disabled, which gives a direct
measure of what the optimiser is worth.

## The modelled network

Sixteen refineries totalling 250.5 MMTPA (million metric tonnes per annum),
covering most of India's import-fed refining capacity. The 18 MMTPA excluded
either runs on domestic crude, such as Barmer on Rajasthan production and the
Assam cluster, or is currently shut, so none of it places load on the import
corridors.

At baseline the solver delivers 5.01 million barrels a day at $82.02 a barrel
landed, averaging 14.3 days in transit. The largest suppliers are Russian ESPO
(Eastern Siberia Pacific Ocean) at 20.5%, Iraq at 17.2% and Saudi Arabia at 16%.

Two of those baseline figures are worth reading carefully, because they are
constraints rather than discoveries. Hormuz carries exactly 40% of supply and
Russia supplies exactly 35%, and both numbers sit on their policy ceilings in
`data/parameters.json`. The solver would buy more from both if the ceilings let
it. Anyone comparing the 40% against the widely quoted 40 to 45% Hormuz exposure
figure should know that the model was told to stop there.

A full Hormuz closure removes 32.2% of supply and moves the crude benchmark by
50.6%. That price figure comes from three sourced inputs multiplied together,
the Hormuz transit volume, the bypass pipeline capacity and a demand elasticity,
so treat it as a construction rather than a calibration.

## Diversification ceilings are priced rather than absolute

The supplier, chokepoint and Cape of Good Hope ceilings express procurement
policy. Enforced absolutely they produce a bad answer, because the solver will
leave a refinery short rather than exceed a self-imposed guideline by a fraction
of a percent, and that shortfall then reaches the economic model as a physical
supply loss driving the growth drag and the power stress flag.

They are therefore soft, with the breach priced. The ranking is what makes the
behaviour defensible. A breach costs $250 a barrel, which is far above any
realistic cost spread between sources, so the ceilings hold during ordinary
optimisation. It sits below the $500 a barrel charge on strategic reserve draw
and far below the penalty on unserved demand, so the solver will exceed a
ceiling ahead of draining the reserve, and drain the reserve ahead of running a
refinery short.

Whatever goes past a ceiling is reported. The interface names the constraint and
the volume, so a barrel withheld by policy stays distinguishable from one the
network genuinely cannot deliver.

Undisrupted, the plan sits about 24,500 barrels a day over the Russian supplier
ceiling. Vizag cannot be filled inside every ceiling at once, and half a
percentage point of extra concentration is the cheapest of the three available
concessions.

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv venv
venv/Scripts/activate          # Windows
# source venv/bin/activate     # macOS and Linux
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open http://localhost:8000. The API (application programming interface)
reference is at `/docs`.

The agents call Groq for signal extraction and for writing the decision brief.
Without a key the system still runs end to end, since the replay uses
pre-validated events and the brief falls back to a deterministic version
assembled from module output. To enable the language model:

```bash
cp .env.example .env
# GROQ_API_KEYS=key1,key2    comma separated keys are load balanced
```

Live sensing is disabled by default so that demonstrations stay reproducible.
Enable it with `LIVE_INGESTION_ENABLED=true`, or call `POST /api/live/poll-now`
for a single cycle.

## Working with it

`POST /api/replay/run` advances the crisis timeline one event at a time. The
timeline covers the 2026 Strait of Hormuz crisis, 18 events running from the
airstrikes on 28 February to the Houthi blockade declaration on 20 July. Two
unrelated headlines are included so the extraction agent can be seen rejecting
them, and one genuine but minor energy story sits below the action threshold.

`POST /api/scenario/apply` applies a named scenario such as `hormuz_full`,
`red_sea_suspension`, `opec_cut` or `correlated_gulf_crisis`. Selecting any node
or corridor on the map allows its availability to be set directly. Named
scenarios stack, so applying a second one never reverses the first, while the map
control assigns outright.

`POST /api/twin/simulate` runs the forward simulation. Pass `compare_no_reroute`
to return the counterfactual alongside it.

`POST /api/signal` accepts raw headline text and pushes it through the full
pipeline, which is the quickest way to exercise the system end to end.

`GET /api/backtest/april2025` decomposes the April 2025 US-Iran standoff. Brent
rose 8% in a single session on an escalation that removed no barrels from the
world market, so the physical channel prices it at zero and the entire move
resolves as $6.24 a barrel of risk premium.

`POST /api/nl-ops` accepts a request such as "model a 60 day Hormuz closure" and
converts it into the corresponding simulation. It works but is not yet surfaced
in the interface.

## Architecture

```
signals (news, sanctions, weather, replay)
        |
        v
  extraction  ->  graph update  ->  routing  ->  reserve  ->  economics  ->  critic
                                       ^                                      |
                                       +--------- re-solve once --------------+
                                                                              |
                                                                              v
                                                                       decision brief
        |
        v
  FastAPI (REST and websocket)  ->  browser interface (MapLibre, deck.gl, D3)
```

The agent pipeline is a LangGraph state machine. The policy critic can return an
infeasible plan to the optimiser with tightened constraints, but only once per
signal, and only when it has supplied a constraint the solver can act on.
Re-running with nothing changed would return the same plan indefinitely. The
critic reads the breaches the solver reports rather than recomputing its own
ratios, so there is one account of what a plan did.

| Path | Contents |
| --- | --- |
| `agents/` | extraction, orchestration, policy critic, explainer, scenario agent, live sensing adapters |
| `graph_engine/` | network construction, routing solver, digital twin, reserve optimiser, economic model, resilience analytics |
| `api/` | FastAPI application, REST (representational state transfer) endpoints and a `/ws/live` push channel |
| `frontend/` | single page interface, map view and network graph view of the same state |
| `data/` | network definition, parameters, replay timeline |
| `scripts/` | generators that produce the data files |

## What the solver determines, and what it does not

The solver works in flow per edge and per grade. It fixes the volume leaving
each source, the volume crossing each corridor, the volume reaching each
refinery, and the reserve draw. Those four are the quantities the interface
reports and the ones a procurement team acts on.

It does not fix which individual source feeds which individual refinery. Once
crude pools in a shared corridor the barrels are fungible, so a plan that sends
2 million barrels a day through Hormuz and 226,000 to Bathinda has many equally
optimal ways of labelling which cargo went where. The decomposition picks one
stable attribution (sorted sources, minimum-id tie-breaking) and the per-route
tables say so. Nothing in the system reports a change based on that pairing,
because a relabelling is not a procurement decision.

## Data

`data/edges.json` is generated rather than hand written. Run
`python scripts/build_network.py` to rebuild it, or
`python scripts/build_network.py --check` to confirm it has not drifted.

Sea lane distances come from `searoute`, which returns a navigable route rather
than a straight line, so a cargo travelling from Hormuz to Paradip rounds Sri
Lanka instead of crossing the subcontinent. Freight is modelled as a fixed per
voyage component plus a distance component, fitted to the three sourced rate
anchors in `parameters.json` and reproducing each within about 3%. Transit time
is distance at 13 knots laden, with an additional day for the Suez convoy.

Crude differentials are quoted against Brent for August 2026 loading. Seven are
verified against published assessments or official selling prices, one follows
from a benchmark identity, and nine have no public assessment available at that
granularity. Each node records which category applies.

Every entry in `data/parameters.json` carries a source and a status tag. Where a
figure is a snapshot of a volatile quantity it is labelled as such and dated.
Saudi Arabia's official selling price to Asia moved from plus 19.50 in May to
minus 1.50 in August, so any single crude differential should be treated as a
point in time rather than a constant.

## Scope and limitations

The nine estimated crude differentials are the least reliable inputs in the
model. A procurement ranking that turns on any of them should be confirmed
against a live price feed before being used. Pipeline tariffs and pipeline
velocity carry the same caveat, since no single published per barrel tariff
covers the Indian crude trunk lines.

Crude differentials respond to scarcity uniformly across every source still able
to deliver. During the 2026 crisis they did not move uniformly, as Gulf grades
firmed considerably more than Russian ones, so relative supplier ranking under a
severe disruption is outside what the model represents.

The three routing objectives frequently coincide. When the remaining network is
pinned to capacity, delivering every available barrel outranks cost, speed and
risk alike, and there is only one way to allocate what is left. The interface
detects this and collapses the three options into one rather than showing three
cards with identical numbers. A genuine trade-off appears at baseline and under
a production cut, and disappears under a corridor closure.

The strategic reserve can reach only three refineries. This reflects the
physical network rather than a simplification, since all three ISPRL (Indian
Strategic Petroleum Reserves Limited) caverns are on the south and west coast.
Under a Hormuz closure the shortfall therefore falls partly on refineries the
reserve cannot supply.

Reserve draw is bounded by what each cavern can sustain across the planning
horizon, so the solver cannot commit a rate that would empty a facility inside
it. The simulation executes that plan and depletes storage accordingly, which
means shipped barrels and stored barrels always agree.

Vessel availability, port congestion and berth scheduling are not represented.
Neither is anything downstream of the refinery gate, so product distribution,
retail supply and demand response all fall outside scope.

National figures are reported as lower bound exposure rather than extrapolated
forecasts, and market risk premia are kept separate from modelled physical
shortfall throughout.
