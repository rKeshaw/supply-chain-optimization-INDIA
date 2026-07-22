"""
Economic cascade model: translates crude supply shortfalls into
price, retail fuel, GDP, and power-sector impact metrics.

All parameters come from data/parameters.json and are explicitly named —
never hardcoded. The explainer agent is strictly forbidden from inventing numbers;
it can only cite values produced by this module.
"""

import math
import logging

logger = logging.getLogger(__name__)


def global_supply_loss_bbl_day(G, params: dict) -> float:
    """Barrels a scenario removes from the WORLD market, in bbl/day.

    Two ways supply leaves the market:

    1. Source-side capacity cuts — an OPEC+ quota, a field outage, sanctions.
    2. Stranding at an egress chokepoint. A chokepoint with no sea alternative
       traps the crude behind it: there is no route out of the Persian Gulf
       except Hormuz, so closing it removes barrels outright, net of the
       pipeline capacity that reaches water beyond the strait. Chokepoints
       flagged ``has_sea_alternative`` (Bab-el-Mandeb, Suez, Malacca) strand
       nothing — traffic diverts around the Cape at extra cost and time.

    This is deliberately NOT India's supply gap. A transit disruption that
    strands India's cargoes but leaves the barrels available to other buyers
    moves India's landed cost, not the global benchmark.
    """
    bypass = float(params.get("hormuz_bypass_pipeline_capacity_bbl_day", {}).get("value", 0.0))

    lost = 0.0
    for _, data in G.nodes(data=True):
        if data.get("type") == "source":
            lost += float(data.get("capacity_bbl_day") or 0) * (1.0 - float(data.get("openness", 1.0)))

    for _, data in G.nodes(data=True):
        if data.get("type") != "chokepoint" or data.get("has_sea_alternative", True):
            continue
        transit = float(data.get("global_transit_bbl_day") or 0)
        stranded = transit * (1.0 - float(data.get("openness", 1.0)))
        lost += max(0.0, stranded - bypass)

    return lost


def compute_cascade(
    gap_bbl_day: float,
    baseline_supply_bbl_day: float,
    days_elapsed: int,
    params: dict,
    reroute_cost_premium_usd_per_bbl: float = 0.0,
    delivered_volume_bbl_day: float = 0.0,
    market_supply_loss_bbl_day: float = 0.0,
) -> dict:
    """
    Compute the full economic cascade from a supply gap.

    Two distinct, separately-labelled impact channels:

    A. VOLUME channel (physical shortfall the reroute could NOT cover):
       1. Supply shortfall % → crude price change % (via elasticity)
       2. Crude price change → retail pump price change (via passthrough, with lag)
       3. Sustained (>lag_days) crude price change → GDP drag (via sensitivity param)
       4. Refinery output drop → power sector stress flag

    B. COST channel (physical volume WAS covered, but rerouting cost more):
       The solver's freight/procurement premium ($/bbl above baseline landed cost)
       is turned into a landed-cost % change, an annualised extra import bill, and
       a lagged retail pass-through. This is what makes cost-shock disruptions —
       an OPEC+ cut or a Red Sea reroute that raises cost without cutting volume —
       register a real economic impact instead of showing as "zero".

    All parameter values are read from params dict (traceable to parameters.json).
    Every output field is labeled with its formula for auditability.

    Args:
        gap_bbl_day: Supply gap = (demand - actual_supply) in bbl/day.
                     Must be >= 0. If 0, the volume channel is flat.
        baseline_supply_bbl_day: Undisrupted baseline supply to compute % shortfall.
        days_elapsed: How many days into the disruption (for lag gating and GDP).
        params: Parameters dict from parameters.json.
        reroute_cost_premium_usd_per_bbl: Extra $/bbl the current (disrupted) routing
                     costs versus the undisrupted baseline routing. Drives channel B.
                     Defaults to 0.0 → channel B is inert (backward-compatible).
        delivered_volume_bbl_day: Volume actually delivered under the reroute, used to
                     annualise the cost premium. Defaults to baseline_supply if unset.

    Returns:
        Dict with all cascade metrics. All values are floats or None.
        Labeled fields include the formula used.
    """
    if gap_bbl_day < 0:
        gap_bbl_day = 0.0

    # The graph represents five refineries, not all of India. Preserve that
    # distinction: refinery shortfall is operationally exact within the model;
    # national shortfall is a lower-bound exposure, not an extrapolated forecast.
    national_consumption = params.get("national_consumption_bbl_day", {}).get("value", 5_000_000)
    modelled_network_share = min(1.0, baseline_supply_bbl_day / max(national_consumption, 1))

    # Unpack parameters (each with fallback default so missing keys don't crash)
    elasticity = params.get("short_run_price_elasticity_abs", {})
    elasticity_central = abs(elasticity.get("value", 0.30))
    elasticity_low = abs(elasticity.get("low", 0.24))
    elasticity_high = abs(elasticity.get("high", 0.36))
    passthrough = params.get("passthrough_coefficient", {}).get("value", 0.72)
    lag_days = params.get("lag_days", {}).get("value", 14)
    gdp_sensitivity = params.get("gdp_sensitivity_param", {}).get("value", 0.2)
    power_threshold = params.get("power_sector_stress_threshold", {}).get("value", 0.15)
    price_cap = params.get("max_modelled_crude_price_change_pct", {}).get("value", 75.0)
    import_bill_baseline = params.get("baseline_annual_crude_import_bill_usd_bn", {}).get("value", 140.0)
    landed_cost_baseline = params.get("assumed_landed_crude_cost_usd_per_bbl", {}).get("value", 86.0)
    petrol_baseline_inr = params.get("baseline_petrol_price_delhi_inr_per_liter", {}).get("value", 102.12)
    diesel_baseline_inr = params.get("baseline_diesel_price_delhi_inr_per_liter", {}).get("value", 95.20)

    global_supply = params.get("global_oil_supply_bbl_day", {}).get("value", 102_000_000)

    # 1. Supply shortfall %
    modelled_shortfall_pct = gap_bbl_day / max(baseline_supply_bbl_day, 1) * 100
    national_shortfall_lower_bound_pct = gap_bbl_day / max(national_consumption, 1) * 100

    # Two physically-distinct price drivers, each with the correct denominator:
    #  A) India's uncovered refinery gap (transit disruption strands India's barrels):
    #     India must bid for replacement cargoes → national-relative tightness.
    #  B) Barrels removed from the GLOBAL market at source (OPEC+ cut): the benchmark
    #     moves even when India reroutes to fully cover its refineries → global-relative.
    market_supply_loss_bbl_day = max(0.0, float(market_supply_loss_bbl_day or 0.0))
    market_supply_loss_pct_global = market_supply_loss_bbl_day / max(global_supply, 1) * 100

    # The benchmark moves only on barrels removed from the world market. A
    # transit disruption strands India's cargoes without destroying them, so the
    # crude still reaches other buyers and the benchmark barely reacts. What
    # moves is the cost of landing a replacement cargo, which the landed-cost
    # channel below carries.
    def price_pressure(elasticity_value: float) -> float:
        return min(price_cap, market_supply_loss_pct_global / max(elasticity_value, 0.01))

    crude_price_change_pct = price_pressure(elasticity_central)
    crude_price_change_low_pct = price_pressure(elasticity_high)
    crude_price_change_high_pct = price_pressure(elasticity_low)

    # Cost channel: the extra dollars per barrel the disrupted routing costs
    # against the undisrupted baseline. Solver output rather than an elasticity
    # estimate.
    freight_premium = max(0.0, float(reroute_cost_premium_usd_per_bbl or 0.0))
    delivered_volume = float(delivered_volume_bbl_day or 0.0)
    if delivered_volume <= 0.0:
        # Fall back to the covered volume (baseline minus the uncovered gap).
        delivered_volume = max(0.0, baseline_supply_bbl_day - gap_bbl_day)

    landed_cost_change_pct = freight_premium / max(landed_cost_baseline, 1.0) * 100
    daily_reroute_cost_premium_usd = freight_premium * delivered_volume
    annualized_reroute_premium_usd_bn = daily_reroute_cost_premium_usd * 365 / 1e9

    # 3. Retail fuel price change (after lag). Retail revisions reflect BOTH the
    # crude benchmark move (volume channel) and the higher landed freight cost
    # (cost channel); both pass through on the same administered pricing cycle.
    # Formula: retail_Δ = (crude_Δ + landed_cost_Δ) × passthrough_coefficient
    projected_retail_price_change_pct = (crude_price_change_pct + landed_cost_change_pct) * passthrough
    if days_elapsed >= lag_days:
        retail_price_change_pct = projected_retail_price_change_pct
    else:
        # Within lag period, the ADMINISTERED price hasn't been revised yet —
        # retail_price_change_pct stays 0 to reflect that specific fact.
        retail_price_change_pct = 0.0

    # Expected pump prices, built on the unlagged projected change so the figure
    # answers what the price becomes once the disruption is priced in. The
    # lag-gated value sits at zero on day zero by construction, which would leave
    # this metric flat however severe the event. Both fuels share one pass-through
    # coefficient because it is not separately calibrated for petrol and diesel.
    expected_petrol_price_inr = petrol_baseline_inr * (1.0 + projected_retail_price_change_pct / 100.0)
    expected_diesel_price_inr = diesel_baseline_inr * (1.0 + projected_retail_price_change_pct / 100.0)

    # 4. GDP drag, applied only after the lag threshold is crossed.
    #
    # The sensitivity parameter is calibrated so that a 10 percentage-point rise
    # in crude prices reduces India's annual GDP by gdp_sensitivity_param pp
    # (0.2 pp per 10% crude increase). The /10 converts crude_price_change_pct
    # (a percentage, not a ratio) to the scale at which the parameter was
    # calibrated. Calibration basis: Mohan & Patra (2009), "Monetary Policy and
    # Inflation in India", RBI Occasional Papers Vol. 30 No. 3; cross-checked
    # against IEA World Energy Outlook sensitivity tables for oil-importing EMEs.
    if days_elapsed >= lag_days and crude_price_change_pct > 0:
        gdp_drag_pct = (crude_price_change_pct / 10.0) * gdp_sensitivity
    else:
        gdp_drag_pct = 0.0

    # 5. Power sector stress — flagged when the physical shortfall exceeds the
    # threshold fraction of baseline refinery throughput, indicating that
    # oil-fired and dual-fuel peakers may face feedstock constraints.
    refinery_output_drop_fraction = gap_bbl_day / max(baseline_supply_bbl_day, 1)
    power_sector_stress = (
        "ELEVATED" if refinery_output_drop_fraction > power_threshold else "NORMAL"
    )

    # 6. Annualized import-bill increase (illustrative order-of-magnitude, not
    # an accounting forecast): crude-benchmark-driven portion (volume channel)
    # plus the exact solver-computed reroute freight premium (cost channel).
    import_bill_increase_usd_bn = import_bill_baseline * (crude_price_change_pct / 100)
    total_import_cost_increase_usd_bn = round(
        import_bill_increase_usd_bn + annualized_reroute_premium_usd_bn, 2
    )
    has_economic_impact = (crude_price_change_pct > 0.0) or (freight_premium > 0.0)

    result = {
        # Supply metrics
        "gap_bbl_day": gap_bbl_day,
        "shortfall_pct": round(modelled_shortfall_pct, 2),
        "shortfall_scope": "modelled five-refinery network",
        "national_shortfall_lower_bound_pct": round(national_shortfall_lower_bound_pct, 2),
        "national_shortfall_scope": "lower bound; no unmodelled-refinery extrapolation",
        "modelled_network_share_of_national_consumption": round(modelled_network_share, 4),

        # Price cascade
        "crude_price_change_pct": round(crude_price_change_pct, 2),
        "crude_price_change_range_pct": {
            "low": round(crude_price_change_low_pct, 2),
            "high": round(crude_price_change_high_pct, 2),
        },
        "crude_price_change_formula": "global_market_loss_pct / short_run_price_elasticity_abs",
        "crude_price_change_scope": "physical-shortfall scenario only; excludes market risk premium",
        "market_supply_loss_bbl_day": round(market_supply_loss_bbl_day, 0),
        "market_supply_loss_pct_global": round(market_supply_loss_pct_global, 3),
        "crude_price_driver": "global_market_loss" if market_supply_loss_bbl_day > 0 else "none",
        "national_gap_price_treatment": (
            "not_converted_to_benchmark — a transit disruption strands India's cargoes without "
            "destroying the barrels, so it moves India's landed cost, not the world price. The gap "
            "is reported in full under shortfall_pct and drives the landed-cost channel."
        ),

        # Retail
        "retail_price_change_pct": round(retail_price_change_pct, 2),
        "retail_price_delayed": days_elapsed < lag_days,
        "retail_lag_days_remaining": max(0, lag_days - days_elapsed),
        "retail_formula": "crude_price_change_pct × passthrough_coefficient",

        # Absolute pump prices — built on projected_retail_price_change_pct
        # (unlag-gated), NOT the lag-gated retail_price_change_pct above. See
        # the projected_retail_price_change_pct comment for why.
        "projected_retail_price_change_pct": round(projected_retail_price_change_pct, 2),
        "petrol_price_inr_per_liter": round(expected_petrol_price_inr, 2),
        "diesel_price_inr_per_liter": round(expected_diesel_price_inr, 2),
        "petrol_baseline_inr_per_liter": petrol_baseline_inr,
        "diesel_baseline_inr_per_liter": diesel_baseline_inr,
        "pump_price_formula": "baseline_price_inr_per_liter × (1 + projected_retail_price_change_pct / 100)",
        "pump_price_scope": "Delhi reference city (PPAC/media standard); state VAT makes actual retail prices vary — not a national figure. Reflects the expected price once the disruption's severity is fully reflected, not the current-instant administered price (which stays at the pre-disruption level until PPAC's next revision cycle).",

        # GDP
        "gdp_drag_pct": round(gdp_drag_pct, 3),
        "gdp_formula": "(crude_price_change_pct / 10) × gdp_sensitivity_param",
        "gdp_applicable": days_elapsed >= lag_days,

        # Power sector
        "power_sector_stress": power_sector_stress,
        "refinery_output_drop_fraction": round(refinery_output_drop_fraction, 4),

        # Cost channel (reroute freight/procurement premium — exact solver output)
        "reroute_cost_premium_usd_per_bbl": round(freight_premium, 3),
        "landed_cost_change_pct": round(landed_cost_change_pct, 2),
        "landed_cost_change_formula": "reroute_cost_premium_usd_per_bbl / assumed_landed_crude_cost_usd_per_bbl",
        "daily_reroute_cost_premium_usd": round(daily_reroute_cost_premium_usd, 0),
        "annualized_reroute_premium_usd_bn": round(annualized_reroute_premium_usd_bn, 3),
        "has_economic_impact": has_economic_impact,

        # Finance
        "import_bill_increase_usd_bn": round(import_bill_increase_usd_bn, 2),
        "import_bill_formula": "baseline_annual_crude_import_bill_usd_bn × crude_price_change_pct",
        "total_import_cost_increase_usd_bn": total_import_cost_increase_usd_bn,
        "total_import_cost_formula": "import_bill_increase_usd_bn + annualized_reroute_premium_usd_bn",

        # Meta
        "days_elapsed": days_elapsed,
        "recovery_estimate_available": False,

        # Parameter echo (for explainer agent — it reads directly from here)
        "_params_used": {
            "short_run_price_elasticity_abs": elasticity_central,
            "short_run_price_elasticity_range_abs": [elasticity_low, elasticity_high],
            "passthrough_coefficient": passthrough,
            "lag_days": lag_days,
            "gdp_sensitivity_param": gdp_sensitivity,
            "power_sector_stress_threshold": power_threshold,
            "max_modelled_crude_price_change_pct": price_cap,
            "baseline_petrol_price_delhi_inr_per_liter": petrol_baseline_inr,
            "baseline_diesel_price_delhi_inr_per_liter": diesel_baseline_inr,
        },
    }

    return result


def compute_backtest_cascade(
    market_supply_loss_bbl_day: float,
    actual_gap_bbl_day: float,
    baseline_brent_price_usd: float,
    actual_brent_change_pct: float,
    days_elapsed: int,
    params: dict,
) -> dict:
    """
    Decompose an observed market move into the part this model explains
    physically and the part it deliberately does not.

    The model's benchmark channel responds ONLY to barrels removed from the
    world market (see compute_cascade). That is a modelling choice, not an
    omission: a transit disruption strands India's cargoes without destroying
    them. So the residual between the physical channel and the observed move is
    not model error — it is the forward-looking risk premium, which is exactly
    the quantity the rest of this system is careful to report separately.

    The benchmark channel keys off global supply loss and nothing else, so that
    figure is the input that has to be supplied here for the comparison to mean
    anything.

    Args:
        market_supply_loss_bbl_day: Barrels physically removed from the WORLD
            market by the event. Zero for a pure threat/escalation event where
            no production or egress was actually lost — which is itself the
            substantive finding for such an event.
        actual_gap_bbl_day: India's own uncovered refinery gap, reported for
            context; it drives shortfall_pct, never the benchmark.
        baseline_brent_price_usd: Brent on the day before the event.
        actual_brent_change_pct: Observed Brent move, %.
        days_elapsed: Days since onset.
        params: Parameters dict.

    Returns:
        Dict with the physical channel, the observed move, and the implied risk
        premium in both percentage points and $/bbl.
    """
    model_cascade = compute_cascade(
        actual_gap_bbl_day, 5_000_000, days_elapsed, params,
        market_supply_loss_bbl_day=market_supply_loss_bbl_day,
    )
    physical_channel_pct = model_cascade["crude_price_change_pct"]

    implied_premium_pct = actual_brent_change_pct - physical_channel_pct
    implied_premium_usd = baseline_brent_price_usd * implied_premium_pct / 100.0
    explained_share = (
        physical_channel_pct / actual_brent_change_pct * 100.0
        if actual_brent_change_pct else None
    )

    return {
        "market_supply_loss_bbl_day": market_supply_loss_bbl_day,
        "model_physical_channel_pct": physical_channel_pct,
        "observed_brent_change_pct": actual_brent_change_pct,
        "baseline_brent_price_usd": baseline_brent_price_usd,
        "implied_risk_premium_pct": round(implied_premium_pct, 2),
        "implied_risk_premium_usd_per_bbl": round(implied_premium_usd, 2),
        "share_explained_by_physical_loss_pct": (
            round(explained_share, 1) if explained_share is not None else None
        ),
        "model_cascade": model_cascade,
        "decomposition": "observed_move = physical_supply_channel + risk_premium",
        "_interpretation": (
            f"Observed Brent move +{actual_brent_change_pct:.1f}%. Barrels physically removed from "
            f"the world market: {market_supply_loss_bbl_day:,.0f} bbl/day, which this model prices at "
            f"+{physical_channel_pct:.1f}%. The residual +{implied_premium_pct:.1f}% "
            f"(${implied_premium_usd:.2f}/bbl on a ${baseline_brent_price_usd:.0f} base) is "
            "forward-looking risk premium — the market pricing the probability of a disruption that "
            "had not yet removed a barrel. This model deliberately does not forecast that component; "
            "the decomposition is the result, not an error term."
        ),
    }
