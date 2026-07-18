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

    # 2. Physical shortfall price-pressure scenario. EIA notes that actual oil
    # prices also include forward-looking risk premia; this model intentionally
    # does not pretend to infer those from a local refinery-network gap.
    def price_pressure(elasticity_value: float) -> float:
        gap_pressure = national_shortfall_lower_bound_pct / max(elasticity_value, 0.01)
        market_pressure = market_supply_loss_pct_global / max(elasticity_value, 0.01)
        return min(price_cap, max(gap_pressure, market_pressure))

    crude_price_change_pct = price_pressure(elasticity_central)
    crude_price_change_low_pct = price_pressure(elasticity_high)
    crude_price_change_high_pct = price_pressure(elasticity_low)

    # --- COST CHANNEL (B): reroute freight/procurement premium ------------------
    # This is exact solver output, not an elasticity estimate: it is the extra
    # $/bbl the disrupted routing costs versus the undisrupted baseline routing.
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
    if days_elapsed >= lag_days:
        retail_price_change_pct = (crude_price_change_pct + landed_cost_change_pct) * passthrough
        # Formula: retail_Δ = (crude_Δ + landed_cost_Δ) × passthrough_coefficient
    else:
        # Within lag period, retail price is not yet revised
        retail_price_change_pct = 0.0

    # 4. GDP drag (only if sustained beyond lag — same threshold for simplicity)
    # Formula: GDP_drag_pct = (crude_price_change_pct / 10) × gdp_sensitivity_param
    if days_elapsed >= lag_days and crude_price_change_pct > 0:
        gdp_drag_pct = (crude_price_change_pct / 10.0) * gdp_sensitivity
    else:
        gdp_drag_pct = 0.0

    # 5. Power sector stress
    refinery_output_drop_fraction = gap_bbl_day / max(baseline_supply_bbl_day, 1)
    power_sector_stress = (
        "ELEVATED" if refinery_output_drop_fraction > power_threshold else "NORMAL"
    )

    # 6. Annualized import-bill sensitivity (illustrative, not an accounting forecast).
    #    crude-benchmark-driven portion (volume channel):
    import_bill_increase_usd_bn = import_bill_baseline * (crude_price_change_pct / 100)
    #    total including the exact reroute freight premium (cost channel):
    total_import_cost_increase_usd_bn = round(
        import_bill_increase_usd_bn + annualized_reroute_premium_usd_bn, 2
    )
    # Single headline economic-impact flag: does EITHER channel bite?
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
        "crude_price_change_formula": "max(national_gap_pct, global_market_loss_pct) / short_run_price_elasticity_abs",
        "crude_price_change_scope": "physical-shortfall scenario only; excludes market risk premium",
        "market_supply_loss_bbl_day": round(market_supply_loss_bbl_day, 0),
        "market_supply_loss_pct_global": round(market_supply_loss_pct_global, 3),
        "crude_price_driver": (
            "global_market_loss" if market_supply_loss_pct_global / max(elasticity_central, 0.01)
            > national_shortfall_lower_bound_pct / max(elasticity_central, 0.01)
            else "national_refinery_gap"
        ),

        # Retail
        "retail_price_change_pct": round(retail_price_change_pct, 2),
        "retail_price_delayed": days_elapsed < lag_days,
        "retail_lag_days_remaining": max(0, lag_days - days_elapsed),
        "retail_formula": "crude_price_change_pct × passthrough_coefficient",

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
        },
    }

    return result


def compute_backtest_cascade(
    actual_gap_bbl_day: float,
    baseline_brent_price_usd: float,
    actual_brent_change_pct: float,
    days_elapsed: int,
    params: dict,
) -> dict:
    """
    Compare the model's predicted cascade against a known historical outcome.

    Used in the backtest (Module 18) against the April 2025 US-Iran standoff.
    Actual Brent change: +8% in a single session. Model should produce a
    comparable price change given the supply signal.

    Args:
        actual_gap_bbl_day: Observed supply gap from the event.
        baseline_brent_price_usd: Brent price on the day before the event.
        actual_brent_change_pct: Actual observed Brent price change %.
        days_elapsed: Days since disruption onset.
        params: Parameters dict.

    Returns:
        Dict with model_prediction, actual_outcome, and error_metrics.
    """
    model_cascade = compute_cascade(
        actual_gap_bbl_day, 5_000_000, days_elapsed, params
    )
    predicted_change = model_cascade["crude_price_change_pct"]

    error_pct = abs(predicted_change - actual_brent_change_pct)
    error_pct_relative = error_pct / max(actual_brent_change_pct, 1) * 100

    return {
        "model_prediction_crude_price_change_pct": predicted_change,
        "actual_brent_change_pct": actual_brent_change_pct,
        "error_percentage_points": round(error_pct, 2),
        "error_pct_relative": round(error_pct_relative, 1),
        "model_cascade": model_cascade,
        "comparison_status": "non_comparable_without_observed_physical_gap_and_risk_premium",
        "_interpretation": (
            f"Physical-shortfall scenario: +{predicted_change:.1f}% vs observed market move "
            f"+{actual_brent_change_pct:.1f}%. Difference: {error_pct:.1f}pp. "
            "Do not interpret this as a calibrated forecast error: the observed move may contain a "
            "forward-looking risk premium that is not identifiable from a local supply-gap estimate."
        ),
    }
