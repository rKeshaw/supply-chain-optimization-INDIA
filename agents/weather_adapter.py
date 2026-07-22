import logging
import urllib.request
import urllib.parse
import json
from datetime import datetime, timezone

from agents.schema import Event

logger = logging.getLogger(__name__)

# Coordinates for major Indian Single Point Mooring (SPM) buoys
SPM_LOCATIONS = {
    "ref_jamnagar_in": {"lat": 22.47, "lon": 69.93, "name": "Gulf of Kutch SPM"},
    "ref_mangalore_in": {"lat": 12.91, "lon": 74.80, "name": "Mangalore SPM"},
    "ref_paradip_in": {"lat": 20.26, "lon": 86.68, "name": "Paradip SPM"}
}

# Thresholds for safe operations (in meters)
WAVE_HEIGHT_WARNING = 2.5
WAVE_HEIGHT_CRITICAL = 3.5

# Severity caps. A berth closed by weather reopens when the swell drops, so even
# a critical reading is a temporary berthing suspension rather than the loss of a
# refinery. Combined with event_type_capacity_impact's 0.5 weather coefficient,
# a critical reading removes at most ~30% of that terminal's throughput.
SEVERITY_CRITICAL = 0.6
SEVERITY_WARNING = 0.3
# A terminal is only marked down after this many consecutive breaching polls.
REQUIRED_CONSECUTIVE_READINGS = 2
_CONSECUTIVE_BREACHES: dict[str, int] = {}

# Network timeout for the live marine API. Keeps the digital twin from
# hanging when the demo host is offline or the API is slow.
WEATHER_FETCH_TIMEOUT_S = 4.0


def fetch_marine_weather(lat: float, lon: float) -> dict:
    """Fetch current wave height from Open-Meteo Marine API."""
    url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height&timezone=auto"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'EnergyResilience/1.0'})
        with urllib.request.urlopen(req, timeout=WEATHER_FETCH_TIMEOUT_S) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        logger.error(f"Failed to fetch marine weather: {e}")
        return {}

def check_weather_disruptions() -> list[Event]:
    """
    Check all critical SPM locations for severe weather.
    Returns a list of synthetic Events if operational thresholds are exceeded.
    """
    events = []
    timestamp = datetime.now(timezone.utc)
    
    for node_id, loc in SPM_LOCATIONS.items():
        data = fetch_marine_weather(loc["lat"], loc["lon"])
        current_data = data.get("current", {})
        wave_height = current_data.get("wave_height")
        
        if wave_height is None:
            continue
            
        logger.info(f"Weather for {loc['name']} ({node_id}): wave_height = {wave_height}m")
        
        if wave_height >= WAVE_HEIGHT_CRITICAL:
            severity = SEVERITY_CRITICAL
            justification = f"CRITICAL: Wave height at {loc['name']} is {wave_height}m, exceeding safe SPM operational limits of {WAVE_HEIGHT_CRITICAL}m."
        elif wave_height >= WAVE_HEIGHT_WARNING:
            severity = SEVERITY_WARNING
            justification = f"WARNING: Wave height at {loc['name']} is {wave_height}m. Partial delays expected."
        else:
            _CONSECUTIVE_BREACHES.pop(node_id, None)
            continue

        # Heavy seas suspend berthing; they do not shut a refinery. Requiring two
        # consecutive readings stops a single spike closing a terminal, which
        # matters most during the June-September monsoon when both thresholds are
        # routinely exceeded off the west coast.
        _CONSECUTIVE_BREACHES[node_id] = _CONSECUTIVE_BREACHES.get(node_id, 0) + 1
        if _CONSECUTIVE_BREACHES[node_id] < REQUIRED_CONSECUTIVE_READINGS:
            logger.info(
                "Weather at %s breached threshold (%sm) on reading %d of %d; holding.",
                loc["name"], wave_height, _CONSECUTIVE_BREACHES[node_id], REQUIRED_CONSECUTIVE_READINGS,
            )
            continue
            
        event = Event(
            id=f"weather_{node_id}_{int(timestamp.timestamp())}",
            source="Open-Meteo Marine API",
            timestamp=timestamp,
            entity=loc["name"],
            location=loc["name"],
            event_type="weather_disruption",
            severity=severity,
            confidence=0.9,  # Live telemetry, but a forecast point is not a berth closure notice
            affected_graph_element=node_id,
            justification=justification
        )
        events.append(event)
        
    return events
