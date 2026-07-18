import logging
from typing import Optional
from pydantic import BaseModel, Field
from agents.groq_balancer import GroqBalancer

logger = logging.getLogger(__name__)

class NLOpsRequest(BaseModel):
    scenario_id: Optional[str] = Field(
        None, 
        description="The ID of a named scenario if the user's request matches one (e.g. 'hormuz_partial', 'hormuz_full', 'red_sea_suspension', 'opec_cut'). Leave null if the user specifies a custom configuration."
    )
    custom_scenario: Optional[dict[str, float]] = Field(
        None,
        description="A dictionary mapping canonical graph node/edge IDs to an openness multiplier (0.0 to 1.0) for custom scenarios. E.g., {'ref_jamnagar_in': 0.0} or {'src_saudi': 0.5}. Leave null if using a named scenario."
    )
    horizon_days: int = Field(
        30,
        description="The number of days to run the digital twin simulation. Default is 30. Extract this if the user specifies a timeframe."
    )

def parse_nl_command(query: str, available_scenarios: dict, available_nodes: list[dict]) -> Optional[NLOpsRequest]:
    """
    Translates a natural language query into structured twin simulation parameters.
    """
    prompt = f"""You are an elite Natural Language Operations (NL-Ops) AI for the Indian Energy Supply Chain Resilience system.
Your job is to translate a user's natural language command into structured parameters for the digital twin simulator.

Available named scenarios (use scenario_id if it's a perfect match):
{list(available_scenarios.keys())}

Available canonical graph elements for custom_scenario (id: name):
{[f"{n['id']}: {n['name']}" for n in available_nodes]}

User Query: "{query}"

Analyze the query and determine:
1. Does it match a named scenario? If so, set scenario_id and leave custom_scenario null.
2. Is it a custom disruption? If so, map the mentioned locations to the correct canonical IDs and set their openness multiplier (e.g., 0.0 for closed, 0.5 for 50% capacity). Set custom_scenario.
3. How many days should it simulate? (Default 30).
"""
    
    balancer = GroqBalancer()
    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Parse this command: {query}. Respond in JSON format exactly matching the schema. Only return the JSON."}
        ]
        
        # We need the JSON schema for the prompt
        schema = NLOpsRequest.model_json_schema()
        messages[0]["content"] += f"\n\nJSON Schema to follow:\n{schema}"
        
        result_str = balancer.execute_completion(
            messages, 
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return NLOpsRequest.model_validate_json(result_str)
    except Exception as e:
        logger.error(f"NL-Ops parsing failed: {e}")
        return None
