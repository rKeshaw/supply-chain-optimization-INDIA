import os
from pathlib import Path
from dotenv import load_dotenv

# Try to load from .env file if it exists (for local development)
# The .env file should be at the root of the energy-resilience project
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

def get_groq_api_keys() -> list[str]:
    """
    Safely extract the Groq API keys from the environment.
    Handles comma-separated lists and avoids breaking on syntax errors.
    """
    keys_str = os.environ.get("GROQ_API_KEYS", "").strip()
    
    # If keys_str has quotes around it, strip them
    if keys_str.startswith('"') and keys_str.endswith('"'):
        keys_str = keys_str[1:-1]
    if keys_str.startswith("'") and keys_str.endswith("'"):
        keys_str = keys_str[1:-1]
        
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys
            
    # Fallback
    single_key = os.environ.get("GROQ_API_KEY", "").strip()
    if single_key.startswith('"') and single_key.endswith('"'):
        single_key = single_key[1:-1]
        
    if single_key:
        return [single_key]
        
    return []

def get_provider_health() -> dict:
    """Return redacted provider health status."""
    keys = get_groq_api_keys()
    return {
        "provider": "Groq",
        "keys_loaded": len(keys),
        "status": "healthy" if keys else "no_keys"
    }
