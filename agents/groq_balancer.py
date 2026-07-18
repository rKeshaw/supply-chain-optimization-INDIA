import os
import random
import logging
from typing import Optional, List, Tuple

# Import the Groq library
try:
    from groq import Groq, RateLimitError, GroqError
except ImportError as e:
    raise ImportError("The 'groq' package is not installed. Add it to requirements.txt and run pip install.") from e

logger = logging.getLogger(__name__)


class GroqBalancer:
    """
    Manages and schedules API calls across multiple Groq API keys to avoid rate limits.
    
    Reads a comma-separated list of keys from the GROQ_API_KEYS environment variable.
    If GROQ_API_KEYS is not set, falls back to the single GROQ_API_KEY environment variable.
    
    Calls are scheduled randomly across the active keys. If a key hits a rate limit
    (HTTP 429 / RateLimitError), the balancer automatically switches to another key
    and retries the call.
    """

    def __init__(self):
        self._keys: List[str] = self._load_keys()

    def _load_keys(self) -> List[str]:
        """Load keys from environment variables using the centralized settings module."""
        # Use centralized settings to load keys and avoid .env parsing bugs
        try:
            from api.settings import get_groq_api_keys
            keys = get_groq_api_keys()
            if keys:
                logger.info(f"GroqBalancer: Loaded {len(keys)} API keys via settings.")
                return keys
        except ImportError:
            pass

        logger.warning(
            "GroqBalancer: No API keys found! Set GROQ_API_KEYS (comma-separated list) "
            "or GROQ_API_KEY in your environment."
        )
        return []

    def get_keys(self) -> List[str]:
        """Return the list of loaded API keys."""
        # Reload keys dynamically to capture any runtime changes in env vars
        self._keys = self._load_keys()
        return self._keys

    def execute_completion(
        self,
        messages: List[dict],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        response_format: Optional[dict] = None,
        max_retries_per_key: int = 1,
    ) -> str:
        """
        Execute a chat completion call with automatic key rotation on rate limit failures.

        Args:
            messages: OpenAI-style message list.
            model: Groq model name (default Llama-3.3-70b).
            temperature: Sampling temperature.
            response_format: Optional format configuration (e.g. {"type": "json_object"}).
            max_retries_per_key: How many times to retry on a single key before rotating.

        Returns:
            The completion content string.

        Raises:
            RuntimeError: If all API keys are exhausted or rate-limited.
            ValueError: If no API keys are loaded.
        """
        keys = self.get_keys()
        if not keys:
            raise ValueError(
                "GroqBalancer: No API keys available. Ensure GROQ_API_KEYS or "
                "GROQ_API_KEY is defined in your environment."
            )

        # Work on a copy of keys so we can prune rate-limited ones during this call lifecycle
        available_keys = list(keys)
        random.shuffle(available_keys)  # shuffle to randomize schedule

        last_exception = None

        while available_keys:
            current_key = available_keys.pop()
            logger.debug(f"GroqBalancer: Scheduling call using key ending in ...{current_key[-6:]}")
            
            client = Groq(api_key=current_key)
            
            for attempt in range(max_retries_per_key + 1):
                try:
                    kwargs = {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                    }
                    if response_format:
                        kwargs["response_format"] = response_format

                    completion = client.chat.completions.create(**kwargs)
                    content = completion.choices[0].message.content
                    logger.debug(f"GroqBalancer: Call succeeded with key ending in ...{current_key[-6:]}")
                    return content

                except RateLimitError as e:
                    logger.warning(
                        f"GroqBalancer: Key ending in ...{current_key[-6:]} rate limited. "
                        f"Attempt {attempt + 1}/{max_retries_per_key + 1}. Error: {e}"
                    )
                    last_exception = e
                    # Move to next key immediately on rate limit
                    break

                except GroqError as e:
                    logger.error(
                        f"GroqBalancer: Groq error with key ending in ...{current_key[-6:]}: {e}. "
                        "Rotating to next key..."
                    )
                    last_exception = e
                    break

                except Exception as e:
                    logger.error(
                        f"GroqBalancer: Unexpected error with key ending in ...{current_key[-6:]}: {e}."
                    )
                    last_exception = e
                    break

        raise RuntimeError(
            f"GroqBalancer: All {len(keys)} keys exhausted or rate-limited. "
            f"Last exception: {last_exception}"
        )


# Global singleton instance for project-wide use
balancer = GroqBalancer()
