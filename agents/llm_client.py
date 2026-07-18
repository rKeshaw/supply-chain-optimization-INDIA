"""
LLM client: thin wrapper around the Groq API load balancer for all agent calls.

Single point of configuration for model selection, temperature, and API key scheduling.
All agents import this rather than calling the API directly.
"""

import os
import logging
from typing import Optional

from agents.groq_balancer import balancer

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    system_instruction: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_retries: int = 2,
    retry_delay_s: float = 2.0,  # kept for signature compatibility
    expect_json: bool = True,
) -> Optional[str]:
    """
    Call the Groq API with the given prompt and system instruction.
    Uses the GroqBalancer singleton to schedule calls across multiple keys
    and handle rate limit (429) retries automatically.

    Args:
        prompt: User prompt text.
        system_instruction: System-level instruction for the model.
        model: Groq model name (default: llama-3.3-70b-versatile).
        temperature: Sampling temperature.
        max_retries: Number of retry attempts per key.
        retry_delay_s: Unused, kept for backward compatibility.
        expect_json: If True, request JSON response format from Groq.

    Returns:
        Model response text string, or None on failure.
    """
    # 1. Structure message payload
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    # 2. Structure format requirements
    response_format = {"type": "json_object"} if expect_json else None

    try:
        content = balancer.execute_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            max_retries_per_key=max_retries
        )
        return content

    except Exception as e:
        logger.error(f"call_llm: GroqBalancer execution failed: {e}")
        return None
