import os
import pytest
from unittest.mock import patch, MagicMock
from agents.groq_balancer import GroqBalancer
from groq import RateLimitError, GroqError

def test_groq_balancer_key_loading():
    """Verify that balancer loads GROQ_API_KEYS and falls back to GROQ_API_KEY."""
    with patch.dict(os.environ, {"GROQ_API_KEYS": "key1, key2,  key3", "GROQ_API_KEY": "key_single"}):
        balancer = GroqBalancer()
        assert balancer.get_keys() == ["key1", "key2", "key3"]

    with patch.dict(os.environ, {"GROQ_API_KEYS": "", "GROQ_API_KEY": "key_single"}):
        balancer = GroqBalancer()
        assert balancer.get_keys() == ["key_single"]

@patch("agents.groq_balancer.Groq")
def test_groq_balancer_rotation(mock_groq_class):
    """Verify that balancer rotates keys when one of them hits a rate limit."""
    # Set up mock clients
    mock_client1 = MagicMock()
    # RateLimitError requires a response argument; we mock the exception
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_client1.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={}
    )
    
    mock_client2 = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Success response from second key"
    mock_client2.chat.completions.create.return_value = mock_completion
    
    # Configure mock class constructor to return client1 first, then client2
    mock_groq_class.side_effect = [mock_client1, mock_client2]
    
    with patch.dict(os.environ, {"GROQ_API_KEYS": "key1,key2"}):
        balancer = GroqBalancer()
        
        # When execute_completion runs, it should try key1 (fails with 429),
        # rotate to key2, and return success
        res = balancer.execute_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="llama-3.3-70b-versatile"
        )
        
        assert res == "Success response from second key"
        assert mock_groq_class.call_count == 2
