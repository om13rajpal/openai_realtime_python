import os
import pytest
from unittest.mock import patch

# Set test environment variables before any imports
@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for all tests."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test-key-123",
        "UPSTASH_REDIS_URL": "rediss://default:testpass@test.upstash.io:6379"
    }):
        yield
