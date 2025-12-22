import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_get_redis_returns_client():
    """get_redis should return a Redis client instance."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "sk-test",
        "UPSTASH_REDIS_URL": "rediss://default:pass@test.upstash.io:6379"
    }):
        from app.redis_client import get_redis

        # Mock the redis connection
        with patch("app.redis_client.redis_client") as mock_client:
            mock_client.ping = AsyncMock(return_value=True)
            client = get_redis()
            assert client is not None


def test_redis_module_exists():
    """Redis client module should be importable."""
    from app import redis_client
    assert hasattr(redis_client, "get_redis")
    assert hasattr(redis_client, "init_redis")
    assert hasattr(redis_client, "close_redis")
