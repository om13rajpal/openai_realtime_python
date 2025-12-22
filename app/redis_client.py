from typing import Optional
import redis.asyncio as redis
from app.config import get_settings

# Global redis client instance
redis_client: Optional[redis.Redis] = None


async def init_redis() -> redis.Redis:
    """Initialize Redis connection pool."""
    global redis_client
    settings = get_settings()

    redis_client = redis.from_url(
        settings.upstash_redis_url,
        encoding="utf-8",
        decode_responses=True
    )

    # Test connection
    await redis_client.ping()
    return redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None


def get_redis() -> redis.Redis:
    """Get the Redis client instance."""
    if redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return redis_client
