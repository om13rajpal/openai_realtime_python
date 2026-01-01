"""
Geocoding Service for AiSeaSafe

This module provides geocoding functionality using OpenStreetMap Nominatim API
to convert location names to coordinates (latitude, longitude).

Uses Nominatim API: https://nominatim.openstreetmap.org/search
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# Nominatim API configuration
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = "AiSeaSafe/1.0 (Maritime Safety Assistant)"

# Cache for geocoding results to reduce API calls
_geocoding_cache: dict[str, Optional[list[float]]] = {}


async def geocode_location(location: str) -> Optional[list[float]]:
    """
    Convert a location name to coordinates using OpenStreetMap Nominatim API.

    Args:
        location: Location name (e.g., "Malibu, California", "Dubai Marina")

    Returns:
        Coordinates as [longitude, latitude] for searoute compatibility,
        or None if geocoding fails.
    """
    if not location or not location.strip():
        return None

    location_key = location.lower().strip()

    # Check cache first
    if location_key in _geocoding_cache:
        logger.debug("Geocoding cache hit for: %s", location)
        return _geocoding_cache[location_key]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{NOMINATIM_BASE_URL}/search",
                params={
                    "q": location,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 0,
                },
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                }
            )

            if response.status_code == 200:
                results = response.json()

                if results and len(results) > 0:
                    result = results[0]
                    lat = float(result.get("lat", 0))
                    lon = float(result.get("lon", 0))

                    # Return [longitude, latitude] for searoute compatibility
                    coords = [lon, lat]

                    # Cache the result
                    _geocoding_cache[location_key] = coords

                    logger.info("Geocoded '%s' to coordinates: [%f, %f]", location, lon, lat)
                    return coords
                else:
                    logger.warning("No geocoding results for: %s", location)
                    _geocoding_cache[location_key] = None
                    return None
            else:
                logger.error("Nominatim API error: %s - %s", response.status_code, response.text)
                return None

    except httpx.TimeoutException:
        logger.error("Geocoding timeout for: %s", location)
        return None
    except Exception as e:
        logger.error("Geocoding error for '%s': %s", location, e)
        return None


async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Convert coordinates to a location name using OpenStreetMap Nominatim API.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Location name string or None if reverse geocoding fails.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{NOMINATIM_BASE_URL}/reverse",
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "json",
                    "addressdetails": 1,
                },
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                }
            )

            if response.status_code == 200:
                result = response.json()

                if result and "display_name" in result:
                    display_name = result["display_name"]

                    # Try to get a shorter, more readable name
                    address = result.get("address", {})
                    city = address.get("city") or address.get("town") or address.get("village")
                    state = address.get("state")
                    country = address.get("country")

                    if city and state:
                        return f"{city}, {state}"
                    elif city and country:
                        return f"{city}, {country}"
                    else:
                        # Return first part of display name (usually most specific)
                        parts = display_name.split(",")
                        if len(parts) >= 2:
                            return f"{parts[0].strip()}, {parts[1].strip()}"
                        return display_name

                return None
            else:
                logger.error("Nominatim reverse API error: %s", response.status_code)
                return None

    except Exception as e:
        logger.error("Reverse geocoding error: %s", e)
        return None


def clear_geocoding_cache():
    """Clear the geocoding cache."""
    global _geocoding_cache
    _geocoding_cache = {}
    logger.info("Geocoding cache cleared")


def get_cache_stats() -> dict:
    """Get geocoding cache statistics."""
    return {
        "cached_locations": len(_geocoding_cache),
        "cache_entries": list(_geocoding_cache.keys())[:10]  # First 10 entries
    }
