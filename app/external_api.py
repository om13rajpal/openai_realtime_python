"""External API client for user settings and weather data."""

import httpx
from typing import Optional, Any
from app.config import get_settings
from app.system_prompt import MARITIME_SAFETY_PROMPT


# External API base URL
EXTERNAL_API_BASE = "http://52.5.27.89:3000"


async def fetch_user_settings(auth_token: str) -> Optional[dict]:
    """Fetch user settings from external API.
    
    Returns user settings including:
    - language preference
    - vessel information
    - location data
    - other preferences
    """
    if not auth_token:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{EXTERNAL_API_BASE}/users/me/settings",
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle different response structures
                if isinstance(data, dict):
                    if data.get("success") is False:
                        return None
                    return data.get("data", data)
                return data
            return None
    except Exception as e:
        print(f"Error fetching user settings: {e}")
        return None


async def fetch_weather_data(
    auth_token: str,
    coordinates: list[list[float]]
) -> Optional[dict]:
    """Fetch weather data for given coordinates.
    
    Args:
        auth_token: User's auth token
        coordinates: List of [lat, lon] pairs
        
    Returns:
        Weather conditions for the coordinates
    """
    if not auth_token or not coordinates:
        return None
    
    try:
        # Format coordinates as query parameter
        coord_str = str(coordinates).replace(" ", "")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{EXTERNAL_API_BASE}/weather",
                params={"coordinates": coord_str},
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    if data.get("success") is False:
                        return None
                    return data.get("data", data)
                return data
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def format_user_context(
    user_settings: Optional[dict],
    weather_data: Optional[dict],
    user_location: Optional[str]
) -> str:
    """Format user context for AI instructions.
    
    Builds a context string that includes:
    - User's current location
    - User's vessels
    - Current weather conditions
    - Language preference
    """
    context_parts = []
    
    # User location
    if user_location:
        context_parts.append(f"[User's current location: {user_location}]")
    
    # User settings
    if user_settings:
        # Vessels
        vessels = user_settings.get("vessels", [])
        if vessels:
            vessel_info = []
            for v in vessels:
                if isinstance(v, dict):
                    make = v.get("make", "Unknown")
                    model = v.get("model", "Unknown")
                    year = v.get("year", "Unknown")
                    vessel_info.append({"make": make, "model": model, "year": year})
            if vessel_info:
                context_parts.append(f"[Vessel Info: {vessel_info}]")
        
        # Language
        language = user_settings.get("language")
        if language:
            context_parts.append(f"[User's preferred language: {language}]")
    
    # Weather data
    if weather_data:
        context_parts.append(f"[Current weather conditions: {weather_data}]")
    
    return "\n".join(context_parts) if context_parts else ""


async def build_enriched_instructions(
    auth_token: Optional[str],
    user_location: Optional[str],
    coordinates: Optional[list[list[float]]]
) -> str:
    """Build enriched instructions with user context.
    
    Uses the backend system prompt and enriches it with
    user context fetched from external APIs.
    """
    # Fetch external data if auth token available
    user_settings = None
    weather_data = None
    
    if auth_token:
        user_settings = await fetch_user_settings(auth_token)
        
        if coordinates:
            weather_data = await fetch_weather_data(auth_token, coordinates)
    
    # Format context
    context = format_user_context(user_settings, weather_data, user_location)
    
    # Combine context with system prompt (backend owns the prompt)
    if context:
        return f"{context}\n\n{MARITIME_SAFETY_PROMPT}"
    
    return MARITIME_SAFETY_PROMPT
