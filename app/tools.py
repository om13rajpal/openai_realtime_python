"""
Maritime Chat Tools for LLM Function Calling

This module provides tools for the maritime-chat endpoint:
- get_marine_weather: Fetch marine weather data from NestJS API
- get_local_assistance: Get local maritime assistance using LLM
- plan_and_analyze_marine_route: Plan marine routes using searoute package
"""

import json
import logging
from typing import Optional, Any
import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from app.config import get_settings
from app.geocoding import geocode_location as nominatim_geocode

logger = logging.getLogger(__name__)

# External API base URL
EXTERNAL_API_BASE = "http://52.5.27.89:3000"


# ============================================================================
# Tool Parameter Models (for OpenAI function calling)
# ============================================================================

class GetMarineWeatherParams(BaseModel):
    """Parameters for get_marine_weather tool."""
    location: str = Field(description="Location name for weather query (e.g., 'Miami, FL', 'Dubai Marina')")
    coordinates: Optional[list[list[float]]] = Field(
        default=None,
        description="Optional coordinates as [[lat, lon]] pairs. If provided, uses coordinates directly."
    )


class GetLocalAssistanceParams(BaseModel):
    """Parameters for get_local_assistance tool."""
    location: str = Field(description="Location to search for local maritime assistance")
    assistance_type: Optional[str] = Field(
        default=None,
        description="Type of assistance needed (e.g., 'marina', 'fuel', 'repair', 'emergency', 'supplies')"
    )


class PlanMarineRouteParams(BaseModel):
    """Parameters for plan_and_analyze_marine_route tool."""
    origin: str = Field(description="Starting location name")
    origin_coordinates: list[float] = Field(description="Origin coordinates as [longitude, latitude]")
    destination: str = Field(description="Destination location name")
    destination_coordinates: list[float] = Field(description="Destination coordinates as [longitude, latitude]")
    vessel_make: Optional[str] = Field(default=None, description="Vessel manufacturer")
    vessel_model: Optional[str] = Field(default=None, description="Vessel model")
    vessel_year: Optional[str] = Field(default=None, description="Vessel year")


# ============================================================================
# Tool Definitions (OpenAI function schemas)
# ============================================================================

MARITIME_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_marine_weather",
            "description": "Retrieves marine weather information for a specified location. Use this when users ask about weather conditions, sea state, wind, waves, or sailing conditions at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name for weather query (e.g., 'Miami, FL', 'Dubai Marina')"
                    },
                    "coordinates": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "Optional coordinates as [[lat, lon]] pairs"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_assistance",
            "description": "Get local maritime assistance information including marinas, fuel stations, repair services, and emergency contacts. Use when users ask for local services or assistance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to search for local maritime assistance"
                    },
                    "assistance_type": {
                        "type": "string",
                        "enum": ["marina", "fuel", "repair", "emergency", "supplies", "general"],
                        "description": "Type of assistance needed"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_and_analyze_marine_route",
            "description": "Plan a marine route between two locations with weather analysis and safety assessment. Use when users want to plan a trip, voyage, or route between locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Starting location name"
                    },
                    "origin_coordinates": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Origin coordinates as [longitude, latitude]"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination location name"
                    },
                    "destination_coordinates": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Destination coordinates as [longitude, latitude]"
                    },
                    "vessel_make": {
                        "type": "string",
                        "description": "Vessel manufacturer"
                    },
                    "vessel_model": {
                        "type": "string",
                        "description": "Vessel model"
                    },
                    "vessel_year": {
                        "type": "string",
                        "description": "Vessel year"
                    }
                },
                "required": ["origin", "origin_coordinates", "destination", "destination_coordinates"]
            }
        }
    }
]


# ============================================================================
# Helper Functions
# ============================================================================

async def fetch_user_weather_preferences(auth_token: str) -> Optional[dict]:
    """Fetch user weather unit preferences from external API."""
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
                if isinstance(data, dict):
                    if data.get("success") is False:
                        return None
                    settings = data.get("data", data)
                    return {
                        "temperature_unit": settings.get("temperature_unit", "celsius"),
                        "wind_speed_unit": settings.get("wind_speed_unit", "knots"),
                        "distance_unit": settings.get("distance_unit", "nautical_miles")
                    }
            return None
    except Exception as e:
        logger.error("Error fetching user preferences: %s", e)
        return None


async def geocode_location(location: str) -> Optional[list[float]]:
    """
    Convert location name to coordinates using OpenStreetMap Nominatim API.
    Returns [longitude, latitude] for searoute compatibility.

    This function uses the Nominatim geocoding service to support any location
    worldwide, not just hardcoded maritime locations.
    """
    if not location or not location.strip():
        return None

    # Use the Nominatim-based geocoding service
    coords = await nominatim_geocode(location)

    if coords:
        logger.info("Successfully geocoded '%s' to [%f, %f]", location, coords[0], coords[1])
        return coords
    else:
        logger.warning("Could not geocode location: %s", location)
        return None


# ============================================================================
# Tool Implementations
# ============================================================================

async def get_marine_weather(
    location: str,
    coordinates: Optional[list[list[float]]] = None,
    auth_token: Optional[str] = None
) -> dict:
    """
    Fetch marine weather data from the NestJS API.

    Args:
        location: Location name for the weather query
        coordinates: Optional coordinates [[lat, lon]] to use directly
        auth_token: User's auth token for preferences

    Returns:
        Weather data dictionary with marine conditions
    """
    try:
        # Get user preferences for units
        preferences = await fetch_user_weather_preferences(auth_token) if auth_token else None

        # If no coordinates provided, try to geocode the location
        if not coordinates:
            geocoded = await geocode_location(location)
            if geocoded:
                # Convert [lon, lat] to [[lat, lon]] for API
                coordinates = [[geocoded[1], geocoded[0]]]
            else:
                return {
                    "success": False,
                    "error": f"Could not find coordinates for location: {location}",
                    "location": location
                }

        # Format coordinates for API
        coord_str = str(coordinates).replace(" ", "")

        # Fetch weather from NestJS API
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            logger.info("Weather API request with auth token (first 20 chars): %s...", auth_token[:20] if len(auth_token) > 20 else auth_token)
        else:
            logger.warning("Weather API request WITHOUT auth token")

        async with httpx.AsyncClient(timeout=15.0) as client:
            logger.info("Calling weather API: %s with coordinates: %s", f"{EXTERNAL_API_BASE}/weather", coord_str)
            response = await client.get(
                f"{EXTERNAL_API_BASE}/weather",
                params={"coordinates": coord_str},
                headers=headers
            )

            logger.info("Weather API response status: %s", response.status_code)

            if response.status_code == 200:
                data = response.json()
                logger.info("Weather API response data keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
                weather_data = data.get("data", data) if isinstance(data, dict) else data

                return {
                    "success": True,
                    "location": location,
                    "coordinates": coordinates,
                    "weather": weather_data,
                    "units": preferences or {
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "knots",
                        "distance_unit": "nautical_miles"
                    }
                }
            else:
                logger.error("Weather API error response: %s", response.text[:500] if response.text else "No response body")
                return {
                    "success": False,
                    "error": f"Weather API returned status {response.status_code}",
                    "location": location,
                    "details": response.text[:200] if response.text else None
                }

    except Exception as e:
        logger.error("Error fetching marine weather: %s", e)
        return {
            "success": False,
            "error": str(e),
            "location": location
        }


async def get_local_assistance(
    location: str,
    assistance_type: Optional[str] = None,
    auth_token: Optional[str] = None
) -> dict:
    """
    Get local maritime assistance information using LLM.

    Args:
        location: Location to search for assistance
        assistance_type: Type of assistance needed
        auth_token: User's auth token

    Returns:
        Structured assistance information
    """
    try:
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        assistance_prompt = f"""You are a maritime assistance expert. Provide helpful information about local maritime services and assistance for recreational boaters.

Location: {location}
Assistance Type: {assistance_type or 'general'}

Provide a JSON response with the following structure:
{{
    "location": "{location}",
    "assistance_type": "{assistance_type or 'general'}",
    "services": [
        {{
            "name": "Service name",
            "type": "marina|fuel|repair|emergency|supplies",
            "description": "Brief description",
            "contact": "Contact information if known",
            "notes": "Any relevant notes for boaters"
        }}
    ],
    "emergency_contacts": {{
        "coast_guard": "Local coast guard contact",
        "harbor_master": "Harbor master contact if applicable",
        "emergency": "General emergency number"
    }},
    "tips": ["Helpful tip 1", "Helpful tip 2"],
    "safety_notes": "Any important safety information for the area"
}}

Provide realistic and helpful information based on the location. If specific details are not known, provide general guidance appropriate for the region."""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a maritime assistance expert. Always respond with valid JSON."},
                {"role": "user", "content": assistance_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        result = json.loads(response.choices[0].message.content)
        result["success"] = True
        return result

    except Exception as e:
        logger.error("Error getting local assistance: %s", e)
        return {
            "success": False,
            "error": str(e),
            "location": location,
            "assistance_type": assistance_type
        }


async def plan_and_analyze_marine_route(
    origin: str,
    origin_coordinates: list[float],
    destination: str,
    destination_coordinates: list[float],
    vessel_make: Optional[str] = None,
    vessel_model: Optional[str] = None,
    vessel_year: Optional[str] = None,
    auth_token: Optional[str] = None
) -> dict:
    """
    Plan a marine route using searoute package and analyze weather conditions.

    Args:
        origin: Starting location name
        origin_coordinates: [longitude, latitude] for origin
        destination: Destination location name
        destination_coordinates: [longitude, latitude] for destination
        vessel_make: Vessel manufacturer
        vessel_model: Vessel model
        vessel_year: Vessel year
        auth_token: User's auth token

    Returns:
        Route data with weather analysis and safety assessment
    """
    try:
        # Import searoute - will be installed via requirements.txt
        try:
            import searoute as sr
        except ImportError:
            logger.error("searoute package not installed")
            return {
                "success": False,
                "error": "Route planning service unavailable. Please install searoute package.",
                "origin": origin,
                "destination": destination
            }

        # Generate route using searoute
        # searoute expects [longitude, latitude] format
        route = sr.searoute(origin_coordinates, destination_coordinates)

        # Extract route information
        route_coords = route.get("geometry", {}).get("coordinates", [])
        route_properties = route.get("properties", {})
        distance_km = route_properties.get("length", 0)
        distance_nm = distance_km * 0.539957  # Convert km to nautical miles

        # Get weather data along the route (sample points)
        weather_points = []
        if route_coords and len(route_coords) > 0:
            # Sample points along route for weather (start, middle, end)
            sample_indices = [0, len(route_coords) // 2, -1]
            weather_coords = []
            for idx in sample_indices:
                if idx < len(route_coords):
                    coord = route_coords[idx]
                    # Convert [lon, lat] to [[lat, lon]] for weather API
                    weather_coords.append([coord[1], coord[0]])

            # Fetch weather for route points
            if weather_coords:
                try:
                    coord_str = str(weather_coords).replace(" ", "")
                    headers = {}
                    if auth_token:
                        headers["Authorization"] = f"Bearer {auth_token}"

                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.get(
                            f"{EXTERNAL_API_BASE}/weather",
                            params={"coordinates": coord_str},
                            headers=headers
                        )

                        if response.status_code == 200:
                            data = response.json()
                            weather_points = data.get("data", data) if isinstance(data, dict) else data
                except Exception as e:
                    logger.warning("Could not fetch route weather: %s", e)

        # Build vessel info
        vessel_info = None
        if vessel_make or vessel_model or vessel_year:
            vessel_info = {
                "make": vessel_make,
                "model": vessel_model,
                "year": vessel_year
            }

        # Calculate estimated time (assuming average speed of 8 knots for recreational vessel)
        avg_speed_knots = 8
        estimated_hours = distance_nm / avg_speed_knots if distance_nm > 0 else 0

        # Analyze safety based on weather (simplified)
        safety_assessment = "Low"  # Default
        safety_notes = []

        if weather_points:
            # Check weather conditions for safety
            for point in weather_points if isinstance(weather_points, list) else [weather_points]:
                if isinstance(point, dict):
                    wind_speed = point.get("wind_speed", 0)
                    wave_height = point.get("wave_height", 0)

                    if wind_speed > 25 or wave_height > 2:
                        safety_assessment = "High"
                        safety_notes.append("Strong winds or rough seas expected")
                    elif wind_speed > 15 or wave_height > 1:
                        if safety_assessment == "Low":
                            safety_assessment = "Moderate"
                        safety_notes.append("Moderate conditions - experience recommended")

        return {
            "success": True,
            "type": "route",
            "origin": {
                "name": origin,
                "coordinates": origin_coordinates
            },
            "destination": {
                "name": destination,
                "coordinates": destination_coordinates
            },
            "route": {
                "coordinates": route_coords,
                "distance_km": round(distance_km, 2),
                "distance_nm": round(distance_nm, 2),
                "estimated_time_hours": round(estimated_hours, 1),
                "properties": route_properties
            },
            "vessel": vessel_info,
            "weather_along_route": weather_points,
            "safety_assessment": {
                "risk_level": safety_assessment,
                "notes": safety_notes if safety_notes else ["Conditions appear favorable for the voyage"]
            }
        }

    except Exception as e:
        logger.error("Error planning marine route: %s", e)
        return {
            "success": False,
            "error": str(e),
            "origin": origin,
            "destination": destination
        }


# ============================================================================
# Tool Executor
# ============================================================================

async def execute_tool(
    tool_name: str,
    tool_args: dict,
    auth_token: Optional[str] = None
) -> dict:
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        auth_token: User's auth token

    Returns:
        Tool execution result
    """
    tool_map = {
        "get_marine_weather": get_marine_weather,
        "get_local_assistance": get_local_assistance,
        "plan_and_analyze_marine_route": plan_and_analyze_marine_route
    }

    if tool_name not in tool_map:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }

    tool_func = tool_map[tool_name]

    # Add auth_token to args if the function supports it
    if auth_token:
        tool_args["auth_token"] = auth_token

    try:
        return await tool_func(**tool_args)
    except Exception as e:
        logger.error("Error executing tool %s: %s", tool_name, e)
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name
        }
