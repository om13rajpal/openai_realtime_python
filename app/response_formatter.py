"""
Response Formatter for AiSeaSafe API

This module provides functions to format tool results into the structured
JSON format required by the Flutter frontend.

Response Types:
- weather: Weather information with detailed report
- assistance: Local maritime assistance contacts
- route: Trip planning with route analysis
- normal: Regular chat responses
"""

import json
import logging
from typing import Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Response type enumeration."""
    WEATHER = "weather"
    ASSISTANCE = "assistance"
    ROUTE = "route"
    NORMAL = "normal"


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


def calculate_risk_level(weather_data: dict) -> str:
    """
    Calculate risk level based on weather conditions.

    Args:
        weather_data: Weather data dictionary

    Returns:
        Risk level string (low, moderate, high, extreme)
    """
    wind_speed = weather_data.get("wind_speed", 0)
    wave_height = weather_data.get("wave_height", 0)
    visibility = weather_data.get("visibility", 10)
    wind_gusts = weather_data.get("wind_gusts", 0)

    # Convert to common units if needed (assuming mph for wind, ft for waves)
    if isinstance(wind_speed, str):
        wind_speed = float(wind_speed.replace("mph", "").replace("kt", "").strip())
    if isinstance(wave_height, str):
        wave_height = float(wave_height.replace("ft", "").replace("m", "").strip())

    # Risk calculation based on recreational boating standards
    if wind_speed > 35 or wave_height > 6 or visibility < 1:
        return RiskLevel.EXTREME.value
    elif wind_speed > 25 or wave_height > 4 or visibility < 3:
        return RiskLevel.HIGH.value
    elif wind_speed > 15 or wave_height > 2 or visibility < 5:
        return RiskLevel.MODERATE.value
    else:
        return RiskLevel.LOW.value


def format_weather_response(
    message: str,
    weather_data: dict,
    location: str = ""
) -> dict:
    """
    Format weather data into the structured response format.

    Args:
        message: AI-generated message about the weather
        weather_data: Raw weather data from API
        location: Location name

    Returns:
        Structured weather response dictionary
    """
    # Extract weather values with fallbacks
    weather = weather_data.get("weather", {})
    if isinstance(weather, list) and len(weather) > 0:
        weather = weather[0]

    # Get values from nested structure or flat structure
    def get_value(key: str, default: Any = 0) -> Any:
        if isinstance(weather_data, dict):
            # Try nested 'weather' key first
            if isinstance(weather, dict) and key in weather:
                return weather.get(key, default)
            # Try flat structure
            return weather_data.get(key, default)
        return default

    # Calculate or get risk level
    risk_level = get_value("risk_level", None)
    if not risk_level:
        risk_level = calculate_risk_level(weather_data)

    # Format temperature
    temp = get_value("temperature", 0)
    temp_unit = weather_data.get("units", {}).get("temperature_unit", "fahrenheit")
    temp_str = f"{temp}°F" if "fahr" in temp_unit.lower() else f"{temp}°C"

    # Format other values with units
    wave_height = get_value("wave_height", 0)
    wind_speed = get_value("wind_speed", 0)
    wind_gusts = get_value("wind_gusts", 0)
    visibility = get_value("visibility", 10)

    wind_unit = weather_data.get("units", {}).get("wind_speed_unit", "mph")
    dist_unit = weather_data.get("units", {}).get("distance_unit", "mi")

    # Build safety message based on risk level
    safety_messages = {
        "low": "Conditions are favorable for recreational boating.",
        "moderate": "Caution—moderate seas; ensure vessel is seaworthy and crew is experienced.",
        "high": "Warning—rough conditions; only experienced mariners should proceed.",
        "extreme": "Danger—do not sail; seek shelter immediately."
    }

    report_message = safety_messages.get(risk_level, safety_messages["moderate"])

    return {
        "response": {
            "message": message,
            "report": {
                "message": report_message,
                "weather": get_value("weather_condition", get_value("description", "Unknown")),
                "temperature": temp_str,
                "wave_height": f"{wave_height} ft",
                "wave_direction": f"{get_value('wave_direction', 0)}°",
                "wind_speed": f"{wind_speed} {wind_unit}",
                "wind_direction": f"{get_value('wind_direction', 0)}°",
                "wind_gusts": f"{wind_gusts} {wind_unit}",
                "humidity": f"{get_value('humidity', 0)}%",
                "pressure_surface_level": f"{get_value('pressure_surface_level', get_value('pressure', 29.92))} inHg",
                "rain_intensity": f"{get_value('rain_intensity', get_value('precipitation', 0))} in/h",
                "cloud_cover": f"{get_value('cloud_cover', 0)}%",
                "visibility": f"{visibility} {dist_unit}",
                "risk_level": risk_level
            },
            "type": ResponseType.WEATHER.value
        }
    }


def format_assistance_response(
    message: str,
    assistance_data: dict
) -> dict:
    """
    Format local assistance data into the structured response format.

    Args:
        message: AI-generated message about assistance
        assistance_data: Raw assistance data from LLM

    Returns:
        Structured assistance response dictionary
    """
    # Extract services from the assistance data
    services = assistance_data.get("services", [])
    emergency_contacts = assistance_data.get("emergency_contacts", {})

    # Format contacts list
    local_assistance = []

    # Add services
    for service in services:
        local_assistance.append({
            "name": service.get("name", "Unknown Service"),
            "type": service.get("type", ""),
            "phone": service.get("contact", service.get("phone", "")),
            "email": service.get("email", ""),
            "address": service.get("address", ""),
            "notes": service.get("notes", service.get("description", ""))
        })

    # Add emergency contacts
    if emergency_contacts:
        if emergency_contacts.get("coast_guard"):
            local_assistance.append({
                "name": "Coast Guard",
                "type": "emergency",
                "phone": emergency_contacts.get("coast_guard", ""),
                "email": "",
                "address": "",
                "notes": "Emergency maritime rescue services"
            })
        if emergency_contacts.get("harbor_master"):
            local_assistance.append({
                "name": "Harbor Master",
                "type": "port_authority",
                "phone": emergency_contacts.get("harbor_master", ""),
                "email": "",
                "address": "",
                "notes": "Port and harbor operations"
            })

    return {
        "response": {
            "message": message,
            "type": ResponseType.ASSISTANCE.value,
            "local_assistance": local_assistance
        }
    }


def format_route_response(
    message: str,
    route_data: dict,
    start_trip: bool = False
) -> dict:
    """
    Format route planning data into the structured response format.

    Args:
        message: AI-generated message about the route
        route_data: Raw route data from searoute and weather analysis
        start_trip: Whether user confirmed to start the trip

    Returns:
        Structured route response dictionary
    """
    origin = route_data.get("origin", {})
    destination = route_data.get("destination", {})
    route_info = route_data.get("route", {})
    weather_along = route_data.get("weather_along_route", [])
    safety = route_data.get("safety_assessment", {})
    vessel = route_data.get("vessel", {})

    # Parse route coordinates
    route_coords = route_info.get("coordinates", [])
    route_path = [[coord[0], coord[1]] for coord in route_coords[:10]] if route_coords else []  # Limit points

    # Determine trip status based on safety assessment
    risk_level = safety.get("risk_level", "Moderate").lower()
    vessel_compatible = True
    issues = []

    if risk_level == "extreme" or risk_level == "high":
        status = "UNSAFE"
        vessel_compatible = False
    elif risk_level == "moderate":
        status = "CAUTION"
    else:
        status = "SAFE"

    # Build issues list from safety notes
    safety_notes = safety.get("notes", [])
    for i, note in enumerate(safety_notes):
        if "rough" in note.lower() or "strong" in note.lower() or "unsafe" in note.lower():
            # Get weather for this waypoint
            wp_weather = weather_along[i] if i < len(weather_along) else {}
            if isinstance(wp_weather, dict):
                issues.append({
                    "waypoint": i,
                    "problem": note,
                    "marine_condition": f"waves ~{wp_weather.get('wave_height', 0)} m, winds ~{wp_weather.get('wind_speed', 0)} kt",
                    "vessel_concern": f"Conditions may exceed vessel limits" if vessel else "",
                    "distance": 0.0,
                    "weather": format_waypoint_weather(wp_weather)
                })

    # Format source and destination weather
    source_weather = weather_along[0] if weather_along else {}
    dest_weather = weather_along[-1] if weather_along else {}

    # Build recommendation
    if status == "UNSAFE":
        recommendation = "NO-GO: Delay departure or choose an alternative route until sea state, wind, and visibility are comfortably within your vessel's limits."
    elif status == "CAUTION":
        recommendation = "PROCEED WITH CAUTION: Conditions are manageable but require experience. Monitor weather updates closely."
    else:
        recommendation = "GO: Conditions are favorable for the voyage. Standard safety precautions apply."

    # Estimate time based on distance and average speed
    distance_nm = route_info.get("distance_nm", route_info.get("distance_nautical_miles", 0))
    avg_speed = 8  # knots for recreational vessel
    est_hours = distance_nm / avg_speed if distance_nm > 0 else 0

    if est_hours < 1:
        est_time = f"Approx {int(est_hours * 60)} minutes"
    elif est_hours < 24:
        est_time = f"Approx {int(est_hours)}-{int(est_hours) + 1} hours"
    else:
        days = int(est_hours / 24)
        est_time = f"Approx {days}-{days + 1} days"

    return {
        "response": {
            "message": message,
            "trip_plan": {
                "route": {
                    "source": {
                        "name": origin.get("name", ""),
                        "coordinates": origin.get("coordinates", [])
                    },
                    "destination": {
                        "name": destination.get("name", ""),
                        "coordinates": destination.get("coordinates", [])
                    },
                    "route_path": route_path,
                    "distance_nautical_miles": round(distance_nm, 2),
                    "distance_kilometers": round(route_info.get("distance_km", 0), 2),
                    "total_waypoints": len(route_path)
                },
                "trip_analysis": {
                    "status": status,
                    "vessel_compatible": vessel_compatible,
                    "summary": safety.get("notes", [""])[0] if safety.get("notes") else "",
                    "issues": issues,
                    "source": {
                        "weather": format_waypoint_weather(source_weather)
                    },
                    "destination": {
                        "weather": format_waypoint_weather(dest_weather)
                    },
                    "recommendation": recommendation
                }
            },
            "estimated_time": est_time,
            "start_trip": start_trip,
            "type": ResponseType.ROUTE.value
        }
    }


def format_waypoint_weather(weather_data: dict) -> dict:
    """Format weather data for a single waypoint."""
    if not weather_data or not isinstance(weather_data, dict):
        return {
            "temperature": 0,
            "wave_height": 0,
            "wave_direction": "0",
            "wind_speed": 0,
            "wind_direction": "0",
            "wind_gusts": 0,
            "humidity": 0,
            "pressure_surface_level": 29.92,
            "rain_intensity": 0,
            "cloud_cover": 0,
            "visibility": 10,
            "risk_level": "low"
        }

    return {
        "temperature": weather_data.get("temperature", 0),
        "wave_height": weather_data.get("wave_height", 0),
        "wave_direction": str(weather_data.get("wave_direction", 0)),
        "wind_speed": weather_data.get("wind_speed", 0),
        "wind_direction": str(weather_data.get("wind_direction", 0)),
        "wind_gusts": weather_data.get("wind_gusts", 0),
        "humidity": weather_data.get("humidity", 0),
        "pressure_surface_level": weather_data.get("pressure_surface_level", weather_data.get("pressure", 29.92)),
        "rain_intensity": weather_data.get("rain_intensity", weather_data.get("precipitation", 0)),
        "cloud_cover": weather_data.get("cloud_cover", 0),
        "visibility": weather_data.get("visibility", 10),
        "risk_level": calculate_risk_level(weather_data)
    }


def format_normal_response(message: str) -> dict:
    """
    Format a normal chat response.

    Args:
        message: AI-generated message

    Returns:
        Structured normal response dictionary
    """
    return {
        "response": {
            "message": message,
            "type": ResponseType.NORMAL.value
        }
    }


def format_response_from_tool_calls(
    message: str,
    tool_calls: list[dict],
    route_data: Optional[dict] = None
) -> dict:
    """
    Determine response type and format based on tool calls made.

    Args:
        message: AI-generated message
        tool_calls: List of tool calls with results
        route_data: Optional route data if route was planned

    Returns:
        Structured response dictionary
    """
    if not tool_calls:
        return format_normal_response(message)

    # Check what tools were called
    for tool_call in tool_calls:
        tool_name = tool_call.get("tool_name", "")
        result = tool_call.get("result", {})

        if not result.get("success", False):
            continue

        if tool_name == "get_marine_weather":
            weather_data = result.get("weather", result)
            weather_data["units"] = result.get("units", {})
            return format_weather_response(message, weather_data, result.get("location", ""))

        elif tool_name == "get_local_assistance":
            return format_assistance_response(message, result)

        elif tool_name == "plan_and_analyze_marine_route":
            return format_route_response(message, result)

    # If route_data provided separately
    if route_data and route_data.get("success"):
        return format_route_response(message, route_data)

    # Default to normal response
    return format_normal_response(message)


def create_streaming_chunk(
    field: str,
    value: Any,
    response_type: str = "normal"
) -> str:
    """
    Create a streaming JSON chunk for partial updates.

    Args:
        field: Field name being updated
        value: New value for the field
        response_type: Type of response

    Returns:
        JSON string for streaming
    """
    chunk = {
        "type": "delta",
        "response_type": response_type,
        "field": field,
        "value": value
    }
    return json.dumps(chunk)
