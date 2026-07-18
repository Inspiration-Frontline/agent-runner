import httpx
from agents import function_tool

from agent_runner.config import get_settings


@function_tool(failure_error_function=None)
async def get_current_weather(location: str) -> dict[str, object]:
    """Get current weather for a city or place using Open-Meteo.

    Args:
        location: Place name such as Shanghai, China.
    """
    normalized = location.strip()
    if not normalized:
        raise ValueError("Location cannot be blank.")

    timeout = get_settings().tool_http_timeout_seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        geocoding = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": normalized, "count": 1, "language": "en", "format": "json"},
        )
        geocoding.raise_for_status()
        matches = geocoding.json().get("results") or []
        if not matches:
            raise ValueError(f"No weather location matched: {normalized}")
        place = matches[0]

        forecast = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        weather = forecast.json()
    return {
        "location": {
            "name": place.get("name"),
            "admin1": place.get("admin1"),
            "country": place.get("country"),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
        },
        "timezone": weather.get("timezone"),
        "timezone_abbreviation": weather.get("timezone_abbreviation"),
        "current": weather.get("current", {}),
        "current_units": weather.get("current_units", {}),
    }
