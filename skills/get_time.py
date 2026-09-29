"""
Reads the actual system clock. No LLM guessing, no web search — just the
real time, instantly. This exists because Gemma was leaking placeholder
templates like "[insert current time here]" when time questions got
routed through web_search, which only returns links, not a live reading.
"""

from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones

# Common city/place names Master might say instead of a formal IANA zone.
# Not exhaustive — falls through to a direct IANA lookup and then a
# best-effort city-name match against the full zone list before giving up.
_CITY_ALIASES = {
    "new york": "America/New_York", "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "dubai": "Asia/Dubai",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "kolkata": "Asia/Kolkata",
    "dhaka": "Asia/Dhaka",
    "sydney": "Australia/Sydney",
    "moscow": "Europe/Moscow",
    "toronto": "America/Toronto",
    "sao paulo": "America/Sao_Paulo",
}


def _resolve_timezone(tz_input: str):
    """Try a direct IANA match, then a known city alias, then a loose
    match against the last segment of every real IANA zone. Returns
    the IANA zone string, or None if nothing matched."""
    tz_input = tz_input.strip()

    if tz_input in available_timezones():
        return tz_input

    key = tz_input.lower()
    if key in _CITY_ALIASES:
        return _CITY_ALIASES[key]

    for zone in available_timezones():
        if zone.split("/")[-1].replace("_", " ").lower() == key:
            return zone

    return None


def get_current_time(timezone: str | None = None) -> str:
    """
    Return the current date/time as a plain string.

    timezone: None -> Master's local system time.
              An IANA zone ('America/New_York') or a common city name
              ('Tokyo', 'New York') otherwise.
    """
    try:
        if not timezone:
            now = datetime.now()
            return now.strftime("%A, %B %d, %Y — %I:%M %p (local time)")

        resolved = _resolve_timezone(timezone)
        if resolved is None:
            return (
                f"I don't recognize the timezone/place '{timezone}'. "
                "Try an IANA timezone like 'America/New_York' or a major city name like 'Tokyo'."
            )

        now = datetime.now(ZoneInfo(resolved))
        return now.strftime(f"%A, %B %d, %Y — %I:%M %p ({resolved})")

    except Exception as exc:
        return f"Couldn't get the time: {exc}"
