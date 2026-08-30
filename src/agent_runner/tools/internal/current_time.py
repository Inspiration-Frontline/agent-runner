from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agents import function_tool


@function_tool(failure_error_function=None)
async def get_current_time(timezone: str | None) -> dict[str, str | int]:
    """Return the current local time, UTC offset, and Unix timestamp for an IANA timezone.

    Args:
        timezone: IANA timezone such as Asia/Shanghai. Null uses the server timezone.

    Returns:
        Current local time, UTC offset, and Unix timestamp for the requested IANA timezone.
    """
    try:
        if timezone:
            now: datetime = datetime.now(ZoneInfo(timezone))
            timezone_name: str = timezone
        else:
            now = datetime.now().astimezone()
            timezone_name = str(now.tzinfo)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown IANA timezone: {timezone}") from error

    offset: timedelta | None = now.utcoffset()
    offset_seconds: int = int(offset.total_seconds()) if offset is not None else 0
    sign: str = "+" if offset_seconds >= 0 else "-"
    absolute_offset: int = abs(offset_seconds)
    offset_text: str = f"{sign}{absolute_offset // 3600:02d}:{absolute_offset % 3600 // 60:02d}"
    return {
        "timezone": timezone_name,
        "local_datetime": now.isoformat(timespec="seconds"),
        "utc_offset": offset_text,
        "unix_timestamp": int(now.timestamp()),
    }
