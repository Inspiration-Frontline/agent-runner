from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_runner.tools.registry import BaseTool


class CurrentTimeTool(BaseTool):
    @property
    def tool_key(self) -> str:
        return "builtin.current_time"

    @property
    def tool_name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "Return the current local time, UTC offset, and Unix timestamp for an IANA timezone."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": ["string", "null"],
                    "description": "IANA timezone such as Asia/Shanghai. Null uses the server timezone.",
                },
            },
            "required": ["timezone"],
            "additionalProperties": False,
        }

    @property
    def strict(self) -> bool:
        return True

    async def execute(self, timezone: str | None) -> dict:
        try:
            if timezone:
                now = datetime.now(ZoneInfo(timezone))
                timezone_name = timezone
            else:
                now = datetime.now().astimezone()
                timezone_name = str(now.tzinfo)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown IANA timezone: {timezone}") from error

        offset = now.utcoffset()
        offset_seconds = int(offset.total_seconds()) if offset is not None else 0
        sign = "+" if offset_seconds >= 0 else "-"
        absolute_offset = abs(offset_seconds)
        offset_text = f"{sign}{absolute_offset // 3600:02d}:{absolute_offset % 3600 // 60:02d}"
        return {
            "timezone": timezone_name,
            "local_datetime": now.isoformat(timespec="seconds"),
            "utc_offset": offset_text,
            "unix_timestamp": int(now.timestamp()),
        }
