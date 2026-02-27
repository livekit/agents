from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ...llm import Tool, Toolset, function_tool
from ...log import logger
from ...voice.events import RunContext


class DateTimeHelperTool(Toolset):
    def __init__(
        self,
        *,
        tz: str | None = None,
        on_tool_called: Callable[[Toolset.ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[Toolset.ToolCompletedEvent], Awaitable[None]] | None = None,
    ):
        """
        This tool allows the agent to access the current date and time, with an optional offset and timezone. It can be used to answer questions about the current date and time, compute relative datetimes (e.g., "What day will it be in 3 days?"), and convert between timezones. By default, timezone is UTC, but if the relevant timezone was set in advance or can be inferred from the conversation, it will be used instead.

        Args:
            tz: The Timezone string to be used to compute current and relative datetimes.
            on_tool_called: Callback to call when the tool is called.
            on_tool_completed: Callback to call when the tool is completed.
        """
        super().__init__(id="date_time_helper")
        self._tz = tz
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

        self._compute_datetime_tool = function_tool(
            self._compute_datetime,
            name="date_time_helper",
            description="""Use this tool any time the current date or time is relevant to the conversation, such as when the user asks for the current time, wants to schedule a meeting, or references a specific date. Always use this tool to get the current date and time instead of relying on the LLM's internal clock, as the LLM's clock may be inaccurate or not in sync with the relevant timezone. The tool returns a dict with three fields: "local" (the computed date/time in ISO format using the chosen timezone), "utc" (the same moment in UTC, ISO format), and "timezone" (the timezone name used for the computation). If no timezone is known, it defaults to UTC.

            Args:
                offset: An optional integer representing a number of seconds to add (positive) or subtract (negative) from the current date and time. This can be used to compute relative datetimes, such as "What will the date and time be in 3 days?" (offset would be 259200 seconds) or "What was the date and time 2 hours ago?" (offset would be -7200 seconds).
                tz_override: An optional string representing a timezone to use for this specific computation, if different from the default timezone set for the tool. This can be used when the user references a specific timezone in their question (e.g., "What time is it in New York?") or when the relevant timezone can be inferred from the conversation. The tool will return the date and time in this specified timezone instead of the default.
            """,
        )

    async def _compute_datetime(self, ctx: RunContext, offset: int | None = None, tz_override: str | None = None) -> Any | None:
        logger.debug("date_time_helper tool called")
        if self._on_tool_called:
            await self._on_tool_called(
                Toolset.ToolCalledEvent(
                    ctx=ctx,
                    arguments={
                        "offset": offset,
                        "tz_override": tz_override,
                    },
                )
            )

        tz_name = tz_override or self._tz or "UTC"
        try:
            tzinfo = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            output = f"Invalid timezone: {tz_name}"
            logger.warning(output)
            completed_ev = Toolset.ToolCompletedEvent(ctx=ctx, output=output)
            if self._on_tool_completed:
                await self._on_tool_completed(completed_ev)
            return completed_ev.output

        current = datetime.now(tzinfo)
        if offset is not None:
            current = current + timedelta(seconds=offset)

        output = {
            "local": self._format_datetime(current),
            "utc": self._format_datetime(current.astimezone(timezone.utc)),
            "timezone": tz_name,
        }
        completed_ev = Toolset.ToolCompletedEvent(ctx=ctx, output=output)
        if self._on_tool_completed:
            await self._on_tool_completed(completed_ev)

        return completed_ev.output

    @staticmethod
    def _format_datetime(value: datetime) -> str:
        if value.tzinfo is not None and value.utcoffset() == timedelta(0):
            return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return value.isoformat()

    @property
    def tools(self) -> list[Tool]:
        return [self._compute_datetime_tool]

