from livekit.agents.llm import function_tool


@function_tool
async def check_schedule(date: str) -> str:
    """Check the user's schedule for a given date.

    Args:
        date: The date to check (e.g. "2024-03-15" or "tomorrow").
    """
    # In a real implementation, this would query a calendar API
    return f"You have 2 meetings on {date}: standup at 9am and design review at 2pm."


@function_tool
async def create_event(title: str, date: str, time: str) -> str:
    """Create a new calendar event.

    Args:
        title: The title of the event.
        date: The date for the event.
        time: The time for the event.
    """
    # In a real implementation, this would create an event via calendar API
    return f"Created event '{title}' on {date} at {time}."
