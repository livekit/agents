from livekit.agents.llm import function_tool


@function_tool
async def check_schedule(date: str) -> str:
    """Check calendar for a specific date.

    Args:
        date: The date to check (YYYY-MM-DD).
    """
    return f"No events on {date}"


@function_tool
async def book_meeting(date: str, time: str, title: str) -> str:
    """Book a new meeting.

    Args:
        date: Meeting date (YYYY-MM-DD).
        time: Meeting time (HH:MM).
        title: Meeting title.
    """
    return f"Booked '{title}' on {date} at {time}"
