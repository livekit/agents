from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("Elicitation demo")


class SeatPreference(BaseModel):
    seat: str = Field(description="Seat to reserve, e.g. 12A")
    extra_legroom: bool = Field(default=False, description="Upgrade to extra legroom")


@mcp.tool()
async def reserve_seat(flight_number: str, ctx: Context) -> str:
    """Reserve a seat on a flight. Asks the user which seat they want."""
    result = await ctx.elicit(f"Pick a seat on flight {flight_number}", schema=SeatPreference)
    if result.action != "accept":
        return f"The user did not pick a seat ({result.action})."

    legroom = " with extra legroom" if result.data.extra_legroom else ""
    return f"Seat {result.data.seat}{legroom} reserved on flight {flight_number}."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
