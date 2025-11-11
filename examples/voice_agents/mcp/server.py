from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")


@mcp.tool()
def get_weather(location: str) -> str:
    return f"The weather in {location} is a perfect sunny 70°F today. Enjoy your day!"


if __name__ == "__main__":
    mcp.run(transport="sse")
