# Value of the `x-api-source` header on every Gradium websocket, so sessions opened
# by this plugin can be told apart from direct API traffic on the same API key.
API_SOURCE = "livekit-plugin"


def ws_headers(api_key: str) -> dict[str, str]:
    """Headers for a Gradium websocket: authentication plus source attribution."""
    return {"x-api-key": api_key, "x-api-source": API_SOURCE}
