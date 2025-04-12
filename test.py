from mitmproxy import http, ctx

def response(flow: http.HTTPFlow) -> None:
    # Rewrite Location header if needed.
    if flow.response.status_code in (301, 302) and "Location" in flow.response.headers:
        location = flow.response.headers["Location"]
        if ":25565" in location:
            new_location = location.replace(":25565", "")
            flow.response.headers["Location"] = new_location
            ctx.log.info(f"Rewrote Location header from {location} to {new_location}")

    # Rewrite the body content if it's HTML.
    if flow.response.headers.get("content-type", "").startswith("text/html"):
        # Check if the unwanted port is present in the body.
        if b":25565" in flow.response.content:
            new_content = flow.response.content.replace(b":25565", b"")
            flow.response.content = new_content
            ctx.log.info("Rewrote HTML body to remove port 25565")
