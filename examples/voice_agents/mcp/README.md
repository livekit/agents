# MCP examples

[`mcp-agent.py`](./mcp-agent.py) connects to the local [`server.py`](./server.py)
for simulated weather and flight booking tools. The booking tool demonstrates
progress notifications and cancellation.

## Add remote web search with Parallel

You can add public web search and page extraction to the same agent using
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).
The anonymous endpoint needs no Parallel account, API key, or authorization
header. Free access is rate limited.

The voice examples already include the `livekit-agents[mcp]` dependency. In
`mcp-agent.py`, add this entry to the existing `AgentSession(tools=[...])` list,
alongside the local toolset:

```python
mcp.MCPToolset(
    id="parallel_search",
    mcp_server=mcp.MCPServerHTTP(
        url="https://search.parallel.ai/mcp",
        transport_type="streamable_http",
        allowed_tools=["web_search", "web_fetch"],
        # Web retrieval can take longer than the default five-second tool timeout.
        client_session_timeout_seconds=30,
    ),
)
```

Keep the local server running for the existing weather and booking tools. The
session connects to both servers and discovers their tools when it starts.
There is no separate Parallel plugin or local Parallel server to run.

Add guidance to `MyAgent`'s instructions, such as: "Use web_search for current
public information and web_fetch to read a public URL. Name the sources you
used, and say when retrieval fails instead of guessing."

Once this toolset is enabled, the agent can invoke its tools during the
conversation. Search queries, requested URLs, and any supplied objectives,
context, or metadata are sent to Parallel. Avoid putting private conversation
details in retrieval requests. Remove the added toolset and its instruction
text to disable this integration.

Anonymous search does not replace the voice agent's credentials. The existing
LiveKit connection and LiveKit Inference still need their normal configuration
and may incur charges. This addition keeps the agent's speech and inference
settings, local tools, and booking progress/cancellation behavior unchanged.
