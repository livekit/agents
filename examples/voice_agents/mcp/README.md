# MCP tools for a voice agent

This example connects a voice agent to the local MCP server in `server.py`.
Start that server before running `mcp-agent.py` with the credentials and setup
in the [voice agent examples](../README.md).

## Search private documents with a remote MCP server

[ClickHouse Loom](https://github.com/ClickHouse/loom) exposes document search
over MCP. To use it here, set `LOOM_MCP_URL` to the full `/mcp` URL of the
deployment and `LOOM_TOKEN` to a namespace-scoped viewer token. Use HTTPS for a
remote deployment. Use a server that exposes the `loom_knowledge_search` tool.

Add `import os` to `mcp-agent.py`. Build this toolset in the entrypoint and pass
`tools=tools` to `AgentSession`:

```python
tools = [
    mcp.MCPToolset(
        id="loom_documents",
        mcp_server=mcp.MCPServerHTTP(
            url=os.environ["LOOM_MCP_URL"],
            transport_type="streamable_http",
            headers={"Authorization": f"Bearer {os.environ['LOOM_TOKEN']}"},
            allowed_tools=["loom_knowledge_search"],
        ),
    ),
]
```

Update the agent instructions to use `loom_knowledge_search` for questions
about the indexed documents. Start the voice agent without the local demo
server. The tool allowlist limits the tools exposed to the agent, and the
Loom token controls its access to documents. See the
[Loom MCP reference](https://github.com/ClickHouse/loom/blob/main/docs/reference/mcp.md).
