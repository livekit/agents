# MCP tools for a voice agent

This example connects a voice agent to the local MCP server in `server.py`.
Start that server before running `mcp-agent.py` with the credentials and setup
in the [voice agent examples](../README.md).

## Search public documents with a remote MCP server

[ClickHouse Loom](https://github.com/ClickHouse/loom) exposes document search
over MCP. Create a Loom namespace containing only public documents, then
[create a viewer token](https://github.com/ClickHouse/loom/blob/v1.3.0/docs/guides/tokens-and-grants.md)
scoped to that namespace. Set `LOOM_PUBLIC_DOCS_TOKEN` to that token and
`LOOM_MCP_URL` to your deployment's HTTPS URL with `/mcp?toolset=assistant`.

The worker uses this token for every caller, and the agent's replies can be
heard by other participants in the room. For private search, authenticate the
caller in your application, use a Loom token scoped to that caller's grants in
the session, and restrict the room to participants authorized to hear the results.

Add `import os` to `mcp-agent.py`. Build this toolset in the entrypoint and pass
`tools=tools` to `AgentSession`:

```python
tools = [
    mcp.MCPToolset(
        id="loom_public_documents",
        mcp_server=mcp.MCPServerHTTP(
            url=os.environ["LOOM_MCP_URL"],
            transport_type="streamable_http",
            headers={"Authorization": f"Bearer {os.environ['LOOM_PUBLIC_DOCS_TOKEN']}"},
            allowed_tools=["loom_knowledge_search"],
        ),
    ),
]
```

Update the agent instructions to use `loom_knowledge_search` for questions
about the public documents. Start the voice agent without the local demo
server. The tool allowlist limits which tools the agent can call, and the
Loom token controls which documents it can read. See the
[Loom MCP reference](https://github.com/ClickHouse/loom/blob/main/docs/reference/mcp.md).
