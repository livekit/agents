# MCP lifecycle checks

Run from the repository root with the Python MCP SDK and uvicorn installed:

```sh
MCP_TEST_PYTHON="$PWD/.venv/bin/python" cargo test --manifest-path rust/Cargo.toml \
  -p lk-agents-runtime --test runtime -- mcp_lifecycle::
uv run pytest --unit tests/test_mcp_lifecycle.py
uv run pytest --e2e tests/test_mcp_lifecycle_e2e.py
```

The SDK peer serves paginated discovery, typed RPC errors and a blocked call over HTTP and
stdio. Checks observe cancellation, reader/call settlement, failed HTTP DELETE cleanup and
native stdio exit/reaping. Python checks exercise the documented server/toolset constructors,
per-tool options and partial-setup cleanup.
