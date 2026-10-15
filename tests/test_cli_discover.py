from __future__ import annotations

import sys
from pathlib import Path

import pytest

from livekit.agents import AgentServer
from livekit.agents.__main__ import _discover_server

pytestmark = pytest.mark.unit


@pytest.fixture
def entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def write(name: str, source: str) -> str:
        path = tmp_path / f"{name}.py"
        path.write_text(source)
        monkeypatch.delitem(sys.modules, name, raising=False)
        return str(path)

    return write


def test_discovers_module_level_server(entrypoint) -> None:
    path = entrypoint(
        "agent_server_style",
        """
from livekit.agents import AgentServer, JobContext

server = AgentServer()

@server.rtc_session(agent_name="named")
async def entry(ctx: JobContext) -> None:
    pass
""",
    )
    server = _discover_server(path)
    assert isinstance(server, AgentServer)
    assert server._agent_name == "named"


def test_discovers_worker_options_run_from_main(entrypoint) -> None:
    path = entrypoint(
        "worker_options_style",
        """
from livekit.agents import JobContext, WorkerOptions, cli

async def entrypoint(ctx: JobContext) -> None:
    pass

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="legacy"))
""",
    )
    server = _discover_server(path)
    assert isinstance(server, AgentServer)
    assert server._agent_name == "legacy"
    # job processes import the entrypoint by reference, which a __main__ function defeats
    assert server._entrypoint_fnc is not None
    assert server._entrypoint_fnc.__module__ == "worker_options_style"
