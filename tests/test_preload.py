from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.ipc import _preload
from livekit.agents.plugin import Plugin
from livekit.agents.worker import AgentServer, WorkerOptions

pytestmark = pytest.mark.unit


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(),
    reason="forkserver is not available on this platform",
)
def test_forkserver_preload_preserves_child_gc() -> None:
    helper = Path(__file__).parent / "utils" / "forkserver_preload.py"
    result = subprocess.run(
        [sys.executable, str(helper)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    outcome = json.loads(result.stdout)

    assert outcome["freeze_count"] > 0
    assert outcome["child_cycle_collected"] is True


def test_preload_resolves_every_inference_export() -> None:
    # a fresh interpreter, as a job process is: the first job imports nothing for these
    code = (
        "import sys\n"
        "import livekit.agents.ipc._preload\n"
        "from livekit.agents import inference\n"
        "before = set(sys.modules)\n"
        "for name in inference.__all__: getattr(inference, name)\n"
        "print('\\n'.join(sorted(set(sys.modules) - before)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE": "0"},
    )

    assert result.stdout.split() == []


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(),
    reason="forkserver is not available on this platform",
)
@pytest.mark.parametrize(
    ("preload_modules", "expected"),
    [
        (None, ["plugin.a"]),
        (["app.a", "app.b"], ["plugin.a", "app.a", "app.b"]),
        (["app.a", "plugin.a", "app.a"], ["plugin.a", "app.a"]),
    ],
)
async def test_preload_modules(preload_modules: list[str] | None, expected: list[str]) -> None:
    server = AgentServer.from_server_options(
        WorkerOptions(
            entrypoint_fnc=AsyncMock(),
            multiprocessing_context="forkserver",
            num_idle_processes=0,
            preload_modules=preload_modules,
        )
    )
    server._simulation = True
    server._mp_ctx = mp_context = MagicMock()

    pool_started = asyncio.Event()
    process_pool = MagicMock()
    process_pool.start = AsyncMock(side_effect=lambda: pool_started.set())
    process_pool.aclose = AsyncMock()

    with (
        patch.object(Plugin, "registered_plugins", [MagicMock(package="plugin.a")]),
        patch("livekit.agents.ipc.proc_pool.ProcPool", return_value=process_pool),
    ):
        run_task = asyncio.create_task(server.run(unregistered=True))
        try:
            await asyncio.wait_for(pool_started.wait(), timeout=5)
            assert mp_context.set_forkserver_preload.call_args.args[0] == expected + [
                "livekit.agents.ipc._preload",
                "livekit.agents.ipc._preload_freeze",
            ]
        finally:
            await server.aclose()
            await run_task


@pytest.fixture
def local_inference_calls() -> Iterator[list[str]]:
    """Record which local-inference singletons the warm-up touches."""
    calls: list[str] = []
    with (
        patch("livekit.local_inference.init_vad", side_effect=lambda: calls.append("vad")),
        patch("livekit.local_inference.init_eot", side_effect=lambda: calls.append("eot")),
    ):
        yield calls


@pytest.mark.parametrize("spawned, expected", [(False, ["vad", "eot"]), (True, ["vad"])])
def test_default_preloads_eot_only_without_a_parent_process(
    monkeypatch: pytest.MonkeyPatch,
    local_inference_calls: list[str],
    spawned: bool,
    expected: list[str],
) -> None:
    # the forkserver has no parent and shares the weights; a spawned job process would not
    monkeypatch.delenv(_preload.ENV_PRELOAD_LOCAL_INFERENCE, raising=False)
    monkeypatch.setattr("multiprocessing.parent_process", lambda: MagicMock() if spawned else None)

    _preload._local_inference_models()

    assert local_inference_calls == expected


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " off "])
def test_local_inference_preload_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, local_inference_calls: list[str], value: str
) -> None:
    monkeypatch.setenv(_preload.ENV_PRELOAD_LOCAL_INFERENCE, value)

    _preload._local_inference_models()

    assert local_inference_calls == []


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_truthy_values_preload_both_models(
    monkeypatch: pytest.MonkeyPatch, local_inference_calls: list[str], value: str
) -> None:
    # an explicit value wins over the default, even in a spawned job process
    monkeypatch.setenv(_preload.ENV_PRELOAD_LOCAL_INFERENCE, value)
    monkeypatch.setattr("multiprocessing.parent_process", lambda: MagicMock())

    _preload._local_inference_models()

    assert local_inference_calls == ["vad", "eot"]
