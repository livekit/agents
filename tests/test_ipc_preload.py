from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.ipc import _preload

pytestmark = pytest.mark.unit


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
