from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

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


def test_preloads_vad_and_eot_by_default(
    monkeypatch: pytest.MonkeyPatch, local_inference_calls: list[str]
) -> None:
    monkeypatch.delenv(_preload.ENV_PRELOAD_EOT, raising=False)

    _preload._local_inference_models()

    assert local_inference_calls == ["vad", "eot"]


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " off "])
def test_eot_preload_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, local_inference_calls: list[str], value: str
) -> None:
    # the expensive half is opt-out; the VAD is cheap and stays either way
    monkeypatch.setenv(_preload.ENV_PRELOAD_EOT, value)

    _preload._local_inference_models()

    assert local_inference_calls == ["vad"]


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_truthy_values_keep_the_eot_preload(
    monkeypatch: pytest.MonkeyPatch, local_inference_calls: list[str], value: str
) -> None:
    monkeypatch.setenv(_preload.ENV_PRELOAD_EOT, value)

    _preload._local_inference_models()

    assert local_inference_calls == ["vad", "eot"]
