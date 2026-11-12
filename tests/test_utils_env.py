import importlib.util
import os
import subprocess
import sys

import pytest

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.env import resolve_env_int, resolve_env_var

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("explicit", "env", "default", "expected"),
    [
        (NOT_GIVEN, {}, "", ""),
        (NOT_GIVEN, {}, "default", "default"),
        (NOT_GIVEN, {"LIVEKIT_INFERENCE_URL": "first", "LIVEKIT_URL": "second"}, "d", "first"),
        (NOT_GIVEN, {"LIVEKIT_URL": "second"}, "default", "second"),
        ("explicit", {"LIVEKIT_INFERENCE_URL": "first"}, "default", "explicit"),
        (NOT_GIVEN, {"LIVEKIT_INFERENCE_URL": ""}, "default", "default"),
        (NOT_GIVEN, {"LIVEKIT_INFERENCE_URL": " "}, "default", " "),
    ],
    ids=[
        "unset",
        "default",
        "first_env",
        "later_env",
        "explicit",
        "empty_is_unset",
        "blank_is_set",
    ],
)
def test_resolve_env_var(
    monkeypatch: pytest.MonkeyPatch,
    explicit: NotGivenOr[str],
    env: dict[str, str],
    default: str,
    expected: str,
) -> None:
    monkeypatch.delenv("LIVEKIT_INFERENCE_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert (
        resolve_env_var(explicit, "LIVEKIT_INFERENCE_URL", "LIVEKIT_URL", default=default)
        == expected
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 3), ("", 3), (" ", 3), ("true", 3), ("on", 3), ("1.5", 3)]
    + [("1", 1), ("0", 0), (" 2 ", 2)],
)
def test_resolve_env_int(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int) -> None:
    # the LK_* debug flags are parsed at import time, so an unparsable value must not raise
    if raw is None:
        monkeypatch.delenv("LK_DUMP_TTS", raising=False)
    else:
        monkeypatch.setenv("LK_DUMP_TTS", raw)

    assert resolve_env_int("LK_DUMP_TTS", default=3) == expected


def test_blank_debug_flags_keep_the_packages_importable() -> None:
    # agent workers import the openai plugin eagerly, and it reads LK_OPENAI_DEBUG at import
    modules = ["livekit.agents"]
    if importlib.util.find_spec("livekit.plugins.openai") is not None:
        modules.append("livekit.plugins.openai")
    blank = ["LK_DUMP_TTS", "LK_OPENAI_DEBUG", "LK_KEYTERMS_DEBUG", "LIVEKIT_EVALS_VERBOSE"]

    result = subprocess.run(
        [sys.executable, "-c", f"import {', '.join(modules)}"],
        env={**os.environ, **dict.fromkeys(blank, "")},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
