import importlib.util
import os
import subprocess
import sys

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils.env import resolve_env_int, resolve_env_var

pytestmark = pytest.mark.unit


class TestResolveEnvVar:
    """Tests for the resolve_env_var helper contract."""

    def test_returns_empty_string_when_no_env_or_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LIVEKIT_INFERENCE_URL", raising=False)

        assert resolve_env_var(NOT_GIVEN, "LIVEKIT_INFERENCE_URL") == ""

    def test_returns_default_when_no_matching_env_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LIVEKIT_INFERENCE_URL", raising=False)

        assert (
            resolve_env_var(
                NOT_GIVEN,
                "LIVEKIT_INFERENCE_URL",
                default="https://default.example.com",
            )
            == "https://default.example.com"
        )

    def test_returns_first_matching_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://inference.example.com")
        monkeypatch.setenv("LIVEKIT_URL", "https://livekit.example.com")

        assert (
            resolve_env_var(
                NOT_GIVEN,
                "LIVEKIT_INFERENCE_URL",
                "LIVEKIT_URL",
                default="https://default.example.com",
            )
            == "https://inference.example.com"
        )

    def test_falls_back_to_later_env_when_earlier_env_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LIVEKIT_INFERENCE_URL", raising=False)
        monkeypatch.setenv("LIVEKIT_URL", "https://livekit.example.com")

        assert (
            resolve_env_var(
                NOT_GIVEN,
                "LIVEKIT_INFERENCE_URL",
                "LIVEKIT_URL",
                default="https://default.example.com",
            )
            == "https://livekit.example.com"
        )

    def test_prefers_explicit_value_over_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://env.example.com")

        assert (
            resolve_env_var(
                "https://explicit.example.com",
                "LIVEKIT_INFERENCE_URL",
                default="https://default.example.com",
            )
            == "https://explicit.example.com"
        )

    def test_treats_empty_env_value_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "")

        assert (
            resolve_env_var(
                NOT_GIVEN,
                "LIVEKIT_INFERENCE_URL",
                default="https://default.example.com",
            )
            == "https://default.example.com"
        )

    def test_treats_whitespace_env_value_as_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_INFERENCE_URL", " ")

        assert (
            resolve_env_var(
                NOT_GIVEN,
                "LIVEKIT_INFERENCE_URL",
                default="https://default.example.com",
            )
            == " "
        )


class TestResolveEnvInt:
    """Contract for the ``LK_*`` debug flags, which are parsed at import time."""

    def test_unset_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LK_DUMP_TTS", raising=False)

        assert resolve_env_int("LK_DUMP_TTS") == 0

    @pytest.mark.parametrize("raw", ["", " ", "true", "on", "1.5"])
    def test_unparsable_value_falls_back_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        # an empty value is how an unset variable reaches a container, and `int("")` at
        # import time used to leave the whole package unimportable
        monkeypatch.setenv("LK_DUMP_TTS", raw)

        assert resolve_env_int("LK_DUMP_TTS", default=3) == 3

    @pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("0", 0), (" 2 ", 2)])
    def test_integer_values_are_parsed(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
    ) -> None:
        monkeypatch.setenv("LK_DUMP_TTS", raw)

        assert resolve_env_int("LK_DUMP_TTS") == expected


def test_blank_debug_flag_keeps_the_package_importable() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import livekit.agents"],
        env={
            **os.environ,
            "LK_DUMP_TTS": "",
            "LK_OPENAI_DEBUG": "",
            "LK_KEYTERMS_DEBUG": "",
            "LIVEKIT_EVALS_VERBOSE": "",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_blank_debug_flag_keeps_the_openai_plugin_importable() -> None:
    # the plugin reads LK_OPENAI_DEBUG at import time too, and agent workers import
    # it eagerly, so an unparsable value has to fall back there as well
    if importlib.util.find_spec("livekit.plugins.openai") is None:
        pytest.skip("livekit-plugins-openai is not installed")

    result = subprocess.run(
        [sys.executable, "-c", "import livekit.plugins.openai"],
        env={**os.environ, "LK_OPENAI_DEBUG": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
