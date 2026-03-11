import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils.env import resolve_env_var


class TestResolveEnvVar:
    """Tests for the resolve_env_var helper contract."""

    def test_returns_empty_string_when_no_env_or_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
