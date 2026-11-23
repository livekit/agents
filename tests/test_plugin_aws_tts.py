"""Amazon Polly TTS request setup, without AWS access."""

from __future__ import annotations

from typing import Any

import pytest
from aiobotocore.session import AioSession, get_session

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.aws import TTS

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_aws_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))


async def _synthesis_region(
    monkeypatch: pytest.MonkeyPatch, instance: TTS, session: AioSession
) -> str | None:
    regions: list[str | None] = []

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def synthesize_speech(self, **kwargs: object) -> dict:
            raise RuntimeError("no AWS access in tests")

    def _create_client(service_name: str, **kwargs: Any) -> _Client:
        assert service_name == "polly"
        # like botocore, an explicit region_name wins over the session's configured region
        regions.append(kwargs.get("region_name") or session.get_config_variable("region"))
        return _Client()

    monkeypatch.setattr(session, "create_client", _create_client)
    with pytest.raises(APIConnectionError):
        await instance.synthesize("hi", conn_options=APIConnectOptions(max_retry=0)).collect()

    assert len(regions) == 1
    return regions[0]


async def test_region_applies_to_a_given_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = get_session()
    instance = TTS(region="eu-west-1", session=session)
    assert await _synthesis_region(monkeypatch, instance, session) == "eu-west-1"


async def test_region_defaults_to_us_east_1(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = TTS()
    assert await _synthesis_region(monkeypatch, instance, instance._session) == "us-east-1"


async def test_session_region_is_used_without_a_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    instance = TTS()
    assert await _synthesis_region(monkeypatch, instance, instance._session) == "ap-southeast-2"
