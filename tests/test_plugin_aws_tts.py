"""Amazon Polly TTS request setup, without AWS access."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiobotocore.config import AioConfig
from aiobotocore.session import AioSession, get_session
from aiobotocore.stub import AioStubber

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.aws import TTS

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_aws_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    # dummy credentials, so creating a client never looks anywhere else for them
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


async def _synthesis_region(
    monkeypatch: pytest.MonkeyPatch, instance: TTS, session: AioSession
) -> str:
    """Synthesize once with a real Polly client whose call is stubbed, and return its region."""
    regions: list[str] = []
    create_client = session.create_client

    @asynccontextmanager
    async def _create_client(service_name: str, **kwargs: Any) -> AsyncIterator[Any]:
        assert service_name == "polly"
        async with create_client(service_name, **kwargs) as client:
            # the region botocore resolved, so the SDK's own precedence is what is tested
            regions.append(client.meta.region_name)
            with AioStubber(client) as stubber:
                stubber.add_client_error("synthesize_speech", http_status_code=400)
                yield client

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


async def test_aws_region_env_is_used_without_a_region(monkeypatch: pytest.MonkeyPatch) -> None:
    # as the STT reads it; botocore itself only reads AWS_DEFAULT_REGION
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    instance = TTS()
    assert await _synthesis_region(monkeypatch, instance, instance._session) == "ap-south-1"


async def test_session_default_client_config_region_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    session = get_session()
    session.set_default_client_config(AioConfig(region_name="eu-central-1"))
    instance = TTS(session=session)
    assert await _synthesis_region(monkeypatch, instance, session) == "eu-central-1"


async def test_region_wins_over_session_default_client_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = get_session()
    session.set_default_client_config(AioConfig(region_name="eu-central-1"))
    instance = TTS(region="eu-west-1", session=session)
    assert await _synthesis_region(monkeypatch, instance, session) == "eu-west-1"
