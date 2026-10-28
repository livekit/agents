"""Google SDK failures retain their structured diagnostic details."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.api_core.exceptions import BadRequest
from google.auth.credentials import AnonymousCredentials

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, utils
from livekit.plugins.google import STT, TTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("modality", ["stt", "tts"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_error_details_reach_the_caller(modality: str, streaming: bool) -> None:
    details = [{"reason": "invalid_request", "domain": "speech.googleapis.com"}]
    sdk_error = BadRequest("request rejected", details=details)
    client = SimpleNamespace(
        recognize=AsyncMock(side_effect=sdk_error),
        streaming_recognize=AsyncMock(side_effect=sdk_error),
        synthesize_speech=AsyncMock(side_effect=sdk_error),
        streaming_synthesize=AsyncMock(side_effect=sdk_error),
    )
    options = APIConnectOptions(max_retry=0)
    if modality == "stt":
        provider = STT(credentials=AnonymousCredentials(), project="test", model="chirp_3")

        async def connect(conn_options: APIConnectOptions):
            return client

        provider._pool = utils.ConnectionPool(connect_cb=connect)
    else:
        provider = TTS(credentials=AnonymousCredentials())
        provider._client = client
    try:
        with pytest.raises(APIStatusError) as failure:
            if modality == "stt" and not streaming:
                await provider.recognize(
                    rtc.AudioFrame(bytes(320), 16000, 1, 160), conn_options=options
                )
            else:
                stream = (
                    provider.stream(conn_options=options)
                    if streaming
                    else provider.synthesize("Hello", conn_options=options)
                )
                async with stream:
                    if streaming:
                        if modality == "tts":
                            stream.push_text("Hello.")
                        stream.end_input()
                    _ = [event async for event in stream]
        assert failure.value.status_code == 400
        assert failure.value.body == details
        assert failure.value.__cause__ is sdk_error
    finally:
        await provider.aclose()
