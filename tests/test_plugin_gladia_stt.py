from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, stt

from .stt_contract_cases import CASES_BY_NAME, assert_control_write_failure, make_provider

pytestmark = pytest.mark.plugin("gladia")


async def test_update_options_stores_region_for_new_streams():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key", region="eu-west")
    stt.update_options(region="us-west")

    assert stt._opts.region == "us-west"


async def test_update_options_rejects_translation_without_target_languages():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key")
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_enabled=True, region="us-west")
    assert stt._opts.translation_config.enabled is False
    assert stt._opts.region == "eu-west"

    stt = STT(api_key="test-key", translation_enabled=True, translation_target_languages=["fr"])
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_target_languages=[])
    assert stt._opts.translation_config.target_languages == ["fr"]

    stt.update_options(translation_enabled=False, translation_target_languages=[])
    assert stt._opts.translation_config.enabled is False


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
@pytest.mark.parametrize("completion", ["close", "post_final_transcript", "end_session"])
async def test_gladia_drains_final_transcript_after_input_ends(monkeypatch, completion):
    _, provider, transport = make_provider(CASES_BY_NAME["gladia"], monkeypatch)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))

    async def collect():
        return [event async for event in stream]

    try:
        stream.end_input()
        await transport.connected.wait()
        socket = transport.sockets[0]
        await _wait_for_stop(socket)
        await asyncio.sleep(0.01)
        socket.messages.put_nowait(
            aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                json.dumps(
                    {
                        "type": "transcript",
                        "data": {"is_final": True, "utterance": {"text": "final words"}},
                    }
                ),
                "",
            )
        )
        if completion == "close":
            socket.server_close()
        else:
            socket.messages.put_nowait(
                aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps({"type": completion}), "")
            )
        events = await asyncio.wait_for(collect(), 1)
        assert [
            event.alternatives[0].text
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["final words"]
    finally:
        await stream.aclose()
        await provider.aclose()


async def _wait_for_stop(socket):
    async def wait():
        while not any(isinstance(data, str) and "stop_recording" in data for data in socket.writes):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 1)


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_stop_recording_write_failure_is_reported(monkeypatch):
    await assert_control_write_failure(
        CASES_BY_NAME["gladia"], monkeypatch, "stop_recording", "end"
    )
