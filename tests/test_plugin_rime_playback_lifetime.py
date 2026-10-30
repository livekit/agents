"""Playback reference cleanup through the real SDK and a local protocol peer."""

from __future__ import annotations

import asyncio

import pytest
from rimelabs_sdk import Rime, realtime as r
from websockets.asyncio.server import serve

from livekit import rtc
from livekit.plugins.rime.realtime._playback import Playback

from .test_plugin_rime import Peer, wait_until

pytestmark = pytest.mark.unit


async def test_queued_interruptions_keep_holds_past_cleanup_deadline():
    # SDK alpha26 has no local reference holds. Its existing reference-cache
    # limit remains the supported boundary for that configuration.
    count = 140 if hasattr(r.RealtimeSession, "hold_response") else 2
    peer = Peer()
    first_truncate = asyncio.Event()
    release_truncate = asyncio.Event()
    emit = peer.emit

    async def delayed_emit(kind, **body):
        if kind == "conversation.item.truncated" and not first_truncate.is_set():
            first_truncate.set()
            await release_truncate.wait()
        await emit(kind, **body)

    peer.emit = delayed_emit
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with (
            Rime(api_key="test") as client,
            client.realtime.connect(
                endpoint=endpoint, timeouts=r.RealtimeTimeouts(request_s=1.0)
            ) as sdk,
        ):
            tasks = []

            def spawn(operation):
                task = asyncio.create_task(operation)
                tasks.append(task)
                return task

            playback = Playback(sdk, "auto", 1.0, sdk.report_playback, spawn, lambda: True)
            outputs = {}
            tokens = {}
            ended = set()

            async def consume():
                async for event in sdk.events:
                    payload = event.payload
                    if isinstance(payload, r.ResponseStarted):
                        if payload.response.response_id in {"first", "second"}:
                            playback.start(payload.response)
                    elif isinstance(payload, r.MessageStarted):
                        outputs[payload.output.response.response_id] = payload.output
                        tokens[payload.output.item_id] = playback.output(payload.output)
                    elif isinstance(payload, r.AudioDelta):
                        frame = rtc.AudioFrame(
                            payload.audio.data, 24000, 1, len(payload.audio.data) // 2
                        )
                        playback.tag(tokens[payload.output.item_id], frame)
                    elif isinstance(payload, r.ResponseEnded):
                        playback.end(payload.response)
                        ended.add(payload.response.response_id)

            reader = asyncio.create_task(consume())
            try:
                for response_id in ("first", "second"):
                    await peer.start(response_id=response_id)
                    await peer.message(response_id)
                    await peer.end(response_id)
                await wait_until(lambda: "second" in ended)
                # Age both responses, then block the first report while the second
                # waits in the playback queue. Their cleanup deadlines now expire
                # before either acknowledgment, but before the SDK request timeout.
                await asyncio.sleep(0.8)
                playback.interrupt(r.PlaybackInterrupted(output=outputs["first"], audio_end_ms=10))
                await asyncio.wait_for(first_truncate.wait(), 1)
                playback.interrupt(r.PlaybackInterrupted(output=outputs["second"], audio_end_ms=10))
                await asyncio.sleep(0.25)
                for number in range(count):
                    response_id = f"ordinary-{number}"
                    await peer.start(response_id=response_id)
                    await peer.end(response_id)
                    await wait_until(lambda response_id=response_id: response_id in ended)
                release_truncate.set()
                await asyncio.wait_for(playback.settle(), 1)
                reported = [await asyncio.wait_for(peer.truncations.get(), 1) for _ in range(2)]
                assert [event["item_id"] for event in reported] == [
                    outputs["first"].item_id,
                    outputs["second"].item_id,
                ]
                assert peer.receipts.empty()
            finally:
                release_truncate.set()
                playback.close()
                reader.cancel()
                await asyncio.gather(reader, *tasks, return_exceptions=True)
