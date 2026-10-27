"""Run the same public STT lifecycle against each provider's real local peer.

uv run pytest tests/stt_harness/test_contract.py --unit -k deepgram
"""

import asyncio
from dataclasses import replace

import pytest

from livekit.agents import APIError, stt, utils

from .discovery import websocket_providers
from .drivers import audio, push, running
from .server import STTServer

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.usefixtures("job_process")]
PROVIDERS = websocket_providers()


@pytest.fixture(params=PROVIDERS, ids=lambda provider: provider.name)
def provider(request):
    return request.param


async def test_streaming_audio_produces_events_before_input_ends(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, collector, server, events, errors):
        push(stream, provider, audio(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)

        async def wait_for_event():
            while not any(e.alternatives for e in events):
                if collector.done():
                    await collector
                    pytest.fail("the public stream ended while input was open")
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_event(), 1)
        assert not collector.done()
        assert not errors


async def test_final_audio_and_response_are_drained(provider, monkeypatch):
    if provider.completion_unknown:
        pytest.skip(provider.completion_unknown)
    server = STTServer(provider, segments=audio(provider))
    server.release_results.clear()
    async with running(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for segment in audio(provider):
            push(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        assert not collector.done(), "an acknowledgement was mistaken for completion"
        server.release_results.set()
        await asyncio.wait_for(collector, provider.final_timeout + 1)
        text = " ".join(
            e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        )
        assert text == "first tail"
        assert not errors
        assert bytes(server.received_audio) == server.expected_audio


async def test_flush_allows_more_audio(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, collector, server, _, errors):
        first, tail = audio(provider)
        push(stream, provider, first)
        stream.flush()
        await asyncio.wait_for(server.audio_received.wait(), 1)
        assert not collector.done()
        server.audio_received.clear()
        push(stream, provider, tail)
        stream.flush()
        await asyncio.wait_for(server.wait_for_audio(len(first.audio) + 1), 1)
        assert not collector.done()
        assert not errors


@pytest.mark.parametrize("draining", [False, True])
async def test_cancel_does_not_emit_error_or_retry(provider, monkeypatch, draining):
    server = STTServer(provider, segments=audio(provider))
    server.release_results.clear()
    async with running(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        for segment in audio(provider):
            push(stream, provider, segment)
        await asyncio.wait_for(server.audio_received.wait(), 1)
        if draining:
            stream.end_input()
            if not provider.completion_unknown:
                await asyncio.wait_for(server.final_requested.wait(), 1)
        server.allow_partial_input = True
        await asyncio.wait_for(stream.aclose(), 1)
        await asyncio.wait_for(collector, 1)
        assert not errors
        assert len(server.connections) == 1


@pytest.mark.parametrize("failure", ["close", "abort", "silent", "final_then_close"])
async def test_failed_connection_emits_error(provider, monkeypatch, failure):
    if failure == "silent" and not provider.monitored:
        pytest.skip("Provider owns heartbeat; its default timing needs transport-specific coverage")
    async with running(provider, monkeypatch) as (stream, collector, server, _, errors):
        push(stream, provider, audio(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        connection = server.connections[0]
        if failure in ("close", "final_then_close"):
            if failure == "final_then_close":
                await connection.finalize()
            connection.close()
        elif failure == "abort":
            connection.abort()
        else:
            server.answer_pings = False
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert errors
        assert errors[-1].recoverable is False


async def test_pongs_do_not_confirm_final_completion(provider, monkeypatch):
    if provider.completion_unknown:
        pytest.skip(provider.completion_unknown)
    server = STTServer(provider, segments=audio(provider))
    server.release_results.clear()
    async with running(provider, monkeypatch, server=server) as (stream, collector, _, _, errors):
        for segment in audio(provider):
            push(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert errors and not errors[-1].recoverable
        if provider.monitored:
            assert server.connections[0].pings > 0


async def test_server_ping_is_answered(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, _, server, _, errors):
        push(stream, provider, audio(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        connection = server.connections[0]
        await connection.ws.ping(b"server-probe")
        await asyncio.wait_for(connection.pong_received.wait(), 1)
        assert not errors


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.name == "deepgram"], ids=lambda p: p.name
)
async def test_quiet_server_stays_connected_when_pings_are_answered(provider, monkeypatch):
    if not provider.monitored:
        pytest.skip("Provider owns heartbeat; its default timing needs transport-specific coverage")
    server = STTServer(provider, segments=audio(provider))
    server.processing_stalled = True
    async with running(provider, monkeypatch, server=server) as (stream, collector, _, _, errors):
        push(stream, provider, audio(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        await asyncio.sleep(0.6)
        assert server.connections[0].pings >= 2
        assert not collector.done()
        assert not errors
        assert len(server.connections) == 1


async def test_failure_has_a_bounded_retry_budget(provider, monkeypatch):
    server = STTServer(provider)
    server.fault_after_audio = "abort"
    async with running(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):

        async def feed():
            while not collector.done():
                push(stream, provider, audio(provider)[0])
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed())
        try:
            with pytest.raises(APIError):
                await asyncio.wait_for(collector, 3)
        finally:
            await utils.aio.gracefully_cancel(feeder)
        assert len(server.connections) == provider.attempts_after_audio
        assert errors and not errors[-1].recoverable


async def test_connection_failure_reaches_fallback(provider, monkeypatch):
    server = STTServer(provider)
    server.fault_after_audio = "abort"
    async with running(provider, monkeypatch, server=server, fallback=True) as (
        stream,
        collector,
        _,
        events,
        _,
    ):

        async def feed():
            while not collector.done():
                push(stream, provider, audio(provider)[0])
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed())
        try:

            async def wait_for_backup():
                while not any(
                    e.alternatives and e.alternatives[0].text == "backup" for e in events
                ):
                    if collector.done():
                        await collector
                        pytest.fail("the stream ended before fallback produced a transcript")
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_backup(), 3)
        finally:
            await utils.aio.gracefully_cancel(feeder)


async def test_unverified_shutdown_uses_a_clean_bounded_drain(provider, monkeypatch):
    if not provider.completion_unknown:
        pytest.skip("provider has a specified completion sequence")
    async with running(provider, monkeypatch) as (stream, collector, server, _, errors):
        for segment in audio(provider):
            push(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert not errors
        assert bytes(server.received_audio) == server.expected_audio


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.name == "deepgram"], ids=lambda p: p.name
)
async def test_open_input_has_no_final_response_deadline(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, collector, server, _, errors):
        push(stream, provider, audio(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        await asyncio.sleep(provider.final_timeout + 0.1)
        assert not collector.done()
        assert not errors


async def test_empty_input_has_a_bounded_outcome(provider, monkeypatch):
    if provider.completion_unknown:
        pytest.skip(provider.completion_unknown)
    server = STTServer(provider)
    async with running(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        stream.end_input()
        try:
            await asyncio.wait_for(collector, provider.final_timeout + 1)
        except APIError:
            # Some services cannot commit an empty audio buffer. The public
            # stream must still finish and report the failure without retrying.
            assert errors and not errors[-1].recoverable
        else:
            assert not errors
        assert len(server.connections) <= 1
        assert not any(server.received_audio), "empty input may add only silence padding"
        assert not any(e.alternatives and e.alternatives[0].text for e in events)


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.name == "openai_server_vad"], ids=lambda p: p.name
)
async def test_out_of_order_commit_completions_are_all_drained(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, collector, server, events, errors):
        first, tail = audio(provider)
        server.automatic_finals = True
        await asyncio.wait_for(server.connected.wait(), 1)
        peer = server.connections[0]
        send = peer.send
        held = []

        def hold(data, *, final=False):
            if final:
                held.append(data)
            else:
                send(data)

        monkeypatch.setattr(peer, "send", hold)

        async def wait_for_results(count):
            while len(held) < count:
                if collector.done():
                    await collector
                    pytest.fail("stream ended before the requested commits arrived")
                await asyncio.sleep(0.01)

        push(stream, provider, first)
        await asyncio.wait_for(wait_for_results(1), 1)
        push(stream, provider, tail)
        stream.end_input()
        await asyncio.wait_for(wait_for_results(2), 1)
        # Both commit ACKs and the clear ACK have already been sent. The last
        # completion must not satisfy the earlier outstanding transcription.
        send(held[1])
        await asyncio.sleep(0.05)
        assert not collector.done()
        send(held[0])
        await asyncio.wait_for(collector, 1)
        assert sorted(
            e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ) == ["first", "tail"]
        assert not errors


async def test_provider_progress_restores_the_retry_budget(provider, monkeypatch):
    if provider.attempts_after_audio == 1:
        pytest.skip("provider does not replay consumed input")
    first = audio(provider)[0]
    server = STTServer(provider, segments=(first,) * 100)
    server.automatic_finals = provider.progress_requires_final
    async with running(provider, monkeypatch, server=server, max_retry=1) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):

        async def feed():
            while not collector.done():
                push(stream, provider, first)
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed())
        try:

            async def wait_for_progress():
                while not any(
                    e.alternatives
                    and e.alternatives[0].text
                    and (
                        not provider.progress_requires_final
                        or e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                    )
                    for e in events
                ):
                    if collector.done():
                        await collector
                        pytest.fail("the retry ended before the provider made progress")
                    await asyncio.sleep(0.01)

            initial = await asyncio.wait_for(server.connections_opened.get(), 1)
            await asyncio.wait_for(wait_for_progress(), 1)
            initial.abort()
            useful = await asyncio.wait_for(server.connections_opened.get(), 1)
            await asyncio.wait_for(useful.audio_received.wait(), 1)
            events.clear()
            await asyncio.wait_for(wait_for_progress(), 1)
            useful.abort()
            # Progress on the second connection permits a third attempt.
            last = await asyncio.wait_for(server.connections_opened.get(), 1)
            last.abort()
            with pytest.raises(APIError):
                await asyncio.wait_for(collector, 2)
            assert len(server.connections) == 3
            assert errors and not errors[-1].recoverable
        finally:
            await utils.aio.gracefully_cancel(feeder)


@pytest.mark.parametrize(
    ("provider", "code"),
    [(p, 1011) for p in PROVIDERS]
    + [(p, code) for p in PROVIDERS if p.name == "deepgram" for code in (1002, 1008, 1006)],
    ids=lambda value: value.name if hasattr(value, "name") else str(value),
)
async def test_abnormal_close_during_drain_is_an_error(provider, monkeypatch, code):
    server = STTServer(provider)
    server.release_results.clear()
    async with running(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        for segment in audio(provider):
            push(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.wait_for_audio(sum(len(s.audio) for s in audio(provider))), 1)
        server.fault_after_audio = "abort"
        peer = server.connections[0]
        if code == 1006:
            peer.abort()
        else:
            # A close control frame must not be held behind the processing gate.
            await peer.ws.close(code=code, message=b"provider close details")
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert errors and not errors[-1].recoverable
        assert 1 <= len(server.connections) <= provider.attempts_after_audio


async def test_queued_flushes_preserve_all_audio_and_results(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, collector, server, events, errors):
        first, tail = audio(provider)
        push(stream, provider, first)
        stream.flush()
        push(stream, provider, tail)
        stream.end_input()
        await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert bytes(server.received_audio) == server.expected_audio
        assert not errors
        if not provider.completion_unknown:
            assert (
                " ".join(
                    e.alternatives[0].text
                    for e in events
                    if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                )
                == "first tail"
            )


@pytest.mark.parametrize(
    "provider",
    [p for p in PROVIDERS if not p.completion_unknown and not p.completion_on_close],
    ids=lambda p: p.name,
)
async def test_normal_close_cannot_replace_a_required_completion(provider, monkeypatch):
    server = STTServer(provider, segments=audio(provider))
    server.release_results.clear()
    async with running(provider, monkeypatch, server=server) as (stream, collector, _, _, errors):
        for segment in audio(provider):
            push(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.wait_for_audio(len(server.expected_audio)), 1)
        await server.connections[0].ws.close()
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert errors and not errors[-1].recoverable


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.completion_unknown], ids=lambda p: p.name
)
async def test_best_effort_drain_accepts_normal_peer_close(provider, monkeypatch):
    async with running(provider, monkeypatch) as (stream, collector, server, _, errors):
        for segment in audio(provider):
            push(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.wait_for_audio(len(server.expected_audio)), 1)
        server.connections[0].close()
        await asyncio.wait_for(collector, provider.final_timeout + 1)
        assert not errors


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p.monitored], ids=lambda p: p.name)
async def test_fatal_close_does_not_retry(provider, monkeypatch):
    class RejectInput(provider.protocol):
        async def receive(self, message):
            await self.connection.ws.close(code=1008, message=b"request rejected")

    rejected = replace(provider, protocol=RejectInput)
    async with running(rejected, monkeypatch, max_retry=2) as (
        stream,
        collector,
        server,
        _,
        errors,
    ):
        push(stream, rejected, audio(rejected)[0])
        with pytest.raises(APIError) as failure:
            await asyncio.wait_for(collector, 2)
        assert not failure.value.retryable
        assert errors and not errors[-1].recoverable
        assert len(server.connections) == 1
