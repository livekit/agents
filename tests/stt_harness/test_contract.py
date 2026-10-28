"""Run the same public STT lifecycle against each provider's real local peer.

uv run pytest tests/stt_harness/test_contract.py --unit -k deepgram
"""

import asyncio
import logging
from dataclasses import replace

import pytest

from livekit.agents import APIError, APIStatusError, stt, utils

from .discovery import websocket_providers
from .drivers import make_audio_segments, open_stream, push_audio, wait_for_final
from .protocol import Transcript
from .server import Segment, STTServer

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.usefixtures("job_process")]
PROVIDERS = websocket_providers()


@pytest.fixture(params=PROVIDERS, ids=lambda provider: provider.name)
def provider(request):
    return request.param


async def test_streaming_audio_produces_events_before_input_ends(provider, monkeypatch):
    async with open_stream(provider, monkeypatch) as (stream, collector, server, events, errors):
        push_audio(stream, provider, make_audio_segments(provider)[0])
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
    if provider.completion_unknown_reason:
        pytest.skip(provider.completion_unknown_reason)
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.release_results.clear()
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        assert not collector.done(), "an acknowledgement was mistaken for completion"
        server.release_results.set()
        await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        text = " ".join(
            e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        )
        assert text == "first tail"
        assert not errors
        assert bytes(server.received_audio) == server.expected_audio


async def test_flush_allows_more_audio(provider, monkeypatch):
    async with open_stream(provider, monkeypatch) as (stream, collector, server, _, errors):
        first, tail = make_audio_segments(provider)
        push_audio(stream, provider, first)
        stream.flush()
        await asyncio.wait_for(server.audio_received.wait(), 1)
        assert not collector.done()
        server.audio_received.clear()
        push_audio(stream, provider, tail)
        stream.flush()
        await asyncio.wait_for(server.wait_for_audio(len(first.audio) + 1), 1)
        assert not collector.done()
        assert not errors


@pytest.mark.parametrize("quiet_stop", [False, True])
async def test_finalized_turns_do_not_end_the_stream(provider, monkeypatch, quiet_stop):
    if quiet_stop and provider.require_completion and not provider.completion_unknown_reason:
        pytest.skip("this protocol requires a stop exchange")
    first, tail = make_audio_segments(provider)
    # Use complete wire chunks: some providers deliberately ignore flush().
    segments = (replace(first, audio=first.audio * 2), replace(tail, audio=first.audio * 2))
    server = STTServer(provider, segments=segments)
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for index, segment in enumerate(segments):
            push_audio(stream, provider, segment)
            stream.flush()
            size = sum(len(s.audio) for s in segments[: index + 1])
            await asyncio.wait_for(server.wait_for_audio(size), 1)
            await server.connections[0].protocol.finish_turn()

            await asyncio.wait_for(wait_for_final(collector, events, segment.text), 1)
            assert not collector.done()
        server.ignore_client_messages = quiet_stop
        stream.end_input()
        await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert not errors


@pytest.mark.parametrize("draining", [False, True])
async def test_cancel_does_not_emit_error_or_retry(provider, monkeypatch, draining):
    provider = replace(provider, final_response_timeout=5)
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.release_results.clear()
    async with open_stream(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        await asyncio.wait_for(server.audio_received.wait(), 1)
        if draining:
            stream.end_input()
            if not provider.completion_unknown_reason:
                await asyncio.wait_for(server.final_requested.wait(), 1)
        server.allow_partial_input = True
        await asyncio.wait_for(stream.aclose(), 1)
        await asyncio.wait_for(collector, 1)
        assert not errors
        assert len(server.connections) == 1


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.drain_quiet_timeout is not None], ids=lambda p: p.name
)
async def test_finalized_input_finishes_after_a_short_quiet_window(provider, monkeypatch):
    provider = replace(provider, final_response_timeout=provider.drain_quiet_timeout + 2)
    first = make_audio_segments(provider)[0]
    server = STTServer(provider, segments=(first,))
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        push_audio(stream, provider, first)
        await asyncio.wait_for(server.wait_for_audio(len(first.audio)), 1)
        await server.connections[0].protocol.finish_turn()
        await asyncio.wait_for(wait_for_final(collector, events, first.text), 1)
        server.ignore_client_messages = True
        start = asyncio.get_running_loop().time()
        stream.end_input()
        await asyncio.wait_for(collector, provider.drain_quiet_timeout + 0.5)
        assert asyncio.get_running_loop().time() - start >= provider.drain_quiet_timeout
        assert not errors


async def test_cancel_during_remote_failure_releases_resources(provider, monkeypatch):
    async with open_stream(provider, monkeypatch, max_retry=2) as (stream, _, server, _, _):
        push_audio(stream, provider, make_audio_segments(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        server.allow_partial_input = True
        server.connections[0].abort()
        # Let receive observe the failure before the client cancels the stream.
        await asyncio.sleep(0)
        await asyncio.wait_for(stream.aclose(), 1)
        # open_stream and fail_on_leaked_tasks check connections and child tasks.


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.supports_prewarm], ids=lambda p: p.name
)
async def test_plugin_close_releases_prewarmed_connections(provider, monkeypatch):
    async with STTServer(provider) as server:
        instance = provider.create_stt(server, monkeypatch)
        try:
            instance.prewarm()
            await asyncio.wait_for(server.connected.wait(), 1)
        finally:
            await instance.aclose()
            await server.wait_for_disconnects()


@pytest.mark.parametrize("failure", ["close", "abort", "silent", "final_then_close"])
async def test_failed_connection_emits_error(provider, monkeypatch, failure):
    if failure == "final_then_close" and provider.reconnect_on_idle_close:
        pytest.skip("idle retirement is tested with the next utterance")
    if failure == "silent" and not provider.uses_websocket_monitor:
        pytest.skip("Provider owns heartbeat; its default timing needs transport-specific coverage")
    async with open_stream(provider, monkeypatch) as (stream, collector, server, _, errors):
        push_audio(stream, provider, make_audio_segments(provider)[0])
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
            await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert errors
        assert errors[-1].recoverable is False


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.reconnect_on_idle_close], ids=lambda p: p.name
)
async def test_idle_retirement_preserves_the_next_utterance(provider, monkeypatch):
    server = STTServer(provider, segments=make_audio_segments(provider))
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for index, segment in enumerate(make_audio_segments(provider)):
            push_audio(stream, provider, segment)
            stream.flush()
            connection = await asyncio.wait_for(server.connections_opened.get(), 1)
            size = sum(len(s.audio) for s in make_audio_segments(provider)[: index + 1])
            await asyncio.wait_for(server.wait_for_audio(size), 1)
            await connection.finalize()

            await asyncio.wait_for(wait_for_final(collector, events, segment.text), 1)
            connection.close()
            await asyncio.wait_for(connection.disconnected.wait(), 1)
            await asyncio.sleep(0.01)
        stream.end_input()
        await asyncio.wait_for(collector, 1)
        assert len(server.connections) == 2
        assert not errors


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.reconnect_on_idle_close], ids=lambda p: p.name
)
@pytest.mark.parametrize("end_after_close", [False, True])
async def test_idle_retirement_preserves_a_partial_audio_chunk(
    provider, monkeypatch, end_after_close
):
    first, tail = make_audio_segments(provider)
    partial = tail.audio[: provider.sample_rate // 50 * 2]
    if end_after_close:
        tail = replace(tail, audio=partial)
    server = STTServer(provider, segments=(first, tail))
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        push_audio(stream, provider, first)
        stream.flush()
        peer = await asyncio.wait_for(server.connections_opened.get(), 1)
        await asyncio.wait_for(wait_for_final(collector, events, first.text), 1)

        push_audio(stream, provider, Segment(partial, tail.text))
        # A round trip lets the ready sender consume the frame into its chunk buffer.
        await peer.ws.ping(b"partial-audio-buffered")
        await asyncio.wait_for(peer.pong_received.wait(), 1)
        assert bytes(server.received_audio) == first.audio
        peer.close()
        await asyncio.wait_for(peer.disconnected.wait(), 1)
        await asyncio.sleep(0.01)

        if not end_after_close:
            push_audio(stream, provider, replace(tail, audio=tail.audio[len(partial) :]))
        stream.end_input()
        await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert bytes(server.received_audio) == first.audio + tail.audio
        assert len(server.connections) == 2
        assert bytes(server.connections[1].audio) == tail.audio
        assert not errors


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.name == "reson8"], ids=lambda p: p.name
)
@pytest.mark.parametrize("final_response_timeout", [0.2, 1.3])
@pytest.mark.parametrize("progress", ["start", "candidate", "next_turn"])
async def test_started_turn_requires_turn_end(
    provider, monkeypatch, final_response_timeout, progress
):
    provider = replace(provider, final_response_timeout=final_response_timeout)
    first = make_audio_segments(provider)[0]
    server = STTServer(provider, segments=(first,))
    server.processing_stalled = True
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        push_audio(stream, provider, first)
        stream.flush()
        await asyncio.wait_for(server.wait_for_audio(len(first.audio)), 1)
        peer = server.connections[0]
        peer.send({"type": "turn_start"})
        if progress != "start":
            peer.send({"type": "turn_end_candidate", "text": first.text})
        if progress == "next_turn":
            peer.send({"type": "turn_end"})
            peer.send({"type": "turn_start"})

        async def wait_for_turns():
            count = 2 if progress == "next_turn" else 1
            while sum(e.type == stt.SpeechEventType.START_OF_SPEECH for e in events) < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_turns(), 1)
        stream.end_input()
        with pytest.raises(APIError, match="final response timed out"):
            await asyncio.wait_for(collector, final_response_timeout + 1)
        assert errors and not errors[-1].recoverable


@pytest.mark.parametrize(
    "provider",
    [p for p in PROVIDERS if p.name == "reson8" or p.drain_quiet_timeout is not None],
    ids=lambda p: p.name,
)
async def test_active_turn_uses_the_configured_final_response_budget(provider, monkeypatch):
    quiet_timeout = provider.drain_quiet_timeout or 1.0
    provider = replace(provider, final_response_timeout=quiet_timeout + 1)
    first = make_audio_segments(provider)[0]
    server = STTServer(provider, segments=(first,))
    server.processing_stalled = True
    server.release_results.clear()
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        push_audio(stream, provider, first)
        await asyncio.wait_for(server.wait_for_audio(len(first.audio)), 1)
        peer = server.connections[0]
        await peer.protocol.send_transcript(Transcript(first.text, False, 0, 0.4, 0))

        async def wait_for_partial():
            while not any(event.alternatives for event in events):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_partial(), 1)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        await asyncio.sleep(quiet_timeout + 0.05)
        assert not collector.done(), "the quiet window ended an active turn"
        server.release_results.set()
        await asyncio.wait_for(collector, quiet_timeout + 0.5)
        await wait_for_final(collector, events, first.text)
        assert not errors


async def test_pongs_do_not_confirm_final_completion(provider, monkeypatch):
    if provider.completion_unknown_reason:
        pytest.skip(provider.completion_unknown_reason)
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.processing_stalled = True
    server.release_results.clear()
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        if provider.require_completion:
            with pytest.raises(APIError):
                await asyncio.wait_for(collector, provider.final_response_timeout + 1)
            assert errors and not errors[-1].recoverable
        else:
            await asyncio.wait_for(collector, provider.final_response_timeout + 1)
            assert not errors
        if provider.uses_websocket_monitor:
            assert server.connections[0].pings > 0


async def test_server_ping_is_answered(provider, monkeypatch):
    async with open_stream(provider, monkeypatch) as (stream, _, server, _, errors):
        push_audio(stream, provider, make_audio_segments(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        connection = server.connections[0]
        await connection.ws.ping(b"server-probe")
        await asyncio.wait_for(connection.pong_received.wait(), 1)
        assert not errors


@pytest.mark.parametrize(
    "provider",
    [p for p in PROVIDERS if p.name in {"deepgram", "assemblyai", "fireworksai", "gradium"}],
    ids=lambda p: p.name,
)
async def test_quiet_server_stays_connected_when_pings_are_answered(provider, monkeypatch, caplog):
    if not provider.uses_websocket_monitor:
        pytest.skip("Provider owns heartbeat; its default timing needs transport-specific coverage")
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.processing_stalled = True
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        push_audio(stream, provider, make_audio_segments(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        caplog.clear()
        await asyncio.sleep(0.6)
        assert server.connections[0].pings >= 2
        assert not collector.done()
        assert not errors
        assert len(server.connections) == 1
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


async def test_failure_has_a_bounded_retry_budget(provider, monkeypatch):
    server = STTServer(provider)
    server.fault_after_audio = "abort"
    async with open_stream(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):

        async def feed():
            while not collector.done():
                push_audio(stream, provider, make_audio_segments(provider)[0])
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(feed())
        try:
            with pytest.raises(APIError):
                await asyncio.wait_for(collector, 3)
        finally:
            await utils.aio.gracefully_cancel(feeder)
        assert len(server.connections) == provider.expected_attempts_after_audio
        assert errors and not errors[-1].recoverable


REQUEST_ERRORS = tuple(
    (provider, error) for provider in PROVIDERS for error in provider.request_errors
)


@pytest.mark.parametrize("provider,request_error", REQUEST_ERRORS, ids=lambda value: value.name)
async def test_request_error_obeys_retry_policy(provider, request_error, monkeypatch):
    server = STTServer(provider)
    server.request_error = request_error
    async with open_stream(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        # Keep input open so end_input() cannot suppress an incorrect retry.
        push_audio(stream, provider, make_audio_segments(provider)[0])
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, 3)
        attempts = provider.expected_attempts_after_audio if request_error.retryable else 1
        assert server.rejected_requests == attempts
        assert [error.recoverable for error in errors] == [True] * (attempts - 1) + [False]
        assert all(error.error.retryable == request_error.retryable for error in errors)
        if request_error.http_status is not None:
            assert all(isinstance(error.error, APIStatusError) for error in errors)
            assert all(error.error.status_code == request_error.http_status for error in errors)


@pytest.mark.parametrize(
    "provider,request_error",
    [(provider, None) for provider in PROVIDERS]
    + [(provider, error) for provider, error in REQUEST_ERRORS if not error.retryable],
    ids=lambda value: value.name if value is not None else "abort",
)
async def test_connection_failure_reaches_fallback(provider, monkeypatch, request_error):
    server = STTServer(provider)
    if request_error is None:
        server.fault_after_audio = "abort"
    else:
        server.request_error = request_error
    async with open_stream(provider, monkeypatch, server=server, fallback=True) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):

        async def feed():
            while not collector.done():
                push_audio(stream, provider, make_audio_segments(provider)[0])
                stream.flush()
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
            assert any(not error.recoverable for error in errors)
            if request_error is not None:
                assert len(errors) == 1
                assert not errors[0].error.retryable
                assert server.rejected_requests == 1
        finally:
            await utils.aio.gracefully_cancel(feeder)


async def test_unverified_shutdown_uses_a_clean_bounded_drain(provider, monkeypatch):
    if not provider.completion_unknown_reason:
        pytest.skip("provider has a specified completion sequence")
    async with open_stream(provider, monkeypatch) as (stream, collector, server, _, errors):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert not errors
        assert bytes(server.received_audio) == server.expected_audio


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.name == "deepgram"], ids=lambda p: p.name
)
async def test_open_input_has_no_final_response_deadline(provider, monkeypatch):
    async with open_stream(provider, monkeypatch) as (stream, collector, server, _, errors):
        push_audio(stream, provider, make_audio_segments(provider)[0])
        await asyncio.wait_for(server.audio_received.wait(), 1)
        await asyncio.sleep(provider.final_response_timeout + 0.1)
        assert not collector.done()
        assert not errors


async def test_empty_input_has_a_bounded_outcome(provider, monkeypatch):
    if provider.completion_unknown_reason:
        pytest.skip(provider.completion_unknown_reason)
    server = STTServer(provider)
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        stream.end_input()
        try:
            await asyncio.wait_for(collector, provider.final_response_timeout + 1)
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
    async with open_stream(provider, monkeypatch) as (stream, collector, server, events, errors):
        first, tail = make_audio_segments(provider)
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

        push_audio(stream, provider, first)
        await asyncio.wait_for(wait_for_results(1), 1)
        push_audio(stream, provider, tail)
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
    if provider.expected_attempts_after_audio == 1:
        pytest.skip("provider does not replay consumed input")
    first = make_audio_segments(provider)[0]
    server = STTServer(provider, segments=(first,) * 100)
    server.automatic_finals = provider.progress_requires_final
    async with open_stream(provider, monkeypatch, server=server, max_retry=1) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):

        async def feed():
            while not collector.done():
                push_audio(stream, provider, first)
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
    async with open_stream(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        _,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(
            server.wait_for_audio(sum(len(s.audio) for s in make_audio_segments(provider))), 1
        )
        server.fault_after_audio = "abort"
        peer = server.connections[0]
        if code == 1006:
            peer.abort()
        else:
            # A close control frame must not be held behind the processing gate.
            await peer.ws.close(code=code, message=b"provider close details")
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert errors and not errors[-1].recoverable
        assert 1 <= len(server.connections) <= provider.expected_attempts_after_audio


async def test_queued_flushes_preserve_all_audio_and_results(provider, monkeypatch):
    async with open_stream(provider, monkeypatch) as (stream, collector, server, events, errors):
        first, tail = make_audio_segments(provider)
        push_audio(stream, provider, first)
        stream.flush()
        push_audio(stream, provider, tail)
        stream.end_input()
        await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert bytes(server.received_audio) == server.expected_audio
        assert not errors
        if not provider.completion_unknown_reason:
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
    [p for p in PROVIDERS if not p.completion_unknown_reason and not p.completion_on_close],
    ids=lambda p: p.name,
)
async def test_normal_close_cannot_replace_a_required_completion(provider, monkeypatch):
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.release_results.clear()
    async with open_stream(provider, monkeypatch, server=server, max_retry=2) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.wait_for_audio(len(server.expected_audio)), 1)
        await asyncio.wait_for(server.final_requested.wait(), 1)

        async def wait_for_progress():
            while not any(event.alternatives for event in events):
                if collector.done():
                    await collector
                    pytest.fail("the stream ended before provider progress arrived")
                await asyncio.sleep(0.01)

        # Audio alone does not establish a turn in protocols with optional idle flushes.
        await asyncio.wait_for(wait_for_progress(), 1)
        await server.connections[0].ws.close()
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert errors and not errors[-1].recoverable

        assert len(server.connections) == 1


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.completion_unknown_reason], ids=lambda p: p.name
)
async def test_best_effort_drain_accepts_normal_peer_close(provider, monkeypatch):
    async with open_stream(provider, monkeypatch) as (stream, collector, server, _, errors):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.wait_for_audio(len(server.expected_audio)), 1)
        await server.connections[0].finalize()
        server.connections[0].close()
        await asyncio.wait_for(collector, provider.final_response_timeout + 1)
        assert not errors


@pytest.mark.parametrize(
    "provider", [p for p in PROVIDERS if p.uses_websocket_monitor], ids=lambda p: p.name
)
async def test_fatal_close_does_not_retry(provider, monkeypatch):
    class RejectInput(provider.protocol):
        async def receive(self, message):
            await self.connection.ws.close(code=1008, message=b"request rejected")

    rejected = replace(provider, protocol=RejectInput)
    async with open_stream(rejected, monkeypatch, max_retry=2) as (
        stream,
        collector,
        server,
        _,
        errors,
    ):
        push_audio(stream, rejected, make_audio_segments(rejected)[0])
        with pytest.raises(APIError) as failure:
            await asyncio.wait_for(collector, 2)
        assert not failure.value.retryable
        assert errors and not errors[-1].recoverable
        assert len(server.connections) == 1
