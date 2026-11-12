"""NVIDIA Riva plugin: STT word timestamps and TTS failure propagation against a local Riva server."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures

import grpc
import pytest
from riva.client.proto import (
    riva_asr_pb2 as asr,
    riva_asr_pb2_grpc as asr_grpc,
    riva_tts_pb2 as tts,
    riva_tts_pb2_grpc as tts_grpc,
)

from livekit.agents import APIConnectionError, APIConnectOptions, stt
from livekit.plugins.nvidia import STT, TTS
from tests.fakes import audio_frame


class _Riva(asr_grpc.RivaSpeechRecognitionServicer, tts_grpc.RivaSpeechSynthesisServicer):
    def StreamingRecognize(  # noqa: N802
        self,
        requests: Iterator[asr.StreamingRecognizeRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[asr.StreamingRecognizeResponse]:
        for _ in requests:
            pass
        word = asr.WordInfo(word="Hello", start_time=1000, end_time=1500)
        alternative = asr.SpeechRecognitionAlternative(
            transcript="Hello", confidence=0.9, words=[word]
        )
        result = asr.StreamingRecognitionResult(alternatives=[alternative], is_final=True)
        yield asr.StreamingRecognizeResponse(results=[result])

    def SynthesizeOnline(  # noqa: N802
        self, request: tts.SynthesizeSpeechRequest, context: grpc.ServicerContext
    ) -> Iterator[tts.SynthesizeSpeechResponse]:
        context.abort(grpc.StatusCode.UNAVAILABLE, "Riva synthesis failed")


@pytest.fixture
def riva() -> Iterator[str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    # the Riva stubs mark every RPC abstract; the ones left out answer UNIMPLEMENTED
    riva = _Riva()  # type: ignore[abstract]
    asr_grpc.add_RivaSpeechRecognitionServicer_to_server(riva, server)
    tts_grpc.add_RivaSpeechSynthesisServicer_to_server(riva, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield f"127.0.0.1:{port}"
    server.stop(grace=None)


async def test_word_timestamps_use_the_transcript_seconds_and_offset(riva: str) -> None:
    """Riva's millisecond word times become seconds shifted by the stream's start offset."""
    stream = STT(use_ssl=False, server=riva).stream(language="en-US")
    stream.start_time_offset = 0.25

    stream.push_frame(audio_frame(0.1))
    stream.end_input()
    [final] = [ev async for ev in stream if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]

    [data] = final.alternatives
    offset = stream.start_time_offset
    assert offset >= 0.25
    assert (data.start_time, data.end_time) == pytest.approx((offset + 1.0, offset + 1.5))
    assert data.words is not None
    assert (data.words[0].start_time, data.words[0].end_time) == pytest.approx(
        (offset + 1.0, offset + 1.5)
    )


async def test_synthesis_failure_reaches_the_reader_with_its_sdk_cause(riva: str) -> None:
    """A Riva synthesis error surfaces as APIConnectionError caused by the gRPC error."""
    model = TTS(use_ssl=False, server=riva)

    with pytest.raises(APIConnectionError) as caught:
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("Hello there.")
            stream.end_input()
            async for _ in stream:
                pass

    assert isinstance(caught.value.__cause__, grpc.RpcError)
    assert caught.value.__cause__.details() == "Riva synthesis failed"
