"""The shipped connection options reach the network in every provider.

Each request path runs against a transport that records the attempt and fails it, so a
provider that refuses its own defaults (or the all-disabled policy) fails before any I/O.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import socket
from collections.abc import Callable
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    llm,
    stt,
    tts,
    utils,
)

pytestmark = pytest.mark.unit

POLICIES = {
    "defaults": dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0),
    "disabled": APIConnectOptions(
        max_retry=0,
        pool_timeout=None,
        connect_timeout=None,
        write_timeout=None,
        read_timeout=None,
        total_timeout=None,
    ),
}

_CREDENTIALS = {
    **dict.fromkeys(
        """ANAM_API_KEY ANTHROPIC_API_KEY ASSEMBLYAI_API_KEY ASYNCAI_API_KEY AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY AZURE_SPEECH_KEY BASETEN_API_KEY BEY_API_KEY BLAND_API_KEY
        CAMB_API_KEY CARTESIA_API_KEY CEREBRAS_API_KEY CLOVA_STT_SECRET_KEY DEEPGRAM_API_KEY
        DID_API_KEY ELEVEN_API_KEY FAL_KEY FIREWORKS_API_KEY FISH_API_KEY GLADIA_API_KEY
        GNANI_API_KEY GOOGLE_API_KEY GRADIUM_API_KEY GROQ_API_KEY HUME_API_KEY INWORLD_API_KEY
        LMNT_API_KEY META_API_KEY MINIMAX_API_KEY MISTRAL_API_KEY MURF_API_KEY NABRAH_API_KEY
        NEUPHONIC_API_KEY NVIDIA_API_KEY OPENAI_API_KEY PALABRA_API_KEY PERPLEXITY_API_KEY
        RESEMBLE_API_KEY RESPEECHER_API_KEY RIME_API_KEY RTZR_CLIENT_ID RTZR_CLIENT_SECRET
        SARVAM_API_KEY SIMPLISMART_API_KEY SLNG_API_KEY SMALLEST_API_KEY SONIOX_API_KEY
        SPEECHIFY_API_KEY SPEECHMATICS_API_KEY SPITCH_API_KEY TELNYX_API_KEY UPLIFTAI_API_KEY
        VAKYAM_API_KEY XAI_API_KEY""".split(),
        "test",
    ),
    "AWS_REGION": "us-east-1",
    "AZURE_SPEECH_REGION": "eastus",
    "CLOVA_STT_INVOKE_URL": "https://clova.invalid",
}


def _model(path: str, **kwargs: Any) -> Callable[[], Any]:
    module, name = path.rsplit(".", 1)
    return lambda: getattr(importlib.import_module(f"livekit.plugins.{module}"), name)(**kwargs)


MODELS = {
    "anthropic.LLM": _model("anthropic.LLM"),
    "assemblyai.STT": _model("assemblyai.STT"),
    "asyncai.TTS": _model("asyncai.TTS"),
    "aws.LLM": _model("aws.LLM"),
    "aws.STT": _model("aws.STT"),
    "aws.TTS": _model("aws.TTS"),
    "azure.STT": _model("azure.STT"),
    "azure.TTS": _model("azure.TTS"),
    "baseten.TTS": _model("baseten.TTS", model_endpoint="https://baseten.invalid/predict"),
    "bland.TTS": _model("bland.TTS"),
    "cambai.TTS": _model("cambai.TTS"),
    "cartesia.STT": _model("cartesia.STT"),
    "cartesia.TTS": _model("cartesia.TTS"),
    "clova.STT": _model("clova.STT"),
    "deepgram.STT": _model("deepgram.STT"),
    "deepgram.STTv2": _model("deepgram.STTv2"),
    "deepgram.TTS": _model("deepgram.TTS"),
    "deepgram.TTSv2": _model("deepgram.TTSv2"),
    "elevenlabs.STT": _model("elevenlabs.STT"),
    "elevenlabs.TTS": _model("elevenlabs.TTS"),
    "fal.WizperSTT": _model("fal.WizperSTT"),
    "fishaudio.TTS": _model("fishaudio.TTS"),
    "gladia.STT": _model("gladia.STT"),
    "gnani.STT": _model("gnani.STT"),
    "gnani.TTS": _model("gnani.TTS"),
    "google.LLM": _model("google.LLM"),
    "gradium.STT": _model("gradium.STT"),
    "gradium.TTS": _model("gradium.TTS"),
    "groq.TTS": _model("groq.TTS"),
    "hume.TTS": _model("hume.TTS"),
    "inworld.STT": _model("inworld.STT"),
    "inworld.TTS": _model("inworld.TTS"),
    "lmnt.TTS": _model("lmnt.TTS"),
    "meta.STT": _model("meta.STT"),
    "minimax.TTS": _model("minimax.TTS"),
    "mistralai.LLM": _model("mistralai.LLM"),
    "mistralai.STT": _model("mistralai.STT"),
    "mistralai.TTS": _model("mistralai.TTS"),
    "murf.TTS": _model("murf.TTS"),
    "nabrah.STT": _model("nabrah.STT"),
    "neuphonic.TTS": _model("neuphonic.TTS"),
    "openai.LLM": _model("openai.LLM"),
    "openai.STT": _model("openai.STT"),
    "openai.TTS": _model("openai.TTS"),
    "palabra.STT": _model("palabra.STT"),
    "palabra.TTS": _model("palabra.TTS"),
    "resemble.TTS": _model("resemble.TTS"),
    "respeecher.TTS": _model("respeecher.TTS"),
    "rime.TTS": _model("rime.TTS"),
    "rtzr.STT": _model("rtzr.STT"),
    "sarvam.STT": _model("sarvam.STT"),
    "sarvam.TTS": _model("sarvam.TTS"),
    "simplismart.STT": _model("simplismart.STT"),
    "simplismart.TTS": _model("simplismart.TTS"),
    "smallestai.STT": _model("smallestai.STT"),
    "smallestai.TTS": _model("smallestai.TTS"),
    "soniox.STT": _model("soniox.STT"),
    "speechify.TTS": _model("speechify.TTS"),
    "speechmatics.STT": _model("speechmatics.STT"),
    "speechmatics.TTS": _model("speechmatics.TTS"),
    "spitch.STT": _model("spitch.STT"),
    "spitch.TTS": _model("spitch.TTS"),
    "telnyx.STT": _model("telnyx.STT"),
    "telnyx.TTS": _model("telnyx.TTS"),
    "vakyam.TTS": _model("vakyam.TTS"),
    "xai.STT": _model("xai.STT"),
}

STREAMING_ONLY = {
    "assemblyai.STT",
    "asyncai.TTS",
    "aws.STT",
    "azure.STT",
    "cartesia.STT",
    "deepgram.STTv2",
    "gradium.STT",
    "inworld.STT",
    "meta.STT",
    "nabrah.STT",
    "palabra.STT",
    "rtzr.STT",
    "soniox.STT",
    "speechmatics.STT",
}

# SDKs whose sockets are opened in native code: their client is the network boundary
NATIVE_CLIENTS = [
    ("livekit.plugins.aws.stt", "TranscribeStreamingClient"),
    ("livekit.plugins.azure.stt", "_create_speech_recognizer"),
]

# one SDK limit applies to every phase, and no request can disable it
SINGLE_LIMIT_SDKS = {"mistralai.LLM", "mistralai.STT", "mistralai.TTS", "speechify.TTS"}


class _Transport:
    """Counts each attempt to reach the network and fails it as an unreachable peer."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attempts = 0

        def getaddrinfo(*args: Any, **kwargs: Any) -> Any:
            self.attempts += 1
            raise socket.gaierror(socket.EAI_NONAME, "unreachable test peer")

        async def create_connection(*args: Any, **kwargs: Any) -> Any:
            self.attempts += 1
            raise ConnectionRefusedError("unreachable test peer")

        def native_client(*args: Any, **kwargs: Any) -> Any:
            self.attempts += 1
            raise ConnectionRefusedError("unreachable test peer")

        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", create_connection)
        for module, name in NATIVE_CLIENTS:
            monkeypatch.setattr(importlib.import_module(module), name, native_client)


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes(320 * 2), sample_rate=16000, num_channels=1, samples_per_channel=320
    )


async def _until_attempt(transport: _Transport, run: Callable[[], Any]) -> None:
    before = transport.attempts
    task = asyncio.ensure_future(run())
    try:
        while transport.attempts == before and not task.done():
            await asyncio.sleep(0.01)
        if transport.attempts == before:
            await task  # surfaces the refusal
    finally:
        await utils.aio.cancel_and_wait(task)
    assert transport.attempts > before, "the request never reached the transport"


def _operations(name: str, model: Any, options: APIConnectOptions) -> list[Callable[[], Any]]:
    async def consume(stream: Any) -> None:
        async with stream:
            async for _ in stream:
                pass

    async def tts_stream() -> None:
        stream = model.stream(conn_options=options)
        stream.push_text("hello")
        stream.end_input()
        await consume(stream)

    async def stt_stream() -> None:
        stream = model.stream(conn_options=options)
        for _ in range(25):
            stream.push_frame(_frame())
        stream.flush()
        await consume(stream)

    ops: list[Callable[[], Any]] = []
    if isinstance(model, tts.TTS):
        if name not in STREAMING_ONLY:
            ops.append(lambda: consume(model.synthesize("hello", conn_options=options)))
        if model.capabilities.streaming:
            ops.append(tts_stream)
        return ops
    if isinstance(model, stt.STT):
        if name not in STREAMING_ONLY:
            ops.append(lambda: model.recognize([_frame()], conn_options=options))
        if model.capabilities.streaming:
            ops.append(stt_stream)
        return ops
    assert isinstance(model, llm.LLM)
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    return [lambda: consume(model.chat(chat_ctx=chat_ctx, conn_options=options))]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", MODELS)
async def test_provider_request_reaches_the_network_under_shipped_policies(
    name: str, policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in _CREDENTIALS.items():
        monkeypatch.setenv(key, value)
    transport = _Transport(monkeypatch)
    async with utils.http_context.open():
        model = MODELS[name]()
        try:
            for operation in _operations(name, model, POLICIES[policy]):
                if policy == "disabled" and name in SINGLE_LIMIT_SDKS:
                    with pytest.raises(APIError, match="must be equal") as error:
                        await asyncio.wait_for(_until_attempt(transport, operation), 10)
                    assert not error.value.retryable and transport.attempts == 0
                else:
                    await asyncio.wait_for(_until_attempt(transport, operation), 10)
        finally:
            await model.aclose()


class _UnreachableSTT(stt.STT):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.attempts = 0

    async def _recognize_impl(self, buffer: Any, *, language: Any, conn_options: Any) -> Any:
        self.attempts += 1
        raise APIConnectionError("unreachable test peer")


async def test_total_timeout_spans_every_retry_of_an_operation() -> None:
    model = _UnreachableSTT()
    options = APIConnectOptions(max_retry=100, retry_interval=0.05, total_timeout=0.3)
    with pytest.raises(APITimeoutError) as error:
        await model.recognize([_frame()], conn_options=options)
    assert error.value.phase == "total"
    assert 1 < model.attempts < options.max_retry


@pytest.mark.parametrize("policy", POLICIES)
async def test_native_realtime_model_accepts_shipped_policies(
    policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.plugins.openai import realtime

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    model = realtime.RealtimeModel(conn_options=POLICIES[policy])
    await model.aclose()
