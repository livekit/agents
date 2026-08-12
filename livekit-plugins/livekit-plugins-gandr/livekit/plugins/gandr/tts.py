"""Gandr TTS plugin for LiveKit Agents.

    from livekit.plugins import gandr

    session = AgentSession(
        stt=..., llm=...,
        tts=gandr.TTS(voice="gandr-mia"),   # key from GANDR_API_KEY
    )

Voices: gandr-mia, gandr-ava, gandr-jenny, gandr-dane, gandr-leo,
gandr-lewis, or a `gnd:` clone id. Swap mid-session with
`session.tts.update_options(voice=...)`.

Notes on behaviour that matter on a live call:

1. A request the API asks you to retry is retried once so the turn stays
   alive. `tts.prewarm()` is exposed if you would rather open the path on
   the SIP invite than on first speech.

2. A stream the API marks as truncated raises rather than playing a partial
   sentence, which hands the turn to the agent retry and to FallbackAdapter
   if one is configured. A visible retry beats a silent half sentence when
   the sentence carries a dose or an address.

3. Output is 24 kHz by default. Requesting 8000 is almost never what you want,
   even on a telephony call; let the transport resample.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio

DEFAULT_BASE = "https://tts.gandr.ai"

#: Rates the API renders at. 24000 is the engine's native rate.
SAMPLE_RATES = (8000, 16000, 22050, 24000)

#: Marker the API returns when a request should simply be retried.
_RETRY = "upstream_cold"


@dataclass
class _Opts:
    api_key: str
    voice: str
    lang: str
    base_url: str
    sample_rate: int
    timeout: float
    speed: float | None
    volume: float | None
    extra: dict | None


class GandrTTS(tts.TTS):
    """Gandr TTS for LiveKit Agents.

    Args:
        api_key:     `gnd_…` key. Defaults to the GANDR_API_KEY env var.
        voice:       stock id or a `gnd:` clone id. Default `gandr-mia`.
        lang:        language code for the input text. Default `en`.
        sample_rate: output rate in Hz, 8000, 16000, 22050 or 24000.

            LEAVE THIS AT 24000, INCLUDING ON TELEPHONY, and take the
            downsample locally. Asking the server for 8 kHz costs extra
            time to first audio on every single utterance, because the
            resample happens server-side before the first chunk is
            released, and LiveKit resamples into the SIP leg for free.
            If your pipeline genuinely needs a narrowband source, 16000
            is the cheapest honest choice.

        speed:       0.6 to 1.5, pitch preserving, applied after synthesis.
        volume:      0.5 to 2.0, soft-ceiling mastered, never clips.
        extra:       merged into every request body, `pronunciation_dict`,
                     `temperature`, `cfg_weight`, `seed`. Not
                     `expressiveness`: the engine remapped it off on
                     2026-07-29, and the door never fills it in, so it is
                     accepted and inert either way. Omit `temperature` and
                     the door picks one for you from the voice (stock
                     jenny/ava/mia/lewis 0.5, dane 0.65, leo 0.8, the
                     floor for a voice the door's map does not name; what
                     a clone inherits was not part of that read, so pass
                     the field yourself on a cloned voice); omit
                     `cfg_weight` and nothing is sent at all. Pass both if
                     the delivery matters, note `voice` defaults to
                     gandr-mia, which the door reads at 0.5, not at the
                     floor.
        base_url:    leave as-is; requests are routed for you.
        timeout:     socket read timeout on the audio stream, seconds.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: str = "gandr-mia",
        lang: str = "en",
        base_url: str = DEFAULT_BASE,
        sample_rate: int = 24000,
        timeout: float = 30.0,
        speed: float | None = None,
        volume: float | None = None,
        extra: dict | None = None,
        http_session: aiohttp.ClientSession | None = None,
        prewarm_on_start: bool = True,
    ):
        """`prewarm_on_start` opens the path as soon as the plugin is
        constructed. Leave it on unless you are building many instances
        you do not intend to speak through."""
        super().__init__(
            # streaming=False is correct: the HTTP lane renders one
            # utterance at a time. livekit-agents wraps this in its own
            # stream adapter and the result is gap-free.
            #
            # aligned_transcript stays False on purpose even though the
            # API does return word timestamps. They arrive in the FINAL
            # event, after the last audio chunk, LiveKit wants them
            # interleaved with audio, so claiming support here would
            # promise a sync we cannot deliver. Ask for them explicitly
            # with extra={"add_timestamps": "word"} if you want the data.
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        key = api_key or os.environ.get("GANDR_API_KEY")
        if not key:
            raise ValueError(
                "No Gandr API key. Pass api_key=… or set GANDR_API_KEY. "
                "Keys: https://gandr.ai/waitlist/"
            )
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")
        self._opts = _Opts(key, voice, lang, base_url, sample_rate, timeout, speed, volume, extra)
        self._session = http_session
        self._prewarm_task: asyncio.Task | None = None
        if prewarm_on_start:
            self._prewarm_soon()

    @property
    def provider(self) -> str:
        return "Gandr"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    # ── opening the path ────────────────────────────────────────────
    def prewarm(self) -> None:
        """Open the path in the background. Returns as soon as the API
        has accepted the request.

        Sync on purpose: livekit-agents calls this fire-and-forget from
        synchronisation hooks, so scheduling, not awaiting, is the
        contract. Call it on the SIP invite, or whenever you know a call
        is about to start, and first audio is at full speed by the time
        your LLM has a sentence to say."""
        self._prewarm_soon()

    async def _prewarm_async(self) -> None:
        """The awaitable form of `prewarm`, kept separate so the sync
        public method never returns a coroutine to a fire-and-forget
        caller."""
        try:
            async with self._ensure_session().get(
                self._opts.base_url + "/v1/prewarm",
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.read()
        except Exception:
            # Prewarming is an optimisation. It must never be the reason
            # a call fails, so every failure here is swallowed,
            # synthesize() retries on its own.
            pass

    def _prewarm_soon(self) -> None:
        """Fire prewarm without blocking, from sync code, whether or not
        a loop is already running."""
        if self._prewarm_task is not None and not self._prewarm_task.done():
            return  # a prewarm is already in flight
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # constructed before the loop; synthesize() covers it
        self._prewarm_task = loop.create_task(self._prewarm_async())

    async def aclose(self) -> None:
        """Shut down the plugin, cancelling any in-flight prewarm so no
        task outlives the object."""
        if self._prewarm_task is not None:
            await aio.cancel_and_wait(self._prewarm_task)
            self._prewarm_task = None

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice: str | None = None,
        lang: str | None = None,
        speed: float | None = None,
        volume: float | None = None,
    ) -> None:
        """Change delivery mid-session. The next utterance picks it up."""
        if voice is not None:
            self._opts.voice = voice
        if lang is not None:
            self._opts.lang = lang
        if speed is not None:
            self._opts.speed = speed
        if volume is not None:
            self._opts.volume = volume


class ChunkedStream(tts.ChunkedStream):
    """One utterance, streamed from Gandr's SSE lane as base64 PCM."""

    def __init__(self, *, tts: GandrTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._o = tts._opts
        self._tts: GandrTTS = tts

    def _body(self) -> dict:
        o = self._o
        body: dict = {
            "text": self._input_text,
            "lang": o.lang,
            "voice": {"mode": "id", "id": o.voice},
            "output_format": {"sample_rate": o.sample_rate},
        }
        if o.speed is not None:
            body["speed"] = o.speed
        if o.volume is not None:
            body["volume"] = o.volume
        if o.extra:
            body.update(o.extra)
        return body

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            await self._attempt(output_emitter)
        except _Retryable:
            # The API asked for a retry. Take one more run at it rather than
            # handing the caller silence.
            await self._tts._prewarm_async()
            await asyncio.sleep(1.0)
            try:
                await self._attempt(output_emitter)
            except _Retryable:
                # Still asking for a retry. Convert the internal marker to a
                # retryable API error so livekit-agents owns the retry
                # instead of letting an internal exception escape.
                raise APIStatusError(
                    message="Gandr asked to retry again",
                    status_code=503,
                    request_id=None,
                    body=None,
                ) from None

    async def _attempt(self, output_emitter: tts.AudioEmitter) -> None:
        o = self._o
        try:
            async with self._tts._ensure_session().post(
                o.base_url + "/v1/tts/sse",
                headers={
                    "x-api-key": o.api_key,
                    "content-type": "application/json",
                },
                json=self._body(),
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=o.timeout,
                ),
            ) as resp:
                if resp.status >= 400:
                    detail = (await resp.text())[:300]
                    if resp.status in (502, 503) and _RETRY in detail:
                        raise _Retryable()
                    raise APIStatusError(
                        message=f"Gandr {resp.status}: {detail}",
                        status_code=resp.status,
                        request_id=None,
                        body=detail,
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=o.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                truncated = False
                done_seen = False
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    # The SSE spec allows `data:{…}` with no space. Our
                    # API sends one, but a proxy is entitled to restripe
                    # this, and a dropped chunk is a dropped syllable.
                    payload = line[5:].lstrip()
                    if not payload:
                        continue
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("done"):
                        done_seen = True
                        truncated = bool(evt.get("truncated"))
                        break
                    data = evt.get("data")
                    if data:
                        output_emitter.push(base64.b64decode(data))

                # Raise BEFORE flushing: a truncated render must not be
                # played as if it were the whole sentence. This turns a
                # silent half-utterance into a retry livekit-agents can
                # see, and into a provider switch if you have wrapped
                # this in a FallbackAdapter.
                # A stream that ends without a `done` event is the same
                # failure: the connection closed cleanly but the render
                # never finished, so what we hold is a partial utterance,
                # not the sentence.
                if truncated or not done_seen:
                    raise APIStatusError(
                        message="Gandr stream ended early (truncated); retrying",
                        status_code=503,
                        request_id=None,
                        body=None,
                    )

                output_emitter.flush()

        except (_Retryable, APIStatusError):
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=str(e.message), status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # Keep the cause visible. The previous revision raised a bare
            # APIConnectionError(), which made a DNS failure and a bad
            # payload look identical in the logs.
            raise APIConnectionError(f"Gandr request failed: {e}") from e


class _Retryable(Exception):
    """Internal: the API asked for a retry. Not surfaced to callers."""


# livekit-plugins convention: the service class is exported as TTS
TTS = GandrTTS
