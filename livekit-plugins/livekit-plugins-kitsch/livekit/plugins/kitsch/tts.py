from __future__ import annotations

import base64
import binascii
from dataclasses import replace
from urllib.parse import quote
from uuid import uuid4

from livekit.agents import APIConnectOptions, tts

from ._client import DEFAULT_OPTIONS, SAMPLE_RATE, Client, duplex, error, receive


class TTS(tts.TTS):
    """Kitsch TTS. Each stream represents one flush-delimited utterance.

    voice_id must be accessible with the supplied Kitsch workspace API key.
    Output is mono PCM16 at 24 kHz.
    """

    def __init__(
        self,
        *,
        voice_id: str,
        api_key: str | None = None,
        language_code: str | None = None,
        base_url: str = "https://api.kitschlabs.com",
    ) -> None:
        if not voice_id.strip():
            raise ValueError("voice_id is required.")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._client = Client(api_key, base_url)
        self._path = f"/v1/text-to-speech/{quote(voice_id, safe='')}"
        self._language = language_code

    @property
    def model(self) -> str:
        return "kitsch-voice"

    @property
    def provider(self) -> str:
        return "Kitsch"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize complete text through the native HTTP audio stream."""
        if not text.strip() or len(text) > 5000:
            raise ValueError("text must contain 1–5000 characters.")
        return ChunkedStream(
            tts=self, input_text=text, conn_options=replace(conn_options, max_retry=0)
        )

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_OPTIONS) -> SynthesizeStream:
        """Accept incremental text; flush/end_input completes this utterance."""
        return SynthesizeStream(self, replace(conn_options, max_retry=0))

    async def aclose(self) -> None:
        await self._client.close()


class ChunkedStream(tts.ChunkedStream):
    _tts: TTS

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body = {"text": self.input_text}
        if self._tts._language:
            body["language_code"] = self._tts._language
        async with self._tts._client.post(
            self._tts._path + "/stream",
            self._conn_options,
            params={"output_format": "pcm_24000"},
            json=body,
        ) as response:
            if response.content_type not in {"audio/pcm", "application/octet-stream"}:
                raise error("Kitsch returned an unexpected audio format.")
            output_emitter.initialize(
                request_id=response.headers.get("x-request-id", uuid4().hex),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
                frame_size_ms=20,
            )
            length = 0
            async for chunk in response.content.iter_chunked(4800):
                length += len(chunk)
                output_emitter.push(chunk)
            if not length or length % 2:
                raise error("Kitsch returned empty or incomplete PCM audio.")


class SynthesizeStream(tts.SynthesizeStream):
    _tts: TTS

    def __init__(self, tts_instance: TTS, options: APIConnectOptions) -> None:
        self._characters = 0
        self._flushed = False
        super().__init__(tts=tts_instance, conn_options=options)

    def push_text(self, token: str) -> None:
        if self._flushed and token:
            raise ValueError("Create a new TTS stream after flush.")
        if self._characters + len(token) > 5000:
            raise ValueError("A TTS utterance cannot exceed 5000 characters.")
        self._characters += len(token)
        super().push_text(token)

    def flush(self) -> None:
        self._flushed = True
        super().flush()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=uuid4().hex,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
            stream=True,
        )
        first = None
        async for part in self._input_ch:
            if isinstance(part, str):
                first = part
                break
        if first is None:
            return
        params = {"output_format": "pcm_24000"}
        if self._tts._language:
            params["language_code"] = self._tts._language
        async with self._tts._client.websocket(
            self._tts._path + "/stream-input", self._conn_options, params=params
        ) as ws:
            output_emitter.start_segment(segment_id=uuid4().hex)
            self._mark_started()

            async def send() -> None:
                await ws.send_json({"text": first})
                async for part in self._input_ch:
                    if isinstance(part, str):
                        await ws.send_json({"text": part})
                    else:
                        # Empty text closes the socket; a flush has no text field.
                        await ws.send_json({"flush": True})
                        return

            async def read() -> None:
                length = 0
                while True:
                    data = await receive(ws, self._conn_options.timeout)
                    if "error" in data:
                        raise error("Kitsch synthesis failed.")
                    if data.get("audio"):
                        try:
                            chunk = base64.b64decode(data["audio"], validate=True)
                        except (ValueError, TypeError, binascii.Error):
                            raise error("Invalid Kitsch audio response.") from None
                        length += len(chunk)
                        output_emitter.push(chunk)
                    if data.get("isFinal") is True:
                        if not length or length % 2:
                            raise error("Kitsch returned empty or incomplete PCM audio.")
                        output_emitter.end_segment()
                        return

            await duplex(send(), read())
