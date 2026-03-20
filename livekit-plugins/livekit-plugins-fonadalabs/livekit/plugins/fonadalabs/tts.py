"""FonadaLabs TTS plugin for LiveKit - WebSocket implementation"""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import os
from dataclasses import dataclass

import aiohttp
from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    APIStatusError,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
    utils,
)

from .log import logger

FONADALABS_TTS_WS_URL = "wss://api.fonada.ai/tts/generate-audio-ws"
FONADALABS_SUPPORTED_VOICES_URL = "https://api.fonada.ai/supported-voices"


@dataclass
class _Catalog:
    """Everything we know about supported languages and voices, from the API."""
    # lang_code -> list of user-facing voice display names  e.g. {"hi": ["Vaanee", "Swara", ...]}
    voices: dict[str, list[str]]
    # lang_code -> display name                             e.g. {"hi": "Hindi"}
    code_to_name: dict[str, str]
    # lowercase display name -> lang_code                  e.g. {"hindi": "hi"}
    name_to_code: dict[str, str]

    def supported_languages(self) -> list[str]:
        """Return language display names for the languages that have TTS voices."""
        return [self.code_to_name[c] for c in self.voices if c in self.code_to_name]

    def resolve_language(self, language: str) -> tuple[str, str] | None:
        """
        Accept either a code ("hi") or display name ("Hindi").
        Returns (code, display_name) or None if not found.
        """
        # Try direct code match first
        if language in self.voices:
            return language, self.code_to_name.get(language, language)
        # Try case-insensitive display name match
        code = self.name_to_code.get(language.lower())
        if code and code in self.voices:
            return code, self.code_to_name.get(code, language)
        return None

    def default_voice(self, lang_code: str) -> str | None:
        voices = self.voices.get(lang_code, [])
        return voices[0] if voices else None


_catalog_cache: _Catalog | None = None

async def _load_catalog(session: aiohttp.ClientSession) -> _Catalog:
    """
    Fetch /supported-voices once and build the Catalog.
    All language and voice data comes from the API — nothing hardcoded.

    Response shape used:
      tts_languages.fonadalabs -> {lang_code: {voices: {internal_id: display_name}}}
      asr_languages.fonadalabs -> {lang_code: display_name}
    """
    global _catalog_cache

    if _catalog_cache is not None:
        return _catalog_cache

    try:
        async with session.get(
            FONADALABS_SUPPORTED_VOICES_URL,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # ── TTS voices (keyed by lang code) ──────────────────────────────
        tts_langs: dict = data.get("tts_languages", {}).get("fonadalabs", {})
        voices: dict[str, list[str]] = {}
        for code, info in tts_langs.items():
            voice_map = info.get("voices", {})
            # API format: {internal_id: display_name} — we want display names
            voices[code] = list(voice_map.values())

        # ── Language name mapping (from same response) ────────────────────
        # asr_languages.fonadalabs has {code: display_name} for ALL languages
        asr_langs: dict = data.get("asr_languages", {}).get("fonadalabs", {})
        code_to_name: dict[str, str] = {}
        name_to_code: dict[str, str] = {}
        for code, display_name in asr_langs.items():
            if code in voices:            # only care about TTS-supported languages
                code_to_name[code] = display_name
                name_to_code[display_name.lower()] = code

        _catalog_cache = _Catalog(
            voices=voices,
            code_to_name=code_to_name,
            name_to_code=name_to_code,
        )

        logger.info(
            f"[FonadaLabs] Catalog loaded — "
            f"languages: {_catalog_cache.supported_languages()} | "
            f"voices: { {c: len(v) for c, v in voices.items()} }"
        )

    except Exception as exc:
        logger.warning(
            f"[FonadaLabs] Could not load catalog from {FONADALABS_SUPPORTED_VOICES_URL}: {exc}. "
            "Language/voice validation will be skipped — server will validate instead."
        )
        _catalog_cache = _Catalog(voices={}, code_to_name={}, name_to_code={})

    return _catalog_cache


def _invalidate_catalog() -> None:
    """Force re-fetch on next TTS call (e.g. after an unsupported_voice error)."""
    global _catalog_cache
    _catalog_cache = None


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

class ConnectionState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING   = "connecting"
    CONNECTED    = "connected"
    FAILED       = "failed"


# ---------------------------------------------------------------------------
# TTS class
# ---------------------------------------------------------------------------

class TTS(tts.TTS):
    """FonadaLabs Text-to-Speech plugin for LiveKit.

    Languages and voices are resolved at runtime from:
        https://api.fonada.ai/supported-voices
    Nothing is hardcoded — the supported list always reflects the live API.

    Args:
        api_key:      FonadaLabs API key (or set FONADALABS_API_KEY env var).
        language:     Language code ("hi", "ta", "te", "en") **or** display name
                      ("Hindi", "Tamil", "Telugu", "English").
                      Defaults to "Hindi".
        voice:        Voice display name (e.g. "Vaanee", "Isai").
                      If omitted, the first available voice for the language is used.
        api_url:      Override the API base URL (optional).
        sample_rate:  Audio sample rate in Hz (default 24000).
        num_channels: Audio channels (default 1 / mono).
        http_session: Reuse an existing aiohttp.ClientSession (optional).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = "Hindi",
        voice: str | None = None,
        api_url: str | None = None,
        sample_rate: int = 24000,
        num_channels: int = 1,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.environ.get("FONADALABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FonadaLabs API key is required. "
                "Pass it directly or set the FONADALABS_API_KEY environment variable."
            )

        # Store as provided — validated at _run() time against the live catalog
        self._language_input = (language or "").strip()
        self._voice_input     = (voice or "").strip() or None

        # Resolved at _run() time
        self._lang_code:    str = ""
        self._lang_name:    str = ""
        self._voice:        str = ""

        # WebSocket URL
        if api_url:
            ws = api_url.replace("http://", "ws://").replace("https://", "wss://")
            self._ws_url = ws.rstrip("/") + (
                "" if ws.endswith("/tts/generate-audio-ws") else "/tts/generate-audio-ws"
            )
        else:
            self._ws_url = FONADALABS_TTS_WS_URL

        self._http_session = http_session
        self._own_session  = http_session is None
        if self._own_session:
            self._http_session = aiohttp.ClientSession()

    async def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> "SynthesizeStream":
        stream = self.stream(conn_options=conn_options)
        stream.push_text(text)
        stream.end_input()
        return stream

    def stream(
        self,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> "SynthesizeStream":
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )

    async def aclose(self) -> None:
        if self._own_session and self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await super().aclose()


# ---------------------------------------------------------------------------
# SynthesizeStream
# ---------------------------------------------------------------------------

class SynthesizeStream(tts.SynthesizeStream):

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts              = tts
        self._session_id       = f"fonadalabs_{id(self)}"
        self._connection_state = ConnectionState.DISCONNECTED
        self._ws_conn          = None
        self._send_task        = None
        self._recv_task        = None

    # ------------------------------------------------------------------
    # _run — called by the LiveKit framework when iteration starts
    # ------------------------------------------------------------------

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import uuid
        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        # ── Resolve language & voice from live catalog ─────────────────
        catalog = await _load_catalog(self._tts._http_session)
        await self._resolve_language_and_voice(catalog)
        # ──────────────────────────────────────────────────────────────

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                # Collect text from the input channel
                segments: list[str] = []
                async for chunk in self._input_ch:
                    if isinstance(chunk, str):
                        segments.append(chunk)
                    elif hasattr(chunk, "text"):
                        segments.append(chunk.text)
                    else:
                        logger.debug(f"Skipping non-text chunk: {type(chunk).__name__}")
                        continue

                if not segments:
                    raise ValueError("No text received from input channel.")

                text = " ".join(segments)
                payload = {
                    "api_key":  self._tts._api_key,
                    "text":     text,
                    "voice_id": self._tts._voice,
                    "language": self._tts._lang_name,   # display name, e.g. "Hindi"
                }
                await ws.send_str(json.dumps(payload))
                logger.debug(
                    "TTS request sent",
                    extra={**self._log_ctx(), "text_len": len(text),
                           "voice": self._tts._voice, "language": self._tts._lang_name},
                )
            except Exception as exc:
                logger.error(f"send_task error: {exc}", extra=self._log_ctx())
                raise

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if not await self._handle_text_msg(msg.data, output_emitter):
                            break
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        output_emitter.push(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise APIConnectionError(f"WebSocket error: {ws.exception()}")
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        output_emitter.end_input()
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"recv_task error: {exc}", extra=self._log_ctx())
                raise

        try:
            self._connection_state = ConnectionState.CONNECTING
            ws = await self._tts._http_session.ws_connect(
                self._tts._ws_url,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            )
            self._ws_conn          = ws
            self._connection_state = ConnectionState.CONNECTED
            logger.info("WebSocket connected", extra=self._log_ctx())

            self._send_task = asyncio.create_task(send_task(ws))
            self._recv_task = asyncio.create_task(recv_task(ws))
            try:
                await asyncio.gather(self._send_task, self._recv_task)
            except Exception:
                raise
            finally:
                await utils.aio.gracefully_cancel(self._send_task, self._recv_task)
                self._send_task = self._recv_task = None
                if not ws.closed:
                    await ws.close()

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
            self._connection_state = ConnectionState.FAILED
            raise APIConnectionError(f"Failed to connect: {exc}") from exc
        except (APIConnectionError, APIStatusError):
            self._connection_state = ConnectionState.FAILED
            raise
        except Exception as exc:
            self._connection_state = ConnectionState.FAILED
            raise APIStatusError(f"TTS session failed: {exc}") from exc
        finally:
            self._connection_state = ConnectionState.DISCONNECTED
            self._ws_conn          = None

    # ------------------------------------------------------------------
    # Resolve language + voice against the live catalog
    # ------------------------------------------------------------------

    async def _resolve_language_and_voice(self, catalog: _Catalog) -> None:
        lang_input = self._tts._language_input

        if catalog.voices:
            # Resolve language
            result = catalog.resolve_language(lang_input)
            if result is None:
                supported = catalog.supported_languages()
                raise APIStatusError(
                    message=(
                        f"Language '{lang_input}' is not supported. "
                        f"Supported languages: {', '.join(supported)}"
                    ),
                    status_code=400,
                )
            lang_code, lang_name = result

            # Resolve voice
            available = catalog.voices.get(lang_code, [])
            voice_input = self._tts._voice_input
            if voice_input:
                if voice_input not in available:
                    raise APIStatusError(
                        message=(
                            f"Voice '{voice_input}' is not available for {lang_name}. "
                            f"Available ({len(available)}): "
                            f"{', '.join(available[:8])}"
                            + ("..." if len(available) > 8 else "")
                        ),
                        status_code=400,
                    )
                voice = voice_input
            else:
                voice = catalog.default_voice(lang_code) or "Vaanee"

        else:
            # Catalog unavailable — accept whatever the user passed, server will validate
            lang_code = lang_input
            lang_name = lang_input
            voice     = self._tts._voice_input or "Vaanee"
            logger.warning(
                "Catalog unavailable — skipping client-side validation",
                extra=self._log_ctx(),
            )

        self._tts._lang_code = lang_code
        self._tts._lang_name = lang_name
        self._tts._voice     = voice

        logger.info(
            f"Resolved: language={lang_name!r} (code={lang_code!r}), voice={voice!r}",
            extra=self._log_ctx(),
        )

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_text_msg(
        self, raw: str, output_emitter: tts.AudioEmitter
    ) -> bool:
        """Return True to keep going, False to stop."""
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON WS message ignored: {raw[:120]}", extra=self._log_ctx())
            return True

        status   = resp.get("status")
        msg_type = resp.get("type")

        if status == "complete" or msg_type in ("complete", "final"):
            output_emitter.end_input()
            return False

        if (status == "error" or msg_type == "error"
                or resp.get("error") or resp.get("error_message")):
            await self._handle_error(resp)
            return False

        if msg_type == "audio":
            audio_b64 = resp.get("data", {}).get("audio", "")
            if audio_b64:
                output_emitter.push(base64.b64decode(audio_b64))
            return True

        if status in ("streaming",) or msg_type in ("streaming", "status"):
            return True

        logger.debug(f"Unhandled WS message: {resp}", extra=self._log_ctx())
        return True

    async def _handle_error(self, resp: dict) -> None:
        data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
        msg  = (
            data.get("message") or data.get("error") or
            resp.get("message") or resp.get("error") or
            resp.get("error_message") or resp.get("detail") or
            f"API error: {json.dumps(resp)}"
        )
        err_type = data.get("error") or data.get("type") or resp.get("error_type", "")

        logger.error(f"TTS API error [{err_type}]: {msg}", extra=self._log_ctx())

        if "credits" in msg.lower():
            raise APIStatusError(message=f"Credits exhausted: {msg}", status_code=402)
        if "rate limit" in msg.lower() or err_type == "rate_limit_exceeded":
            raise APIStatusError(message=f"Rate limit: {msg}", status_code=429)
        if "unsupported_voice" in msg.lower() or err_type == "unsupported_voice":
            _invalidate_catalog()
            raise APIStatusError(message=f"Unsupported voice: {msg}", status_code=400)
        if "invalid_language" in msg.lower() or err_type == "invalid_language":
            _invalidate_catalog()
            raise APIStatusError(message=f"Invalid language: {msg}", status_code=400)
        if "invalid api key" in msg.lower() or "authentication" in msg.lower():
            raise APIStatusError(message=f"Auth failed: {msg}", status_code=401)
        if any(w in msg.lower() for w in ("rate_limit", "timeout", "temporary")):
            raise APIConnectionError(f"Recoverable error: {msg}")

        raise APIStatusError(message=f"TTS error: {msg}", status_code=500)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_ctx(self) -> dict:
        return {
            "session_id":        self._session_id,
            "connection_state":  self._connection_state.value,
            "language":          getattr(self._tts, "_lang_name", self._tts._language_input),
            "voice":             getattr(self._tts, "_voice", self._tts._voice_input),
        }

    async def aclose(self) -> None:
        self._connection_state = ConnectionState.DISCONNECTED
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self._ws_conn and not self._ws_conn.closed:
            try:
                await self._ws_conn.close()
            except Exception:
                pass
        await super().aclose()
