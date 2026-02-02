# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
import queue
import time
import wave
import weakref
from dataclasses import dataclass, replace
from typing import Literal

import aiohttp

try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-untyped]
except ImportError as err:
    raise ImportError(
        "azure-cognitiveservices-speech is required for streaming. "
        "Install with: pip install azure-cognitiveservices-speech"
    ) from err

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger

SUPPORTED_OUTPUT_FORMATS = {
    8000: "raw-8khz-16bit-mono-pcm",
    16000: "raw-16khz-16bit-mono-pcm",
    22050: "raw-22050hz-16bit-mono-pcm",
    24000: "raw-24khz-16bit-mono-pcm",
    44100: "raw-44100hz-16bit-mono-pcm",
    48000: "raw-48khz-16bit-mono-pcm",
}

# Azure SDK output format mapping - use Raw formats to avoid WAV headers
SDK_OUTPUT_FORMATS = {
    8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
    16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
    22050: speechsdk.SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm,
    24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
    44100: speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
    48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
}


@dataclass
class ProsodyConfig:
    rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] | float | None = None
    volume: Literal["silent", "x-soft", "soft", "medium", "loud", "x-loud"] | float | None = None
    pitch: Literal["x-low", "low", "medium", "high", "x-high"] | None = None

    def validate(self) -> None:
        if self.rate:
            if isinstance(self.rate, float) and not 0.5 <= self.rate <= 2:
                raise ValueError("Prosody rate must be between 0.5 and 2")
            if isinstance(self.rate, str) and self.rate not in [
                "x-slow",
                "slow",
                "medium",
                "fast",
                "x-fast",
            ]:
                raise ValueError(
                    "Prosody rate must be one of 'x-slow', 'slow', 'medium', 'fast', 'x-fast'"
                )
        if self.volume:
            if isinstance(self.volume, float) and not 0 <= self.volume <= 100:
                raise ValueError("Prosody volume must be between 0 and 100")
            if isinstance(self.volume, str) and self.volume not in [
                "silent",
                "x-soft",
                "soft",
                "medium",
                "loud",
                "x-loud",
            ]:
                raise ValueError(
                    "Prosody volume must be one of 'silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud'"  # noqa: E501
                )
        if self.pitch and self.pitch not in [
            "x-low",
            "low",
            "medium",
            "high",
            "x-high",
        ]:
            raise ValueError(
                "Prosody pitch must be one of 'x-low', 'low', 'medium', 'high', 'x-high'"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class StyleConfig:
    style: str
    degree: float | None = None

    def validate(self) -> None:
        if self.degree is not None and not 0.1 <= self.degree <= 2.0:
            raise ValueError("Style degree must be between 0.1 and 2.0")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class _TTSOptions:
    sample_rate: int
    subscription_key: str | None
    region: str | None
    voice: str
    language: str | None
    speech_endpoint: str | None
    deployment_id: str | None
    prosody: NotGivenOr[ProsodyConfig]
    style: NotGivenOr[StyleConfig]
    lexicon_uri: NotGivenOr[str]
    auth_token: str | None = None

    def get_endpoint_url(self) -> str:
        base = (
            self.speech_endpoint
            or f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )
        if self.deployment_id:
            return f"{base}?deploymentId={self.deployment_id}"
        return base


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "en-US-JennyNeural",
        language: str | None = None,
        sample_rate: int = 24000,
        prosody: NotGivenOr[ProsodyConfig] = NOT_GIVEN,
        style: NotGivenOr[StyleConfig] = NOT_GIVEN,
        lexicon_uri: NotGivenOr[str] = NOT_GIVEN,
        speech_key: str | None = None,
        speech_region: str | None = None,
        speech_endpoint: str | None = None,
        deployment_id: str | None = None,
        speech_auth_token: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        num_prewarm: int = 3,
    ) -> None:
        """
        Create a new instance of Azure TTS.

        Args:
            voice: Voice name (e.g., "en-US-JennyNeural")
            language: Language code (e.g., "en-US")
            sample_rate: Audio sample rate in Hz
            prosody: Prosody configuration for rate, volume, pitch
            style: Style configuration for expression
            speech_key: Azure Speech API key (or set AZURE_SPEECH_KEY env var)
            speech_region: Azure region (or set AZURE_SPEECH_REGION env var)
            speech_endpoint: Custom endpoint URL
            deployment_id: Custom deployment ID
            speech_auth_token: Authentication token
            http_session: Optional aiohttp session
            num_prewarm: Number of synthesizers to prewarm on initialization (default: 3)
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if sample_rate not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. Supported: {list(SUPPORTED_OUTPUT_FORMATS)}"  # noqa: E501
            )

        if not speech_key:
            speech_key = os.environ.get("AZURE_SPEECH_KEY")

        if not speech_region:
            speech_region = os.environ.get("AZURE_SPEECH_REGION")

        if not speech_endpoint:
            speech_endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")

        has_endpoint = bool(speech_endpoint)
        has_key_and_region = bool(speech_key and speech_region)
        has_token_and_region = bool(speech_auth_token and speech_region)
        if not (has_endpoint or has_key_and_region or has_token_and_region):
            raise ValueError(
                "Authentication requires one of: speech_endpoint (AZURE_SPEECH_ENDPOINT), "
                "speech_key & speech_region (AZURE_SPEECH_KEY & AZURE_SPEECH_REGION), "
                "or speech_auth_token & speech_region."
            )

        if is_given(prosody):
            prosody.validate()
        if is_given(style):
            style.validate()

        self._session = http_session
        self._opts = _TTSOptions(
            sample_rate=sample_rate,
            subscription_key=speech_key,
            region=speech_region,
            speech_endpoint=speech_endpoint,
            voice=voice,
            deployment_id=deployment_id,
            language=language,
            prosody=prosody,
            style=style,
            lexicon_uri=lexicon_uri,
            auth_token=speech_auth_token,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._target_pool_size = num_prewarm  # Track target pool size for replacements
        self._prewarm_task: asyncio.Task[None] | None = None  # Track prewarm task for cleanup
        self._pool = utils.ConnectionPool[speechsdk.SpeechSynthesizer](
            connect_cb=self._create_and_warmup_synthesizer,
            close_cb=self._close_synthesizer,
            max_session_duration=300,  # 5 minutes (jitter applied per-connection during prewarm)
            mark_refreshed_on_get=True,
        )

        # Prewarm a small number of synthesizers for fast first synthesis
        # Pool will grow on-demand like Cartesia (no background maintenance)
        if num_prewarm > 0:
            self.prewarm(count=num_prewarm)

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Azure TTS"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prosody: NotGivenOr[ProsodyConfig] = NOT_GIVEN,
        style: NotGivenOr[StyleConfig] = NOT_GIVEN,
        lexicon_uri: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(prosody):
            prosody.validate()
            self._opts.prosody = prosody
        if is_given(style):
            style.validate()
            self._opts.style = style
        if is_given(lexicon_uri):
            self._opts.lexicon_uri = lexicon_uri

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _create_and_warmup_synthesizer(self, timeout: float) -> speechsdk.SpeechSynthesizer:
        """Create and warm up a new Azure Speech synthesizer.

        This runs in a thread pool since Azure SDK is synchronous.
        Uses Connection.from_speech_synthesizer() and connection.open() to pre-connect.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            Warmed-up synthesizer ready for use
        """

        def _sync_create_and_warmup() -> speechsdk.SpeechSynthesizer:
            import threading
            import time

            # Build WebSocket v2 endpoint
            if self._opts.speech_endpoint:
                endpoint = self._opts.speech_endpoint.replace(
                    "/cognitiveservices/v1", "/cognitiveservices/websocket/v2"
                )
            else:
                endpoint = f"wss://{self._opts.region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"

            # Create speech config
            speech_config = speechsdk.SpeechConfig(
                endpoint=endpoint,
                subscription=self._opts.subscription_key or "",
            )

            # Set deployment ID if provided
            if self._opts.deployment_id:
                speech_config.endpoint_id = self._opts.deployment_id

            # Set voice and output format
            speech_config.speech_synthesis_voice_name = self._opts.voice

            # Use SDK format if available
            if self._opts.sample_rate in SDK_OUTPUT_FORMATS:
                speech_config.set_speech_synthesis_output_format(
                    SDK_OUTPUT_FORMATS[self._opts.sample_rate]
                )
            else:
                # Default to 24kHz raw format
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
                )

            # Create synthesizer (no audio config - we'll use events)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=None
            )

            # Pre-connect using Connection.from_speech_synthesizer() and open()
            logger.info("pre-connecting azure tts synthesizer")
            warmup_start = time.time()

            # Get connection from synthesizer
            connection = speechsdk.Connection.from_speech_synthesizer(synthesizer)

            # Use threading event to wait for connection
            connected_event = threading.Event()

            def on_connected(evt: speechsdk.ConnectionEventArgs) -> None:
                logger.debug("synthesizer connected callback received")
                connected_event.set()

            def on_disconnected(evt: speechsdk.ConnectionEventArgs) -> None:
                logger.debug("synthesizer disconnected callback received")

            # Subscribe to connection events
            connection.connected.connect(on_connected)
            connection.disconnected.connect(on_disconnected)

            # Open connection (for_continuous_recognition has no effect for synthesizer)
            connection.open(for_continuous_recognition=True)

            # Wait for connection to be established
            if not connected_event.wait(timeout=timeout):
                raise APIConnectionError("Azure TTS pre-connect timed out waiting for connection")

            # Disconnect event handlers after successful connection
            connection.connected.disconnect_all()
            connection.disconnected.disconnect_all()

            warmup_time = time.time() - warmup_start
            logger.info(
                "azure tts pre-connect completed", extra={"duration": f"{warmup_time:.3f}s"}
            )

            return synthesizer

        # Run in thread pool (Azure SDK is synchronous)
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _sync_create_and_warmup), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Azure TTS synthesizer creation timed out") from None
        except Exception as e:
            raise APIConnectionError(f"Failed to create synthesizer: {e}") from e

    async def _close_synthesizer(self, synthesizer: speechsdk.SpeechSynthesizer) -> None:
        """Close and clean up an Azure Speech synthesizer.

        Non-blocking cleanup that disconnects event handlers and lets
        the SDK destructor handle WebSocket closure.

        Args:
            synthesizer: The synthesizer to close
        """
        try:
            # Disconnect event handlers to prevent callbacks on next use
            synthesizer.synthesizing.disconnect_all()
            synthesizer.synthesis_completed.disconnect_all()
            synthesizer.synthesis_canceled.disconnect_all()
        except Exception as e:
            logger.debug(f"error disconnecting handlers during close: {e}")

        # Delete synthesizer - SDK destructor handles WebSocket cleanup
        del synthesizer
        logger.debug("synthesizer closed (non-blocking)")

    def prewarm(self, count: int = 1) -> None:
        """Prewarm the connection pool with multiple synthesizers.

        Pool grows on-demand after prewarming (like Cartesia).
        When a synthesizer is interrupted or expires, ConnectionPool
        automatically creates a new one on the next get() call.

        Args:
            count: Number of synthesizers to create and warm up. Defaults to 1.
        """
        import asyncio

        async def _prewarm_multiple() -> None:
            """Create and warm up multiple synthesizers sequentially with jittered expiry times."""
            import random

            for i in range(count):
                try:
                    logger.info(f"prewarming synthesizer {i + 1}/{count}")
                    # Use the pool's connect method which handles locking
                    async with self._pool._connect_lock:
                        if len(self._pool._connections) < i + 1:  # Only create if needed
                            # Use longer timeout for prewarming (Azure warmup takes 2-3s)
                            conn = await self._pool._connect(timeout=30.0)

                            # Add random jitter to connection timestamp to stagger expiry times
                            # This prevents all prewarmed synthesizers from expiring simultaneously
                            # Jitter range: -60 to 0 seconds (spreads expiries over 1 minute)
                            jitter = random.uniform(-60, 0)
                            self._pool._connections[conn] += jitter

                            self._pool.put(conn)
                            logger.info(
                                f"synthesizer {i + 1}/{count} prewarmed",
                                extra={
                                    "pool_total": len(self._pool._connections),
                                    "pool_available": len(self._pool._available),
                                },
                            )
                except Exception as e:
                    logger.warning(f"failed to prewarm synthesizer {i + 1}/{count}: {e}")

        # Run prewarm in background and store task for cleanup
        self._prewarm_task = asyncio.create_task(_prewarm_multiple())

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        # Cancel prewarm task if running
        if self._prewarm_task is not None:
            await utils.aio.gracefully_cancel(self._prewarm_task)
            self._prewarm_task = None

        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ssml(self) -> str:
        lang = self._opts.language or "en-US"
        ssml = (
            f'<speak version="1.0" '
            f'xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" '
            f'xml:lang="{lang}">'
        )
        ssml += f'<voice name="{self._opts.voice}">'

        if is_given(self._opts.lexicon_uri):
            ssml += f'<lexicon uri="{self._opts.lexicon_uri}"/>'

        if is_given(self._opts.style):
            degree = f' styledegree="{self._opts.style.degree}"' if self._opts.style.degree else ""
            ssml += f'<mstts:express-as style="{self._opts.style.style}"{degree}>'

        if is_given(self._opts.prosody):
            p = self._opts.prosody

            rate_attr = f' rate="{p.rate}"' if p.rate is not None else ""
            vol_attr = f' volume="{p.volume}"' if p.volume is not None else ""
            pitch_attr = f' pitch="{p.pitch}"' if p.pitch is not None else ""
            ssml += f"<prosody{rate_attr}{vol_attr}{pitch_attr}>{self.input_text}</prosody>"
        else:
            ssml += self.input_text

        if is_given(self._opts.style):
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"
        return ssml

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        headers = {
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": SUPPORTED_OUTPUT_FORMATS[self._opts.sample_rate],
            "User-Agent": "LiveKit Agents",
        }
        if self._opts.auth_token:
            headers["Authorization"] = f"Bearer {self._opts.auth_token}"

        elif self._opts.subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = self._opts.subscription_key

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        try:
            async with self._tts._ensure_session().post(
                url=self._opts.get_endpoint_url(),
                headers=headers,
                data=self._build_ssml(),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError(str(e)) from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS using Azure Speech SDK with TextStream input."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._text_ch = utils.aio.Chan[str | None]()
        # Check environment variable for audio saving
        self._save_audio = int(os.getenv("AZURE_TTS_SAVE_AUDIO", 0)) > 0
        self._audio_data: list[bytes] = []
        self._text_data: list[str] = []  # Store text chunks

    async def _create_replacement_synthesizers(self, target_count: int) -> None:
        """Create replacement synthesizers in the background to maintain pool health.

        Args:
            target_count: Number of synthesizers to create to reach target pool size
        """
        import random

        for i in range(target_count):
            try:
                logger.debug(f"creating replacement synthesizer {i + 1}/{target_count} for pool")
                # Use the pool's connect method which handles locking and jitter
                async with self._tts._pool._connect_lock:
                    # Create new synthesizer with longer timeout (Azure warmup takes 2-3s)
                    conn = await self._tts._pool._connect(timeout=30.0)

                    # Add random jitter to prevent simultaneous expiry
                    jitter = random.uniform(-60, 0)
                    self._tts._pool._connections[conn] += jitter

                    self._tts._pool.put(conn)
                    logger.info(
                        f"replacement synthesizer {i + 1}/{target_count} created",
                        extra={
                            "pool_total": len(self._tts._pool._connections),
                            "pool_available": len(self._tts._pool._available),
                        },
                    )
            except Exception as e:
                logger.warning(
                    f"failed to create replacement synthesizer {i + 1}/{target_count}: {e}"
                )

            # Sleep between creations to avoid thundering herd
            # Skip sleep after last iteration
            if i < target_count - 1:
                await asyncio.sleep(0.5)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _forward_input() -> None:
            """Forward text chunks directly to synthesis."""
            async for input in self._input_ch:
                if isinstance(input, str):
                    self._text_ch.send_nowait(input)
                elif isinstance(input, self._FlushSentinel):
                    # Flush marks segment boundary
                    self._text_ch.send_nowait(None)
            self._text_ch.close()

        # Log pool status before acquiring connection
        pool_total = len(self._tts._pool._connections)
        pool_available = len(self._tts._pool._available)
        logger.info(
            "acquiring synthesizer from pool",
            extra={
                "pool_total": pool_total,
                "pool_available": pool_available,
                "pool_in_use": pool_total - pool_available,
            },
        )

        # Use context manager for automatic pool lifecycle management
        # On success: synthesizer returned to pool
        # On error/interruption: synthesizer removed from pool (disposed and discarded)
        synthesis_failed = False
        synthesis_interrupted = False
        synthesizer = None
        expired_count = 0
        try:
            synthesizer = await self._tts._pool.get(timeout=self._conn_options.timeout)

            # Log pool status after acquiring
            pool_total_after = len(self._tts._pool._connections)
            pool_available_after = len(self._tts._pool._available)

            # Check if connections were expired during acquisition
            expired_count = pool_total - pool_total_after
            if expired_count > 0:
                logger.info(
                    "expired synthesizers removed during acquisition",
                    extra={
                        "expired_count": expired_count,
                        "pool_total_before": pool_total,
                        "pool_total_after": pool_total_after,
                    },
                )
                # Proactively create replacements for expired synthesizers
                # The pool already created 1 new connection, but it may not be warm yet
                # Create additional warm synthesizers to maintain target pool size
                target_size = self._tts._target_pool_size
                current_size = pool_total_after
                needed = max(0, target_size - current_size)

                if needed > 0:
                    logger.info(
                        f"creating {needed} replacement synthesizer(s) for expired connections",
                        extra={
                            "target_pool_size": target_size,
                            "current_pool_size": current_size,
                            "expired_count": expired_count,
                        },
                    )
                    # Store background tasks for proper lifecycle management
                    task = asyncio.create_task(self._create_replacement_synthesizers(needed))
                    # Optionally add to a set to prevent GC and allow cleanup
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

            logger.info(
                "synthesizer acquired from pool",
                extra={
                    "pool_total": pool_total_after,
                    "pool_available": pool_available_after,
                    "pool_in_use": pool_total_after - pool_available_after,
                },
            )

            tasks = [
                asyncio.create_task(_forward_input()),
                asyncio.create_task(self._synthesize_segment(output_emitter, synthesizer)),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                # User interrupted - remove synthesizer to be safe (may be in bad state)
                logger.debug("synthesis cancelled, removing synthesizer from pool")
                synthesis_interrupted = True
                raise  # Re-raise so base TTS class knows synthesis was cancelled
            except asyncio.TimeoutError:
                synthesis_failed = True
                raise APITimeoutError() from None
            except (APIStatusError, APIConnectionError):
                synthesis_failed = True
                raise  # Synthesizer will be removed in finally block
            except Exception as e:
                synthesis_failed = True
                raise APIConnectionError(str(e)) from e
            finally:
                logger.debug("cancelling forward_input and synthesis tasks")
                await utils.aio.gracefully_cancel(*tasks)

            # Success - return synthesizer to pool
            self._tts._pool.put(synthesizer)
            synthesizer = None  # Mark as returned
        except BaseException:
            # If synthesizer wasn't returned yet, remove it from pool
            # (either failed or interrupted - both need fresh synthesizer)
            if synthesizer is not None:
                logger.debug("removing synthesizer from pool due to failure/interrupt")
                self._tts._pool.remove(synthesizer)
                # Immediately close the connection instead of waiting for next drain cycle
                await self._tts._pool._drain_to_close()
                synthesizer = None  # Mark as removed
            raise
        finally:
            # Log pool status after returning/removing synthesizer
            pool_total_final = len(self._tts._pool._connections)
            pool_available_final = len(self._tts._pool._available)

            if synthesis_failed:
                log_msg = "removed failed synthesizer from pool"
            elif synthesis_interrupted:
                log_msg = "removed interrupted synthesizer from pool"
            else:
                log_msg = "returning synthesizer to pool"

            logger.info(
                log_msg,
                extra={
                    "pool_total": pool_total_final,
                    "pool_available": pool_available_final,
                    "pool_in_use": pool_total_final - pool_available_final,
                    "synthesis_failed": synthesis_failed,
                    "synthesis_interrupted": synthesis_interrupted,
                },
            )

            # Proactively replace interrupted or failed synthesizers to maintain pool health
            if synthesis_failed or synthesis_interrupted:
                # Calculate how many synthesizers needed to reach target pool size
                target_size = self._tts._target_pool_size
                current_size = pool_total_final
                needed = max(0, target_size - current_size)

                if needed > 0:
                    logger.info(
                        f"proactively creating {needed} replacement synthesizer(s) to reach target pool size",
                        extra={
                            "target_pool_size": target_size,
                            "current_pool_size": current_size,
                            "replacements_needed": needed,
                        },
                    )
                    # Store background tasks for proper lifecycle management
                    task = asyncio.create_task(self._create_replacement_synthesizers(needed))
                    # Optionally add to a set to prevent GC and allow cleanup
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

    async def _synthesize_segment(
        self, output_emitter: tts.AudioEmitter, synthesizer: speechsdk.SpeechSynthesizer
    ) -> None:
        """Synthesize using Azure SDK with streaming text and audio."""
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        # Use asyncio Queue to bridge sync callbacks to async
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        synthesis_error: list[Exception] = []
        text_queue: queue.Queue[str | None] = queue.Queue()  # Sync queue for thread-safe access
        cancelled = [False]  # Flag to signal cancellation to SDK thread
        first_audio_received = [False]  # Track first audio chunk
        synthesis_start_time = [0.0]  # Track when synthesis starts
        audio_chunks_from_sdk = [0]  # Track total chunks received from SDK
        audio_chunks_queued = [0]  # Track chunks successfully queued
        import threading

        audio_lock = threading.Lock()  # Lock to synchronize audio callbacks
        synthesis_complete_event = threading.Event()  # Signal when synthesis truly complete

        # Get the event loop before entering the thread
        loop = asyncio.get_running_loop()

        def _run_sdk_synthesis() -> None:
            """Run Azure SDK synthesis in sync mode with streaming callbacks."""
            import threading

            thread_id = threading.current_thread().ident
            logger.debug(f"SDK synthesis thread started (thread_id={thread_id})")

            # Send audio to async queue (thread-safe)
            async def _put_audio(audio_chunk: bytes | None, reason: str | None = None) -> None:
                # await asyncio.sleep(0.2)
                await audio_queue.put(audio_chunk)
                if audio_chunk:
                    logger.debug(
                        "_put_audio",
                        extra={
                            "chunk_number": audio_chunks_from_sdk[0],
                            "chunk_size": len(audio_chunk),
                        },
                    )
                if reason:
                    logger.debug(
                        "_put_audio",
                        extra={
                            "reason": reason,
                        },
                    )

            def synthesizing_callback(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
                """Called when audio chunks are available during synthesis."""
                import time

                if cancelled[0]:
                    return  # Discard audio if cancelled
                if evt.result.audio_data:
                    with audio_lock:  # Ensure completion callback waits for us
                        # Raw PCM format - no headers to strip
                        audio_chunk = evt.result.audio_data

                        # Track chunks received from SDK
                        audio_chunks_from_sdk[0] += 1

                        # Save audio data if enabled
                        if self._save_audio:
                            self._audio_data.append(bytes(audio_chunk))

                        # Log first audio chunk received
                        if not first_audio_received[0]:
                            first_audio_received[0] = True
                            ttfb = time.time() - synthesis_start_time[0]
                            logger.info(
                                "first audio chunk received from Azure",
                                extra={
                                    "time_to_first_byte": f"{ttfb:.3f}s",
                                    "chunk_size": len(audio_chunk),
                                },
                            )

                        future = asyncio.run_coroutine_threadsafe(_put_audio(audio_chunk), loop)
                        # Wait for it to complete to ensure ordering
                        future.result()
                        audio_chunks_queued[0] += 1

            def completed_callback(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
                """Called when synthesis completes successfully."""
                # Signal that synthesis is complete from SDK perspective
                synthesis_complete_event.set()
                with audio_lock:
                    logger.debug(
                        f"Synthesis completed callback, {audio_chunks_queued[0]} chunks queued so far"
                    )

            def canceled_callback(evt: speechsdk.SpeechSynthesisEventArgs) -> None:
                """Called when synthesis is canceled or fails."""
                cancellation = evt.result.cancellation_details

                # Signal that synthesis is complete (even though cancelled)
                synthesis_complete_event.set()

                # Check if this was a user cancellation (synthesizer still healthy)
                # vs a server/connection error (synthesizer broken)
                if cancellation.reason == speechsdk.CancellationReason.CancelledByUser:
                    # User interrupted - synthesizer is still healthy
                    logger.debug(f"synthesis cancelled by user: {cancellation.error_details}")
                    # Don't add to synthesis_error - this is not a failure
                    # Signal completion normally
                    asyncio.run_coroutine_threadsafe(
                        _put_audio(None, reason=cancellation.error_details), loop
                    )
                else:
                    # Real error (connection error, server error, etc.)
                    error = APIStatusError(
                        f"Azure TTS synthesis canceled: {cancellation.reason}. "
                        f"Error: {cancellation.error_details}"
                    )
                    synthesis_error.append(error)
                    # Signal error completion
                    asyncio.run_coroutine_threadsafe(
                        _put_audio(None, reason=cancellation.error_details), loop
                    )

            # Connect event handlers
            synthesizer.synthesizing.connect(synthesizing_callback)
            synthesizer.synthesis_completed.connect(completed_callback)
            synthesizer.synthesis_canceled.connect(canceled_callback)

            # Create streaming request
            tts_request = speechsdk.SpeechSynthesisRequest(
                input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
            )

            try:
                # Start synthesis (returns result future)
                result_future = synthesizer.speak_async(tts_request)

                # Stream text pieces as they arrive from the queue
                chunk_count = [0]
                while True:
                    if cancelled[0]:
                        logger.debug("sdk thread detected cancellation, stopping text streaming")
                        break
                    try:
                        # Use timeout to periodically check cancelled flag and avoid hang
                        text_piece = text_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue  # Check cancelled flag and try again
                    if text_piece is None:
                        break
                    if cancelled[0]:
                        break
                    chunk_count[0] += 1

                    # Start timer when first text chunk is sent
                    if chunk_count[0] == 1:
                        import time

                        synthesis_start_time[0] = time.time()
                        logger.debug("sending first text chunk to Azure TTS")

                    tts_request.input_stream.write(text_piece)
                    # Log every 20 chunks to reduce log verbosity
                    if chunk_count[0] % 20 == 0:
                        logger.info(
                            "streaming tts chunk",
                            extra={"chunk": chunk_count[0], "text": text_piece},
                        )

                # Close input stream to signal completion
                tts_request.input_stream.close()

                # Wait for synthesis to complete with timeout to prevent hanging
                # Poll with timeout to allow checking cancelled flag
                result = None
                while not cancelled[0]:
                    # Wait for completion event with short timeout
                    if synthesis_complete_event.wait(timeout=0.1):
                        # Event was set, get the result (should be ready now)
                        result = result_future.get()
                        logger.debug(
                            f"synthesis complete event set, retrieving result{result.reason}"
                        )
                        break

                # Now wait for all audio chunks to be queued by acquiring the lock
                # This ensures any in-flight synthesizing_callback calls complete
                with audio_lock:
                    logger.debug(
                        f"Synthesis result received, {audio_chunks_queued[0]} total chunks queued"
                    )
                    if (
                        result
                        and result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
                    ):
                        asyncio.run_coroutine_threadsafe(
                            _put_audio(None, reason="SynthesizingAudioCompleted"), loop
                        )

                # Disconnect event handlers to prevent them from firing on next use
                synthesizer.synthesizing.disconnect_all()
                synthesizer.synthesis_completed.disconnect_all()
                synthesizer.synthesis_canceled.disconnect_all()

            except Exception as e:
                synthesis_error.append(e)
                asyncio.run_coroutine_threadsafe(_put_audio(None, reason=str(e)), loop)
                # Disconnect handlers even on error
                try:
                    synthesizer.synthesizing.disconnect_all()
                    synthesizer.synthesis_completed.disconnect_all()
                    synthesizer.synthesis_canceled.disconnect_all()
                except Exception:
                    pass

        async def _stream_text_input() -> None:
            """Stream text chunks to the SDK as they arrive."""
            async for text_chunk in self._text_ch:
                if text_chunk is None:
                    # End of segment
                    break
                self._mark_started()
                # Save text if audio saving is enabled
                if self._save_audio:
                    self._text_data.append(text_chunk)
                # Send to sync thread via sync queue
                text_queue.put(text_chunk)
            # Signal end of text
            text_queue.put(None)

        async def _receive_audio() -> None:
            """Receive audio chunks as they arrive."""
            chunk_count = 0
            try:
                while True:
                    if cancelled[0]:
                        logger.debug("audio reception cancelled, discarding remaining chunks")
                        # Drain remaining audio
                        while not audio_queue.empty():
                            try:
                                await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                            except Exception:
                                break
                        break

                    audio_chunk = await audio_queue.get()

                    # None signals completion
                    if audio_chunk is None:
                        logger.debug(
                            f"Received completion signal, pushed {chunk_count} audio chunks to playback (SDK sent {audio_chunks_from_sdk[0]}, queued {audio_chunks_queued[0]})"
                        )
                        break

                    # Only push audio if not cancelled
                    if not cancelled[0]:
                        chunk_count += 1
                        output_emitter.push(audio_chunk)
            except asyncio.CancelledError:
                cancelled[0] = True
                raise

        try:
            # Start SDK synthesis in thread pool
            synthesis_task = loop.run_in_executor(None, _run_sdk_synthesis)
            logger.debug("Started synthesis_task in thread pool executor")

            # Run text streaming and audio receiving concurrently
            await asyncio.gather(_stream_text_input(), _receive_audio())

            # Check for errors
            if synthesis_error:
                raise synthesis_error[0]

            # Wait for synthesis thread to complete
            logger.debug("Waiting for synthesis_task to complete (success path)")
            await synthesis_task
            logger.debug("synthesis_task completed successfully")

            output_emitter.end_segment()

            # Save audio to file if enabled
            if self._save_audio and len(self._audio_data) > 0:
                try:
                    os.makedirs("audio_debug", exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    filename = f"audio_debug/response_{segment_id}_{timestamp}.wav"
                    text_filename = f"audio_debug/response_{segment_id}_{timestamp}.txt"

                    # Save audio
                    with wave.open(filename, "wb") as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 2 bytes for PCM16
                        wav_file.setframerate(self._opts.sample_rate)
                        wav_file.writeframes(b"".join(self._audio_data))

                    # Save text
                    with open(text_filename, "w", encoding="utf-8") as text_file:
                        text_file.write("".join(self._text_data))

                    total_audio_bytes = sum(len(chunk) for chunk in self._audio_data)
                    logger.info(
                        f"Saved audio response to {filename} ({len(self._audio_data)} chunks, {total_audio_bytes} bytes)"
                    )
                    logger.info(
                        f"Saved text to {text_filename} ({''.join(self._text_data)[:50]}...)"
                    )
                    # Clear data for next synthesis
                    self._audio_data.clear()
                    self._text_data.clear()
                except Exception as e:
                    logger.error(f"Failed to save audio file: {e}")

        except asyncio.CancelledError:
            # Clean up on interruption
            logger.info("synthesis interrupted, stopping synthesizer")
            cancelled[0] = True

            # Stop the Azure synthesizer to terminate ongoing synthesis
            try:
                if synthesizer:
                    # Fire-and-forget stop - don't wait for Azure SDK
                    synthesizer.stop_speaking_async()
                    logger.debug("stop signal sent (non-blocking)")
            except Exception as e:
                logger.warning(f"error stopping synthesizer: {e}")

            # Signal SDK thread to stop
            text_queue.put(None)
            logger.debug("Sent None to text_queue to signal SDK thread to stop")

            # Wait for SDK thread to complete before cleanup
            # This ensures the thread releases its hold on the synthesizer
            try:
                await asyncio.wait_for(synthesis_task, timeout=2.0)
                logger.debug("synthesis_task completed after cancellation")
            except asyncio.TimeoutError:
                logger.warning("synthesis_task did not complete within timeout")
            except Exception as e:
                logger.debug(f"synthesis_task error during cleanup: {e}")

            # Flush any queued audio immediately
            output_emitter.flush()
            # End segment on cancellation
            try:
                output_emitter.end_segment()
            except Exception:
                pass  # Segment might already be ended
            # Drain queues
            while not text_queue.empty():
                try:
                    text_queue.get_nowait()
                except Exception:
                    break
            while not audio_queue.empty():
                try:
                    await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                except Exception:
                    break

            # Re-raise to allow pool cleanup logic to remove synthesizer
            logger.debug("re-raising cancellation to trigger pool cleanup")
            raise
        except Exception as e:
            # End segment on error
            try:
                output_emitter.end_segment()
            except Exception:
                pass  # Segment might already be ended
            # Clean up queues on error
            while not text_queue.empty():
                try:
                    text_queue.get_nowait()
                except Exception:
                    break
            while not audio_queue.empty():
                try:
                    await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                except Exception:
                    break

            raise APIConnectionError(f"Azure SDK synthesis failed: {e}") from e
