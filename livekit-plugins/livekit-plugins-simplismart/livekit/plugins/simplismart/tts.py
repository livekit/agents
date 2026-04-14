import asyncio
import os
import traceback

import aiohttp

from .log import logger
from pydantic import BaseModel

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
    tts,
    utils,
)

from .models import TTSModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/tts"


class SimplismartTTSOptions(BaseModel):
    """Configuration options for SimpliSmart TTS models."""

    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    max_tokens: int = 1000


class QwenTTSOptions(BaseModel):
    """Configuration options for Qwen 3 TTS."""

    language: str = "English"
    leading_silence: bool = True


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str = SIMPLISMART_BASE_URL,
        model: TTSModels | str = "canopylabs/orpheus-3b-0.1-ft",
        voice: str = "tara",
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        sample_rate: int = 24000,
        options: SimplismartTTSOptions | QwenTTSOptions | None = SimplismartTTSOptions(),
    ) -> None:
        """
        Configuration options for SimpliSmart TTS (Text-to-Speech) models.

        Supports both OpenAI compatible and Qwen 3 TTS models. Use SimplismartTTSOptions
        for OpenAI-compatible configuration or QwenTTSOptions for Qwen 3 configuration.

        Args:
            base_url: Base URL for the TTS endpoint. Auto-detected based on options type if not provided.
            model: TTS model to use (default: "canopylabs/orpheus-3b-0.1-ft" for Orpheus, "qwen-tts" for Qwen).
            voice: Voice/speaker identifier for synthesis (default: "tara" for Orpheus, "Chelsie" for Qwen).
            api_key: API key for authentication (defaults to SIMPLISMART_API_KEY env var).
            http_session: Optional aiohttp session for reuse.
            sample_rate: Audio sample rate in Hz (default: 24000).
            options: Configuration options - use SimplismartTTSOptions for OpenAI-compatible or QwenTTSOptions for Qwen 3.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._base_url = base_url or SIMPLISMART_BASE_URL
        self._model = model
        self._voice = voice
        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        self._session = http_session

        # Determine options type - default to Orpheus for backwards compatibility
        if options is None:
            self._opts = SimplismartTTSOptions()
        else:
            self._opts = options

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "SimpliSmart"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            conn_options: Connection options for the API request.

        Returns:
            ChunkedStream: Stream of audio data.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis and stream audio chunks.

        Uses type-based detection to determine which API format to use:
        - QwenTTSOptions → Qwen 3 endpoint format
        - SimplismartTTSOptions → Orpheus endpoint format
        """
        logger.info(
            f"TTS synthesis starting - text: {self._input_text[:50]}... voice: {self._tts._voice}"
        )

        # Determine payload format based on options type
        if isinstance(self._opts, QwenTTSOptions):
            # Qwen 3 TTS format
            payload = {
                "text": self._input_text,
                "language": self._opts.language,
                "speaker": self._tts._voice,
                "leading_silence": self._opts.leading_silence,
            }
            headers = {
                "Authorization": f"Bearer {self._tts._api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/L16",
            }
        else:
            # Orpheus TTS format
            payload = self._opts.model_dump()
            payload["prompt"] = self._input_text
            payload["voice"] = self._tts._voice
            payload["model"] = self._tts._model
            headers = {
                "Authorization": f"Bearer {self._tts._api_key}",
                "Content-Type": "application/json",
            }

        logger.info(f"TTS request to {self._tts._base_url} with payload type: {type(self._opts).__name__}")

        try:
            async with self._tts._ensure_session().post(
                self._tts._base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                logger.info(f"TTS response status: {resp.status}")
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Simplismart TTS API error: {resp.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Simplismart TTS API Error: {error_text}",
                        status_code=resp.status,
                        request_id=None,
                        body=error_text,
                    )

                # Initialize audio emitter
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )
                logger.info(f"TTS audio emitter initialized - sample_rate: {self._tts.sample_rate}")

                # Stream audio chunks
                chunk_count = 0
                total_bytes = 0
                async for audio_data, _ in resp.content.iter_chunks():
                    if audio_data:
                        chunk_count += 1
                        total_bytes += len(audio_data)
                        output_emitter.push(audio_data)

                logger.info(f"TTS received {chunk_count} chunks, {total_bytes} bytes total")
                output_emitter.flush()
                logger.info("TTS synthesis completed successfully")

        except asyncio.TimeoutError as e:
            logger.error(f"Simplismart TTS API timeout: {e}")
            raise APITimeoutError("Simplismart TTS API request timed out") from e
        except aiohttp.ClientError as e:
            logger.error(f"Simplismart TTS API client error: {e}")
            raise APIConnectionError(f"Simplismart TTS API connection error: {e}") from e
        except APIStatusError:
            raise
        except Exception as e:
            logger.error(f"Error during Simplismart TTS processing: {traceback.format_exc()}")
            raise APIConnectionError(f"Unexpected error in Simplismart TTS: {e}") from e