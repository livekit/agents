from __future__ import annotations

import os
from dataclasses import dataclass

from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

# Gemini TTS models that support native text-to-speech
GEMINI_TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

# Available voice options for Gemini TTS
GEMINI_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"
DEFAULT_SAMPLE_RATE = 24000  # not configurable
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: str
    voice_name: str
    vertexai: bool
    project: str | None
    location: str | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice_name: str = DEFAULT_VOICE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Gemini TTS.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file.
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            model (str, optional): The Gemini TTS model to use. Defaults to "gemini-2.5-flash-preview-tts".
            voice_name (str, optional): The voice to use for synthesis. Defaults to "Kore".
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 24000.
            api_key (str, optional): The API key for Google Gemini. If not provided, it attempts to read from the `GOOGLE_API_KEY` environment variable.
            vertexai (bool, optional): Whether to use VertexAI. Defaults to False.
            project (str, optional): The Google Cloud project to use (only for VertexAI).
            location (str, optional): The location to use for VertexAI API requests. Defaults to "us-central1".
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False
            ),  # Gemini TTS doesn't support streaming yet
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        if model not in GEMINI_TTS_MODELS:
            raise ValueError(
                f"Model {model} is not supported. Available models: {GEMINI_TTS_MODELS}"
            )

        if voice_name not in GEMINI_VOICES:
            raise ValueError(
                f"Voice {voice_name} is not supported. Available voices: {GEMINI_VOICES}"
            )

        # Setup authentication similar to LLM plugin
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location = (
            location
            if is_given(location)
            else os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )
        use_vertexai = (
            vertexai
            if is_given(vertexai)
            else os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in ["true", "1"]
        )
        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")

        if use_vertexai:
            if not gcp_project:
                from google.auth._default_async import default_async

                _, gcp_project = default_async(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            gemini_api_key = None  # VertexAI does not require an API key
        else:
            gcp_project = None
            gcp_location = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        self._opts = _TTSOptions(
            model=model,
            voice_name=voice_name,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
        )

        self._client = Client(
            api_key=gemini_api_key,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice_name (str, optional): The voice to use for synthesis.
        """
        if is_given(voice_name):
            if voice_name not in GEMINI_VOICES:
                raise ValueError(
                    f"Voice {voice_name} is not supported. Available voices: {GEMINI_VOICES}"
                )
            self._opts.voice_name = voice_name

    async def aclose(self) -> None:
        # Close any resources if needed
        pass


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            # Configure the request for Gemini TTS
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._tts._opts.voice_name,
                        )
                    )
                ),
            )

            # Make the request to Gemini TTS
            response = await self._tts._client.aio.models.generate_content(
                model=self._tts._opts.model,
                contents=self._input_text,
                config=config,
            )

            # Check if we have a valid response
            if not response.candidates or not response.candidates[0].content:
                raise APIStatusError("No audio content generated")

            # Extract audio data from the response
            audio_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                    audio_data = part.inline_data.data
                    # audio/L16;codec=pcm;rate=24000
                    mime_type = "audio/pcm"
                    break

            if not audio_data:
                raise APIStatusError("No audio data found in response")

            # Initialize the audio emitter
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type=mime_type,
            )

            # Push the audio data
            output_emitter.push(audio_data)

        except ClientError as e:
            raise APIStatusError(
                "gemini tts: client error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=False if e.code != 429 else True,
            ) from e
        except ServerError as e:
            raise APIStatusError(
                "gemini tts: server error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=True,
            ) from e
        except APIError as e:
            raise APIStatusError(
                "gemini tts: api error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=True,
            ) from e
        except Exception as e:
            raise APIConnectionError(
                f"gemini tts: error generating speech {str(e)}",
                retryable=True,
            ) from e
