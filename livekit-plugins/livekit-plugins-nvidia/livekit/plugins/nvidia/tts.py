import json
import logging
import os
from dataclasses import dataclass

import riva.client
import riva.client.proto.riva_tts_pb2 as riva_tts

from livekit.agents import (
    APIConnectOptions,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import is_given

logger = logging.getLogger(__name__)


@dataclass
class TTSOptions:
    voice: str
    function_id: str
    server: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice: str = "Magpie-Multilingual.EN-US.Sofia",
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        api_key: str | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=16000,
            num_channels=1,
        )

        if is_given(api_key):
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set. Either pass api_key parameter or set NVIDIA_API_KEY environment variable"
                )

        self._opts = TTSOptions(voice=voice, function_id=function_id, server=server)
        self._tts_service = None

    def _ensure_session(self) -> riva.client.SpeechSynthesisService:
        if not self._tts_service:
            # Use the same Auth pattern as STT (which works!)
            auth = riva.client.Auth(
                uri=self._opts.server,
                use_ssl=True,
                metadata_args=[
                    ["authorization", f"Bearer {self.nvidia_api_key}"],
                    ["function-id", self._opts.function_id],
                ],
            )

            # WORKAROUND: Manually patch the auth metadata since get_auth_metadata() is broken
            auth.metadata = [
                ("authorization", f"Bearer {self.nvidia_api_key}"),
                ("function-id", self._opts.function_id),
            ]

            logger.debug(f"Manually set auth metadata: {auth.metadata}")
            self._tts_service = riva.client.SpeechSynthesisService(auth)
        return self._tts_service

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        raise NotImplementedError("Chunked synthesis is not supported for NVIDIA TTS")

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options, opts=self._opts)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: TTSOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Test NVIDIA TTS authentication - simplified for debugging."""
        try:
            logger.debug("Testing NVIDIA TTS authentication...")
            service = self._tts._ensure_session()
            logger.info(f"TTS service created successfully: {type(service)}")

            # Check the manually patched auth metadata
            logger.debug(f"TTS service auth metadata: {service.auth.metadata}")

            self.list_voices(service)
            logger.info("NVIDIA TTS service initialization successful!")

        except Exception as e:
            logger.exception(f"Error in NVIDIA TTS: {e}")
            logger.warning("NVIDIA TTS authentication failed, but continuing...")

        # Don't call end_input() or end_segment() since we're not producing audio
        # Let the TTS framework handle the lifecycle

    def list_voices(self, service: riva.client.SpeechSynthesisService) -> None:
        """List available TTS voices from NVIDIA."""
        try:
            config_response = service.stub.GetRivaSynthesisConfig(
                riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
            )
            tts_models = {}
            for model_config in config_response.model_config:
                language_code = model_config.parameters.get("language_code", "unknown")
                voice_name = model_config.parameters.get("voice_name", "unknown")
                subvoices_str = model_config.parameters.get("subvoices", "")

                if subvoices_str:
                    subvoices = [voice.split(":")[0] for voice in subvoices_str.split(",")]
                    full_voice_names = [voice_name + "." + subvoice for subvoice in subvoices]
                else:
                    full_voice_names = [voice_name]

                if language_code in tts_models:
                    tts_models[language_code]["voices"].extend(full_voice_names)
                else:
                    tts_models[language_code] = {"voices": full_voice_names}

            tts_models = dict(sorted(tts_models.items()))
            logger.info("Available TTS voices:")
            logger.info(json.dumps(tts_models, indent=4))

        except Exception as e:
            logger.error(f"Error listing TTS voices: {e}")
            # Don't raise to allow debugging
            logger.warning("TTS voice listing failed, skipping...")
            return
