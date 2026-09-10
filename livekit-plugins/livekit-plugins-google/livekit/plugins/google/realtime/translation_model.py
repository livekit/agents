from __future__ import annotations

from dataclasses import replace

import google.auth.credentials
from google.genai import types
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .realtime_api import RealtimeModel, RealtimeSession

DEFAULT_TRANSLATION_MODEL = "gemini-3.5-live-translate-preview"


class RealtimeTranslationModel(RealtimeModel):
    """Gemini Live realtime translation model (``gemini-3.5-live-translate-preview``).

    A thin specialization of the conversational :class:`RealtimeModel`: the Gemini
    translate model speaks the same Live API protocol (each translated utterance is a
    turn delimited by ``turn_complete``), so the entire connection / receive /
    generation / transcription / metrics machinery is inherited unchanged. The only
    difference is the connect config — see :class:`RealtimeTranslationSession`.

    It performs live speech-to-speech translation: audio streamed in one language is
    translated to ``target_language`` and returned as audio + transcript while the
    speaker is still talking. Use it as a drop-in realtime model, with no STT/TTS/VAD::

        session = AgentSession(llm=RealtimeTranslationModel(target_language="es"))

    or drive a :meth:`session` directly (e.g. to translate many tracks into many
    languages at once — see the multi-user translator example).

    Notes:
        - One model instance translates into a single ``target_language`` (one-way).
        - Pair it with an empty ``Agent`` (no instructions/tools/chat history): the
          translate model takes none of those, and they are not sent.
    """

    def __init__(
        self,
        *,
        target_language: str,
        model: str = DEFAULT_TRANSLATION_MODEL,
        echo_target_language: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        http_options: NotGivenOr[types.HttpOptions] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        credentials: google.auth.credentials.Credentials | None = None,
    ) -> None:
        """
        Args:
            target_language: Target language code, e.g. ``"es"`` (BCP-47). Sent as
                ``TranslationConfig.target_language_code``.
            model: Translation model name. Defaults to ``gemini-3.5-live-translate-preview``.
            echo_target_language: When True (default) the model produces translated
                audio. When False no audio is generated (translated text only).
            api_key: Google Gemini API key. If None, read from ``GOOGLE_API_KEY``.
            vertexai: Use VertexAI instead of the Gemini Developer API. Defaults to
                the ``GOOGLE_GENAI_USE_VERTEXAI`` env var.
            project: GCP project (VertexAI only). Defaults to ``GOOGLE_CLOUD_PROJECT``.
            location: GCP location (VertexAI only). Defaults to ``GOOGLE_CLOUD_LOCATION``
                or ``us-central1``.
            http_options: Optional ``types.HttpOptions`` for the genai client.
            conn_options: Retry/backoff and connection settings.
            credentials: Google auth credentials (VertexAI only).
        """
        super().__init__(
            model=model,
            modalities=[types.Modality.AUDIO],
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            http_options=http_options,
            conn_options=conn_options,
            credentials=credentials,
        )
        self._target_language = target_language
        self._echo_target_language = echo_target_language
        # a translator has no system prompt, tools, mutable history, or generate_reply;
        # disabling mutation also makes the inherited generate_reply reject cleanly.
        self._capabilities = replace(
            self._capabilities,
            mutable_chat_context=False,
            mutable_instructions=False,
            auto_tool_reply_generation=False,
        )

    @property
    def target_language(self) -> str:
        return self._target_language

    @property
    def echo_target_language(self) -> bool:
        return self._echo_target_language

    def session(self) -> RealtimeTranslationSession:
        sess = RealtimeTranslationSession(self)
        self._sessions.add(sess)
        return sess

    def update_options(  # type: ignore[override]
        self, *, target_language: NotGivenOr[str] = NOT_GIVEN
    ) -> None:
        """Update the target language. Existing sessions reconnect to apply it."""
        if is_given(target_language) and target_language != self._target_language:
            self._target_language = target_language
            for sess in self._sessions:
                sess._mark_restart_needed()


class RealtimeTranslationSession(RealtimeSession):
    """Session for the Gemini translate model.

    Inherits the full turn-based lifecycle from :class:`RealtimeSession` and only
    overrides the connect config: the translate model is configured solely by
    ``translation_config`` and takes no system prompt, tools, voice, or history
    (matching Google's translate example and reference LiveKit bridge).
    """

    def __init__(self, realtime_model: RealtimeTranslationModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeTranslationModel = realtime_model

    def _build_connect_config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            translation_config=types.TranslationConfig(
                target_language_code=self._realtime_model.target_language,
                echo_target_language=self._realtime_model.echo_target_language,
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
