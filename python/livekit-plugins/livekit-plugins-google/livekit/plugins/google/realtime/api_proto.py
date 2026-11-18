"""The Gemini Live protocol (BidiGenerateContent over a WebSocket), shared by the realtime model
and Gemini STT: where a session connects, and the messages the plugin reads and sends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import JsonValue

import google.auth.credentials
import google.auth.transport.requests
from google.genai import types
from livekit.agents.llm import realtime as llm_realtime

from ..utils import auth
from ..version import __version__

# Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations
# Gemini API release notes with preview deprecations: https://ai.google.dev/gemini-api/docs/changelog
# live models: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
# VertexAI retirement: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#retired-models
LiveAPIModels = Literal[
    # Models available on both APIs
    "gemini-3.8-live",
    # VertexAI models
    "gemini-live-2.5-flash-native-audio",  # GA https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api#live-2.5-flash
    # Gemini API models
    "gemini-3.8-live-extended-thinking",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",  # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-live
]

Voice = Literal[
    "Achernar",
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Charon",
    "Despina",
    "Enceladus",
    "Erinome",
    "Fenrir",
    "Gacrux",
    "Iapetus",
    "Kore",
    "Laomedeia",
    "Leda",
    "Orus",
    "Pulcherrima",
    "Puck",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Sulafat",
    "Umbriel",
    "Vindemiatrix",
    "Zephyr",
    "Zubenelgenubi",
]

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_MULTI_REGIONAL_LOCATIONS = {"us", "eu"}


@dataclass
class LiveEndpoint:
    """A Live API backend: the Gemini API with a key, or Vertex AI with a project."""

    vertexai: bool
    api_key: str | None
    project: str | None
    location: str | None
    credentials: google.auth.credentials.Credentials | None
    http_options: types.HttpOptions | None
    api_version: str | None = None

    async def target(self, model: str) -> tuple[llm_realtime.WebSocketTarget, str]:
        """The socket to open, with fresh credentials, and the model as the setup names it."""
        options = self.http_options or types.HttpOptions()
        headers = {**(options.headers or {}), "x-goog-api-client": f"livekit-agents/{__version__}"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        version = options.api_version or self.api_version
        if not self.vertexai:
            base = options.base_url or "https://generativelanguage.googleapis.com/"
            method = "BidiGenerateContent"
            if self.api_key and self.api_key.startswith("auth_tokens/"):
                # an ephemeral token
                method = "BidiGenerateContentConstrained"
                headers["Authorization"] = f"Token {self.api_key}"
            path = (
                f"ws/google.ai.generativelanguage.{version or 'v1beta'}.GenerativeService.{method}"
            )
            name = model if model.startswith(("models/", "tunedModels/")) else f"models/{model}"
            return _target(base, path, headers), name

        if options.base_url:
            base = options.base_url
        elif (self.api_key and not self.location) or self.location == "global":
            base = "https://aiplatform.googleapis.com/"
        elif self.location in _MULTI_REGIONAL_LOCATIONS:
            base = f"https://aiplatform.{self.location}.rep.googleapis.com/"
        else:
            base = f"https://{self.location}-aiplatform.googleapis.com/"
        path = (
            f"ws/google.cloud.aiplatform.{version or 'v1beta1'}.LlmBidiService/BidiGenerateContent"
        )
        if not self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {await self._token()}"
        if model.startswith(("projects/", "models/", "publishers/")):
            name = model
        elif "/" in model:
            publisher, model_id = model.split("/", 1)
            name = f"publishers/{publisher}/models/{model_id}"
        else:
            name = f"publishers/google/models/{model}"
        if name.startswith("publishers/") and self.project and self.location:
            name = f"projects/{self.project}/locations/{self.location}/{name}"
        return _target(base, path, headers), name

    async def _token(self) -> str:
        def fresh() -> str:
            credentials = self.credentials
            if credentials is None:
                credentials, _ = auth.default(scopes=_SCOPES)
            if not (credentials.token and credentials.valid):
                credentials.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
            return str(credentials.token)

        # google-auth refreshes over blocking HTTP
        return await asyncio.to_thread(fresh)


def _target(base: str, path: str, headers: dict[str, str]) -> llm_realtime.WebSocketTarget:
    scheme, sep, rest = base.partition("://")
    ws_scheme = {"https": "wss", "http": "ws"}.get(scheme, scheme)
    return llm_realtime.WebSocketTarget(
        f"{ws_scheme}{sep}{rest.rstrip('/')}/{path}", headers=headers
    )


def dump(value: Any) -> Any:
    """A google-genai type as the Live API reads it: camelCase JSON."""
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


# server messages: only the fields the plugins read


class Transcription(TypedDict, total=False):
    text: str
    languageCode: str


class InlineData(TypedDict):
    data: str  # base64
    mimeType: NotRequired[str]


class Part(TypedDict, total=False):
    text: str
    thought: bool
    inlineData: InlineData


class ModelTurn(TypedDict, total=False):
    parts: list[Part]


class ServerContent(TypedDict, total=False):
    modelTurn: ModelTurn
    inputTranscription: Transcription
    interimInputTranscription: Transcription
    outputTranscription: Transcription
    generationComplete: bool
    turnComplete: bool
    turnCompleteReason: str
    interrupted: bool


class FunctionCall(TypedDict, total=False):
    id: str
    name: str
    args: dict[str, JsonValue]


class ToolCall(TypedDict, total=False):
    functionCalls: list[FunctionCall]


class ToolCallCancellation(TypedDict, total=False):
    ids: list[str]


class ModalityTokenCount(TypedDict, total=False):
    modality: str
    tokenCount: int


class UsageMetadata(TypedDict, total=False):
    promptTokenCount: int
    cachedContentTokenCount: int
    responseTokenCount: int
    candidatesTokenCount: int  # Vertex AI's name for responseTokenCount
    toolUsePromptTokenCount: int
    thoughtsTokenCount: int
    promptTokensDetails: list[ModalityTokenCount]
    cacheTokensDetails: list[ModalityTokenCount]
    responseTokensDetails: list[ModalityTokenCount]
    candidatesTokensDetails: list[ModalityTokenCount]
    toolUsePromptTokensDetails: list[ModalityTokenCount]


class GoAway(TypedDict, total=False):
    timeLeft: str


class SessionResumptionUpdate(TypedDict, total=False):
    newHandle: str
    resumable: bool


class ServerMessage(TypedDict, total=False):
    """One server message: the fields present say what it carries."""

    setupComplete: dict[str, JsonValue]
    serverContent: ServerContent
    toolCall: ToolCall
    toolCallCancellation: ToolCallCancellation
    usageMetadata: UsageMetadata
    goAway: GoAway
    sessionResumptionUpdate: SessionResumptionUpdate


# client messages: each carries one of its fields, a google-genai type dumped by ``dump``


class ClientMessage(TypedDict, total=False):
    setup: dict[str, Any]
    clientContent: dict[str, Any]
    realtimeInput: dict[str, Any]
    toolResponse: dict[str, Any]
