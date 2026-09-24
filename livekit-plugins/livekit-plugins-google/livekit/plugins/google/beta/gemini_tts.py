from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.tts._provider_format import split_expr_markup
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger

GEMINI_TTS_MODELS = Literal[
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
]

# Models taking a per-part ``speech_metadata.style`` instead of one prompt-wide
# instruction. Substring-matched, so a dated build of either counts. The list stays
# narrow -- gemini-3.1 and 2.5 reject the field outright ("Speech metadata is not
# supported for this model"), and an unlisted model quietly loses its styles while a
# wrongly listed one fails every request.
_STYLE_METADATA_MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")


def _styles_per_part(model: str) -> bool:
    """Whether *model* accepts a delivery style on each part of the request."""
    return any(family in model for family in _STYLE_METADATA_MODELS)


# Headerless PCM, asked for explicitly: the emitter is initialized for raw audio/pcm, and
# the docs only guarantee a headerless stream when response_format names it. The docs
# spell it "audio/l16"; on the wire it is an enum (AUDIO_MULAW and AUDIO_ALAW are the
# other two) nested under a ResponseFormatConfig that the typed SDK has no field for.
_RESPONSE_FORMAT = {"audio": {"mime_type": "AUDIO_L16"}}

# where one part ends and the next begins: Gemini takes a style per part, and an
# expression marker is the only thing that changes it
_EXPRESSION_MARKER_RE = re.compile(r'<expr\b(?=[^>]*type="expression")[^>]*?/\s*>')


GEMINI_VOICES = Literal[
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

DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_VOICE = "Kore"
DEFAULT_SAMPLE_RATE = 24000  # not configurable
NUM_CHANNELS = 1

DEFAULT_INSTRUCTIONS = "Say the text with a proper tone, don't omit or add any words"


@dataclass
class _TTSOptions:
    model: GEMINI_TTS_MODELS | str
    voice_name: GEMINI_VOICES | str
    vertexai: bool
    project: str | None
    location: str | None
    instructions: str | None
    speakers: dict[str, str] | None
    speaker: str | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: GEMINI_TTS_MODELS | str = DEFAULT_MODEL,
        voice_name: GEMINI_VOICES | str = DEFAULT_VOICE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str | None] = NOT_GIVEN,
        speakers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Gemini TTS.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file.
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            model (str, optional): The Gemini TTS model to use. Defaults to "gemini-3.1-flash-tts-preview".
            voice_name (str, optional): The voice to use for synthesis. Defaults to "Kore".
            api_key (str, optional): The API key for Google Gemini. If not provided, it attempts to read from the `GOOGLE_API_KEY` environment variable.
            vertexai (bool, optional): Whether to use VertexAI. Defaults to False.
            project (str, optional): The Google Cloud project to use (only for VertexAI).
            location (str, optional): The location to use for VertexAI API requests. Defaults to "us-central1".
            instructions (str, optional): Control the style, tone, accent, and pace using prompts. See https://ai.google.dev/gemini-api/docs/speech-generation#controllable
            speakers (dict[str, str], optional): Speaker name -> voice, for a multi-speaker voice config. Replaces `voice_name`.
            speaker (str, optional): Which of `speakers` this instance voices. Required with `speakers`, and switchable via `update_options`.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        gcp_project: str | None = (
            project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        gcp_location: str | None = (
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

                _, gcp_project = default_async(  # type: ignore
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

        speaker_map = dict(speakers) if is_given(speakers) else None
        current_speaker = speaker if is_given(speaker) else None
        if speaker_map is not None:
            # not "up to two": the API rejects any other count outright, with
            # "the number of speaker_voice_configs must equal 2"
            if len(speaker_map) != 2:
                raise ValueError(f"`speakers` must name exactly 2 speakers, got {len(speaker_map)}")
            # the API rejects a multi-speaker turn whose speech_metadata names no speaker,
            # so there is no useful default to fall back to here
            if current_speaker is None:
                raise ValueError("`speaker` is required when `speakers` is set")
            if current_speaker not in speaker_map:
                raise ValueError(
                    f"speaker {current_speaker!r} is not one of the configured speakers: "
                    f"{', '.join(sorted(speaker_map))}"
                )
            if not _styles_per_part(model):
                raise ValueError(
                    f"multi-speaker needs a model that takes per-part speech_metadata; "
                    f"{model!r} does not"
                )

        self._opts = _TTSOptions(
            speakers=speaker_map,
            speaker=current_speaker,
            model=model,
            voice_name=voice_name,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
            # this family speaks anything left in a part's text, so the preamble would
            # be read aloud (measured: it roughly triples a short line). _styled_part
            # puts instructions in speech_metadata.style instead.
            instructions=instructions
            if is_given(instructions)
            else (None if _styles_per_part(model) else DEFAULT_INSTRUCTIONS),
        )

        self._client = Client(
            api_key=gemini_api_key,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
        )

    class Markup(tts.TTS.Markup):
        # only a model that can carry a style out of band declares the dialect; on the
        # others a marker would have nowhere to go and would be read aloud
        def _provider_key(self) -> str:
            assert isinstance(self._tts, TTS)
            return "gemini" if _styles_per_part(self._tts._opts.model) else ""

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        if self._client.vertexai:
            return "Vertex AI"
        else:
            return "Gemini"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice_name: NotGivenOr[str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice_name (str, optional): The voice to use for synthesis.
            speaker (str, optional): Which configured speaker to voice from now on.
        """
        if is_given(voice_name):
            self._opts.voice_name = voice_name
        if is_given(speaker):
            if not self._opts.speakers:
                raise ValueError("`speaker` needs a TTS constructed with `speakers`")
            if speaker not in self._opts.speakers:
                raise ValueError(
                    f"speaker {speaker!r} is not one of the configured speakers: "
                    f"{', '.join(sorted(self._opts.speakers))}"
                )
            self._opts.speaker = speaker

    async def aclose(self) -> None:
        """Close the TTS and release its GenAI HTTP clients."""
        try:
            await self._client.aio.aclose()
        except Exception:
            logger.warning("failed to close the genai client", exc_info=True)


def _speech_config(opts: _TTSOptions) -> types.SpeechConfig:
    """One voice, or a speaker->voice table when the TTS was given `speakers`."""
    if opts.speakers:
        return types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker=name,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        ),
                    )
                    for name, voice in opts.speakers.items()
                ]
            )
        )
    return types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=opts.voice_name)
        )
    )


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    def _styled_parts(self) -> list[dict[str, Any]] | None:
        """Build one request part per delivery style, or ``None`` if none is carried.

        Discrete events are already inline Gemini tags here (``convert`` lowered them) and
        belong in the words; the ``expression`` marker is the other channel and comes out,
        splitting the text wherever the delivery changes::

            {"parts": [{"text": "\"<chuckle> Sienna?\"",
                        "speech_metadata": {"style": "Thoughtful, Quiet"}},
                       {"text": "\"What's on your mind?\"",
                        "speech_metadata": {"style": "Wistful"}}]}

        The agent's stream adapter hands over one sentence at a time, so a turn usually
        makes one part; a direct ``synthesize()`` call may carry several sentences, and
        each keeps the style that governs it -- including a leading one with no style of
        its own, which travels as a plain part. Hence ``split_expr_markup``, not
        ``split_all_markup`` -- the latter would take the inline tags out too.
        """
        opts = self._tts._opts
        if not _styles_per_part(opts.model):
            return None

        markup = self._tts.markup
        # the stream adapter has already lowered; a direct synthesize() call has not
        text = markup.convert(markup.normalize(self._input_text))
        # whatever `convert` lowered only exists here, so these parts are the only copy
        # of it -- handing back None would send the raw input, markers and all
        lowered = text != self._input_text
        # slice at each marker, keeping it at the head of its span so the shared splitter
        # reads the label off it
        bounds = [0, *(m.start() for m in _EXPRESSION_MARKER_RE.finditer(text)), len(text)]
        spans = [text[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]

        parts: list[dict[str, Any]] = []
        stripped_a_marker = False
        for span in spans:
            words, markers = split_expr_markup(span)
            stripped_a_marker |= any(t["type"] == "expression" for t in markers)
            if not (words := words.strip()):
                continue
            marker = next((t["value"] for t in markers if t["type"] == "expression"), "")
            part: dict[str, Any] = {"text": f'"{words}"'}
            metadata: dict[str, str] = {}
            if style := ", ".join(p for p in (opts.instructions, marker) if p):
                metadata["style"] = style
            if opts.speaker:
                # every turn of a multi-speaker request has to name its speaker, so a
                # part carries one even with no style of its own
                metadata["speaker"] = opts.speaker
            if metadata:
                part["speech_metadata"] = metadata
            parts.append(part)

        # a span with no direction is still a part: dropping the whole request over it
        # would send the raw text for Gemini to read out. Only hand back None when the
        # input carried no markup at all, where the plain prompt says the same thing.
        carries_markup = lowered or stripped_a_marker
        if not parts or not (carries_markup or any("speech_metadata" in p for p in parts)):
            return None
        return parts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            opts = self._tts._opts
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=_speech_config(opts),
            )
            input_text = self._input_text
            if opts.instructions is not None:
                input_text = f'{opts.instructions}:\n"{input_text}"'

            # neither speech_metadata nor response_format has a field on the typed config
            # (types.Part and SpeechConfig both forbid extras), so both ride extra_body:
            # it merges into the request body, replacing `contents` wholesale since lists
            # overwrite, and merging into `generationConfig` since dicts recurse.
            extra_body: dict[str, Any] = {}
            if _styles_per_part(opts.model):
                extra_body["generationConfig"] = {"response_format": _RESPONSE_FORMAT}
            if (styled_parts := self._styled_parts()) is not None:
                extra_body["contents"] = [{"parts": styled_parts}]
            if extra_body:
                config.http_options = types.HttpOptions(extra_body=extra_body)

            response = await self._tts._client.aio.models.generate_content_stream(
                model=self._tts._opts.model,
                contents=input_text,
                config=config,
            )

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type="audio/pcm",
            )

            async for chunk in response:
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    for part in chunk.candidates[0].content.parts:
                        if (
                            (inline_data := part.inline_data)
                            and inline_data.data
                            and inline_data.mime_type
                            and inline_data.mime_type.startswith("audio/")
                        ):
                            # mime_type: audio/L16;codec=pcm;rate=24000
                            output_emitter.push(inline_data.data)

        except ClientError as e:
            raise APIStatusError(
                "gemini tts: client error",
                status_code=e.code,
                body=f"{e.message} {e.status}",
                retryable=True if e.code in {429, 499} else False,
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
