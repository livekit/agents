from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict

from google.auth.credentials import Credentials
from google.genai import Client, types
from google.genai.errors import APIError
from livekit.agents import APIConnectionError, APIStatusError, tokenize, tts, utils
from livekit.agents.tts import split_expr_markup
from livekit.agents.types import (
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .utils import gemini_backend, genai_http_options

GEMINI_TTS_MODELS = Literal[
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

DEFAULT_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "Kore"
DEFAULT_SAMPLE_RATE = 24000  # not configurable
NUM_CHANNELS = 1

DEFAULT_INSTRUCTIONS = "Say the text with a proper tone, don't omit or add any words"


class _SpeechMetadata(TypedDict, total=False):
    style: str
    speaker: str


class _Part(TypedDict):
    text: str
    speech_metadata: NotRequired[_SpeechMetadata]


@dataclass
class _TTSOptions:
    model: GEMINI_TTS_MODELS | str
    voice_name: GEMINI_VOICES | str
    instructions: str | None
    speakers: dict[str, str] | None
    speaker: str | None


class GeminiTTS(tts.TTS):
    """Text-to-speech through the Gemini API's speech generation."""

    def __init__(
        self,
        *,
        model: GEMINI_TTS_MODELS | str = DEFAULT_MODEL,
        voice_name: GEMINI_VOICES | str = DEFAULT_VOICE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        credentials: NotGivenOr[Credentials] = NOT_GIVEN,
        credentials_info: NotGivenOr[Mapping[str, object]] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str | None] = NOT_GIVEN,
        speakers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Gemini TTS.

        Args:
            model: The Gemini TTS model to use. Defaults to "gemini-3.8-flash-tts".
            voice_name: The voice to use for synthesis. Defaults to "Kore".
            api_key: Gemini API key. Defaults to the `GOOGLE_API_KEY` environment variable.
            vertexai: Whether to use Vertex AI. Defaults to `GOOGLE_GENAI_USE_VERTEXAI`, else
                to Vertex AI when credentials, a project or a location are given.
            credentials: Credentials for Vertex AI.
            credentials_info: Service account info for Vertex AI.
            credentials_file: Path to a service account JSON file for Vertex AI.
            project: Google Cloud project for Vertex AI. Defaults to `GOOGLE_CLOUD_PROJECT`,
                then the credentials' project.
            location: Google Cloud region for Vertex AI. Defaults to `GOOGLE_CLOUD_LOCATION`,
                then "us-central1".
            instructions: Control the style, tone, accent, and pace using prompts. See
                https://ai.google.dev/gemini-api/docs/speech-generation#controllable
            speakers: Speaker name -> voice, for a multi-speaker voice config. Replaces
                `voice_name`.
            speaker: Which of `speakers` this instance voices. Required with `speakers`, and
                switchable via `update_options`.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        backend = gemini_backend(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            credentials_info=credentials_info,
            credentials_file=credentials_file,
            default_location="us-central1",
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
            # this family speaks anything left in a part's text, so the preamble would
            # be read aloud (measured: it roughly triples a short line). _styled_part
            # puts instructions in speech_metadata.style instead.
            instructions=instructions
            if is_given(instructions)
            else (None if _styles_per_part(model) else DEFAULT_INSTRUCTIONS),
        )

        self._client = Client(
            api_key=backend.api_key,
            vertexai=backend.vertexai,
            credentials=backend.credentials,
            project=backend.project,
            location=backend.location,
        )

    class Markup(tts.TTS.Markup):
        # only a model that can carry a style out of band declares the dialect; on the
        # others a marker would have nowhere to go and would be read aloud
        def _provider_key(self) -> str:
            assert isinstance(self._tts, GeminiTTS)
            return "gemini" if _styles_per_part(self._tts.model) else ""

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        if self._client.vertexai:
            return "vertex_ai"
        else:
            return "gemini"

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            options=replace(self._opts),
            client=self._client,
        )

    def update_options(
        self,
        *,
        voice_name: NotGivenOr[str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice_name (str, optional): The voice to use for synthesis.
            speaker (str, optional): Which configured speaker to voice from now on.
            sentence_tokenizer (tokenize.SentenceTokenizer, optional): What splits the text into the
                chunks the TTS is sent.
            pacing (tts.SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or
                ``None`` to send each chunk as soon as it is split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(speaker):
                if not self._opts.speakers:
                    raise ValueError("`speaker` needs a TTS constructed with `speakers`")
                if speaker not in self._opts.speakers:
                    raise ValueError(
                        f"speaker {speaker!r} is not one of the configured speakers: "
                        f"{', '.join(sorted(self._opts.speakers))}"
                    )
        if is_given(voice_name):
            self._opts.voice_name = voice_name
        if is_given(speaker):
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
    def __init__(
        self,
        *,
        tts: GeminiTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        options: _TTSOptions,
        client: Client,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._options = options
        self._client = client

    def _styled_parts(self) -> list[_Part] | None:
        """Build one request part per delivery style, or ``None`` if none is carried.

        Discrete events are already inline Gemini tags here and belong in the words; the ``expression`` marker is the other channel and comes out,
        splitting the text wherever the delivery changes::

            {"parts": [{"text": "\"<chuckle> Sienna?\"",
                        "speech_metadata": {"style": "Thoughtful, Quiet"}},
                       {"text": "\"What's on your mind?\"",
                        "speech_metadata": {"style": "Wistful"}}]}

        A text may carry several sentences, and each keeps the style that governs it --
        including a leading one with no style of its own, which travels as a plain part. ``split_expr_markup`` keeps the inline
        tags in the text.
        """
        opts = self._options
        if not _styles_per_part(opts.model):
            return None

        text = self._input_text
        # slice at each marker, keeping it at the head of its span so the shared splitter
        # reads the label off it
        bounds = [0, *(m.start() for m in _EXPRESSION_MARKER_RE.finditer(text)), len(text)]
        spans = [text[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]

        parts: list[_Part] = []
        stripped_a_marker = False
        for span in spans:
            split = split_expr_markup(span)
            words, markers = split.text, split.tags
            stripped_a_marker |= any(t["type"] == "expression" for t in markers)
            if not (words := words.strip()):
                continue
            marker = next((t["value"] for t in markers if t["type"] == "expression"), "")
            part: _Part = {"text": f'"{words}"'}
            metadata: _SpeechMetadata = {}
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
        # would send the markers for Gemini to read out. Only hand back None when the
        # input carried no marker, where the plain prompt says the same thing.
        if not parts or not (stripped_a_marker or any("speech_metadata" in p for p in parts)):
            return None
        return parts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        http_options = genai_http_options(self._conn_options, None)
        try:
            opts = self._options
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=_speech_config(opts),
                http_options=http_options,
            )
            input_text = self._input_text
            if opts.instructions is not None:
                input_text = f'{opts.instructions}:\n"{input_text}"'

            # neither speech_metadata nor response_format has a field on the typed config
            # (types.Part and SpeechConfig both forbid extras), so both ride extra_body:
            # it merges into the request body, replacing `contents` wholesale since lists
            # overwrite, and merging into `generationConfig` since dicts recurse.
            extra_body: dict[str, object] = {}
            if _styles_per_part(opts.model):
                extra_body["generationConfig"] = {"response_format": _RESPONSE_FORMAT}
            if (styled_parts := self._styled_parts()) is not None:
                extra_body["contents"] = [{"parts": styled_parts}]
            if extra_body:
                http_options.extra_body = extra_body

            response = await self._client.aio.models.generate_content_stream(
                model=self._options.model,
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

        except APIError as e:
            raise APIStatusError(
                "gemini tts: api error", status_code=e.code, body=f"{e.message} {e.status}"
            ) from e
        except Exception as e:
            raise APIConnectionError(f"gemini tts: error generating speech {str(e)}") from e
