from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, tts
from livekit.plugins.google.beta.gemini_tts import TTS

pytestmark = pytest.mark.plugin("google")


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_gemini_tts_success(mock_genai_client_class) -> None:
    # Setup mocks for GenAI Client
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    # Mock chunk response candidates
    class MockInlineData:
        def __init__(self, data: bytes):
            self.data = data
            self.mime_type = "audio/pcm"

    class MockPart:
        def __init__(self, data: bytes):
            self.inline_data = MockInlineData(data)

    class MockContent:
        def __init__(self, data: bytes):
            self.parts = [MockPart(data)]

    class MockCandidate:
        def __init__(self, data: bytes):
            self.content = MockContent(data)

    class MockChunk:
        def __init__(self, data: bytes):
            self.candidates = [MockCandidate(data)]

    async def mock_generator(*args, **kwargs):
        yield MockChunk(b"\x00" * 4800)
        yield MockChunk(b"\x01" * 4800)

    mock_stream.side_effect = mock_generator

    # Initialize TTS
    google_tts = TTS(api_key="test-api-key")

    # Create output emitter mock
    mock_emitter = MagicMock(spec=tts.AudioEmitter)

    stream = google_tts._synthesize(
        tts.SpeechChunk(text="Hello world", said="Hello world"),
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )
    try:
        await stream._run(mock_emitter)
    finally:
        await stream.aclose()

    # Assertions
    mock_stream.assert_called_once()
    mock_emitter.initialize.assert_called_once()
    assert mock_emitter.push.call_count == 2


def _audio_response(data: bytes = bytes(4800)):
    """A generate_content_stream stand-in yielding one PCM chunk."""
    inline = MagicMock(data=data, mime_type="audio/pcm")
    chunk = MagicMock(
        candidates=[MagicMock(content=MagicMock(parts=[MagicMock(inline_data=inline)]))]
    )

    async def _stream(*args, **kwargs):
        yield chunk

    return _stream


def test_expressive_only_on_the_models_that_style_per_part() -> None:
    # only a model that can carry a style out of band may declare the dialect
    with patch("livekit.plugins.google.beta.gemini_tts.Client"):
        for model in ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"):
            assert TTS(api_key="k", model=model).markup._provider_key() == "gemini"
        # `model` is typed `| str`: a dated build of one still counts
        boom = TTS(api_key="k", model="gemini-3.8-flash-tts-preview-09-2026")
        assert boom.markup._provider_key() == "gemini"
        assert boom.markup.llm_instructions() is not None

        older = TTS(api_key="k", model="gemini-2.5-flash-preview-tts")
        assert older.markup._provider_key() == ""
        assert older.markup.llm_instructions() is None


def _request_body(mock_stream) -> dict:
    """The contents the SDK will actually send, after extra_body is merged in."""
    config = mock_stream.call_args.kwargs["config"]
    body: dict = {"contents": ["<typed contents, replaced below>"]}
    if config.http_options and config.http_options.extra_body:
        body.update(config.http_options.extra_body)
    return body


async def _speak(google_tts: TTS, text: str) -> None:
    """Synthesize ``text`` through the framework, which lowers it for Gemini."""
    async with google_tts.synthesize(text) as stream:
        async for _ in stream:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"])
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_expressive_markup_becomes_speech_metadata(mock_genai_client_class, model) -> None:
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model=model)
    await _speak(
        google_tts, '<expr type="expression" label="Thoughtful, Quiet, American accent"/> Sienna?'
    )

    assert _request_body(mock_stream)["contents"] == [
        {
            "parts": [
                {
                    "text": '"Sienna?"',
                    "speech_metadata": {"style": "Thoughtful, Quiet, American accent"},
                }
            ]
        }
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_unmarked_text_keeps_the_plain_prompt(mock_genai_client_class) -> None:
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    await _speak(google_tts, "Hello world")

    config = mock_stream.call_args.kwargs["config"]
    # nothing to style, so contents is left alone and the words go as they are
    assert "contents" not in (config.http_options.extra_body or {})
    assert mock_stream.call_args.kwargs["contents"] == "Hello world"


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_markers_reach_the_plugin_through_the_stream_adapter(
    mock_genai_client_class,
) -> None:
    """End to end for the seam that makes any of this reachable.

    Gemini TTS isn't streaming, so the agent drives it through ``tts.StreamAdapter``.
    """
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream

    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    adapter = tts.StreamAdapter(tts=google_tts)
    assert adapter.markup._provider_key() == "gemini"

    try:
        async with adapter.stream() as stream:
            # split mid-marker, the way tokens actually arrive from an LLM
            for chunk in ['<expr type="expression" label="Warm, ', 'Welcoming"/> Hey there.']:
                stream.push_text(chunk)
            stream.end_input()
            async for _ in stream:
                pass
    finally:
        await adapter.aclose()

    assert _request_body(mock_stream)["contents"] == [
        {"parts": [{"text": '"Hey there."', "speech_metadata": {"style": "Warm, Welcoming"}}]}
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_instructions_ride_the_style_not_the_spoken_text(mock_genai_client_class) -> None:
    """A preamble in a part's text is read out loud by this family.

    Measured: prefixing "Say the text with a proper tone..." roughly tripled a short
    sentence, 6.96s against 2.60s. The direction channel is speech_metadata.style.
    """
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    # the default preamble is dropped entirely for this family
    assert TTS(api_key="k", model="gemini-3.8-flash-tts")._opts.instructions is None
    assert TTS(api_key="k", model="gemini-2.5-flash-preview-tts")._opts.instructions is not None

    google_tts = TTS(
        api_key="test-api-key", model="gemini-3.8-flash-tts", instructions="Speak slowly"
    )
    await _speak(google_tts, '<expr type="expression" label="Wistful"/> Sienna?')

    assert _request_body(mock_stream)["contents"] == [
        {"parts": [{"text": '"Sienna?"', "speech_metadata": {"style": "Speak slowly, Wistful"}}]}
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_inline_events_stay_in_the_words(mock_genai_client_class) -> None:
    """The two channels have to part ways correctly, or one of them is lost.

    An inline tag in the style field does nothing; a style label left in the text is
    read out loud.
    """
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    await _speak(
        google_tts,
        '<expr type="expression" label="Easygoing, Warm"/> Yeah, '
        '<expr type="sound" label="chuckle"/> I get that a lot.',
    )

    assert _request_body(mock_stream)["contents"] == [
        {
            "parts": [
                {
                    "text": '"Yeah, <chuckle> I get that a lot."',
                    "speech_metadata": {"style": "Easygoing, Warm"},
                }
            ]
        }
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_headerless_pcm_is_asked_for_explicitly(mock_genai_client_class) -> None:
    # the emitter is initialized for raw audio/pcm, and the docs only guarantee a
    # headerless stream when response_format names it
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    await _speak(google_tts, "Hello world")

    extra = mock_stream.call_args.kwargs["config"].http_options.extra_body
    assert extra["generationConfig"] == {"response_format": {"audio": {"mime_type": "AUDIO_L16"}}}


def test_multi_speaker_config_is_validated() -> None:
    with patch("livekit.plugins.google.beta.gemini_tts.Client"):
        speakers = {"Sienna": "Kore", "Comanchero": "Puck"}

        # a multi-speaker turn whose speech_metadata names no speaker is rejected by the
        # API, so there is no useful default to fall back to
        with pytest.raises(ValueError, match="`speaker` is required"):
            TTS(api_key="k", model="gemini-3.8-flash-tts", speakers=speakers)

        # the API rejects any count but two -- a single speaker, and an empty table
        # that would otherwise slip past into the single-voice config
        for bad in ({}, {"Solo": "Kore"}, {"A": "Kore", "B": "Puck", "C": "Charon"}):
            with pytest.raises(ValueError, match="exactly 2 speakers"):
                TTS(api_key="k", model="gemini-3.8-flash-tts", speakers=bad, speaker="A")

        with pytest.raises(ValueError, match="not one of the configured speakers"):
            TTS(api_key="k", model="gemini-3.8-flash-tts", speakers=speakers, speaker="Nobody")

        # the speaker travels in speech_metadata, which the older models have no field for
        with pytest.raises(ValueError, match="per-part speech_metadata"):
            TTS(
                api_key="k",
                model="gemini-2.5-flash-preview-tts",
                speakers=speakers,
                speaker="Sienna",
            )

        google_tts = TTS(
            api_key="k", model="gemini-3.8-flash-tts", speakers=speakers, speaker="Sienna"
        )
        google_tts.update_options(speaker="Comanchero")
        assert google_tts._opts.speaker == "Comanchero"
        with pytest.raises(ValueError, match="not one of the configured speakers"):
            google_tts.update_options(speaker="Nobody")


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_every_multi_speaker_turn_names_its_speaker(mock_genai_client_class) -> None:
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(
        api_key="test-api-key",
        model="gemini-3.8-flash-tts",
        speakers={"Sienna": "Kore", "Comanchero": "Puck"},
        speaker="Sienna",
    )
    # plain words, no markers: the part is still sent, because the speaker has to travel
    await _speak(google_tts, "Sienna?")

    assert _request_body(mock_stream)["contents"] == [
        {"parts": [{"text": '"Sienna?"', "speech_metadata": {"speaker": "Sienna"}}]}
    ]

    speech = mock_stream.call_args.kwargs["config"].speech_config
    assert speech.voice_config is None
    assert [c.speaker for c in speech.multi_speaker_voice_config.speaker_voice_configs] == [
        "Sienna",
        "Comanchero",
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_speaker_and_style_travel_together(mock_genai_client_class) -> None:
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(
        api_key="test-api-key",
        model="gemini-3.8-flash-lite-tts",
        speakers={"Sienna": "Kore", "Comanchero": "Puck"},
        speaker="Sienna",
    )
    await _speak(google_tts, '<expr type="expression" label="Wistful"/> Sienna?')

    assert _request_body(mock_stream)["contents"] == [
        {
            "parts": [
                {
                    "text": '"Sienna?"',
                    "speech_metadata": {"style": "Wistful", "speaker": "Sienna"},
                }
            ]
        }
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_each_styled_sentence_gets_its_own_part(mock_genai_client_class) -> None:
    """Gemini takes a style per part, so a style change has to open a new one.

    A text can carry several styled sentences; folding them into one part would speak
    the later ones with the first one's delivery.
    """
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    await _speak(
        google_tts,
        '<expr type="expression" label="Warm"/> Hello. '
        '<expr type="expression" label="Sad"/> <expr type="sound" label="sigh"/> Goodbye.',
    )

    assert _request_body(mock_stream)["contents"] == [
        {
            "parts": [
                {"text": '"Hello."', "speech_metadata": {"style": "Warm"}},
                # the inline event stays in the words it belongs to
                {"text": '"<sigh> Goodbye."', "speech_metadata": {"style": "Sad"}},
            ]
        }
    ]


@pytest.mark.asyncio
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_an_unstyled_span_does_not_sink_the_request(mock_genai_client_class) -> None:
    """A turn that opens before its first marker still has to have its markers taken out.

    Abandoning the parts would fall back to the raw text, and since gemini conversion
    deliberately leaves expression markers standing, Gemini would read them aloud.
    """
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    await _speak(google_tts, 'Hello. <expr type="expression" label="Sad"/> Goodbye.')

    assert _request_body(mock_stream)["contents"] == [
        {
            "parts": [
                {"text": '"Hello."'},  # no direction of its own, and no metadata key
                {"text": '"Goodbye."', "speech_metadata": {"style": "Sad"}},
            ]
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ('Hello <expr type="sound" label="laugh"/> there.', "Hello <laugh> there."),
        ('Hello <expr type="break" label="300ms"/> there.', "Hello <short pause> there."),
        ('Say <expr type="prosody" label="emphasis">this</expr> now.', "Say THIS now."),
    ],
)
@patch("livekit.plugins.google.beta.gemini_tts.Client")
async def test_lowered_markup_without_a_style_keeps_the_plain_prompt(
    mock_genai_client_class, written, spoken
) -> None:
    """The framework lowers the markup before the plugin, so with no style to carry the
    lowered words go as the plain prompt."""
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client
    mock_stream = AsyncMock()
    mock_client.aio.models.generate_content_stream = mock_stream
    mock_stream.side_effect = _audio_response()

    google_tts = TTS(api_key="test-api-key", model="gemini-3.8-flash-tts")
    await _speak(google_tts, written)

    assert "contents" not in (mock_stream.call_args.kwargs["config"].http_options.extra_body)
    assert mock_stream.call_args.kwargs["contents"] == spoken
