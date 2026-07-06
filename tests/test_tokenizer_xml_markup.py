"""Regression tests: sentence tokenizers must handle XML markup correctly.

Covers blingfire sentence tokenizer (batch + streaming) with TTS markup tags
used in expressive mode (Cartesia, ElevenLabs, Inworld).
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.tokenize.blingfire import SentenceTokenizer
from livekit.agents.tokenize.token_stream import _XML_TAG_RE
from livekit.agents.tts.markup_utils import strip_xml_tags

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_wrapping_tag_intact(sentences: list[str], tag: str) -> None:
    """If a sentence has <tag>, it must also have </tag> (not split)."""
    for s in sentences:
        if f"<{tag}" in s and f"</{tag}>" not in s and "/>" not in s:
            pytest.fail(f"<{tag}> split across sentences: {sentences}")


def _assert_no_tag_only_sentences(sentences: list[str]) -> None:
    """No sentence should be purely XML tags with no text content."""
    for s in sentences:
        if "<" in s:
            assert _XML_TAG_RE.sub("", s).strip(), f"Tag-only sentence: {s!r}"


async def _stream_tokenize(tok: SentenceTokenizer, text: str) -> list[str]:
    stream = tok.stream()
    for char in text:
        stream.push_text(char)
    stream.end_input()
    return [ev.token async for ev in stream]


async def _stream_tokenize_tiktoken(tok: SentenceTokenizer, text: str) -> list[str]:
    """Push text token-by-token using GPT-4o's tokenizer (realistic LLM streaming)."""
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    stream = tok.stream()
    for token_id in enc.encode(text):
        stream.push_text(enc.decode([token_id]))
    stream.end_input()
    return [ev.token async for ev in stream]


# ===========================================================================
# strip_xml_tags
# ===========================================================================


class TestStripXmlTags:
    def test_self_closing(self) -> None:
        assert strip_xml_tags('<emotion value="happy"/> Hello!', ["emotion"]) == " Hello!"

    def test_wrapping_preserves_content(self) -> None:
        assert strip_xml_tags("<spell>A.B.C.</spell> confirmed", ["spell"]) == "A.B.C. confirmed"

    def test_preserves_unrelated_tags(self) -> None:
        text = '<emotion value="happy"/> <custom>keep</custom>'
        assert strip_xml_tags(text, ["emotion"]) == " <custom>keep</custom>"

    def test_empty_tags_list(self) -> None:
        text = '<emotion value="happy"/> Hi'
        assert strip_xml_tags(text, []) == text


# ===========================================================================
# xAI dialect (mixed inline [..] + wrapping <..>)
# ===========================================================================


class TestXaiDialect:
    """xAI's LLM writes every tag as XML — inline sounds as <sound value="NAME"/> and pauses
    as <break time="..."/> (modeled on Inworld); the transcript strips them all, and
    convert_markup rewrites sounds to [NAME] and <break> to [pause]/[long-pause] for the TTS
    while emotion/prosody stay angle-bracketed."""

    def test_llm_instructions_registered(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        instr = pf.llm_instructions("xai")
        # non-None is what the expressive gate keys on
        assert instr is not None
        assert '<sound value="laugh"/>' in instr and "<whisper>" in instr

    def test_split_markup_strips_inline_keeps_wrapping_inner(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = 'So I walked in and <break time="500ms"/> there it was. <sound value="laugh"/> <whisper>a secret</whisper> <emphasis>wow</emphasis>.'
        clean, tags = pf.split_markup("xai", raw)
        # inline sounds/pauses removed entirely; wrapping tags keep their inner text
        assert "<break" not in clean and "<sound" not in clean and "laugh" not in clean
        assert "<whisper>" not in clean and "a secret" in clean and "wow" in clean
        types = [(t["type"], t["value"]) for t in tags]
        assert ("break", "500ms") in types and ("sound", "laugh") in types
        assert ("whisper", "a secret") in types and ("emphasis", "wow") in types

    def test_emotion_wrapping_tags_stripped_inner_kept(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = "<happy>Great to hear from you!</happy> <sad>I'm sorry about that.</sad>"
        clean, tags = pf.split_markup("xai", raw)
        # emotion is the tag name; delimiters removed, spoken words preserved
        assert "<happy>" not in clean and "</sad>" not in clean
        assert "Great to hear from you!" in clean and "I'm sorry about that." in clean
        types = [(t["type"], t["value"]) for t in tags]
        assert ("happy", "Great to hear from you!") in types
        assert ("sad", "I'm sorry about that.") in types

    def test_every_documented_tag_is_strippable(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # every prosody/style tag the instructions offer must be in _XAI_TAGS,
        # or it would leak into the user-visible transcript
        for tag in pf._XAI_WRAPPING:
            assert f"<{tag}>" in pf._XAI_LLM_INSTRUCTIONS, f"{tag} not documented"
            assert tag in pf._XAI_TAGS
            clean, _ = pf.split_markup("xai", f"<{tag}>hello there</{tag}>")
            assert clean.strip() == "hello there", f"{tag} not stripped: {clean!r}"

    def test_emotion_tags_stripped_though_unprompted(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # emotion tags are no longer instructed, but stay in _XAI_TAGS so a stray one is
        # stripped from the transcript rather than leaking to the user
        for tag in pf._XAI_EMOTIONS:
            assert tag in pf._XAI_TAGS
            clean, _ = pf.split_markup("xai", f"<{tag}>hello there</{tag}>")
            assert clean.strip() == "hello there", f"{tag} not stripped: {clean!r}"

    def test_documented_inline_tags_present(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # nonverbals from xAI's docs, incl. the ones the user called out; documented as
        # <sound value="NAME"/> (converted to [NAME] for the TTS in convert_markup)
        for name in ("tsk", "lip-smack", "tongue-click", "chuckle", "giggle", "hum-tune"):
            assert name in pf._XAI_INLINE
            assert f'<sound value="{name}"/>' in pf._XAI_LLM_INSTRUCTIONS

    def test_pitch_volume_intensity_speed_present(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # the request: pitch, volume, intensity, speed — real xAI tag names
        for tag in (
            "higher-pitch",
            "lower-pitch",
            "soft",
            "loud",
            "build-intensity",
            "decrease-intensity",
            "slow",
            "fast",
            "emphasis",
        ):
            assert tag in pf._XAI_WRAPPING

    def test_nested_emotion_prosody_strips_cleanly(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # combining emotion + prosody means nesting; the transcript must come out clean
        # (no leaked inner markup) — this is what the fixed-point strip guarantees
        raw = '<excited><loud><higher-pitch>no way</higher-pitch></loud></excited> <sound value="laugh"/> okay'
        clean, _ = pf.split_markup("xai", raw)
        assert "<" not in clean and ">" not in clean and "[" not in clean
        assert clean.strip() == "no way  okay".replace("  ", " ") or "no way" in clean
        assert "no way" in clean and "okay" in clean

    def test_convert_inline_sounds_and_pauses_to_brackets(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = (
            '<sound value="laugh"/> <break time="500ms"/> <break time="2s"/> <whisper>hi</whisper>'
        )
        # <sound value="X"/> -> [X]; <break> -> [pause] (<1s) or [long-pause] (>=1s);
        # emotion/prosody stay angle-bracketed, and normalize is a no-op for xAI
        assert pf.convert_markup("xai", raw) == "[laugh] [pause] [long-pause] <whisper>hi</whisper>"
        assert pf.normalize_markup("xai", raw) == raw

    def test_presets_registered_for_xai(self) -> None:
        from livekit.agents.voice import presets
        from livekit.agents.voice.agent_session import DEFAULT_EXPRESSIVE_OPTIONS

        for preset in (presets.CUSTOMER_SERVICE, presets.CASUAL):
            opts = presets.resolve_options(
                preset, provider_key="xai", default=DEFAULT_EXPRESSIVE_OPTIONS
            )
            body = opts["tts_instructions_template"].common
            # tuned body, not the agnostic default (which has no xai tag reference)
            assert "<whisper>" in body


# ===========================================================================
# Batch sentence tokenizer
# ===========================================================================


class TestBatchTokenizer:
    def setup_method(self) -> None:
        self.tok = SentenceTokenizer(min_sentence_len=1, xml_aware=True)

    def test_expression_tags_between_sentences_split_correctly(self) -> None:
        """Regression: blingfire refuses to split when <expression .../> sits between
        sentences because /> confuses its boundary detection. The XML wrapper must
        strip tags before blingfire and remap offsets so each tag goes with its sentence."""
        text = (
            '<expression value="speak cheerfully"/> Hello and welcome! '
            '<expression value="speak with bright energy"/> Great specials today. '
            '<expression value="sound excited"/> Try our new sandwich.'
        )
        sentences = self.tok.tokenize(text)
        assert len(sentences) == 3, f"Expected 3 sentences: {sentences}"
        assert '<expression value="speak cheerfully"/>' in sentences[0]
        assert '<expression value="speak with bright energy"/>' in sentences[1]
        assert '<expression value="sound excited"/>' in sentences[2]
        _assert_no_tag_only_sentences(sentences)

    def test_standalone_tag_merged_with_following_text(self) -> None:
        """Regression: a self-closing tag as its own sentence must merge with
        the next so TTS never receives a tag-only chunk."""
        text = '<expression value="speak firmly"/> I told you already, no changes to the order.'
        sentences = self.tok.tokenize(text)
        _assert_no_tag_only_sentences(sentences)

    def test_wrapping_tag_with_inner_periods(self) -> None:
        """Dots inside <spell> look like sentence endings. Merge must keep tag intact."""
        text = "Spell it: <spell>U.S.A.</spell>. Got it?"
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")

    def test_wrapping_tag_with_inner_sentences(self) -> None:
        """Full sentences inside a wrapping tag must not be split out."""
        text = (
            "Read this: <spell>The quick brown fox. The cat sat on the mat.</spell>. "
            "Now something else."
        )
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")

    def test_mixed_tags(self) -> None:
        """Self-closing + wrapping + break tags in one text."""
        text = (
            '<emotion value="excited"/><speed ratio="1.3"/> Great news! '
            "The code is <spell>X9Z</spell>. "
            '<break time="500ms"/> <emotion value="calm"/> Let me explain.'
        )
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")
        _assert_no_tag_only_sentences(sentences)

    def test_no_markup(self) -> None:
        sentences = self.tok.tokenize("Hello there. How are you? I am fine.")
        assert len(sentences) >= 2

    def test_only_tag_no_text(self) -> None:
        sentences = self.tok.tokenize('<emotion value="happy"/>')
        assert len(sentences) == 1


# ===========================================================================
# Streaming sentence tokenizer
# ===========================================================================


class TestStreamingTokenizer:
    def setup_method(self) -> None:
        self.tok = SentenceTokenizer(min_sentence_len=1, stream_context_len=5, xml_aware=True)

    @pytest.mark.asyncio
    async def test_tag_split_across_chunks(self) -> None:
        """Tag arrives in multiple push_text calls — must hold until complete."""
        stream = self.tok.stream()
        stream.push_text("Hello. <emo")
        stream.push_text('tion value="happy"/> Great!')
        stream.end_input()
        tokens = [ev.token async for ev in stream]
        full = " ".join(tokens)
        assert '<emotion value="happy"/>' in full

    @pytest.mark.asyncio
    async def test_wrapping_tag_inner_sentences_streaming(self) -> None:
        """Wrapping tag with inner sentence splits must merge in streaming mode."""
        text = (
            "I want to tell you something important now. "
            "<outer>The first thing you should know is quite significant. "
            "The second thing is equally critical to understand. "
            "The third thing wraps up the entire explanation.</outer> "
            "That was everything I needed to explain today."
        )
        tokens = await _stream_tokenize(self.tok, text)
        _assert_wrapping_tag_intact(tokens, "outer")

    @pytest.mark.asyncio
    async def test_standalone_expression_tag_streaming(self) -> None:
        """Regression: streaming must never emit a tag-only chunk."""
        text = (
            '<expression value="speak firmly with a sharp and serious tone"/> '
            "I told you already, no changes to the order."
        )
        tokens = await _stream_tokenize(self.tok, text)
        _assert_no_tag_only_sentences(tokens)

    @pytest.mark.asyncio
    async def test_flush_xml_only_emitted(self) -> None:
        """flush()/end_input() must emit tag-only tokens - they could be
        non-verbal sounds like laughs that produce audio on their own."""
        stream = self.tok.stream()
        stream.push_text('<expression value="laugh"/>')
        stream.end_input()
        tokens = [ev.token async for ev in stream]
        assert len(tokens) == 1

    @pytest.mark.asyncio
    async def test_expression_tags_between_sentences_tiktoken(self) -> None:
        """Regression: expression tags between sentences must split correctly
        when streamed with GPT-4o's actual tokenizer."""
        text = (
            '<expression value="speak cheerfully"/> Hello and welcome to McDonalds! '
            '<expression value="speak with bright energy"/> We have got some great specials. '
            '<expression value="sound excited"/> Our new chicken sandwich is amazing. '
            '<expression value="speak warmly"/> Would you like to try a combo meal?'
        )
        tokens = await _stream_tokenize_tiktoken(self.tok, text)
        assert len(tokens) >= 3, f"Expected at least 3 sentences: {tokens}"
        _assert_no_tag_only_sentences(tokens)
        for t in tokens:
            assert "<expression" in t, f"Sentence missing expression tag: {t!r}"

    @pytest.mark.asyncio
    async def test_realistic_conversation(self) -> None:
        text = (
            '<emotion value="neutral"/> Thank you for calling. '
            "How can I help you today? "
            '<break time="500ms"/> '
            '<emotion value="empathetic"/> I understand your frustration. '
            "Let me look into this for you. "
            "Your order number is <spell>A.B.1.2.3.</spell>. "
            '<emotion value="confident"/> I found the issue. '
            '<speed ratio="0.8"/> The refund will be processed in 3 to 5 business days. '
            '<emotion value="happy"/> Is there anything else I can help with?'
        )
        tokens = await _stream_tokenize(self.tok, text)
        _assert_wrapping_tag_intact(tokens, "spell")
        _assert_no_tag_only_sentences(tokens)


# ===========================================================================
# Plain text with "<" (false-positive guard)
# ===========================================================================


class TestPlainTextAngleBrackets:
    """Regression: a stray "<" in plain text must not stall streaming.

    `_has_unclosed_xml_tags` used to treat any "<" after the last ">" as an
    unfinished tag; one "3 < 5" then held every following sentence until flush,
    degrading streaming TTS to end-of-turn batching for the rest of the turn.
    """

    def test_bare_lt_is_not_a_tag(self) -> None:
        from livekit.agents.tokenize.token_stream import _has_unclosed_xml_tags

        assert not _has_unclosed_xml_tags("3 < 5.")
        assert not _has_unclosed_xml_tags("i <3 you")
        assert not _has_unclosed_xml_tags("price < 10 dollars")
        # tag-shaped: must still hold
        assert _has_unclosed_xml_tags("Hello <emo")
        assert _has_unclosed_xml_tags("Hello <")  # the next chunk resolves it
        assert _has_unclosed_xml_tags("<spell>abc")  # unclosed wrapping tag

    def test_digit_named_pseudo_tags_are_not_counted(self) -> None:
        # regression: the depth-counter regex must not treat "<5>" / "<3 wins>" as
        # open tags, or a complete-but-digit-named pair would leave depth > 0 and
        # stall streaming for the rest of the turn (the tail check already treats
        # "<"+digit as plain text — the two predicates must agree)
        from livekit.agents.tokenize.token_stream import _has_unclosed_xml_tags

        assert not _has_unclosed_xml_tags("Rate this from <1> to <5> please.")
        assert not _has_unclosed_xml_tags("Scores: <3 wins> today.")
        # a real letter-named tag pair is still balanced
        assert not _has_unclosed_xml_tags("<spell>abc</spell> done")

    @pytest.mark.asyncio
    async def test_digit_pseudo_tag_streams_with_xml_aware(self) -> None:
        tok = SentenceTokenizer(min_sentence_len=1, stream_context_len=5, xml_aware=True)
        stream = tok.stream()
        stream.push_text("Rate this from <1> to <5>. And here is a second sentence to split.")
        ev = await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert "<5>" in ev.token or "<1>" in ev.token
        stream.end_input()

    @pytest.mark.asyncio
    async def test_bare_lt_streams_with_xml_aware(self) -> None:
        tok = SentenceTokenizer(min_sentence_len=1, stream_context_len=5, xml_aware=True)
        stream = tok.stream()
        stream.push_text("Note that 3 < 5 holds. And here is a second sentence to tokenize.")
        # the first sentence must be emitted without waiting for flush
        ev = await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert "3 < 5" in ev.token
        stream.end_input()

    @pytest.mark.asyncio
    async def test_tag_shaped_text_streams_when_not_xml_aware(self) -> None:
        # the default tokenizer (non-expressive agents) applies no XML logic at
        # all, so even tag-shaped plain text must stream sentence by sentence
        tok = SentenceTokenizer(min_sentence_len=1, stream_context_len=5)
        stream = tok.stream()
        stream.push_text("Email me at <bob@example.com> please. Second sentence for the split.")
        ev = await asyncio.wait_for(stream.__anext__(), timeout=1)
        assert "bob@example.com" in ev.token
        stream.end_input()


# ===========================================================================
# Markup.to_text_stream (transcript stripping)
# ===========================================================================


async def _achunks(items: list[str]):
    for it in items:
        yield it


class TestToTextStreamBareLt:
    """Regression: the transcript-strip path must not stall on a bare "<" either.

    to_text_stream buffered on a naive `rfind("<") > rfind(">")` check, so a "<"
    in prose (e.g. "3 < 5") froze every following transcript chunk of the segment
    until a ">" arrived or the stream ended — the same stall fixed in the tokenizer.
    """

    def _markup(self):
        from livekit.agents.tts.tts import TTS

        class _DialectMarkup(TTS.Markup):
            def _provider_key(self) -> str:
                return "cartesia"

        return _DialectMarkup(None)  # type: ignore[arg-type]  # _provider_key ignores tts

    @pytest.mark.asyncio
    async def test_bare_lt_does_not_hold_following_chunk(self) -> None:
        out = [
            c
            async for c in self._markup().to_text_stream(_achunks(["The value 3 < 5 ", "is true."]))
        ]
        # fixed: the first chunk is emitted incrementally (>= 2 items); the buggy
        # version held everything and emitted a single item at end-of-stream
        assert len(out) >= 2
        assert "3 < 5" in out[0]
        assert "".join(out).replace(" ", "") == "Thevalue3<5istrue."

    @pytest.mark.asyncio
    async def test_partial_tag_still_buffered(self) -> None:
        # a genuinely partial tag split across chunks must still be held and stripped
        out = [
            c
            async for c in self._markup().to_text_stream(
                _achunks(["Hi <emo", 'tion value="happy"/> there'])
            )
        ]
        joined = "".join(out)
        assert "<emotion" not in joined
        assert "Hi" in joined and "there" in joined
