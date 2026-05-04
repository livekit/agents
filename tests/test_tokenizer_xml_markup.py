"""Regression tests: sentence tokenizers must never split inside XML tags.

Covers blingfire sentence tokenizer (batch + streaming) with TTS markup tags
used in expressiveness mode (Cartesia, ElevenLabs).
"""

from __future__ import annotations

import pytest

from livekit.agents.tokenize.blingfire import SentenceTokenizer
from livekit.agents.tts.markup_utils import strip_xml_tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_tags_balanced(sentences: list[str], label: str = "") -> None:
    """Every sentence must have balanced < > and no orphaned closing tags."""
    for s in sentences:
        assert s.count("<") == s.count(">"), f"Unbalanced <> in {label}: {s!r}"


def _assert_wrapping_tag_intact(sentences: list[str], tag: str) -> None:
    """If a sentence has <tag>, it must also have </tag> (not split)."""
    for s in sentences:
        if f"<{tag}" in s and f"</{tag}>" not in s and "/>" not in s:
            pytest.fail(f"<{tag}> split across sentences: {sentences}")


async def _stream_tokenize(tok: SentenceTokenizer, text: str) -> list[str]:
    stream = tok.stream()
    for char in text:
        stream.push_text(char)
    stream.end_input()
    return [ev.token async for ev in stream]


async def _stream_tokenize_chunked(
    tok: SentenceTokenizer, text: str, chunk_size: int
) -> list[str]:
    """Push text in chunks of *chunk_size* characters (simulates LLM streaming)."""
    stream = tok.stream()
    for i in range(0, len(text), chunk_size):
        stream.push_text(text[i : i + chunk_size])
    stream.end_input()
    return [ev.token async for ev in stream]


# ===========================================================================
# strip_xml_tags
# ===========================================================================


class TestStripXmlTags:
    def test_self_closing(self) -> None:
        assert strip_xml_tags('<emotion value="happy"/> Hello!', ["emotion"]) == " Hello!"

    def test_wrapping(self) -> None:
        assert strip_xml_tags("Code: <spell>A7X9</spell>.", ["spell"]) == "Code: A7X9."

    def test_multiple_tags(self) -> None:
        text = '<speed ratio="1.5"/><emotion value="excited"/> Great!'
        assert strip_xml_tags(text, ["speed", "emotion"]) == " Great!"

    def test_break(self) -> None:
        assert strip_xml_tags('<break time="1s"/>', ["break"]) == ""

    def test_phoneme(self) -> None:
        text = '<phoneme alphabet="ipa" ph="x">word</phoneme>'
        assert strip_xml_tags(text, ["phoneme"]) == "word"

    def test_empty_tags_list(self) -> None:
        text = '<emotion value="happy"/> Hi'
        assert strip_xml_tags(text, []) == text

    def test_no_tags_in_text(self) -> None:
        assert strip_xml_tags("Hello world.", ["emotion"]) == "Hello world."

    def test_preserves_unrelated_tags(self) -> None:
        text = '<emotion value="happy"/> <custom>keep</custom>'
        assert strip_xml_tags(text, ["emotion"]) == " <custom>keep</custom>"

    def test_nested_content_preserved(self) -> None:
        text = "<spell>A.B.C.</spell> confirmed"
        assert strip_xml_tags(text, ["spell"]) == "A.B.C. confirmed"


# ===========================================================================
# Batch sentence tokenizer
# ===========================================================================


class TestBatchTokenizer:
    def setup_method(self) -> None:
        self.tok = SentenceTokenizer(min_sentence_len=1)

    # -- Self-closing tags (Cartesia primary tags) --

    def test_emotion_self_closing(self) -> None:
        text = '<emotion value="happy"/> Hello! How are you?'
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "emotion")
        assert '<emotion value="happy"/>' in sentences[0]

    def test_speed_decimal_in_attribute(self) -> None:
        text = '<speed ratio="1.5"/> This is fast. And this is normal.'
        sentences = self.tok.tokenize(text)
        assert '<speed ratio="1.5"/>' in sentences[0]

    def test_volume_decimal_in_attribute(self) -> None:
        text = '<volume ratio="0.5"/> Whispering now. Back to normal.'
        sentences = self.tok.tokenize(text)
        assert '<volume ratio="0.5"/>' in sentences[0]

    def test_break_between_sentences(self) -> None:
        text = 'First sentence. <break time="1.5s"/> Second sentence.'
        sentences = self.tok.tokenize(text)
        for s in sentences:
            if "<break" in s:
                assert "/>" in s, f"<break> split: {sentences}"

    def test_multiple_emotions_across_sentences(self) -> None:
        text = (
            '<emotion value="sad"/> I am sorry about that. '
            '<emotion value="calm"/> Let us figure this out together.'
        )
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "multi-emotion")

    def test_emotion_mid_sentence(self) -> None:
        text = 'I feel <emotion value="excited"/> really great today!'
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "mid-sentence")

    # -- Wrapping tags --

    def test_spell_simple(self) -> None:
        text = "The code is <spell>A7X9</spell>. Please confirm."
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")

    def test_spell_with_periods(self) -> None:
        text = "Spell it: <spell>U.S.A.</spell>. Got it?"
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")

    def test_spell_with_abbreviation(self) -> None:
        text = "The org is <spell>N.A.S.A.</spell>. They do space stuff."
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")

    def test_phoneme_simple(self) -> None:
        text = 'Say <phoneme alphabet="cmu-arpabet" ph="M AE1">Madison</phoneme>. It is a city.'
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "phoneme")

    def test_phoneme_with_ipa(self) -> None:
        text = 'Pronounce <phoneme alphabet="ipa" ph="ˈæktʃuəli">actually</phoneme>. Easy right?'
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "phoneme")

    # -- Multiple tags in one text --

    def test_mixed_tags(self) -> None:
        text = (
            '<emotion value="excited"/><speed ratio="1.3"/> Great news! '
            "The code is <spell>X9Z</spell>. "
            '<break time="500ms"/> <emotion value="calm"/> Let me explain.'
        )
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "mixed")
        _assert_wrapping_tag_intact(sentences, "spell")

    # -- Edge cases --

    def test_no_markup(self) -> None:
        text = "Hello there. How are you? I am fine."
        sentences = self.tok.tokenize(text)
        assert len(sentences) >= 2

    def test_only_tag_no_text(self) -> None:
        text = '<emotion value="happy"/>'
        sentences = self.tok.tokenize(text)
        assert len(sentences) == 1
        assert sentences[0].strip() == '<emotion value="happy"/>'

    def test_tag_with_many_attributes(self) -> None:
        text = '<phoneme alphabet="cmu-arpabet" ph="P R AH0 N AH0 N S IY EY1 SH AH0 N">pronunciation</phoneme> is key.'
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "phoneme")

    def test_adjacent_wrapping_tags(self) -> None:
        text = "<spell>A.B.</spell><spell>C.D.</spell>. Done."
        sentences = self.tok.tokenize(text)
        _assert_wrapping_tag_intact(sentences, "spell")

    def test_unicode_text_with_tags(self) -> None:
        text = '<emotion value="happy"/> こんにちは。元気ですか？'
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "unicode")
        assert '<emotion value="happy"/>' in sentences[0]

    def test_emoji_adjacent_to_tag(self) -> None:
        text = '<emotion value="excited"/> 🎉 Great news! You won.'
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "emoji")

    def test_long_text_many_sentences(self) -> None:
        text = (
            '<emotion value="neutral"/> Welcome to the service. '
            "My name is Alex. "
            '<emotion value="happy"/> I am glad to help you today. '
            "What can I do for you? "
            '<emotion value="curious"/> Tell me more about your issue. '
            "I will do my best. "
            '<break time="1s"/> '
            '<emotion value="calm"/> Take your time.'
        )
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "long")

    def test_exclamation_and_question_marks(self) -> None:
        text = '<emotion value="excited"/> Wow! Really? <emotion value="happy"/> That is amazing!'
        sentences = self.tok.tokenize(text)
        _assert_tags_balanced(sentences, "punctuation")

    def test_decimal_outside_tag_still_splits(self) -> None:
        """Decimal in regular text (not in a tag) should still allow sentence splitting."""
        text = "The price is 1.5 dollars. That is cheap."
        sentences = self.tok.tokenize(text)
        # Should split at the period after "dollars"
        assert len(sentences) >= 2


# ===========================================================================
# Streaming sentence tokenizer
# ===========================================================================


class TestStreamingTokenizer:
    def setup_method(self) -> None:
        self.tok = SentenceTokenizer(min_sentence_len=1, stream_context_len=5)

    # -- Self-closing tags --

    @pytest.mark.asyncio
    async def test_emotion_streaming(self) -> None:
        text = '<emotion value="happy"/> Hello! How are you today?'
        tokens = await _stream_tokenize(self.tok, text)
        full = " ".join(tokens)
        assert '<emotion value="happy"/>' in full
        _assert_tags_balanced(tokens, "stream-emotion")

    @pytest.mark.asyncio
    async def test_speed_decimal_streaming(self) -> None:
        text = '<speed ratio="1.5"/> Fast speech here. Normal speech here.'
        tokens = await _stream_tokenize(self.tok, text)
        full = " ".join(tokens)
        assert '<speed ratio="1.5"/>' in full

    @pytest.mark.asyncio
    async def test_break_streaming(self) -> None:
        text = 'Wait. <break time="2s"/> OK continue now.'
        tokens = await _stream_tokenize(self.tok, text)
        _assert_tags_balanced(tokens, "stream-break")

    # -- Wrapping tags --

    @pytest.mark.asyncio
    async def test_spell_streaming(self) -> None:
        text = "Code: <spell>A7X9</spell>. Please confirm it."
        tokens = await _stream_tokenize(self.tok, text)
        _assert_wrapping_tag_intact(tokens, "spell")

    @pytest.mark.asyncio
    async def test_spell_with_dots_streaming(self) -> None:
        text = "Country: <spell>U.S.A.</spell>. Is that right?"
        tokens = await _stream_tokenize(self.tok, text)
        _assert_wrapping_tag_intact(tokens, "spell")

    @pytest.mark.asyncio
    async def test_phoneme_streaming(self) -> None:
        text = 'Say <phoneme alphabet="cmu-arpabet" ph="M AE1">Madison</phoneme>. Nice city.'
        tokens = await _stream_tokenize(self.tok, text)
        _assert_wrapping_tag_intact(tokens, "phoneme")

    # -- Multiple tags --

    @pytest.mark.asyncio
    async def test_multiple_emotions_streaming(self) -> None:
        text = '<emotion value="sad"/> Sorry. <emotion value="calm"/> We will fix it.'
        tokens = await _stream_tokenize(self.tok, text)
        _assert_tags_balanced(tokens, "stream-multi")

    # -- Chunk boundary edge cases --

    @pytest.mark.asyncio
    async def test_tag_split_across_chunks(self) -> None:
        """Tag arrives in multiple push_text calls."""
        stream = self.tok.stream()
        stream.push_text("Hello. <emo")
        stream.push_text('tion value="happy"/> Great!')
        stream.end_input()
        tokens = [ev.token async for ev in stream]
        full = " ".join(tokens)
        assert '<emotion value="happy"/>' in full

    @pytest.mark.asyncio
    async def test_closing_tag_split_across_chunks(self) -> None:
        """Closing tag arrives across chunk boundary."""
        stream = self.tok.stream()
        stream.push_text("Code: <spell>A.B.")
        stream.push_text("C.</spell>. Done.")
        stream.end_input()
        tokens = [ev.token async for ev in stream]
        _assert_wrapping_tag_intact(tokens, "spell")

    @pytest.mark.asyncio
    async def test_attribute_split_across_chunks(self) -> None:
        """Attribute value split mid-decimal."""
        stream = self.tok.stream()
        stream.push_text('<speed ratio="1.')
        stream.push_text('5"/> Go fast. Then slow.')
        stream.end_input()
        tokens = [ev.token async for ev in stream]
        full = " ".join(tokens)
        assert '<speed ratio="1.5"/>' in full

    @pytest.mark.asyncio
    async def test_small_chunks_char_by_char(self) -> None:
        """Simulate worst-case: one character at a time."""
        text = '<emotion value="sad"/> I am sorry. <spell>S.O.S.</spell>. Help!'
        tokens = await _stream_tokenize(self.tok, text)
        _assert_tags_balanced(tokens, "char-by-char")
        _assert_wrapping_tag_intact(tokens, "spell")

    @pytest.mark.asyncio
    async def test_large_chunks(self) -> None:
        """Push in large chunks (typical LLM token size ~4-8 chars)."""
        text = (
            '<emotion value="excited"/> Amazing news! '
            "The confirmation code is <spell>X.Y.Z.</spell>. "
            '<break time="1s"/> <emotion value="calm"/> Please verify.'
        )
        tokens = await _stream_tokenize_chunked(self.tok, text, chunk_size=7)
        _assert_tags_balanced(tokens, "large-chunks")
        _assert_wrapping_tag_intact(tokens, "spell")

    # -- No markup --

    @pytest.mark.asyncio
    async def test_no_markup_streaming(self) -> None:
        text = "Hello there. How are you? I am fine."
        tokens = await _stream_tokenize(self.tok, text)
        full = " ".join(tokens)
        assert "Hello" in full
        assert "fine" in full

    # -- Unicode --

    @pytest.mark.asyncio
    async def test_unicode_streaming(self) -> None:
        text = '<emotion value="happy"/> Bonjour. Comment ça va?'
        tokens = await _stream_tokenize(self.tok, text)
        _assert_tags_balanced(tokens, "unicode-stream")

    @pytest.mark.asyncio
    async def test_chinese_with_tags(self) -> None:
        text = '<emotion value="neutral"/> 你好。你今天怎么样？'
        tokens = await _stream_tokenize(self.tok, text)
        _assert_tags_balanced(tokens, "chinese-stream")

    # -- Realistic expressiveness example --

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
        _assert_tags_balanced(tokens, "realistic")
        _assert_wrapping_tag_intact(tokens, "spell")
