"""Regression tests: sentence tokenizers must never split inside XML tags.

These tests cover the blingfire sentence tokenizer (both batch and streaming)
with TTS markup tags used in expressiveness mode (Cartesia, ElevenLabs).
"""

from __future__ import annotations

import pytest

from livekit.agents.tokenize.blingfire import SentenceTokenizer
from livekit.agents.tts.markup_utils import strip_xml_tags

# -- strip_xml_tags -----------------------------------------------------------


class TestStripXmlTags:
    def test_self_closing_tags(self) -> None:
        text = '<emotion value="happy"/> Hello there!'
        assert strip_xml_tags(text, ["emotion"]) == " Hello there!"

    def test_wrapping_tags(self) -> None:
        text = "Code: <spell>A7X9</spell>. Confirm?"
        assert strip_xml_tags(text, ["spell"]) == "Code: A7X9. Confirm?"

    def test_multiple_tags(self) -> None:
        text = '<speed ratio="1.5"/><emotion value="excited"/> Great news!'
        result = strip_xml_tags(text, ["speed", "emotion"])
        assert result == " Great news!"

    def test_break_tag(self) -> None:
        text = 'Hold on. <break time="1s"/> OK, ready.'
        assert strip_xml_tags(text, ["break"]) == "Hold on.  OK, ready."

    def test_phoneme_wrapping(self) -> None:
        text = 'Say <phoneme alphabet="cmu-arpabet" ph="M AE1">Madison</phoneme>.'
        assert strip_xml_tags(text, ["phoneme"]) == "Say Madison."

    def test_empty_tags_list(self) -> None:
        text = '<emotion value="happy"/> Hello'
        assert strip_xml_tags(text, []) == text

    def test_no_tags_in_text(self) -> None:
        assert strip_xml_tags("Hello world.", ["emotion"]) == "Hello world."


# -- Sentence tokenizer: batch mode ------------------------------------------


class TestSentenceTokenizerBatch:
    def setup_method(self) -> None:
        self.tok = SentenceTokenizer(min_sentence_len=1)

    def test_self_closing_tag_not_split(self) -> None:
        text = '<emotion value="happy"/> Hello! How are you?'
        sentences = self.tok.tokenize(text)
        assert len(sentences) >= 1
        assert '<emotion value="happy"/>' in sentences[0]

    def test_speed_ratio_decimal_not_split(self) -> None:
        text = '<speed ratio="1.5"/> This is fast. And this is slow.'
        sentences = self.tok.tokenize(text)
        # The decimal in 1.5 must not cause a sentence break inside the tag
        assert '<speed ratio="1.5"/>' in sentences[0]

    def test_multiple_emotion_tags(self) -> None:
        text = '<emotion value="sad"/> I am sorry. <emotion value="calm"/> Let us figure this out.'
        sentences = self.tok.tokenize(text)
        # Each emotion tag should stay with its following text
        for s in sentences:
            # no sentence should have an unclosed tag
            assert s.count("<") == s.count(">"), f"Unbalanced tags in: {s!r}"

    def test_spell_wrapping_tag_preserved(self) -> None:
        text = "The code is <spell>A7X9</spell>. Please confirm."
        sentences = self.tok.tokenize(text)
        # <spell>...</spell> must not be split across sentences
        full = " ".join(sentences)
        assert "<spell>A7X9</spell>" in full

    def test_spell_with_dots_not_split(self) -> None:
        text = "Spell it: <spell>U.S.A.</spell>. Got it?"
        sentences = self.tok.tokenize(text)
        # The dots inside <spell> must not cause sentence breaks
        full = " ".join(sentences)
        assert "<spell>" in full
        assert "</spell>" in full
        # both tags must be in the same sentence
        for s in sentences:
            if "<spell>" in s:
                assert "</spell>" in s, f"<spell> split across sentences: {sentences}"

    def test_phoneme_wrapping_tag_preserved(self) -> None:
        text = 'Say <phoneme alphabet="cmu-arpabet" ph="M AE1">Madison</phoneme>. It is a city.'
        sentences = self.tok.tokenize(text)
        for s in sentences:
            if "<phoneme" in s:
                assert "</phoneme>" in s, f"<phoneme> split: {sentences}"

    def test_break_tag(self) -> None:
        text = 'Hold on. <break time="1s"/> OK ready.'
        sentences = self.tok.tokenize(text)
        for s in sentences:
            if "<break" in s:
                assert "/>" in s, f"<break> tag split: {sentences}"


# -- Sentence tokenizer: streaming mode ---------------------------------------


class TestSentenceTokenizerStreaming:
    def setup_method(self) -> None:
        self.tok = SentenceTokenizer(min_sentence_len=1, stream_context_len=5)

    async def _stream_text(self, text: str) -> list[str]:
        stream = self.tok.stream()
        for char in text:
            stream.push_text(char)
        stream.end_input()

        tokens: list[str] = []
        async for ev in stream:
            tokens.append(ev.token)
        return tokens

    @pytest.mark.asyncio
    async def test_self_closing_tag_streaming(self) -> None:
        text = '<emotion value="happy"/> Hello! How are you today?'
        tokens = await self._stream_text(text)
        assert len(tokens) >= 1
        full = " ".join(tokens)
        assert '<emotion value="happy"/>' in full

    @pytest.mark.asyncio
    async def test_speed_decimal_streaming(self) -> None:
        text = '<speed ratio="1.5"/> Fast speech here. Normal speech here.'
        tokens = await self._stream_text(text)
        full = " ".join(tokens)
        assert '<speed ratio="1.5"/>' in full

    @pytest.mark.asyncio
    async def test_spell_wrapping_streaming(self) -> None:
        text = "Code: <spell>A7X9</spell>. Please confirm it."
        tokens = await self._stream_text(text)
        for t in tokens:
            if "<spell>" in t:
                assert "</spell>" in t, f"<spell> split in streaming: {tokens}"

    @pytest.mark.asyncio
    async def test_spell_with_dots_streaming(self) -> None:
        text = "Country: <spell>U.S.A.</spell>. Is that right?"
        tokens = await self._stream_text(text)
        for t in tokens:
            if "<spell>" in t:
                assert "</spell>" in t, f"<spell> split in streaming: {tokens}"

    @pytest.mark.asyncio
    async def test_multiple_emotions_streaming(self) -> None:
        text = '<emotion value="sad"/> Sorry about that. <emotion value="calm"/> We will fix it.'
        tokens = await self._stream_text(text)
        for t in tokens:
            assert t.count("<") == t.count(">"), f"Unbalanced tags in streaming: {t!r}"

    @pytest.mark.asyncio
    async def test_no_markup_unchanged(self) -> None:
        text = "Hello there. How are you? I am fine."
        tokens = await self._stream_text(text)
        full = " ".join(tokens)
        assert "Hello" in full
        assert "fine" in full

    @pytest.mark.asyncio
    async def test_incomplete_tag_buffered(self) -> None:
        """Simulate a tag arriving across multiple push_text calls."""
        stream = self.tok.stream()

        stream.push_text("Hello. <emo")
        stream.push_text('tion value="happy"/> Great!')
        stream.end_input()

        tokens: list[str] = []
        async for ev in stream:
            tokens.append(ev.token)

        full = " ".join(tokens)
        assert '<emotion value="happy"/>' in full
