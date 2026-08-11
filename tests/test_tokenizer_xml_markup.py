"""Regression tests: sentence tokenizers must handle XML markup correctly.

Covers blingfire sentence tokenizer (batch + streaming) with TTS markup tags
used in expressive mode (Cartesia, ElevenLabs, Inworld).
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.tokenize.blingfire import SentenceTokenizer
from livekit.agents.tokenize.token_stream import _XML_TAG_RE
from livekit.agents.tts.markup_utils import extract_and_strip

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
# extract_and_strip
# ===========================================================================


def _strip(text: str, tags: list[str]) -> str:
    return extract_and_strip(text, xml_tags=tags)[0]


class TestExtractAndStrip:
    def test_self_closing(self) -> None:
        assert _strip('<emotion value="happy"/> Hello!', ["emotion"]) == " Hello!"

    def test_wrapping_preserves_content(self) -> None:
        assert _strip("<spell>A.B.C.</spell> confirmed", ["spell"]) == "A.B.C. confirmed"

    def test_preserves_unrelated_tags(self) -> None:
        text = '<emotion value="happy"/> <custom>keep</custom>'
        assert _strip(text, ["emotion"]) == " <custom>keep</custom>"

    def test_empty_tags_list(self) -> None:
        text = '<emotion value="happy"/> Hi'
        assert _strip(text, []) == text

    def test_square_brackets_are_never_markup(self) -> None:
        # only XML is markup here — bracket spans reach transcripts as prose/markdown
        text = 'Press [Enter] <emotion value="happy"/> to open [the docs](https://lk.io)'
        assert _strip(text, ["emotion"]) == "Press [Enter] to open [the docs](https://lk.io)"

    def test_removal_leaves_a_single_space(self) -> None:
        # a tag between two spaces must not leave both behind: the transcript would show
        # a double space after the punctuation the tag followed
        assert _strip('Right. <emotion value="sad"/> Anyway.', ["emotion"]) == "Right. Anyway."
        # the space survives when it is the only separator between the words
        assert _strip('Right.<emotion value="sad"/> Anyway.', ["emotion"]) == "Right. Anyway."
        assert _strip('Right. <emotion value="sad"/>Anyway.', ["emotion"]) == "Right. Anyway."
        # trailing: the space may separate words still streaming in, so it is kept
        assert _strip('Right. <emotion value="sad"/>', ["emotion"]) == "Right. "
        # a wrapping tag keeps its content, so nothing is doubled to begin with
        assert _strip("a <spell>b</spell> c", ["spell"]) == "a b c"
        # a lone closing tag is a removal too
        assert _strip("a </spell> b", ["spell"]) == "a b"
        # newlines are structure, not a doubled separator
        assert _strip('a\n<emotion value="sad"/>\nb', ["emotion"]) == "a\n\nb"

    def test_reports_stripped_tags(self) -> None:
        clean, tags = extract_and_strip(
            '<emotion value="happy"/>hi <spell>A7</spell>', xml_tags=["emotion", "spell"]
        )
        assert clean == "hi A7"
        assert tags == [("emotion", "happy"), ("spell", "A7")]


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
        # this branch instructs the unified expr dialect; convert_markup lowers it to
        # xAI's native syntax (see tests/test_expr_markup.py)
        assert '<expr type="sound" label="breath"/>' in instr
        assert '<expr type="prosody" label="' in instr

    def test_strip_removes_inline_keeps_wrapping_inner(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = 'So I walked in and <break time="500ms"/> there it was. <sound value="laugh"/> <whisper>a secret</whisper> <emphasis>wow</emphasis>.'
        clean, tags = pf.split_all_markup(raw)
        # inline sounds/pauses removed entirely; wrapping tags keep their inner text
        assert "<break" not in clean and "<sound" not in clean and "laugh" not in clean
        assert "<whisper>" not in clean and "a secret" in clean and "wow" in clean
        types = [(t["type"], t["value"]) for t in tags]
        assert ("break", "500ms") in types and ("sound", "laugh") in types
        assert ("whisper", "a secret") in types and ("emphasis", "wow") in types

    def test_emotion_wrapping_tags_stripped_inner_kept(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = "<happy>Great to hear from you!</happy> <sad>I'm sorry about that.</sad>"
        clean, tags = pf.split_all_markup(raw)
        # emotion is the tag name; delimiters removed, spoken words preserved
        assert "<happy>" not in clean and "</sad>" not in clean
        assert "Great to hear from you!" in clean and "I'm sorry about that." in clean
        types = [(t["type"], t["value"]) for t in tags]
        assert ("happy", "Great to hear from you!") in types
        assert ("sad", "I'm sorry about that.") in types

    def test_every_documented_tag_is_strippable(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # every prosody label the expr instructions offer must be in _XAI_TAGS,
        # or a hallucinated native form would leak into the user-visible transcript
        instructions = pf.llm_instructions("xai")
        assert instructions is not None
        for tag in pf._XAI_WRAPPING:
            assert tag in instructions, f"{tag} not documented"
            assert tag in pf._XAI_TAGS
            clean, _ = pf.split_all_markup(f"<{tag}>hello there</{tag}>")
            assert clean.strip() == "hello there", f"{tag} not stripped: {clean!r}"

    def test_emotion_tags_stripped_though_unprompted(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # emotion tags are no longer instructed, but stay in _XAI_TAGS so a stray one is
        # stripped from the transcript rather than leaking to the user
        for tag in pf._XAI_EMOTIONS:
            assert tag in pf._XAI_TAGS
            clean, _ = pf.split_all_markup(f"<{tag}>hello there</{tag}>")
            assert clean.strip() == "hello there", f"{tag} not stripped: {clean!r}"

    def test_documented_inline_tags_present(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # nonverbals from xAI's docs, incl. the ones the user called out; documented in
        # the expr sound-label vocabulary (lowered to [NAME] for the TTS in convert_markup)
        instructions = pf.llm_instructions("xai")
        assert instructions is not None
        for name in ("tsk", "lip-smack", "tongue-click", "chuckle", "giggle", "hum-tune"):
            assert name in pf._XAI_INLINE
            assert name in instructions

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
        clean, _ = pf.split_all_markup(raw)
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


class TestFishAudioDialect:
    """Fish's native dialect is square brackets: expr markers lower to [very EMOTION] /
    [SOUND] / [break] / [long-break] / [emphasis] word for the TTS, and the transcript
    strips both the expr markers and any hallucinated native form (XML or brackets)."""

    def test_llm_instructions_registered(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        instr = pf.llm_instructions("fishaudio")
        # non-None is what the expressive gate keys on
        assert instr is not None
        # discrete emotion vocabulary, Fish's sounds, and the emphasis-only prosody
        for emotion in pf._FISHAUDIO_EMOTIONS:
            assert emotion in instr
        assert '<expr type="sound" label="laughing"/>' in instr
        assert "clear throat" in instr
        assert '<expr type="prosody" label="emphasis">' in instr

    def test_convert_expr_to_fish_brackets(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = (
            '<expr type="expression" label="excited"/> We won! '
            '<expr type="sound" label="laughing"/> <expr type="break" label="500ms"/> '
            'That was <expr type="prosody" label="emphasis">really</expr> close. '
            '<expr type="break" label="2s"/>'
        )
        assert pf.convert_markup("fishaudio", raw) == (
            "[very excited] We won! [laughing] [break] "
            "That was [emphasis] really close. [long-break]"
        )

    def test_convert_expression_intensified_once(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # "very" is prepended so the emotion lands harder, but never doubled
        assert pf.convert_markup("fishaudio", '<expr type="expression" label="sad"/>') == (
            "[very sad]"
        )
        assert pf.convert_markup("fishaudio", '<expression value="very sad"/>') == "[very sad]"

    def test_convert_sound_alias(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # other providers advertise "laugh"; a hallucinated one still lowers to a
        # sound Fish renders
        assert pf.convert_markup("fishaudio", '<expr type="sound" label="laugh"/>') == "[laughing]"
        assert pf.convert_markup("fishaudio", '<expr type="sound" label="sigh"/>') == "[sighing]"
        assert pf.convert_markup("fishaudio", '<expr type="sound" label="cry"/>') == "[sobbing]"

    def test_convert_tone_wrap_to_prefix_marker(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # Fish tones have no closing form: the wrap lowers to a prefix marker
        raw = '<expr type="prosody" label="whispering">don\'t tell anyone</expr> okay?'
        assert pf.convert_markup("fishaudio", raw) == "[whispering] don't tell anyone okay?"
        # a self-closing tone (taught as wrapping, but Fish's native form is a
        # prefix anyway) is salvaged rather than dropped
        assert pf.convert_markup("fishaudio", '<expr type="prosody" label="soft"/> hey') == (
            "[soft] hey"
        )
        # labels outside the tone vocabulary still unwrap to their words
        raw = '<expr type="prosody" label="like a pirate">ahoy</expr>'
        assert pf.convert_markup("fishaudio", raw) == "ahoy"

    def test_tone_wrap_stripped_from_transcript(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = '<expr type="prosody" label="shouting">We won the whole thing!</expr>'
        clean, tags = pf.split_all_markup(raw)
        assert clean == "We won the whole thing!"
        assert ("prosody", "shouting") in [(t["type"], t["value"]) for t in tags]

    def test_new_emotions_advertised_and_mapped(self) -> None:
        from livekit.agents.tts import _provider_format as pf
        from livekit.agents.tts._mood import match_mood

        instr = pf.llm_instructions("fishaudio")
        assert instr is not None
        for tone in pf._FISHAUDIO_TONES:
            assert tone in instr
        # every advertised emotion resolves to a real mood, not the fallback —
        # lk.expression consumers get a meaningful enum for the whole vocabulary
        for emotion in pf._FISHAUDIO_EMOTIONS:
            assert match_mood(emotion, fallback=None) is not None, emotion

    def test_split_markup_strips_expr(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        raw = (
            '<expr type="expression" label="empathetic"/> That sounds '
            '<expr type="prosody" label="emphasis">really</expr> hard. '
            '<expr type="sound" label="clear throat"/>'
        )
        clean, tags = pf.split_all_markup(raw)
        assert clean.strip() == "That sounds really hard."
        types = [(t["type"], t["value"]) for t in tags]
        assert ("expression", "empathetic") in types
        assert ("prosody", "emphasis") in types
        assert ("sound", "clear throat") in types

    def test_hallucinated_native_markup_never_leaks(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # hallucinated fish-native XML must vanish from the transcript. Bracket cues
        # never reach the sinks from the LLM (it only writes expr) — natives produced
        # by convert_markup are removed on the aligned path by drop_bracket_cues —
        # so brackets here are prose and survive, matching the other providers.
        raw = '<expression value="happy"/> Hey there <emphasis>friend</emphasis>'
        clean, _ = pf.split_all_markup(raw)
        assert "<" not in clean
        assert "Hey" in clean and "there" in clean and "friend" in clean
        # and the same native XML still converts for the TTS
        assert pf.convert_markup("fishaudio", raw) == "[very happy] Hey there [emphasis] friend"

    def test_steering_filters_sounds_and_examples(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # everything off: the Sounds section and any example demonstrating a sound
        # are omitted entirely, not advertised and then revoked
        instr = pf.llm_instructions("fishaudio", {"nonverbal_sounds": False})
        assert instr is not None
        assert "laughing" not in instr and "clear throat" not in instr
        assert "Examples:" in instr  # the sound-free example survives

        # sparse opt-out: removing reflex sounds keeps the laugh family
        instr = pf.llm_instructions("fishaudio", {"nonverbal_sounds": {"reflex_sounds": False}})
        assert instr is not None
        assert "laughing" in instr and "clear throat" not in instr

    def test_supported_nonverbals(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # the queryable capabilities matrix: every Fish sound governed by one field
        assert pf.supported_nonverbals("fishaudio") == {
            "laughing": ["laughing", "chuckling"],
            "breathing": ["gasping"],
            "sighing": ["sighing"],
            "crying": ["sobbing"],
            "vocalizing": ["groaning"],
            "reflex_sounds": ["clear throat", "yawning"],
        }
        governed = [
            label for labels in pf._NONVERBAL_SOUND_LABELS["fishaudio"].values() for label in labels
        ]
        assert sorted(governed) == sorted(pf._FISHAUDIO_SOUNDS)

    def test_register_rule_in_shared_preamble(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # register inference is provider-neutral: every markup-capable provider's
        # block carries the rule via the shared preamble, not just fish
        for provider in ("fishaudio", "inworld", "xai", "cartesia"):
            instr = pf.llm_instructions(provider)
            assert instr is not None
            assert "REGISTER of the moment" in instr, provider

    def test_register_supplement_matches_steering(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        default = pf.llm_instructions("fishaudio")
        assert default is not None
        assert "Laughter belongs only" in default
        assert "Save fillers for relaxed moments" in default

        # an opted-out concept must be absent from the ENTIRE block — not even
        # mentioned prohibitively, or the LLM receives contradictory directions
        composed = pf.llm_instructions(
            "fishaudio", {"nonverbal_sounds": False, "disfluencies": False}
        )
        assert composed is not None
        assert "laugh" not in composed.lower()
        assert "filler" not in composed.lower()
        assert "Um, uh" not in composed

    def test_nonverbal_sounds_accepts_bool(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # False disables the whole vocabulary; True (like omission) keeps it all
        off = pf.llm_instructions("fishaudio", {"nonverbal_sounds": False})
        on = pf.llm_instructions("fishaudio", {"nonverbal_sounds": True})
        default = pf.llm_instructions("fishaudio")
        assert off is not None and on is not None and default is not None
        assert 'type="sound"' not in off and "laughing" not in off
        for instr in (on, default):
            assert ", ".join(pf._FISHAUDIO_SOUNDS) in instr
        assert pf._allowed_sounds("fishaudio", {"nonverbal_sounds": False}) == []
        assert pf._allowed_sounds("fishaudio", {"nonverbal_sounds": True}) == pf._FISHAUDIO_SOUNDS

    def test_all_on_forms_render_like_omission(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # equivalent configurations must produce identical instructions: the
        # explicit all-on forms add no sound guidance the default doesn't have
        for provider in ("fishaudio", "inworld", "xai"):
            for steering in ({"nonverbal_sounds": True}, {"nonverbal_sounds": {}}):
                assert pf.steering_instructions(provider, steering) == "", (provider, steering)
                assert pf.llm_instructions(provider, steering) == pf.llm_instructions(provider), (
                    provider,
                    steering,
                )
        # all-off leaves nothing to guide; the vocabulary removal happens in
        # llm_instructions, not here
        assert pf.steering_instructions("fishaudio", {"nonverbal_sounds": False}) == ""
        # a genuine opt-out still draws guidance about what remains — and only
        # about what remains
        partial = pf.steering_instructions("fishaudio", {"nonverbal_sounds": {"laughing": False}})
        assert "clear-throat" in partial
        assert "laugh" not in partial.lower()

    def test_nonverbal_dict_is_sparse(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # a dict is a sparse opt-out: omitted categories stay enabled, so
        # {"laughing": False} removes laughter and nothing else
        steering = {"nonverbal_sounds": {"laughing": False}}
        assert pf._allowed_sounds("fishaudio", steering) == [
            "clear throat",
            "sighing",
            "gasping",
            "groaning",
            "yawning",
            "sobbing",
        ]
        inworld = pf._allowed_sounds("inworld", steering)
        assert "laugh" not in inworld
        for kept in ("sigh", "breathe", "clear throat", "cough", "yawn"):
            assert kept in inworld, kept
        # xai's laugh-family prosody is governed by the same field
        assert "laugh-speak" not in pf._allowed_prosody("xai", steering)
        assert "whisper" in pf._allowed_prosody("xai", steering)

    def test_steering_is_sparse_over_default(self) -> None:
        from livekit.agents.voice.agent_session import (
            DEFAULT_EXPRESSIVE_OPTIONS,
            resolve_expressive_options,
        )

        # bare default: fillers on, no sound filtering
        r = resolve_expressive_options(
            {"speech_steering": {}},
            provider_key="fishaudio",
            default=DEFAULT_EXPRESSIVE_OPTIONS,
        )["speech_steering"]
        assert r["disfluencies"] is True
        assert "nonverbal_sounds" not in r
        # a composed agent: both taken away, regardless of context
        r2 = resolve_expressive_options(
            {"speech_steering": {"nonverbal_sounds": False, "disfluencies": False}},
            provider_key="fishaudio",
            default=DEFAULT_EXPRESSIVE_OPTIONS,
        )["speech_steering"]
        assert r2["nonverbal_sounds"] is False
        assert r2["disfluencies"] is False

    def test_disfluent_examples_follow_steering(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        # fillers are few-shotted only while disfluencies are enabled, so the
        # examples never contradict the "no fillers" delivery guideline
        on = pf.llm_instructions("fishaudio", {"disfluencies": True})
        off = pf.llm_instructions("fishaudio", {"disfluencies": False})
        default = pf.llm_instructions("fishaudio")  # disfluencies default on
        assert on is not None and off is not None and default is not None
        assert "Um, uh" in on and "Um, uh" in default
        assert "Um, uh" not in off and ", um," not in off

    def test_normalize_closes_unclosed_tags(self) -> None:
        from livekit.agents.tts import _provider_format as pf

        assert pf.normalize_markup("fishaudio", '<expr type="sound" label="laughing"> hi') == (
            '<expr type="sound" label="laughing"/> hi'
        )
        assert pf.normalize_markup("fishaudio", '<expression value="happy"> hi') == (
            '<expression value="happy"/> hi'
        )


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
# Universal transcript stripping (provider-agnostic, used by the transcript sinks)
# ===========================================================================


class TestUniversalMarkupStrip:
    """The transcript sinks strip downstream without knowing the provider, so they remove
    the union of every provider's XML tags — but never square brackets, which reach the
    transcript as markdown/prose. See split_all_markup / TranscriptMarkupStripper."""

    def test_split_all_markup_across_providers(self) -> None:
        from livekit.agents.tts._provider_format import split_all_markup

        # Cartesia <emotion> and Inworld/xAI <expression>/<sound> all strip regardless of
        # which provider produced them; a [bracket] span is left for the reader
        clean, tags = split_all_markup(
            '<emotion value="happy"/>Hi <expression value="warm"/>there '
            '<sound value="giggle"/>[pause] friend'
        )
        assert clean == "Hi there [pause] friend"
        types = [(t["type"], t["value"]) for t in tags]
        assert ("emotion", "happy") in types
        assert ("expression", "warm") in types
        assert ("sound", "giggle") in types
        assert ("", "pause") not in types

    def test_expression_attribute_shape(self) -> None:
        from livekit.agents.tts._provider_format import expression_attribute, split_all_markup

        _, tags = split_all_markup('<emotion value="sad"/>oh no')
        attr = expression_attribute(tags)
        assert attr == {"lk.expression": '{"expression":"sad","mood":"sad"}'}

        # no expression/emotion tag -> no attribute
        _, tags = split_all_markup('<break time="1s"/>hi')
        assert expression_attribute(tags) is None

    def test_streaming_stripper_holds_partial_tags(self) -> None:
        from livekit.agents.tts._provider_format import TranscriptMarkupStripper

        s = TranscriptMarkupStripper()
        # a tag split across pushes is held until it closes, never emitted half-stripped
        out = s.push("Hi <emo")
        out += s.push('tion value="happy"/> the')
        out += s.push("re")
        out += s.flush()
        assert "<emotion" not in out
        assert out.replace(" ", "") == "Hithere"
        assert s.expression_attribute() == {
            "lk.expression": '{"expression":"happy","mood":"happy"}'
        }

    def test_streaming_stripper_markdown_link_survives(self) -> None:
        from livekit.agents.tts._provider_format import TranscriptMarkupStripper

        s = TranscriptMarkupStripper()
        # an unclosed "[" must not stall the chunk (brackets aren't markup), and the link
        # must arrive intact rather than collapsed to its (url) tail
        first = s.push("Read [the docs](https:")
        assert first == "Read [the docs](https:"
        rest = s.push("//docs.livekit.io) now.") + s.flush()
        assert first + rest == "Read [the docs](https://docs.livekit.io) now."

    def test_streaming_stripper_bare_lt_not_stalled(self) -> None:
        from livekit.agents.tts._provider_format import TranscriptMarkupStripper

        s = TranscriptMarkupStripper()
        # a bare "<" in prose must not freeze the following chunk
        first = s.push("The value 3 < 5 ")
        assert "3 < 5" in first
        rest = s.push("is true.") + s.flush()
        assert (first + rest).replace(" ", "") == "Thevalue3<5istrue."
