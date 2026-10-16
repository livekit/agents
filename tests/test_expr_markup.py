"""Tests for the LiveKit expression marker (expr) dialect.

The LLM emits a single marker tag — ``<expr type="..." label="..."/>`` (self-closing
for expression/break/sound, wrapping for prosody/spell) — and the framework lowers it
to each provider's native markup before synthesis while stripping it from transcripts.
The syntax is shared, but the kinds and label vocabularies are per provider: each
provider's instruction block advertises only what that provider supports.
"""

from __future__ import annotations

import pytest

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.tts._provider_format import (
    TranscriptMarkupStripper,
    convert_markup,
    expression_attribute,
    llm_instructions,
    normalize_markup,
    split_all_markup,
    strip_all_markup,
    strip_expr_markup,
)

pytestmark = pytest.mark.unit

# Gemini-flavored turn: free-form style labels only, lifted out into speech_metadata
GEMINI_TURN = (
    '<expr type="expression" label="Thoughtful, Quiet, American accent"/> Sienna? '
    '<expr type="expression" label="Wistful, American accent"/> '
    "What's on your mind, Comanchero?"
)

# Inworld-flavored turn: free-form expression + sound + break
JOKE = (
    '<expr type="expression" label="say playfully"/> Why did the burger go to the gym? '
    '<expr type="break" label="500ms"/> Because it wanted better buns! '
    '<expr type="sound" label="laugh"/>'
)

# ---------------------------------------------------------------------------
# convert_markup: expr -> xAI (sounds, breaks, wrapping prosody; no expression)
# ---------------------------------------------------------------------------


def test_convert_expr_xai() -> None:
    text = (
        'So I walked in and <expr type="break" label="500ms"/> there it was! '
        '<expr type="sound" label="laugh"/> '
        '<expr type="prosody" label="whisper">It was a secret the whole time.</expr>'
    )
    assert convert_markup("xai", text) == (
        "So I walked in and [pause] there it was! [laugh] "
        "<whisper>It was a secret the whole time.</whisper>"
    )


def test_convert_expr_xai_break_durations() -> None:
    assert convert_markup("xai", '<expr type="break" label="50ms"/>') == "[pause]"
    assert convert_markup("xai", '<expr type="break" label="2s"/>') == "[long-pause]"


def test_convert_expr_xai_sound_alias() -> None:
    # tolerance: an Inworld-style "breathe" label maps to xAI's native [breath] cue
    assert convert_markup("xai", '<expr type="sound" label="breathe"/>') == "[breath]"


def test_convert_expr_xai_prosody_multiword_label() -> None:
    # multi-word labels normalize to xAI's hyphenated tag names
    text = '<expr type="prosody" label="higher pitch">no way</expr>'
    assert convert_markup("xai", text) == "<higher-pitch>no way</higher-pitch>"


def test_convert_expr_xai_prosody_unknown_label_unwraps() -> None:
    text = '<expr type="prosody" label="like a pirate">ahoy there</expr>'
    assert convert_markup("xai", text) == "ahoy there"


def test_convert_expr_xai_drops_expression() -> None:
    # xAI has no free-form delivery descriptions; a hallucinated expression marker is
    # dropped from the audio path (it still surfaces in transcript tags)
    text = '<expr type="expression" label="say playfully"/> Hello!'
    assert convert_markup("xai", text) == " Hello!"


# ---------------------------------------------------------------------------
# convert_markup: expr -> Inworld (free-form expression, its sound list, breaks)
# ---------------------------------------------------------------------------


def test_convert_expr_inworld() -> None:
    # expression/sound lower to Inworld's bracket syntax; <break> stays native SSML
    assert convert_markup("inworld", JOKE) == (
        "[say playfully] Why did the burger go to the gym? "
        '<break time="500ms"/> Because it wanted better buns! [laugh]'
    )


def test_convert_expr_inworld_stray_prosody_becomes_expression_hint() -> None:
    text = '<expr type="prosody" label="whisper">keep it secret</expr>'
    assert convert_markup("inworld", text) == "[whisper]keep it secret"


# ---------------------------------------------------------------------------
# convert_markup: expr -> Cartesia (discrete emotions, breaks, spell; no sounds)
# ---------------------------------------------------------------------------


def test_convert_expr_cartesia() -> None:
    text = (
        '<expr type="expression" label="excited"/> We won! '
        '<expr type="break" label="1s"/> <expr type="sound" label="laugh"/> Unbelievable.'
    )
    # expression -> <emotion>, break stays, sound is dropped (no Cartesia support) —
    # without leaving the space it sat between behind as a doubled separator
    assert convert_markup("cartesia", text) == (
        '<emotion value="excited"/> We won! <break time="1s"/> Unbelievable.'
    )


def test_convert_expr_cartesia_spell() -> None:
    text = 'Your code is <expr type="spell">A7X9</expr>.'
    assert convert_markup("cartesia", text) == "Your code is <spell>A7X9</spell>."


def test_convert_expr_spell_unwraps_elsewhere() -> None:
    # spell is Cartesia-only; other providers keep the characters, drop the marker
    text = 'Your code is <expr type="spell">A7X9</expr>.'
    assert convert_markup("xai", text) == "Your code is A7X9."
    assert convert_markup("inworld", text) == "Your code is A7X9."


def test_convert_expr_cartesia_prosody_point_controls() -> None:
    # Cartesia prosody labels lower to its native speed/volume ratio tags
    assert convert_markup("cartesia", '<expr type="prosody" label="slow"/> One moment.') == (
        '<speed ratio="0.85"/> One moment.'
    )
    assert convert_markup("cartesia", '<expr type="prosody" label="loud"/> We won!') == (
        '<volume ratio="1.3"/> We won!'
    )
    # wrapping form applies the control before the span
    assert convert_markup("cartesia", '<expr type="prosody" label="soft">bad news</expr>') == (
        '<volume ratio="0.9"/>bad news'
    )


def test_convert_expr_cartesia_prosody_unknown_label_unwraps() -> None:
    text = '<expr type="prosody" label="whisper">keep it secret</expr>'
    assert convert_markup("cartesia", text) == "keep it secret"


def test_convert_stray_expr_never_reaches_tts() -> None:
    # an unpaired prosody open/close (e.g. split across stream chunks) is dropped,
    # keeping the words
    assert convert_markup("xai", '<expr type="prosody" label="loud">hello there') == "hello there"
    assert convert_markup("xai", "hello there</expr>") == "hello there"


# ---------------------------------------------------------------------------
# transcript stripping (provider-agnostic: the sinks strip without knowing the TTS)
# ---------------------------------------------------------------------------


def test_split_all_markup_strips_expr() -> None:
    clean, tags = split_all_markup(JOKE)
    assert clean.strip() == "Why did the burger go to the gym? Because it wanted better buns!"
    assert tags == [
        {"type": "expression", "value": "say playfully"},
        {"type": "break", "value": "500ms"},
        {"type": "sound", "value": "laugh"},
    ]


def test_split_all_markup_wrapping_keeps_inner_text() -> None:
    text = (
        'She said <expr type="prosody" label="whisper">keep it secret</expr> — '
        'code <expr type="spell">A7X9</expr>.'
    )
    clean, tags = split_all_markup(text)
    assert clean == "She said keep it secret — code A7X9."
    assert tags == [
        {"type": "prosody", "value": "whisper"},
        {"type": "spell", "value": ""},
    ]


def test_split_all_markup_mixed_expr_and_native() -> None:
    text = '<expr type="expression" label="say playfully"/> Hello! <sound value="laugh"/>'
    clean, tags = split_all_markup(text)
    assert clean.strip() == "Hello!"
    assert {"type": "expression", "value": "say playfully"} in tags
    assert {"type": "sound", "value": "laugh"} in tags


def test_split_all_markup_keeps_square_brackets() -> None:
    # bracket spans are a TTS-only native form (convert_markup emits them on the audio
    # path), so the transcript strip must leave markdown links and prose brackets intact
    text = 'Press [Enter], then read [the docs](https://docs.livekit.io). <sound value="sigh"/>'
    clean, tags = split_all_markup(text)
    assert clean == "Press [Enter], then read [the docs](https://docs.livekit.io)."
    assert tags == [{"type": "sound", "value": "sigh"}]


def test_expr_regex_does_not_match_native_expression_tag() -> None:
    # "<expr" is a prefix of "<expression" — the word boundary in the expr regexes
    # must keep the native Inworld tag on the generic strip path with its own type
    text = '<expression value="speak calmly"/> Hi <expr type="break" label="1s"/> there.'
    clean, tags = split_all_markup(text)
    # the native tag opens the text, so its separator goes with it
    assert clean == "Hi there."
    assert {"type": "expression", "value": "speak calmly"} in tags
    assert {"type": "break", "value": "1s"} in tags
    # conversion must also leave the native tag for the provider pipeline, not eat it
    assert convert_markup("inworld", text) == '[speak calmly] Hi <break time="1s"/> there.'


def test_transcript_stripper_streaming_chunks() -> None:
    stripper = TranscriptMarkupStripper()
    out = ""
    # split mid-tag so the partial "<expr ..." must be held back, never half-emitted
    for chunk in [
        '<expr type="expr',
        'ession" label="say playfully"/> Hello',
        ' <expr type="prosody" label="whisper">wor',
        "ld</expr>!",
    ]:
        out += stripper.push(chunk)
    out += stripper.flush()
    assert out == "Hello world!"
    assert stripper.tags[0] == {"type": "expression", "value": "say playfully"}
    assert {"type": "prosody", "value": "whisper"} in stripper.tags


def test_split_all_markup_removed_tag_leaves_one_space() -> None:
    # a marker between two spaces must not leave both behind, or punctuation ends up
    # followed by a double space in the transcript
    assert strip_all_markup('Right. <expr type="sound" label="laugh"/> Anyway.') == (
        "Right. Anyway."
    )
    assert strip_all_markup('Right. <sound value="laugh"/> Anyway.') == "Right. Anyway."
    # a wrapping marker keeps its inner text, so its spacing is untouched
    assert strip_all_markup('a <expr type="prosody" label="loud">b</expr> c') == "a b c"
    # only the doubled separator goes: a marker with text on one side keeps the space
    assert strip_all_markup('Right.<expr type="sound" label="laugh"/> Anyway.') == "Right. Anyway."
    assert strip_all_markup('Right. <expr type="sound" label="laugh"/>Anyway.') == "Right. Anyway."
    # newlines are structure, not a separator a strip may collapse
    assert strip_all_markup('a\n<expr type="sound" label="laugh"/>\nb') == "a\n\nb"


def test_split_all_markup_keeps_trailing_space_for_stream() -> None:
    # mid-stream that space is the separator for words still arriving
    chunk = 'Right. <expr type="sound" label="laugh"/>'
    assert split_all_markup(chunk, at_line_start=False, at_text_end=False)[0] == "Right. "
    # as a whole segment there is nothing still arriving, so the separator goes
    assert strip_all_markup(chunk) == "Right."


def test_transcript_stripper_dedups_space_across_chunks() -> None:
    # the space before the marker goes out with the previous chunk, so the in-text dedup
    # can't see it — the stripper has to close that seam itself
    for chunks in (
        ["Right. ", '<expr type="sound" label="laugh"/>', " Anyway."],
        ["Right. ", '<expr type="sound" label="laugh"/> Anyway.'],
        ["Right. ", '<sound value="laugh"/>', " Anyway."],
    ):
        stripper = TranscriptMarkupStripper()
        out = "".join(stripper.push(c) for c in chunks) + stripper.flush()
        assert out == "Right. Anyway.", chunks


def test_transcript_stripper_leaves_untagged_whitespace_alone() -> None:
    # without a stripped tag at the seam there is nothing to dedup: whitespace the LLM
    # itself emitted is passed through untouched
    stripper = TranscriptMarkupStripper()
    out = "".join(stripper.push(c) for c in ["Right. ", " Anyway."]) + stripper.flush()
    assert out == "Right.  Anyway."

    # a tag stripped earlier in the chunk doesn't license collapsing the seam either:
    # the whitespace here trails "hello", not the removed tag
    stripper = TranscriptMarkupStripper()
    chunks = ['<sound value="x"/>hello  ', "   world"]
    out = "".join(stripper.push(c) for c in chunks) + stripper.flush()
    assert out == "hello     world"


def test_expression_attribute_from_expr() -> None:
    _, tags = split_all_markup(JOKE)
    attr = expression_attribute(tags)
    assert attr is not None
    assert '"say playfully"' in next(iter(attr.values()))


# ---------------------------------------------------------------------------
# normalize_markup: fix unclosed self-closing expr markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["xai", "inworld", "cartesia"])
def test_normalize_closes_unclosed_expr(provider: str) -> None:
    text = '<expr type="sound" label="laugh"> Hello'
    assert normalize_markup(provider, text) == '<expr type="sound" label="laugh"/> Hello'


def test_normalize_leaves_wrapping_and_closed_tags_alone() -> None:
    text = (
        '<expr type="prosody" label="whisper">hi</expr> <expr type="break" label="1s"/> '
        '<expr type="spell">A7X9</expr>'
    )
    assert normalize_markup("xai", text) == text


# ---------------------------------------------------------------------------
# llm instructions: shared syntax, per-provider kinds and vocabularies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["xai", "inworld", "cartesia", "gemini"])
def test_llm_instructions_use_expr_syntax(provider: str) -> None:
    instructions = llm_instructions(provider)
    assert instructions is not None
    assert "<expr" in instructions
    assert '<expr type="break" label="' in instructions


def test_llm_instructions_cartesia_types() -> None:
    instructions = llm_instructions("cartesia")
    assert instructions is not None
    # discrete emotion vocabulary, not free-form descriptions
    assert '<expr type="expression" label="EMOTION"/>' in instructions
    assert "NOT free-form" in instructions
    assert '<expr type="spell">' in instructions
    # coarse self-closing prosody point controls
    assert '<expr type="prosody" label="slow"/>' in instructions
    # no non-verbal sounds
    assert 'type="sound"' not in instructions


def test_llm_instructions_inworld_kinds() -> None:
    instructions = llm_instructions("inworld")
    assert instructions is not None
    # free-form delivery descriptions + Inworld's own sound list
    assert '<expr type="expression" label="DESCRIPTION"/>' in instructions
    assert "free-form" in instructions
    assert "clear throat" in instructions
    # no wrapping prosody, no spell
    assert 'type="prosody"' not in instructions
    assert 'type="spell"' not in instructions


def test_llm_instructions_xai_kinds() -> None:
    instructions = llm_instructions("xai")
    assert instructions is not None
    # xAI's own sound cues + wrapping prosody vocabulary
    assert "tongue-click" in instructions
    assert '<expr type="prosody" label="STYLE">' in instructions
    assert "sing-song" in instructions
    # no free-form delivery descriptions, no spell
    assert 'type="expression"' not in instructions
    assert 'type="spell"' not in instructions


def test_llm_instructions_gemini_kinds() -> None:
    instructions = llm_instructions("gemini")
    assert instructions is not None
    # free-form style descriptors plus an accent, Gemini's own prompting vocabulary
    assert '<expr type="expression" label="DESCRIPTORS"/>' in instructions
    assert "free-form natural language" in instructions
    assert '"<PLACE> accent"' in instructions
    # discrete events are real for this voice, and land inline
    assert 'type="sound"' in instructions
    assert "laugh, chuckle, sigh, breath, cough, argh, gasp, giggle, cry" in instructions
    assert 'type="break"' in instructions
    # emphasis is the only wrapping prosody, and there is no spell marker
    assert '<expr type="prosody" label="emphasis">' in instructions
    assert 'type="spell"' not in instructions


def test_gemini_lowers_events_inline_and_leaves_the_style_marker() -> None:
    # a discrete event becomes an inline tag; the delivery marker is left standing
    assert convert_markup("gemini", GEMINI_TURN) == GEMINI_TURN
    turn = '<expr type="expression" label="Easygoing"/> Yeah, <expr type="sound" label="chuckle"/> I get that.'
    assert convert_markup("gemini", turn) == (
        '<expr type="expression" label="Easygoing"/> Yeah, <chuckle> I get that.'
    )


def test_gemini_pauses_and_emphasis_lower_to_native_forms() -> None:
    # Gemini documents one pause length, so a duration is only a hint that a beat belongs
    assert convert_markup("gemini", 'Oh no. <expr type="break" label="300ms"/> Let me look.') == (
        "Oh no. <short pause> Let me look."
    )
    assert convert_markup("gemini", '<expr type="break" label="2s"/> Right.') == (
        "<short pause> Right."
    )
    # its one in-text prosody control is capitalizing the word
    assert (
        convert_markup(
            "gemini", 'Your code is <expr type="prosody" label="emphasis">B four</expr>.'
        )
        == "Your code is B FOUR."
    )


def test_gemini_sound_aliases_and_unknown_labels() -> None:
    # other providers advertise "breathe"; it still lands on a tag Gemini renders
    assert convert_markup("gemini", '<expr type="sound" label="breathe"/> Okay.') == (
        "<breath> Okay."
    )
    # the general docs write these plural or gerund; they lower onto the stems
    for written, native in (("giggles", "giggle"), ("crying", "cry"), ("sighs", "sigh")):
        assert convert_markup("gemini", f'<expr type="sound" label="{written}"/> Okay.') == (
            f"<{native}> Okay."
        )
    # a label from no vocabulary at all is dropped rather than invented into a tag
    assert convert_markup("gemini", 'Hi <expr type="sound" label="kazoo"/> there.') == "Hi there."


def test_gemini_native_inline_tags_are_stripped_from_transcripts() -> None:
    # they belong in the words sent to the provider, never in what the user reads
    clean, _ = split_all_markup("Yeah, <chuckle> I get that. <short pause> Right?")
    assert clean == "Yeah, I get that. Right?"


def test_gemini_sound_steering_removes_the_section() -> None:
    off = llm_instructions("gemini", {"nonverbal_sounds": False})
    assert off is not None and 'type="sound"' not in off
    # ... and a single category takes only its own labels with it
    no_laughs = llm_instructions("gemini", {"nonverbal_sounds": {"laughing": False}})
    assert no_laughs is not None
    assert "chuckle" not in no_laughs and "laugh" not in no_laughs
    assert "giggle" not in no_laughs  # the whole laughter family, not just the stem
    assert "sigh, breath, cough, argh, gasp, cry" in no_laughs


def test_gemini_markers_strip_to_transcript_and_tags() -> None:
    clean, tags = split_all_markup(GEMINI_TURN)
    assert clean == "Sienna? What's on your mind, Comanchero?"
    assert tags == [
        {"type": "expression", "value": "Thoughtful, Quiet, American accent"},
        {"type": "expression", "value": "Wistful, American accent"},
    ]
    # the free-form label still normalizes to a mood for `lk.expression`
    attribute = expression_attribute(tags)
    assert attribute is not None
    assert '"expression":"Thoughtful, Quiet, American accent"' in next(iter(attribute.values()))


def test_gemini_normalizes_an_unclosed_marker() -> None:
    # the marker is the only thing carrying the style
    text = '<expr type="expression" label="Warm, Welcoming"> Hey there.'
    clean, tags = split_all_markup(normalize_markup("gemini", text))
    assert clean == "Hey there."
    assert tags == [{"type": "expression", "value": "Warm, Welcoming"}]


def test_llm_instructions_none_for_unknown_provider() -> None:
    assert llm_instructions("") is None
    assert llm_instructions("openai") is None


# ---------------------------------------------------------------------------
# strip_expr_markup + ChatMessage.text_content / raw_text_content
# ---------------------------------------------------------------------------

# assistant text mixing expr markers with content that must survive an expr-only strip:
# provider-native tags, bracket spans, markdown links, and stray angle brackets
MIXED = (
    '<expr type="expression" label="happy"/> Press [Enter] to see <b>bold</b>, '
    'read [the docs](https://docs.livekit.io), then 1 < 2. <break time="1s"/> '
    '<expr type="prosody" label="whisper">keep it secret</expr>'
)
MIXED_CLEAN = (
    "Press [Enter] to see <b>bold</b>, "
    'read [the docs](https://docs.livekit.io), then 1 < 2. <break time="1s"/> '
    "keep it secret"
)


def test_strip_expr_markup_only_touches_expr() -> None:
    assert strip_expr_markup(MIXED) == MIXED_CLEAN


def test_strip_expr_markup_noop_without_expr() -> None:
    text = 'plain text with [brackets] and <sound value="laugh"/>'
    assert strip_expr_markup(text) == text


def test_assistant_text_content_strips_expr_only() -> None:
    msg = ChatMessage(role="assistant", content=[MIXED])
    assert msg.text_content == MIXED_CLEAN
    assert msg.raw_text_content == MIXED


@pytest.mark.parametrize("role", ["user", "system", "developer"])
def test_non_assistant_text_content_stays_raw(role: str) -> None:
    # only assistant messages carry expressive markup; other roles are never stripped
    msg = ChatMessage(role=role, content=[JOKE])
    assert msg.text_content == JOKE
    assert msg.raw_text_content == JOKE


def test_text_content_none_without_text() -> None:
    msg = ChatMessage(role="assistant", content=[])
    assert msg.text_content is None
    assert msg.raw_text_content is None


def test_to_dict_strip_markup_is_expr_only_and_assistant_only() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=[MIXED])
    chat_ctx.add_message(role="assistant", content=[MIXED])

    items = chat_ctx.to_dict(strip_markup=True)["items"]
    assert items[0]["content"] == [MIXED]  # user content untouched
    assert items[1]["content"] == [MIXED_CLEAN]  # assistant loses only expr tags

    # default keeps the raw content for persistence
    items = chat_ctx.to_dict()["items"]
    assert items[1]["content"] == [MIXED]


# ---------------------------------------------------------------------------
# a marker opening a turn takes its separator with it
# ---------------------------------------------------------------------------


def test_opening_marker_leaves_no_leading_space() -> None:
    # a marker opens every turn, with only the space after it
    assert strip_all_markup(JOKE).startswith("Why did")
    assert strip_expr_markup(JOKE).startswith("Why did")
    assert split_all_markup(JOKE)[0].startswith("Why did")


def test_closing_marker_leaves_no_trailing_space() -> None:
    # the mirror case: a turn ending on a sound strands the space before the marker
    turn = JOKE + " Ha!"
    assert strip_all_markup(JOKE).endswith("better buns!")
    assert strip_expr_markup(JOKE).endswith("better buns!")
    assert strip_all_markup(turn).endswith("Ha!")  # a marker mid-turn keeps its separator


def test_streamed_closing_marker_leaves_no_trailing_space() -> None:
    stripper = TranscriptMarkupStripper()
    chunks = ["Hi there! ", '<expr type="sound" label="laugh"/>']
    assert "".join(stripper.push(c) for c in chunks) + stripper.flush() == "Hi there!"


@pytest.mark.parametrize(
    "chunks",
    [
        ['<expr type="expression" label="Warm"/> Hi there.'],
        # the marker lands in a chunk of its own, so its space falls to the next one
        ['<expr type="expression" label="Warm"/>', " Hi there."],
        ['<expr type="expression" la', 'bel="Warm"/> Hi there.'],
        ['  <expr type="expression" label="Warm"/> Hi there.'],
    ],
)
def test_streamed_opening_marker_leaves_no_leading_space(chunks: list[str]) -> None:
    stripper = TranscriptMarkupStripper()
    assert "".join(stripper.push(c) for c in chunks) + stripper.flush() == "Hi there."


def test_streamed_marker_mid_segment_does_not_glue_words() -> None:
    # a chunk boundary is not a segment boundary
    stripper = TranscriptMarkupStripper()
    chunks = ["Hi there.", '<expr type="expression" label="Bright"/> Lovely day.']
    assert "".join(stripper.push(c) for c in chunks) + stripper.flush() == "Hi there. Lovely day."


# ---------------------------------------------------------------------------
# a marker heading a line takes its separator with it
# ---------------------------------------------------------------------------

# one sentence per line puts every marker after the first at the head of a line
PER_LINE = (
    '<expr type="expression" label="Warm"/> Hi there.\n'
    '<expr type="expression" label="Amused"/> Why so late?'
)


def test_line_opening_marker_leaves_no_leading_space() -> None:
    assert strip_all_markup(PER_LINE) == "Hi there.\nWhy so late?"
    assert strip_expr_markup(PER_LINE) == "Hi there.\nWhy so late?"


@pytest.mark.parametrize(
    "chunks",
    [
        [PER_LINE],
        # the line break ends one chunk, so the next one has no newline of its own to see
        [PER_LINE[: PER_LINE.index("\n") + 1], PER_LINE[PER_LINE.index("\n") + 1 :]],
        ["Hi there.\n", '<expr type="expression" label="Amused"/> Why so late?'],
    ],
)
def test_streamed_line_opening_marker_leaves_no_leading_space(chunks: list[str]) -> None:
    stripper = TranscriptMarkupStripper()
    out = "".join(stripper.push(c) for c in chunks) + stripper.flush()
    assert out == "Hi there.\nWhy so late?"


def test_streamed_chunk_starting_mid_line_keeps_its_separator() -> None:
    # the counterpart trap: that leading space is all there is between two words
    stripper = TranscriptMarkupStripper()
    chunks = ['<expr type="expression" label="Warm"/> Hi', " there."]
    assert "".join(stripper.push(c) for c in chunks) + stripper.flush() == "Hi there."


def test_paragraph_structure_is_never_collapsed() -> None:
    # only the horizontal run a marker stranded goes, never the line break
    assert strip_all_markup('a\n<expr type="sound" label="laugh"/>\nb') == "a\n\nb"
    for text in ("   Indented on purpose.  ", "a\n   indented line"):
        assert strip_all_markup(text) == text
        assert strip_expr_markup(text) == text


def test_only_the_stranded_separator_goes() -> None:
    # the drop happens at the removal, so whitespace on lines the marker never touched
    # is left alone -- indented blocks survive a turn that carries markers
    turn = '<expr type="expression" label="Warm"/> Intro:\n    indented text'
    assert strip_all_markup(turn) == "Intro:\n    indented text"
    assert strip_expr_markup(turn) == "Intro:\n    indented text"
    # and a marker mid-line still leaves exactly one separator
    assert strip_all_markup('a <expr type="sound" label="laugh"/> b') == "a b"
