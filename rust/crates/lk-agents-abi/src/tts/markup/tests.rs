//! Markup calculators: ordered input, assert on output. [`TranscriptMarkupStripper`] is the one
//! accumulator, asserted on its final emission and tag list.

use lk_agents_abi::{ChatContent, ChatContext, ChatMessage, Role};

use super::*;

fn t(secs: u64) -> std::time::SystemTime {
    std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs)
}

// --- the whitespace rule (#6717) --------------------------------------------------------------

#[test]
fn whitespace_the_model_wrote_mid_chunk_is_not_collapsed_at_the_seam() {
    // #6717 regression: a tag stripped at chunk start leaves the model's own spaces alone.
    let mut s = TranscriptMarkupStripper::new();
    let a = s.push("<sound value=\"x\"/>hello  ");
    let b = s.push("   world");
    assert_eq!(format!("{a}{b}{}", s.flush()), "hello     world");
}

#[test]
fn a_marker_ending_a_chunk_collapses_with_the_next_chunks_space() {
    // The spaces either side of the marker are split across chunks; only the stripper sees both.
    let mut s = TranscriptMarkupStripper::new();
    let a = s.push("Ready. <expr type=\"break\" label=\"1s\"/>");
    let b = s.push(" Go.");
    assert_eq!(format!("{a}{b}{}", s.flush()), "Ready. Go.");
}

// --- brackets are prose, not markup (#6590) ----------------------------------------------------

#[test]
fn a_bare_less_than_in_prose_does_not_stall_the_transcript() {
    // #6116: `"3 < 5"` must not freeze the segment.
    let mut s = TranscriptMarkupStripper::new();
    assert_eq!(s.push("3 < 5 and 4 "), "3 < 5 and 4");
    assert_eq!(s.flush(), " ", "only the trailing separator is held, and only until flush");
    // but a tag-shaped tail is still held until it closes
    let mut s = TranscriptMarkupStripper::new();
    assert_eq!(s.push("hi <sou"), "");
    assert_eq!(s.push("nd value=\"laugh\"/> there"), "hi there");
}

// --- nesting and the fixed point (#6116) --------------------------------------------------------

#[test]
fn nested_wrapping_tags_are_fully_removed() {
    // #6116: nested emotion + prosody (xAI) needs more than one pass.
    let (clean, tags) =
        split_all_markup("<excited><loud><higher-pitch>no way</higher-pitch></loud></excited>");
    assert_eq!(clean, "no way");
    // the outer tag's value is the *raw* inner markup, because it was recorded on the first pass
    assert_eq!(tags[0].kind, "excited");
    assert_eq!(tags[0].value, "<loud><higher-pitch>no way</higher-pitch></loud>");
    assert_eq!(tags.last().unwrap(), &ExpressiveTag::new("higher-pitch", "no way"));
}

// --- normalization (#6347) ----------------------------------------------------------------------

#[test]
fn normalizing_an_already_closed_marker_is_a_no_op() {
    // `[^/>\s]` keeps the rewrite idempotent (no `...//>`).
    let m = Markup::from_provider(Provider::Inworld);
    for text in ["<expr type=\"sound\" label=\"laugh\"/>", "<expr type=\"break\" label=\"1s\" />"] {
        assert_eq!(m.normalize(text), text);
        assert_eq!(m.normalize(&m.normalize(text)), m.normalize(text));
    }
}

#[test]
fn the_native_closer_eats_the_first_angle_bracket_when_a_second_follows() {
    // CPython's `([^>]*[^/])`: `[^/]` admits `>`, so `<break>  >` matches as a whole.
    let m = Markup::from_provider(Provider::Cartesia);
    assert_eq!(m.normalize("<break>  >"), "<break>/>");
    // and with no second `>`, the documented case: a bare `<break>` is left alone
    assert_eq!(m.normalize("<break> x"), "<break> x");
    assert_eq!(m.normalize("<break time=\"1s\" >"), "<break time=\"1s\" />");
}

// --- per-provider lowering ----------------------------------------------------------------------

#[test]
fn an_unsupported_marker_type_loses_the_marker_and_keeps_the_words() {
    // A hallucinated prosody for Inworld: keep the words, no doubled space.
    let i = Markup::from_provider(Provider::Inworld);
    assert_eq!(
        i.convert("say <expr type=\"prosody\" label=\"whisper\">this</expr> now"),
        "say [whisper]this now"
    );
    // xAI has no free-form delivery descriptions, so an expression marker vanishes entirely
    let x = Markup::from_provider(Provider::Xai);
    assert_eq!(x.convert("a <expr type=\"expression\" label=\"happy\"/> b"), "a b");
}

#[test]
fn a_provider_without_a_dialect_passes_text_through_untouched() {
    // `""` key (rime, deepgram): both hooks are no-ops and `llm_instructions` is `None`.
    let m = Markup::from_provider_key("rime");
    let text = "<expr type=\"sound\" label=\"laugh\"/> hi";
    assert_eq!(m.provider_key(), "");
    assert_eq!(m.normalize(text), text);
    assert_eq!(m.convert(text), text);
    assert_eq!(m.llm_instructions(None), None);
}

// --- moods --------------------------------------------------------------------------------------

#[test]
fn keyword_stems_match_at_word_starts_only() {
    // #6698: stems match at word starts only.
    assert_eq!(match_mood("like a pirate", Some(DEFAULT_MOOD)), Some(Mood::Calm));
    assert_eq!(match_mood("irate", Some(DEFAULT_MOOD)), Some(Mood::Angry));
    assert_eq!(match_mood("irately snapping", Some(DEFAULT_MOOD)), Some(Mood::Angry));
}

#[test]
fn a_named_mood_outweighs_a_supporting_descriptor() {
    // Weight 2 names a mood, weight 1 only supports one.
    assert_eq!(match_mood("gently curious, welcoming", Some(DEFAULT_MOOD)), Some(Mood::Curious));
}

#[test]
fn ties_break_by_priority_and_an_unmatched_label_is_recessive() {
    // Strict `>`: the first entry in MOOD_PRIORITY wins a tie.
    assert_eq!(match_mood("furious and heartbroken", Some(DEFAULT_MOOD)), Some(Mood::Angry));
    assert_eq!(match_mood("zzzz", Some(DEFAULT_MOOD)), Some(Mood::Calm));
    // the caller can opt out of the fallback and handle the miss itself
    assert_eq!(match_mood("zzzz", None), None);
}

#[test]
fn the_expression_attribute_is_compact_and_ascii_escaped() {
    // `json.dumps` defaults to `ensure_ascii=True`.
    let tags = vec![ExpressiveTag::new("expression", "trés\u{a0}calme")];
    let (key, json) = expression_attribute(&tags).unwrap();
    assert_eq!(key, lk_agents_abi::ATTRIBUTE_TRANSCRIPTION_EXPRESSION);
    assert_eq!(json, r#"{"expression":"tr\u00e9s\u00a0calme","mood":"calm"}"#);
    // no expression or emotion tag at all -> no attribute, not an empty one
    assert_eq!(expression_attribute(&[ExpressiveTag::new("sound", "laugh")]), None);
}

// --- steering ------------------------------------------------------------------------------------

#[test]
fn disfluencies_false_renders_the_opposite_guideline_not_silence() {
    // `disfluencies ... is not None`: a presence test, not truthiness.
    let m = Markup::from_provider(Provider::Cartesia);
    let off = SpeechSteering { disfluencies: Some(false), ..Default::default() };
    assert_eq!(
        m.steering_instructions(&off),
        "Delivery guidelines:\n- No fillers (um, uh). Sound composed and fluent."
    );
    assert!(m.steering_instructions(&SpeechSteering::default()).is_empty());
}

#[test]
fn a_normal_pace_is_not_a_guideline() {
    // `pace != "normal"`: the default renders nothing.
    let m = Markup::from_provider(Provider::Cartesia);
    let normal = SpeechSteering { pace: Some(Pace::Normal), ..Default::default() };
    assert_eq!(m.steering_instructions(&normal), "");
    let slow = SpeechSteering { pace: Some(Pace::Slow), ..Default::default() };
    assert_eq!(
        m.steering_instructions(&slow),
        "Delivery guidelines:\n- Keep a slow overall speaking pace."
    );
}

#[test]
fn a_field_that_governs_no_label_is_not_reported_as_supported() {
    // `if labels:`: xAI's empty `reflex_sounds` is omitted.
    let nv = supported_nonverbals(Some(Provider::Xai));
    assert!(!nv.iter().any(|(f, _)| *f == NonverbalField::ReflexSounds));
    assert!(nv.iter().any(|(f, l)| *f == NonverbalField::Laughing
        && *l == vec!["laugh", "chuckle", "giggle", "laugh-speak"]));
    // and the two tables merge in order, sounds before prosody
    assert!(supported_nonverbals(Some(Provider::Cartesia)).is_empty());
    assert!(supported_nonverbals(None).is_empty());
}

// --- aligned transcript --------------------------------------------------------------------------

#[test]
fn a_dropped_cue_takes_exactly_one_of_its_two_spaces() {
    // `dedup_removal_space` on the aligned-token path.
    let toks = |s: &str| vec![lk_agents_abi::TimedString::new(s)];
    let mut held = Vec::new();
    let out = drop_bracket_cues(toks("Hello [laugh] world"), &mut held, true);
    assert_eq!(out.iter().map(|t| t.text.as_str()).collect::<String>(), "Hello world");
}

#[test]
fn a_cue_split_across_alignment_messages_is_still_judged_whole() {
    // An unclosed `[` is held across alignment messages.
    let mut held = Vec::new();
    let a = drop_bracket_cues(vec![lk_agents_abi::TimedString::new("hi [lau")], &mut held, false);
    assert_eq!(a.iter().map(|t| t.text.as_str()).collect::<String>(), "hi ");
    let b =
        drop_bracket_cues(vec![lk_agents_abi::TimedString::new("gh] there")], &mut held, false);
    assert_eq!(b.iter().map(|t| t.text.as_str()).collect::<String>(), "there");
    assert!(held.is_empty());
}

#[test]
fn a_lone_bracket_is_released_rather_than_stalling_the_transcript() {
    // The hold is capped at 256 characters.
    let long = format!("[{}", "x".repeat(300));
    let mut held = Vec::new();
    let out = drop_bracket_cues(vec![lk_agents_abi::TimedString::new(&long)], &mut held, false);
    assert_eq!(out.iter().map(|t| t.text.as_str()).collect::<String>(), long);
    assert!(held.is_empty());
}

#[test]
fn surviving_tokens_keep_their_alignment_timings() {
    // A partly dropped token keeps its timings.
    let tok = lk_agents_abi::TimedString {
        text: "a[cue]b".into(),
        start_time: Some(1.5),
        end_time: Some(2.5),
        ..Default::default()
    };
    let mut held = Vec::new();
    let out = drop_bracket_cues(vec![tok], &mut held, true);
    assert_eq!(out[0].text, "ab");
    assert_eq!((out[0].start_time, out[0].end_time), (Some(1.5), Some(2.5)));
}

// Catches: the plan losing the index channel, or reporting an unchanged token as rewritten (a
// `Some` would make the host rebuild the `TimedString` and drop its alignment fields).
#[test]
fn drop_bracket_cues_plans_by_index() {
    let texts = vec!["hey ".to_string(), "[laugh]".to_string(), " there".to_string()];
    let (emit, keep) = drop_bracket_cues_plan(&texts, true);
    assert!(keep.is_empty());
    assert_eq!(emit, vec![(0, Some("hey".to_string())), (2, None)]);
}

// --- chat-context edits ----------------------------------------------------------------------------

#[test]
fn only_assistant_messages_are_scrubbed_of_markup() {
    // History markup few-shots the model; a user turn quoting a tag is left alone.
    let mut ctx = ChatContext::new(vec![
        ChatMessage::new(
            "u",
            Role::User,
            vec![ChatContent::Text("what does <sound value=\"laugh\"/> mean?".into())],
            t(1),
        )
        .into(),
        ChatMessage::new(
            "a",
            Role::Assistant,
            vec![ChatContent::Text("ha <sound value=\"laugh\"/> ha".into())],
            t(2),
        )
        .into(),
    ]);
    assert_eq!(strip_assistant_markup(&mut ctx), 1);
    let text = |i: usize| match &ctx.items[i] {
        lk_agents_abi::ChatItem::Message(m) => match &m.content[0] {
            ChatContent::Text(t) => t.clone(),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    assert_eq!(text(0), "what does <sound value=\"laugh\"/> mean?");
    assert_eq!(text(1), "ha ha");
}

#[test]
fn expressive_chunking_raises_the_minimum_only_when_expressive_is_on() {
    // Batch up to the send limit, and xml-aware only under expressive (#6116's class of stall).
    let m = Markup::from_provider(Provider::Inworld);
    assert_eq!(
        m.sentence_tokenizer(true),
        SentenceTokenizerConfig {
            max_token_len: Some(900),
            min_token_len: Some(900),
            xml_aware: true
        }
    );
    assert_eq!(
        m.sentence_tokenizer(false),
        SentenceTokenizerConfig { max_token_len: Some(900), min_token_len: None, xml_aware: false }
    );
    // Fish is deliberately uncapped.
    assert_eq!(Markup::from_provider(Provider::FishAudio).max_input_len(), None);
}

// --- gaps the nightly mutation run named -----------------------------------------------------

#[test]
fn a_provider_key_string_resolves_to_its_dialect() {
    // The gateway's `_provider_key()` string is the production entry point.
    for (key, provider) in [
        ("cartesia", Provider::Cartesia),
        ("inworld", Provider::Inworld),
        ("xai", Provider::Xai),
        ("fishaudio", Provider::FishAudio),
    ] {
        let m = Markup::from_provider_key(key);
        assert_eq!(m, Markup::from_provider(provider));
        assert_eq!(m.provider_key(), key);
        assert!(m.llm_instructions(None).is_some());
    }
    let none = Markup::from_provider_key("nope");
    assert_eq!(none.provider_key(), "");
    assert!(none.llm_instructions(None).is_none());
    assert!(nonverbals_of("nope").is_empty());
    assert!(!nonverbals_of("xai").is_empty());
    assert_eq!(none.max_input_len(), None);
    assert_eq!(Markup::from_provider_key("cartesia").max_input_len(), Some(400));
}

#[test]
fn removing_the_guide_leaves_the_rest_of_the_history_alone() {
    // An inverted predicate would still remove exactly one item.
    let mut ctx = ChatContext::new(vec![
        ChatMessage::new("u", Role::User, vec![ChatContent::Text("hi".into())], t(1)).into(),
        ChatMessage::new(
            EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID,
            Role::System,
            vec![ChatContent::Text("guide".into())],
            t(2),
        )
        .into(),
        ChatMessage::new("a", Role::Assistant, vec![ChatContent::Text("hey".into())], t(3)).into(),
    ]);
    assert_eq!(remove_expressive_instructions(&mut ctx), 1);
    assert_eq!(ctx.items.iter().map(|i| i.id()).collect::<Vec<_>>(), ["u", "a"]);
    assert_eq!(remove_expressive_instructions(&mut ctx), 0, "a second pass finds nothing");
}

#[test]
fn a_stray_expr_closer_is_dropped_without_leaving_its_delimiters() {
    // An orphaned `</expr>` from a split wrapper must never be spoken.
    assert_eq!(strip_expr_markup("done </expr> here"), "done here");
    assert_eq!(strip_all_markup("done </expr> here"), "done here");
    assert_eq!(Markup::from_provider(Provider::Xai).convert("done </expr> here"), "done here");
    // `\s*` inside the closer, and no `\b` after `expr`, so `</expression>` is not one
    assert_eq!(strip_expr_markup("a </expr > b"), "a b");
    assert_eq!(strip_expr_markup("a </expression> b"), "a </expression> b");
}

#[test]
fn only_prosody_and_spell_markers_wrap_their_span() {
    // The prosody|spell lookahead decides wrapping; otherwise words between markers are swallowed.
    let x = Markup::from_provider(Provider::Xai);
    assert_eq!(
        x.convert("<expr type=\"prosody\" label=\"whisper\">psst</expr> ok"),
        "<whisper>psst</whisper> ok"
    );
    // A self-closing marker then an orphaned `</expr>`: the pause must not vanish.
    assert_eq!(
        x.convert("<expr type=\"break\" label=\"1s\"/> words </expr>"),
        "[long-pause] words "
    );
    assert_eq!(
        Markup::from_provider(Provider::Inworld)
            .convert("<expr type=\"expression\" label=\"warm\"/>hi</expr>"),
        "[warm]hi"
    );
    // and the span of a real wrapper runs to the *earliest* closer, not the last
    assert_eq!(
        x.convert("<expr type=\"prosody\" label=\"loud\">a</expr>b</expr>c"),
        "<loud>a</loud>bc"
    );
}

#[test]
fn every_marker_type_lowers_to_each_dialect() {
    // One row per (marker type, provider): the crate's dialect rules as a matrix.
    let rows: &[(&str, [&str; 4])] = &[
        // (input, [cartesia, inworld, xai, fishaudio])
        (
            r#"<expr type="expression" label="happy"/>x"#,
            [r#"<emotion value="happy"/>x"#, "[happy]x", "x", "[very happy]x"],
        ),
        (r#"<expr type="sound" label="laugh"/>x"#, ["x", "[laugh]x", "[laugh]x", "[laughing]x"]),
        (
            // the alias tables: Inworld's "breathe" is xAI's "breath", Fish's "sigh" is "sighing"
            r#"<expr type="sound" label="breathe"/><expr type="sound" label="sigh"/>x"#,
            ["x", "[breathe][sigh]x", "[breath][sigh]x", "[breathe][sighing]x"],
        ),
        (
            r#"<expr type="break" label="1s"/>x"#,
            [r#"<break time="1s"/>x"#, r#"<break time="1s"/>x"#, "[long-pause]x", "[long-break]x"],
        ),
        (
            r#"<expr type="prosody" label="slow">go</expr>"#,
            [r#"<speed ratio="0.85"/>go"#, "[slow]go", "<slow>go</slow>", "go"],
        ),
        (
            r#"<expr type="prosody" label="whispering">go</expr>"#,
            [r#"go"#, "[whispering]go", "go", "[whispering] go"],
        ),
        (
            // self-closing prosody: a point control for Cartesia, a salvaged prefix for Fish
            r#"<expr type="prosody" label="loud"/>x"#,
            [r#"<volume ratio="1.3"/>x"#, "x", "x", "x"],
        ),
        (r#"<expr type="spell">A7X9</expr>."#, ["<spell>A7X9</spell>.", "A7X9.", "A7X9.", "A7X9."]),
    ];
    let providers = [Provider::Cartesia, Provider::Inworld, Provider::Xai, Provider::FishAudio];
    for (input, expected) in rows {
        for (p, want) in providers.iter().zip(expected) {
            assert_eq!(
                &Markup::from_provider(*p).convert(input),
                want,
                "{:?} on {input:?}",
                p.as_str()
            );
        }
    }
}

#[test]
fn fish_emphasis_is_a_prefix_marker_on_the_stressed_word() {
    // Fish has no wrapping stress tag: unwind to a prefix `[emphasis]`, case-insensitively.
    let f = Markup::from_provider(Provider::FishAudio);
    assert_eq!(f.convert("the <emphasis>ninth</emphasis>."), "the [emphasis] ninth.");
    assert_eq!(f.convert("<EMPHASIS> ninth </EMPHASIS>"), "[emphasis] ninth");
    assert_eq!(f.convert(r#"<emphasis class="x">ninth</emphasis>"#), "[emphasis] ninth");
    // `([^<]*)` cannot cross a `<`, so an unclosed one is left for the transcript stripper
    assert_eq!(f.convert("<emphasis>ninth"), "<emphasis>ninth");
}

#[test]
fn break_durations_bucket_at_one_second_and_parse_the_way_pythons_float_does() {
    // Boundary `>= 1.0` s; `float(str)` accepts whitespace and `_` between digits only;
    // unparseable is 0.0 (short), never a crash.
    let x = Markup::from_provider(Provider::Xai);
    for (time, want) in [
        ("999ms", "[pause]"),
        ("1000ms", "[long-pause]"),
        ("1s", "[long-pause]"),
        ("soon", "[pause]"),
        ("500 ms", "[pause]"),
        ("1_000ms", "[long-pause]"),
        ("1_000_000ms", "[long-pause]"),
        ("_1000ms", "[pause]"),
        ("1000_ms", "[pause]"),
        ("1__000ms", "[pause]"),
        ("_ms", "[pause]"),
        ("1e3ms", "[long-pause]"),
        (" 2s ", "[long-pause]"),
        ("", "[pause]"),
        ("s", "[pause]"),
        // Accepted divergence: CPython would transliterate these digits (see `py_float`).
        ("\u{661}\u{662}\u{663}\u{664}ms", "[pause]"),
    ] {
        assert_eq!(x.convert(&format!(r#"<break time="{time}"/>"#)), want, "time={time:?}");
    }
    // the tag itself needs `\s+` before `time` and accepts either closer
    assert_eq!(x.convert(r#"<break time="1s">"#), "[long-pause]");
    assert_eq!(x.convert(r#"<break time="1s" />"#), "[long-pause]");
    assert_eq!(x.convert(r#"<breaktime="1s"/>"#), r#"<breaktime="1s"/>"#);
    let f = Markup::from_provider(Provider::FishAudio);
    assert_eq!(f.convert("<break time=\"2s\"/>"), "[long-break]");
    assert_eq!(f.convert("<break time=\"500ms\"/>"), "[break]");
}

#[test]
fn silencing_every_sound_removes_the_section_for_each_provider_that_has_one() {
    // Each builder's `if sounds:` guard: all-off must drop the Sounds section.
    let off = SpeechSteering { nonverbal_sounds: Some(NonverbalSounds::Off), ..Default::default() };
    for provider in [Provider::Inworld, Provider::Xai, Provider::FishAudio] {
        let on = Markup::from_provider(provider).llm_instructions(None).unwrap();
        let quiet = Markup::from_provider(provider).llm_instructions(Some(&off)).unwrap();
        assert!(on.contains("Sounds -"), "{}: has sounds by default", provider.as_str());
        assert!(!quiet.contains("Sounds -"), "{}: and loses the section", provider.as_str());
        assert!(!quiet.contains(r#"type="sound""#), "{}: examples go too", provider.as_str());
        assert!(quiet.len() < on.len(), "{}: strictly smaller", provider.as_str());
    }
    // Inworld's clear-throat aside and breathe warning are conditional on those labels surviving
    let no_reflex = SpeechSteering {
        nonverbal_sounds: Some(NonverbalSounds::Per(NonverbalOptions {
            reflex_sounds: Some(false),
            breathing: Some(false),
            ..Default::default()
        })),
        ..Default::default()
    };
    let i = Markup::from_provider(Provider::Inworld).llm_instructions(Some(&no_reflex)).unwrap();
    assert!(!i.contains("clear-throat when shifting"));
    assert!(!i.contains("Use the \"breathe\" sound"));
    // Cartesia has no sounds, so steering cannot change its block
    let c = Markup::from_provider(Provider::Cartesia);
    let all_off = SpeechSteering { disfluencies: Some(false), ..off };
    assert_eq!(c.llm_instructions(Some(&all_off)), c.llm_instructions(None));
}

#[test]
fn fish_salvages_a_self_closing_tone_and_drops_anything_else() {
    // Fish tones: a self-closed one is usable; an unknown label must not reach the TTS.
    let f = Markup::from_provider(Provider::FishAudio);
    assert_eq!(f.convert(r#"<expr type="prosody" label="Whispering"/>psst"#), "[whispering]psst");
    assert_eq!(f.convert(r#"<expr type="prosody" label="nonsense"/>psst"#), "psst");
    // and the same label on a provider without tones is dropped rather than mistranslated
    assert_eq!(
        Markup::from_provider(Provider::Xai)
            .convert(r#"<expr type="prosody" label="whispering"/>x"#),
        "x"
    );
}

#[test]
fn all_three_self_closing_expr_types_are_closed_and_nothing_else_is() {
    // The lookahead lists exactly `expression|break|sound` (#6347).
    let m = Markup::from_provider(Provider::Cartesia);
    for ty in ["expression", "break", "sound"] {
        let raw = format!(r#"<expr type="{ty}" label="x">"#);
        assert_eq!(m.normalize(&raw), format!(r#"<expr type="{ty}" label="x"/>"#));
    }
    for ty in ["prosody", "spell"] {
        let raw = format!(r#"<expr type="{ty}" label="x">a</expr>"#);
        assert_eq!(m.normalize(&raw), raw, "a wrapping marker must stay open");
    }
}

#[test]
fn a_tag_name_needs_a_word_boundary_before_it_counts() {
    // `<(TAGS)\b`: `<soundtrack>` is not a `<sound>`.
    let m = Markup::from_provider(Provider::Inworld);
    assert_eq!(m.normalize(r#"<expressionist value="x">"#), r#"<expressionist value="x">"#);
    assert_eq!(m.normalize(r#"<expression value="x">"#), r#"<expression value="x"/>"#);
    // and the same for the expr dialect: `<expr` must not swallow `<expression`
    assert_eq!(strip_expr_markup(r#"<expression value="x"/>"#), r#"<expression value="x"/>"#);
}

