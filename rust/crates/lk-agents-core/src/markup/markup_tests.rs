use lk_agents_abi::{ChatContent, ChatContext, ChatMessage, Role};

use super::*;

fn t(secs: u64) -> std::time::SystemTime {
    std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs)
}

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

#[test]
fn a_bare_less_than_in_prose_does_not_stall_the_transcript() {
    // #6116: `"3 < 5"` must not freeze the segment.
    let mut s = TranscriptMarkupStripper::new();
    assert_eq!(s.push("3 < 5 and 4 "), "3 < 5 and 4");
    assert_eq!(s.flush(), " ", "only the trailing separator is held, and only until flush");
    // but a tail that could still be a tag is held until it closes
    let mut s = TranscriptMarkupStripper::new();
    assert_eq!(s.push("hi <sou"), "hi");
    assert_eq!(s.push("nd value=\"laugh\"/> there"), " there");
}

#[test]
fn nested_wrapping_tags_are_fully_removed() {
    // #6116: nested emotion + prosody (xAI) needs more than one pass.
    let (clean, tags) =
        split_all_markup("<excited><loud><higher-pitch>no way</higher-pitch></loud></excited>");
    assert_eq!(clean, "no way");
    // was: a wrapping tag's value was its raw inner markup; a tag's value is its first attribute
    let kinds: Vec<&str> = tags.iter().map(|t| t.kind.as_str()).collect();
    assert_eq!(kinds, ["excited", "loud", "higher-pitch"]);
    assert!(tags.iter().all(|t| t.value.is_empty()));
}

#[test]
fn normalizing_an_already_closed_marker_is_a_no_op() {
    // `[^/>\s]` keeps the rewrite idempotent (no `...//>`).
    let m = Markup::from_provider(Provider::Inworld);
    let canonical = "<expr type=\"sound\" label=\"laugh\"/>";
    assert_eq!(m.normalize(canonical), canonical);
    // was: kept as written; a settled tag now takes its canonical spelling
    let spaced = "<expr type=\"break\" label=\"1s\" />";
    assert_eq!(m.normalize(spaced), "<expr type=\"break\" label=\"1s\"/>");
    assert_eq!(m.normalize(&m.normalize(spaced)), m.normalize(spaced));
    // a tag the delta ends inside stays as written for the next delta to complete
    assert_eq!(m.normalize("hi <expr type=\"sou"), "hi <expr type=\"sou");
}

#[test]
fn a_native_the_provider_writes_self_closed_is_self_closed() {
    // was: CPython-regex quirks (`<break>  >` -> `<break>/>`, a bare `<break>` left open)
    let m = Markup::from_provider(Provider::Cartesia);
    assert_eq!(m.normalize("<break>  >"), "<break/>  >");
    assert_eq!(m.normalize("<break> x"), "<break/> x");
    assert_eq!(m.normalize("<break time=\"1s\" >"), "<break time=\"1s\"/>");
}

#[test]
fn an_unsupported_marker_type_loses_the_marker_and_keeps_the_words() {
    // A hallucinated prosody for Inworld: keep the words, no doubled space.
    let i = Markup::from_provider(Provider::Inworld);
    assert_eq!(
        i.lower("say <expr type=\"prosody\" label=\"whisper\">this</expr> now").text,
        "say [whisper] this now"
    );
    // xAI has no free-form delivery descriptions, so an expression marker vanishes entirely
    let x = Markup::from_provider(Provider::Xai);
    assert_eq!(x.lower("a <expr type=\"expression\" label=\"happy\"/> b").text, "a b");
}

// Catches: a voice with no dialect sent a marker it would read aloud.
#[test]
fn a_provider_without_a_dialect_drops_markers_and_keeps_the_words() {
    // `""` key (rime, deepgram): nothing to normalize, no instructions, markers dropped.
    let m = Markup::from_provider_key("rime");
    let text = "well <expr type=\"sound\" label=\"laugh\"/> hi <b>there</b>";
    assert_eq!(m.provider_key(), "");
    assert_eq!(m.normalize(text), text);
    // a point marker written open is closed, so it never holds text as a wrapper would
    assert_eq!(
        m.normalize("a <expr type=\"break\" label=\"1s\"> b"),
        "a <expr type=\"break\" label=\"1s\"/> b"
    );
    let lowered = m.lower(text);
    assert_eq!(lowered.text, "well hi <b>there</b>");
    assert_eq!(lowered.said, lowered.text);
    assert_eq!(m.llm_instructions(None), None);
}

// Catches: markup aligned as words (a cue counted as said, a tag glued to a word), or a real
// bracketed phrase taken for a cue: what is said comes from the lowering, never re-read.
#[test]
fn a_lowered_chunk_says_exactly_its_words() {
    let lower = |p, text| Markup::from_provider(p).lower(text);
    let laugh = lower(Provider::Inworld, "Okay <expr type=\"sound\" label=\"laugh\"/> sure.");
    assert_eq!((laugh.text.as_str(), laugh.said.as_str()), ("Okay [laugh] sure.", "Okay sure."));
    let happy = lower(Provider::Cartesia, "<expr type=\"expression\" label=\"happy\"/>Sure.");
    assert_eq!(happy.text, "<emotion value=\"happy\"/>Sure.");
    assert_eq!(happy.said, "Sure.");
    let stress =
        lower(Provider::Gemini, "<expr type=\"prosody\" label=\"emphasis\">nine</expr> dollars");
    assert_eq!((stress.text.as_str(), stress.said.as_str()), ("NINE dollars", "nine dollars"));
    let whisper =
        lower(Provider::Xai, "a <expr type=\"prosody\" label=\"whisper\">quiet</expr> word");
    assert_eq!(whisper.text, "a <whisper>quiet</whisper> word");
    assert_eq!(whisper.said, "a quiet word");
    let prose = lower(Provider::Inworld, "see [the docs] now");
    assert_eq!(prose.said, "see [the docs] now");
}

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

// Catches: the `lk.expression` value losing its documented `{"expression", "mood"}` shape, or the
// provider's words altered on the way.
#[test]
fn the_expression_attribute_is_the_documented_compact_json() {
    let tags = vec![ExpressiveTag::new("expression", "trés\u{a0}calme")];
    let (key, json) = expression_attribute(&tags).unwrap();
    assert_eq!(key, lk_agents_abi::ATTRIBUTE_TRANSCRIPTION_EXPRESSION);
    assert_eq!(json, "{\"expression\":\"trés\u{a0}calme\",\"mood\":\"calm\"}");
    // no expression or emotion tag at all -> no attribute, not an empty one
    assert_eq!(expression_attribute(&[ExpressiveTag::new("sound", "laugh")]), None);
}

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

#[test]
fn a_dialect_knows_its_providers_request_limit() {
    assert_eq!(Markup::from_provider(Provider::Inworld).max_input_len(), Some(900));
    assert_eq!(Markup::from_provider(Provider::Cartesia).max_input_len(), Some(400));
    // Fish is deliberately uncapped.
    assert_eq!(Markup::from_provider(Provider::FishAudio).max_input_len(), None);
}

#[test]
fn a_provider_key_string_resolves_to_its_dialect() {
    // The gateway's `_provider_key()` string is the production entry point.
    for (key, provider) in [
        ("cartesia", Provider::Cartesia),
        ("inworld", Provider::Inworld),
        ("xai", Provider::Xai),
        ("fishaudio", Provider::FishAudio),
        ("fishaudio_drama", Provider::FishAudioDrama),
        ("gemini", Provider::Gemini),
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
    assert_eq!(Markup::from_provider(Provider::Xai).lower("done </expr> here").text, "done here");
    // `\s*` inside the closer, and no `\b` after `expr`, so `</expression>` is not one
    assert_eq!(strip_expr_markup("a </expr > b"), "a b");
    assert_eq!(strip_expr_markup("a </expression> b"), "a </expression> b");
}

#[test]
fn only_prosody_and_spell_markers_wrap_their_span() {
    // The prosody|spell lookahead decides wrapping; otherwise words between markers are swallowed.
    let x = Markup::from_provider(Provider::Xai);
    assert_eq!(
        x.lower("<expr type=\"prosody\" label=\"whisper\">psst</expr> ok").text,
        "<whisper>psst</whisper> ok"
    );
    // A self-closing marker then an orphaned `</expr>`: the pause must not vanish.
    assert_eq!(
        x.lower("<expr type=\"break\" label=\"1s\"/> words </expr>").text,
        "[long-pause] words "
    );
    assert_eq!(
        Markup::from_provider(Provider::Inworld)
            .lower("<expr type=\"expression\" label=\"warm\"/>hi</expr>")
            .text,
        "[warm] hi"
    );
    // and the span of a real wrapper runs to the *earliest* closer, not the last
    assert_eq!(
        x.lower("<expr type=\"prosody\" label=\"loud\">a</expr>b</expr>c").text,
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
            [r#"<emotion value="happy"/>x"#, "[happy] x", "x", "[very happy] x"],
        ),
        (r#"<expr type="sound" label="laugh"/>x"#, ["x", "[laugh] x", "[laugh] x", "[laughing] x"]),
        (
            // the alias tables: Inworld's "breathe" is xAI's "breath", Fish's "sigh" is "sighing"
            r#"<expr type="sound" label="breathe"/><expr type="sound" label="sigh"/>x"#,
            ["x", "[breathe][sigh] x", "[breath][sigh] x", "[breathe][sighing] x"],
        ),
        (
            r#"<expr type="break" label="1s"/>x"#,
            [
                r#"<break time="1s"/>x"#,
                r#"<break time="1s"/>x"#,
                "[long-pause] x",
                "[long-break] x",
            ],
        ),
        (
            r#"<expr type="prosody" label="slow">go</expr>"#,
            [r#"<speed ratio="0.85"/>go"#, "[slow] go", "<slow>go</slow>", "go"],
        ),
        (
            r#"<expr type="prosody" label="whispering">go</expr>"#,
            [r#"go"#, "[whispering] go", "go", "[whispering] go"],
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
                &Markup::from_provider(*p).lower(input).text,
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
    assert_eq!(f.lower("the <emphasis>ninth</emphasis>.").text, "the [emphasis] ninth.");
    assert_eq!(f.lower("<EMPHASIS> ninth </EMPHASIS>").text, "[emphasis] ninth");
    assert_eq!(f.lower(r#"<emphasis class="x">ninth</emphasis>"#).text, "[emphasis] ninth");
    // was: an unclosed one reached the TTS raw; now its delimiter goes and the word stays
    assert_eq!(f.lower("<emphasis>ninth").text, "ninth");
}

#[test]
fn fish_drama_keeps_free_form_directions_and_scopes_prosody_to_the_words() {
    let drama = Markup::from_provider_key("fishaudio_drama");
    assert_eq!(
        drama.lower(
            r#"<expr type="expression" label="a little sheepish, apologetic"/> I am <expr type="prosody" label="whispering">sorry</expr>. <expr type="sound" label="sigh"/>"#,
        ).text,
        "[a little sheepish, apologetic] I am <whisper>sorry</whisper>. [sighing]",
    );
    assert_eq!(
        drama.lower(r#"<expr type="prosody" label="hurried">Your total</expr> is <expr type="prosody" label="emphasis">nine</expr>."#).text,
        "<fast>Your total</fast> is <emphasis>nine</emphasis>.",
    );
    assert_eq!(
        drama.lower(r#"A <break time="500ms"/> B <break time="2s"/> C"#).text,
        "A [short pause] B [pause] C"
    );
    let instructions = drama.llm_instructions(None).unwrap();
    assert!(instructions.contains("a free-form direction"));
    assert!(instructions.contains("whisper, soft, slow, fast, emphasis"));
    assert!(!instructions.contains("regretful, hopeful, happy"));
    assert_eq!(strip_all_markup("<whisper>just</whisper> <stress>nine</stress>"), "just nine");
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
        assert_eq!(x.lower(&format!(r#"<break time="{time}"/>"#)).text, want, "time={time:?}");
    }
    // the tag itself needs `\s+` before `time` and accepts either closer
    assert_eq!(x.lower(r#"<break time="1s">"#).text, "[long-pause]");
    assert_eq!(x.lower(r#"<break time="1s" />"#).text, "[long-pause]");
    assert_eq!(x.lower(r#"<breaktime="1s"/>"#).text, r#"<breaktime="1s"/>"#);
    let f = Markup::from_provider(Provider::FishAudio);
    assert_eq!(f.lower("<break time=\"2s\"/>").text, "[long-break]");
    assert_eq!(f.lower("<break time=\"500ms\"/>").text, "[break]");
}

#[test]
fn silencing_every_sound_removes_the_section_for_each_provider_that_has_one() {
    // Each builder's `if sounds:` guard: all-off must drop the Sounds section.
    let off = SpeechSteering {
        nonverbal_sounds: Some(NonverbalSounds::Enabled(false)),
        ..Default::default()
    };
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
    assert_eq!(
        f.lower(r#"<expr type="prosody" label="Whispering"/>psst"#).text,
        "[whispering] psst"
    );
    assert_eq!(f.lower(r#"<expr type="prosody" label="nonsense"/>psst"#).text, "psst");
    // and the same label on a provider without tones is dropped rather than mistranslated
    assert_eq!(
        Markup::from_provider(Provider::Xai)
            .lower(r#"<expr type="prosody" label="whispering"/>x"#)
            .text,
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

// Catches: a lowering that lengthens a chunk, so a chunk the splitter kept within the provider's
// request limit (measured before lowering) would pass it once lowered.
#[test]
fn lowering_never_lengthens_a_chunk() {
    let markers = [
        r#"<expr type="expression" label="sad"/>"#,
        r#"<expr type="expression" label="very warm"/>"#,
        r#"<expr type="sound" label="laugh"/>"#,
        r#"<expr type="sound" label="hm"/>"#,
        r#"<expr type="break" label="1s"/>"#,
        r#"<expr type="break" label="300ms"/>"#,
        r#"<expr type="prosody" label="slow"/>"#,
        r#"<expr type="prosody" label="whispering">a few words</expr>"#,
        r#"<expr type="prosody" label="emphasis">nine</expr>"#,
        r#"<expr type="prosody" label="hurried">your total</expr>"#,
        r#"<expr type="spell">A7X9</expr>"#,
        r#"<emotion value="sad"/>"#,
        r#"<expression value="sad"/>"#,
        r#"<sound value="hm"/>"#,
        r#"<break time="1s"/>"#,
        r#"<emphasis>ninth</emphasis>"#,
        r#"<whisper>psst</whisper>"#,
        r#"<speed ratio="1.2"/>"#,
        r#"<spell>A7X9</spell>"#,
        "</expr>",
    ];
    let providers = [
        Provider::Cartesia,
        Provider::Inworld,
        Provider::Xai,
        Provider::FishAudio,
        Provider::FishAudioDrama,
        Provider::Gemini,
    ];
    for provider in providers.map(Some).into_iter().chain([None]) {
        let markup = Markup(provider);
        for marker in markers {
            let text = format!("Okay {marker} then.");
            let lowered = markup.lower(&markup.normalize(&text));
            let (got, had) = (lowered.text.chars().count(), text.chars().count());
            assert!(got <= had, "{provider:?} lengthened {text:?} to {:?}", lowered.text);
            assert!(lowered.said.chars().count() <= got, "{provider:?}: said past text");
        }
    }
}

// Catches: a bracket cue lowered against the word the model glued its marker to, so the provider
// read `[laugh]Sure` as one token and echoed it as one word in its timestamps (survey seed 152).
#[test]
fn a_bracket_cue_is_its_own_token() {
    let laugh = r#"<expr type="sound" label="laugh"/>Sure."#;
    assert_eq!(Markup::from_provider(Provider::Inworld).lower(laugh).text, "[laugh] Sure.");
    let warm = r#"<expr type="expression" label="warm"/>Ref"#;
    assert_eq!(Markup::from_provider(Provider::FishAudio).lower(warm).text, "[very warm] Ref");
    // a cue before a blank, or an XML tag, stays as it is
    assert_eq!(
        Markup::from_provider(Provider::Inworld)
            .lower(r#"<expr type="sound" label="laugh"/> Sure."#)
            .text,
        "[laugh] Sure."
    );
    let happy = r#"<expr type="expression" label="happy"/>Sure."#;
    assert_eq!(
        Markup::from_provider(Provider::Cartesia).lower(happy).text,
        "<emotion value=\"happy\"/>Sure."
    );
}
