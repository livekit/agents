use super::*;

const SPOKEN: View = View::Spoken;
const DISPLAY: View = View::Display;

fn units(deltas: &[&str]) -> Vec<Unit> {
    let mut lexer = Lexer::new(Rules::speech(Vocabulary::LiveKit));
    let mut sync = Units::new(SPOKEN, DISPLAY);
    let mut out = Vec::new();
    for d in deltas {
        out.extend(sync.push(&lexer.push(d)));
    }
    out.extend(sync.push(&lexer.finish()));
    // neighbouring units of a kind are one stretch, however the deltas split them
    let mut merged: Vec<Unit> = Vec::new();
    for u in out {
        match (merged.last_mut(), &u) {
            (Some(Unit::Said(a)), Unit::Said(b)) => a.push_str(b),
            (Some(Unit::Silent { shown, tags }), Unit::Silent { shown: next, tags: t })
                if tags.is_empty() && t.is_empty() =>
            {
                shown.push_str(next)
            }
            _ => merged.push(u),
        }
    }
    merged
}

fn s(text: &str) -> Unit {
    Unit::Said(text.to_owned())
}

fn q(text: &str) -> Unit {
    Unit::Silent { shown: text.to_owned(), tags: Vec::new() }
}

#[test]
fn what_is_not_said_is_its_own_silent_unit_in_place() {
    assert_eq!(
        units(&["Say **very", " important** now 🎉"]),
        [s("Say "), q("**"), s("very important"), q("**"), s(" now "), q("🎉")]
    );
}

#[test]
fn a_link_url_and_unspoken_blocks_are_silent() {
    assert_eq!(
        units(&["see [docs](https://x.y) ok"]),
        [s("see "), q("["), s("docs"), q("](https://x.y)"), s(" ok")]
    );
    assert_eq!(
        units(&["Run:\n```sh\nmake\n```\nthen ~~stop~~ go."]),
        [s("Run:\n"), q("```sh\nmake\n```\n"), s("then "), q("~~stop~~"), s(" go.")]
    );
}

#[test]
fn spoken_text_leaves_with_its_delta() {
    let mut lexer = Lexer::new(Rules::speech(Vocabulary::LiveKit));
    let mut sync = Units::new(SPOKEN, DISPLAY);
    assert_eq!(sync.push(&lexer.push("hel")), [s("hel")]);
    assert_eq!(sync.push(&lexer.push("lo wo")), [s("lo wo")]);
}

#[test]
fn a_markers_delivery_takes_effect_in_place() {
    let units = units(&[r#"Hi <expr type="expression" label="warm"/> there"#]);
    let Unit::Silent { tags, .. } = &units[1] else { panic!("{units:?}") };
    assert_eq!(tags[0].attr("label"), Some("warm"));
    assert_eq!(units[0], s("Hi "));
    assert_eq!(units.last(), Some(&s("there")));
}

#[test]
fn emoji_kept_in_the_transcript_cost_the_voice_nothing() {
    assert_eq!(units(&["great 👍 job"]), [s("great "), q("👍"), s(" job")]);
}

#[test]
fn a_dropped_fragment_between_spoken_words_leaves_one_said_unit() {
    let tokens = Lexer::all(Rules::speech(Vocabulary::LiveKit), "a <expr type x> b");
    assert_eq!(Units::new(SPOKEN, DISPLAY).push(&tokens), [s("a b")]);
}

#[test]
fn a_markers_delivery_is_never_folded_into_the_silent_text_before_it() {
    let tokens = Lexer::all(Rules::speech(Vocabulary::LiveKit), "~~x~~<expr label='y'/> go");
    let units = Units::new(SPOKEN, DISPLAY).push(&tokens);
    let Unit::Silent { tags, .. } = &units[1] else { panic!("{units:?}") };
    assert_eq!(units[0], q("~~x~~"));
    assert_eq!(tags[0].attr("label"), Some("y"));
}
