use super::*;
use crate::tts::markup::{Provider, ProviderExt};

fn vocab(p: Provider) -> Vocabulary {
    Vocabulary::of(Some(p))
}

fn tags_only(v: &Vocabulary) -> Rules {
    Rules { tags: Some(v.clone()), ..Rules::default() }
}

/// The transcript of `text` under `v`, and the tags it carried.
fn display(v: &Vocabulary, text: &str) -> (String, Vec<Tag>) {
    let tokens = Lexer::all(tags_only(v), text);
    let shown = Projector::new(View::Display { keep_emoji: true }).text(&tokens);
    let tags = tokens.into_iter().filter_map(|t| match t.kind {
        Kind::Tag(tag) => Some(tag),
        _ => None,
    });
    (shown, tags.collect())
}

fn shown(v: &Vocabulary, text: &str) -> String {
    display(v, text).0
}

#[test]
fn a_stream_cut_inside_a_marker_shows_none_of_it() {
    let v = Vocabulary::LiveKit;
    assert_eq!(shown(&v, r#"Sure thing <expr type="sou"#), "Sure thing ");
    assert_eq!(shown(&v, "Sure thing <ex"), "Sure thing ");
    // a lone `<` at the end means nothing
    assert_eq!(shown(&v, "Sure thing <"), "Sure thing ");
}

#[test]
fn prose_angle_brackets_are_never_held_or_touched() {
    let v = vocab(Provider::Cartesia);
    let mut lexer = Lexer::new(tags_only(&v));
    let mut view = Projector::new(View::Display { keep_emoji: true });
    let mut out = String::new();
    for d in ["if a", "<b then ", "we ship. ", "3 < 5 and <3 and <b>bold</b>."] {
        out.push_str(&view.text(&lexer.push(d)));
        assert_eq!(lexer.held(), 0, "held after {d:?}");
    }
    out.push_str(&view.text(&lexer.finish()));
    assert_eq!(out, "if a<b then we ship. 3 < 5 and <3 and <b>bold</b>.");
}

#[test]
fn a_greater_than_inside_a_quoted_label_does_not_end_the_tag() {
    let (text, tags) =
        display(&Vocabulary::LiveKit, r#"Hi <expr type="expression" label="calm > angry"/> there"#);
    assert_eq!(text, "Hi there");
    assert_eq!(tags[0].attr("label"), Some("calm > angry"));
    assert_eq!(tags[0].render(), r#"<expr type="expression" label="calm   angry"/>"#);
}

#[test]
fn case_and_quote_style_do_not_hide_a_marker() {
    let v = Vocabulary::LiveKit;
    assert_eq!(shown(&v, "Hi <EXPR type='sound' label='laugh'/> there"), "Hi there");
    let (_, tags) = display(&v, "<Expr TYPE='sound' label='laugh' />");
    assert_eq!(tags[0].render(), r#"<expr type="sound" label="laugh"/>"#);
    let (_, tags) = display(&v, "<expr / >");
    assert_eq!(tags[0].render(), "<expr/>");
    let (_, tags) = display(&v, "<expr type = 'x' my_key='1' data-x='2'>");
    assert_eq!(tags[0].render(), r#"<expr type="x" my_key="1" data-x="2">"#);
    assert_eq!(shown(&v, "Hi <expr type='expression' label='it's fine'/> there"), "Hi there");
}

#[test]
fn a_repeated_attribute_keeps_its_last_value() {
    let (_, tags) = display(&Vocabulary::LiveKit, r#"<expr label="a" label="b"/>"#);
    assert_eq!(tags[0].attr("label"), Some("b"));
    assert_eq!(tags[0].attr("type"), None);
}

#[test]
fn only_the_active_voices_native_tags_are_markup() {
    let text = "Say <emphasis>this</emphasis> and <break time=\"1s\"/> and <speed ratio=\"1\"/>.";
    assert_eq!(shown(&vocab(Provider::FishAudio), text), "Say this and and <speed ratio=\"1\"/>.");
    assert_eq!(shown(&Vocabulary::LiveKit, text), text);
}

#[test]
fn a_name_needs_a_boundary_after_it() {
    assert_eq!(shown(&vocab(Provider::Inworld), "a <expressive> b"), "a <expressive> b");
    assert_eq!(shown(&Vocabulary::LiveKit, "a <exprs/> b"), "a <exprs/> b");
}

#[test]
fn a_botched_marker_loses_itself_and_keeps_the_words_after_it() {
    let v = Vocabulary::LiveKit;
    // a bare word: only the marker's own bytes go
    assert_eq!(
        shown(
            &v,
            r#"Hi <expr type="sound" label="laugh" there, <expr type="break" label="1s"/>ok"#
        ),
        "Hi there, ok"
    );
    // an unclosed quote: the model's `>` still ends it when the reply goes on
    assert_eq!(
        shown(&v, "Hi <expr type=\"sound\" label=\"laugh/> there.\nNext"),
        "Hi there.\nNext"
    );
    // at the end of the stream it is indistinguishable from a cut: dropped
    assert_eq!(shown(&v, r#"Hi <expr type="sound" label="laugh/> there"#), "Hi ");
    // a stray character where `>` belongs, then a `>` before any other `<`
    assert_eq!(shown(&v, r#"a <expr type="x" / junk> b"#), "a b");
    // a closing tag takes no attributes and does not close itself
    assert_eq!(shown(&v, "a </expr x> b"), "a b");
    assert_eq!(display(&v, "a </expr/> b"), ("a b".to_owned(), vec![]));
    assert_eq!(display(&v, "a </expr x='1'> b"), ("a b".to_owned(), vec![]));
    // an attribute needs a name
    assert_eq!(display(&v, "a <expr ='1'/> b"), ("a b".to_owned(), vec![]));
    // a stray character after `/` and no `>` to come: only the marker goes
    assert_eq!(shown(&v, "a <expr / x"), "a x");
    // no `=` after a name, and no quote after `=`
    assert_eq!(shown(&v, "a <expr type x> b"), "a b");
    assert_eq!(shown(&v, "a <expr type x><b"), "a <b");
    assert_eq!(shown(&v, "a <expr type=x> b"), "a b");
}

#[test]
fn a_recognized_tag_never_holds_past_its_limit() {
    let v = vocab(Provider::Xai);
    let mut lexer = Lexer::new(tags_only(&v));
    let mut view = Projector::new(View::Display { keep_emoji: true });
    assert_eq!(view.text(&lexer.push(r#"Hi <expr type="sound" label="laugh""#)), "Hi ");
    assert!(lexer.held() > 0);
    let out = view.text(&lexer.push(&" and then a long reply".repeat(20)));
    assert_eq!(lexer.held(), 0, "still held past the tag limit");
    assert!(out.contains("and then a long reply"));
    // an open quoted value past the limit
    let mut lexer = Lexer::new(tags_only(&v));
    lexer.push(r#"<expr label=""#);
    let out = view.text(&lexer.push(&"x".repeat(400)));
    assert_eq!(lexer.held(), 0);
    assert!(!out.contains("label"));
}

#[test]
fn gemini_names_with_a_space_are_read_whole() {
    let v = vocab(Provider::Gemini);
    assert_eq!(shown(&v, "Wait <short pause> now"), "Wait now");
    let mut lexer = Lexer::new(tags_only(&v));
    let mut view = Projector::new(View::Display { keep_emoji: true });
    assert_eq!(view.text(&lexer.push("Wait <short")), "Wait ");
    assert!(lexer.held() > 0);
    assert_eq!(view.text(&lexer.push(" pause> now")), "now");
}

#[test]
fn a_developer_vocabulary_adds_names_and_keeps_expr() {
    let v = Vocabulary::names(["Aside", " cue ", ""]);
    let text = "a <aside>quiet</aside> <CUE/> <expr type=\"sound\" label=\"x\"/>b <b>c</b>";
    assert_eq!(shown(&v, text), "a quiet b <b>c</b>");
    assert_eq!(Vocabulary::names([]), Vocabulary::LiveKit);
}

#[test]
fn all_reads_any_tag_shaped_text_as_markup() {
    let v = Vocabulary::All;
    assert_eq!(shown(&v, "a <b>bold</b> <my-tag x=\"1\"/>c"), "a bold c");
    assert_eq!(shown(&v, "3 < 5, <3, a<b."), "3 < 5, <3, a<b.");
    let mut lexer = Lexer::new(tags_only(&v));
    let mut view = Projector::new(View::Display { keep_emoji: true });
    assert_eq!(view.text(&lexer.push("x <bol")), "x ");
    assert_eq!(view.text(&lexer.push("d>y</bold>")), "y");
}

#[test]
fn every_vocabulary_names_expr_and_only_its_voices_natives() {
    for p in [
        Provider::Cartesia,
        Provider::Inworld,
        Provider::Xai,
        Provider::FishAudio,
        Provider::FishAudioDrama,
        Provider::Gemini,
    ] {
        let names = vocab(p).listed().expect("listed").to_vec();
        assert!(names.iter().any(|n| n == EXPR));
        assert_eq!(names.len(), 1 + p.native_tags().iter().filter(|n| **n != EXPR).count());
        assert!(names.windows(2).all(|w| w[0].len() >= w[1].len()), "longest first: {names:?}");
    }
    assert_eq!(Vocabulary::of(None), Vocabulary::LiveKit);
    assert_eq!(Vocabulary::All.listed(), None);
}

#[test]
fn the_synthesis_view_spells_each_tag_canonically() {
    let rules = tags_only(&Vocabulary::names(["emphasis"]));
    let tokens = Lexer::all(rules, "Say <EMPHASIS level='strong'>this</Emphasis > now");
    let synthesis = View::Synthesis { keep_markdown: false, keep_emoji: false };
    assert_eq!(
        Projector::new(synthesis).text(&tokens),
        r#"Say <emphasis level="strong">this</emphasis> now"#
    );
}

#[test]
fn tag_spans_count_characters_and_keep_a_botched_tag_whole() {
    let span = |start, end, closing, self_closing| TagSpan { start, end, closing, self_closing };
    assert_eq!(
        tag_spans(&Vocabulary::LiveKit, "é <expr/> a<b <expr type x> </expr>"),
        [span(2, 9, false, true), span(14, 27, false, true), span(28, 35, true, false)]
    );
}

#[test]
fn only_an_unfinished_tag_may_follow() {
    let v = Vocabulary::LiveKit;
    assert!(tag_may_follow(&v, "Hi <ex"));
    assert!(!tag_may_follow(&v, "Hi <b"));
    assert!(!tag_may_follow(&v, "Hi <expr/>"));
    assert!(!tag_may_follow(&v, "Hi"));
}

#[test]
fn the_pending_tag_is_the_first_one_still_unfinished() {
    let v = Vocabulary::LiveKit;
    assert_eq!(pending_tag(&v, "a <b <expr/> <ex"), Some(13));
    assert_eq!(pending_tag(&v, "a <expr type x> <ex"), Some(16));
    assert_eq!(pending_tag(&v, "x <expr label='y"), Some(2));
    assert_eq!(pending_tag(&v, "Hi <expr/> x <ex>"), None);
}

#[test]
fn a_tag_past_its_limit_drops_at_most_the_limit_and_keeps_the_rest_as_text() {
    let rules = tags_only(&Vocabulary::LiveKit);
    // the limit falls inside a two-byte character; a `>` in an earlier value does not end the tag
    for text in [
        format!("<expr label=\"{} tail", "é".repeat(200)),
        format!("<expr a='>' label='{} tail", "é".repeat(200)),
    ] {
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        for deltas in [vec![text.as_str()], chars] {
            let mut lexer = Lexer::new(rules.clone());
            let mut tokens: Vec<Token> = deltas.iter().flat_map(|d| lexer.push(d)).collect();
            tokens.extend(lexer.finish());
            let shown = Projector::new(View::Display { keep_emoji: true }).text(&tokens);
            assert!(text.ends_with(&shown) && shown.ends_with("é tail"), "{shown:?}");
            assert!(!shown.contains("label"), "{shown:?}");
            let dropped = text.len() - shown.len();
            assert!(dropped <= markup::MAX_TAG, "dropped {dropped}");
        }
    }
}

#[test]
fn what_no_rule_reads_is_prose() {
    let text = "*a* ~~b~~ `c` [d](e) !x _f_ 🚀 <b> <expr/> end";
    let spoken = View::Spoken { keep_markdown: false, keep_emoji: false };
    for (rules, want) in [
        (Rules::default(), text),
        (Rules::model(ModelSyntax::DEFAULT), text),
        (tags_only(&Vocabulary::LiveKit), "*a* ~~b~~ `c` [d](e) !x _f_ 🚀 <b> end"),
    ] {
        assert_eq!(
            Projector::new(spoken).text(&Lexer::all(rules.clone(), text)),
            want,
            "{rules:?}"
        );
    }
}
