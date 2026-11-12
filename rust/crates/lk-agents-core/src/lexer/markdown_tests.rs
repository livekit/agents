use super::*;

const SPOKEN: View = View::Spoken;
const DISPLAY: View = View::Display;

fn markdown() -> Rules {
    Rules { markdown: true, emoji: true, ..Rules::default() }
}

fn view(view: View, deltas: &[&str]) -> String {
    let mut lexer = Lexer::new(markdown());
    let mut p = Projector::new(view);
    let mut out: String = deltas.iter().map(|d| p.text(&lexer.push(d))).collect();
    out.push_str(&p.text(&lexer.finish()));
    out
}

/// The former filter's table. `// was:` marks each intended change.
#[test]
fn markdown_is_not_spoken() {
    let cases: &[(&[&str], &str)] = &[
        (&["这是**重点**内容。"], "这是重点内容。"),
        (&["タイ語*強調*です。"], "タイ語強調です。"),
        (&["a ** b and *  x* here."], "a ** b and *  x* here."),
        // was: "\nprint(1)\n\n" -- code is shown, not spoken
        (&["```py\nprint(1)\n```\n"], ""),
        (&["say `unterminated here."], "say `unterminated here."),
        // was: "`ab and  ` x." -- a code span closes on a run of its own length
        (&["``a``b and `` `` x."], "ab and x."),
        // was: " but ~~ a~~ and xz." -- no leading blank where something was removed
        (&["~~gone~~ but ~~ a~~ and x~~y~~z."], "but ~~ a~~ and xz."),
        // was: "\n" -- a rule takes its line
        (&["***\n"], ""),
        (&["- - -\n"], ""),
        (&["___\n"], ""),
        (&["* * *x\n"], "* *x\n"),
        (&["see [a](b) and ![img](u) and [a] (b) end."], "see a and img and [a] (b) end."),
        (&["[open( never closes."], "[open( never closes."),
        (&["a_b_c and __bold__ and _ital_ and ***both***."], "a_b_c and bold and ital and both."),
        (
            &["* item\n1. one\n> quote\n#tag\n##  h2\n+ plus\n"],
            "item\n1. one\nquote\n#tag\nh2\nplus\n",
        ),
        (&["**", "bold", "** after"], "bold after"),
        // was: " y"
        (&["~~", "x", "~~ y"], "y"),
        (&["`", "code", "` z"], "code z"),
        // a closing run that grows past its opener's length closes nothing
        (&["``a``", "` z"], "``a``` z"),
        (&["[link", "](url) done"], "link done"),
        (&["![i", "](u) done"], "i done"),
        (&["see !", "[img](u)"], "see img"),
        (&["[a]", "(u) z"], "a z"),
        (&["see [a]"], "see [a]"),
        (&["*a<b* ok"], "a<b ok"),
        (&["*a*b*c*."], "*a*b*c*."),
        (&["_a_b_ c."], "_a_b_ c."),
        (&["**a**b."], "**a**b."),
        // was: "mixed  ok."
        (&["mixed ~~*x*~~ ok."], "mixed ok."),
        (&["\u{2014} dash *em* \u{2014}."], "\u{2014} dash em \u{2014}."),
        (&["tail with trailing *"], "tail with trailing *"),
        // was: "ok  done"
        (&["ok \u{2B50}\u{FE0F} done"], "ok done"),
        (&["한국어*강조*입니다 ｱｲ*ｳｴ*ｵ 豈*更*車"], "한국어강조입니다 ｱｲｳｴｵ 豈更車"),
    ];
    for (deltas, want) in cases {
        assert_eq!(view(SPOKEN, deltas), *want, "{deltas:?}");
    }
}

#[test]
fn the_transcript_keeps_markdown_and_emoji_as_written() {
    let text = "## Plan\n- **ship** it 🚀\n```sh\nmake\n```\nsee [docs](u)";
    assert_eq!(view(DISPLAY, &[text]), text);
    assert_eq!(view(SPOKEN, &["go 🚀 now"]), "go now");
}

#[test]
fn history_is_plain_and_keeps_code_and_link_targets() {
    let text = "## Plan\n- **ship** ~~not~~ it 🚀\n```sh\nmake\n```\nsee [docs](https://x.y)";
    assert_eq!(view(View::Plain, &[text]), "Plan\nship it 🚀\nmake\nsee docs (https://x.y)");
}

#[test]
fn markup_and_code_inside_emphasis_keep_their_meaning() {
    let rules = Rules::speech(Vocabulary::LiveKit);
    let text = r#"**wow <expr type="sound" label="a*b"/> `x*y`** done"#;
    let tokens = Lexer::all(rules, text);
    assert_eq!(Projector::new(SPOKEN).text(&tokens), "wow x*y done");
    assert!(
        tokens
            .iter()
            .any(|t| matches!(&t.kind, Kind::Tag(tag) if tag.attr("label") == Some("a*b")))
    );
}

#[test]
fn nested_emphasis_of_the_other_delimiter_is_read() {
    assert_eq!(view(SPOKEN, &["_**x**_ and **_y_**."]), "x and y.");
}

#[test]
fn a_spaced_or_intraword_delimiter_is_released_at_once() {
    let mut lexer = Lexer::new(markdown());
    let mut p = Projector::new(SPOKEN);
    assert_eq!(p.text(&lexer.push("5 * 3 is 15")), "5 * 3 is 15");
    assert_eq!(lexer.held(), 0);
    assert_eq!(p.text(&lexer.push(", 2*3*4")), ", 2*3*4");
    assert_eq!(lexer.held(), 0);
}

#[test]
fn an_inline_opener_waits_no_longer_than_its_line_and_window() {
    let mut lexer = Lexer::new(markdown());
    let mut p = Projector::new(SPOKEN);
    assert_eq!(p.text(&lexer.push("a *b")), "a ");
    assert!(lexer.held() > 0);
    // the line ends: the `*` was literal
    assert_eq!(p.text(&lexer.push(" c\nd")), "*b c\nd");
    let mut lexer = Lexer::new(markdown());
    lexer.push("a *b");
    let out = p.text(&lexer.push(&"x".repeat(INLINE_WINDOW + 1)));
    assert!(out.starts_with('*'), "{out:?}");
    assert_eq!(lexer.held(), 0);
}

#[test]
fn fenced_code_streams_to_the_transcript_line_by_line() {
    let mut lexer = Lexer::new(markdown());
    let mut shown = Projector::new(DISPLAY);
    let mut said = Projector::new(SPOKEN);
    let mut push = |d: &str| {
        let t = lexer.push(d);
        (shown.text(&t), said.text(&t))
    };
    assert_eq!(push("Here:\n```py\npri"), ("Here:\n```py\npri".into(), "Here:\n".into()));
    assert_eq!(push("nt(1)\n``"), ("nt(1)\n".into(), String::new()));
    // a line inside the block that only starts like a fence is content
    assert_eq!(push("`x\n"), ("```x\n".into(), String::new()));
    assert_eq!(push("```\nDone."), ("```\nDone.".into(), "Done.".into()));
}

#[test]
fn an_unclosed_fence_runs_to_the_end_and_is_never_spoken() {
    assert_eq!(view(SPOKEN, &["a\n```\ncode\nmore"]), "a\n");
    assert_eq!(view(SPOKEN, &["a\n~~~\ncode\n~~~~\nb"]), "a\nb");
    // a closing fence shorter than the opener is content
    assert_eq!(view(SPOKEN, &["````\nx\n```\ny\n````\nz"]), "z");
    // a closing fence with blanks after it, arriving piece by piece
    assert_eq!(view(SPOKEN, &["```\nx\n``", "` ", " \nafter"]), "after");
}

#[test]
fn line_markers_need_their_blank_and_their_line_start() {
    for (text, want) in [
        ("#tag", "#tag"),
        ("####### seven", "####### seven"),
        ("###### six", "six"),
        ("-not a list", "-not a list"),
        ("  - indented", "indented"),
        ("a - b", "a - b"),
        (">quote", ">quote"),
        ("> quote", "quote"),
        ("    ---", "    ---"),
        ("   ---", ""),
        ("--", "--"),
        // a backtick fence's info string has no backtick: this is a failed code span, literal
        ("```a`b", "```a`b"),
    ] {
        assert_eq!(view(SPOKEN, &[text]), want, "{text:?}");
    }
}

#[test]
fn a_link_needs_its_parts_on_one_line() {
    for (text, want) in [
        ("[a\n](b)", "[a\n](b)"),
        ("[a](b\n)", "[a](b\n)"),
        ("[a]x", "[a]x"),
        ("![a]x", "![a]x"),
        ("!x", "!x"),
        ("wow!", "wow!"),
        ("[f(x)](u)", "f(x)"),
        ("[**b**](u)", "b"),
    ] {
        assert_eq!(view(SPOKEN, &[text]), want, "{text:?}");
    }
}

#[test]
fn emphasis_edges() {
    for (text, want) in [
        ("****x****", "****x****"),
        ("*x **", "*x **"),
        ("**x*", "**x*"),
        ("*x*_", "*x*_"),
        ("_x_a", "_x_a"),
        ("*x\ny*", "*x\ny*"),
        ("a ~~~x~~~", "a ~~~x~~~"),
        ("~~x ~~", "~~x ~~"),
        ("~~x~", "~~x~"),
        ("~", "~"),
        ("`", "`"),
        ("``", "``"),
        ("`a\nb`", "`a\nb`"),
    ] {
        assert_eq!(view(SPOKEN, &[text]), want, "{text:?}");
    }
}

#[test]
fn emoji_sequences_are_one_token_however_they_arrive() {
    let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
    let mut lexer = Lexer::new(markdown());
    let mut tokens = lexer.push("hi \u{1F468}");
    tokens.extend(lexer.push("\u{200D}\u{1F469}\u{200D}\u{1F467} there"));
    tokens.extend(lexer.finish());
    let emoji: Vec<_> = tokens.iter().filter(|t| t.kind == Kind::Emoji).collect();
    assert_eq!(emoji.len(), 1);
    assert_eq!(emoji[0].raw, family);
}

#[test]
fn code_is_literal_even_when_it_quotes_markup() {
    let rules = Rules::speech(Vocabulary::LiveKit);
    let tokens = Lexer::all(rules, "use `<expr type=\"sound\"/>` like this");
    assert!(tokens.iter().all(|t| !matches!(t.kind, Kind::Tag(_))));
    assert!(tokens.iter().any(|t| t.kind == Kind::Prose && t.raw == "<expr type=\"sound\"/>"));
    assert_eq!(Projector::new(SPOKEN).text(&tokens), "use <expr type=\"sound\"/> like this");
}

#[test]
fn a_tilde_fence_at_a_line_start_opens_a_block() {
    assert_eq!(view(SPOKEN, &["~~~x~~~\ncode\n~~~\nafter"]), "after");
}

#[test]
fn a_construct_running_past_its_window_is_literal_however_it_arrives() {
    let long = "x".repeat(INLINE_WINDOW + 1);
    for text in [
        format!("`{long}` end"),
        format!("[{long}](u) end"),
        format!("*{long}* end"),
        // a rule or a fence is decided within the window too
        format!("{}\nend", "-".repeat(INLINE_WINDOW + 1)),
        format!("```{long}\nend"),
    ] {
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(view(DISPLAY, &[&text]), text);
        assert_eq!(view(SPOKEN, &[&text]), text, "{text:?}");
        assert_eq!(view(SPOKEN, &chars), text, "{text:?} one character at a time");
    }
}

#[test]
fn the_synthesis_view_reads_markdown_and_emoji_as_the_spoken_view_does() {
    let tokens = Lexer::all(markdown(), "**a** ~~b~~ `c` 🚀\n```\nd\n```\n");
    assert_eq!(
        Projector::new(View::Synthesis).text(&tokens),
        Projector::new(View::Spoken).text(&tokens),
    );
}

#[test]
fn a_line_marker_is_decided_within_the_window() {
    // the window counts the line's newline
    let rule = format!("{}\n", "-".repeat(INLINE_WINDOW - 1));
    assert_eq!(view(SPOKEN, &[&rule, "end"]), "end");
    let long = format!("{}\n", "-".repeat(INLINE_WINDOW));
    assert_eq!(view(SPOKEN, &[&long, "end"]), format!("{long}end"));
}

#[test]
fn an_emoji_run_longer_than_the_window_is_cut_at_it() {
    let tokens = Lexer::all(markdown(), &"\u{1F680}".repeat(INLINE_WINDOW));
    let runs: Vec<usize> = tokens.iter().map(|t| t.raw.len()).collect();
    assert_eq!(runs, [INLINE_WINDOW; 4]);
    assert!(tokens.iter().all(|t| t.kind == Kind::Emoji));
}

#[test]
fn markup_is_transparent_to_the_intraword_rule() {
    let rules = Rules::speech(Vocabulary::LiveKit);
    for (text, want) in [("a<expr/>_b_ c", "a_b_ c"), ("**x a<expr/>_b_ c**", "x a_b_ c")] {
        assert_eq!(Projector::new(SPOKEN).text(&Lexer::all(rules.clone(), text)), want, "{text:?}");
    }
}

#[test]
fn blanks_at_a_line_start_wait_for_what_follows_them() {
    let text = "a\n    - item\n  ";
    let chars: Vec<String> = text.chars().map(String::from).collect();
    let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
    assert_eq!(view(SPOKEN, &[text]), "a\nitem\n  ");
    assert_eq!(view(SPOKEN, &chars), "a\nitem\n  ");
}

#[test]
fn an_inline_closer_at_the_windows_last_byte_still_closes() {
    for n in [INLINE_WINDOW - 1, INLINE_WINDOW] {
        let x = "x".repeat(n);
        let u = "u".repeat(n - 3);
        let within = n < INLINE_WINDOW;
        for (text, spoken) in [
            (format!("`{x}`"), x.clone()),
            (format!("*{x}*"), x.clone()),
            (format!("~~{}~~", &x[1..]), String::new()),
            (format!("[a]({u})"), "a".to_owned()),
        ] {
            let want = if within { spoken } else { text.clone() };
            assert_eq!(view(SPOKEN, &[&text]), want, "{n}: {text:?}");
        }
    }
}
