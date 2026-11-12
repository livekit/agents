use super::*;

const DEFAULT: ModelSyntax = ModelSyntax::DEFAULT;
const GEMMA: ModelSyntax = ModelSyntax {
    spans: &[Span { open: "<|channel>thought", close: Some("<channel|>"), why: Hidden::Reasoning }],
    ..ModelSyntax::DEFAULT
};
const PREFILLED: ModelSyntax = ModelSyntax { starts_open: true, ..ModelSyntax::DEFAULT };
/// A Hermes-style template: reasoning, text tool calls, Mistral's call prefix and its end token.
const HERMES: ModelSyntax = ModelSyntax {
    spans: &[
        ModelSyntax::THINK,
        Span { open: "<tool_call>", close: Some("</tool_call>"), why: Hidden::ToolCall },
        Span { open: "[TOOL_CALLS]", close: None, why: Hidden::ToolCall },
    ],
    starts_open: false,
    turns: Some(ModelSyntax::CHATML),
    controls: &[],
};

fn run(tags: ModelSyntax, deltas: &[&str]) -> String {
    let mut lexer = Lexer::new(Rules::model(tags));
    let mut view = Projector::new(View::Model);
    let mut out: String = deltas.iter().map(|d| view.text(&lexer.push(d))).collect();
    out.push_str(&view.text(&lexer.finish()));
    out
}

#[test]
fn reasoning_is_hidden_however_it_is_chunked() {
    for (deltas, want) in [
        (&["<think>I should be brief</think>Hello there"][..], "Hello there"),
        (&["<thi", "nk>hidden", " more</thi", "nk>", "visible"], "visible"),
        (&["if a < b then", " done"], "if a < b then done"),
        (&["<think>one</think>middle<think>two</think>end"], "middleend"),
        (&["héllo 🎉 wörld"], "héllo 🎉 wörld"),
        (&["Hello <think>x</think> world"], "Hello world"),
    ] {
        assert_eq!(run(DEFAULT, deltas), want, "{deltas:?}");
    }
}

#[test]
fn a_stream_ending_inside_reasoning_or_on_part_of_a_delimiter_shows_none_of_it() {
    assert_eq!(run(DEFAULT, &["visible<think>cut off here"]), "visible");
    assert_eq!(run(DEFAULT, &["trailing <thi"]), "trailing ");
    assert_eq!(run(DEFAULT, &["visible<think>a</thi"]), "visible");
}

#[test]
fn a_stray_close_is_dropped() {
    assert_eq!(run(DEFAULT, &["a</think>b"]), "ab");
}

#[test]
fn a_prefilled_template_hides_everything_up_to_the_orphan_close() {
    assert_eq!(run(PREFILLED, &["the user wants", " X</thi", "nk>Sure."]), "Sure.");
    assert_eq!(run(PREFILLED, &["still reasoning"]), "");
}

#[test]
fn a_finished_lexer_starts_over() {
    let mut lexer = Lexer::new(Rules::model(PREFILLED));
    let mut view = Projector::new(View::Model);
    lexer.push("plan</think>one");
    lexer.finish();
    // a second stream through the same lexer starts inside reasoning again
    let mut out = view.text(&lexer.push("plan</think>two"));
    out.push_str(&view.text(&lexer.finish()));
    assert_eq!(out, "two");
}

#[test]
fn gemma_channel_delimiters_are_hidden_like_think_tags() {
    assert_eq!(run(GEMMA, &["<|chan", "nel>thought plan<chan", "nel|>Hi"]), "Hi");
}

#[test]
fn every_chunking_and_every_cut_agrees_with_the_whole() {
    let mut rng = fastrand::Rng::with_seed(0x7417);
    let pieces = ["Hi ", "there. ", "<think>", "</think>", "plan ", "a < b ", "<thi", "é "];
    for tags in [DEFAULT, PREFILLED, GEMMA] {
        for _ in 0..1500 {
            let text: String =
                (0..rng.usize(1..12)).map(|_| pieces[rng.usize(0..pieces.len())]).collect();
            let whole = run(tags, &[&text]);
            let per_char: Vec<String> = text.chars().map(String::from).collect();
            let per_char: Vec<&str> = per_char.iter().map(String::as_str).collect();
            assert_eq!(run(tags, &per_char), whole, "{text:?} one character at a time");
            for (cut, _) in text.char_indices() {
                let got = run(tags, &[&text[..cut]]);
                assert!(
                    whole.starts_with(got.trim_end()),
                    "{:?}: {got:?} vs {whole:?}",
                    &text[..cut]
                );
                for d in tags.spans.iter().flat_map(|s| std::iter::once(s.open).chain(s.close)) {
                    assert!(!got.contains(d), "{got:?}");
                }
            }
        }
    }
}

#[test]
fn tool_calls_written_as_text_are_hidden_and_marked_for_a_warning() {
    let text = "Sure.<tool_call>{\"name\": \"x\"}</tool_call> Done.<|im_end|>";
    assert_eq!(run(HERMES, &[text]), "Sure. Done.");
    let tokens = Lexer::all(Rules::model(HERMES), text);
    assert!(tokens.iter().any(|t| t.kind == Kind::Hidden(Hidden::ToolCall)));
    // a call prefix with no close runs to the end of the reply
    assert_eq!(run(HERMES, &["Okay [TOOL_CALLS]", "[{\"name\": \"x\"}]"]), "Okay ");
}

#[test]
fn template_tokens_drop_only_for_the_model_that_has_them() {
    assert_eq!(run(HERMES, &["a<|im_e", "nd|>b"]), "a");
    assert_eq!(run(DEFAULT, &["a<|im_end|>b"]), "a<|im_end|>b");
    let controls = ModelSyntax { controls: &["<|python_tag|>"], ..DEFAULT };
    assert_eq!(run(controls, &["a<|python_t", "ag|>b"]), "ab");
}

const LLAMA: ModelSyntax = ModelSyntax { turns: Some(ModelSyntax::LLAMA3), ..DEFAULT };

/// Every template, replies ending in the model inventing the user's next turn.
fn templates() -> [(ModelSyntax, &'static str, &'static str); 4] {
    [
        (
            HERMES,
            "<|im_start|>assistant\nSure, done.<|im_end|>\n<|im_start|>user\nthanks!<|im_end|>",
            "Sure, done.",
        ),
        (HERMES, "Sure, done.<|im_end|>\n<|im_start|>user\nthanks!", "Sure, done."),
        (
            LLAMA,
            "<|start_header_id|>assistant<|end_header_id|>\n\nSure, done.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nthanks!",
            "Sure, done.",
        ),
        (
            ModelSyntax::HARMONY,
            "<|start|>assistant<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>final<|message|>Sure, done.<|return|><|start|>user<|message|>thanks!",
            "Sure, done.",
        ),
    ]
}

#[test]
fn only_the_assistants_turn_is_said_however_it_arrives_and_wherever_it_is_cut() {
    for (syntax, reply, said) in templates() {
        assert_eq!(run(syntax, &[reply]), said, "{reply:?}");
        let chars: Vec<String> = reply.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(run(syntax, &chars), said, "{reply:?} one character at a time");
        for (cut, _) in reply.char_indices() {
            let got = run(syntax, &[&reply[..cut]]);
            assert!(said.starts_with(&got), "{:?}: {got:?}", &reply[..cut]);
        }
    }
}

#[test]
fn a_turn_opener_naming_no_known_role_hides_its_turn() {
    assert_eq!(run(HERMES, &["Hi<|im_end|><|im_start|>tool\n{}<|im_end|>"]), "Hi");
    assert_eq!(
        run(HERMES, &["<|im_start|>system\nrules<|im_end|><|im_start|>assistant\nHi"]),
        "Hi"
    );
}

const HARMONY: ModelSyntax = ModelSyntax::HARMONY;

#[test]
fn only_harmonys_final_channel_is_said() {
    // a header channel that is not final, and one with a recipient
    assert_eq!(
        run(HARMONY, &["<|start|>assistant<|channel|>commentary to=x<|message|>{}<|call|>"]),
        ""
    );
    let reply = "<|start|>assistant<|channel|>analysis<|message|>User wants X.<|end|>\
<|start|>assistant<|channel|>commentary to=functions.get<|constrain|>json<|message|>{\"a\":1}<|call|>\
<|start|>assistant<|channel|>final<|message|>Here you go.<|return|>";
    assert_eq!(run(HARMONY, &[reply]), "Here you go.");
    let chars: Vec<String> = reply.chars().map(String::from).collect();
    let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
    assert_eq!(run(HARMONY, &chars), "Here you go.");
}

#[test]
fn harmony_text_with_no_message_structure_is_said() {
    assert_eq!(run(HARMONY, &["Plain reply."]), "Plain reply.");
    // a message whose header the server stripped is the reply
    assert_eq!(run(HARMONY, &["<|message|>Hi<|end|>after"]), "Hi");
}

#[test]
fn a_harmony_reply_cut_anywhere_says_nothing_of_its_analysis() {
    let reply = "<|channel|>analysis<|message|>secret plan<|end|><|start|>assistant<|channel|>final<|message|>Hello there<|return|>";
    for (cut, _) in reply.char_indices() {
        let got = run(HARMONY, &[&reply[..cut]]);
        assert!("Hello there".starts_with(&got), "{:?}: {got:?}", &reply[..cut]);
    }
}

#[test]
fn a_harmony_turn_with_no_channel_is_said() {
    assert_eq!(run(HARMONY, &["<|start|>assistant<|message|>Hi<|end|>"]), "Hi");
}

#[test]
fn a_header_end_split_across_deltas_holds_only_its_start() {
    const TURNED: ModelSyntax = ModelSyntax {
        turns: Some(Turns {
            opens: &[Turn { open: "<|turn|>model", said: true }],
            header_end: Some("<|body|>"),
            ends: &["<|eot|>"],
            channel: None,
        }),
        ..DEFAULT
    };
    let mut lexer = Lexer::new(Rules::model(TURNED));
    lexer.push("<|turn|>model lang=en<|bo");
    assert_eq!(lexer.held(), "<|bo".len());
    assert_eq!(run(TURNED, &["<|turn|>model lang=en<|bo", "dy|>Hi"]), "Hi");
}

#[test]
fn the_blank_lines_after_a_header_are_hidden_as_they_arrive() {
    let mut lexer = Lexer::new(Rules::model(HERMES));
    let mut view = Projector::new(View::Model);
    lexer.push("<|im_start|>assistant\n");
    for _ in 0..500 {
        assert_eq!(view.text(&lexer.push("\n")), "");
        assert_eq!(lexer.held(), 0);
    }
    assert_eq!(view.text(&lexer.push(" Hi\n")), "Hi\n");
}

#[test]
fn a_whole_delimiter_is_settled_at_once() {
    let mut lexer = Lexer::new(Rules::model(DEFAULT));
    for d in ["Hi", "<think>", "plan", "</think>"] {
        lexer.push(d);
        assert_eq!(lexer.held(), 0, "after {d:?}");
    }
}

#[test]
fn a_final_channel_with_more_header_after_it_is_said() {
    for header in ["final<|constrain|>text", "final to=user"] {
        let reply = format!("<|start|>assistant<|channel|>{header}<|message|>Hi<|return|>");
        assert_eq!(run(HARMONY, &[&reply]), "Hi", "{reply:?}");
    }
}
