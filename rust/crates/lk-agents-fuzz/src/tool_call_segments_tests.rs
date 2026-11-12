//! A reply that speaks, calls a tool and speaks again voices each side as its own segment.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs};

// Catches: the words before and after a call in one response fed to one synthesis, glued with
// no separator ("Let me check that for you.Your booking is confirmed."), so the voice's
// sentence splitter had to guess where the first sentence ended (livekit/agents#3804).
#[test]
fn the_call_ends_the_sentence_before_it() {
    let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
    let text = |text: &str| Input::LlmText { chat: Ref::new(0), text: AgentText(text.into()) };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(100) },
        text("Let me check that for you."),
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs(r#"{"label":"booking"}"#.into()),
        },
        text("Your booking is confirmed."),
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(3_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        // The speaker plays each segment out, so the next one is synthesized.
        for input in &inputs {
            crate::playout::apply_with_playout(&mut run, input).await;
        }
        run
    });
    let mut syntheses: Vec<(u32, String)> = Vec::new();
    for (_, tts, _, text) in run.record.produced() {
        match syntheses.last_mut() {
            Some((nth, said)) if nth == tts => said.push_str(text),
            _ => syntheses.push((*tts, text.clone())),
        }
    }
    let said: Vec<&str> = syntheses.iter().map(|(_, said)| said.trim()).collect();
    assert_eq!(said, ["Let me check that for you.", "Your booking is confirmed."]);
}
