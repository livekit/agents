//! `InputRequired(tool=)`: the person's next turn is a call of that tool, with their words.
use lk_agents_abi::ChatItem;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
};
use lk_agents_voice::events::SessionEvent;

// Catches: the claim never made on a call (`_claim_next_turn`, `_answer_arguments` absent): the
// person's answer went to the model to interpret, and the tool that asked was called with the
// model's arguments (finding 7 of the tools review).
#[test]
fn the_claimed_tool_takes_the_persons_words() {
    let setup = Setup {
        vad: true,
        stt: true,
        turn: TurnMode::Stt,
        tools: vec!["lookup".into()],
        ..Setup::default()
    };
    let call = |chat| Input::LlmToolCall {
        chat: Ref::new(chat),
        name: "lookup".into(),
        args: ToolArgs(r#"{"label":"guess"}"#.into()),
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(100) },
        call(0),
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::AskVerbatim,
            text: AgentText("Which day?".into()),
        },
        Input::Advance { ms: Millis(3_000) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal { utterance: Ref::new(0), text: UserText("Monday".into()), speaker: None },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(3_000) },
        // The turn's reply: the model calls the claimed tool with arguments of its own.
        call(1),
        Input::LlmEnd { chat: Ref::new(1) },
        Input::Advance { ms: Millis(100) },
        Input::ToolAnswered {
            tool: Ref::new(1),
            answer: ToolAnswer::Value,
            text: AgentText("Booked.".into()),
        },
        Input::Advance { ms: Millis(1_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let calls: Vec<String> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::FunctionCall(c) => Some(c.arguments.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(calls, [r#"{"label":"guess"}"#, r#"{"label":"Monday"}"#], "calls: {calls:?}");
}
