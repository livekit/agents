//! On a stored lane, the person's answer to a read-back is recorded as that call's output.
use lk_agents_abi::ChatItem;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs, TurnMode, UserText};
use lk_agents_voice::events::SessionEvent;

/// The fixture tool the runs declare with `needs_approval=True`.
const APPROVAL_TOOL: &str = "transfer";

// Catches: the preemptive reply to the person's answer planned its request with the answer already in the
// waiting output, rendered for the model ("not run yet: the person answered ..."). The lane's
// store gave that rendering back for the row it had not stored yet, and the session's record took
// it as the lane's edit: the output stopped waiting before the answer was recorded, so the stored
// output was the model's rendering, never the person's words (`tool_output_follows_outcome`,
// survey seed 21687).
#[test]
fn the_answer_is_the_read_backs_output() {
    // A store that keeps its lease for the whole call.
    let setup = Setup {
        tools: vec![APPROVAL_TOOL.into()],
        stt: true,
        vad: true,
        preemptive: true,
        turn: TurnMode::Stt,
        storage_lost_at: Some(1_000),
        ..Setup::default()
    };
    let answer = || UserText("yes please".into());
    let inputs = [
        Input::GenerateReply {
            user_input: Some(UserText("transfer me".into())),
            instructions: None,
            allow: None,
        },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: APPROVAL_TOOL.into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
        // The person answers; the reply to it is planned from their preflight words.
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(600) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
        Input::SttPreflight { utterance: Ref::new(0), text: answer(), speaker: None },
        Input::Advance { ms: Millis(50) },
        Input::SttFinal { utterance: Ref::new(0), text: answer(), speaker: None },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let outputs: Vec<String> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let asked = outputs.iter().any(|o| o.starts_with("waiting for the person's answer"));
    assert!(asked, "the scenario: the call waits on its read-back: {outputs:?}");
    assert_eq!(outputs.last().map(String::as_str), Some("yes please"), "{outputs:?}");
}
