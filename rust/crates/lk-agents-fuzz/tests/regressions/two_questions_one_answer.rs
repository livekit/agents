//! Two `InputRequired` calls in one step: the person's next message answers both.
use std::collections::BTreeMap;

use lk_agents_abi::ChatItem;
use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
};
use lk_agents_voice::events::SessionEvent;

// Catches: the record answering only the last question asked (`pending_input`) while the request
// answered every waiting output: the first call's output stayed waiting on the record forever.
#[test]
fn the_answer_is_every_waiting_calls_output() {
    let setup = Setup {
        vad: true,
        stt: true,
        turn: TurnMode::Stt,
        tools: vec!["lookup".into()],
        ..Setup::default()
    };
    let call = |label: &str| Input::LlmToolCall {
        chat: Ref::new(0),
        name: "lookup".into(),
        args: ToolArgs(format!(r#"{{"label":"{label}"}}"#)),
    };
    let ask = |tool, text: &str| Input::ToolAnswered {
        tool: Ref::new(tool),
        answer: ToolAnswer::AskVerbatim,
        text: AgentText(text.into()),
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(100) },
        call("day"),
        call("time"),
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        ask(0, "Which day?"),
        ask(1, "Which time?"),
        Input::Advance { ms: Millis(5_000) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("Monday at noon".into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(3_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let mut outputs = BTreeMap::new();
    for (_, ev) in run.record.host_view().events() {
        if let SessionEvent::ConversationItemAdded { item, .. } = ev
            && let ChatItem::FunctionCallOutput(o) = item.as_ref()
        {
            outputs.insert(o.call_id.clone(), o.output.clone());
        }
    }
    assert_eq!(outputs.len(), 2, "outputs: {outputs:?}");
    assert!(
        outputs.values().all(|o| o == "Monday at noon"),
        "an output still waits for the answer: {outputs:?}"
    );
    assert!(outputs.values().all(|o| !o.starts_with(WAITING_PREFIX)));
}
