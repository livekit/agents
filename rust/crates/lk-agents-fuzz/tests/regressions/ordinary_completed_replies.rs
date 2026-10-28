//! An ordinary tool's `Completed(value)` is its call's output, and the agent replies to it.
use lk_agents_abi::ChatItem;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
use lk_agents_fuzz::record::Kind;
use lk_agents_voice::events::SessionEvent;

// Catches: a `Completed` from a tool outside any VoiceTask taken as a task's completion, its
// output asking for no reply and the step ending there: the agent went silent (the host's
// `task` flag shadowed, finding 1 of the tools review). The fake host hands a value with every
// completion, so the core alone decides.
#[test]
fn an_ordinary_completion_is_replied_to() {
    let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(100) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs(r#"{"label":"order"}"#.into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Completed,
            text: AgentText("booked".into()),
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
    let outputs: Vec<(String, bool)> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::FunctionCallOutput(o) => Some((o.output.clone(), o.reply_required)),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(outputs, [("booked".to_owned(), true)], "outputs: {outputs:?}");
    assert_eq!(run.record.resources(Kind::Chat).count(), 2, "the follow-up was never asked");
}
