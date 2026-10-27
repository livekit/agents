//! A background tool's update that lands before its step's join heard it detach is kept.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
use lk_agents_voice::events::{SessionEvent, ToolLifecycle};
use lk_agents_voice::trace::Step;

// Catches: `ctx.update()` reaches the driver directly while the detach is heard at the join, in
// the model's order: with a blocking call ahead of it, the second call's update arrived first and
// was refused (`UpdateRefused`), never told nor replied to (finding 3 of the tools review).
#[test]
fn an_update_ahead_of_the_join_is_told() {
    let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
    let call = |args: &str| Input::LlmToolCall {
        chat: Ref::new(0),
        name: "lookup".into(),
        args: ToolArgs(args.into()),
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(100) },
        call("{}"),
        call(r#"{"seconds":3}"#),
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        // The second call detaches and updates while the first still runs.
        Input::ToolAnswered {
            tool: Ref::new(1),
            answer: ToolAnswer::Detached,
            text: AgentText("Checking.".into()),
        },
        Input::Advance { ms: Millis(10) },
        Input::ToolUpdate {
            tool: Ref::new(1),
            progress: Some(AgentText("Found it.".into())),
            end: ToolAnswer::Return,
        },
        Input::Advance { ms: Millis(100) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("12C".into()),
        },
        Input::Advance { ms: Millis(3_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let told: Vec<String> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                id,
                ..
            })) => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert!(told.iter().any(|id| id.ends_with("_update_1")), "updates told: {told:?}");
}
