//! A close longer than 30 s of virtual time still ends every task, once the session exits.
use lk_agents_fuzz::drive::{Case, RoomSetup, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Models, Ref, SayInput, ToolArgs, UserText};

// Catches: the run joining tasks with the session frozen mid-close (campaign seed 2037). Every
// phase of this close runs to its bound: the inline task's unwind (5 s), the `on_exit` nobody
// answers (the 15 s hook bound), the drain waiting on replies parked behind the caller's open
// utterance (5 s), the edge waiting on a tool's `function_tools_executed` (5 s), then the trailing
// transcript. That outlasted the 30 s `AwaitExit` used to wait, and the room forwarder, the
// drains and every reply task were reported as never finishing although the close would have
// ended them.
#[test]
fn a_close_longer_than_the_wait_ends_every_task() {
    let line = |text: &str| AgentText(text.into());
    let inputs = vec![
        Input::GenerateReply {
            user_input: None,
            instructions: Some(line("Greet.")),
            allow: Some(false),
        },
        Input::Say { text: line("Sure."), input: SayInput::Words, allow: None, to_ctx: true },
        Input::GenerateReply { user_input: None, instructions: Some(line("Recap.")), allow: None },
        Input::GenerateReply {
            user_input: Some(UserText("can you hear me".into())),
            instructions: None,
            allow: None,
        },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Task { agent: "lookup_order".into(), speech: None },
        Input::SttStart { stt: Ref::new(0) },
        Input::UpdateModels { models: Models::Stt },
        Input::Advance { ms: Millis(3_000) },
        Input::LlmToolCall {
            chat: Ref::new(3),
            name: "caller".into(),
            args: ToolArgs("{}".into()),
        },
    ];
    let setup = Setup {
        stt: true,
        hook_bound_ms: Some(15_000),
        room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: true }),
        ..Setup::default()
    };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let stuck: Vec<_> =
        outcome.failures.iter().filter(|f| f.contract == "every_task_finishes").collect();
    assert!(stuck.is_empty(), "{stuck:#?}");
}
