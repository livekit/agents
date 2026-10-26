//! A tool call reported started reaches the runner even when its speech is interrupted at once.
use lk_agents_fuzz::drive::{Policy, Run, Setup, simulate_with};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};

// Catches: the runner task of a scheduled speech's call racing adoption against interruption.
// The start was already journalled hosted and `ToolExecutionUpdated::Started` emitted, but an
// interruption landing before the task's first poll dropped the body unrun (survey seed 170).
// Schedule 1 under `Pos` dispatches the call, then takes the interrupt before the task runs.
#[test]
fn a_started_call_runs_though_its_speech_is_interrupted() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate_with(1, async {
        let mut run = Run::start(&Setup::default(), 1, Policy::Pos).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let r = &run.record;
    let started = r.internal_view().tool_calls_started();
    assert_eq!(started.len(), 1, "the call was dispatched");
    let (_, call_id, hosted) = started[0];
    assert!(hosted);
    assert!(r.host_view().tool_ran(call_id), "the runner never saw {call_id}");
}
