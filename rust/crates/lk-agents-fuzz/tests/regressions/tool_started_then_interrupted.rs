//! A tool call reported started reaches the runner even when its speech is interrupted at once.
use lk_agents_fuzz::drive::{Policy, Run, Setup, simulate_with};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};

// Catches: the runner task of a scheduled speech's call racing adoption against interruption.
// The start was already journalled hosted and `ToolExecutionUpdated::Started` emitted, but an
// interruption landing before the task's first poll dropped the body unrun (survey seed 170;
// 9f070e85f). The race is the scheduler's, so 32 schedules of each queue policy run.
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
    let mut started = 0;
    for policy in [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge] {
        for schedule in 0..32 {
            let run = simulate_with(schedule, async {
                let mut run = Run::start(&Setup::default(), schedule, policy).await;
                for input in &inputs {
                    run.apply(input).await;
                }
                run
            });
            let r = &run.record;
            for (_, call_id, hosted) in r.internal_view().tool_calls_started() {
                started += 1;
                assert!(
                    !hosted || r.host_view().tool_ran(call_id),
                    "{policy:?} schedule {schedule}: the runner never saw {call_id}"
                );
            }
        }
    }
    assert!(started > 0, "the scenario: a call started before the interruption");
}
