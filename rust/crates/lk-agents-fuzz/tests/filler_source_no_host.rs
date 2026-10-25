//! A filler whose callable has no host to run it ends at its first fire.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};
use lk_agents_voice::trace::{FillerEnd, GenerationStep, Step};

// Catches: a `FillerSource` hook nobody can answer taken as a declined fire. With no delay and no
// interval the scope cooled down, dwelt and fired again in the same instant, forever: the driver
// looped without virtual time moving (survey seed 4372 at 5120 seeds).
#[test]
fn a_filler_callable_without_a_host_ends_the_filler() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::FillerStart {
            tool: Ref::new(0),
            speech: Ref::new(0),
            delay: Millis(0),
            interval: Some(Millis(0)),
            text: None,
        },
        Input::Advance { ms: Millis(50) },
    ];
    // No `hook_bound_ms`: the session has no host to call hooks on.
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    assert_eq!(run.record.livelock(), None);
    let ends: Vec<_> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Generation(GenerationStep::FillerFired { .. }) => Some(None),
            Step::Generation(GenerationStep::FillerStopped { cause, .. }) => Some(Some(*cause)),
            _ => None,
        })
        .collect();
    assert_eq!(ends, [Some(FillerEnd::Refused)], "no fire, the filler ends");
}
