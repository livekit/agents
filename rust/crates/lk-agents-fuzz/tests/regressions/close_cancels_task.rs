//! A close that cancels a running task cuts the task's speech as the close, not as the host.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

// Catches: the close's `AgentTask.cancel()` journalling its forced interrupt as `Host`, an
// interruption no host call asked for (`interruption_has_cause`, survey seeds 3 and 979).
#[test]
fn a_task_cancelled_by_the_close_is_cut_for_closing() {
    let inputs = [
        Input::Task { agent: "transfer".into(), speech: None },
        Input::Advance { ms: Millis(20) },
        Input::Say {
            text: AgentText("Let me pull up your account -- one moment.".into()),
            input: SayInput::Words,
            allow: Some(true),
            to_ctx: true,
        },
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: true },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let causes: Vec<InterruptCause> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Interrupt(InterruptStep::Interrupted { cause, .. }) => Some(*cause),
            _ => None,
        })
        .collect();
    assert!(!causes.is_empty(), "the task's speech was cut");
    assert!(causes.iter().all(|c| *c == InterruptCause::Closing), "{causes:?}");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"interruption_has_cause"), "{failed:?}");
}
