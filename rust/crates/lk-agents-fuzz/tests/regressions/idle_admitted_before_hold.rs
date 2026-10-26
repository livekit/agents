//! A wait admitted ahead of a hold granted in the same idle instant is not judged as held off.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref};
use lk_agents_fuzz::record::Answer;

// Catches: `idle_respects_holds` judging a wait by when its answer reached the host. The driver
// admits the asks of one idle instant in ask order, so a `WaitForIdle` queued ahead of a
// `WaitForIdleAndHold` passes first and the hold is granted after it; both answers arrive after
// the pass, so the hold looked taken before the wait was answered (survey seeds 10 and 16).
#[test]
fn a_wait_queued_ahead_of_a_hold_passes_first() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::WaitForIdle { hold: None },
        Input::WaitForIdleAndHold {},
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let answers: Vec<_> = run
        .record
        .host_view()
        .api_calls()
        .iter()
        .filter_map(|c| c.returned().map(|(_, a)| a.clone()))
        .collect();
    assert!(answers.contains(&Answer::IdleOk), "{answers:?}");
    assert_eq!(run.record.host_view().holds().len(), 1, "the hold was granted");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"idle_respects_holds"), "{failed:?}");
}
