//! A floor hold the host never released ends at the close's hard edge.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref};
use lk_agents_voice::trace::{CloseStep, SpeechStep, Step};

// Catches: a `wait_for_idle_and_hold()` granted before the session closes outliving it. Past the
// edge `ReleaseIdleHold` is dropped and nothing else released the hold (soak seed 6). The late
// release must stay harmless.
#[test]
fn a_hold_ends_with_the_session() {
    let inputs = [
        Input::WaitForIdleAndHold {},
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: false },
        Input::Advance { ms: Millis(6_000) },
        Input::ReleaseIdleHold { hold: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let granted = journal
        .iter()
        .find_map(|(_, s)| match s {
            Step::Speech(SpeechStep::IdleAnswered { hold: Some(hold), .. }) => Some(*hold),
            _ => None,
        })
        .expect("the hold was granted");
    let closing = journal
        .iter()
        .position(|(_, s)| matches!(s, Step::Close(CloseStep::Requested { .. })))
        .expect("the close was requested");
    let released: Vec<_> = journal
        .iter()
        .enumerate()
        .filter_map(|(i, (_, s))| match s {
            Step::Speech(SpeechStep::IdleHoldReleased { hold }) => Some((i > closing, *hold)),
            _ => None,
        })
        .collect();
    assert_eq!(released, [(true, granted)], "released once, by the close");
}
