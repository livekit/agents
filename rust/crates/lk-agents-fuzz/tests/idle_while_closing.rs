//! Idle reached during a close's drain grants no floor hold: the waiter is refused as closing.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::trace::{CloseStep, SpeechStep, Step};

// Catches: `settle_waiters` admitting idle waiters while the session closes. The reference's
// `wait_for_idle` raises `ActivityClosedError` once closing; here the drain's last speech ending
// granted a `wait_for_idle_and_hold()` its hold, which nothing could release past the edge, so it
// outlived the session (soak seed 0).
#[test]
fn idle_during_a_close_grants_no_hold() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
        Input::WaitForIdleAndHold {},
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: true },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ];
    let setup = Setup { hook_bound_ms: Some(1_000), ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let closing = journal
        .iter()
        .position(|(_, s)| matches!(s, Step::Close(CloseStep::Requested { .. })))
        .expect("the close was requested");
    let answers: Vec<_> = journal[closing..]
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Speech(SpeechStep::IdleAnswered { hold, closing }) => Some((*hold, *closing)),
            _ => None,
        })
        .collect();
    assert_eq!(answers, [(None, true)], "the waiter is refused, not granted");
}
