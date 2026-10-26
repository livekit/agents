//! A speech cut by a close the session began on its own has a cause.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_voice::trace::{CloseStep, InterruptCause, InterruptStep, Step};

// Catches: `interruption_has_cause` permitting a `Closing` interruption only after the case's
// `Close`. The error policy closes the session on its own (`CloseReason::Error`), and the speech
// it cuts is as caused as one a host close cuts (survey seed 0, 547 failures over seeds 0 to 1023).
#[test]
fn a_speech_cut_by_the_error_policy_close_is_caused() {
    let setup = Setup { max_unrecoverable_errors: 0, ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        // Words already out: the failure is not retried.
        Input::LlmText { chat: Ref::new(0), text: AgentText("Hello".into()) },
        Input::LlmFail { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let close = journal
        .iter()
        .position(|(_, s)| matches!(s, Step::Close(CloseStep::Requested { .. })))
        .expect("the error policy began the close");
    let cut = journal
        .iter()
        .position(|(_, s)| {
            matches!(
                s,
                Step::Interrupt(InterruptStep::Interrupted { cause: InterruptCause::Closing, .. })
            )
        })
        .expect("the close cut the speech");
    assert!(close < cut);
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"interruption_has_cause"), "{failed:?}");
}
