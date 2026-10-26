//! The interrupt a DTMF collection makes before its reply has a cause: the key pressed.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Count, DtmfInput, DtmfKey, Input, Millis, SayInput};
use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

fn dtmf(input: DtmfInput, key: DtmfKey) -> Input {
    Input::Dtmf { input, key, digits: Count(2), confirm: false, timeout: Millis(120) }
}

// Catches: `interruption_has_cause` blaming no one for the collection's `session.interrupt()`
// (`GetDtmfTask`'s debounced reply), which the driver journals as the host's: the person's key
// is what asked for it (survey seeds 106, 343, 858).
#[test]
fn a_dtmf_reply_interrupt_is_caused() {
    let inputs = [
        dtmf(DtmfInput::Start, DtmfKey::Pound),
        Input::Say {
            text: AgentText("Your balance is due on the fourteenth of next month.".into()),
            input: SayInput::Words,
            allow: Some(true),
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        dtmf(DtmfInput::Digit, DtmfKey::Two),
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let cut = run.record.journal().iter().any(|(_, s)| {
        matches!(s, Step::Interrupt(InterruptStep::Interrupted { cause: InterruptCause::Host, .. }))
    });
    assert!(cut, "the collection's reply interrupted the speech");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"interruption_has_cause"), "{failed:?}");
}
