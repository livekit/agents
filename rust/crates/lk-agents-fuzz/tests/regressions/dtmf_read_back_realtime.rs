//! A DTMF read-back on a realtime model is answered on its socket.
use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
use lk_agents_fuzz::input::{Count, DtmfInput, DtmfKey, Input, Millis};

// Catches: the confirm arm's read-back built a pipeline reply on a realtime session, which has no
// LLM, so it failed with "the session has no llm" though nothing failed (`speech_error_is_injected`,
// survey seeds 320, 621, 731). `GetDtmfTask` calls `session.generate_reply()`, which a realtime
// model answers on its own socket.
#[test]
fn a_read_back_on_a_realtime_model_is_spoken() {
    let setup =
        Setup { realtime: Some(RealtimeSetup { turn_detection: false }), ..Setup::default() };
    // One digit, read back for confirmation.
    let dtmf = |input, key| Input::Dtmf {
        input,
        key,
        digits: Count(0),
        confirm: true,
        timeout: Millis(120),
    };
    let inputs = vec![
        dtmf(DtmfInput::Start, DtmfKey::Pound),
        Input::Advance { ms: Millis(20) },
        dtmf(DtmfInput::Digit, DtmfKey::Two),
        Input::Advance { ms: Millis(1_000) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    assert!(outcome.ok(), "{:#?}", outcome.failures);
    let spoke = outcome.reached.iter().any(|v| v == "Realtime::GenerationStatus");
    assert!(spoke, "no realtime generation answered the read-back");
}
