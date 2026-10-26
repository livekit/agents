//! A close ends a DTMF collection as it begins, before its drain.
use lk_agents_fuzz::drive::{Case, run};
use lk_agents_fuzz::input::{AgentText, Count, DtmfInput, DtmfKey, Input, Millis, SayInput};

fn dtmf(input: DtmfInput, key: DtmfKey) -> Input {
    Input::Dtmf { input, key, digits: Count(0), confirm: false, timeout: Millis(120) }
}

// Catches: the collection kept listening until the close's hard edge, so a key pressed while the
// close drained completed it with a digit pressed after the session began closing (survey seed
// 2112). `GetDtmfTask.on_exit`, which the close runs first, stops listening.
#[test]
fn a_key_pressed_while_closing_is_not_collected() {
    let inputs = vec![
        dtmf(DtmfInput::Start, DtmfKey::Pound),
        // A speech the close's drain waits on, so the session is still up when the key comes.
        Input::Say {
            text: AgentText("Your balance is due on the fourteenth of next month.".into()),
            input: SayInput::Words,
            allow: Some(false),
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::Close { drain: true },
        dtmf(DtmfInput::Digit, DtmfKey::Two),
        Input::Advance { ms: Millis(1_000) },
    ];
    let outcome = run(Case { inputs, ..Case::default() });
    assert!(outcome.ok(), "{:#?}", outcome.failures);
}
