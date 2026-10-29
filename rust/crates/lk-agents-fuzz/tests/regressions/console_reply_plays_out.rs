//! A console reply settles on the CLI's playback receipt.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

// Catches: the console refusing the CLI's unkeyed `AudioPlaybackFinished` (0ef7f361b): every
// spoken reply waited out its playback deadline and failed "playback report timed out", which the
// rules excused as a sink that lost its receipt. The CLI acks each flush it plays.
#[test]
fn a_console_say_finishes_on_the_cli_receipt() {
    let setup = Setup { console: true, ..Setup::default() };
    let say = |text: &str| Input::Say {
        text: AgentText(text.into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let inputs = vec![
        Input::Advance { ms: Millis(100) },
        say("Hello from the console."),
        Input::Advance { ms: Millis(5_000) },
        say("And a second reply."),
        Input::Advance { ms: Millis(5_000) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
