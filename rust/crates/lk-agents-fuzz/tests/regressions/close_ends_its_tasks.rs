//! A closing session's tasks end within its cleanup bound: its close cancels nothing still running.
use lk_agents_fuzz::drive::{Case, Outcome, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

// Catches: an interrupted `say` whose caller's text stream stays open went on reading it, and its
// voice, its synthesis failed, went on waiting for the words, so the speech's task outlived the
// close's release and was cancelled at the cleanup bound (`every_task_finishes`, 82 of the first
// 1,024 survey seeds once the DST ran the runtime's real close, e.g. seed 15). `_tts_task`
// cancels its forwarding at the interruption.
#[test]
fn an_interrupted_say_stops_reading_its_stream() {
    let outcome = closed(
        Setup { tts: true, ..Setup::default() },
        vec![
            Input::TtsOutage { down: true },
            Input::Say {
                text: AgentText(String::new()),
                input: SayInput::TextStream,
                allow: None,
                to_ctx: true,
            },
            Input::FeedText { stream: Ref::new(0), text: AgentText("One moment please.".into()) },
            Input::Advance { ms: Millis(2_000) },
        ],
    );
    assert!(outcome.ok(), "{:#?}", outcome.failures);
}

/// `inputs`, then the close, waited out before anything still pending is released.
fn closed(setup: Setup, mut inputs: Vec<Input>) -> Outcome {
    inputs.extend([
        Input::Close { drain: false },
        Input::Advance { ms: Millis(20_000) },
        Input::AwaitExit {},
    ]);
    run(Case { setup, inputs, ..Case::default() })
}
