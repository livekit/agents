//! A closing session's tasks end within its cleanup bound: its close cancels nothing still running.
use lk_agents_fuzz::drive::{Case, Outcome, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Models, Ref, SayInput};

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

// Catches: a provider stream a handoff was still opening when the session closed kept its open
// running, which nothing would take any more, past the close's cleanup bound (survey seed 313).
#[test]
fn a_close_abandons_a_stream_still_opening() {
    let outcome = closed(
        Setup { stt: true, connect_ms: Some(30_000), ..Setup::default() },
        vec![Input::UpdateModels { models: Models::All }, Input::Advance { ms: Millis(20) }],
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
