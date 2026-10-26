//! A replaced output still gets what the reference sends it: the rest of a step already running,
//! and a synchroniser's closing flushes.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};

fn clean(setup: Setup, inputs: Vec<Input>) {
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"output_replacement_is_clean"), "{failed:?}");
    assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
}

// Catches: `output_replacement_is_clean` counting a synchroniser's closing flushes on the text
// output it wrapped (one when the swap rotates it, one when the session closes it) as output
// escaping onto a detached sink (`_capture_task`'s `finally`; survey seeds 0, 34, 278).
#[test]
fn a_synchroniser_flushes_its_old_output_as_it_closes() {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let inputs = vec![
        Input::SetTranscriptionOutput { attach: true },
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: true },
    ];
    clean(setup, inputs);
}

// Catches: the same rule counting the rest of a step already playing, whose reply task had read
// the old output at its start, as output escaping (`_tts_task` and `_pipeline_reply_task_impl`
// read `self._session.output` once, agent_activity.py:3406, 3717; survey seeds 5, 9, 15).
#[test]
fn a_step_playing_at_the_swap_finishes_on_its_output() {
    let inputs = vec![
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Audio,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(20) },
        Input::FeedAudio { stream: Ref::new(0), chunks: Count(1) },
        Input::Advance { ms: Millis(20) },
        Input::SetAudioOutput { attach: true },
        Input::Advance { ms: Millis(20) },
        Input::FeedAudio { stream: Ref::new(0), chunks: Count(1) },
        Input::FeedEnd { stream: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
    ];
    clean(Setup::default(), inputs);
}
