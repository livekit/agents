//! A replaced output still gets what the reference sends it: the rest of a step already running,
//! and a synchroniser's closing flushes; the swap itself interrupts nothing.
use lk_agents_abi::voice::events::AgentState;
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Case, Run, Setup, run, simulate};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};
use lk_agents_voice::events::{SessionEvent, SpeechOutcome};
use lk_agents_voice::trace::{InterruptCause, InterruptStep, PlayoutStep, Step};

fn clean(setup: Setup, inputs: Vec<Input>) {
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"output_replacement_is_clean"), "{failed:?}");
    assert!(outcome.expectations.get("output_replacement_is_clean").is_some_and(|n| *n > 0));
}

// A physically admitted old text destination gets its segment's closing flush while retained.
// The replacement must not force that work onto the new destination or retire it before delivery.
#[test]
fn a_synchroniser_flushes_its_old_output_as_it_closes() {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let inputs = vec![
        Input::Say {
            text: AgentText("Hello there, thanks for calling the front desk.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(200) },
        Input::SetTranscriptionOutput { attach: true },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
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

// Catches: a sink swap force-interrupting the speech playing into the old output. Replacement takes
// effect at a segment boundary and does not interrupt speech (voice-session.md "Output handover
// policy"): the playing segment finishes on its original output.
#[test]
fn a_swap_lets_the_playing_speech_finish() {
    let inputs = [
        Input::Say {
            text: AgentText("Hello there, thanks for calling the front desk.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(200) },
        Input::SetAudioOutput { attach: true },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let journal = run.record.journal();
    let swapped =
        journal.iter().any(|(_, s)| matches!(s, Step::Playout(PlayoutStep::OutputReplaced { .. })));
    assert!(swapped, "the output was replaced");
    let speaking = journal.iter().any(|(_, s)| {
        matches!(
            s,
            Step::Emitted(SessionEvent::AgentStateChanged { new: AgentState::Speaking, .. })
        )
    });
    assert!(speaking, "the agent was speaking when the output was replaced");
    let cut = journal.iter().any(|(_, s)| {
        matches!(
            s,
            Step::Interrupt(InterruptStep::Interrupted {
                cause: InterruptCause::OutputReplaced,
                ..
            })
        )
    });
    assert!(!cut, "the swap interrupted the playing speech");
    let completed = journal.iter().any(|(_, s)| {
        matches!(
            s,
            Step::Emitted(SessionEvent::SpeechDone { outcome: SpeechOutcome::Completed, .. })
        )
    });
    assert!(completed, "the speech completed");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(failed.is_empty(), "{failed:?}");
}
