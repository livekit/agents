//! A playout whose finish the output never reports settles once the output played all of it.
use lk_agents_fuzz::drive::{Case, Run, Setup, run, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

// Catches: an output that played the speech's audio and lost its "playback finished" report (a
// transport drops it) held the speech waiting on its playout until the session closed, and every
// speech queued behind it (DST seeds 229, 803 with the lost-finish fault).
#[test]
fn a_lost_finish_releases_the_speech_after_its_audio() {
    let inputs = [
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::Advance { ms: Millis(30_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the speech still waits on a finish the output lost");
}

// Catches: a sink that reset mid-speech, dropping what it held, then lost the finish of the
// speech's later audio held the speech until the close: the dropped audio never counted as
// played, so the margin never armed (DST seed 10521 with the lost-finish fault).
#[test]
fn a_lost_finish_after_a_reset_releases_the_speech() {
    let text = |t: &str| AgentText(t.into());
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: text("Sure, I can") },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::SinkReset { output: Ref::new(0) },
        Input::LlmText { chat: Ref::new(0), text: text(" help with that.") },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::PlayoutFinishLost { segment: Ref::new(1) },
        Input::Advance { ms: Millis(30_000) },
    ];
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the speech still waits on a finish the output lost");
}

// Catches: a speech cut after its output played all of its audio and lost the finish left that
// segment outstanding on the output's count, so every later playout waited for one finish more
// than it was owed: the next speech's real finish settled the old segment and the speech waited
// out the lost-finish margin (DST seeds 7836, 6883 with finishes lost after a reset).
#[test]
fn a_cut_after_a_lost_finish_keeps_the_next_playout_in_step() {
    let say = |text: &str| Input::Say {
        text: AgentText(text.into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let inputs = [
        say("Hello there."),
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::PlayoutFinishLost { segment: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(3_000) },
        say("Goodbye."),
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(1) },
        Input::PlayoutProgressed { segment: Ref::new(1), ms: Millis(10_000) },
        Input::PlayoutFinished { segment: Ref::new(1) },
        Input::Advance { ms: Millis(200) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [_, next] = &speeches[..] else { panic!("two speeches, got {}", speeches.len()) };
    assert!(next.is_done(), "the next speech waits for a finish its output already reported");
}

// Catches: the lost-finish margin started at the playout's next periodic look, up to a margin after
// the report that completed what the output owed, so a lost finish held the speech for twice the
// margin and the DST's release phase, waiting only on a journalled due time, never saw it go quiet
// (DST seed 6883 with finishes lost after a reset).
#[test]
fn a_lost_finish_is_due_a_margin_after_the_last_report() {
    let inputs = [
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(1_500) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::PlayoutFinishLost { segment: Ref::new(0) },
        Input::Advance { ms: Millis(2_100) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done(), "the lost finish was not due a margin after the last report");
}

// Catches: the speaker's own finish arriving after the playout's lost-finish margin had settled
// the segment for it was refused by the output fake, which had already let the segment go: the
// segment stayed open past the close and read as a leak (`nothing_left_live`, campaign seed
// 11256, its output replaced). A late report of a finish the product settled still ends the
// segment.
#[test]
fn a_finish_reported_after_the_margin_settles_the_segment() {
    let inputs = vec![
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
        Input::Advance { ms: Millis(5_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        // The output replaced: no clear at the close settles what it still holds.
        Input::SetAudioOutput { attach: true },
        Input::Advance { ms: Millis(100) },
        Input::Close { drain: true },
        Input::AwaitExit {},
    ];
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"nothing_left_live"), "{failed:?}");
}
