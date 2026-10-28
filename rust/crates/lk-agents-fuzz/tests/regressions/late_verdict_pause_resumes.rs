//! A speech paused by a detector verdict that came after the person stopped resumes.
use lk_agents_fuzz::drive::{Case, Outcome, RealtimeSetup, Run, Setup, finish, simulate};
use lk_agents_fuzz::input::{Count, Input, Millis, Ref};

// Catches: the adaptive detector's verdict landing after the person's end of speech paused the
// playing speech, and nothing ever armed its resume: the end of speech that arms it had already
// come. With no reply to the turn to interrupt it (scheduling paused by a drain), the agent sat
// silent mid-sentence for the rest of the call (`pause_resolves`, survey seed 273). The
// reference's transcript paths arm the resume for a pause decided after the end of speech; every
// pause decided with the person quiet now does.
#[test]
fn a_late_verdicts_pause_resumes() {
    // A realtime model with the session's own detector, recogniser and adaptive interruption.
    let setup = Setup {
        realtime: Some(RealtimeSetup::default()),
        stt: true,
        vad: true,
        interruption: true,
        pausable_output: true,
        ..Setup::default()
    };
    let outcome = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        // Warm up echo cancellation on a completed, observed first reply. The tested reply
        // then gets its overlap before its own missing-completion deadline can expire.
        run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None })
            .await;
        super::playout::apply_with_playout(&mut run, &Input::Advance { ms: Millis(10_000) }).await;
        let next_segment = run.record.live().opened(lk_agents_fuzz::record::Kind::Segment);
        let inputs = vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
            Input::PlayoutStarted { segment: Ref::new(next_segment) },
            Input::Advance { ms: Millis(500) },
            Input::Drain { wait: false },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(30) },
            Input::Advance { ms: Millis(300) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
            Input::Advance { ms: Millis(100) },
            // The detector rules after the person has stopped.
            Input::OverlapVerdict { port: Ref::new(0), is_interruption: true, agent_ended: false },
            Input::Advance { ms: Millis(10_000) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        finish(
            run,
            Outcome { case: Case { setup, inputs, ..Case::default() }, ..Outcome::default() },
        )
        .await
    });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"pause_resolves"), "{:#?}", outcome.failures);
    assert!(outcome.expectations.get("pause_resolves").is_some_and(|n| *n > 0), "no pause");
}
