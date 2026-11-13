//! Regression suites grouped by behavior, linked once for deterministic simulation tests.

#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    reason = "the DST harness: a broken fake is a test failure, and runs report to stdout"
)]

mod playout {
    //! The successful speaker in workflow regressions: report real queued PCM as the clock advances.
    use lk_agents_fuzz::drive::Run;
    use lk_agents_fuzz::input::{Input, Millis, Ref};
    use lk_agents_fuzz::record::{Kind, Sink};

    /// The fixture holds a segment until it is flushed, then plays its accepted duration at normal
    /// speed. Adversarial report-loss tests continue to drive these controls themselves.
    pub async fn apply_with_playout(run: &mut Run, input: &Input) {
        play(run, input, true).await;
    }

    /// The same speaker reporting only its start and its finish, never its progress, as
    /// `tests/fake_io.FakeAudioOutput` and many custom outputs do.
    pub async fn apply_with_bare_playout(run: &mut Run, input: &Input) {
        play(run, input, false).await;
    }

    async fn play(run: &mut Run, input: &Input, progress: bool) {
        let Input::Advance { ms } = input else {
            run.apply(input).await;
            return;
        };
        let mut remaining = ms.0;
        while remaining > 0 {
            let elapsed = remaining.min(100);
            remaining -= elapsed;
            run.apply(&Input::Advance { ms: Millis(elapsed) }).await;
            let now = run.record.now().at_ms;
            let pending = run.record.live().of(Kind::Segment);
            for segment in pending {
                let flushed = run.record.sink().iter().find_map(|(at, sink)| match sink {
                    Sink::Flush { segment: id, ms, .. } if *id == segment => Some((at.at_ms, *ms)),
                    _ => None,
                });
                let Some((at, duration)) = flushed else { continue };
                run.apply(&Input::PlayoutStarted { segment: Ref::new(segment) }).await;
                if progress {
                    run.apply(&Input::PlayoutProgressed {
                        segment: Ref::new(segment),
                        ms: Millis(elapsed),
                    })
                    .await;
                }
                if now >= at.saturating_add(duration) {
                    run.apply(&Input::PlayoutFinished { segment: Ref::new(segment) }).await;
                }
            }
        }
    }
}

#[path = "audio_tests.rs"]
mod audio;
#[path = "avatar_tests.rs"]
mod avatar;
#[path = "avatar_lifecycle_tests.rs"]
mod avatar_lifecycle;
#[path = "conversation_tests.rs"]
mod conversation;
#[path = "end_of_turn_tests.rs"]
mod end_of_turn;
#[path = "input_tests.rs"]
mod input;
#[path = "interruption_tests.rs"]
mod interruption;
#[path = "metrics_tests.rs"]
mod metrics;
#[path = "output_failure_tests.rs"]
mod output_failure;
#[path = "realtime_lifecycle_tests.rs"]
mod realtime_lifecycle;
#[path = "realtime_tools_tests.rs"]
mod realtime_tools;
#[path = "realtime_turns_tests.rs"]
mod realtime_turns;
#[path = "recording_tests.rs"]
mod recording;
#[path = "remote_session_tests.rs"]
mod remote_session;
#[path = "room_io_tests.rs"]
mod room_io;
#[path = "room_registration_tests.rs"]
mod room_registration;
#[path = "session_resources_tests.rs"]
mod session_resources;
#[path = "session_state_tests.rs"]
mod session_state;
#[path = "speech_completion_tests.rs"]
mod speech_completion;
#[path = "speech_playback_tests.rs"]
mod speech_playback;
#[path = "storage_fault_tests.rs"]
mod storage_fault;
#[path = "telephony_tests.rs"]
mod telephony;
#[path = "tool_answers_tests.rs"]
mod tool_answers;
#[path = "tool_call_segments_tests.rs"]
mod tool_call_segments;
#[path = "tool_lifecycle_tests.rs"]
mod tool_lifecycle;
#[path = "tool_speech_tests.rs"]
mod tool_speech;
#[path = "transcript_text_tests.rs"]
mod transcript_text;
#[path = "user_away_tests.rs"]
mod user_away;
#[path = "user_turn_tests.rs"]
mod user_turn;
