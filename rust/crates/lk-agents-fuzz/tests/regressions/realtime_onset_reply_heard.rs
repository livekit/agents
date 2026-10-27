//! The reply to a turn the realtime server heard over a playing reply is heard.
use lk_agents_fuzz::drive::{Case, RealtimeSetup, Setup, run};
use lk_agents_fuzz::input::{Count, Input, Millis, Ref, RtEvent};

// Catches: c6881abb9's silence, as the person on the call met it (`speech_is_heard`). The
// server's onset over a reply still playing (its generation long done) paused the reply instead
// of interrupting it; the person's turn then interrupted the paused reply, and that interrupt's
// cancel reached the server after it had created its reply to the turn: the new response was
// cancelled before a word of it played, and the agent said nothing.
#[test]
fn the_reply_to_a_turn_over_a_playing_reply_is_heard() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
        pausable_output: true,
        ..Setup::default()
    };
    let inputs = vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        // Past the echo canceller's warm-up; the person talks over the reply, then stops.
        Input::Advance { ms: Millis(4_000) },
        Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SpeechStarted },
        Input::Advance { ms: Millis(1_000) },
        Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SpeechStopped },
        Input::Advance { ms: Millis(20_000) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_is_heard"), "{:#?}", outcome.failures);
    assert!(outcome.expectations.get("speech_is_heard").is_some_and(|n| *n > 0));
}
