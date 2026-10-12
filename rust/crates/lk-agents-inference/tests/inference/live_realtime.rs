//! Tier 4 (§4): the real OpenAI Realtime API, over the real transport.
//!
//! `#[ignore]`d and behind a feature, for the same reason as
//! `lk-agents-inference/tests/live_gateway.rs`: CI has no credentials, and this is the only test
//! in the crate that can fail for a reason that is not a bug. Run it with:
//!
//! ```sh
//! set -a && . ~/livekit/.env && set +a
//! cargo test -p lk-agents-realtime --features live \
//!   --test live_realtime -- --ignored --nocapture
//! ```
//!
//! What it is for is narrow: the 57 offline tests pin the logic, and none of them can catch a
//! wrong assumption about the wire itself -- the URL shape, the `Authorization` scheme, or a
//! `session.update` the server silently rejects because a field moved between API versions. This
//! test asserts exactly those three and nothing else, because everything else is cheaper to test
//! offline.

#![cfg(feature = "live")]

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::TokioSpawner;
use lk_agents_abi::RealClock;
use lk_agents_inference::native::TungsteniteTransport;
use lk_agents_inference::testkit::NoImages;
use lk_agents_inference::realtime::{
    AudioDelta, Event, GenerationCreated, GenerationDone, NativeRealtime, OPENAI,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and reaches the network"]
async fn a_real_session_accepts_our_opening_update_and_answers() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("OPENAI_API_KEY unset; skipping");
        return;
    };

    let session = NativeRealtime::new(
        OPENAI,
        api_key,
        Arc::new(TungsteniteTransport),
        Arc::new(TokioSpawner::current()),
        Arc::new(RealClock::new()),
        Arc::new(NoImages),
    )
    .session(false)
    .await;

    let req_id = session.generate_reply(Some("Say the single word: pong.".into()));

    // Bounded, like every other await in this crate's tests. A live server that says nothing is a
    // failed test, not a hung one.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut saw_audio = false;
    loop {
        let Ok(Some(ev)) = tokio::time::timeout_at(deadline, session.next_event()).await else {
            panic!("no answer within 20 s; saw_audio={saw_audio}");
        };
        match ev {
            // The server rejecting the opening `session.update` arrives here and nowhere else.
            Event::Error(e) => panic!("the server rejected something: {}", e.message),
            Event::GenerationCreated(GenerationCreated { req_id: Some(got), .. }) => {
                assert_eq!(got, req_id, "the correlation survives a real round trip");
            }
            Event::AudioDelta (AudioDelta { .. }) => saw_audio = true,
            Event::GenerationDone (GenerationDone { status, usage, .. }) => {
                assert_eq!(status, lk_agents_inference::realtime::GenerationStatus::Completed);
                assert!(usage.output_tokens > 0, "the usage block is where we think it is");
                assert!(saw_audio, "an audio session that produced no audio");
                break;
            }
            _ => {}
        }
    }
    session.aclose().await;
}
