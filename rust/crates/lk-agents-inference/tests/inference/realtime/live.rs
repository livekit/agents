//! Real-peer realtime protocol acceptance. Run with `cargo test -p lk-agents-runtime --test realtime_live -- --ignored`.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::llm::realtime::GenerationDone;
use lk_agents_inference::realtime::openai::{
    AudioDelta, Event, GenerationCreated, OPENAI, OpenAIRealtime,
};
use lk_agents_inference::testkit::NoImages;
use lk_agents_net::clock::RealClock;
use lk_agents_net::task::TokioSpawner;
use lk_agents_net::websocket::TungsteniteTransport;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and reaches the network"]
async fn a_real_session_accepts_our_opening_update_and_answers() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("OPENAI_API_KEY unset; skipping");
        return;
    };

    let session = <OpenAIRealtime>::new(
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
            Event::AudioDelta(AudioDelta { .. }) => saw_audio = true,
            Event::GenerationDone(GenerationDone { status, usage, .. }) => {
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
