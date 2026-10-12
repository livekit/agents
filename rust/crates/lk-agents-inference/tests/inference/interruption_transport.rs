//! The driver's lifecycle: reconnect, the retry ladder, and input teardown.
//!
//! These are the three things in `transport.rs` that no core test can reach, because each is a
//! property of the *loop* rather than of a decision. `retry.rs` proves the ladder's arithmetic; a
//! test here proves the loop actually climbs it.

use crate::support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::RealClock;
use lk_agents_inference::interruption::{
    AdaptiveInterruptionDetector, AUDIO_PREFIX_DURATION, ConnectionConfig, DETECTION_INTERVAL,
    DetectorEvent, InputEvent, InterruptionOptions, MAX_AUDIO_DURATION, MIN_INTERRUPTION_DURATION,
    REMOTE_INFERENCE_TIMEOUT, StreamClosed,
};
use lk_agents_abi::ApiConnectOptions;
use support::within;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use lk_agents_inference::headers::HEADER_SESSION_ID;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::Request;

fn options(base_url: String, threshold: Option<f64>) -> InterruptionOptions {
    InterruptionOptions::new(
        threshold,
        MIN_INTERRUPTION_DURATION,
        MAX_AUDIO_DURATION,
        AUDIO_PREFIX_DURATION,
        DETECTION_INTERVAL,
        REMOTE_INFERENCE_TIMEOUT,
        base_url,
        "devkey".to_owned(),
        "secret".to_owned(),
    )
    .unwrap()
}

/// Accepts connections forever, forwarding each `session.create` it receives with the
/// connection's `X-LiveKit-Session-ID`.
async fn recording_gateway() -> (u16, mpsc::UnboundedReceiver<(String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                let mut session_id = String::new();
                let record = |req: &Request, resp| {
                    session_id = req.headers().get(HEADER_SESSION_ID).map(|v| v.to_str().unwrap().to_owned()).unwrap_or_default();
                    Ok(resp)
                };
                let Ok(mut ws) =
                    within("the websocket handshake", tokio_tungstenite::accept_hdr_async(stream, record))
                        .await
                else {
                    return;
                };
                // Bounded: the loop ends when the client disconnects, but a wedged client must
                // not leave this task parked for the life of the run.
                let _ = within("the client to finish talking", async {
                    while let Some(Ok(msg)) = ws.next().await {
                        if let Message::Text(text) = msg {
                            let _ = tx.send((session_id.clone(), text.to_string()));
                            let _ = ws
                                .send(Message::Text(
                                    r#"{"type":"session.created","default_threshold":0.6}"#.into(),
                                ))
                                .await;
                        }
                    }
                })
                .await;
            });
        }
    });
    (port, rx)
}

/// A port with nothing listening on it.
async fn dead_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

// Catches: `update_options` not reaching live streams. The reference shares one options object
// between the detector and every stream, so the *only* way a new threshold gets to the gateway is
// the reconnect -- drop the notify (or invert `run`'s "reconnect or finish" branch) and the
// session keeps running at the old sensitivity for the rest of the call, silently. Also catches
// headers read once rather than per connection (`_connect_ws`): every reconnect would reuse the
// first session id.
#[tokio::test]
async fn update_options_forces_a_reconnect_that_carries_the_new_settings() {
    let (port, mut creates) = recording_gateway().await;
    let connections = Arc::new(AtomicUsize::new(0));
    let config = ConnectionConfig {
        headers: Arc::new(move || {
            let n = connections.fetch_add(1, Ordering::SeqCst);
            vec![(HEADER_SESSION_ID.to_owned(), format!("inference_{n}"))]
        }),
        ..Default::default()
    };
    let (detector, _events) = AdaptiveInterruptionDetector::new(
        options(format!("ws://127.0.0.1:{port}"), Some(0.4)),
        config,
        Arc::new(RealClock::new()),
    );
    let _stream = detector.stream();

    let (sid, first) = within("the first session.create", creates.recv()).await.unwrap();
    assert_eq!(sid, "inference_0");
    assert!(first.contains(r#""threshold":0.4"#), "{first}");
    assert!(first.contains(r#""min_frames":2"#), "{first}");

    detector.update_options(Some(0.85), Some(0.25));
    let (sid, second) = within("a reconnect after update_options", creates.recv()).await.unwrap();
    assert_eq!(sid, "inference_1", "each connection reads the headers again");
    assert!(second.contains(r#""threshold":0.85"#), "{second}");
    assert!(second.contains(r#""min_frames":10"#), "0.25s at 40 frames/s: {second}");

    // The per-stream nudge is the same path with no options change behind it -- `update_options`
    // reaches live streams through exactly this.
    _stream.request_reconnect();
    let (sid, third) = within("a reconnect after request_reconnect", creates.recv()).await.unwrap();
    assert_eq!(sid, "inference_2");
    assert_eq!(third, second, "a bare reconnect must re-send the current settings");
}

// Catches: the retry counter never advancing. `retry::decide` is pure and tested, but it is only
// correct if the driver increments `num_retries` -- leave that out and a gateway that is simply
// down produces an infinite reconnect loop that never reports anything, which is indistinguishable
// from "interruption is enabled and quiet". Also catches `emit_error` being dropped: the two
// recoverable errors before the fatal one are the only warning a user gets.
#[tokio::test]
async fn an_unreachable_gateway_climbs_the_ladder_and_then_gives_up() {
    let port = dead_port().await;
    let config = ConnectionConfig {
        conn_options: ApiConnectOptions::new(2, 0.01, 1.0).unwrap(),
        ..Default::default()
    };
    let (detector, mut events) = AdaptiveInterruptionDetector::new(
        options(format!("ws://127.0.0.1:{port}"), Some(0.5)),
        config,
        Arc::new(RealClock::new()),
    );
    let stream = detector.stream();

    let err = within(
        "the retry ladder to terminate -- num_retries may not be advancing",
        stream.finish(),
    )
    .await
    .expect_err("an unreachable gateway must fail the stream");
    assert_eq!(err.message, "failed to detect interruption after 2 attempts");

    let mut recoverable = 0;
    let mut fatal = 0;
    while let Ok(DetectorEvent::Error(e)) = events.try_recv() {
        if e.recoverable {
            recoverable += 1;
        } else {
            fatal += 1;
        }
        assert_eq!(e.label, "livekit.agents.inference.interruption.AdaptiveInterruptionDetector");
    }
    assert_eq!((recoverable, fatal), (2, 1), "two retries announced, then one give-up");
}

// Catches: `end_input` not actually closing the channel. The reference closes `_input_ch` so the
// send task can drain, write `session.close` and let the gateway hang up cleanly; a no-op leaves
// the socket open for the rest of the process and the caller with no way to tell.
#[tokio::test]
async fn end_input_closes_the_input_and_further_pushes_are_rejected() {
    let (port, _creates) = recording_gateway().await;
    let (detector, _events) = AdaptiveInterruptionDetector::new(
        options(format!("ws://127.0.0.1:{port}"), Some(0.5)),
        ConnectionConfig::default(),
        Arc::new(RealClock::new()),
    );
    let mut stream = detector.stream();

    stream.push(InputEvent::AgentSpeechStarted).unwrap();
    stream.end_input().unwrap();
    assert_eq!(stream.push(InputEvent::AgentSpeechEnded), Err(StreamClosed::InputEnded));
    assert_eq!(stream.flush(), Err(StreamClosed::InputEnded));
}
