//! One end-to-end pass over the real socket.
//!
//! Everything else in this crate is tested against [`InterruptionMachine`] with no I/O, which is the
//! point of the split. This file exists because that leaves one thing unproven: that the driver
//! actually *wires* the core to a WebSocket -- handshake, `session.create`, the 8-byte binary
//! header, and a verdict coming back out as an `OverlappingSpeechEvent`. A unit test cannot
//! distinguish "the core is right" from "the core is right and never called", and every one of
//! those four steps is a silent failure in production: no interruption, no error, no log.

use crate::support;

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::turn::detector::LegEvent;
use lk_agents_abi::turn::interruption::{InterruptionSignal, InterruptionSource};
use lk_agents_abi::{Clock, RealClock, WallAnchor};
use lk_agents_inference::interruption::{
    AUDIO_PREFIX_DURATION, AdaptiveInterruptionDetector, AudioFrame, ConnectionConfig,
    DETECTION_INTERVAL, InputEvent, InterruptionOptions, MAX_AUDIO_DURATION,
    MIN_INTERRUPTION_DURATION, OverlapStart, REMOTE_INFERENCE_TIMEOUT,
};
use support::within;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

const SR: u32 = 16_000;

fn frame(n: usize) -> AudioFrame {
    AudioFrame {
        data: std::iter::repeat_n(1234_i16, n).flat_map(i16::to_le_bytes).collect(),
        sample_rate: SR,
        num_channels: 1,
        samples_per_channel: n as u32,
    }
}

fn detector(port: u16) -> AdaptiveInterruptionDetector {
    let opts = InterruptionOptions::new(
        Some(0.7),
        MIN_INTERRUPTION_DURATION,
        MAX_AUDIO_DURATION,
        AUDIO_PREFIX_DURATION,
        DETECTION_INTERVAL,
        REMOTE_INFERENCE_TIMEOUT,
        format!("ws://127.0.0.1:{port}"),
        "devkey".to_owned(),
        "secret".to_owned(),
    )
    .unwrap();
    AdaptiveInterruptionDetector::new(opts, ConnectionConfig::default(), Arc::new(RealClock::new()))
}

/// 1 s of agent-only audio, then the user starts over it for one detection interval: exactly one
/// request.
fn push_one_overlap(stream: &lk_agents_inference::interruption::InterruptionStream) {
    stream.push(InputEvent::AgentSpeechStarted).unwrap();
    stream.push(InputEvent::Audio(frame(SR as usize))).unwrap();
    stream
        .push(InputEvent::OverlapSpeechStarted(OverlapStart {
            speech_duration: 0.2,
            started_at: 1.0,
            user_speaking_span: None,
        }))
        .unwrap();
    stream.push(InputEvent::Audio(frame(1600))).unwrap();
}

/// A `/bargein` gateway that answers the first audio frame with a interruption.
///
/// Returns the port and a handle that yields the `session.create` the client sent, so the test can
/// assert on the handshake as well as on the verdict.
async fn fake_gateway() -> (u16, tokio::task::JoinHandle<(String, u64)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        let (stream, _) = within("a client to connect", listener.accept()).await.unwrap();
        let mut ws = within("the websocket handshake", tokio_tungstenite::accept_async(stream))
            .await
            .unwrap();

        let create = match within("session.create", ws.next()).await.unwrap().unwrap() {
            Message::Text(t) => t.to_string(),
            other => panic!("expected session.create, got {other:?}"),
        };
        within(
            "session.created to be written",
            ws.send(Message::Text(r#"{"type":"session.created","default_threshold":0.6}"#.into())),
        )
        .await
        .unwrap();

        let audio = within("an audio frame", async {
            loop {
                match ws.next().await.unwrap().unwrap() {
                    Message::Binary(b) => break b,
                    _ => continue,
                }
            }
        })
        .await;
        let created_at = u64::from_le_bytes(audio[..8].try_into().unwrap());
        // Echo the header back, which is the only way the client can match the verdict to the
        // request it timed.
        within(
            "the verdict to be written",
            ws.send(Message::Text(
                format!(
                    r#"{{"type":"bargein_detected","created_at":{created_at},"prediction_duration":0.02,"probabilities":[0.95,0.91,0.2]}}"#
                )
                .into(),
            )),
        )
        .await
        .unwrap();
        (create, (audio.len() - 8) as u64 / 2)
    });
    (port, handle)
}

// Catches: the driver never reaching the wire. Any of -- `session.create` not sent (the gateway
// waits forever), the audio sent without or with a byte-swapped 8-byte header (every verdict misses
// the cache and reports a zero RTT), the verdict parsed but never emitted to the consumer -- ends
// with interruption silently not working and nothing in the logs.
#[tokio::test]
async fn a_verdict_from_a_real_socket_reaches_the_consumer_as_an_event() {
    let (port, gateway) = fake_gateway().await;
    let detector = detector(port);
    let mut stream = detector.stream();
    push_one_overlap(&stream);

    let event = within("the verdict to reach the consumer", stream.recv())
        .await
        .expect("the stream closed without emitting");

    assert!(event.is_interruption);
    assert!(!event.agent_ended);
    assert_eq!(event.num_requests, 1);
    assert_eq!(event.prediction_duration, 0.02);
    // The 2nd-highest of [0.95, 0.91, 0.2].
    assert_eq!(event.probability, 0.91_f32 as f64);
    assert_eq!(event.overlap_started_at, Some(1.0));
    assert!(event.total_duration > 0.0, "the echoed header must produce a real round trip");
    // Wall clock, not the monotonic scheduling clock: `created_at`/`detected_at` are `time.time()`
    // in the reference and every consumer treats them as epoch seconds. A driver that fed the
    // monotonic clock in here would stamp every event a few milliseconds after 1970.
    assert!(
        event.created_at > 1_700_000_000.0,
        "created_at {} is not epoch seconds",
        event.created_at
    );
    assert!(event.detected_at > 1_700_000_000.0);
    // 1 s buffered (nothing trimmed: prefix 1.0 s + 0.2 s of speech already exceeds it) plus the
    // 0.1 s batch. The verdict has to land on *that* entry, not on a fresh one.
    assert_eq!(
        event.speech_input.as_ref().map(Vec::len),
        Some(SR as usize + 1600),
        "the cache entry the verdict landed on must be the one that carried the audio"
    );

    let (create, samples_sent) = within("the gateway task to finish", gateway).await.unwrap();
    assert_eq!(
        create,
        r#"{"type":"session.create","settings":{"sample_rate":16000,"num_channels":1,"threshold":0.7,"min_frames":2,"encoding":"s16le"}}"#
    );
    // 1.0 s prefix + 0.2 s of already-detected speech, trimmed out of the 1 s buffered, plus the
    // 0.1 s batch that triggered the send.
    assert_eq!(samples_sent, (1.1 * SR as f64) as u64);
}

// Catches: interruption metrics never reaching the session. They were sent only on the detector's
// own channel, which nothing drained, so `metrics_collected` never carried an `InterruptionMetrics`.
#[tokio::test]
async fn the_session_leg_reports_interruption_metrics_with_the_verdict() {
    let (port, gateway) = fake_gateway().await;
    let detector = detector(port);
    let clock = RealClock::new();
    let leg = detector.open(WallAnchor { wall: 1_787_000_000.0, at: clock.now() });
    leg.signal(InterruptionSignal::AgentSpeechStarted);
    leg.audio().push(frame(SR as usize)).unwrap();
    leg.signal(InterruptionSignal::OverlapStarted {
        started_at: clock.now(),
        speech_duration: Duration::from_millis(200),
    });
    leg.audio().push(frame(1600)).unwrap();

    let event = within("the verdict to reach the session", leg.next_event()).await;
    let LegEvent::Verdict { verdict, .. } = event else { panic!("expected a verdict: {event:?}") };
    let metrics = verdict.metrics.expect("the verdict carries the detector's metrics");
    assert_eq!((metrics.num_interruptions, metrics.num_backchannels), (1, 0));
    assert_eq!(metrics.num_requests, 1);
    assert!(metrics.total_duration > 0.0);
    within("the gateway task to finish", gateway).await.unwrap();
}
