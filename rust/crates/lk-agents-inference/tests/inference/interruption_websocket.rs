//! One end-to-end pass over the real socket.
//!
//! Everything else in this crate is tested against [`InterruptionCore`] with no I/O, which is the
//! point of the split. This file exists because that leaves one thing unproven: that the driver
//! actually *wires* the core to a WebSocket -- handshake, `session.create`, the 8-byte binary
//! header, and a verdict coming back out as an `OverlappingSpeechEvent`. A unit test cannot
//! distinguish "the core is right" from "the core is right and never called", and every one of
//! those four steps is a silent failure in production: no barge-in, no error, no log.

use crate::support;

use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::errors::RefRegistry;
use lk_agents_abi::{HostRef, RealClock};
use lk_agents_inference::interruption::{
    AUDIO_PREFIX_DURATION, AdaptiveInterruptionDetector, AudioFrame, ConnectionConfig,
    DETECTION_INTERVAL, DetectorEvent, InputEvent, InterruptionOptions, MAX_AUDIO_DURATION,
    MIN_INTERRUPTION_DURATION, OverlapStart, REMOTE_INFERENCE_TIMEOUT, StreamEvent,
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

fn detector(port: u16) -> (AdaptiveInterruptionDetector, tokio::sync::mpsc::UnboundedReceiver<DetectorEvent>) {
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
fn push_one_overlap(stream: &lk_agents_inference::interruption::InterruptionStream, span: Option<HostRef>) {
    stream.push(InputEvent::AgentSpeechStarted).unwrap();
    stream.push(InputEvent::Audio(frame(SR as usize))).unwrap();
    stream
        .push(InputEvent::OverlapSpeechStarted(OverlapStart {
            speech_duration: 0.2,
            started_at: 1.0,
            user_speaking_span: span,
        }))
        .unwrap();
    stream.push(InputEvent::Audio(frame(1600))).unwrap();
}

/// A `/bargein` gateway that answers the first audio frame with a barge-in.
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
// with barge-in silently not working and nothing in the logs.
#[tokio::test]
async fn a_verdict_from_a_real_socket_reaches_the_consumer_as_an_event() {
    let (port, gateway) = fake_gateway().await;
    let (detector, mut detector_events) = detector(port);
    let mut stream = detector.stream();
    push_one_overlap(&stream, None);

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

    // Both the event and its metrics reach the detector's channel.
    let mut saw_event = false;
    let mut saw_metrics = false;
    while let Ok(ev) = detector_events.try_recv() {
        match ev {
            DetectorEvent::OverlappingSpeech(_) => saw_event = true,
            DetectorEvent::MetricsCollected(m) => {
                saw_metrics = true;
                assert_eq!((m.num_interruptions, m.num_backchannels), (1, 0));
                assert_eq!(m.num_requests, 1);
            }
            _ => {}
        }
    }
    assert!(saw_event && saw_metrics, "the detector must fan the verdict out to both channels");
}

// Catches: a host-driven stream's events reaching the detector's channel, where the session's
// reader takes them -- a user's `detector.stream()` during a session then loses its own error or
// span write to that reader -- or the span write going anywhere but the stream that pushed the
// span.
#[tokio::test]
async fn a_private_stream_keeps_its_verdict_and_span_write_to_itself() {
    struct NoRelease;
    impl RefRegistry for NoRelease {
        fn release(&self, _id: u64) {}
    }
    let registry: Arc<dyn RefRegistry> = Arc::new(NoRelease);

    let (port, gateway) = fake_gateway().await;
    let (detector, mut detector_events) = detector(port);
    let mut stream = detector.private_stream();
    push_one_overlap(&stream, Some(HostRef::new(7, &registry)));

    let span = within("the span write", stream.recv_event()).await;
    assert!(matches!(&span, Some(StreamEvent::SpanUpdate { span, .. }) if span.id() == 7), "{span:?}");
    let verdict = within("the verdict", stream.recv_event()).await;
    assert!(matches!(verdict, Some(StreamEvent::Overlap(_))), "{verdict:?}");
    within("the gateway task to finish", gateway).await.unwrap();
    assert!(detector_events.try_recv().is_err(), "the detector's channel is the session stream's alone");
}
