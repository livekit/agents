use std::sync::Arc;

use crate::testkit::fallback::within;
use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::ApiConnectOptions;
use lk_agents_inference::interruption::{
    AUDIO_PREFIX_DURATION, AdaptiveInterruptionDetector, ConnectionConfig, DETECTION_INTERVAL,
    InputEvent, InterruptionOptions, MAX_AUDIO_DURATION, MIN_INTERRUPTION_DURATION,
    REMOTE_INFERENCE_TIMEOUT, StreamClosed, StreamEvent,
};
use lk_agents_net::clock::RealClock;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
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
/// connection's test marker.
async fn recording_gateway() -> (u16, mpsc::UnboundedReceiver<(String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                let mut session_id = String::new();
                #[allow(
                    clippy::result_large_err,
                    reason = "The WebSocket handshake callback requires this external response type"
                )]
                let record = |req: &Request, resp| {
                    session_id = req
                        .headers()
                        .get("X-LiveKit-Agent-ID")
                        .map(|v| v.to_str().unwrap().to_owned())
                        .unwrap_or_default();
                    Ok(resp)
                };
                let Ok(mut ws) = within(
                    "the websocket handshake",
                    tokio_tungstenite::accept_hdr_async(stream, record),
                )
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
// the job's identity read once rather than per connection (`_connect_ws`): every reconnect would
// miss the agent that joined after the first.
#[tokio::test]
async fn update_options_forces_a_reconnect_that_carries_the_new_settings() {
    let (port, mut creates) = recording_gateway().await;
    let cx = crate::testkit::job_cx(Default::default());
    let job = cx.job().cloned().unwrap();
    let detector = AdaptiveInterruptionDetector::new(
        options(format!("ws://127.0.0.1:{port}"), Some(0.4)),
        ConnectionConfig::default(),
        Arc::new(RealClock::new()),
    );
    let _stream = detector.stream(&cx);

    let (sid, first) = within("the first session.create", creates.recv()).await.unwrap();
    assert_eq!(sid, "", "the agent had not joined yet");
    assert!(first.contains(r#""threshold":0.4"#), "{first}");
    assert!(first.contains(r#""min_frames":2"#), "{first}");

    lk_agents_net::job_identity::agent_joined(&job, "PA_1".into());
    detector.update_options(Some(0.85), Some(0.25));
    let (sid, second) = within("a reconnect after update_options", creates.recv()).await.unwrap();
    assert_eq!(sid, "PA_1", "each connection reads the job's identity again");
    assert!(second.contains(r#""threshold":0.85"#), "{second}");
    assert!(second.contains(r#""min_frames":10"#), "0.25s at 40 frames/s: {second}");

    // The per-stream nudge is the same path with no options change behind it -- `update_options`
    // reaches live streams through exactly this.
    _stream.request_reconnect();
    let (sid, third) = within("a reconnect after request_reconnect", creates.recv()).await.unwrap();
    assert_eq!(sid, "PA_1");
    assert_eq!(third, second, "a bare reconnect must re-send the current settings");
}

// Catches: the driver failing to advance the shared policy's retry counter. A gateway that is
// down produces an infinite reconnect loop that never reports anything, which is indistinguishable
// from "interruption is enabled and quiet". Also catches `emit_error` being dropped: the two
// recoverable errors before the fatal one are the only warning a user gets.
#[tokio::test]
async fn an_unreachable_gateway_climbs_the_ladder_and_then_gives_up() {
    let port = dead_port().await;
    let config = ConnectionConfig {
        conn_options: ApiConnectOptions {
            max_retry: 2,
            retry_interval: 0.01,
            connect_timeout: Some(1.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let detector = AdaptiveInterruptionDetector::new(
        options(format!("ws://127.0.0.1:{port}"), Some(0.5)),
        config,
        Arc::new(RealClock::new()),
    );
    let mut stream = detector.stream(&lk_agents_abi::telemetry::Cx::NONE);

    let mut recoverable = 0;
    let mut fatal = 0;
    while let Some(event) = within("the stream's errors", stream.recv_event()).await {
        let StreamEvent::Error { recoverable: retrying, request, .. } = event else {
            panic!("expected an error: {event:?}")
        };
        if retrying {
            recoverable += 1;
        } else {
            fatal += 1;
        }
        assert_eq!(request, None);
    }
    assert_eq!((recoverable, fatal), (2, 1), "two retries announced, then one give-up");
    let err = within(
        "the retry ladder to terminate -- num_retries may not be advancing",
        stream.finish(),
    )
    .await
    .expect_err("an unreachable gateway must fail the stream");
    // The count includes the first attempt (7dc5c87d6): two retries are three attempts.
    assert_eq!(err.message, "failed to detect interruption after 3 attempts");
}

// Catches: `end_input` not actually closing the channel. The reference closes `_input_ch` so the
// send task can drain, write `session.close` and let the gateway hang up cleanly; a no-op leaves
// the socket open for the rest of the process and the caller with no way to tell.
#[tokio::test]
async fn end_input_closes_the_input_and_further_pushes_are_rejected() {
    let (port, _creates) = recording_gateway().await;
    let detector = AdaptiveInterruptionDetector::new(
        options(format!("ws://127.0.0.1:{port}"), Some(0.5)),
        ConnectionConfig::default(),
        Arc::new(RealClock::new()),
    );
    let mut stream = detector.stream(&lk_agents_abi::telemetry::Cx::NONE);

    stream.push(InputEvent::AgentSpeechStarted).unwrap();
    stream.end_input();
    assert_eq!(stream.push(InputEvent::AgentSpeechEnded), Err(StreamClosed::InputEnded));
    assert_eq!(stream.flush(), Err(StreamClosed::InputEnded));
}

mod websocket {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::testkit::fallback::within;
    use futures_util::{SinkExt, StreamExt};
    use lk_agents_abi::turn::detector::LegEvent;
    use lk_agents_abi::turn::interruption::{InterruptionSignal, InterruptionSource};
    use lk_agents_abi::{Clock, WallAnchor};
    use lk_agents_inference::interruption::{
        AUDIO_PREFIX_DURATION, AdaptiveInterruptionDetector, AudioFrame, ConnectionConfig,
        DETECTION_INTERVAL, InputEvent, InterruptionOptions, MAX_AUDIO_DURATION,
        MIN_INTERRUPTION_DURATION, OverlapStart, REMOTE_INFERENCE_TIMEOUT,
    };
    use lk_agents_net::clock::RealClock;
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
        AdaptiveInterruptionDetector::new(
            opts,
            ConnectionConfig::default(),
            Arc::new(RealClock::new()),
        )
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
                ws.send(Message::Text(
                    r#"{"type":"session.created","default_threshold":0.6}"#.into(),
                )),
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
        let mut stream = detector.stream(&lk_agents_abi::telemetry::Cx::NONE);
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
        let anchor = WallAnchor { wall: 1_787_000_000.0, at: clock.now() };
        let leg = detector.open(anchor, &lk_agents_abi::telemetry::Cx::NONE);
        leg.signal(InterruptionSignal::AgentSpeechStarted);
        leg.audio().push(frame(SR as usize)).unwrap();
        leg.signal(InterruptionSignal::OverlapStarted {
            started_at: clock.now(),
            speech_duration: Duration::from_millis(200),
        });
        leg.audio().push(frame(1600)).unwrap();

        let event = within("the verdict to reach the session", leg.next_event()).await;
        let LegEvent::Verdict { verdict, .. } = event else {
            panic!("expected a verdict: {event:?}")
        };
        let metrics = verdict.metrics.expect("the verdict carries the detector's metrics");
        assert_eq!((metrics.num_interruptions, metrics.num_backchannels), (1, 0));
        assert_eq!(metrics.num_requests, 1);
        assert!(metrics.total_duration > 0.0);
        within("the gateway task to finish", gateway).await.unwrap();
    }
}
