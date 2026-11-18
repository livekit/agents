use super::*;
use crate::testkit::{FakeWs, arm_then_advance, fallback::until, text};
use lk_agents_abi::{SpeechEventType, Ts, VirtualClock};
use lk_agents_net::task::TokioSpawner;
use serde_json::json;

fn client(ws: Arc<FakeWs>, clock: Arc<VirtualClock>) -> GatewayStt {
    GatewayStt::new(
        SttOptions {
            model: Some("deepgram/nova-3".into()),
            api_key: "key".into(),
            api_secret: "secret".into(),
            conn_options: ApiConnectOptions { max_retry: 0, ..Default::default() },
            ..Default::default()
        },
        ws,
        Arc::new(TokioSpawner::current()),
        clock,
    )
}

// Catches a half-open gateway staying live because its audio writes still succeed.
#[tokio::test]
async fn a_gateway_that_accepts_audio_but_stops_answering_pings_fails_once() {
    let ws = Arc::new(
        FakeWs::new(vec![text(json!({"type":"session.created"}))]).keep_open().unresponsive(),
    );
    let clock = Arc::new(VirtualClock::new());
    let (sink, mut source) = client(ws.clone(), clock.clone())
        .stream_native(&telemetry::Cx::NONE, None, 0.0)
        .unwrap()
        .split();
    sink.push(AudioFrame::new(vec![0; 1600], 16_000, 1, 800)).unwrap();
    until("audio sent on the open connection", || ws.sent_json().len() >= 2).await;
    arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("the liveness ping", || ws.ping_count() == 1).await;
    arm_then_advance(&clock, Duration::from_secs(15)).await;
    let error = source.next().await.expect_err("half-open socket must fail");
    assert!(error.retryable());
    assert_eq!(ws.connect_count(), 1);
    Box::new(source).close(Ts::from_nanos(u64::MAX)).await;
    assert_eq!(ws.close_count(), 1, "the connection is closed once");
}

// Catches an uncorrelated finalize ack dropping the tail, or EOF progress extending the cap.
#[tokio::test]
async fn the_gateway_drains_a_final_after_its_ack_within_the_absolute_budget() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type":"session.created"}))]).keep_open());
    let clock = Arc::new(VirtualClock::new());
    let (sink, mut source) = client(ws.clone(), clock.clone())
        .stream_native(&telemetry::Cx::NONE, None, 0.0)
        .unwrap()
        .split();
    sink.push(AudioFrame::new(vec![0; 1600], 16_000, 1, 800)).unwrap();
    sink.end_input();
    until("finalize sent", || ws.sent_json().iter().any(|m| m["type"] == "session.finalize")).await;
    ws.feed(text(json!({"type":"session.finalized"})));
    arm_then_advance(&clock, Duration::from_secs(28)).await;
    ws.feed(text(json!({"type":"final_transcript", "transcript":"tail"})));
    loop {
        let event = source.next().await.unwrap().expect("the tail must arrive");
        if event.kind == SpeechEventType::FinalTranscript {
            assert_eq!(event.alternatives[0].text, "tail");
            break;
        }
    }
    // The final's three-second quiet window cannot extend the thirty-second absolute cap.
    arm_then_advance(&clock, Duration::from_secs(2)).await;
    while let Some(event) = source.next().await.unwrap() {
        assert_ne!(event.kind, SpeechEventType::FinalTranscript, "delivered once");
    }
    assert_eq!(ws.connect_count(), 1);
    Box::new(source).close(Ts::from_nanos(u64::MAX)).await;
    assert_eq!(ws.close_count(), 1, "the connection is closed once");
}

// Catches transport pongs being skipped without updating the native STT watchdog.
#[tokio::test]
async fn an_idle_gateway_that_answers_pings_keeps_transcribing() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type":"session.created"}))]).keep_open());
    let clock = Arc::new(VirtualClock::new());
    let (sink, mut source) = client(ws.clone(), clock.clone())
        .stream_native(&telemetry::Cx::NONE, None, 0.0)
        .unwrap()
        .split();
    sink.push(AudioFrame::new(vec![0; 1600], 16_000, 1, 800)).unwrap();
    until("the audio", || ws.sent_json().len() >= 2).await;
    arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("the first ping", || ws.ping_count() == 1).await;
    arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("the answered connection's next ping", || ws.ping_count() == 2).await;
    ws.feed(text(json!({"type":"final_transcript", "transcript":"still here"})));
    loop {
        let event = source.next().await.unwrap().expect("healthy connection remains open");
        if event.kind == SpeechEventType::FinalTranscript {
            assert_eq!(event.alternatives[0].text, "still here");
            break;
        }
    }
    assert_eq!(ws.connect_count(), 1);
    Box::new(source).close(Ts::from_nanos(u64::MAX)).await;
    assert_eq!(ws.close_count(), 1);
}
