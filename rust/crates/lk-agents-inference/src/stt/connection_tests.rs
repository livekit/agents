use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::errors::{ApiErrorKind, ProviderError};
use lk_agents_abi::{ApiConnectOptions, AudioFrame, StreamInput, StreamOutput, Ts};
use lk_agents_net::clock::RealClock;
use lk_agents_net::task::TokioSpawner;
use lk_agents_net::websocket::TungsteniteTransport;
use serde_json::json;
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message;

use super::{GatewayStt, SttOptions};

/// A half-open gateway socket fails recognition with a retryable connection error.
/// https://github.com/livekit/agents/pull/8023
#[tokio::test]
#[ignore = "https://github.com/livekit/agents/pull/8023 reproduces: half-open gateway socket never fails"]
async fn a_gateway_that_stops_answering_pings_returns_a_retryable_connection_error() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("ws://{}", listener.local_addr().unwrap());
    let (heard_audio, audio_received) = oneshot::channel();
    let (release, hold) = oneshot::channel();
    let gateway = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        let create = socket.next().await.unwrap().unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(create.to_text().unwrap()).unwrap()["type"],
            "session.create"
        );
        socket.send(Message::text(json!({"type":"session.created"}).to_string())).await.unwrap();
        loop {
            let message = socket.next().await.unwrap().unwrap();
            if let Message::Text(text) = message
                && serde_json::from_str::<serde_json::Value>(&text).unwrap()["type"]
                    == "input_audio"
            {
                break;
            }
        }
        heard_audio.send(()).unwrap();
        // Stop polling the real socket: TCP stays open, but no ping can get a pong.
        let _ = hold.await;
        drop(socket);
    });
    let options = SttOptions {
        model: Some("deepgram/nova-3".into()),
        base_url,
        api_key: "local-key".into(),
        api_secret: "local-secret".into(),
        conn_options: ApiConnectOptions { max_retry: 0, ..Default::default() },
        ..Default::default()
    };
    let client = GatewayStt::new(
        options,
        Arc::new(TungsteniteTransport),
        Arc::new(TokioSpawner::current()),
        Arc::new(RealClock::new()),
    );
    let (sink, mut source) = client.stream_native(None, 0.0).unwrap().split();
    sink.push(AudioFrame::new(vec![0; 1600], 16_000, 1, 800)).unwrap();
    tokio::time::timeout(Duration::from_secs(5), audio_received)
        .await
        .expect("the healthy gateway must receive streaming silence")
        .unwrap();

    // The historical policy is a 30-second ping and a 15-second pong deadline.
    tokio::time::pause();
    let result = tokio::time::timeout(Duration::from_secs(75), source.next()).await;
    let _ = release.send(());
    gateway.await.unwrap();
    Box::new(source).close(Ts::from_nanos(u64::MAX)).await;
    let error = result
        .expect("a half-open STT socket must not leave recognition waiting forever")
        .expect_err("a lost gateway connection must fail recognition");
    let ProviderError::Local(error) = error else { panic!("a native connection error") };
    assert_eq!(error.kind, ApiErrorKind::Connection);
    assert!(error.retryable);
}
