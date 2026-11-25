//! A view's extra headers on the gateway's connections.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{ApiConnectOptions, StreamOutput, Stt};
use lk_agents_net::task::TokioSpawner;
use lk_agents_net::transport::TransportError;

use super::{GatewayStt, SttOptions};
use crate::testkit::FakeWs;

fn opts() -> SttOptions {
    SttOptions {
        model: Some("deepgram/nova-3".into()),
        base_url: "https://gw/v1".into(),
        api_key: "k".into(),
        api_secret: "s".into(),
        sample_rate: 16_000,
        conn_options: ApiConnectOptions { max_retry: 1, retry_interval: 0.0, ..Default::default() },
        ..SttOptions::default()
    }
}

// Catches: a view's extra headers missing from a reconnect, or leaking into the client's other
// streams. Python: `stream(extra_headers=...)` sends them "on every connection, reconnects
// included", as AMD's client feature tag (`_stt.py` passes `client_feature_kwargs`).
#[tokio::test]
async fn a_views_extra_headers_ride_on_every_connection_of_its_streams_only() {
    let ws = Arc::new(FakeWs::failing(TransportError::Connect("refused".into())));
    let opts = opts();
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    let client = GatewayStt::new(
        opts,
        ws.clone(),
        spawner,
        Arc::new(lk_agents_net::clock::RealClock::new()),
    );
    let tag = std::collections::BTreeMap::from([("X-LiveKit-Client-Feature".into(), "AMD".into())]);
    let tagged = Stt::with_extra_headers(&client, &tag).expect("an Inference STT takes headers");
    let (_sink, mut source) = tagged
        .stream(&lk_agents_abi::telemetry::Cx::NONE, None, None)
        .await
        .unwrap()
        .unwrap()
        .split();
    let _ = lk_agents_abi::timeout(
        &lk_agents_net::clock::RealClock::new(),
        Duration::from_secs(5),
        source.next(),
    )
    .await
    .expect("the retries settle");
    let connects = ws.connects();
    assert_eq!(connects.len(), 2, "the first attempt and one retry");
    for (_, headers) in &connects {
        assert_eq!(headers.get("X-LiveKit-Client-Feature"), Some("AMD"));
    }
    let (_sink, mut plain) =
        client.stream_native(&lk_agents_abi::telemetry::Cx::NONE, None, 0.0).unwrap().split();
    let _ = lk_agents_abi::timeout(
        &lk_agents_net::clock::RealClock::new(),
        Duration::from_secs(5),
        plain.next(),
    )
    .await;
    let (_, headers) = ws.last_connect().expect("the client's own stream connected");
    assert_eq!(headers.get("X-LiveKit-Client-Feature"), None, "the client's streams are untagged");
}
