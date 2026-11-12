use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_net::clock::RealClock;
use lk_agents_net::transport::{Headers, TransportError, WsReceiver, WsSender, WsTransport};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use super::{
    AUDIO_PREFIX_DURATION, AdaptiveInterruptionDetector, ConnectionConfig, DETECTION_INTERVAL,
    InterruptionOptions, MAX_AUDIO_DURATION, MIN_INTERRUPTION_DURATION, REMOTE_INFERENCE_TIMEOUT,
};
use crate::testkit::FakeWs;

// Reuse the scripted socket, failing only the final application-level close write.
struct ClosingPeer {
    socket: FakeWs,
    created: Mutex<Option<oneshot::Sender<()>>>,
    already_closed: bool,
}

struct ClosingSender {
    inner: Arc<dyn WsSender>,
    created: Mutex<Option<oneshot::Sender<()>>>,
    already_closed: bool,
}

#[async_trait]
impl WsTransport for ClosingPeer {
    async fn connect(
        &self,
        url: &str,
        headers: Headers,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        let (inner, receiver) = self.socket.connect(url, headers).await?;
        Ok((
            Arc::new(ClosingSender {
                inner,
                created: Mutex::new(self.created.lock().take()),
                already_closed: self.already_closed,
            }),
            receiver,
        ))
    }
}

#[async_trait]
impl WsSender for ClosingSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        let frame: serde_json::Value = serde_json::from_str(&text).unwrap();
        if frame["type"] == "session.close" {
            return Err(TransportError::Closed);
        }
        self.inner.send_text(text).await?;
        if let Some(created) = self.created.lock().take() {
            let _ = created.send(());
        }
        Ok(())
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        self.inner.send_binary(data).await
    }

    fn is_closed(&self) -> bool {
        self.already_closed || self.inner.is_closed()
    }

    async fn close(&self) {
        self.inner.close().await;
    }
}

/// A peer closing during the final close write ends the stream without a detector error.
/// https://github.com/livekit/agents/pull/6893
#[tokio::test]
#[ignore = "https://github.com/livekit/agents/pull/6893 reproduces: final close write emits a detector error"]
async fn a_peer_closing_during_the_final_write_does_not_fail_the_interruption_stream() {
    for already_closed in [false, true] {
        let (created, session_created) = oneshot::channel();
        let peer = Arc::new(ClosingPeer {
            socket: FakeWs::new(Vec::new()).keep_open(),
            created: Mutex::new(Some(created)),
            already_closed,
        });
        let options = InterruptionOptions::new(
            Some(0.7),
            MIN_INTERRUPTION_DURATION,
            MAX_AUDIO_DURATION,
            AUDIO_PREFIX_DURATION,
            DETECTION_INTERVAL,
            REMOTE_INFERENCE_TIMEOUT,
            "ws://local-gateway".into(),
            "local-key".into(),
            "local-secret".into(),
        )
        .unwrap();
        let detector = AdaptiveInterruptionDetector::new(
            options,
            ConnectionConfig {
                transport: peer,
                conn_options: ApiConnectOptions { max_retry: 0, ..Default::default() },
                ..Default::default()
            },
            Arc::new(RealClock::new()),
        );
        let mut stream = detector.stream();
        tokio::time::timeout(Duration::from_secs(2), session_created).await.unwrap().unwrap();
        stream.end_input();
        let event = tokio::time::timeout(Duration::from_secs(2), stream.recv_event())
            .await
            .expect("teardown must settle even when the peer is already closing");
        let result = tokio::time::timeout(Duration::from_secs(2), stream.finish())
            .await
            .expect("the detector must join its socket tasks");
        assert!(event.is_none(), "benign teardown published a detector failure: {event:?}");
        result.expect("a final close write is best effort");
    }
}
