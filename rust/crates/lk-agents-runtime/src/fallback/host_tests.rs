//! The declared facade preserves native provider dispatch and stream cleanup.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use lk_agents_abi::ChoiceDelta;
use lk_agents_abi::llm::fallback::FallbackLlm as DeclaredFallback;

use super::*;

struct Provider(Arc<AtomicBool>);

#[async_trait::async_trait]
impl Llm for Provider {
    fn label(&self) -> String {
        "scripted".into()
    }

    fn model(&self) -> String {
        "native-model".into()
    }

    async fn chat(&self, _: &ChatRequest) -> Result<LlmStream, ProviderError> {
        Ok(Box::new(Response { sent: false, closed: self.0.clone() }))
    }
}

struct Response {
    sent: bool,
    closed: Arc<AtomicBool>,
}

impl StreamSource<LlmChunk> for Response {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            if self.sent {
                return std::future::pending().await;
            }
            self.sent = true;
            Ok(Some(LlmChunk {
                id: "reply".into(),
                delta: Some(ChoiceDelta {
                    content: Some("native response".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }))
        })
    }

    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.closed.store(true, Ordering::SeqCst);
            Closed::Clean
        })
    }
}

#[tokio::test]
async fn declared_fallback_keeps_native_dispatch_and_cancels_its_open_stream() {
    let empty = <FallbackLlm as DeclaredFallback>::new(vec![], 5.0, 0, 0.5, false, None);
    assert!(empty.is_err());

    let closed = Arc::new(AtomicBool::new(false));
    let fallback = <FallbackLlm as DeclaredFallback>::new(
        vec![Arc::new(Provider(closed.clone()))],
        5.0,
        0,
        0.5,
        false,
        None,
    )
    .expect("native provider accepted");
    assert_eq!(fallback.model(), "native-model");
    let mut stream = Llm::chat(&fallback, &ChatRequest::default()).await.expect("opened");
    let chunk = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("first chunk arrived")
        .expect("provider succeeded")
        .expect("provider emitted a chunk");
    assert_eq!(chunk.delta.unwrap().content.as_deref(), Some("native response"));
    assert_eq!(DeclaredFallback::active_index(&fallback), 0);

    let deadline = crate::process::host_clock().now().saturating_add(Duration::from_secs(5));
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), stream.close(deadline))
            .await
            .expect("close settled"),
        Closed::Clean,
    );
    assert!(closed.load(Ordering::SeqCst));
    DeclaredFallback::aclose(&fallback).await;
}
