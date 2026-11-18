use super::*;
use lk_agents_abi::{ChoiceDelta, Closed, FunctionToolCall, LlmChunk, LlmStream, StreamOutput, Ts};
use lk_agents_net::task::TaskSet;
use parking_lot::Mutex;
use std::collections::VecDeque;

struct Script {
    requests: Mutex<Vec<ChatRequest>>,
    chunks: Vec<Result<LlmChunk, ProviderError>>,
    closed: Arc<Mutex<usize>>,
    pending: bool,
}

struct Chunks {
    items: VecDeque<Result<LlmChunk, ProviderError>>,
    closed: Arc<Mutex<usize>>,
    pending: bool,
}

impl StreamOutput<LlmChunk> for Chunks {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            if let Some(item) = self.items.pop_front() {
                return item.map(Some);
            }
            if self.pending { std::future::pending().await } else { Ok(None) }
        })
    }

    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            *self.closed.lock() += 1;
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Chunks {}

impl Llm for Script {
    fn label(&self) -> String {
        "routing-only".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        request: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        self.requests.lock().push(request.clone());
        Box::pin(async move {
            Ok(Box::new(Chunks {
                items: self.chunks.clone().into(),
                closed: self.closed.clone(),
                pending: self.pending,
            }) as LlmStream)
        })
    }
}

fn verdict(arguments: &str) -> LlmChunk {
    LlmChunk {
        id: "verdict".into(),
        delta: Some(ChoiceDelta {
            tool_calls: vec![FunctionToolCall {
                name: routing::TOOL.into(),
                arguments: arguments.into(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        usage: Some(Box::new(lk_agents_abi::usage::TokenUsage {
            input_tokens: 11,
            output_tokens: 3,
            ..Default::default()
        })),
        ..Default::default()
    }
}

fn input() -> CallRoutingRequest {
    CallRoutingRequest {
        remote_id: "travel".into(),
        endpoint: "https://travel.test".into(),
        proposed: serde_json::json!({"request":"Find a hotel"}),
        selected_task: None,
        history: vec![],
        candidates: vec![],
    }
}

fn background() -> Background {
    let (clock, spawner, _) = crate::testkit::fallback::parts();
    Background::new(Arc::<TaskSet>::default(), spawner, clock)
}

fn script(chunks: Vec<Result<LlmChunk, ProviderError>>) -> Arc<Script> {
    Arc::new(Script { requests: Mutex::default(), chunks, closed: Arc::default(), pending: false })
}

#[tokio::test]
async fn call_routing_uses_its_configured_model_and_named_tool_and_preserves_usage() {
    let model = script(vec![Ok(verdict(r#"{"action":"new","candidate":null,"response":null}"#))]);
    let (result, usage) = classify(
        model.clone(),
        &background(),
        &telemetry::Cx::NONE,
        input(),
        lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
    )
    .await;
    assert_eq!(result.unwrap(), CallRoute::New);
    assert_eq!(usage.input_tokens(), 11);
    assert_eq!(usage.output_tokens(), 3);
    assert_eq!(*model.closed.lock(), 1);
    let requests = model.requests.lock();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools[0].name, routing::TOOL);
    assert_eq!(requests[0].tool_choice, Some(ToolChoice::Named(routing::TOOL.into())));
    assert_eq!(requests[0].parallel_tool_calls, Some(false));
}

#[tokio::test]
async fn unusable_or_partial_call_verdicts_fail_without_accepting_or_leaking_the_stream() {
    let valid = verdict(r#"{"action":"new","candidate":null,"response":null}"#);
    let mut wrong_tool = valid.clone();
    wrong_tool.delta.as_mut().unwrap().tool_calls[0].name = "route_input_response".into();
    for chunks in [
        vec![],
        vec![Ok(wrong_tool)],
        vec![Ok(verdict("not json"))],
        vec![Ok(valid.clone()), Ok(valid.clone())],
        vec![Ok(valid), Err(crate::testkit::fallback::boom("failed after verdict"))],
    ] {
        let model = script(chunks);
        let (result, _) = classify(
            model.clone(),
            &background(),
            &telemetry::Cx::NONE,
            input(),
            ApiConnectOptions { max_retry: 0, ..lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(*model.closed.lock(), 1);
        assert_eq!(model.requests.lock().len(), 1, "a partial verdict is never replayed");
    }
}

#[tokio::test]
async fn cancelling_call_classification_still_closes_its_owned_stream() {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    let mut model = script(vec![]);
    Arc::get_mut(&mut model).unwrap().pending = true;
    let background = background();
    let mut running = Box::pin(classify(
        model.clone(),
        &background,
        &telemetry::Cx::NONE,
        input(),
        lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
    ));
    assert!(matches!(
        running.as_mut().poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ));
    drop(running);
    assert!(background.tasks.wait_empty(background.clock.as_ref(), None).await);
    assert_eq!(*model.closed.lock(), 1);
}
