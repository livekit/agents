//! A live task stream dies after the peer committed the work. The original call follows the
//! same task again (or reads it once) and never resends the business request.

use super::bound_cancellation::invocation;
use super::*;
use lk_agents_abi::a2a::remote::{RemoteEvent, RemoteInteraction as _};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::session::remote::{RemoteAgentView as _, RemoteTaskView as _};
use lk_agents_runtime::net::signal;

/// How the peer answers a resubscription after the first stream dropped.
#[derive(Clone, Copy, PartialEq)]
enum Resubscribe {
    /// The stream continues from the task's current snapshot.
    Streams,
    /// The peer refuses the subscription; its task is read once instead.
    Refused,
    /// The resubscribed stream dies again before any progress, every time.
    AlwaysDrops,
}

/// The production client over a peer that streams SSE and can lose a stream mid-task.
struct StreamingPeer {
    resubscribe: Resubscribe,
    sends: Mutex<u32>,
    subscriptions: Mutex<u32>,
    reads: Mutex<u32>,
}

fn sse(id: &a2a::JsonRpcId, events: &[Value]) -> Vec<Result<bytes::Bytes, AttemptFailure>> {
    events
        .iter()
        .map(|event| {
            let response = json!({"jsonrpc":"2.0","id":id,"result":event});
            Ok(bytes::Bytes::from(format!("data: {response}\n\n")))
        })
        .collect()
}

fn lost() -> Result<bytes::Bytes, AttemptFailure> {
    Err(AttemptFailure::Connection("the stream died after the booking committed".into()))
}

fn working(artifacts: Value) -> Value {
    json!({"task":{"id":"booking","contextId":"trip","status":{"state":"TASK_STATE_WORKING"},
        "artifacts":artifacts}})
}

fn first_chunk() -> Value {
    json!([{"artifactId":"receipt","parts":[{"text":"Booked "}]}])
}

impl HttpExchangePort for StreamingPeer {
    fn send<'a>(
        &'a self,
        method: Method,
        request: HttpRequest,
        _policy: ResponsePolicy,
        reporter: Reporter<(u32, Sent)>,
        attempt: u32,
    ) -> BoxFuture<'a, Result<HttpResponse, AttemptFailure>> {
        Box::pin(async move {
            reporter.report((attempt, Sent::Connected)).await;
            reporter.report((attempt, Sent::RequestWritten)).await;
            let json_body = |value: Value| vec![Ok(bytes::Bytes::from(value.to_string()))];
            let (content_type, body) = if method == Method::GET {
                (
                    "application/json",
                    json_body(json!({"name":"booking","version":"1",
                    "description":"Book a seat","defaultInputModes":["text/plain"],
                    "defaultOutputModes":["text/plain"],"skills":[],
                    "supportedInterfaces":[{"url":format!("{ENDPOINT}/rpc"),
                        "protocolBinding":"JSONRPC","protocolVersion":"1.0"}],
                    "capabilities":{"streaming":true}})),
                )
            } else {
                let rpc: a2a::JsonRpcRequest = serde_json::from_slice(&request.body).unwrap();
                match rpc.method.as_str() {
                    a2a::methods::SEND_STREAMING_MESSAGE => {
                        *self.sends.lock() += 1;
                        // The peer booked the seat, streamed its first chunk, then the socket died.
                        let mut body = sse(
                            &rpc.id,
                            &[
                                working(json!([])),
                                json!({"artifactUpdate":{"taskId":"booking","contextId":"trip",
                                "artifact":first_chunk()[0].clone()}}),
                            ],
                        );
                        body.push(lost());
                        ("text/event-stream", body)
                    }
                    a2a::methods::SUBSCRIBE_TO_TASK => {
                        *self.subscriptions.lock() += 1;
                        match self.resubscribe {
                            Resubscribe::Refused => (
                                "application/json",
                                json_body(json!({
                                "jsonrpc":"2.0","id":rpc.id,"error":{
                                    "code":a2a::error_code::UNSUPPORTED_OPERATION,
                                    "message":"subscriptions are unavailable"}})),
                            ),
                            Resubscribe::AlwaysDrops => {
                                let mut body = sse(&rpc.id, &[working(first_chunk())]);
                                body.push(lost());
                                ("text/event-stream", body)
                            }
                            Resubscribe::Streams => (
                                "text/event-stream",
                                sse(
                                    &rpc.id,
                                    &[
                                        working(first_chunk()),
                                        json!({"artifactUpdate":{"taskId":"booking","contextId":"trip",
                                    "artifact":{"artifactId":"receipt","parts":[{"text":"Friday"}]},
                                    "append":true,"lastChunk":true}}),
                                        json!({"statusUpdate":{"taskId":"booking","contextId":"trip",
                                    "status":{"state":"TASK_STATE_COMPLETED"}}}),
                                    ],
                                ),
                            ),
                        }
                    }
                    a2a::methods::GET_TASK => {
                        *self.reads.lock() += 1;
                        (
                            "application/json",
                            json_body(json!({"jsonrpc":"2.0","id":rpc.id,
                            "result":{"id":"booking","contextId":"trip",
                                "status":{"state":"TASK_STATE_COMPLETED"},
                                "artifacts":[{"artifactId":"receipt","parts":[
                                    {"text":"Booked "},{"text":"Friday"}]}]}})),
                        )
                    }
                    method => panic!("a lost stream never sends {method}"),
                }
            };
            let mut headers = HeaderMap::new();
            headers.insert("content-type", content_type.parse().unwrap());
            Ok(HttpResponse {
                status: StatusCode::OK,
                headers,
                body: Box::pin(futures_util::stream::iter(body)),
            })
        })
    }
}

#[tokio::test(start_paused = true)]
async fn a_lost_task_stream_is_resubscribed_without_resending_or_repeating_artifacts() {
    lost_stream(Resubscribe::Streams).await;
}

#[tokio::test(start_paused = true)]
async fn a_refused_resubscription_reads_the_task_once_instead_of_resending() {
    lost_stream(Resubscribe::Refused).await;
}

#[tokio::test(start_paused = true)]
async fn a_stream_that_keeps_dying_leaves_the_outcome_unknown_within_the_retry_policy() {
    lost_stream(Resubscribe::AlwaysDrops).await;
}

async fn lost_stream(resubscribe: Resubscribe) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(173, false, clock.clone()).await;
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (owner, updates, binding, view) = owner(&env, store.clone()).await;
    let peer = Arc::new(StreamingPeer {
        resubscribe,
        sends: Mutex::new(0),
        subscriptions: Mutex::new(0),
        reads: Mutex::new(0),
    });
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        peer.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let interaction = interaction();
    let (ready, _) = signal::once();
    let options = lk_agents_abi::ApiConnectOptions::default();
    let request = invocation(&env, owner.clone(), ready, options);
    let result = interaction.run_with_client(&env.process, client, request).await;
    interaction.settled().await;
    assert_eq!(*peer.sends.lock(), 1, "a lost connection never restarts the booking");
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    match resubscribe {
        Resubscribe::Streams | Resubscribe::Refused => {
            let Ok(RemoteEvent::Outcome { outcome: ToolOutcome::Value { text, .. } }) = result
            else {
                panic!("the recovered task delivers its result: {result:?}")
            };
            assert_eq!(text.matches("Booked").count(), 1, "{text}");
            assert!(text.contains("Booked Friday"), "{text}");
            let saved = task.task().unwrap();
            assert_eq!(saved.status.state, a2a::TaskState::Completed);
            let parts: Vec<_> = saved.artifacts.unwrap()[0]
                .parts
                .iter()
                .filter_map(|part| part.as_text().map(str::to_owned))
                .collect();
            assert_eq!(parts, ["Booked ", "Friday"], "a snapshot never appends twice");
            assert_eq!(*peer.subscriptions.lock(), 1);
            let reads = u32::from(resubscribe == Resubscribe::Refused);
            assert_eq!(*peer.reads.lock(), reads);
        }
        Resubscribe::AlwaysDrops => {
            assert!(result.is_err(), "a task that cannot be observed is not a result");
            assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Working);
            assert_eq!(
                *peer.subscriptions.lock(),
                options.max_retry,
                "recovery without progress stops within the connection policy"
            );
        }
    }
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}
