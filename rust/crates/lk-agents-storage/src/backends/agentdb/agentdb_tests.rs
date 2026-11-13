use lk_agents_abi::storage::{Storage as _, TaskState};
use lk_agents_abi::{OwnedTask, SpawnerExt as _};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;
use crate::storage::Namespace;

#[tokio::test]
async fn unavailable_after_sending_a_commit_is_uncertain_and_is_not_resent() {
    use connection::tests::{dial, hello_ok, reply, serve};
    use connection::wire as pb;
    fn dying(message: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        match &message.message {
            Some(pb::client_message::Message::Hello(_)) => vec![hello_ok()],
            Some(pb::client_message::Message::Batch(_)) => {
                vec![reply(pb::server_message::Message::Error(pb::Error {
                    code: "unavailable".into(),
                    message: "database is shutting down".into(),
                }))]
            }
            _ => Vec::new(),
        }
    }
    let (url, seen) = serve(dying, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let pooled = Pooled {
        conn,
        schema: SchemaCell::default(),
        pool: Arc::new(DbPool::new(Arc::new(lk_agents_net::clock::RealClock::new()))),
        closed: AtomicBool::new(false),
    };
    let result =
        pooled.batch(vec![Statement::raw("UPDATE partitions SET version = version + 1")]).await;
    assert!(
        matches!(result, Err(StorageError::Uncertain(ref why)) if why.to_string().contains("database is shutting down")),
        "{result:?}"
    );
    let batches = seen
        .lock()
        .iter()
        .filter(|message| matches!(message.message, Some(pb::client_message::Message::Batch(_))))
        .count();
    assert_eq!(batches, 1, "a commit with an unknown outcome must not be resent");
    pooled.conn.aclose().await;
}

/// agent-store answers `retry` when a write applied nothing and `outcome_unknown` when its durable
/// flush failed after SQLite committed: the first is a retryable refusal, the second `Uncertain`.
/// Before, both were unknown codes, so a write failed `Invalid` and was never retried.
#[tokio::test]
async fn retry_refuses_a_sent_write_and_outcome_unknown_leaves_it_uncertain() {
    use connection::tests::{dial, hello_ok, reply, serve};
    use connection::wire as pb;
    fn answers(code: &str, message: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        match &message.message {
            Some(pb::client_message::Message::Hello(_)) => vec![hello_ok()],
            Some(pb::client_message::Message::Batch(_)) => {
                vec![reply(pb::server_message::Message::Error(pb::Error {
                    code: code.into(),
                    message: code.into(),
                }))]
            }
            _ => Vec::new(),
        }
    }
    fn retry(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        answers("retry", m)
    }
    fn outcome_unknown(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        answers("outcome_unknown", m)
    }
    type Script = fn(&pb::ClientMessage) -> Vec<pb::ServerMessage>;
    for (code, script) in [("retry", retry as Script), ("outcome_unknown", outcome_unknown)] {
        let (url, _) = serve(script, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let pooled = Pooled {
            conn,
            schema: SchemaCell::default(),
            pool: Arc::new(DbPool::new(Arc::new(lk_agents_net::clock::RealClock::new()))),
            closed: AtomicBool::new(false),
        };
        let result = pooled.batch(vec![Statement::raw("UPDATE partitions SET version = 2")]).await;
        match code {
            "retry" => assert!(
                matches!(result, Err(StorageError::Unavailable(ref e)) if e.contains("retry")),
                "{result:?}"
            ),
            _ => assert!(matches!(result, Err(StorageError::Uncertain(_))), "{result:?}"),
        }
        pooled.conn.aclose().await;
    }
}

/// A fake `wsdb` server that answers `Hello`, then closes the socket on the next request it
/// reads, unanswered: the frame was written and read, the answer never comes.
async fn closes_after_reading() -> String {
    use connection::tests::hello_ok;
    use connection::wire as pb;
    use futures_util::{SinkExt as _, StreamExt as _};
    use lk_agents_net::websocket::Message;
    use prost::Message as _;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    tokio::spawn(async move {
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                while let Some(Ok(Message::Binary(b))) = ws.next().await {
                    let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                    match msg.message {
                        Some(pb::client_message::Message::Hello(_)) => {
                            let mut ok = hello_ok();
                            ok.request_id = msg.request_id;
                            ws.send(Message::binary(ok.encode_to_vec())).await.unwrap();
                        }
                        Some(pb::client_message::Message::Ping(_)) => {}
                        _ => return, // dropped: the socket closes with the request unanswered
                    }
                }
            });
        }
    });
    url
}

/// A socket closed between a request's frame and its answer leaves every write uncertain,
/// whether the client redials (`Disconnected`) or has given up (`Closed`), on every statement
/// kind; a read changed nothing and is only retryable.
#[tokio::test]
async fn a_socket_closed_after_a_write_was_sent_leaves_it_uncertain() {
    for redial in [true, false] {
        let url = closes_after_reading().await;
        let dial = || {
            let mut opts = connection::Options::new(url.clone(), "DB_1");
            opts.reconnect.disable = !redial;
            opts.reconnect.min_backoff = Duration::from_millis(10);
            opts.reconnect.budget = Duration::from_secs(5);
            let minter = TokenMinter::new("devkey", "secret").unwrap();
            let background = crate::testing::background();
            Conn::dial(opts, minter, background.spawner, background.clock)
        };
        let exec = dial().await.unwrap().exec(&Statement::raw("UPDATE t SET n = 1")).await;
        let exec = StorageError::from(exec.unwrap_err());
        assert!(matches!(exec, StorageError::Uncertain(_)), "redial {redial}: {exec:?}");
        let pooled = Pooled {
            conn: dial().await.unwrap(),
            schema: SchemaCell::default(),
            pool: Arc::new(DbPool::new(Arc::new(lk_agents_net::clock::RealClock::new()))),
            closed: AtomicBool::new(false),
        };
        let batch = pooled.batch(vec![Statement::raw("UPDATE t SET n = 2")]).await;
        assert!(matches!(batch, Err(StorageError::Uncertain(_))), "redial {redial}: {batch:?}");
        let read = pooled.query(Statement::raw("SELECT n FROM t")).await;
        assert!(matches!(read, Err(StorageError::Unavailable(_))), "redial {redial}: {read:?}");
        pooled.conn.aclose().await;
    }
}

#[tokio::test]
async fn the_data_plane_sits_under_the_project_public_api() {
    let minter = || TokenMinter::new("devkey", "secret").unwrap();
    let storage = |url: &str| {
        AgentDbStorage::new(AgentDbOptions::new(url), minter(), &ProcessEnv::current_runtime())
    };
    let staging = storage("wss://my-app.staging.livekit.cloud").unwrap();
    assert_eq!(staging.opts.url, "https://beta-api.staging.livekit.cloud");
    assert_eq!(staging.data_plane, "wss://beta-api.staging.livekit.cloud/v0/agentstore/db");
    let local = storage("ws://localhost:9000/").unwrap();
    assert_eq!(local.data_plane, "ws://localhost:9000/v0/agentstore/db");
    assert!(storage("ftp://x").is_err());
}

struct Peer {
    url: String,
    /// Dropped with the peer, which cancels it.
    task: Option<OwnedTask>,
}

impl Peer {
    async fn tasks() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = lk_agents_net::task::TokioSpawner::current().spawn(async move {
            for (index, status) in ["INPUT_REQUIRED", "not-a-state"].into_iter().enumerate() {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut bytes = [0; 4096];
                while !request.ends_with(b"\r\n\r\n") {
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&bytes[..n]);
                }
                let head = String::from_utf8(request).unwrap();
                if index == 0 {
                    assert!(head.starts_with("GET /v0/projects/p_/threads/AT_1/tasks?"));
                    assert!(head.contains("status=INPUT_REQUIRED"));
                }
                let body = serde_json::to_vec(&json!({
                    "items": [
                        {"taskId": "AST_1", "threadId": "AT_1", "status": status},
                        {"taskId": "AST_2", "threadId": "AT_1", "status": status, "name": ""}
                    ],
                    "pageInfo": {}
                }))
                .unwrap();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len(),
                );
                socket.write_all(header.as_bytes()).await.unwrap();
                socket.write_all(&body).await.unwrap();
            }
        });
        Self { url, task: Some(task) }
    }

    async fn close(&mut self) {
        self.task.take().unwrap().await.unwrap();
    }
}

#[tokio::test]
async fn cloud_task_views_preserve_waiting_state_and_label_presence_and_refuse_unknown_states() {
    let mut peer = Peer::tasks().await;
    let storage = Namespace::new(
        &ProcessEnv::current_runtime(),
        None,
        Some(peer.url.clone()),
        Some("key".into()),
        Some("secret".into()),
        None,
        None,
        None,
    )
    .unwrap();
    let tasks =
        storage.get_thread_tasks("AT_1".into(), Some(TaskState::InputRequired)).await.unwrap();
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].status, TaskState::InputRequired);
    assert_eq!(tasks[0].name, None);
    assert_eq!(tasks[1].name.as_deref(), Some(""));
    assert!(matches!(
        storage.get_thread_tasks("AT_1".into(), None).await,
        Err(StorageError::Invalid(_))
    ));
    storage.aclose().await;
    peer.close().await;
}
