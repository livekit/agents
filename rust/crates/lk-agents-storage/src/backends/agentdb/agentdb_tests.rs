use lk_agents_abi::storage::{Storage as _, TaskState};
use std::sync::atomic::AtomicUsize;

use lk_agents_abi::{OwnedTask, SpawnerExt as _};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;
use crate::storage::Namespace;

#[tokio::test]
async fn unavailable_after_sending_a_commit_is_uncertain_and_is_not_resent() {
    use connection::tests::{dial, error, hello_ok, serve};
    use connection::wire as pb;
    fn dying(message: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        match &message.message {
            Some(pb::client_message::Message::Hello(_)) => vec![hello_ok()],
            Some(pb::client_message::Message::Batch(_)) => {
                vec![error(ErrorCode::COMMIT_UNKNOWN_RESULT, "database is shutting down")]
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

/// agent-store answers `retry` (1003) when a write applied nothing under pressure, `unavailable`
/// (1004) when the owner could not serve it, `snapshot_expired` (1007) when a read held its
/// snapshot too long, and `commit_unknown_result` (1021) when its durable flush failed after
/// SQLite committed: the first three are retryable refusals, the last `Uncertain`. Before, each was
/// an unknown code, so a write failed `Invalid` or `Uncertain` and was never retried.
#[tokio::test]
async fn retryable_codes_refuse_a_sent_write_and_commit_unknown_result_leaves_it_uncertain() {
    use connection::tests::{dial, error, hello_ok, serve};
    use connection::wire as pb;
    fn answers(code: ErrorCode, message: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        match &message.message {
            Some(pb::client_message::Message::Hello(_)) => vec![hello_ok()],
            Some(pb::client_message::Message::Batch(_)) => vec![error(code, "refused")],
            _ => Vec::new(),
        }
    }
    fn retry(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        answers(ErrorCode::RETRY, m)
    }
    fn unavailable(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        answers(ErrorCode::UNAVAILABLE, m)
    }
    fn snapshot_expired(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        answers(ErrorCode::SNAPSHOT_EXPIRED, m)
    }
    fn commit_unknown(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        answers(ErrorCode::COMMIT_UNKNOWN_RESULT, m)
    }
    type Script = fn(&pb::ClientMessage) -> Vec<pb::ServerMessage>;
    for (code, script) in [
        ("retry", retry as Script),
        ("unavailable", unavailable),
        ("snapshot_expired", snapshot_expired),
        ("commit_unknown_result", commit_unknown),
    ] {
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
            "commit_unknown_result" => {
                assert!(matches!(result, Err(StorageError::Uncertain(_))), "{result:?}")
            }
            _ => assert!(
                matches!(result, Err(StorageError::Unavailable(ref e)) if e.starts_with(code)),
                "{code}: {result:?}"
            ),
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

/// A socket closed between a request's frame and its answer leaves every write uncertain, on
/// every statement kind; a read changed nothing and is only retryable.
#[tokio::test]
async fn a_socket_closed_after_a_write_was_sent_leaves_it_uncertain() {
    let url = closes_after_reading().await;
    let dial = || {
        let mut opts = connection::Options::new(url.clone(), "DB_1");
        opts.reconnect.min_backoff = Duration::from_millis(10);
        let minter = TokenMinter::new("devkey", "secret").unwrap();
        let background = crate::testing::background();
        Conn::dial(opts, minter, background.spawner, background.clock)
    };
    let conn = dial().await.unwrap();
    let exec = conn.exec(&Statement::raw("UPDATE t SET n = 1"), conn.deadline()).await;
    let exec = StorageError::from(exec.unwrap_err());
    assert!(matches!(exec, StorageError::Uncertain(_)), "{exec:?}");
    let pooled = Pooled {
        conn: dial().await.unwrap(),
        schema: SchemaCell::default(),
        pool: Arc::new(DbPool::new(Arc::new(lk_agents_net::clock::RealClock::new()))),
        closed: AtomicBool::new(false),
    };
    let batch = pooled.batch(vec![Statement::raw("UPDATE t SET n = 2")]).await;
    assert!(matches!(batch, Err(StorageError::Uncertain(_))), "{batch:?}");
    // a read is sent again on every redial until its deadline, then only retryable
    let clock = lk_agents_net::clock::RealClock::new();
    let deadline = clock.now().saturating_add(Duration::from_millis(300));
    let read = crate::deadline::within(deadline, pooled.query(Statement::raw("SELECT n FROM t")));
    let read = read.await;
    assert!(matches!(read, Err(StorageError::Unavailable(_))), "{read:?}");
    pooled.conn.aclose().await;
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
            let statuses = [
                TaskStatus::Submitted,
                TaskStatus::Working,
                TaskStatus::InputRequired,
                TaskStatus::Completed,
                TaskStatus::Failed,
                TaskStatus::Canceled,
                TaskStatus::Rejected,
            ];
            let responses = statuses
                .into_iter()
                .map(|status| json!(status.as_str_name()))
                .chain([json!(0), json!(999), json!("not-a-state")]);
            for (index, status) in responses.enumerate() {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut bytes = [0; 4096];
                while !request.ends_with(b"\r\n\r\n") {
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&bytes[..n]);
                }
                let head = String::from_utf8(request).unwrap();
                if index < statuses.len() {
                    assert!(head.starts_with("GET /v0/projects/p_/threads/AT_1/tasks?"));
                    assert!(head.contains(&format!("status={}", statuses[index].as_str_name())));
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
async fn cloud_task_views_preserve_every_state_and_label_presence_and_refuse_invalid_states() {
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
    for state in [
        TaskState::Submitted,
        TaskState::Working,
        TaskState::InputRequired,
        TaskState::Completed,
        TaskState::Failed,
        TaskState::Canceled,
        TaskState::Rejected,
    ] {
        let tasks = storage.get_thread_tasks("AT_1".into(), Some(state)).await.unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].status, state);
        assert_eq!(tasks[1].status, state);
        assert_eq!(tasks[0].name, None);
        assert_eq!(tasks[1].name.as_deref(), Some(""));
    }
    for _ in 0..3 {
        assert!(matches!(
            storage.get_thread_tasks("AT_1".into(), None).await,
            Err(StorageError::Invalid(_))
        ));
    }
    storage.aclose().await;
    peer.close().await;
}

/// A writer panic answers `internal` after the write may have applied, and a newer server may send
/// a code this build does not know: either leaves a sent write `Uncertain`, resolved by its key,
/// never a refusal that would drop a write that committed.
#[test]
fn a_sent_write_answered_internal_or_an_unknown_code_is_uncertain() {
    for code in [ErrorCode::INTERNAL, ErrorCode(9999)] {
        let answer = WsdbError::Server { code, message: code.to_string() };
        let err: StorageError = WriteError::Sent(answer.clone()).into();
        assert!(matches!(err, StorageError::Uncertain(_)), "{code}: {err:?}");
        let err: StorageError = WriteError::NotSent(answer).into();
        assert!(matches!(err, StorageError::Invalid(_)), "{code}: unsent, it is final: {err:?}");
    }
}

/// A pooled connection over the fake `wsdb` server `script` answers.
async fn pooled(
    script: fn(&connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage>,
) -> (Pooled, connection::tests::Served) {
    let (url, seen) = connection::tests::serve(script, false).await;
    let conn = connection::tests::dial(&url, "DB_1").await.unwrap();
    let pool = Arc::new(DbPool::new(Arc::new(lk_agents_net::clock::RealClock::new())));
    let pooled =
        Pooled { conn, schema: SchemaCell::default(), pool, closed: AtomicBool::new(false) };
    (pooled, seen)
}

/// How many frames of `seen` `kind` picks.
fn frames(
    seen: &connection::tests::Served,
    kind: fn(&connection::wire::ClientMessage) -> bool,
) -> usize {
    seen.lock().iter().filter(|m| kind(m)).count()
}

fn is_batch(m: &connection::wire::ClientMessage) -> bool {
    matches!(m.message, Some(connection::wire::client_message::Message::Batch(_)))
}

fn is_exec(m: &connection::wire::ClientMessage) -> bool {
    matches!(m.message, Some(connection::wire::client_message::Message::Exec(_)))
}

fn is_query(m: &connection::wire::ClientMessage) -> bool {
    matches!(m.message, Some(connection::wire::client_message::Message::Query(_)))
}

/// Answers `code` to every batch, and the shared script otherwise.
fn batches_answered(
    code: ErrorCode,
    m: &connection::wire::ClientMessage,
) -> Vec<connection::wire::ServerMessage> {
    match is_batch(m) {
        true => vec![connection::tests::error(code, "answered")],
        false => connection::tests::script(m),
    }
}

/// Catches: a read refused unapplied failing at once. A query answered `retry` is sent again and
/// served.
#[tokio::test]
async fn a_read_refused_unapplied_is_sent_again() {
    use std::sync::atomic::AtomicBool as Once;
    static REFUSED: Once = Once::new(false);
    fn refuses_once(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        if is_query(m) && !REFUSED.swap(true, Ordering::SeqCst) {
            return vec![connection::tests::error(ErrorCode::RETRY, "busy")];
        }
        connection::tests::script(m)
    }
    let (pooled, seen) = pooled(refuses_once).await;
    let rows = pooled.query(Statement::raw("SELECT")).await.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(frames(&seen, is_query), 2, "sent again");
    pooled.conn.aclose().await;
}

/// Catches: a write answered `retry` sent again with less than the server's request timeout left
/// before its caller's deadline, whose answer the deadline could cut: an unknown outcome, and a
/// wait past the deadline. Too short for another whole attempt, it reaches its caller as the
/// refusal, sent once, before the deadline.
#[tokio::test]
async fn a_retry_with_no_room_for_another_attempt_reaches_the_caller_unapplied() {
    use std::sync::atomic::AtomicBool as Once;
    static REFUSED: Once = Once::new(false);
    fn refused_then_unanswered(
        m: &connection::wire::ClientMessage,
    ) -> Vec<connection::wire::ServerMessage> {
        match is_batch(m) {
            true if !REFUSED.swap(true, Ordering::SeqCst) => {
                vec![connection::tests::error(ErrorCode::RETRY, "log full")]
            }
            true => Vec::new(),
            false => connection::tests::script(m),
        }
    }
    let (pooled, seen) = pooled(refused_then_unanswered).await;
    let clock = pooled.pool.clock.clone();
    let deadline = clock.now().saturating_add(Duration::from_secs(2));
    let write = crate::deadline::within(deadline, async {
        let resend = Resend::new(clock.clone(), &Cx::NONE, StorageWrite::Context);
        resend.send(|| pooled.batch(vec![Statement::raw("UPDATE t SET n = 1")])).await
    });
    let write = lk_agents_abi::before(&*clock, Some(deadline), write).await;
    assert!(matches!(write, Ok(Err(StorageError::Unavailable(_)))), "{write:?}");
    assert_eq!(frames(&seen, is_batch), 1, "no resend the deadline could cut");
    pooled.conn.aclose().await;
}

/// Catches: a read batch answered `internal` or a code this build does not know read as
/// retryable, and a write so answered sent again. The read fails final, the write `Uncertain`;
/// neither goes out twice.
#[tokio::test]
async fn never_retry_codes_are_final_for_reads_and_uncertain_for_writes() {
    fn internal(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        batches_answered(ErrorCode::INTERNAL, m)
    }
    fn unknown(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        batches_answered(ErrorCode(9999), m)
    }
    type Script = fn(&connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage>;
    for script in [internal as Script, unknown] {
        let (pooled, seen) = pooled(script).await;
        let read = pooled.read_batch(vec![Statement::raw("SELECT")]).await;
        assert!(matches!(read, Err(StorageError::Invalid(_))), "{read:?}");
        assert_eq!(frames(&seen, is_batch), 1, "a read answered a never-retry code is final");
        let clock = Arc::new(lk_agents_net::clock::RealClock::new());
        let resend = Resend::new(clock, &Cx::NONE, StorageWrite::Context);
        let write = resend.send(|| pooled.batch(vec![Statement::raw("UPDATE t SET n = 1")])).await;
        assert!(matches!(write, Err(StorageError::Uncertain(_))), "{write:?}");
        assert_eq!(frames(&seen, is_batch), 2, "the write went out once");
        pooled.conn.aclose().await;
    }
}

/// Catches: a write the server's request timeout cut sent again with what is left of its bound,
/// turning a clean failure into an unknown outcome, and one its canceller's Cancel stopped sent
/// again. A write answered `timed_out` or `cancelled` applied nothing and is final, sent once.
#[tokio::test]
async fn a_timed_out_or_cancelled_write_is_final() {
    fn timed_out(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        match is_exec(m) {
            true => vec![connection::tests::error(ErrorCode::TIMED_OUT, "request timeout")],
            false => connection::tests::script(m),
        }
    }
    fn cancelled(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        match is_exec(m) {
            true => vec![connection::tests::error(ErrorCode::CANCELLED, "canceled by its client")],
            false => connection::tests::script(m),
        }
    }
    async fn exec(pooled: &Pooled) -> Result<(), StorageError> {
        let update = Statement::raw("UPDATE t SET n = 1");
        pooled.conn.exec(&update, pooled.conn.deadline()).await?;
        Ok(())
    }
    type Script = fn(&connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage>;
    let clock = Arc::new(lk_agents_net::clock::RealClock::new());
    for (answer, script) in [("timed_out", timed_out as Script), ("cancelled", cancelled)] {
        let (pooled, seen) = pooled(script).await;
        let resend = Resend::new(clock.clone(), &Cx::NONE, StorageWrite::Context);
        let write = resend.send(|| exec(&pooled)).await;
        assert!(matches!(write, Err(StorageError::Invalid(_))), "{answer}: {write:?}");
        assert_eq!(frames(&seen, is_exec), 1, "a {answer} write is final");
        pooled.conn.aclose().await;
    }
}

/// Catches: a database that was not reachable yet failing `open`. A `Hello` refused `retry` is
/// dialed again within the deadline; one refused for good is final at once.
#[tokio::test]
async fn a_dial_refused_unapplied_is_sent_again_and_a_final_refusal_is_not() {
    use std::sync::atomic::AtomicBool as Once;
    static REFUSED: Once = Once::new(false);
    fn hello_refused_once(
        m: &connection::wire::ClientMessage,
    ) -> Vec<connection::wire::ServerMessage> {
        let hello = matches!(m.message, Some(connection::wire::client_message::Message::Hello(_)));
        if hello && !REFUSED.swap(true, Ordering::SeqCst) {
            return vec![connection::tests::error(ErrorCode::RETRY, "warming")];
        }
        connection::tests::script(m)
    }
    let (url, seen) = connection::tests::serve(hello_refused_once, false).await;
    let http = url.replacen("ws://", "http://", 1).replace("/db", "");
    let minter = TokenMinter::new("devkey", "secret").unwrap();
    let storage =
        AgentDbStorage::new(AgentDbOptions::new(&http), minter, &ProcessEnv::current_runtime())
            .unwrap();
    let conn = storage.dial("DB_1").await.expect("dialed again");
    let hellos = |seen: &connection::tests::Served| {
        frames(seen, |m| {
            matches!(m.message, Some(connection::wire::client_message::Message::Hello(_)))
        })
    };
    assert_eq!(hellos(&seen), 2);
    conn.close();
    let refused = storage.dial("DB_unknown").await;
    assert!(matches!(refused, Err(StorageError::NotFound(_))), "{:?}", refused.map(drop));
    assert_eq!(hellos(&seen), 3, "a final refusal is not dialed again");
    storage.close().await;
}

/// A threads service answering each request with the next `(status, body)` of `script`, then
/// holding every later one unanswered; how many requests it read.
async fn threads_service(
    script: Vec<(u16, &'static str)>,
) -> (String, Arc<AtomicUsize>, OwnedTask) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let counted = count.clone();
    let task = lk_agents_net::task::TokioSpawner::current().spawn(async move {
        let mut held = Vec::new();
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut bytes = [0; 4096];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut bytes).await.unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&bytes[..n]);
            }
            let index = counted.fetch_add(1, Ordering::SeqCst);
            let Some(&(status, body)) = script.get(index) else {
                held.push(socket); // never answered
                continue;
            };
            let header = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
            );
            socket.write_all(header.as_bytes()).await.unwrap();
            socket.write_all(body.as_bytes()).await.unwrap();
        }
    });
    (url, count, task)
}

fn threads_storage(url: &str) -> AgentDbStorage {
    let minter = TokenMinter::new("devkey", "secret").unwrap();
    AgentDbStorage::new(AgentDbOptions::new(url), minter, &ProcessEnv::current_runtime()).unwrap()
}

/// Catches: a thread read failing on a gateway's bare 503 or 504, or a dropped connection. Each
/// is sent again, and the read is served. (`livekit-api` reads a bare 502 as `internal`, the code
/// it also gives a malformed answer, so a 502 stays final.)
#[tokio::test]
async fn a_thread_read_is_sent_again_after_a_bare_gateway_error() {
    let script = vec![(503, "service unavailable"), (504, ""), (200, "{}")];
    let (url, requests, _task) = threads_service(script).await;
    let storage = threads_storage(&url);
    storage.get_thread("AT_1").await.expect("served after the gateway errors");
    assert_eq!(requests.load(Ordering::SeqCst), 3);
    storage.close().await;
}

/// Catches: a thread create refused unapplied failing at once, and an update whose answer may
/// have applied sent again (a lost update). The create is sent again; the update is `Uncertain`
/// after one request.
#[tokio::test]
async fn a_thread_write_is_sent_again_only_while_nothing_applied() {
    let (url, requests, _task) = threads_service(vec![(429, ""), (200, "{}")]).await;
    let storage = threads_storage(&url);
    storage.create_thread(CreateRequest::default()).await.expect("sent again");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    storage.close().await;

    let (url, requests, _task) = threads_service(vec![(503, "")]).await;
    let storage = threads_storage(&url);
    let update = UpdateRequest { thread_id: "AT_1".into(), ..Default::default() };
    let updated = storage.update_thread(update).await;
    assert!(matches!(updated, Err(StorageError::Uncertain(_))), "{updated:?}");
    assert_eq!(requests.load(Ordering::SeqCst), 1, "a maybe-applied update is not sent again");
    storage.close().await;
}

/// Catches: a thread call waiting past the storage deadline on a service that never answers. A
/// read is `Unavailable` at the deadline, a write `Uncertain`.
#[tokio::test]
async fn thread_calls_end_at_the_storage_deadline() {
    let (url, _, _task) = threads_service(Vec::new()).await;
    let storage = threads_storage(&url);
    let clock = lk_agents_net::clock::RealClock::new();
    let soon = || clock.now().saturating_add(Duration::from_millis(300));
    let read = crate::deadline::within(soon(), storage.get_thread("AT_1")).await;
    assert!(matches!(read, Err(StorageError::Unavailable(_))), "{read:?}");
    let update = UpdateRequest { thread_id: "AT_1".into(), ..Default::default() };
    let write = crate::deadline::within(soon(), storage.update_thread(update)).await;
    assert!(matches!(write, Err(StorageError::Uncertain(_))), "{write:?}");
    storage.close().await;
}

/// Runs `request`, stepping `clock` 2 ms at a time once the tasks have run what they can; the
/// virtual instants at which `seen` first held each further frame `kind` picks.
async fn frames_over_virtual_time<T>(
    clock: &lk_agents_abi::VirtualClock,
    seen: &connection::tests::Served,
    kind: fn(&connection::wire::ClientMessage) -> bool,
    request: impl Future<Output = T>,
) -> (T, Vec<Ts>) {
    let (mut request, mut at) = (std::pin::pin!(request), Vec::new());
    loop {
        for _ in 0..20 {
            let poll = std::future::poll_fn(|cx| std::task::Poll::Ready(request.as_mut().poll(cx)));
            if let std::task::Poll::Ready(out) = poll.await {
                at.resize(frames(seen, kind), clock.now());
                return (out, at);
            }
            tokio::task::yield_now().await;
        }
        at.resize(frames(seen, kind), clock.now());
        clock.advance(Duration::from_millis(2));
        assert!(clock.now() < Ts::from_nanos(60_000_000_000), "the request never ended");
    }
}

/// Catches: a database answering every request `wrong_shard_server` or `retry` (1003) resending
/// it in a zero-delay loop. Only a moved request's first resend is immediate; every later one
/// waits a jittered backoff (at least half of 100 ms, doubling), its retry owner's or its
/// redial's, and the retries stop before the caller's deadline. Reads and writes alike reach
/// their caller `Unavailable`.
#[tokio::test]
async fn a_request_refused_on_every_attempt_backs_off_until_its_deadline() {
    fn moving(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        match is_query(m) || is_batch(m) {
            true => vec![connection::tests::error(ErrorCode::WRONG_SHARD_SERVER, "moved")],
            false => connection::tests::script(m),
        }
    }
    fn busy(m: &connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage> {
        match is_query(m) || is_batch(m) {
            true => vec![connection::tests::error(ErrorCode::RETRY, "snapshot too old")],
            false => connection::tests::script(m),
        }
    }
    type Script = fn(&connection::wire::ClientMessage) -> Vec<connection::wire::ServerMessage>;
    let ms = Duration::from_millis;
    for (name, script, immediate) in [("moved", moving as Script, 2), ("retry", busy, 1)] {
        for write in [false, true] {
            let clock = Arc::new(lk_agents_abi::VirtualClock::new());
            let (url, seen) = connection::tests::serve(script, false).await;
            let background = crate::testing::background();
            let opts = connection::Options::new(url, "DB_1");
            let minter = TokenMinter::new("devkey", "secret").unwrap();
            let conn = Conn::dial(opts, minter, background.spawner, clock.clone()).await.unwrap();
            let pool = Arc::new(DbPool::new(clock.clone()));
            let pooled = Pooled {
                conn,
                schema: SchemaCell::default(),
                pool,
                closed: AtomicBool::new(false),
            };
            let deadline = clock.now().saturating_add(crate::deadline::REQUEST_TIMEOUT + ms(1000));
            let kind = if write { is_batch } else { is_query };
            let request = crate::deadline::within(deadline, async {
                let statement = Statement::raw("UPDATE t SET n = 1");
                let out = match write {
                    true => {
                        let clock = pooled.pool.clock.clone();
                        let resend = Resend::new(clock, &Cx::NONE, StorageWrite::Context);
                        resend.send(|| pooled.batch(vec![statement.clone()])).await.map(drop)
                    }
                    false => pooled.query(statement.clone()).await.map(drop),
                };
                pooled.conn.aclose().await;
                out
            });
            let (out, at) = frames_over_virtual_time(&clock, &seen, kind, request).await;
            let case = format!("{name}, write {write}: {at:?}");
            assert!(matches!(out, Err(StorageError::Unavailable(_))), "{case}: {out:?}");
            // the moved request went out again with no backoff
            assert!(at[immediate - 1] < Ts::ZERO.saturating_add(ms(40)), "{case}");
            // each later attempt waited at least half its step (50 ms), less the ticks a frame
            // can take to be seen; a zero-delay loop shows a tick or two
            for pair in at[immediate - 1..].windows(2) {
                assert!(pair[1].saturating_duration_since(pair[0]) >= ms(30), "{case}");
            }
            // a resend starts only while a whole request fits before the deadline: within the
            // first second, a few doubling steps
            assert!(*at.last().unwrap() < deadline && at.len() <= 2 * 5, "{case}");
        }
    }
}

/// agent-db lists newest first with pages of 50 by default and 100 at most, project-wide with no
/// scope; a row it answers with activity and expiry reaches the host without them. Oldest first
/// is refused before anything is sent, until the protocol's `ListRequest` carries the order.
#[tokio::test]
async fn cloud_thread_listing_caps_pages_and_refuses_oldest_first() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let heads = Arc::new(Mutex::new(Vec::new()));
    let seen = heads.clone();
    let peer = lk_agents_net::task::TokioSpawner::current().spawn(async move {
        for _ in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut bytes = [0; 4096];
            while !request.ends_with(b"\r\n\r\n") {
                let n = socket.read(&mut bytes).await.unwrap();
                assert_ne!(n, 0);
                request.extend_from_slice(&bytes[..n]);
            }
            seen.lock().push(String::from_utf8(request).unwrap());
            let body = serde_json::to_vec(&json!({
                "items": [{
                    "threadId": "AT_1", "scope": "user:1", "attributes": {"title": "Renewal"},
                    "createdAt": "2026-01-01T00:00:00Z", "lastActive": "2026-01-02T00:00:00Z",
                    "expiresAt": "2026-01-03T00:00:00Z", "idleTtl": "86400s"
                }],
                "pageInfo": {"nextCursor": "next", "hasMore": true}
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
    let storage = Namespace::new(
        &ProcessEnv::current_runtime(),
        None,
        Some(url),
        Some("key".into()),
        Some("secret".into()),
        None,
        None,
        None,
    )
    .unwrap();
    let page = storage.list_threads(None, SortOrder::NewestFirst, None, None).await.unwrap();
    assert_eq!(page.next_page_token, "next");
    let row = &page.threads[0];
    assert_eq!(
        (row.thread_id.as_str(), row.created_at, row.idle_ttl_seconds),
        ("AT_1", 1767225600, 86400)
    );
    storage.list_threads(None, SortOrder::NewestFirst, None, Some(500)).await.unwrap();
    let refused = storage.list_threads(None, SortOrder::OldestFirst, None, None).await;
    assert!(matches!(refused, Err(StorageError::Invalid(_))), "{refused:?}");
    peer.await.unwrap();
    storage.aclose().await;
    let heads = heads.lock();
    assert!(heads[0].starts_with("GET /v0/projects/p_/threads?"), "{}", heads[0]);
    assert!(heads[0].contains("page.pageSize=50") && !heads[0].contains("scope="), "{}", heads[0]);
    assert!(heads[1].contains("page.pageSize=100"), "{}", heads[1]);
}
