use super::*;

/// Like [`serve`] over [`script`], except that the first frame `drains` picks is never run: the
/// server ends its socket with `notice` on request id 0, once. A `Hello` it picks is answered
/// first, so the notice follows it before any request. `answered`: the notice is that request's
/// answer instead, and the server keeps the socket open.
pub(super) async fn serve_draining(
    notice: ErrorCode,
    drains: fn(&pb::ClientMessage) -> bool,
    answered: bool,
) -> (String, Served) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let log: Served = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    tokio::spawn(async move {
        let drained = Arc::new(AtomicBool::new(false));
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let (seen, drained) = (seen.clone(), drained.clone());
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                while let Some(Ok(Message::Binary(b))) = ws.next().await {
                    let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                    seen.lock().push(msg.clone());
                    let gone = notice == ErrorCode::NOT_FOUND && drained.load(Ordering::SeqCst);
                    let drain = drains(&msg) && !drained.swap(true, Ordering::SeqCst);
                    let hello = matches!(msg.message, Some(pb::client_message::Message::Hello(_)));
                    let mut answers = match () {
                        // a deleted database stays gone: every later `Hello` is refused
                        _ if gone && hello => vec![error(notice, "the database was deleted")],
                        _ if drain && !hello => vec![],
                        _ => script(&msg),
                    };
                    answers.iter_mut().for_each(|a| a.request_id = msg.request_id);
                    if drain {
                        let mut notice = error(notice, "draining");
                        notice.request_id = if answered { msg.request_id } else { 0 };
                        answers.push(notice);
                    }
                    for answer in answers {
                        if ws.send(Message::binary(answer.encode_to_vec())).await.is_err() {
                            return;
                        }
                    }
                    if drain && !answered {
                        return;
                    }
                }
            });
        }
    });
    (url, log)
}

const NOTICES: [ErrorCode; 2] = [ErrorCode::WRONG_SHARD_SERVER, ErrorCode::IDLE_EVICTED];

/// Catches: an exec a drain notice left unanswered failed with the notice instead of sent again.
/// The server answered every request it read before the notice, so the exec applied nothing: it
/// goes out again on the next socket, and its caller sees only the answer.
#[tokio::test]
async fn an_exec_a_drain_notice_left_unanswered_is_served_on_the_next_socket() {
    for notice in NOTICES {
        let (url, log) = serve_draining(notice, is_exec, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let r = conn.exec(&st("INSERT", vec![]), conn.deadline()).await;
        assert_eq!(r.unwrap().rows_affected, 1, "{notice}");
        let ids = sent(&log, is_exec);
        assert!(ids.len() == 2 && ids[0] != ids[1], "{notice}: sent again as a new request");
        assert_eq!(hellos(&log), 2, "{notice}: on the next socket");
        conn.close();
    }
}

/// Catches: a batch a drain notice left unanswered failed instead of sent again.
#[tokio::test]
async fn a_batch_a_drain_notice_left_unanswered_is_served_on_the_next_socket() {
    for notice in NOTICES {
        let (url, log) = serve_draining(notice, is_batch, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let stmts: Vec<Statement> = (0..4).map(|_| st("S", vec![])).collect();
        let out = conn.batch(&stmts, conn.deadline()).await.unwrap();
        assert_eq!(out.results[3].rows().len(), 2, "{notice}");
        assert_eq!(sent(&log, is_batch).len(), 2, "{notice}: sent again");
        assert_eq!(hellos(&log), 2, "{notice}");
        conn.close();
    }
}

/// Catches: a query a drain notice left unanswered failed instead of sent again.
#[tokio::test]
async fn a_query_a_drain_notice_left_unanswered_is_served_on_the_next_socket() {
    for notice in NOTICES {
        let (url, log) = serve_draining(notice, is_query, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        let rows = conn.query_all(&st("SELECT", vec![]), conn.deadline()).await.unwrap();
        assert_eq!(rows.len(), 2, "{notice}");
        assert_eq!(sent(&log, is_query).len(), 2, "{notice}: sent again");
        assert_eq!(hellos(&log), 2, "{notice}");
        conn.close();
    }
}

/// Catches: a request answered `wrong_shard_server` failing `Unavailable`, a read never sent
/// again. It applied nothing: each request kind goes out again on a socket redialed at once, and
/// its caller sees only the answer.
#[tokio::test]
async fn a_request_answered_wrong_shard_server_is_served_on_the_next_socket() {
    let shard = ErrorCode::WRONG_SHARD_SERVER;
    let (url, log) = serve_draining(shard, is_exec, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let r = conn.exec(&st("INSERT", vec![]), conn.deadline()).await;
    assert_eq!(r.unwrap().rows_affected, 1);
    assert_eq!((sent(&log, is_exec).len(), hellos(&log)), (2, 2), "exec: on the next socket");
    conn.close();

    let (url, log) = serve_draining(shard, is_batch, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let stmts: Vec<Statement> = (0..4).map(|_| st("S", vec![])).collect();
    let out = conn.batch(&stmts, conn.deadline()).await.unwrap();
    assert_eq!(out.results[3].rows().len(), 2);
    assert_eq!((sent(&log, is_batch).len(), hellos(&log)), (2, 2), "batch: on the next socket");
    conn.close();

    let (url, log) = serve_draining(shard, is_query, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let rows = conn.query_all(&st("SELECT", vec![]), conn.deadline()).await.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!((sent(&log, is_query).len(), hellos(&log)), (2, 2), "query: on the next socket");
    conn.close();
}

/// Catches: a request answered `wrong_shard_server` ending its socket, which left every other
/// write in flight on it `Disconnected` (outcome unknown). The first socket holds an exec's answer
/// until it refused a query `wrong_shard_server`: the exec still gets its own answer there, and
/// the query is served on the next socket once nothing is in flight on the first.
#[tokio::test]
async fn a_write_in_flight_when_another_request_is_refused_wrong_shard_gets_its_answer() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let log: Served = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    tokio::spawn(async move {
        let first = Arc::new(AtomicBool::new(true));
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let (seen, first) = (seen.clone(), first.swap(false, Ordering::SeqCst));
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                let (mut exec, mut query) = (None, None);
                while let Some(Ok(Message::Binary(b))) = ws.next().await {
                    let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                    seen.lock().push(msg.clone());
                    let mut answers = Vec::new();
                    match &msg.message {
                        Some(pb::client_message::Message::Exec(_)) if first => exec = Some(msg),
                        Some(pb::client_message::Message::Query(_)) if first => query = Some(msg),
                        _ => answers.extend(script(&msg).into_iter().map(|mut a| {
                            a.request_id = msg.request_id;
                            a
                        })),
                    }
                    if let (Some(e), Some(q)) = (&exec, &query) {
                        let mut refused = error(ErrorCode::WRONG_SHARD_SERVER, "moved");
                        refused.request_id = q.request_id;
                        answers.push(refused);
                        answers.extend(script(e).into_iter().map(|mut a| {
                            a.request_id = e.request_id;
                            a
                        }));
                        (exec, query) = (None, None);
                    }
                    for answer in answers {
                        if ws.send(Message::binary(answer.encode_to_vec())).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    });
    let conn = dial(&url, "DB_1").await.unwrap();
    let (insert, select) = (st("INSERT", vec![]), st("SELECT", vec![]));
    let (exec, query) = tokio::join!(
        conn.exec(&insert, conn.deadline()),
        conn.query_all(&select, conn.deadline()),
    );
    assert_eq!(exec.expect("its own answer, not Uncertain").rows_affected, 1);
    assert_eq!(query.expect("served on the next socket").len(), 2);
    assert_eq!(sent(&log, is_exec).len(), 1, "the write went out once");
    assert_eq!(sent(&log, is_query).len(), 2, "the refused query went out again");
    assert_eq!(hellos(&log), 2);
    conn.close();
}

/// Catches: a request racing a notice read right after `Hello` failing, whether it was bound to
/// the drained socket, written to it, or registered after it. A `wrong_shard_server` socket is
/// redialed at once, with no request waiting.
#[tokio::test]
async fn a_drain_notice_right_after_hello_serves_the_next_request_on_the_next_socket() {
    for notice in NOTICES {
        let (url, log) = serve_draining(notice, is_hello, false).await;
        let conn = dial(&url, "DB_1").await.unwrap();
        if notice == ErrorCode::WRONG_SHARD_SERVER {
            for _ in 0..200 {
                if hellos(&log) == 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(hellos(&log), 2, "redialed with no request waiting");
        }
        let r = conn.exec(&st("INSERT", vec![]), conn.deadline()).await;
        assert_eq!(r.unwrap().rows_affected, 1, "{notice}");
        assert_eq!(hellos(&log), 2, "{notice}");
        conn.close();
    }
}

/// Catches: a refused upgrade read as one refusal for both causes. 401 is unauthenticated, 403
/// permission denied: both final, neither redialed.
#[tokio::test]
async fn a_refused_upgrade_names_its_cause() {
    for (status, code) in [(401, ErrorCode::UNAUTHENTICATED), (403, ErrorCode::PERMISSION_DENIED)] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/db", listener.local_addr().unwrap());
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let _ = tcp.read(&mut request).await.unwrap();
            let response = format!("HTTP/1.1 {status} Refused\r\ncontent-length: 0\r\n\r\n");
            tcp.write_all(response.as_bytes()).await.unwrap();
        });
        let err = dial(&url, "DB_1").await.err().unwrap();
        assert!(matches!(&err, WsdbError::Server { code: c, .. } if *c == code), "{status}: {err}");
        assert!(err.is_final(), "{status}: final");
    }
}

/// Catches: a frame the socket never took reported as sent. A write queued to a writer whose
/// socket then fails is not sent: nothing applied, retryable, never uncertain.
#[tokio::test]
async fn a_frame_the_socket_fails_to_send_is_not_sent() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let replace = |tx: Writer| {
        let mut st = conn.0.state.lock();
        let live = st.live.unwrap();
        st.writers.insert(live, tx);
    };
    // the writer gone, its queue refuses every frame
    let (tx, rx) = mpsc::unbounded_channel::<Outgoing>();
    drop(rx);
    replace(tx);
    let err = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.err().unwrap();
    assert!(matches!(err, WriteError::NotSent(_)), "{err}");
    // queued, then dropped by a writer that failed before sending it
    let (tx, mut rx) = mpsc::unbounded_channel::<Outgoing>();
    replace(tx);
    tokio::spawn(async move { drop(rx.recv().await) });
    let err = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.err().unwrap();
    assert!(matches!(err, WriteError::NotSent(_)), "{err}");
    assert!(StorageError::from(err).retryable());
    conn.close();
}

/// Catches: a frame on request id 0 other than a drain notice dropped silently, leaving the
/// requests in flight waiting out their deadlines. It is a protocol error that ends the socket:
/// a write in flight is `Disconnected` (outcome unknown), never sent again.
#[tokio::test]
async fn an_unexpected_frame_on_request_id_0_ends_the_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let log: Served = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                while let Some(Ok(Message::Binary(b))) = ws.next().await {
                    let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                    seen.lock().push(msg.clone());
                    let answers = match &msg.message {
                        // on request id 0, and not a notice; the exec itself goes unanswered
                        Some(pb::client_message::Message::Exec(s)) if s.sql == "stray" => {
                            vec![error(ErrorCode::SQL_ERROR, "not a notice")]
                        }
                        _ => script(&msg)
                            .into_iter()
                            .map(|mut a| {
                                a.request_id = msg.request_id;
                                a
                            })
                            .collect(),
                    };
                    for answer in answers {
                        if ws.send(Message::binary(answer.encode_to_vec())).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    });
    let conn = dial(&url, "DB_1").await.unwrap();
    let deadline = conn.0.clock.now().saturating_add(Duration::from_secs(2));
    let err = conn.exec(&st("stray", vec![]), deadline).await.err().unwrap();
    assert!(matches!(err, WriteError::Sent(WsdbError::Disconnected)), "{err}");
    assert!(matches!(StorageError::from(err), StorageError::Uncertain(_)));
    assert_eq!(sent(&log, is_exec).len(), 1, "never sent again");
    let r = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.unwrap();
    assert_eq!(r.rows_affected, 1, "the next request redials");
    conn.close();
}

/// Catches: the server's `not_found` on request id 0 (the database deleted with the socket open)
/// read as a protocol error: the write in flight failed `Disconnected` (outcome unknown) and the
/// next call redialed. The write applied nothing and fails not found; so does every later call,
/// unsent, with no redial.
#[tokio::test]
async fn a_database_deleted_with_the_socket_open_fails_not_found_for_good() {
    let (url, log) = serve_draining(ErrorCode::NOT_FOUND, is_exec, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let err = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.err().unwrap();
    assert!(
        matches!(&err, WriteError::NotSent(WsdbError::Server { code: ErrorCode::NOT_FOUND, .. })),
        "{err}"
    );
    assert!(matches!(StorageError::from(err), StorageError::NotFound(_)), "applied nothing");
    let err = conn.query_all(&st("SELECT", vec![]), conn.deadline()).await.err().unwrap();
    assert!(matches!(&err, WsdbError::Server { code: ErrorCode::NOT_FOUND, .. }), "{err}");
    assert_eq!(sent(&log, is_exec).len(), 1, "never sent again");
    assert!(sent(&log, is_query).is_empty(), "a later call is not sent");
    assert_eq!(hellos(&log), 1, "no redial");
}

/// Catches: new requests waiting for a socket refused `wrong_shard_server` to finish its
/// requests in flight. As with HTTP/2's GOAWAY, the next socket opens at once: the refused query
/// and a later exec are served there while the first socket still holds a write, which then
/// gets its own answer there.
#[tokio::test]
async fn new_requests_never_wait_behind_a_retiring_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let log: Served = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    let (served, served_next) = signal::signal(false);
    let served = Arc::new(served);
    tokio::spawn(async move {
        let first = Arc::new(AtomicBool::new(true));
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let (seen, first) = (seen.clone(), first.swap(false, Ordering::SeqCst));
            let (served, served_next) = (served.clone(), served_next.clone());
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                let mut held = None;
                while let Some(Ok(Message::Binary(b))) = ws.next().await {
                    let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                    seen.lock().push(msg.clone());
                    let answer = |m: &pb::ClientMessage| -> Vec<pb::ServerMessage> {
                        let mut out = script(m);
                        out.iter_mut().for_each(|a| a.request_id = m.request_id);
                        out
                    };
                    let mut answers = Vec::new();
                    match &msg.message {
                        Some(pb::client_message::Message::Exec(_)) if first => held = Some(msg),
                        Some(pb::client_message::Message::Query(_)) if first => {
                            let mut refused = error(ErrorCode::WRONG_SHARD_SERVER, "moved");
                            refused.request_id = msg.request_id;
                            answers.push(refused);
                        }
                        Some(pb::client_message::Message::Exec(_)) => {
                            answers = answer(&msg);
                            served.set(true);
                        }
                        _ => answers = answer(&msg),
                    }
                    for a in answers {
                        if ws.send(Message::binary(a.encode_to_vec())).await.is_err() {
                            return;
                        }
                    }
                    if let Some(exec) = held.as_ref().filter(|_| first)
                        && !sent(&seen, is_query).is_empty()
                    {
                        // the held write is answered once the next socket served one
                        served_next.wait_for(|served| *served).await.unwrap();
                        for a in answer(exec) {
                            ws.send(Message::binary(a.encode_to_vec())).await.unwrap();
                        }
                        held = None;
                    }
                }
            });
        }
    });
    let conn = dial(&url, "DB_1").await.unwrap();
    let deadline = conn.0.clock.now().saturating_add(Duration::from_secs(3));
    let (held, insert, select) = (st("HOLD", vec![]), st("INSERT", vec![]), st("SELECT", vec![]));
    let (held, (query, exec)) = tokio::join!(conn.exec(&held, deadline), async {
        while sent(&log, is_exec).is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await; // the held write goes out first
        }
        let query = conn.query_all(&select, deadline).await;
        (query, conn.exec(&insert, deadline).await)
    });
    assert_eq!(query.expect("served on the next socket").len(), 2);
    assert_eq!(exec.expect("served on the next socket").rows_affected, 1);
    assert_eq!(held.expect("its own answer on the retiring socket").rows_affected, 1);
    assert_eq!(sent(&log, is_exec).len(), 2, "each write went out once");
    assert_eq!(hellos(&log), 2);
    conn.close();
}

/// Catches: a database answering every request `wrong_shard_server` driving the request and the
/// redial in a zero-delay loop. The request goes out again at once only the first time, then
/// reaches its caller `Drained` (nothing applied) for its retry owner to back off; the redial
/// after a socket that lived no time waits its jittered backoff on the connection's clock.
#[tokio::test]
async fn a_database_that_moves_on_every_request_is_resent_once_and_redialed_after_a_backoff() {
    fn moving(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        match &m.message {
            Some(pb::client_message::Message::Exec(_)) => {
                vec![error(ErrorCode::WRONG_SHARD_SERVER, "moved")]
            }
            _ => script(m),
        }
    }
    let (url, log) = serve(moving, false).await;
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let background = crate::testing::background();
    let opts = Options::new(&url, "DB_1");
    let conn =
        Conn::dial(opts, minter(), background.spawner, clock.clone(), crate::telemetry::Cx::NONE)
            .await
            .unwrap();
    let deadline = clock.now().saturating_add(Duration::from_secs(60));
    let exec = conn.exec(&st("UPDATE t SET n = 1", vec![]), deadline).await;
    assert!(matches!(exec, Err(WriteError::Sent(WsdbError::Drained))), "{exec:?}");
    let refused = StorageError::from(exec.unwrap_err());
    assert!(matches!(refused, StorageError::Unavailable(_)), "applied nothing: {refused:?}");
    assert_eq!(sent(&log, is_exec).len(), 2, "sent again at once, once");
    // the first redial was at once; the next waits at least half the first step on the clock
    assert_eq!(hellos(&log), 2);
    while hellos(&log) < 3 {
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        clock.advance(Duration::from_millis(2));
        assert!(clock.now() < Ts::from_nanos(1_000_000_000), "never redialed");
    }
    let waited = clock.now().saturating_duration_since(Ts::ZERO);
    assert!(waited >= RETRY_MIN_BACKOFF / 2, "redialed after {waited:?}");
    conn.aclose().await;
}
