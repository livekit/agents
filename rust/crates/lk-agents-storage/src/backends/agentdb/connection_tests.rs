use super::*;
use crate::StorageError;
use crate::backends::agentdb::rows::{decode_batch, encode_batch};
use crate::sql::SqlValue;
use std::sync::atomic::AtomicBool;
use tokio::net::TcpListener;

#[test]
fn a_mixed_column_round_trips() {
    let rows: Vec<Row> = vec![
        vec![SqlValue::Int(1), SqlValue::Text("a".into()), SqlValue::Null],
        vec![SqlValue::Text("mixed".into()), SqlValue::Text("".into()), SqlValue::Blob(vec![1, 2])],
        vec![SqlValue::Double(2.5), SqlValue::Text("bb".into()), SqlValue::Int(-7)],
    ];
    let batch = encode_batch(&rows, 3);
    assert_eq!(decode_batch(&batch).unwrap(), rows);
}

#[test]
fn a_malformed_column_is_refused_not_read_as_null() {
    let mut batch = encode_batch(&[vec![SqlValue::Int(1)], vec![SqlValue::Int(2)]], 1);
    batch.columns[0].ints.pop();
    assert!(matches!(decode_batch(&batch), Err(WsdbError::Protocol(_))));
    let mut batch = encode_batch(&[vec!["ok".into()]], 1);
    batch.columns[0].text_data = vec![0xff, 0xfe];
    assert!(matches!(decode_batch(&batch), Err(WsdbError::Protocol(_))));
}

// A fake `wsdb` server on a loopback listener: enough of the protocol to exercise the
// client's framing, the per-statement `Batch` shape, the credit flow and the reconnect.

pub(crate) type Served = Arc<Mutex<Vec<pb::ClientMessage>>>;

/// Accepts connections forever; `script` answers one request. Every client frame is kept in
/// the returned log. An exec of `drop` closes the socket when `drop_once` is set, once. An
/// `idle_notice` among the answers goes on request id 0, and the socket closes after it;
/// [`serve_draining`] ends a socket with a notice in place of a request's answer.
pub(crate) async fn serve(
    script: fn(&pb::ClientMessage) -> Vec<pb::ServerMessage>,
    drop_once: bool,
) -> (String, Served) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let log: Served = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    tokio::spawn(async move {
        let dropped_once = Arc::new(AtomicBool::new(false));
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let (seen, dropped_once) = (seen.clone(), dropped_once.clone());
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
                while let Some(Ok(Message::Binary(b))) = ws.next().await {
                    let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                    seen.lock().push(msg.clone());
                    if drop_once
                        && matches!(&msg.message, Some(pb::client_message::Message::Exec(s)) if s.sql == "drop")
                        && !dropped_once.swap(true, Ordering::SeqCst)
                    {
                        break; // the socket closes with the request unanswered
                    }
                    let mut evicted = false;
                    for mut resp in script(&msg) {
                        if resp == idle_notice() {
                            evicted = true;
                        } else {
                            resp.request_id = msg.request_id;
                        }
                        if ws.send(Message::binary(resp.encode_to_vec())).await.is_err() {
                            break;
                        }
                    }
                    if evicted {
                        break;
                    }
                }
            });
        }
    });
    (url, log)
}

pub(crate) fn reply(m: pb::server_message::Message) -> pb::ServerMessage {
    pb::ServerMessage { request_id: 0, message: Some(m) }
}

/// An error answer of `code`.
pub(crate) fn error(code: ErrorCode, description: &str) -> pb::ServerMessage {
    reply(pb::server_message::Message::Error(pb::Error {
        code: code.0,
        name: code.to_string(),
        description: description.into(),
    }))
}

/// The server's notice that it evicted the database while the socket idled.
pub(crate) fn idle_notice() -> pb::ServerMessage {
    error(ErrorCode::IDLE_EVICTED, "evicted while idle")
}

pub(crate) fn hello_ok() -> pb::ServerMessage {
    reply(pb::server_message::Message::HelloOk(pb::HelloOk {
        tip: 7,
        ping_interval_ms: 0,
        ping_timeout_ms: 0,
    }))
}

pub(crate) fn script(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
    use pb::client_message::Message as C;
    use pb::server_message::Message as S;
    match &m.message {
        Some(C::Hello(h)) if h.token.split('.').count() == 3 && h.database_id == "DB_1" => {
            vec![hello_ok()]
        }
        Some(C::Hello(_)) => vec![error(ErrorCode::NOT_FOUND, "no such database")],
        Some(C::Exec(s)) if s.sql == "boom" => vec![error(ErrorCode::SQL_ERROR, "constraint")],
        Some(C::Exec(s)) if s.sql == "busy" => vec![error(ErrorCode::RETRY, "busy")],
        Some(C::Exec(_)) => vec![reply(S::ExecResult(pb::ExecResult {
            rows_affected: 1,
            last_insert_id: 9,
            tip: 8,
            statement: 0,
        }))],
        Some(C::Query(_)) => {
            // Two batches, so the client's per-batch credit is observable.
            let mut b1 = encode_batch(&[vec![SqlValue::Int(1), "a".into()]], 2);
            let mut b2 = encode_batch(&[vec![SqlValue::Int(2), SqlValue::Null]], 2);
            b1.statement = 0;
            b2.statement = 0;
            vec![
                reply(S::Columns(pb::Columns {
                    names: vec!["n".into(), "s".into()],
                    statement: 0,
                })),
                reply(S::ColumnBatch(b1)),
                reply(S::ColumnBatch(b2)),
                reply(S::Done(pb::Done { tip: 8, total_rows: 2 })),
            ]
        }
        Some(C::Batch(b)) => {
            // Statement 0: an exec. Statement 1: rows. Statement 2: a RETURNING with no rows,
            // which still sends its Columns. Statement 3: rows again. Statements past the
            // fourth get no result.
            assert!(b.statements.len() >= 4);
            let mut rows1 = encode_batch(&[vec![SqlValue::Int(3)]], 1);
            rows1.statement = 1;
            let mut rows3 = encode_batch(&[vec!["x".into()], vec!["y".into()]], 1);
            rows3.statement = 3;
            vec![
                reply(S::ExecResult(pb::ExecResult {
                    rows_affected: 2,
                    last_insert_id: 0,
                    tip: 0,
                    statement: 0,
                })),
                reply(S::Columns(pb::Columns { names: vec!["version".into()], statement: 1 })),
                reply(S::ColumnBatch(rows1)),
                reply(S::Columns(pb::Columns { names: vec!["version".into()], statement: 2 })),
                reply(S::Columns(pb::Columns { names: vec!["id".into()], statement: 3 })),
                reply(S::ColumnBatch(rows3)),
                reply(S::Done(pb::Done { tip: 9, total_rows: 3 })),
            ]
        }
        _ => vec![],
    }
}

fn st(sql: &'static str, params: Vec<SqlValue>) -> Statement {
    Statement::raw(sql).with(params)
}

fn minter() -> TokenMinter {
    TokenMinter::new("devkey", "secret").unwrap()
}

pub(crate) async fn dial(url: &str, db: &str) -> Result<Conn, WsdbError> {
    let mut opts = Options::new(url, db);
    opts.reconnect.min_backoff = Duration::from_millis(10);
    let background = crate::testing::background();
    Conn::dial(opts, minter(), background.spawner, background.clock).await
}

#[tokio::test]
async fn hello_carries_the_token_and_a_refused_database_is_a_server_error() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert_eq!(conn.tip(), 7);
    let err = dial(&url, "DB_nope").await.err().unwrap();
    assert!(matches!(&err, WsdbError::Server { code: ErrorCode::NOT_FOUND, .. }), "{err}");
    conn.close();
}

#[tokio::test]
async fn exec_query_and_credits() {
    let (url, log) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let r = conn.exec(&st("INSERT", vec!["v".into()]), conn.deadline()).await.unwrap();
    assert_eq!((r.rows_affected, r.last_insert_id, r.tip), (1, 9, 8));
    let err = conn.exec(&st("boom", vec![]), conn.deadline()).await.err().unwrap();
    assert!(matches!(err, WriteError::Sent(WsdbError::Server { code: ErrorCode::SQL_ERROR, .. })));
    // Backpressure is returned once; the caller owns retry policy.
    let err = conn.exec(&st("busy", vec![]), conn.deadline()).await.err().unwrap();
    assert!(matches!(err, WriteError::Sent(WsdbError::Server { code: ErrorCode::RETRY, .. })));
    let busy = |log: &Served| {
        log.lock()
            .iter()
            .filter(|m| matches!(&m.message, Some(pb::client_message::Message::Exec(s)) if s.sql == "busy"))
            .count()
    };
    assert_eq!(busy(&log), 1);

    let rows = conn.query_all(&st("SELECT", vec![]), conn.deadline()).await.unwrap();
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int(1), "a".into()], vec![SqlValue::Int(2), SqlValue::Null]]
    );
    // One credit per batch consumed, on the query's request id.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let log = log.lock();
    let query_id = log
        .iter()
        .find(|m| matches!(m.message, Some(pb::client_message::Message::Query(_))))
        .unwrap()
        .request_id;
    let credits = log
        .iter()
        .filter(|m| {
            m.request_id == query_id
                && matches!(m.message, Some(pb::client_message::Message::Credit(_)))
        })
        .count();
    assert_eq!(credits, 2);
}

#[tokio::test]
async fn a_batch_answers_per_statement_and_an_empty_returning_is_rows() {
    let (url, log) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let stmts: Vec<Statement> = (0..4).map(|_| st("S", vec![])).collect();
    let out = conn.batch(&stmts, conn.deadline()).await.unwrap();
    assert_eq!(conn.tip(), 9);
    assert_eq!(out.results[0], StatementResult::Exec { rows_affected: 2, last_insert_id: 0 });
    assert_eq!(out.results[1].rows(), &[vec![SqlValue::Int(3)]]);
    // the acquire that lost: Columns, no ColumnBatch
    assert_eq!(out.results[2], StatementResult::Rows(vec![]));
    assert_eq!(out.results[3].rows().len(), 2);
    // one credit per batch consumed, past the window the server grants: none up front
    let credits = |m: &pb::ClientMessage| {
        matches!(m.message, Some(pb::client_message::Message::Credit(pb::Credit { batches: 1 })))
    };
    for _ in 0..200 {
        if sent(&log, credits).len() >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    let all_credits =
        |m: &pb::ClientMessage| matches!(m.message, Some(pb::client_message::Message::Credit(_)));
    assert_eq!(sent(&log, all_credits), sent(&log, credits), "every credit is for one batch");
    assert_eq!(sent(&log, credits), vec![sent(&log, is_batch)[0]; 2]);
}

#[tokio::test]
async fn a_batch_answer_missing_a_statement_is_malformed_not_a_zero_row_success() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let stmts: Vec<Statement> = (0..5).map(|_| st("S", vec![])).collect();
    assert!(matches!(
        conn.batch(&stmts, conn.deadline()).await,
        Err(WriteError::Sent(WsdbError::Protocol(_)))
    ));
    conn.close();
}

#[tokio::test]
async fn a_dropped_socket_fails_the_request_in_flight_and_redials() {
    let (url, log) = serve(script, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    // The fake drops the socket on this exec: the request fails and the client redials.
    let err = conn.exec(&st("drop", vec![]), conn.deadline()).await.err().unwrap();
    assert!(matches!(err, WriteError::Sent(WsdbError::Disconnected)), "{err}");
    let r = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.unwrap();
    assert_eq!(r.rows_affected, 1);
    let hellos = log
        .lock()
        .iter()
        .filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_))))
        .count();
    assert_eq!(hellos, 2, "one Hello per dial: the first and the redial");
    conn.close();
    assert!(matches!(
        conn.exec(&st("INSERT", vec![]), conn.deadline()).await,
        Err(WriteError::NotSent(WsdbError::Closed))
    ));
}

/// A socket the server evicted while it idled is redialed only once a call wants the
/// connection: two idle connections at a warm cap of one would otherwise evict each other's
/// database in a loop. Dormant meanwhile, it is no pool's to take back.
#[tokio::test]
async fn an_idle_evicted_socket_is_redialed_only_when_a_call_wants_it() {
    fn evicting(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        match &m.message {
            Some(pb::client_message::Message::Exec(s)) if s.sql == "idle" => {
                let mut answer = script(m);
                answer.push(idle_notice());
                answer
            }
            _ => script(m),
        }
    }
    let (url, log) = serve(evicting, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    conn.exec(&st("idle", vec![]), conn.deadline()).await.unwrap();
    for _ in 0..200 {
        if !conn.is_idle() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(!conn.is_idle(), "an idle-evicted connection is dormant, no pool's to take back");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(hellos(&log), 1, "nothing redials an idle socket");
    let r = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.unwrap();
    assert_eq!(r.rows_affected, 1);
    assert_eq!(hellos(&log), 2, "the call redialed once");
    assert!(conn.is_idle());
    conn.close();
}

fn hellos(log: &Served) -> usize {
    let log = log.lock();
    log.iter().filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_)))).count()
}

/// The request ids `log` holds for frames `kind` picks, in the order sent.
fn sent(log: &Served, kind: impl Fn(&pb::ClientMessage) -> bool) -> Vec<u32> {
    log.lock().iter().filter(|m| kind(m)).map(|m| m.request_id).collect()
}

fn is_exec(m: &pb::ClientMessage) -> bool {
    matches!(m.message, Some(pb::client_message::Message::Exec(_)))
}

fn is_batch(m: &pb::ClientMessage) -> bool {
    matches!(m.message, Some(pb::client_message::Message::Batch(_)))
}

fn is_query(m: &pb::ClientMessage) -> bool {
    matches!(m.message, Some(pb::client_message::Message::Query(_)))
}

fn is_hello(m: &pb::ClientMessage) -> bool {
    matches!(m.message, Some(pb::client_message::Message::Hello(_)))
}

/// A socket that stops answering pings is dropped past its read deadline, and the next request
/// redials it: nothing redials while no request waits.
#[tokio::test]
async fn a_socket_that_stops_answering_pings_is_redialed_for_the_next_request() {
    fn silent(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        use pb::client_message::Message as C;
        match &m.message {
            Some(C::Hello(_)) => {
                vec![reply(pb::server_message::Message::HelloOk(pb::HelloOk {
                    tip: 7,
                    ping_interval_ms: 20,
                    ping_timeout_ms: 20,
                }))]
            }
            Some(C::Ping(_)) => vec![], // the node is gone: no pong
            other => {
                script(&pb::ClientMessage { request_id: m.request_id, message: other.clone() })
            }
        }
    }
    let (url, log) = serve(silent, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(!conn.is_idle(), "dropped past the read deadline");
    assert_eq!(hellos(&log), 1, "nothing redials while no request waits");
    assert_eq!(conn.exec(&st("INSERT", vec![]), conn.deadline()).await.unwrap().rows_affected, 1);
    assert_eq!(hellos(&log), 2, "the request redialed");
    conn.close();
}

#[tokio::test]
async fn a_redial_the_server_refuses_for_good_fails_later_calls_with_its_error() {
    fn refuses_again(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        static HELLOS: AtomicU32 = AtomicU32::new(0);
        match &m.message {
            Some(pb::client_message::Message::Hello(_))
                if HELLOS.fetch_add(1, Ordering::SeqCst) > 0 =>
            {
                vec![error(ErrorCode::NOT_FOUND, "the database was deleted")]
            }
            _ => script(m),
        }
    }
    let (url, log) = serve(refuses_again, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert!(matches!(
        conn.exec(&st("drop", vec![]), conn.deadline()).await,
        Err(WriteError::Sent(WsdbError::Disconnected))
    ));
    let err = conn.exec(&st("INSERT", vec![]), conn.deadline()).await.err().unwrap();
    assert!(
        matches!(&err, WriteError::NotSent(WsdbError::Server { code: ErrorCode::NOT_FOUND, .. })),
        "{err}"
    );
    assert_eq!(hellos(&log), 2, "no redial after a final refusal");
}

/// The token rides the upgrade as a bearer header, never in the URL a proxy logs.
#[tokio::test]
async fn the_token_rides_the_upgrade_as_a_bearer_header_not_the_url() {
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut upgrade = None;
        #[allow(clippy::result_large_err, reason = "tungstenite's callback signature")]
        let record = |req: &Request, resp: Response| {
            let auth = req.headers().get("authorization").map(|v| v.to_str().unwrap().to_owned());
            upgrade = Some((req.uri().to_string(), auth));
            Ok(resp)
        };
        let mut ws = tokio_tungstenite::accept_hdr_async(tcp, record).await.unwrap();
        let Some(Ok(Message::Binary(bytes))) = ws.next().await else { panic!("no Hello") };
        let hello = pb::ClientMessage::decode(bytes.as_ref()).unwrap();
        let mut ok = hello_ok();
        ok.request_id = hello.request_id;
        ws.send(Message::binary(ok.encode_to_vec())).await.unwrap();
        upgrade.unwrap()
    });
    let conn = dial(&url, "DB_1").await.unwrap();
    let (uri, auth) = server.await.unwrap();
    assert_eq!(uri, "/db", "nothing rides the URL");
    let token = auth.and_then(|a| a.strip_prefix("Bearer ").map(str::to_owned));
    assert_eq!(token.map(|t| t.split('.').count()), Some(3), "a bearer JWT");
    conn.close();
}

/// Catches: one request whose answers nobody reads holding back every other request's answers.
/// The server streams a query far past its window while its caller reads nothing: an exec on the
/// same socket is still answered.
#[tokio::test]
async fn a_slow_stream_does_not_stall_another_stream() {
    fn flood(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        use pb::server_message::Message as S;
        if !matches!(m.message, Some(pb::client_message::Message::Query(_))) {
            return script(m);
        }
        let mut answer =
            vec![reply(S::Columns(pb::Columns { names: vec!["n".into()], statement: 0 }))];
        answer.extend(
            (0..64).map(|n| reply(S::ColumnBatch(encode_batch(&[vec![SqlValue::Int(n)]], 1)))),
        );
        answer.push(reply(S::Done(pb::Done { tip: 8, total_rows: 64 })));
        answer
    }
    let (url, log) = serve(flood, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let deadline = conn.0.clock.now().saturating_add(Duration::from_secs(2));
    let query = pb::client_message::Message::Query(statement(&st("SELECT", vec![])));
    let mut frame = pb::ClientMessage { request_id: 0, message: Some(query) };
    let mut stalled = conn.send(&mut frame, deadline, conn.window()).await.unwrap();
    let exec = conn.exec(&st("INSERT", vec![]), deadline).await;
    assert_eq!(exec.expect("answered past the stalled stream").rows_affected, 1);
    // the stalled stream reads what its window held, then that it was cancelled
    let mut read = 0;
    let end = loop {
        match stalled.next().await {
            Ok(_) => read += 1,
            Err(error) => break error,
        }
    };
    assert!(matches!(end, WsdbError::Overrun), "{end}");
    assert_eq!(read, conn.window(), "its window, and nothing past it");
    let cancelled = |m: &pb::ClientMessage| {
        m.request_id == stalled.id
            && matches!(m.message, Some(pb::client_message::Message::Cancel(_)))
    };
    for _ in 0..200 {
        if !sent(&log, cancelled).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(sent(&log, cancelled).len(), 1, "the server is told to stop");
    conn.close();
}

#[tokio::test]
async fn a_frame_over_the_limit_is_refused_before_it_is_sent() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let big = SqlValue::Text("x".repeat(MAX_FRAME + 1));
    assert!(matches!(
        conn.exec(&st("INSERT", vec![big]), conn.deadline()).await,
        Err(WriteError::NotSent(WsdbError::TooLarge(_)))
    ));
    assert_eq!(
        conn.exec(&st("INSERT", vec![]), conn.deadline()).await.unwrap().rows_affected,
        1,
        "still open"
    );
    conn.close();
}

#[tokio::test]
async fn dropping_the_last_handle_closes_the_connection() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let inner = Arc::downgrade(&conn.0);
    let other = conn.clone();
    drop(conn);
    assert!(!other.is_closed(), "a clone keeps it open");
    drop(other);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(inner.upgrade().is_none(), "its tasks no longer hold it");
}

/// Real time until armed; then whichever task reads the time panics (waits still work).
#[derive(Default)]
struct Failing {
    real: lk_agents_net::clock::RealClock,
    armed: AtomicBool,
}

impl Failing {
    fn check(&self) {
        assert!(!self.armed.load(Ordering::Acquire), "the clock failed");
    }
}

impl Clock for Failing {
    fn now(&self) -> lk_agents_abi::Ts {
        self.check();
        self.real.now()
    }
    fn wait_until(&self, at: lk_agents_abi::Ts) -> lk_agents_abi::BoxFuture<'_, ()> {
        self.real.wait_until(at)
    }
    fn wall(&self) -> std::time::SystemTime {
        self.real.wall()
    }
}

// Catches: a supervisor that dies while redialing leaving later requests waiting on a ready
// gate nobody will open.
#[tokio::test]
async fn a_supervisor_that_dies_mid_redial_fails_later_requests() {
    let (url, _) = serve(script, true).await;
    let opts = Options::new(&url, "DB_1");
    let clock = Arc::new(Failing::default());
    let spawner = crate::testing::background().spawner;
    let conn = Conn::dial(opts, minter(), spawner, clock.clone()).await.unwrap();
    assert!(conn.exec(&st("drop", vec![]), conn.deadline()).await.is_err());
    // The next request wakes the redial, whose dial deadline panics the supervisor.
    let (insert, deadline) = (st("INSERT", vec![]), conn.deadline());
    clock.armed.store(true, Ordering::Release);
    let next = tokio::time::timeout(Duration::from_secs(5), conn.exec(&insert, deadline));
    assert!(matches!(next.await, Ok(Err(_))), "a request after the supervisor died hung");
    assert!(conn.is_closed());
}

#[test]
fn a_column_cannot_hide_unannounced_values() {
    let mut batch = encode_batch(&[vec![SqlValue::Int(1)]], 1);
    batch.columns[0].ints.push(2);
    assert!(matches!(decode_batch(&batch), Err(WsdbError::Protocol(_))));
    let mut batch = encode_batch(&[vec![SqlValue::Text("ok".into())]], 1);
    batch.columns[0].text_data.extend_from_slice(b"hidden");
    assert!(matches!(decode_batch(&batch), Err(WsdbError::Protocol(_))));
}

#[tokio::test]
async fn a_query_refuses_rows_without_matching_columns() {
    fn malformed(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        if matches!(m.message, Some(pb::client_message::Message::Query(_))) {
            vec![
                reply(pb::server_message::Message::ColumnBatch(encode_batch(
                    &[vec![SqlValue::Int(1)]],
                    1,
                ))),
                reply(pb::server_message::Message::Done(pb::Done { tip: 8, total_rows: 1 })),
            ]
        } else {
            script(m)
        }
    }
    let (url, _) = serve(malformed, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert!(matches!(
        conn.query_all(&st("SELECT", vec![]), conn.deadline()).await,
        Err(WsdbError::Protocol(_))
    ));
    conn.close();
}

#[tokio::test]
async fn duplicate_batch_results_cannot_overwrite_an_authoritative_result() {
    fn malformed(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        if matches!(m.message, Some(pb::client_message::Message::Batch(_))) {
            vec![
                reply(pb::server_message::Message::ExecResult(pb::ExecResult {
                    statement: 0,
                    rows_affected: 1,
                    ..Default::default()
                })),
                reply(pb::server_message::Message::ExecResult(pb::ExecResult {
                    statement: 0,
                    rows_affected: 0,
                    ..Default::default()
                })),
                reply(pb::server_message::Message::Done(pb::Done { tip: 8, total_rows: 0 })),
            ]
        } else {
            script(m)
        }
    }
    let (url, _) = serve(malformed, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert!(matches!(
        conn.batch(&[st("UPDATE", vec![])], conn.deadline()).await,
        Err(WriteError::Sent(WsdbError::Protocol(_)))
    ));
    conn.close();
}

#[tokio::test]
async fn a_hello_for_another_request_cannot_open_the_database() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        let Some(Ok(Message::Binary(bytes))) = ws.next().await else { panic!("no Hello") };
        let hello = pb::ClientMessage::decode(bytes.as_ref()).unwrap();
        let mut response = hello_ok();
        response.request_id = hello.request_id + 1;
        ws.send(Message::binary(response.encode_to_vec())).await.unwrap();
    });
    assert!(matches!(dial(&url, "DB_1").await, Err(WsdbError::Protocol(_))));
    server.await.unwrap();
}

#[tokio::test]
async fn close_settles_all_connection_resources_before_returning() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let inner = Arc::downgrade(&conn.0);
    // Both waiters observe the same receipt, including when one is cancelled.
    {
        let cancelled = conn.aclose();
        tokio::pin!(cancelled);
        tokio::select! {
            _ = &mut cancelled => {},
            _ = tokio::task::yield_now() => {},
        }
        tokio::join!(conn.aclose(), conn.aclose());
    }
    drop(conn);
    assert!(inner.upgrade().is_none(), "no resource task retains the closed connection");
}

#[test]
fn compressed_rows_cannot_amplify_beyond_the_frame_memory_budget() {
    let batch = pb::ColumnBatch {
        rows: 1_000_000,
        columns: vec![pb::Column { types: vec![0; 1_000_000], ..Default::default() }],
        statement: 0,
    };
    assert!(matches!(decode_batch(&batch), Err(WsdbError::Protocol(_))));
}

#[path = "connection_notices_tests.rs"]
mod notices;

#[path = "connection_deadlines_tests.rs"]
mod deadlines;
