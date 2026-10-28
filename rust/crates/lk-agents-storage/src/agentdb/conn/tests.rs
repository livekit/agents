//! WebSocket database protocol conformance.

use super::*;
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
/// the returned log. An exec of `drop` closes the socket when `drop_once` is set, once.
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
                    seen.lock().unwrap().push(msg.clone());
                    if drop_once
                        && matches!(&msg.message, Some(pb::client_message::Message::Exec(s)) if s.sql == "drop")
                        && !dropped_once.swap(true, Ordering::SeqCst)
                    {
                        break; // the socket closes with the request unanswered
                    }
                    for mut resp in script(&msg) {
                        resp.request_id = msg.request_id;
                        if ws.send(Message::binary(resp.encode_to_vec())).await.is_err() {
                            break;
                        }
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
        Some(C::Hello(_)) => vec![reply(S::Error(pb::Error {
            code: "not_found".into(),
            message: "no such database".into(),
        }))],
        Some(C::Exec(s)) if s.sql == "boom" => {
            vec![reply(S::Error(pb::Error { code: "exec".into(), message: "constraint".into() }))]
        }
        Some(C::Exec(s)) if s.sql == "busy" => {
            vec![reply(S::Error(pb::Error { code: "retry".into(), message: "busy".into() }))]
        }
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
    Statement { sql: sql.into(), params }
}

fn minter() -> TokenMinter {
    TokenMinter::new("devkey", "secret").unwrap()
}

pub(crate) async fn dial(url: &str, db: &str) -> Result<Conn, WsdbError> {
    let mut opts = Options::new(url, db);
    opts.reconnect.min_backoff = Duration::from_millis(10);
    opts.reconnect.budget = Duration::from_secs(5);
    Conn::dial(opts, minter()).await
}

#[tokio::test]
async fn hello_carries_the_token_and_a_refused_database_is_a_server_error() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert_eq!(conn.tip(), 7);
    let err = dial(&url, "DB_nope").await.err().unwrap();
    assert!(matches!(&err, WsdbError::Server { code, .. } if code == "not_found"), "{err}");
    conn.close();
}

#[tokio::test]
async fn exec_query_and_credits() {
    let (url, log) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let r = conn.exec(&st("INSERT", vec!["v".into()])).await.unwrap();
    assert_eq!((r.rows_affected, r.last_insert_id, r.tip), (1, 9, 8));
    let err = conn.exec(&st("boom", vec![])).await.err().unwrap();
    assert!(matches!(err, WsdbError::Server { code, .. } if code == "exec"));
    // `retry` applied nothing: sent three times, then the error
    let err = conn.exec(&st("busy", vec![])).await.err().unwrap();
    assert!(matches!(err, WsdbError::Server { code, .. } if code == "retry"));
    let busy = |log: &Served| {
        log.lock()
            .unwrap()
            .iter()
            .filter(|m| matches!(&m.message, Some(pb::client_message::Message::Exec(s)) if s.sql == "busy"))
            .count()
    };
    assert_eq!(busy(&log), 3);

    let rows = conn.query_all(&st("SELECT", vec![])).await.unwrap();
    assert_eq!(
        rows,
        vec![vec![SqlValue::Int(1), "a".into()], vec![SqlValue::Int(2), SqlValue::Null]]
    );
    // One credit per batch consumed, on the query's request id.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let log = log.lock().unwrap();
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
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let stmts: Vec<Statement> = (0..4).map(|_| st("S", vec![])).collect();
    let out = conn.batch(&stmts).await.unwrap();
    assert_eq!(conn.tip(), 9);
    assert_eq!(out.results[0], StatementResult::Exec { rows_affected: 2, last_insert_id: 0 });
    assert_eq!(out.results[1].rows(), &[vec![SqlValue::Int(3)]]);
    // the acquire that lost: Columns, no ColumnBatch
    assert_eq!(out.results[2], StatementResult::Rows(vec![]));
    assert_eq!(out.results[3].rows().len(), 2);
}

#[tokio::test]
async fn a_batch_answer_missing_a_statement_is_malformed_not_a_zero_row_success() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let stmts: Vec<Statement> = (0..5).map(|_| st("S", vec![])).collect();
    assert!(matches!(conn.batch(&stmts).await, Err(WsdbError::Protocol(_))));
    conn.close();
}

#[tokio::test]
async fn a_dropped_socket_fails_the_request_in_flight_and_redials() {
    let (url, log) = serve(script, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    // The fake drops the socket on this exec: the request fails and the client redials.
    let err = conn.exec(&st("drop", vec![])).await.err().unwrap();
    assert!(matches!(err, WsdbError::Disconnected), "{err}");
    let r = conn.exec(&st("INSERT", vec![])).await.unwrap();
    assert_eq!(r.rows_affected, 1);
    let hellos = log
        .lock()
        .unwrap()
        .iter()
        .filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_))))
        .count();
    assert_eq!(hellos, 2, "one Hello per dial: the first and the redial");
    conn.close();
    assert!(matches!(conn.exec(&st("INSERT", vec![])).await, Err(WsdbError::Closed)));
}

fn hellos(log: &Served) -> usize {
    let log = log.lock().unwrap();
    log.iter().filter(|m| matches!(m.message, Some(pb::client_message::Message::Hello(_)))).count()
}

#[tokio::test]
async fn a_socket_that_stops_answering_pings_is_redialed() {
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
    assert!(hellos(&log) >= 2, "redialed past the read deadline");
    assert_eq!(conn.exec(&st("INSERT", vec![])).await.unwrap().rows_affected, 1);
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
                vec![reply(pb::server_message::Message::Error(pb::Error {
                    code: "not_found".into(),
                    message: "the database was deleted".into(),
                }))]
            }
            _ => script(m),
        }
    }
    let (url, log) = serve(refuses_again, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert!(matches!(conn.exec(&st("drop", vec![])).await, Err(WsdbError::Disconnected)));
    let err = conn.exec(&st("INSERT", vec![])).await.err().unwrap();
    assert!(matches!(&err, WsdbError::Server { code, .. } if code == "not_found"), "{err}");
    assert_eq!(hellos(&log), 2, "no redial after a final refusal");
}

#[tokio::test]
async fn a_frame_over_the_limit_is_refused_before_it_is_sent() {
    let (url, _) = serve(script, false).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    let big = SqlValue::Text("x".repeat(MAX_FRAME + 1));
    assert!(matches!(conn.exec(&st("INSERT", vec![big])).await, Err(WsdbError::TooLarge(_))));
    assert_eq!(conn.exec(&st("INSERT", vec![])).await.unwrap().rows_affected, 1, "still open");
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

#[test]
fn the_token_carries_the_database_admin_grant_and_verifies() {
    use livekit_api::access_token::TokenVerifier;
    let token = TokenMinter::new("devkey", "secret").unwrap().mint().unwrap();
    let claims = TokenVerifier::with_api_key("devkey", "secret").verify(&token).unwrap();
    assert!(claims.agent.database_admin);
    assert_eq!(claims.exp - claims.nbf, TOKEN_TTL.as_secs() as usize);
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
    assert!(matches!(conn.query_all(&st("SELECT", vec![])).await, Err(WsdbError::Protocol(_))));
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
    assert!(matches!(conn.batch(&[st("UPDATE", vec![])]).await, Err(WsdbError::Protocol(_))));
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
        drop(cancelled);
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
