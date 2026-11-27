use super::notices::serve_draining;
use super::*;
use crate::ThreadDbConnection as _;
use crate::backends::agentdb::{DbPool, Pooled};
use crate::telemetry::tests::{Collect, recording};
use crate::thread::SchemaCell;
use lk_agents_abi::VirtualClock;

const DURATION: &str = "lk.agents.storage.request_duration";
const RETRIES: &str = "lk.agents.storage.retries";
const REDIALS: &str = "lk.agents.storage.redials";

fn labeled(value: f64, labels: &[&str]) -> (f64, Vec<String>) {
    (value, labels.iter().map(|l| l.to_string()).collect())
}

/// Waits, on real time, until `sink` holds `n` measurements of `instrument`.
async fn recorded(sink: &Collect, instrument: &str, n: usize) {
    for _ in 0..2000 {
        if sink.labeled(instrument).len() >= n {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("{instrument} recorded {:?}, not {n}", sink.labeled(instrument));
}

/// A server whose first query is answered `retry` once `release` fires, telling `got` it read
/// it; every other frame [`script`] answers.
async fn serve_refusing_once(
    got: mpsc::UnboundedSender<()>,
    release: oneshot::Receiver<()>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let mut release = Some((got, release));
        loop {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            while let Some(Ok(Message::Binary(b))) = ws.next().await {
                let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
                let query = matches!(msg.message, Some(pb::client_message::Message::Query(_)));
                let mut answers = match if query { release.take() } else { None } {
                    Some((got, release)) => {
                        got.send(()).unwrap();
                        release.await.unwrap();
                        vec![error(ErrorCode::RETRY, "refused under load")]
                    }
                    None => script(&msg),
                };
                answers.iter_mut().for_each(|a| a.request_id = msg.request_id);
                for answer in answers {
                    ws.send(Message::binary(answer.encode_to_vec())).await.unwrap();
                }
            }
        }
    });
    url
}

/// Catches: a refusal its retry owner sent again left uncounted, or the request measured from
/// the server's side or over its whole retry. The refused query took 30 ms on the connection's
/// clock and is classed `capacity`; its resend after the backoff is a request of its own, `ok`;
/// the resend is counted once under `capacity`.
#[tokio::test]
async fn a_query_refused_for_capacity_then_answered_records_both_requests_and_one_retry() {
    let (got, mut got_rx) = mpsc::unbounded_channel();
    let (release, release_rx) = oneshot::channel();
    let url = serve_refusing_once(got, release_rx).await;
    let (cx, sink) = recording();
    let clock = Arc::new(VirtualClock::new());
    let background = crate::testing::background();
    let opts = Options::new(&url, "DB_1");
    let conn = Conn::dial(opts, minter(), background.spawner, clock.clone(), cx.clone());
    let pooled = Pooled {
        conn: conn.await.unwrap(),
        schema: SchemaCell::default(),
        pool: Arc::new(DbPool::new(clock.clone(), cx)),
        closed: AtomicBool::new(false),
    };
    let (rows, ()) = tokio::join!(pooled.query(Statement::raw("SELECT n, s FROM t")), async {
        got_rx.recv().await.unwrap();
        clock.advance(Duration::from_millis(30)); // the server holds its answer meanwhile
        release.send(()).unwrap();
        recorded(&sink, RETRIES, 1).await;
        clock.advance(Duration::from_secs(5)); // past any first backoff step
    });
    assert_eq!(rows.expect("answered on its resend").len(), 2);
    assert_eq!(
        sink.labeled(DURATION),
        vec![
            labeled(0.0, &["hello", "ok"]),
            labeled(0.03, &["query", "capacity"]),
            labeled(0.0, &["query", "ok"]),
        ]
    );
    assert_eq!(sink.labeled(RETRIES), vec![labeled(1.0, &["capacity"])]);
    assert!(sink.labeled(REDIALS).is_empty());
    pooled.close().await;
}

/// Catches: a socket the server ended because its database moved redialed uncounted, or the
/// connection's own resend of the request it drained left out of the retries. The exec is sent
/// again on the redialed socket and measured once, `ok`, its resend and the redial each counted
/// under `moved`.
#[tokio::test]
async fn a_database_that_moved_mid_request_records_the_redial_and_the_resend() {
    let (url, log) = serve_draining(ErrorCode::WRONG_SHARD_SERVER, is_exec, false).await;
    let (cx, sink) = recording();
    let clock = Arc::new(VirtualClock::new());
    let background = crate::testing::background();
    let opts = Options::new(&url, "DB_1");
    let conn = Conn::dial(opts, minter(), background.spawner, clock.clone(), cx).await.unwrap();
    let deadline = clock.now().saturating_add(Duration::from_secs(10));
    let exec = conn.exec(&st("UPDATE t SET n = 1", vec![]), deadline).await;
    assert_eq!(exec.expect("served on the next socket").rows_affected, 1);
    assert_eq!((hellos(&log), sent(&log, is_exec).len()), (2, 2));
    assert_eq!(
        sink.labeled(DURATION),
        vec![
            labeled(0.0, &["hello", "ok"]),
            labeled(0.0, &["hello", "ok"]),
            labeled(0.0, &["exec", "ok"]),
        ]
    );
    assert_eq!(sink.labeled(RETRIES), vec![labeled(1.0, &["moved"])]);
    assert_eq!(sink.labeled(REDIALS), vec![labeled(1.0, &["moved"])]);
    conn.close();
}
