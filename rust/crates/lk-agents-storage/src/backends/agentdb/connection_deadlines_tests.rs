use super::*;

/// Catches: a request waiting out a redial budget of the connection's own instead of its
/// caller's deadline, or a redial that keeps dialing once no request waits. The request fails
/// unsent at its deadline, and the redial stops with it.
#[tokio::test]
async fn a_request_waits_for_the_redial_only_until_its_deadline() {
    fn unavailable_after_first(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
        static HELLOS: AtomicU32 = AtomicU32::new(0);
        match &m.message {
            Some(pb::client_message::Message::Hello(_))
                if HELLOS.fetch_add(1, Ordering::SeqCst) > 0 =>
            {
                vec![error(ErrorCode::RETRY, "draining")]
            }
            _ => script(m),
        }
    }
    let (url, log) = serve(unavailable_after_first, true).await;
    let conn = dial(&url, "DB_1").await.unwrap();
    assert!(conn.exec(&st("drop", vec![]), conn.deadline()).await.is_err());
    let deadline = conn.0.clock.now().saturating_add(Duration::from_millis(300));
    let err = conn.exec(&st("INSERT", vec![]), deadline).await.err().unwrap();
    assert!(matches!(err, WriteError::NotSent(WsdbError::Timeout)), "{err}");
    assert!(StorageError::from(err).retryable(), "unsent: only retryable");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let dialed = hellos(&log);
    assert!(dialed > 2, "the request's wait redialed with backoff: {dialed}");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(hellos(&log), dialed, "nothing redials once no request waits");
    assert!(!conn.is_closed(), "a refusal that may pass is not final");
    conn.close();
}

/// Catches: a dial attempt bounded only by the dial timeout, outliving the caller's deadline. A
/// server that never answers `Hello` fails the dial at the caller's deadline.
#[tokio::test]
async fn a_dial_attempt_ends_at_the_callers_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        while ws.next().await.is_some() {} // Hello read, never answered
    });
    let mut opts = Options::new(&url, "DB_1");
    opts.dial_timeout = Duration::from_secs(60);
    let background = crate::testing::background();
    let deadline = background.clock.now().saturating_add(Duration::from_millis(200));
    let dial = Conn::dial(opts, minter(), background.spawner, background.clock);
    let dial =
        tokio::time::timeout(Duration::from_secs(5), crate::deadline::within(deadline, dial));
    let err = dial.await.expect("the dial outlived its caller's deadline").err().unwrap();
    assert!(matches!(err, WsdbError::Io(_)), "{err}");
}

/// Catches: a frame the socket already received lost to its writer stopping, which failed an
/// answered request `Disconnected`. Received frames are read first, then the socket ends.
#[tokio::test]
async fn frames_received_before_the_writer_stopped_are_still_read() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        for n in [1, 2] {
            let mut answer = hello_ok();
            answer.request_id = n;
            ws.send(Message::binary(answer.encode_to_vec())).await.unwrap();
        }
        while ws.next().await.is_some() {}
    });
    let ws = connect_with_limit(&url, Headers::new(), Some(MAX_FRAME)).await.unwrap();
    let (_sink, mut read) = ws.split();
    // both frames received before the writer stops
    let first = read.next().await.unwrap().unwrap();
    let (stopped, writer_gone) = oneshot::channel();
    drop(stopped);
    let (reader, _reader_gone) = oneshot::channel();
    let mut socket = Socket {
        generation: 1,
        read,
        writer_gone,
        writer_stopped: false,
        read_deadline: None,
        ping_interval: Duration::ZERO,
        _reader: reader,
    };
    tokio::time::sleep(Duration::from_millis(50)).await; // the second frame arrives
    let clock = lk_agents_net::clock::RealClock::new();
    let id = |m: Message| pb::ServerMessage::decode(m.into_data().as_ref()).unwrap().request_id;
    assert_eq!(id(first), 1);
    assert_eq!(id(socket.next_message(&clock).await.expect("received, so read")), 2);
    assert!(matches!(socket.next_message(&clock).await, Err(WsdbError::Io(_))), "then it ends");
}

/// Catches: a request's answer awaited only until its caller's deadline, spent queued behind a
/// big frame on a busy socket: answered at once, it timed out, outcome unknown. As net/http's
/// `ResponseHeaderTimeout`, the answer is awaited from the moment its frame was sent whole.
#[tokio::test]
async fn a_request_queued_behind_a_big_frame_is_answered() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        let mut stalled = false;
        while let Some(Ok(Message::Binary(b))) = ws.next().await {
            let msg = pb::ClientMessage::decode(b.as_ref()).unwrap();
            if !stalled && matches!(msg.message, Some(pb::client_message::Message::Exec(_))) {
                stalled = true;
                // the big frame's write blocks meanwhile
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            for mut resp in script(&msg) {
                resp.request_id = msg.request_id;
                ws.send(Message::binary(resp.encode_to_vec())).await.unwrap();
            }
        }
    });
    let conn = dial(&url, "DB_1").await.unwrap();
    let (small, big) =
        (st("INSERT", vec![]), st("INSERT", vec![SqlValue::Text("x".repeat(15 << 20))]));
    let deadline = conn.0.clock.now().saturating_add(Duration::from_millis(300));
    let (held, big, small) = tokio::join!(
        conn.exec(&small, conn.deadline()),
        conn.exec(&big, conn.deadline()),
        conn.exec(&small, deadline),
    );
    assert!(held.is_ok() && big.is_ok(), "{held:?} {big:?}");
    assert_eq!(small.unwrap().rows_affected, 1, "sent past its caller's deadline, still answered");
    conn.close();
}

/// Catches: a frame a peer that stopped reading never took waiting forever on the dead socket's
/// writer, held until the redial. The socket ends at its read deadline, its writer with it, and
/// the request is not sent: nothing applied, retryable.
#[tokio::test]
async fn a_frame_a_stalled_peer_never_takes_is_not_sent() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/db", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        let Some(Ok(Message::Binary(b))) = ws.next().await else { return };
        let hello = pb::ClientMessage::decode(b.as_ref()).unwrap();
        let ok = pb::HelloOk { tip: 7, ping_interval_ms: 20, ping_timeout_ms: 20 };
        let mut ok = reply(pb::server_message::Message::HelloOk(ok));
        ok.request_id = hello.request_id;
        ws.send(Message::binary(ok.encode_to_vec())).await.unwrap();
        std::future::pending::<()>().await; // reads nothing more
    });
    let conn = dial(&url, "DB_1").await.unwrap();
    let big = st("INSERT", vec![SqlValue::Text("x".repeat(15 << 20))]);
    let err = conn.exec(&big, conn.deadline()).await.err().unwrap();
    assert!(matches!(err, WriteError::NotSent(_)), "{err}");
    assert!(StorageError::from(err).retryable());
    conn.close();
}
