use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::telemetry;
use lk_agents_abi::worker::http::{ExchangeEvent, ExchangeKind, ExchangeMessage, HttpExchange};
use tokio::io::AsyncWriteExt;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

use crate::http::exchange::tests::{
    CASES, HANDSHAKE, attr, body, default_server, echo_size, exchange, get, host, parse, preamble,
    quick_idle, serve_in_pieces, server, start, traced, wire, wire_sized,
};
use crate::http::{EndpointServer, HttpServerLimits, Origin};
use crate::transport::HttpStreamResetCode as Code;

/// Accept, echo text and reverse binary, and report the close code.
fn echo_websocket(closed: Arc<parking_lot::Mutex<Option<u16>>>) -> Arc<EndpointServer> {
    default_server(move |exchange| {
        let closed = closed.clone();
        async move {
            assert_eq!(exchange.head().kind, ExchangeKind::Websocket);
            assert_eq!(exchange.head().subprotocols, ["chat", "v2"]);
            let accept =
                ExchangeMessage::Accept { subprotocol: Some("chat".into()), headers: vec![] };
            exchange.send(accept).await.unwrap();
            loop {
                let reply = match exchange.receive().await.unwrap() {
                    ExchangeEvent::Text { text } => {
                        ExchangeMessage::Text { text: format!("echo:{text}") }
                    }
                    ExchangeEvent::Binary { mut data } => {
                        data.reverse();
                        ExchangeMessage::Binary { data }
                    }
                    ExchangeEvent::Closed { code } => {
                        *closed.lock() = Some(code);
                        return Ok(());
                    }
                    other => panic!("unexpected {other:?}"),
                };
                exchange.send(reply).await.unwrap();
            }
        }
    })
}

fn upgrade() -> tokio_tungstenite::tungstenite::handshake::client::Request {
    let mut request =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(
            "ws://agents.test/ws",
        )
        .unwrap();
    request.headers_mut().insert("sec-websocket-protocol", "chat, v2".parse().unwrap());
    request
}

#[tokio::test]
async fn a_websocket_upgrade_exchanges_frames_and_the_peers_close() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed.clone());
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, response) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    assert_eq!(response.status(), 101);
    assert_eq!(response.headers()["sec-websocket-protocol"], "chat");

    socket.send(Message::text("hi")).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), Message::text("echo:hi"));
    socket.send(Message::binary(b"abc".to_vec())).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), Message::binary(b"cba".to_vec()));
    socket.send(Message::Ping(b"p".to_vec().into())).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), Message::Pong(b"p".to_vec().into()));

    socket.close(None).await.unwrap();
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Status)));
    assert!(end.finished() && end.reset().is_none());
    assert_eq!(server.connections.open(), 0);
}

#[tokio::test]
async fn drain_closes_open_websockets_with_going_away() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed.clone());
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, _) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    let draining = tokio::spawn({
        let server = server.clone();
        async move { server.drain().await }
    });
    let Some(Ok(Message::Close(Some(frame)))) = socket.next().await else {
        panic!("a close frame");
    };
    assert_eq!(frame.code, CloseCode::Away);
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), draining).await.unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Away)));
    assert!(end.finished());
    assert!(server.websockets.lock().open.is_empty());
}

/// The close code of the last close frame the server sent after its 101.
fn close_code(out: &[u8]) -> u16 {
    let head = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    assert!(out.starts_with(b"HTTP/1.1 101"));
    assert!(String::from_utf8_lossy(&out[..head]).contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    let frame = &out[head..];
    assert_eq!(frame[0], 0x88, "a close frame");
    u16::from_be_bytes([frame[2], frame[3]])
}

#[tokio::test]
async fn websocket_limits_close_with_message_too_big_and_release_payloads() {
    // A declared 9-byte frame exceeds the limit before any payload arrives; two messages
    // overflow a one-message inbox the application does not drain.
    let oversized: &[u8] = b"\x82\x89\x00\x00\x00\x00";
    let flood: &[u8] = b"\x82\x81\x00\x00\x00\x00a\x82\x81\x00\x00\x00\x00a";
    for frames in [oversized, flood] {
        let limits = HttpServerLimits {
            websocket_inbox: 1,
            websocket_frame_bytes: 8,
            close_flush_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let server = server(limits, |exchange| async move {
            exchange
                .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            while !matches!(exchange.receive().await.unwrap(), ExchangeEvent::Closed { .. }) {}
            Ok(())
        });
        let mut raw = HANDSHAKE.to_vec();
        raw.extend_from_slice(frames);
        let (out, end) = exchange(&server, &raw).await;
        assert_eq!(close_code(&out), u16::from(CloseCode::Size));
        assert!(end.finished());
        assert_eq!(server.connections.open(), 0);
    }
}

#[tokio::test]
async fn an_unanswered_or_refused_upgrade_is_an_http_response() {
    let server = default_server(|exchange| async move {
        if exchange.head().path == "/ws" {
            let close = ExchangeMessage::Close { code: 1000, reason: String::new() };
            exchange.send(close).await.unwrap();
        } else {
            exchange.send(start(401, &[("content-length", "4")])).await.unwrap();
            exchange.send(body(b"nope", false)).await.unwrap();
        }
        Ok(())
    });
    let (out, end) = exchange(&server, HANDSHAKE).await;
    assert_eq!(parse(&out).status, 403);
    assert!(end.finished() && end.reset().is_none());

    let denied = String::from_utf8_lossy(HANDSHAKE).replace("/ws", "/denied");
    let (out, end) = exchange(&server, denied.as_bytes()).await;
    let response = parse(&out);
    assert_eq!((response.status, response.body.as_slice()), (401, &b"nope"[..]));
    assert!(end.finished() && end.reset().is_none());
}

#[tokio::test]
async fn an_unanswered_upgrade_is_reset_and_its_application_cancelled() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = dropped.clone();
    let limits = HttpServerLimits { head_timeout: Duration::from_millis(50), ..Default::default() };
    let server = server(limits, move |_| {
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let guard = Dropped(flag.clone());
        async move {
            let _guard = guard;
            std::future::pending().await
        }
    });
    let (out, end) = exchange(&server, HANDSHAKE).await;
    assert!(out.is_empty());
    assert_eq!(end.reset(), Some(Code::HsrTimeout));
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(server.connections.open(), 0);
    assert!(server.websockets.lock().open.is_empty());
}

#[tokio::test]
async fn a_websocket_peer_that_stops_reading_is_dropped() {
    let failed = Arc::new(AtomicBool::new(false));
    let flag = failed.clone();
    let server = server(quick_idle(), move |exchange| {
        let flag = flag.clone();
        async move {
            exchange
                .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
                .await
                .unwrap();
            loop {
                let message = ExchangeMessage::Binary { data: vec![0u8; 64 << 10] };
                if exchange.send(message).await.is_err() {
                    flag.store(true, Ordering::SeqCst);
                    return Ok(());
                }
            }
        }
    });
    let (wire, mut client, end) = wire_sized(1024);
    client.write_all(HANDSHAKE).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), server.serve(wire, Origin::Node(preamble())))
        .await
        .expect("the stall is noticed");
    assert!(failed.load(Ordering::SeqCst));
    assert_eq!(end.reset(), Some(Code::HsrAbort));
    assert_eq!(server.connections.open(), 0);
    drop(client);
}

#[tokio::test]
async fn upgrades_after_drain_are_refused() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed);
    server.drain().await;
    let (out, end) = exchange(&server, HANDSHAKE).await;
    assert!(out.is_empty());
    assert_eq!(end.reset(), Some(Code::HsrRefused));
    let (out, _) = exchange(&echo_size(), &get("/")).await;
    assert_eq!(parse(&out).status, 200, "plain requests still run while draining");
}

#[tokio::test]
async fn close_codes_are_validated_and_reasons_fit_a_close_frame() {
    let server = default_server(|exchange| async move {
        exchange
            .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
            .await
            .unwrap();
        for code in [1005, 1006, 1015, 999, 5000] {
            let close = ExchangeMessage::Close { code, reason: String::new() };
            assert!(exchange.send(close).await.is_err(), "{code} is refused");
        }
        let reason = "\u{e9}".repeat(100);
        exchange.send(ExchangeMessage::Close { code: 4001, reason }).await.unwrap();
        Ok(())
    });
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let request = tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(
        "ws://agents.test/ws",
    )
    .unwrap();
    let (mut socket, _) = tokio_tungstenite::client_async(request, client).await.unwrap();
    let Some(Ok(Message::Close(Some(frame)))) = socket.next().await else {
        panic!("a close frame");
    };
    assert_eq!(u16::from(frame.code), 4001);
    assert_eq!(frame.reason.len(), 122, "123 bytes, cut at a character boundary");
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    assert!(end.finished());
}

#[tokio::test]
async fn stopping_closes_open_websockets_with_going_away() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let server = echo_websocket(closed);
    let (wire, client, end) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, _) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    let stopping = tokio::spawn({
        let server = server.clone();
        async move { server.stop().await }
    });
    let Some(Ok(Message::Close(Some(frame)))) = socket.next().await else {
        panic!("a close frame");
    };
    assert_eq!((frame.code, frame.reason.as_str()), (CloseCode::Away, "worker stopped"));
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
    stopping.await.unwrap();
    assert!(end.finished());
}

fn quick_pings() -> HttpServerLimits {
    HttpServerLimits {
        websocket_ping_interval: Duration::from_millis(20),
        websocket_ping_timeout: Duration::from_millis(20),
        close_flush_timeout: Duration::from_millis(50),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_quiet_peer_is_pinged_and_one_that_answers_stays_connected() {
    let server = server(quick_pings(), |exchange| async move {
        let accept = ExchangeMessage::Accept { subprotocol: Some("chat".into()), headers: vec![] };
        exchange.send(accept).await.unwrap();
        loop {
            match exchange.receive().await.unwrap() {
                ExchangeEvent::Text { text } => {
                    exchange.send(ExchangeMessage::Text { text }).await.unwrap();
                }
                ExchangeEvent::Closed { .. } => return Ok(()),
                _ => {}
            }
        }
    });
    let (wire, client, _) = wire();
    let serving = tokio::spawn({
        let server = server.clone();
        async move { server.serve(wire, Origin::Node(preamble())).await }
    });
    let (mut socket, _) = tokio_tungstenite::client_async(upgrade(), client).await.unwrap();
    // Reading answers each ping with a pong, as every WebSocket client does.
    let mut pings = 0;
    let quiet = tokio::time::sleep(Duration::from_millis(200));
    tokio::pin!(quiet);
    loop {
        tokio::select! {
            biased;
            () = &mut quiet => break,
            message = socket.next() => match message.unwrap().unwrap() {
                Message::Ping(_) => pings += 1,
                other => panic!("unexpected {other:?}"),
            },
        }
    }
    assert!(pings >= 3, "a quiet connection is pinged: {pings}");
    socket.send(Message::text("still here")).await.unwrap();
    loop {
        match socket.next().await.unwrap().unwrap() {
            Message::Text(text) => break assert_eq!(text.as_str(), "still here"),
            Message::Ping(_) => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    socket.close(None).await.unwrap();
    while socket.next().await.is_some() {}
    tokio::time::timeout(Duration::from_secs(5), serving).await.unwrap().unwrap();
}

#[tokio::test]
async fn a_peer_that_answers_no_ping_is_gone() {
    let closed = Arc::new(parking_lot::Mutex::new(None));
    let code = closed.clone();
    let server = server(quick_pings(), move |exchange| {
        let code = code.clone();
        async move {
            exchange
                .send(ExchangeMessage::Accept { subprotocol: None, headers: vec![] })
                .await
                .unwrap();
            loop {
                if let ExchangeEvent::Closed { code: closed } = exchange.receive().await.unwrap() {
                    *code.lock() = Some(closed);
                    return Ok(());
                }
            }
        }
    });
    // The client sends its handshake, then never reads or answers again.
    let (wire, mut client, end) = wire();
    client.write_all(HANDSHAKE).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), server.serve(wire, Origin::Node(preamble())))
        .await
        .expect("an unresponsive peer is noticed");
    assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Abnormal)));
    assert_eq!(end.reset(), Some(Code::HsrAbort));
    assert_eq!(server.connections.open(), 0);
    drop(client);
}

#[tokio::test]
async fn an_accepted_websocket_is_a_101_span_for_the_connections_life() {
    let (server, spans) = traced(HttpServerLimits::default(), |exchange| async move {
        let accept = ExchangeMessage::Accept { subprotocol: None, headers: vec![] };
        exchange.send(accept).await.unwrap();
        while !matches!(exchange.receive().await.unwrap(), ExchangeEvent::Closed { .. }) {}
        Ok(())
    });
    let mut raw = HANDSHAKE.to_vec();
    raw.extend_from_slice(b"\x88\x82\x00\x00\x00\x00\x03\xe8");
    let (out, _) = exchange(&server, &raw).await;
    assert!(out.starts_with(b"HTTP/1.1 101"));
    let recorded = spans.http_requests();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        attr(&recorded[0], "http.response.status_code"),
        Some(&telemetry::record::Value::Int(101))
    );
    assert_eq!(attr(&recorded[0], "lk.endpoint.completion"), None);
}

// Catches: a client gone without a close frame reported to the application as a protocol
// violation (1002) instead of an abnormal closure (1006).
#[tokio::test]
async fn a_client_gone_without_a_close_frame_is_an_abnormal_closure() {
    for partial_frame in [&b""[..], &b"\x81\x85\x00\x00\x00\x00ab"[..]] {
        let closed = Arc::new(parking_lot::Mutex::new(None));
        let code = closed.clone();
        let server = server(HttpServerLimits::default(), move |exchange| {
            let code = code.clone();
            async move {
                let accept = ExchangeMessage::Accept { subprotocol: None, headers: vec![] };
                exchange.send(accept).await.unwrap();
                loop {
                    if let ExchangeEvent::Closed { code: closed } =
                        exchange.receive().await.unwrap()
                    {
                        *code.lock() = Some(closed);
                        return Ok(());
                    }
                }
            }
        });
        let mut raw = HANDSHAKE.to_vec();
        raw.extend_from_slice(partial_frame);
        let (out, _) = exchange(&server, &raw).await;
        assert!(out.starts_with(b"HTTP/1.1 101"));
        assert_eq!(*closed.lock(), Some(u16::from(CloseCode::Abnormal)), "{partial_frame:?}");
    }
}

#[tokio::test]
async fn random_frames_after_an_upgrade_settle_and_free_their_connections() {
    let limits = HttpServerLimits {
        websocket_frame_bytes: 512,
        websocket_message_bytes: 1024,
        websocket_inbox: 2,
        close_flush_timeout: Duration::from_millis(50),
        ..Default::default()
    };
    let server = server(limits, |exchange| async move {
        let accept = ExchangeMessage::Accept { subprotocol: None, headers: vec![] };
        exchange.send(accept).await.map_err(host)?;
        while !matches!(exchange.receive().await.map_err(host)?, ExchangeEvent::Closed { .. }) {}
        Ok(())
    });
    for seed in 0..CASES {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut raw = HANDSHAKE.to_vec();
        for _ in 0..rng.usize(1..8) {
            // A frame header with random flags, opcode, mask and length, then random payload.
            let length = rng.usize(0..2048);
            raw.push(rng.u8(..));
            match length {
                0..=125 => raw.push(0x80 | length as u8),
                126..=0xffff => {
                    raw.push(0x80 | 126);
                    raw.extend_from_slice(&(length as u16).to_be_bytes());
                }
                _ => unreachable!(),
            }
            raw.extend((0..4 + length).map(|_| rng.u8(..)));
        }
        if rng.bool() {
            raw.truncate(rng.usize(HANDSHAKE.len()..raw.len()));
        }
        let (out, end) = serve_in_pieces(&server, &raw, &mut rng).await;
        assert!(out.starts_with(b"HTTP/1.1 101"), "seed {seed}");
        assert!(end.finished() || end.reset().is_some(), "seed {seed}");
        assert_eq!(server.connections.open(), 0, "seed {seed}");
        assert!(server.websockets.lock().open.is_empty(), "seed {seed}");
    }
}
