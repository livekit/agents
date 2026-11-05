//! Seeded random and mutated input against the endpoint server: whatever the bytes and however
//! they are split, an exchange settles, never panics, frees its connection, and either answers or
//! resets. A failing case prints its seed.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::http::{ExchangeError, ExchangeEvent, ExchangeMessage, HttpExchange};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::exchange_tests::{
    FakeEnd, HANDSHAKE, body, default_server, parse, preamble, server, start, wire,
};
use super::{EndpointServer, Exchange, HttpServerLimits, Origin};

const CASES: u64 = 300;

fn host(error: ExchangeError) -> HostError {
    HostError::new(error.to_string())
}

/// Read the whole body and echo it back in random pieces; a disconnect ends quietly.
async fn echo(exchange: Arc<Exchange>) -> Result<(), HostError> {
    let mut all = Vec::new();
    loop {
        match exchange.receive().await.map_err(host)? {
            ExchangeEvent::Body { data, more } => {
                all.extend_from_slice(&data);
                if !more {
                    break;
                }
            }
            ExchangeEvent::Text { .. } | ExchangeEvent::Binary { .. } => {}
            ExchangeEvent::Disconnected | ExchangeEvent::Closed { .. } => return Ok(()),
        }
    }
    exchange.send(start(200, &[])).await.map_err(host)?;
    let mut rng = fastrand::Rng::with_seed(all.len() as u64);
    let mut rest = all.as_slice();
    while !rest.is_empty() {
        let (piece, tail) = rest.split_at(rng.usize(1..=rest.len()));
        exchange.send(body(piece, true)).await.map_err(host)?;
        rest = tail;
    }
    exchange.send(body(b"", false)).await.map_err(host)
}

/// A valid request with a random body, framing, header case and trailers; and its body.
fn valid_request(rng: &mut fastrand::Rng) -> (Vec<u8>, Vec<u8>) {
    let body: Vec<u8> = (0..rng.usize(0..20_000)).map(|_| rng.u8(..)).collect();
    let host = if rng.bool() { "Host" } else { "hOsT" };
    let mut raw = format!("POST /echo?n={} HTTP/1.1\r\n{host}: agents.test\r\n", rng.u32(..));
    if rng.bool() {
        raw.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        let mut raw = raw.into_bytes();
        raw.extend_from_slice(&body);
        return (raw, body);
    }
    raw.push_str("Transfer-Encoding: chunked\r\n\r\n");
    let mut raw = raw.into_bytes();
    let mut rest = body.as_slice();
    while !rest.is_empty() {
        let (chunk, tail) = rest.split_at(rng.usize(1..=rest.len().min(4096)));
        let extension = if rng.bool() { ";name=value" } else { "" };
        raw.extend_from_slice(format!("{:x}{extension}\r\n", chunk.len()).as_bytes());
        raw.extend_from_slice(chunk);
        raw.extend_from_slice(b"\r\n");
        rest = tail;
    }
    raw.extend_from_slice(b"0\r\n");
    if rng.bool() {
        raw.extend_from_slice(b"x-trailer: done\r\n");
    }
    raw.extend_from_slice(b"\r\n");
    (raw, body)
}

/// Write `raw` in random pieces with yields between them, half-close, and read the response.
async fn serve_in_pieces(
    server: &EndpointServer,
    raw: &[u8],
    rng: &mut fastrand::Rng,
) -> (Vec<u8>, Arc<FakeEnd>) {
    let (wire, client, end) = wire();
    let (mut reader, mut writer) = tokio::io::split(client);
    let mut pieces = Vec::new();
    let mut rest = raw;
    while !rest.is_empty() {
        let (piece, tail) = rest.split_at(rng.usize(1..=rest.len().min(8192)));
        pieces.push(piece.to_vec());
        rest = tail;
    }
    let write = async move {
        for piece in pieces {
            if writer.write_all(&piece).await.is_err() {
                return;
            }
            tokio::task::yield_now().await;
        }
        let _ = writer.shutdown().await;
    };
    let read = async move {
        let mut out = Vec::new();
        let _ = reader.read_to_end(&mut out).await;
        out
    };
    let serve = server.serve(wire, Origin::Node(preamble()));
    let ((), (), out) = tokio::time::timeout(
        Duration::from_secs(5),
        futures_util::future::join3(serve, write, read),
    )
    .await
    .expect("the exchange settles");
    (out, end)
}

#[tokio::test]
async fn valid_requests_round_trip_however_they_are_framed_and_split() {
    let server = default_server(echo);
    for seed in 0..CASES {
        let mut rng = fastrand::Rng::with_seed(seed);
        let (raw, body) = valid_request(&mut rng);
        let (out, end) = serve_in_pieces(&server, &raw, &mut rng).await;
        let response = parse(&out);
        assert_eq!(response.status, 200, "seed {seed}");
        assert!(response.body == body, "seed {seed}: the echoed body differs");
        assert!(response.trailers.is_empty(), "seed {seed}");
        assert!(end.finished() && end.reset().is_none(), "seed {seed}");
        assert_eq!(server.connections.open(), 0, "seed {seed}");
    }
}

/// Flip, insert, delete, duplicate or cut bytes.
fn mutate(raw: &mut Vec<u8>, rng: &mut fastrand::Rng) {
    for _ in 0..rng.usize(1..=4) {
        if raw.is_empty() {
            raw.push(rng.u8(..));
        }
        let at = rng.usize(..raw.len());
        match rng.u8(..5) {
            0 => raw[at] = rng.u8(..),
            1 => raw.insert(at, *b"\r\n:;0 \t\x00\xff".get(rng.usize(..9)).unwrap()),
            2 => {
                let end = (at + rng.usize(1..64)).min(raw.len());
                raw.drain(at..end);
            }
            3 => {
                let end = (at + rng.usize(1..64)).min(raw.len());
                let copy = raw[at..end].to_vec();
                raw.splice(at..at, copy);
            }
            _ => raw.truncate(at),
        }
    }
}

#[tokio::test]
async fn mutated_requests_settle_and_free_their_connections() {
    let limits = HttpServerLimits {
        request_head_bytes: 4096,
        body_metadata_bytes: 256,
        head_timeout: Duration::from_millis(200),
        ..Default::default()
    };
    let server = server(limits, echo);
    let mut answered = 0;
    for seed in 0..CASES {
        let mut rng = fastrand::Rng::with_seed(seed);
        let (mut raw, _) = valid_request(&mut rng);
        mutate(&mut raw, &mut rng);
        let (out, end) = serve_in_pieces(&server, &raw, &mut rng).await;
        assert_eq!(server.connections.open(), 0, "seed {seed}");
        if end.reset().is_none() {
            assert!(end.finished(), "seed {seed}: neither answered nor reset");
            assert!(out.starts_with(b"HTTP/1.1 "), "seed {seed}: {:?}", &out[..out.len().min(64)]);
            answered += 1;
        }
    }
    assert!(answered > 0 && answered < CASES, "mutations both pass and fail: {answered}");
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
