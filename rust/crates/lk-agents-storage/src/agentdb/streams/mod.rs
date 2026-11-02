//! The answer streams at `/v0/agentstore/streams`: the A2A event frames a turn sent, as JSON
//! records.
//!
//! * `POST /streams` creates one (`AST_..`, a day's TTL); `POST /streams/{id}` appends, and
//!   `Stream-Closed: true` closes.
//! * `GET /streams/{id}?offset=` reads from an offset (`-1` the beginning); `&live=sse` follows
//!   the tail and rotates about every minute, so the reader reconnects from its offset.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt as _};
use lk_agents_abi::{ApiConnectOptions, Clock, DEFAULT_API_CONNECT_OPTIONS};
use lk_agents_net::http::exchange::{Exchange, HttpEvent, ResponsePolicy};
use lk_agents_net::http::{GatewayHttp, HeaderMap, Method};
use lk_agents_net::sse::{SseDecoder, SseEvent};
use lk_agents_net::transport::HttpRequest;

use crate::StorageError;
use crate::streams::{CONTENT_TYPE, DEFAULT_TTL, ResponseStreams, START, StreamInfo, StreamRead};

use super::conn::TokenMinter;

mod cursor;
use cursor::{MAX_BYTES, MAX_RECORDS, Position, invalid};

/// The streams API's answer to a create.
#[derive(serde::Deserialize)]
struct Created {
    stream_id: String,
    expires_at: i64,
}

/// One attempt with net's phase deadlines: storage owns retries, and an append is not idempotent.
const ATTEMPT: ApiConnectOptions =
    ApiConnectOptions { max_retry: 0, ..DEFAULT_API_CONNECT_OPTIONS };

type Body = Pin<Box<dyn Stream<Item = Result<Bytes, StorageError>> + Send>>;

/// The streams client over the process's HTTP transport, shared by every session a worker serves.
#[derive(Clone)]
pub struct Streams {
    http: Arc<GatewayHttp>,
    clock: Arc<dyn Clock>,
    base: String,
    minter: TokenMinter,
}

impl Streams {
    pub fn new(
        http: Arc<GatewayHttp>,
        clock: Arc<dyn Clock>,
        base: impl Into<String>,
        minter: TokenMinter,
    ) -> Self {
        Self { http, clock, base: base.into().trim_end_matches('/').to_owned(), minter }
    }

    fn url(&self, id: &str, query: &[(&str, &str)]) -> Result<String, StorageError> {
        let url = format!("{}/{id}", self.base);
        if query.is_empty() {
            return Ok(url);
        }
        url::Url::parse_with_params(&url, query)
            .map(String::from)
            .map_err(|e| StorageError::Invalid(format!("stream url: {e}")))
    }

    /// A successful answer; any other status is mapped to its typed error. A `live` answer is
    /// listened to passively: only the total deadline bounds the wait for its next event.
    async fn send(
        &self,
        method: Method,
        url: String,
        headers: &[(&str, &str)],
        body: Vec<u8>,
        live: bool,
    ) -> Result<Answer, StorageError> {
        let headers = std::iter::once((
            "authorization".to_owned(),
            format!("Bearer {}", self.minter.mint()?),
        ))
        .chain(headers.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())))
        .collect();
        let mut exchange = Exchange::with_response_policy(
            self.http.clone(),
            method,
            HttpRequest { url, headers, body },
            &ATTEMPT,
            self.clock.clone(),
            ResponsePolicy::Protocol,
        );
        let headers = loop {
            match exchange.next().await.map_err(unavailable)? {
                HttpEvent::Headers(headers) => break headers,
                HttpEvent::Started => {}
                HttpEvent::Bytes(_) | HttpEvent::End => {
                    return Err(malformed("no response headers"));
                }
            }
        };
        exchange.expect_response(!live);
        let status = exchange.status().expect("a response carries its status");
        let body: Body = Box::pin(futures_util::stream::unfold(Some(exchange), |exchange| async {
            let mut exchange = exchange?;
            loop {
                match exchange.next().await {
                    Ok(HttpEvent::Bytes(bytes)) => return Some((Ok(bytes), Some(exchange))),
                    Ok(HttpEvent::End) => return None,
                    Ok(HttpEvent::Started | HttpEvent::Headers(_)) => {}
                    Err(error) => return Some((Err(unavailable(error)), None)),
                }
            }
        }));
        let answer = Answer { headers, body };
        if status.is_success() { Ok(answer) } else { Err(answer.rejected(status.as_u16()).await) }
    }

    /// Reads from `from` (`None`: the beginning). With `live`, follows the tail until the stream
    /// closes; without, ends at the tail.
    pub fn read(&self, stream_id: &str, from: Option<&str>, live: bool) -> StreamReader {
        let decoded = from.map(Position::decode).unwrap_or_else(|| Position::at(START.to_owned()));
        let (position, error) = match decoded {
            Ok(position) => (position, None),
            Err(error) => (Position::at(START.to_owned()).expect("start"), Some(error)),
        };
        let cursor = position.encode();
        let replay = position.consumed;
        StreamReader {
            streams: self.clone(),
            id: stream_id.to_owned(),
            position: Position { anchor: position.anchor, consumed: 0 },
            cursor,
            replay,
            replay_bytes: 0,
            replay_pages: 0,
            error,
            boundary: None,
            live,
            buffered: VecDeque::new(),
            closed: false,
            sse: None,
        }
    }
}

/// A response: its headers, and its body as it arrives.
struct Answer {
    headers: HeaderMap,
    body: Body,
}

impl Answer {
    fn header(&self, name: &str) -> Option<String> {
        self.headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
    }

    fn flag(&self, name: &str) -> bool {
        self.header(name).is_some_and(|v| v.eq_ignore_ascii_case("true"))
    }

    fn required_next(&self) -> Result<String, StorageError> {
        self.header("Stream-Next-Offset").ok_or_else(|| malformed("response omitted next offset"))
    }

    /// The body read to its end, within the response byte budget.
    async fn bytes(mut self) -> Result<Vec<u8>, StorageError> {
        let mut body = Vec::new();
        while let Some(chunk) = self.body.next().await {
            let chunk = chunk?;
            if chunk.len() > MAX_BYTES.saturating_sub(body.len()) {
                return Err(malformed("response exceeds byte budget"));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }

    /// A non-2xx with its body's message: 503 is retryable (`Unavailable`), 404 is `NotFound`.
    async fn rejected(self, status: u16) -> StorageError {
        let message = match self.bytes().await {
            Ok(body) => String::from_utf8_lossy(&body).into_owned(),
            Err(error) => error.to_string(),
        };
        let message = serde_json::from_str::<serde_json::Value>(&message)
            .ok()
            .and_then(|v| {
                v.get("message")
                    .or_else(|| v.get("msg"))
                    .and_then(|m| m.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or(message);
        match status {
            404 => StorageError::NotFound(message),
            401 | 403 => StorageError::Unauthorized(message),
            500..=599 => StorageError::Unavailable(message),
            s => StorageError::Invalid(format!("http {s}: {message}")),
        }
    }
}

/// A read position over one stream.
pub struct StreamReader {
    streams: Streams,
    id: String,
    position: Position,
    cursor: String,
    replay: usize,
    replay_bytes: usize,
    replay_pages: usize,
    error: Option<StorageError>,
    boundary: Option<Boundary>,
    live: bool,
    buffered: VecDeque<Vec<u8>>,
    closed: bool,
    sse: Option<Sse>,
}

struct Boundary {
    next: String,
    up_to_date: bool,
    closed: bool,
}

struct Sse {
    body: Body,
    decoder: SseDecoder,
    events: VecDeque<SseEvent>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Control {
    stream_next_offset: Option<String>,
    error: Option<String>,
    #[serde(default)]
    stream_closed: bool,
    #[serde(default)]
    up_to_date: bool,
}

#[async_trait]
impl ResponseStreams for Streams {
    /// A new stream, `AST_..`.
    async fn create(&self, ttl: Option<Duration>) -> Result<StreamInfo, StorageError> {
        let body = serde_json::json!({
            "content_type": CONTENT_TYPE,
            "ttl_seconds": ttl.unwrap_or(DEFAULT_TTL).as_secs(),
            "region": "",
        });
        let headers = [("content-type", CONTENT_TYPE)];
        let answer =
            self.send(Method::POST, self.base.clone(), &headers, body.to_string().into(), false);
        let created: Created = serde_json::from_slice(&answer.await?.bytes().await?)
            .map_err(|e| StorageError::Unavailable(e.to_string()))?;
        Ok(StreamInfo { stream_id: created.stream_id, expires_at: created.expires_at })
    }

    /// The resume cursor is immediately after the record.
    async fn append(&self, stream_id: &str, record: &[u8]) -> Result<String, StorageError> {
        let headers = [("content-type", CONTENT_TYPE)];
        let url = self.url(stream_id, &[])?;
        let answer = self.send(Method::POST, url, &headers, record.to_vec(), false).await?;
        Ok(Position::at(answer.required_next()?)?.encode())
    }

    /// Its terminal frame marks the commit.
    async fn close(&self, stream_id: &str) -> Result<(), StorageError> {
        let url = self.url(stream_id, &[])?;
        self.send(Method::POST, url, &[("Stream-Closed", "true")], Vec::new(), false).await?;
        Ok(())
    }

    async fn delete(&self, stream_id: &str) -> Result<(), StorageError> {
        let url = self.url(stream_id, &[])?;
        self.send(Method::DELETE, url, &[], Vec::new(), false).await?;
        Ok(())
    }

    fn read(&self, stream_id: &str, after: Option<&str>, live: bool) -> Box<dyn StreamRead> {
        Box::new(Streams::read(self, stream_id, after, live))
    }
}

#[async_trait]
impl StreamRead for StreamReader {
    /// The tail ends a catch-up read, the close a live one. Any failure settles the reader.
    async fn next(&mut self) -> Result<Option<Vec<u8>>, StorageError> {
        let result = self.next_inner().await;
        if result.is_err() {
            self.closed = true;
            self.buffered.clear();
            self.sse = None;
        }
        result
    }

    fn next_offset(&self) -> &str {
        &self.cursor
    }
}

impl StreamReader {
    async fn next_inner(&mut self) -> Result<Option<Vec<u8>>, StorageError> {
        if let Some(error) = self.error.take() {
            self.closed = true;
            return Err(error);
        }
        loop {
            if let Some(r) = self.buffered.pop_front() {
                self.position.consumed += 1;
                if self.position.consumed > MAX_RECORDS {
                    return Err(invalid("record replay count exceeds budget"));
                }
                self.cursor = self.position.encode();
                return Ok(Some(r));
            }
            if let Some(boundary) = self.boundary.take() {
                self.advance(boundary)?;
            }
            if self.closed {
                return Ok(None);
            }
            if self.live {
                self.pump_live().await?;
            } else {
                self.catch_up().await?;
            }
        }
    }

    fn records(&mut self, body: &[u8]) -> Result<(), StorageError> {
        if body.len() > MAX_BYTES {
            return Err(malformed("response exceeds byte budget"));
        }
        if self.replay > 0 {
            self.replay_bytes = self
                .replay_bytes
                .checked_add(body.len())
                .filter(|n| *n <= MAX_BYTES)
                .ok_or_else(|| malformed("replay exceeds byte budget"))?;
        }
        let records = split_records(body)?;
        if records.len() > MAX_RECORDS {
            return Err(malformed("response exceeds record budget"));
        }
        let skip = self.replay.min(records.len());
        self.replay -= skip;
        self.position.consumed += skip;
        self.buffered.extend(records.into_iter().skip(skip));
        Ok(())
    }

    fn advance(&mut self, boundary: Boundary) -> Result<(), StorageError> {
        if self.replay > 0 && (boundary.up_to_date || boundary.closed) {
            return Err(invalid("replay position is beyond retained records"));
        }
        if boundary.next == self.position.anchor && !boundary.up_to_date && !boundary.closed {
            return Err(malformed("upstream offset did not advance"));
        }
        self.position = Position::at(boundary.next)?;
        if self.replay == 0 {
            self.cursor = self.position.encode();
            self.replay_bytes = 0;
            self.replay_pages = 0;
        }
        self.closed = boundary.closed || (!self.live && boundary.up_to_date);
        Ok(())
    }

    /// One bounded `GET`. Its official boundary is installed only after buffered records drain.
    async fn catch_up(&mut self) -> Result<(), StorageError> {
        self.replay_page()?;
        let url = self.streams.url(&self.id, &[("offset", &self.position.anchor)])?;
        let answer = self.streams.send(Method::GET, url, &[], Vec::new(), false).await?;
        let boundary = Boundary {
            next: answer.required_next()?,
            up_to_date: answer.flag("Stream-Up-To-Date"),
            closed: answer.flag("Stream-Closed"),
        };
        self.records(&answer.bytes().await?)?;
        self.boundary = Some(boundary);
        Ok(())
    }

    /// Reads SSE events until a `data` event lands in the buffer, the stream closes, or the
    /// connection rotates (then reconnects from the offset).
    async fn pump_live(&mut self) -> Result<(), StorageError> {
        let result = self.pump_live_inner().await;
        if result.is_err() {
            self.sse = None;
            self.rewind();
        }
        result
    }

    fn rewind(&mut self) {
        self.replay += self.position.consumed;
        self.position.consumed = 0;
    }

    fn replay_page(&mut self) -> Result<(), StorageError> {
        if self.replay > 0 {
            self.replay_pages += 1;
            if self.replay_pages > 128 {
                return Err(malformed("replay exceeds page budget"));
            }
        }
        Ok(())
    }

    async fn pump_live_inner(&mut self) -> Result<(), StorageError> {
        if self.sse.is_none() {
            let query = [("offset", self.position.anchor.as_str()), ("live", "sse")];
            let url = self.streams.url(&self.id, &query)?;
            let answer = self.streams.send(Method::GET, url, &[], Vec::new(), true).await?;
            self.sse = Some(Sse {
                body: answer.body,
                decoder: SseDecoder::new(),
                events: VecDeque::new(),
            });
        }
        loop {
            let sse = self.sse.as_mut().expect("opened above");
            if let Some(event) = sse.events.pop_front() {
                match event.event.as_str() {
                    "data" => {
                        self.records(event.data.as_bytes())?;
                        return Ok(());
                    }
                    "control" => {
                        let c: Control = serde_json::from_str(&event.data)
                            .map_err(|e| StorageError::Unavailable(e.to_string()))?;
                        if let Some(err) = c.error {
                            self.sse = None;
                            return Err(StorageError::Unavailable(err.to_owned()));
                        }
                        self.replay_page()?;
                        self.advance(Boundary {
                            next: c
                                .stream_next_offset
                                .ok_or_else(|| malformed("control omitted next offset"))?,
                            up_to_date: c.up_to_date,
                            closed: c.stream_closed,
                        })?;
                        if self.closed {
                            self.sse = None;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
                continue;
            }
            match sse.body.next().await {
                Some(Ok(chunk)) => {
                    let mut events = Vec::new();
                    sse.decoder
                        .feed(&chunk, &mut events)
                        .map_err(|e| StorageError::Unavailable(e.to_string()))?;
                    sse.events.extend(events);
                }
                Some(Err(e)) => {
                    self.sse = None;
                    return Err(e);
                }
                None => {
                    let finished = sse.decoder.finish();
                    self.sse = None;
                    finished.map_err(|e| StorageError::Unavailable(e.to_string()))?;
                    // No control may have arrived. Replay the official anchor and skip only
                    // records actually yielded before rotation.
                    self.rewind();
                    return Ok(());
                }
            }
        }
    }
}

/// The catch-up body of a JSON stream: one array, one record per element.
fn split_records(body: &[u8]) -> Result<Vec<Vec<u8>>, StorageError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(Vec::new());
    }
    let records: Vec<Box<serde_json::value::RawValue>> = serde_json::from_slice(body)
        .map_err(|e| StorageError::Unavailable(format!("stream body: {e}")))?;
    Ok(records.into_iter().map(|r| r.get().as_bytes().to_vec()).collect())
}

/// An answer the streams service should not have given: refused, not retried.
fn malformed(message: &str) -> StorageError {
    StorageError::Unavailable(format!("stream answer: {message}"))
}

fn unavailable(e: lk_agents_abi::errors::ApiError) -> StorageError {
    StorageError::Unavailable(e.to_string())
}

#[cfg(test)]
#[path = "../streams_tests.rs"]
mod tests;
