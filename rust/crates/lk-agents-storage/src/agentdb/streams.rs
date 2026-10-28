//! The answer streams at `/v0/agentstore/streams`: the A2A event frames a turn sent, as JSON
//! records.
//!
//! * `POST /streams` creates one (`AST_..`, a day's TTL); `POST /streams/{id}` appends, and
//!   `Stream-Closed: true` closes. Appends are queued and sent in order; a failed one is logged.
//! * `GET /streams/{id}?offset=` reads from an offset (`-1` the beginning); `&live=sse` follows
//!   the tail and rotates about every minute, so the reader reconnects from its offset.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt as _;
use tokio::sync::{mpsc, watch};

use crate::StorageError;
use crate::streams::{CONTENT_TYPE, DEFAULT_TTL, START, StreamInfo};

use super::conn::TokenMinter;
use super::http_ok;
use lk_agents_net::sse::{SseDecoder, SseEvent};

mod cursor;
use cursor::{MAX_BYTES, MAX_RECORDS, Position, invalid};

/// The streams API's answer to a create.
#[derive(serde::Deserialize)]
struct Created {
    stream_id: String,
    expires_at: i64,
}

struct Queue {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    /// Appends handed over and not yet answered.
    pending: watch::Sender<usize>,
}

/// The streams client, shared by every session a worker serves.
#[derive(Clone)]
pub struct Streams {
    http: reqwest::Client,
    base: String,
    minter: TokenMinter,
    /// The runtime each stream's append queue runs on.
    runtime: tokio::runtime::Handle,
    queues: Arc<Mutex<HashMap<String, Arc<Queue>>>>,
}

impl Streams {
    pub fn new(
        http: reqwest::Client,
        base: impl Into<String>,
        minter: TokenMinter,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            http,
            base: base.into().trim_end_matches('/').to_owned(),
            minter,
            runtime,
            queues: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn url(&self, id: &str) -> String {
        format!("{}/{id}", self.base)
    }

    fn token(&self) -> Result<String, StorageError> {
        self.minter.mint()
    }

    /// A new stream, `AST_..`; `None` is the default day. `region` is the session's.
    pub async fn create(
        &self,
        ttl: Option<Duration>,
        region: Option<&str>,
    ) -> Result<StreamInfo, StorageError> {
        let body = serde_json::json!({
            "content_type": CONTENT_TYPE,
            "ttl_seconds": ttl.unwrap_or(DEFAULT_TTL).as_secs(),
            "region": region.unwrap_or(""),
        });
        let resp = self
            .http
            .post(&self.base)
            .bearer_auth(self.token()?)
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE)
            .body(body.to_string())
            .send()
            .await
            .map_err(transport)?;
        let resp = http_ok(resp).await?;
        let created: Created =
            resp.json().await.map_err(|e| StorageError::Unavailable(e.to_string()))?;
        Ok(StreamInfo { stream_id: created.stream_id, expires_at: created.expires_at })
    }

    /// Hands one record to the stream's queue; sent in order, failures logged.
    pub fn append(&self, stream_id: &str, record: Vec<u8>) {
        let queue = self.queue(stream_id);
        queue.pending.send_modify(|n| *n += 1);
        if queue.tx.send(record).is_err() {
            queue.pending.send_modify(|n| *n = n.saturating_sub(1));
        }
    }

    fn queue(&self, stream_id: &str) -> Arc<Queue> {
        let mut queues = self.queues.lock().expect("queues");
        if let Some(q) = queues.get(stream_id) {
            return q.clone();
        }
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (pending, _) = watch::channel(0usize);
        let queue = Arc::new(Queue { tx, pending });
        queues.insert(stream_id.to_owned(), queue.clone());
        let this = self.clone();
        let id = stream_id.to_owned();
        let q = queue.clone();
        self.runtime.spawn(async move {
            while let Some(record) = rx.recv().await {
                if let Err(e) = this.append_now(&id, &record).await {
                    log::warn!(target: "livekit.agents", streamId = id.as_str(); "stream append failed: {e}");
                }
                q.pending.send_modify(|n| *n = n.saturating_sub(1));
            }
        });
        queue
    }

    /// Waits until every queued append of the stream was answered.
    pub async fn flush(&self, stream_id: &str) {
        let queue = self.queues.lock().expect("queues").get(stream_id).cloned();
        if let Some(q) = queue {
            let mut rx = q.pending.subscribe();
            let _ = rx.wait_for(|n| *n == 0).await;
        }
    }

    /// One append, awaited: the resume cursor immediately after the record.
    pub async fn append_now(&self, stream_id: &str, record: &[u8]) -> Result<String, StorageError> {
        let resp = self
            .http
            .post(self.url(stream_id))
            .bearer_auth(self.token()?)
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE)
            .body(record.to_vec())
            .send()
            .await
            .map_err(transport)?;
        let resp = http_ok(resp).await?;
        Ok(Position::at(required_next(&resp)?)?.encode())
    }

    /// Flushes the queue, then closes the stream: its terminal frame marks the commit.
    pub async fn close(&self, stream_id: &str) -> Result<(), StorageError> {
        self.flush(stream_id).await;
        self.queues.lock().expect("queues").remove(stream_id);
        let resp = self
            .http
            .post(self.url(stream_id))
            .bearer_auth(self.token()?)
            .header("Stream-Closed", "true")
            .send()
            .await
            .map_err(transport)?;
        http_ok(resp).await.map(|_| ())
    }

    pub async fn delete(&self, stream_id: &str) -> Result<(), StorageError> {
        let resp = self
            .http
            .delete(self.url(stream_id))
            .bearer_auth(self.token()?)
            .send()
            .await
            .map_err(transport)?;
        http_ok(resp).await.map(|_| ())
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
    body: std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>,
    >,
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

impl StreamReader {
    /// The offset to resume from after the records returned so far.
    pub fn next_offset(&self) -> &str {
        &self.cursor
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The next record, or `None` at the end: the tail for a catch-up read, the close for a live one.
    /// Any failure settles the reader.
    pub async fn next(&mut self) -> Result<Option<Vec<u8>>, StorageError> {
        let result = self.next_inner().await;
        if result.is_err() {
            self.closed = true;
            self.buffered.clear();
            self.sse = None;
        }
        result
    }

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
        let resp = self
            .streams
            .http
            .get(self.streams.url(&self.id))
            .query(&[("offset", self.position.anchor.as_str())])
            .bearer_auth(self.streams.token()?)
            .send()
            .await
            .map_err(transport)?;
        let resp = http_ok(resp).await?;
        let boundary = Boundary {
            next: required_next(&resp)?,
            up_to_date: header(&resp, "Stream-Up-To-Date")
                .is_some_and(|v| v.eq_ignore_ascii_case("true")),
            closed: header(&resp, "Stream-Closed").is_some_and(|v| v.eq_ignore_ascii_case("true")),
        };
        let mut body = Vec::new();
        let mut chunks = resp.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(transport)?;
            if chunk.len() > MAX_BYTES.saturating_sub(body.len()) {
                return Err(malformed("response exceeds byte budget"));
            }
            body.extend_from_slice(&chunk);
        }
        self.records(&body)?;
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
            let resp = self
                .streams
                .http
                .get(self.streams.url(&self.id))
                .query(&[("offset", self.position.anchor.as_str()), ("live", "sse")])
                .bearer_auth(self.streams.token()?)
                .send()
                .await
                .map_err(transport)?;
            let resp = http_ok(resp).await?;
            self.sse = Some(Sse {
                body: Box::pin(resp.bytes_stream()),
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
                    return Err(transport(e));
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

fn required_next(resp: &reqwest::Response) -> Result<String, StorageError> {
    header(resp, "Stream-Next-Offset").ok_or_else(|| malformed("response omitted next offset"))
}

fn header(resp: &reqwest::Response, name: &str) -> Option<String> {
    resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

/// An answer the streams service should not have given: refused, not retried.
fn malformed(message: &str) -> StorageError {
    StorageError::Unavailable(format!("stream answer: {message}"))
}

fn transport(e: reqwest::Error) -> StorageError {
    StorageError::Unavailable(e.to_string())
}

#[cfg(test)]
mod framing_tests {
    use super::*;

    #[tokio::test]
    async fn a_live_stream_accepts_utf8_and_crlf_split_across_chunks() {
        super::super::install_tls_provider();
        let streams = Streams::new(
            reqwest::Client::new(),
            "http://unused/streams",
            TokenMinter::new("key", "secret").unwrap(),
            tokio::runtime::Handle::current(),
        );
        let mut reader = streams.read("AST_test", None, true);
        let body = "event: data\r\ndata: [\"café\"]\r\n\r\n".as_bytes();
        let chunks: Vec<Result<bytes::Bytes, reqwest::Error>> =
            body.iter().map(|b| Ok(bytes::Bytes::copy_from_slice(&[*b]))).collect();
        reader.sse = Some(Sse {
            body: Box::pin(futures_util::stream::iter(chunks)),
            decoder: SseDecoder::new(),
            events: VecDeque::new(),
        });
        reader.pump_live().await.unwrap();
        assert_eq!(reader.buffered.pop_front().unwrap(), "\"café\"".as_bytes());
    }
}

#[cfg(test)]
mod tests;
