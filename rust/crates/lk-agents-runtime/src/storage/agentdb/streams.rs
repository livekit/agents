//! The answer streams at `/v0/agentstore/streams`: the A2A event frames a turn sent, as JSON
//! records.
//!
//! * `POST /streams` creates one (`AST_..`, a day's TTL); `POST /streams/{id}` appends, and
//!   `Stream-Closed: true` closes. Appends are queued and sent in order; a failed one is logged.
//! * `GET /streams/{id}?offset=` reads from an offset (`-1` the beginning); `&live=sse` follows
//!   the tail and rotates about every minute, so the reader reconnects from its offset.

use lk_agents_host_derive::Host;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
use std::collections::VecDeque;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt as _;
use tokio::sync::{mpsc, watch};

use crate::conn::TokenMinter;
use crate::{StorageError, http_ok};

pub const DEFAULT_TTL: Duration = Duration::from_secs(86_400);
pub const CONTENT_TYPE: &str = "application/json";
/// The beginning, as the server spells it.
pub const START: &str = "-1";

/// A created stream: its id (also the task id it answers) and when it expires.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct StreamInfo {
    pub stream_id: String,
    pub expires_at: i64,
}

struct Queue {
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    queues: Arc<Mutex<HashMap<String, Arc<Queue>>>>,
}

impl Streams {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(http: reqwest::Client, base: impl Into<String>, minter: TokenMinter) -> Self {
        Self {
            http,
            base: base.into().trim_end_matches('/').to_owned(),
            minter,
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
        resp.json::<StreamInfo>().await.map_err(|e| StorageError::Decode(e.to_string()))
    }

    /// Hands one record to the stream's queue; sent in order, failures logged.
    pub fn append(&self, stream_id: &str, record: Vec<u8>) {
        let queue = self.queue(stream_id);
        queue.pending.send_modify(|n| *n += 1);
        if queue.tx.send(record).is_err() {
            queue.pending.send_modify(|n| *n = n.saturating_sub(1));
        }
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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
        tokio::spawn(async move {
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

    /// One append, awaited: the record's offset token.
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
        Ok(header(&resp, "Stream-Offset").unwrap_or_default())
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
        StreamReader {
            streams: self.clone(),
            id: stream_id.to_owned(),
            from: from.unwrap_or(START).to_owned(),
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
    /// The next offset to ask for.
    from: String,
    live: bool,
    buffered: VecDeque<Vec<u8>>,
    closed: bool,
    sse: Option<Sse>,
}

struct Sse {
    body: std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>,
    >,
    buf: Vec<u8>,
}

impl StreamReader {
    /// The offset to resume from after the records returned so far.
    pub fn next_offset(&self) -> &str {
        &self.from
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The next record, or `None` at the end: the tail for a catch-up read, the close for a live one.
    pub async fn next(&mut self) -> Result<Option<Vec<u8>>, StorageError> {
        loop {
            if let Some(r) = self.buffered.pop_front() {
                return Ok(Some(r));
            }
            if self.closed {
                return Ok(None);
            }
            if self.live {
                self.pump_live().await?;
            } else {
                let up_to_date = self.catch_up().await?;
                if self.buffered.is_empty() && (up_to_date || self.closed) {
                    self.closed = true;
                }
            }
        }
    }

    /// One `GET`: fills the buffer, advances the offset; `true` at the tail.
    async fn catch_up(&mut self) -> Result<bool, StorageError> {
        let resp = self
            .streams
            .http
            .get(self.streams.url(&self.id))
            .query(&[("offset", self.from.as_str())])
            .bearer_auth(self.streams.token()?)
            .send()
            .await
            .map_err(transport)?;
        let resp = http_ok(resp).await?;
        if let Some(next) = header(&resp, "Stream-Next-Offset") {
            self.from = next;
        }
        let up_to_date =
            header(&resp, "Stream-Up-To-Date").is_some_and(|v| v.eq_ignore_ascii_case("true"));
        if header(&resp, "Stream-Closed").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            self.closed = true;
        }
        let body = resp.bytes().await.map_err(transport)?;
        self.buffered.extend(split_records(&body)?);
        Ok(up_to_date)
    }

    /// Reads SSE events until a `data` event lands in the buffer, the stream closes, or the
    /// connection rotates (then reconnects from the offset).
    async fn pump_live(&mut self) -> Result<(), StorageError> {
        if self.sse.is_none() {
            let resp = self
                .streams
                .http
                .get(self.streams.url(&self.id))
                .query(&[("offset", self.from.as_str()), ("live", "sse")])
                .bearer_auth(self.streams.token()?)
                .send()
                .await
                .map_err(transport)?;
            let resp = http_ok(resp).await?;
            self.sse = Some(Sse { body: Box::pin(resp.bytes_stream()), buf: Vec::new() });
        }
        let sse = self.sse.as_mut().expect("opened above");
        loop {
            if let Some((event, data)) = take_event(&mut sse.buf) {
                match event.as_str() {
                    "data" => {
                        self.buffered.extend(split_records(data.as_bytes())?);
                        return Ok(());
                    }
                    "control" => {
                        let c: serde_json::Value = serde_json::from_str(&data)
                            .map_err(|e| StorageError::Decode(e.to_string()))?;
                        if let Some(next) = c["streamNextOffset"].as_str() {
                            self.from = next.to_owned();
                        }
                        if let Some(err) = c["error"].as_str() {
                            self.sse = None;
                            return Err(StorageError::Http { status: 0, message: err.to_owned() });
                        }
                        if c["streamClosed"].as_bool() == Some(true) {
                            self.closed = true;
                            self.sse = None;
                            return Ok(());
                        }
                    }
                    _ => {}
                }
                continue;
            }
            match sse.body.next().await {
                Some(Ok(chunk)) => sse.buf.extend_from_slice(&chunk),
                Some(Err(e)) => {
                    self.sse = None;
                    return Err(transport(e));
                }
                None => {
                    self.sse = None; // rotation: reconnect from the offset
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
        .map_err(|e| StorageError::Decode(format!("stream body: {e}")))?;
    Ok(records.into_iter().map(|r| r.get().as_bytes().to_vec()).collect())
}

/// One complete SSE event off the front of `buf`: `(event, data)`, data lines joined with `\n`.
fn take_event(buf: &mut Vec<u8>) -> Option<(String, String)> {
    let end = buf.windows(2).position(|w| w == b"\n\n")?;
    let block = String::from_utf8_lossy(&buf[..end]).into_owned();
    buf.drain(..end + 2);
    let mut event = String::from("message");
    let mut data = Vec::new();
    for line in block.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            event = v.trim().to_owned();
        } else if let Some(v) = line.strip_prefix("data:") {
            data.push(v.strip_prefix(' ').unwrap_or(v).to_owned());
        }
    }
    Some((event, data.join("\n")))
}

fn header(resp: &reqwest::Response, name: &str) -> Option<String> {
    resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

fn transport(e: reqwest::Error) -> StorageError {
    StorageError::Unavailable(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_events_are_taken_one_at_a_time() {
        let mut buf = b"event: data\ndata: [{\"a\":1},{\"b\":2}]\n\nevent: control\ndata: {\"streamNextOffset\":\"x\",\"upToDate\":true}\n\nev".to_vec();
        let (e, d) = take_event(&mut buf).unwrap();
        assert_eq!(e, "data");
        assert_eq!(
            split_records(d.as_bytes()).unwrap(),
            vec![b"{\"a\":1}".to_vec(), b"{\"b\":2}".to_vec()]
        );
        let (e, d) = take_event(&mut buf).unwrap();
        assert_eq!(e, "control");
        assert!(d.contains("streamNextOffset"));
        assert!(take_event(&mut buf).is_none());
        assert_eq!(buf, b"ev");
    }
}
