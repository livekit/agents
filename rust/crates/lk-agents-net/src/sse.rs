//! Server-sent events framing over a response body as it arrives: bytes in, events out. Network
//! chunks are not events: a chunk may end inside a UTF-8 sequence, a CRLF pair or a field, and an
//! event is complete only at its blank line (WHATWG HTML, "Parsing an event stream").
//!
//! The decoder is the one SSE reader of the runtime; it fails explicitly instead of decoding
//! lossily or dropping a half-received event.

/// One dispatched event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, `message` when the event named none.
    pub event: String,
    /// Every `data:` line of the event, joined with `\n`.
    pub data: String,
    /// The last event ID, if the server supplied one on this stream (including an empty reset).
    pub id: Option<String>,
    /// A server-supplied reconnect delay, in milliseconds.
    pub retry: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SseError {
    #[error("the event stream is not UTF-8 at byte {offset}")]
    InvalidUtf8 { offset: u64 },
    #[error("the event stream ended inside an event")]
    Truncated,
    #[error("an event exceeds the {limit} byte assembly limit")]
    TooLarge { limit: usize },
}

const DEFAULT_EVENT: &str = "message";
/// Match the runtime's largest accepted protocol frame, including unfinished fields.
pub const DEFAULT_MAX_EVENT_BYTES: usize = 16 << 20;

/// Incremental decoder state for one body.
#[derive(Debug)]
pub struct SseDecoder {
    limit: usize,
    event_bytes: usize,
    /// A UTF-8 sequence the last chunk ended inside.
    partial: Vec<u8>,
    /// Body offset of the next byte not yet decoded (the first of `partial`).
    offset: u64,
    /// The last chunk ended in CR: an LF opening the next chunk belongs to that delimiter.
    after_cr: bool,
    /// The leading byte-order mark is checked once, on the first decoded text.
    started: bool,
    line: String,
    event: String,
    data: String,
    last_id: Option<String>,
    failed: Option<SseError>,
    controls: bool,
    id_changed: bool,
    retry: Option<u64>,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::with_max_event_bytes(DEFAULT_MAX_EVENT_BYTES)
    }
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_event_bytes(limit: usize) -> Self {
        Self {
            limit,
            event_bytes: 0,
            partial: Vec::new(),
            offset: 0,
            after_cr: false,
            started: false,
            line: String::new(),
            event: String::new(),
            data: String::new(),
            last_id: None,
            failed: None,
            controls: false,
            id_changed: false,
            retry: None,
        }
    }

    /// Include ID/retry-only frames for protocols that use SSE reconnect instructions.
    pub fn with_control_events(mut self) -> Self {
        self.controls = true;
        self
    }

    /// Decodes `bytes`, appending each event completed by them to `out`. After an error the
    /// decoder stays failed: the rest of the body cannot be framed reliably.
    pub fn feed(&mut self, mut bytes: &[u8], out: &mut Vec<SseEvent>) -> Result<(), SseError> {
        if let Some(err) = &self.failed {
            return Err(err.clone());
        }
        if !self.partial.is_empty() {
            let full = utf8_len(self.partial[0]);
            let take = (full - self.partial.len()).min(bytes.len());
            self.partial.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.partial.len() < full {
                return Ok(());
            }
            let partial = std::mem::take(&mut self.partial);
            let text = std::str::from_utf8(&partial)
                .map_err(|_| self.fail(SseError::InvalidUtf8 { offset: self.offset }))?;
            self.offset += partial.len() as u64;
            self.push_text(text, out)?;
        }
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                self.offset += bytes.len() as u64;
                self.push_text(text, out)?;
            }
            Err(err) => {
                let (valid, rest) = bytes.split_at(err.valid_up_to());
                let text = std::str::from_utf8(valid).unwrap_or_default();
                self.offset += valid.len() as u64;
                self.push_text(text, out)?;
                if err.error_len().is_some() {
                    return Err(self.fail(SseError::InvalidUtf8 { offset: self.offset }));
                }
                self.partial.extend_from_slice(rest);
            }
        }
        Ok(())
    }

    /// The body ended: clean only between events.
    pub fn finish(&self) -> Result<(), SseError> {
        if let Some(err) = &self.failed {
            return Err(err.clone());
        }
        if !self.partial.is_empty() {
            return Err(SseError::InvalidUtf8 { offset: self.offset });
        }
        if !self.line.is_empty()
            || !self.data.is_empty()
            || !self.event.is_empty()
            || (self.controls && (self.id_changed || self.retry.is_some()))
        {
            return Err(SseError::Truncated);
        }
        Ok(())
    }

    fn fail(&mut self, err: SseError) -> SseError {
        self.failed = Some(err.clone());
        err
    }

    fn push_text(&mut self, mut text: &str, out: &mut Vec<SseEvent>) -> Result<(), SseError> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.started {
            self.started = true;
            text = text.strip_prefix('\u{feff}').unwrap_or(text);
        }
        if self.after_cr {
            self.after_cr = false;
            text = text.strip_prefix('\n').unwrap_or(text);
        }
        while let Some(at) = text.find(['\r', '\n']) {
            self.append_line(&text[..at])?;
            let cr = text.as_bytes()[at] == b'\r';
            text = &text[at + 1..];
            if cr {
                match text.strip_prefix('\n') {
                    Some(rest) => text = rest,
                    None => self.after_cr = text.is_empty(),
                }
            }
            let line = std::mem::take(&mut self.line);
            if line.is_empty() {
                self.event_bytes = 0;
            } else {
                self.event_bytes += line.len();
            }
            self.process_line(&line, out);
        }
        self.append_line(text)?;
        Ok(())
    }

    fn append_line(&mut self, text: &str) -> Result<(), SseError> {
        if text.len() > self.limit.saturating_sub(self.event_bytes + self.line.len()) {
            return Err(self.fail(SseError::TooLarge { limit: self.limit }));
        }
        self.line.push_str(text);
        Ok(())
    }

    fn process_line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            return self.dispatch(out);
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => value.clone_into(&mut self.event),
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
            }
            "id" if !value.contains('\0') => {
                self.last_id = Some(value.to_owned());
                self.id_changed = true;
            }
            "retry" if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) => {
                if let Ok(value) = value.parse() {
                    self.retry = Some(value);
                }
            }
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        let event = std::mem::take(&mut self.event);
        let control = std::mem::take(&mut self.id_changed);
        let retry = self.retry.take();
        if self.data.is_empty() && !(self.controls && (control || retry.is_some())) {
            return;
        }
        let mut data = std::mem::take(&mut self.data);
        data.pop();
        out.push(SseEvent {
            event: if event.is_empty() { DEFAULT_EVENT.to_owned() } else { event },
            data,
            id: self.last_id.clone(),
            retry,
        });
    }
}

/// A demand-driven SSE response. Dropping it releases its HTTP exchange synchronously.
pub struct EventStream {
    exchange: crate::http::exchange::Exchange,
    decoder: SseDecoder,
    pending: std::collections::VecDeque<Result<SseEvent, lk_agents_abi::errors::ApiError>>,
    limit: usize,
    ended: bool,
}

impl EventStream {
    pub fn new(exchange: crate::http::exchange::Exchange, limit: usize) -> Self {
        Self {
            exchange,
            decoder: SseDecoder::with_max_event_bytes(limit).with_control_events(),
            pending: Default::default(),
            limit,
            ended: false,
        }
    }

    pub fn expect_response(&mut self, expected: bool) {
        self.exchange.expect_response(expected);
    }

    pub async fn next(&mut self) -> Result<Option<SseEvent>, lk_agents_abi::errors::ApiError> {
        use crate::http::exchange::HttpEvent;
        use lk_agents_abi::errors::ApiError;

        loop {
            if self.ended {
                return Ok(None);
            }
            if let Err(error) = self.exchange.wait_for_consumer(std::future::ready(())).await {
                self.ended = true;
                return Err(error);
            }
            if let Some(event) = self.pending.pop_front() {
                match event {
                    Ok(event) => {
                        self.exchange.delivered();
                        return Ok(Some(event));
                    }
                    Err(error) => {
                        self.exchange.cancel();
                        self.ended = true;
                        return Err(error);
                    }
                }
            }
            match self.exchange.next().await {
                Ok(HttpEvent::Started) => {
                    self.decoder =
                        SseDecoder::with_max_event_bytes(self.limit).with_control_events();
                }
                Ok(HttpEvent::Headers(_)) => {}
                Ok(HttpEvent::Bytes(bytes)) => {
                    let mut events = Vec::new();
                    let result = self.decoder.feed(&bytes, &mut events);
                    self.pending.extend(events.into_iter().map(Ok));
                    if let Err(error) = result {
                        self.pending
                            .push_back(Err(ApiError::new(error.to_string()).with_retryable(false)));
                    }
                }
                Ok(HttpEvent::End) => {
                    self.ended = true;
                    return self
                        .decoder
                        .finish()
                        .map(|()| None)
                        .map_err(|error| ApiError::new(error.to_string()).with_retryable(false));
                }
                Err(error) => {
                    self.ended = true;
                    return Err(error);
                }
            }
        }
    }

    pub fn into_stream(
        self,
    ) -> futures_util::stream::BoxStream<'static, Result<SseEvent, lk_agents_abi::errors::ApiError>>
    {
        use futures_util::StreamExt;
        futures_util::stream::unfold(self, |mut stream| async move {
            match stream.next().await {
                Ok(Some(event)) => Some((Ok(event), stream)),
                Ok(None) => None,
                Err(error) => Some((Err(error), stream)),
            }
        })
        .boxed()
    }
}

/// The length of the UTF-8 sequence a lead byte opens; only reached with a lead byte the standard
/// library reported as an incomplete (valid) prefix.
fn utf8_len(lead: u8) -> usize {
    match lead {
        0xF0..=0xF7 => 4,
        0xE0..=0xEF => 3,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(chunks: &[&[u8]]) -> (Vec<SseEvent>, Result<(), SseError>) {
        let mut decoder = SseDecoder::new();
        let mut out = Vec::new();
        for chunk in chunks {
            if let Err(err) = decoder.feed(chunk, &mut out) {
                return (out, Err(err));
            }
        }
        let end = decoder.finish();
        (out, end)
    }

    fn data(events: &[SseEvent]) -> Vec<&str> {
        events.iter().map(|e| e.data.as_str()).collect()
    }

    /// The same body cut at every byte boundary decodes to the same events: chunking is never
    /// framing. Covers split UTF-8 (2-, 3- and 4-byte), CRLF split across chunks, bare CR, LF,
    /// multi-line data, a field split mid-name and whitespace that is payload.
    #[test]
    fn every_split_of_a_body_decodes_to_the_same_events() {
        let body = "\u{feff}: keep-alive\r\n\
                    data: caf\u{e9} \u{20ac}\u{1f600}\r\n\r\n\
                    event: delta\rdata:  two spaces\rdata:\r\r\
                    id: 7\ndata: first\ndata: second\n\n"
            .as_bytes();
        let expected = vec![
            SseEvent {
                event: "message".into(),
                data: "caf\u{e9} \u{20ac}\u{1f600}".into(),
                id: None,
                retry: None,
            },
            SseEvent { event: "delta".into(), data: " two spaces\n".into(), id: None, retry: None },
            SseEvent {
                event: "message".into(),
                data: "first\nsecond".into(),
                id: Some("7".into()),
                retry: None,
            },
        ];
        assert_eq!(decode(&[body]), (expected.clone(), Ok(())));
        for at in 0..=body.len() {
            let (head, tail) = body.split_at(at);
            assert_eq!(decode(&[head, tail]), (expected.clone(), Ok(())), "split at byte {at}");
        }
        let bytes: Vec<&[u8]> = body.chunks(1).collect();
        assert_eq!(decode(&bytes), (expected, Ok(())), "one byte per chunk");
    }

    #[test]
    fn reconnect_frames_preserve_server_delays_and_explicit_cursor_resets() {
        let body = b"data: first\n\nretry: 25\nid: resume-1\n\ndata: second\n\nid:\n\n";
        for at in 0..=body.len() {
            let mut decoder = SseDecoder::new().with_control_events();
            let mut events = Vec::new();
            decoder.feed(&body[..at], &mut events).unwrap();
            decoder.feed(&body[at..], &mut events).unwrap();
            decoder.finish().unwrap();
            assert_eq!(
                events
                    .iter()
                    .map(|event| (event.data.as_str(), event.id.as_deref(), event.retry))
                    .collect::<Vec<_>>(),
                [
                    ("first", None, None),
                    ("", Some("resume-1"), Some(25)),
                    ("second", Some("resume-1"), None),
                    ("", Some(""), None),
                ]
            );
        }
    }

    // Catches: a body ending inside an event reported as a clean end (the half-received event
    // silently dropped).
    #[test]
    fn a_body_ending_inside_an_event_is_truncated() {
        assert_eq!(decode(&[b"data: {\"id\":"]).1, Err(SseError::Truncated));
        assert_eq!(decode(&[b"data: x\n"]).1, Err(SseError::Truncated));
        let (events, end) = decode(&[b"data: x\n\ndata: y"]);
        assert_eq!((data(&events), end), (vec!["x"], Err(SseError::Truncated)));
    }

    // Catches: replacing invalid bytes instead of failing, or a stream ending inside a character
    // reported clean.
    #[test]
    fn invalid_utf8_fails_with_its_offset_and_stays_failed() {
        let mut decoder = SseDecoder::new();
        let mut out = Vec::new();
        assert_eq!(
            decoder.feed(b"data: ok\n\ndata: \xff\n\n", &mut out),
            Err(SseError::InvalidUtf8 { offset: 16 })
        );
        assert_eq!(data(&out), vec!["ok"]);
        assert_eq!(
            decoder.feed(b"data: later\n\n", &mut out),
            Err(SseError::InvalidUtf8 { offset: 16 })
        );
        assert_eq!(decode(&[b"data: \xe2\x82"]).1, Err(SseError::InvalidUtf8 { offset: 6 }));
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn an_unterminated_event_cannot_grow_past_the_body_budget() {
        let mut decoder = SseDecoder::new();
        let mut out = Vec::new();
        let event = vec![b'x'; (16 << 20) + 1];
        assert!(decoder.feed(&event, &mut out).is_err());
        assert!(out.is_empty());
    }
}
