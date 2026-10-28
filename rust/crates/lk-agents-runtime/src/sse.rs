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
    /// The last event ID as of this event (it persists across events).
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SseError {
    #[error("the event stream is not UTF-8 at byte {offset}")]
    InvalidUtf8 { offset: u64 },
    #[error("the event stream ended inside an event")]
    Truncated,
}

const DEFAULT_EVENT: &str = "message";

/// Incremental decoder state for one body.
#[derive(Debug, Default)]
pub struct SseDecoder {
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
    last_id: String,
    failed: Option<SseError>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
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
            self.push_text(text, out);
        }
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                self.offset += bytes.len() as u64;
                self.push_text(text, out);
            }
            Err(err) => {
                let (valid, rest) = bytes.split_at(err.valid_up_to());
                let text = std::str::from_utf8(valid).unwrap_or_default();
                self.offset += valid.len() as u64;
                self.push_text(text, out);
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
        if !self.line.is_empty() || !self.data.is_empty() || !self.event.is_empty() {
            return Err(SseError::Truncated);
        }
        Ok(())
    }

    fn fail(&mut self, err: SseError) -> SseError {
        self.failed = Some(err.clone());
        err
    }

    fn push_text(&mut self, mut text: &str, out: &mut Vec<SseEvent>) {
        if text.is_empty() {
            return;
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
            self.line.push_str(&text[..at]);
            let cr = text.as_bytes()[at] == b'\r';
            text = &text[at + 1..];
            if cr {
                match text.strip_prefix('\n') {
                    Some(rest) => text = rest,
                    None => self.after_cr = text.is_empty(),
                }
            }
            let line = std::mem::take(&mut self.line);
            self.process_line(&line, out);
        }
        self.line.push_str(text);
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
            "id" if !value.contains('\0') => value.clone_into(&mut self.last_id),
            // `retry` sets a reconnection delay; no consumer reconnects through it.
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        let event = std::mem::take(&mut self.event);
        if self.data.is_empty() {
            return;
        }
        let mut data = std::mem::take(&mut self.data);
        data.pop();
        out.push(SseEvent {
            event: if event.is_empty() { DEFAULT_EVENT.to_owned() } else { event },
            data,
            id: self.last_id.clone(),
        });
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
                id: "".into(),
            },
            SseEvent { event: "delta".into(), data: " two spaces\n".into(), id: "".into() },
            SseEvent { event: "message".into(), data: "first\nsecond".into(), id: "7".into() },
        ];
        assert_eq!(decode(&[body]), (expected.clone(), Ok(())));
        for at in 0..=body.len() {
            let (head, tail) = body.split_at(at);
            assert_eq!(decode(&[head, tail]), (expected.clone(), Ok(())), "split at byte {at}");
        }
        let bytes: Vec<&[u8]> = body.chunks(1).collect();
        assert_eq!(decode(&bytes), (expected, Ok(())), "one byte per chunk");
    }

    // Catches: decoding each chunk lossily, so a character split across two chunks became two
    // replacement characters (the old `String::from_utf8_lossy` per chunk).
    #[test]
    fn a_character_split_across_chunks_is_decoded_whole() {
        let (events, end) = decode(&[b"data: caf\xc3", b"\xa9\n\n"]);
        assert_eq!((data(&events), end), (vec!["caf\u{e9}"], Ok(())));
    }

    // Catches: emitting each `data:` line as its own event.
    #[test]
    fn data_lines_join_into_one_event_at_the_blank_line() {
        let (events, end) = decode(&[b"data: a\ndata: b\n\n"]);
        assert_eq!((data(&events), end), (vec!["a\nb"], Ok(())));
    }

    // Catches: only LF delimiting lines, so a CR-only stream never produced an event.
    #[test]
    fn a_bare_cr_ends_a_line() {
        let (events, end) = decode(&[b"data: x\r\r"]);
        assert_eq!((data(&events), end), (vec!["x"], Ok(())));
    }

    // Catches: `trim_start` on the value, which strips payload whitespace past the one space the
    // format reserves.
    #[test]
    fn only_the_first_space_after_the_colon_is_stripped() {
        let (events, end) = decode(&[b"data:  x \n\n"]);
        assert_eq!((data(&events), end), (vec![" x "], Ok(())));
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
