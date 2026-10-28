//! Bounded HTTP request parsing and projection into an ASGI scope.

use super::{Headers, READ_SIZE, RequestInfo, Scope, StreamIo};

pub(super) enum HeadError {
    Protocol(String),
    Gone,
}

/// Up to and including the blank line, plus whatever came after it.
pub(super) async fn read_head(io: &dyn StreamIo, max: usize) -> Result<Vec<u8>, HeadError> {
    let mut buf = Vec::new();
    loop {
        if let Some(end) = find(&buf, b"\r\n\r\n") {
            if end + 4 > max {
                return Err(HeadError::Protocol("request head exceeds byte limit".into()));
            }
            return Ok(buf);
        }
        if buf.len() >= max {
            return Err(HeadError::Protocol("request head exceeds byte limit".into()));
        }
        let chunk = io.read(READ_SIZE.min(max - buf.len())).await.map_err(|_| HeadError::Gone)?;
        if chunk.is_empty() {
            return Err(HeadError::Protocol("stream ended before the request head".into()));
        }
        buf.extend_from_slice(&chunk);
    }
}

pub(super) fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

pub(super) struct Head {
    pub(super) method: String,
    pub(super) target: Vec<u8>,
    pub(super) http10: bool,
    /// Names lowercased, as h11 iterates them.
    pub(super) headers: Headers,
    /// The bytes after the head.
    pub(super) rest: Vec<u8>,
}

impl Head {
    pub(super) fn values<'a>(&'a self, name: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + 'a {
        self.headers.iter().filter(move |(k, _)| k == name).map(|(_, v)| v.as_slice())
    }

    pub(super) fn tokens(&self, name: &[u8]) -> Vec<Vec<u8>> {
        self.values(name)
            .flat_map(|v| v.split(|b| *b == b','))
            .map(|t| t.trim_ascii().to_ascii_lowercase())
            .collect()
    }

    pub(super) fn is_websocket_upgrade(&self) -> bool {
        self.method == "GET"
            && self.tokens(b"upgrade").iter().any(|t| t == b"websocket")
            && self.tokens(b"connection").iter().any(|t| t == b"upgrade")
    }
}

/// The request head as h11 takes it: HTTP/1.0 or 1.1, a `Host` on 1.1.
pub(super) fn parse_head(buf: &[u8]) -> Result<Head, String> {
    let mut n = 64;
    loop {
        let mut slots = vec![httparse::EMPTY_HEADER; n];
        let mut req = httparse::Request::new(&mut slots);
        match req.parse(buf) {
            Ok(httparse::Status::Complete(len)) => {
                let http10 = req.version == Some(0);
                let headers: Headers = req
                    .headers
                    .iter()
                    .map(|h| (h.name.to_ascii_lowercase().into_bytes(), h.value.to_vec()))
                    .collect();
                if !http10 && !headers.iter().any(|(k, _)| k == b"host") {
                    return Err("Missing mandatory Host: header".into());
                }
                return Ok(Head {
                    method: req.method.unwrap_or_default().to_owned(),
                    target: req.path.unwrap_or_default().as_bytes().to_vec(),
                    http10,
                    headers,
                    rest: buf[len..].to_vec(),
                });
            }
            Ok(httparse::Status::Partial) => return Err("incomplete request head".into()),
            Err(httparse::Error::TooManyHeaders) if n < 8192 => n *= 4,
            Err(e) => return Err(e.to_string()),
        }
    }
}

pub(super) fn scope_base(
    info: &RequestInfo,
    kind: &'static str,
    headers: Headers,
    target: &[u8],
) -> Scope {
    let (raw_path, query) = match target.iter().position(|b| *b == b'?') {
        Some(i) => (&target[..i], &target[i + 1..]),
        None => (target, &[][..]),
    };
    // `urllib.parse.unquote` of the latin-1 text: percent-escapes decode as UTF-8.
    let latin1: String = raw_path.iter().map(|b| *b as char).collect();
    Scope {
        kind,
        method: String::new(),
        scheme: String::new(),
        path: unquote(&latin1),
        raw_path: raw_path.to_vec(),
        query_string: query.to_vec(),
        headers,
        client: (!info.client_addr.is_empty()).then(|| (info.client_addr.clone(), 0)),
        subprotocols: Vec::new(),
    }
}

/// `urllib.parse.unquote` (UTF-8, invalid sequences replaced).
pub(super) fn unquote(s: &str) -> String {
    if !s.contains('%') {
        return s.to_owned();
    }
    let mut out = String::new();
    let mut bytes = Vec::new();
    let mut chars = s.char_indices().peekable();
    let raw = s.as_bytes();
    while let Some((i, c)) = chars.next() {
        let hex = |j: usize| raw.get(j).and_then(|b| (*b as char).to_digit(16));
        if c == '%'
            && let (Some(h), Some(l)) = (hex(i + 1), hex(i + 2))
        {
            bytes.push((h * 16 + l) as u8);
            chars.next();
            chars.next();
            continue;
        }
        if !bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&std::mem::take(&mut bytes)));
        }
        out.push(c);
    }
    if !bytes.is_empty() {
        out.push_str(&String::from_utf8_lossy(&bytes));
    }
    out
}
