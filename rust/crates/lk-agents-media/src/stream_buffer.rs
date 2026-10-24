//! `StreamBuffer` (`utils/codecs/decoder.py:65-142`), public via `utils.codecs.__all__`: a blocking
//! pipe giving ffmpeg's AVIO a blocking `read`. `end_input` drains then EOFs; `close` EOFs at once
//! even with bytes unread, which is how the decode thread tells a teardown from a failure.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// The reference compacts after 5 MB; `VecDeque::drain` reclaims as it goes, so this is unused.
pub const REFERENCE_COMPACT_THRESHOLD: usize = 5 * 1024 * 1024;

#[derive(Default)]
struct Inner {
    data: VecDeque<u8>,
    eof: bool,
    closed: bool,
}

/// A thread-safe buffer that behaves like an IO stream.
///
/// A byte pipe: write from one thread, read from another. The first paragraph is the class's
/// `__doc__`, and the same sentence heads the impl's doc, which the `.pyi` is written from.
#[lk_agents_abi::class]
#[derive(Default)]
pub struct StreamBuffer {
    inner: Mutex<Inner>,
    ready: Condvar,
}

/// A thread-safe buffer that behaves like an IO stream.
///
/// Allows writing from one thread and reading from another. Unused by the decoder; exported
/// because it is in the reference's `utils.codecs.__all__`.
#[lk_agents_abi::host(object)]
impl StreamBuffer {
    #[host(new)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append bytes and wake the reader.
    ///
    /// A write after `end_input` is dropped rather than reordered behind the EOF the reader has
    /// already been promised.
    #[host(method)]
    pub fn write(&self, data: &[u8]) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.closed || g.eof {
            return;
        }
        g.data.extend(data);
        self.ready.notify_all();
    }

    /// Blocking read of at most `buf.len()` bytes. Returns 0 only at EOF or after `close`.
    pub fn read(&self, buf: &mut [u8]) -> usize {
        if buf.is_empty() {
            return 0;
        }
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if g.closed {
                return 0;
            }
            if !g.data.is_empty() {
                let n = buf.len().min(g.data.len());
                for (slot, b) in buf.iter_mut().zip(g.data.drain(..n)) {
                    *slot = b;
                }
                return n;
            }
            if g.eof {
                return 0;
            }
            g = self.ready.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// No more bytes are coming. The reader drains what is buffered, then sees EOF.
    #[host(method)]
    pub fn end_input(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.eof = true;
        self.ready.notify_all();
    }

    /// Stop now. The reader sees EOF immediately, buffered bytes and all.
    #[host(method)]
    pub fn close(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.closed = true;
        g.data.clear();
        self.ready.notify_all();
    }

    /// Bytes written but not yet read.
    #[host(method(name = "__len__"))]
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).data.len()
    }

    /// `read(size=-1)`: blocks (the binding releases the GIL) until data is available, EOF is
    /// signalled, or the buffer is closed. `-1` reads what is available now, not the whole stream.
    #[host(method(name = "read", detach, defaults(size = -1)))]
    pub fn read_up_to(&self, size: isize) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; if size < 0 { self.len().max(1) } else { size as usize }];
        let n = self.read(&mut buf);
        buf.truncate(n);
        buf
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Catches: treating `end_input` as `close` and dropping buffered bytes. The reference's
    // `end_input` is called on every normal completion, so an implementation that discarded the
    // tail would truncate the last ~4 KB of every single decode -- audible as a clipped word at
    // the end of each utterance, and easy to blame on the TTS.
    #[test]
    fn end_input_drains_what_was_written_before_reporting_eof() {
        let b = StreamBuffer::new();
        b.write(b"hello world");
        b.end_input();
        let mut buf = [0u8; 5];
        assert_eq!(b.read(&mut buf), 5);
        assert_eq!(&buf, b"hello");
        let mut rest = [0u8; 32];
        assert_eq!(b.read(&mut rest), 6);
        assert_eq!(&rest[..6], b" world");
        assert_eq!(b.read(&mut rest), 0, "EOF after the buffer is drained");
        assert_eq!(b.read(&mut rest), 0, "and it stays EOF");
    }

    // Catches: `close` draining like `end_input` does. `aclose()` closes mid-decode to tear the
    // stream down; if the reader kept serving buffered bytes, the decode thread would keep
    // emitting frames for an utterance that was already cancelled -- the agent talks over the
    // user after a interruption.
    #[test]
    fn close_stops_immediately_even_with_bytes_still_unread() {
        let b = StreamBuffer::new();
        b.write(b"hello world");
        b.close();
        let mut buf = [0u8; 32];
        assert_eq!(b.read(&mut buf), 0);
        assert!(b.is_empty(), "close must release the buffered bytes, not just hide them");
    }

    // Catches: a non-blocking read. If `read` returned 0 on an empty-but-open buffer, ffmpeg's
    // AVIO would take it for end of stream and `avformat_open_input` would fail on any stream
    // whose first chunk lands after the demuxer's first probe read -- i.e. every network TTS.
    #[test]
    fn a_read_on_an_empty_open_buffer_waits_for_the_writer() {
        let b = Arc::new(StreamBuffer::new());
        let w = b.clone();
        let h = std::thread::spawn(move || {
            let mut buf = [0u8; 4];
            let n = w.read(&mut buf);
            (n, buf)
        });
        // No sleep: the reader either blocks (and this write wakes it) or it already returned 0,
        // which is the failure this test is for. Either way `join` decides it.
        b.write(b"abcd");
        let (n, buf) = h.join().unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf, b"abcd");
    }

    // Catches: a write racing in behind an EOF the reader was already promised. The reference
    // has the same guarantee by construction (`_eof` is checked under the lock); without it a
    // late chunk from a cancelled request can be appended after the demuxer has been told the
    // stream ended, and it surfaces as trailing audio from the *previous* utterance.
    #[test]
    fn a_write_after_end_input_is_dropped_rather_than_appended() {
        let b = StreamBuffer::new();
        b.write(b"ab");
        b.end_input();
        b.write(b"cd");
        let mut buf = [0u8; 32];
        assert_eq!(b.read(&mut buf), 2);
        assert_eq!(&buf[..2], b"ab");
        assert_eq!(b.read(&mut buf), 0);
    }
}
