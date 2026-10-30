//! `StreamBuffer` (`utils/codecs/decoder.py:65-142`), public via `utils.codecs.__all__`: a blocking
//! pipe giving a decode job a blocking `read`. `end_input` drains then EOFs; `close` EOFs at once
//! even with bytes unread, which is how the decode job tells a teardown from a failure.

use std::collections::VecDeque;
use std::thread::Thread;

use lk_agents_net::lock::Mutex;

/// Writes are kept whole, so a reader can take them as the writer made them.
#[derive(Default)]
struct Inner {
    chunks: VecDeque<Vec<u8>>,
    /// Bytes of the front chunk already read.
    offset: usize,
    len: usize,
    eof: bool,
    closed: bool,
    /// Readers parked until a write or the end of input.
    readers: Vec<Thread>,
}

/// A thread-safe buffer that behaves like an IO stream.
///
/// A byte pipe: write from one thread, read from another. The first paragraph is the class's
/// `__doc__`, and the same sentence heads the impl's doc, which the `.pyi` is written from.
#[derive(Default)]
pub struct StreamBuffer {
    inner: Mutex<Inner>,
}

/// A thread-safe buffer that behaves like an IO stream.
///
/// Allows writing from one thread and reading from another. Unused by the decoder; exported
/// because it is in the reference's `utils.codecs.__all__`.
impl StreamBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append bytes and wake the reader.
    ///
    /// A write after `end_input` is dropped rather than reordered behind the EOF the reader has
    /// already been promised.
    pub fn write(&self, data: &[u8]) {
        let mut g = self.inner.lock();
        if g.closed || g.eof || data.is_empty() {
            return;
        }
        g.len += data.len();
        g.chunks.push_back(data.to_vec());
        g.wake_readers();
    }

    /// Blocking read of at most `buf.len()` bytes. Returns 0 only at EOF or after `close`.
    pub fn read(&self, buf: &mut [u8]) -> usize {
        if buf.is_empty() {
            return 0;
        }
        self.when_readable(|g| {
            let Some(front) = g.chunks.front() else { return 0 };
            let start = g.offset;
            let n = buf.len().min(front.len() - start);
            buf[..n].copy_from_slice(&front[start..start + n]);
            g.consume(n);
            n
        })
    }

    /// Blocking read of the rest of the oldest write, as written; `None` at EOF or after `close`.
    pub fn read_chunk(&self) -> Option<Vec<u8>> {
        self.when_readable(|g| {
            let mut chunk = g.chunks.pop_front()?;
            let read = std::mem::take(&mut g.offset);
            g.len -= chunk.len() - read;
            chunk.drain(..read);
            Some(chunk)
        })
    }

    /// Runs `read` once there is data, or none will come (no chunks then).
    fn when_readable<T>(&self, read: impl FnOnce(&mut Inner) -> T) -> T {
        loop {
            {
                let mut g = self.inner.lock();
                if g.closed || g.eof || !g.chunks.is_empty() {
                    return read(&mut g);
                }
                let me = std::thread::current();
                if !g.readers.iter().any(|r| r.id() == me.id()) {
                    g.readers.push(me);
                }
            }
            // An unpark between the unlock and here makes this return at once.
            std::thread::park();
        }
    }

    /// No more bytes are coming. The reader drains what is buffered, then sees EOF.
    pub fn end_input(&self) {
        let mut g = self.inner.lock();
        g.eof = true;
        g.wake_readers();
    }

    /// Stop now. The reader sees EOF immediately, buffered bytes and all.
    pub fn close(&self) {
        let mut g = self.inner.lock();
        g.closed = true;
        g.chunks.clear();
        g.offset = 0;
        g.len = 0;
        g.wake_readers();
    }

    /// Bytes written but not yet read.
    pub fn len(&self) -> usize {
        self.inner.lock().len
    }

    /// `read(size=-1)`: blocks (the binding releases the GIL) until data is available, EOF is
    /// signalled, or the buffer is closed. `-1` reads what is available now, not the whole stream.
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

impl Inner {
    fn wake_readers(&mut self) {
        for reader in self.readers.drain(..) {
            reader.unpark();
        }
    }

    fn consume(&mut self, n: usize) {
        self.offset += n;
        self.len -= n;
        if self.chunks.front().is_some_and(|front| front.len() == self.offset) {
            self.chunks.pop_front();
            self.offset = 0;
        }
    }
}

impl lk_agents_abi::media::codecs::StreamBuffer for StreamBuffer {
    fn new() -> Self {
        Self::new()
    }
    fn write(&self, data: &[u8]) {
        self.write(data);
    }
    fn end_input(&self) {
        self.end_input();
    }
    fn close(&self) {
        self.close();
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn read(&self, size: isize) -> Vec<u8> {
        self.read_up_to(size)
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
