//! The job channel over its real transport (and the in-memory one): framing, limits, how a
//! connection ends, and a stress run. Allocation checks record only their own thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Duration;

use lk_agents_worker::ipc::ipc_channel::{
    Channel, ChannelError, MAGIC, MAX_FRAME, SendError, VERSION, memory, pair,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};

type Message = (u64, Vec<u8>);
type Side = Channel<Message, Message>;

struct RecordingAllocator;
thread_local! {
    static RECORD: Cell<Option<usize>> = const { Cell::new(None) };
}

// SAFETY: every allocation and deallocation is forwarded unchanged to the system allocator.
// The thread-local counter records sizes only and never retains or accesses an allocation.
unsafe impl GlobalAlloc for RecordingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        RECORD.with(|record| {
            if let Some(max) = record.get() {
                record.set(Some(max.max(layout.size())));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        RECORD.with(|record| {
            if let Some(max) = record.get() {
                record.set(Some(max.max(size)));
            }
        });
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: RecordingAllocator = RecordingAllocator;

/// Two channels over the platform's real transport.
fn real() -> (Side, Side) {
    let (a, b) = pair().unwrap();
    (Channel::from_endpoint(a).unwrap(), Channel::from_endpoint(b).unwrap())
}

/// A channel and the raw stream at its peer, past the channel's preamble.
async fn raw() -> (Side, DuplexStream) {
    let (ours, mut peer) = tokio::io::duplex(1 << 16);
    let channel = Side::new(ours);
    peer.read_exact(&mut [0; 8]).await.unwrap();
    (channel, peer)
}

fn preamble(version: u32) -> Vec<u8> {
    [MAGIC.as_slice(), &version.to_be_bytes()].concat()
}

fn frame(msg: &Message) -> Vec<u8> {
    let data = postcard::to_allocvec(msg).unwrap();
    [(data.len() as u32).to_be_bytes().as_slice(), &data].concat()
}

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future).await.expect("timed out")
}

// Catches: an oversized message allocated at its size, written partially, or ending the channel.
#[tokio::test]
async fn an_oversized_send_is_refused_and_the_channel_stays_usable() {
    let (a, b) = real();
    let big = (1, vec![0; MAX_FRAME]);
    RECORD.with(|record| record.set(Some(0)));
    let refused = a.send(big).await;
    let max_allocation = RECORD.with(|record| record.replace(None).unwrap());
    assert!(matches!(refused, Err(SendError::TooLarge(n)) if n > MAX_FRAME));
    assert!(max_allocation < MAX_FRAME, "the refused message allocated {max_allocation} bytes");
    assert!(matches!(a.try_send((1, vec![0; MAX_FRAME])), Err(SendError::TooLarge(_))));
    a.send((2, b"next".to_vec())).await.unwrap();
    assert_eq!(within(b.recv()).await.unwrap(), (2, b"next".to_vec()));
}

// Catches: a peer's length prefix trusted past the limit (an allocation the size of a bad header).
#[tokio::test]
async fn an_oversized_prefix_is_refused_before_reading_its_payload() {
    let (channel, mut peer) = raw().await;
    peer.write_all(&preamble(VERSION)).await.unwrap();
    RECORD.with(|record| record.set(Some(0)));
    peer.write_all(&u32::MAX.to_be_bytes()).await.unwrap();
    let received = within(channel.recv()).await;
    let max_allocation = RECORD.with(|record| record.replace(None).unwrap());
    assert!(matches!(received, Err(ChannelError::TooLarge(n)) if n == u32::MAX as usize));
    assert!(max_allocation < MAX_FRAME, "the advertised frame allocated {max_allocation} bytes");
}

// Catches: a stream ending inside the preamble or a frame taken as a clean close, or the reverse,
// and a fragmented frame misread.
#[tokio::test]
async fn the_stream_ending_mid_frame_is_truncated_and_between_frames_closed() {
    let msg = (1 << 40, b"payload".to_vec());
    let bytes = [preamble(VERSION), frame(&msg)].concat();
    for cut in 0..=bytes.len() {
        let (channel, mut peer) = raw().await;
        for part in bytes[..cut].chunks(3) {
            peer.write_all(part).await.unwrap();
            tokio::task::yield_now().await;
        }
        drop(peer);
        let received = within(channel.recv()).await;
        match cut {
            0 | 8 => assert!(matches!(received, Err(ChannelError::Closed)), "cut={cut}"),
            n if n == bytes.len() => {
                assert_eq!(received.unwrap(), msg);
                assert!(matches!(within(channel.recv()).await, Err(ChannelError::Closed)));
            }
            _ => assert!(matches!(received, Err(ChannelError::Truncated)), "cut={cut}"),
        }
    }
}

// Catches: a peer of another version, or no peer of ours at all, failing as a malformed message
// instead of saying which version it speaks.
#[tokio::test]
async fn a_version_mismatch_is_named_before_any_message() {
    let (channel, mut peer) = raw().await;
    peer.write_all(&preamble(VERSION + 1)).await.unwrap();
    assert!(
        matches!(within(channel.recv()).await, Err(ChannelError::Version(v)) if v == VERSION + 1)
    );

    let (channel, mut peer) = raw().await;
    peer.write_all(b"GET / HTTP/1.1\r\n").await.unwrap();
    assert!(matches!(within(channel.recv()).await, Err(ChannelError::Malformed(_))));

    let (channel, mut peer) = raw().await;
    peer.write_all(&preamble(VERSION)).await.unwrap();
    peer.write_all(&[0, 0, 0, 1, 0xff]).await.unwrap();
    assert!(matches!(within(channel.recv()).await, Err(ChannelError::Malformed(_))));
}

// Catches: a peer that exits with our messages unread read as a transport failure (Linux reports
// ECONNRESET once its last message is read): it is a close, after that last message.
#[tokio::test]
async fn a_peer_exiting_with_unread_messages_is_a_close() {
    let (a, b) = real();
    for i in 0..200 {
        let _ = a.try_send((i, vec![0; 1024]));
    }
    b.send((7, b"exiting".to_vec())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(b);
    assert_eq!(within(a.recv()).await.unwrap(), (7, b"exiting".to_vec()));
    assert!(matches!(within(a.recv()).await, Err(ChannelError::Closed)));
}

// Catches: a close that drops what is queued, resets the peer's unread data, or waits on a peer
// that already finished.
#[tokio::test]
async fn close_flushes_then_drains_the_peer_to_its_end() {
    let (a, b) = real();
    for i in 0..100 {
        let _ = b.try_send((i, vec![1; 1024]));
    }
    let received = tokio::spawn(async move {
        let mut received = Vec::new();
        loop {
            match b.recv().await {
                Ok((i, _)) => received.push(i),
                Err(error) => return (received, error),
            }
        }
    });
    for i in 0..50 {
        a.send((i, vec![2; 4096])).await.unwrap();
    }
    let started = std::time::Instant::now();
    within(a.close()).await;
    assert!(started.elapsed() < Duration::from_secs(1), "the peer finished when we did");
    let (received, end) = within(received).await.unwrap();
    assert_eq!(received, (0..50).collect::<Vec<_>>());
    assert!(matches!(end, ChannelError::Closed), "{end}");
    assert!(matches!(a.send((0, Vec::new())).await, Err(SendError::Closed)));
}

// Catches: a close that waits for EOF, which a Windows named pipe never gives while its peer
// process lives: the end frame ends the stream on every transport.
#[tokio::test]
async fn the_end_frame_ends_the_stream_without_eof() {
    let (channel, mut peer) = raw().await;
    peer.write_all(&preamble(VERSION)).await.unwrap();
    peer.write_all(&[0; 4]).await.unwrap();
    assert!(matches!(within(channel.recv()).await, Err(ChannelError::Closed)));
    let started = std::time::Instant::now();
    within(channel.close()).await;
    assert!(started.elapsed() < Duration::from_secs(1), "the peer already ended");
    let mut end = [1; 4];
    peer.read_exact(&mut end).await.unwrap();
    assert_eq!(end, [0; 4], "this side ends with its own end frame");
    drop(peer);
}

// Catches: the job process's end inherited by what the job spawns, which keeps the connection
// open after the job process dies.
#[cfg(unix)]
#[tokio::test]
async fn a_process_the_job_spawns_does_not_inherit_the_channel() {
    use std::os::fd::RawFd;
    let (parent, child) = pair().unwrap();
    // multiprocessing hands the job process its end inheritable.
    // SAFETY: a plain flag update on a descriptor this test owns.
    assert_eq!(unsafe { libc::fcntl(child as RawFd, libc::F_SETFD, 0) }, 0);
    let parent = Side::from_endpoint(parent).unwrap();
    let child = Side::from_endpoint(child).unwrap();
    let mut grandchild = std::process::Command::new("sleep").arg("30").spawn().unwrap();
    drop(child);
    let received = tokio::time::timeout(Duration::from_secs(5), parent.recv()).await;
    let _ = grandchild.kill();
    let _ = grandchild.wait();
    assert!(matches!(received, Ok(Err(ChannelError::Closed))), "{received:?}");
}

/// A deterministic xorshift, so a failing run replays.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Mostly small, sometimes up to the frame limit.
    fn size(&mut self) -> usize {
        match self.next() % 64 {
            0 => (self.next() as usize) % (MAX_FRAME - 16),
            1..=8 => (self.next() as usize) % (64 << 10),
            _ => (self.next() as usize) % 256,
        }
    }
}

/// Sends `count` messages of random sizes while the peer sends back as many, then one side
/// closes while the other may still be sending: each receives an in-order prefix of what was
/// sent (whole, unchanged), then a clean close, never a truncation or a transport failure.
async fn stress(a: Side, b: Side, count: u64, seed: u64) {
    let (a, b) = (std::sync::Arc::new(a), std::sync::Arc::new(b));
    let send = |side: std::sync::Arc<Side>, seed: u64| async move {
        let mut random = Random(seed);
        for i in 0..count {
            let size = random.size();
            if side.send((i, vec![(i % 251) as u8; size])).await.is_err() {
                return i;
            }
        }
        count
    };
    let recv = |side: std::sync::Arc<Side>, seed: u64| async move {
        let mut random = Random(seed);
        let mut received = 0;
        loop {
            match side.recv().await {
                Ok((i, data)) => {
                    assert_eq!(i, received, "in order");
                    assert_eq!(data.len(), random.size(), "whole");
                    assert!(data.iter().all(|byte| *byte == (i % 251) as u8), "unchanged");
                    received += 1;
                }
                Err(ChannelError::Closed) => return received,
                Err(error) => panic!("message {received}: {error}"),
            }
        }
    };
    let a_sends = tokio::spawn(send(a.clone(), seed));
    let b_sends = tokio::spawn(send(b.clone(), seed + 1));
    let b_receives = tokio::spawn(recv(b.clone(), seed));
    let a_receives = tokio::spawn(recv(a.clone(), seed + 1));
    let a_sent = within(a_sends).await.unwrap();
    assert_eq!(a_sent, count);
    // `a` closes while `b` may still be sending; `b` finishes once it read `a`'s end.
    within(a.close()).await;
    assert_eq!(within(b_receives).await.unwrap(), count, "the close flushed every message");
    let b_sent = within(b_sends).await.unwrap();
    let a_received = within(a_receives).await.unwrap();
    assert!(a_received <= b_sent, "{a_received} received of {b_sent} sent");
    within(b.close()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "stress (~10 s in debug): CI runs it with --include-ignored"]
async fn stress_both_directions_over_the_real_transport() {
    for seed in 1..=4 {
        let (a, b) = real();
        stress(a, b, 300, seed * 7919).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "stress (~10 s in debug): CI runs it with --include-ignored"]
async fn stress_both_directions_in_memory() {
    for seed in 1..=4 {
        let (a, b) = memory();
        stress(a, b, 300, seed * 104_729).await;
    }
}
