use std::io;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{Clock, OwnedTask, SpawnerExt, timeout};
use lk_agents_net::owned::Background;
use parking_lot::Mutex;
use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot};

// Each side writes `MAGIC` and its `VERSION` (big-endian), then frames of a 4-byte big-endian
// length and a postcard message, and last an empty frame: its end, which a named pipe cannot
// signal by half-closing. Reader and writer tasks own the halves, so a caller that stops waiting
// never cuts a frame.

/// The first bytes either side writes.
pub const MAGIC: [u8; 4] = *b"LKAJ";
/// The protocol both sides must speak; a change to a message is a new version.
pub const VERSION: u32 = 6;
/// The largest message: a job's `Job` protobuf is a few kilobytes.
pub const MAX_FRAME: usize = 4 << 20;
/// Frames queued in each direction.
const QUEUE: usize = 16;
/// The preamble's length: [`MAGIC`], then [`VERSION`].
const PREAMBLE: usize = MAGIC.len() + size_of::<u32>();
/// The empty frame that ends a stream.
const END_FRAME: [u8; 4] = [0; 4];
/// How long [`Channel::close`] flushes what is queued, then how long it drains the peer.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(2);

/// Why the channel stopped receiving.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    /// The peer finished or went away between frames.
    #[error("the IPC peer closed the connection")]
    Closed,
    /// The stream ended inside a frame.
    #[error("the IPC connection ended inside a frame")]
    Truncated,
    #[error("a {0}-byte frame is over the {MAX_FRAME}-byte limit")]
    TooLarge(usize),
    #[error("malformed message: {0}")]
    Malformed(String),
    #[error("the IPC peer speaks protocol version {0}, this side {VERSION}")]
    Version(u32),
    #[error("the IPC connection failed: {message}")]
    Io { kind: io::ErrorKind, os_error: Option<i32>, message: String },
}

/// Why a message was not queued.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SendError {
    #[error("the IPC connection is closed")]
    Closed,
    #[error("the IPC peer is not reading")]
    Full,
    /// Nothing was queued; the channel stays usable.
    #[error("a {0}-byte message is over the {MAX_FRAME}-byte limit")]
    TooLarge(usize),
}

/// One side of the connection: it sends `Out` and receives `In`, each in order.
pub struct Channel<Out, In> {
    outbox: Mutex<Option<mpsc::Sender<Vec<u8>>>>,
    inbox: tokio::sync::Mutex<mpsc::Receiver<Result<In, ChannelError>>>,
    writer: Mutex<Option<OwnedTask>>,
    reader: Mutex<Option<OwnedTask>>,
    /// What `close` bounds its flush and drain on.
    clock: Arc<dyn Clock>,
    _out: PhantomData<fn(Out)>,
}

impl<Out: Serialize, In: DeserializeOwned + Send + 'static> Channel<Out, In> {
    /// The channel over a connected stream, its reader and writer run on `rt`.
    pub fn new<S: AsyncRead + AsyncWrite + Send + 'static>(stream: S, rt: &Background) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        let (outbox, queued) = mpsc::channel(QUEUE);
        let (received, inbox) = mpsc::channel(QUEUE);
        let (reading, finished) = oneshot::channel();
        let spawner = &rt.spawner;
        let writer = spawner.spawn(write_frames(writer, queued, finished, received.clone()));
        let reader = spawner.spawn(read_frames(reader, received, reading));
        Self {
            outbox: Mutex::new(Some(outbox)),
            inbox: tokio::sync::Mutex::new(inbox),
            writer: Mutex::new(Some(writer)),
            reader: Mutex::new(Some(reader)),
            clock: rt.clock.clone(),
            _out: PhantomData,
        }
    }

    /// The channel over one end of [`pair`], taking its ownership. Needs a runtime context.
    pub fn from_endpoint(endpoint: i64, rt: &Background) -> io::Result<Self> {
        Ok(Self::new(transport::adopt(endpoint)?, rt))
    }

    /// Queues a message, waiting while the queue is full.
    pub async fn send(&self, msg: Out) -> Result<(), SendError> {
        let frame = frame(&msg)?;
        let outbox = self.outbox.lock().clone().ok_or(SendError::Closed)?;
        outbox.send(frame).await.map_err(|_| SendError::Closed)
    }

    /// Queues a message without waiting.
    pub fn try_send(&self, msg: Out) -> Result<(), SendError> {
        let frame = frame(&msg)?;
        let outbox = self.outbox.lock().clone().ok_or(SendError::Closed)?;
        outbox.try_send(frame).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => SendError::Full,
            mpsc::error::TrySendError::Closed(_) => SendError::Closed,
        })
    }

    /// The next message, or why there is none; after an error, `Closed`. Cancel safe.
    pub async fn recv(&self) -> Result<In, ChannelError> {
        self.inbox.lock().await.recv().await.unwrap_or(Err(ChannelError::Closed))
    }

    /// Stops sending, flushes what is queued and finishes the stream (each within
    /// [`CLOSE_TIMEOUT`]), then discards what the peer still sends until it finishes too, so
    /// that closing with unread data never resets the connection under the peer.
    pub async fn close(&self) {
        self.outbox.lock().take();
        let writer = self.writer.lock().take();
        if let Some(mut writer) = writer
            && timeout(&*self.clock, CLOSE_TIMEOUT, &mut writer).await.is_err()
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = CLOSE_TIMEOUT.as_millis() as u64; "IPC writer close timed out");
            writer.cancel();
            let _ = writer.await;
        }
        if timeout(&*self.clock, CLOSE_TIMEOUT, async {
            let mut inbox = self.inbox.lock().await;
            while inbox.recv().await.is_some() {}
        })
        .await
        .is_err()
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = CLOSE_TIMEOUT.as_millis() as u64; "IPC peer drain timed out");
        }
        let reader = self.reader.lock().take();
        if let Some(reader) = reader {
            reader.cancel();
            let _ = reader.await;
        }
    }
}

/// Two connected channels over a bounded in-memory stream.
pub fn memory<A, B>(rt: &Background) -> (Channel<A, B>, Channel<B, A>)
where
    A: Serialize + DeserializeOwned + Send + 'static,
    B: Serialize + DeserializeOwned + Send + 'static,
{
    let (a, b) = tokio::io::duplex(64 << 10);
    (Channel::new(a, rt), Channel::new(b, rt))
}

/// A new connection's two endpoints, as raw descriptors (Unix) or handles (Windows) the host
/// transfers: the worker keeps the first, the job process gets the second. Neither is inherited
/// by a process the job spawns.
pub fn pair() -> io::Result<(i64, i64)> {
    transport::pair()
}

fn frame<T: Serialize>(msg: &T) -> Result<Vec<u8>, SendError> {
    // Measured before allocating, so an oversized message never costs its size.
    let size = postcard::serialize_with_flavor(msg, postcard::ser_flavors::Size::default())
        .expect("the messages serialize");
    if size > MAX_FRAME {
        return Err(SendError::TooLarge(size));
    }
    let mut frame = Vec::with_capacity(4 + size);
    frame.extend_from_slice(&(size as u32).to_be_bytes());
    Ok(postcard::to_extend(msg, frame).expect("the messages serialize"))
}

fn preamble() -> [u8; PREAMBLE] {
    let mut preamble = [0; PREAMBLE];
    preamble[..MAGIC.len()].copy_from_slice(&MAGIC);
    preamble[MAGIC.len()..].copy_from_slice(&VERSION.to_be_bytes());
    preamble
}

/// Whether the peer's preamble names this protocol.
fn check_peer(peer: &[u8; PREAMBLE]) -> Result<(), ChannelError> {
    let (magic, version) =
        peer.split_last_chunk::<4>().expect("the preamble ends in a four-byte version");
    if magic != MAGIC {
        return Err(ChannelError::Malformed("the IPC peer is not a LiveKit agents process".into()));
    }
    let version = u32::from_be_bytes(*version);
    if version != VERSION {
        return Err(ChannelError::Version(version));
    }
    Ok(())
}

/// The peer finishing or vanishing is `Closed`; a stream ending inside a frame is `Truncated`.
fn failure(error: io::Error, in_frame: bool) -> ChannelError {
    match error.kind() {
        io::ErrorKind::UnexpectedEof if in_frame => ChannelError::Truncated,
        io::ErrorKind::UnexpectedEof
        | io::ErrorKind::ConnectionReset
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::BrokenPipe
        | io::ErrorKind::NotConnected => ChannelError::Closed,
        kind => {
            ChannelError::Io { kind, os_error: error.raw_os_error(), message: error.to_string() }
        }
    }
}

/// Fills `buf`: `Closed` when the stream ends before its first byte.
async fn read_start<R: AsyncRead + Unpin>(r: &mut R, buf: &mut [u8]) -> Result<(), ChannelError> {
    if r.read(&mut buf[..1]).await.map_err(|error| failure(error, false))? == 0 {
        return Err(ChannelError::Closed);
    }
    r.read_exact(&mut buf[1..]).await.map_err(|error| failure(error, true))?;
    Ok(())
}

async fn read_frame<In: DeserializeOwned, R: AsyncRead + Unpin>(
    r: &mut R,
) -> Result<In, ChannelError> {
    let mut len = [0; 4];
    read_start(r, &mut len).await?;
    let len = u32::from_be_bytes(len) as usize;
    if len == 0 {
        return Err(ChannelError::Closed);
    }
    if len > MAX_FRAME {
        return Err(ChannelError::TooLarge(len));
    }
    let mut data = vec![0; len];
    r.read_exact(&mut data).await.map_err(|error| failure(error, true))?;
    match postcard::take_from_bytes(&data) {
        Ok((msg, [])) => Ok(msg),
        Ok(_) => Err(ChannelError::Malformed("trailing bytes after the message".into())),
        Err(error) => Err(ChannelError::Malformed(error.to_string())),
    }
}

/// The peer's preamble, then its messages until the first error; `finished` drops when it ends.
async fn read_frames<In: DeserializeOwned, R: AsyncRead + Unpin>(
    mut r: R,
    received: mpsc::Sender<Result<In, ChannelError>>,
    finished: oneshot::Sender<()>,
) {
    let mut peer = [0; PREAMBLE];
    let started = read_start(&mut r, &mut peer).await.and_then(|()| check_peer(&peer));
    let error = match started {
        Err(error) => error,
        Ok(()) => match forward_frames(&mut r, &received).await {
            Some(error) => error,
            None => return,
        },
    };
    drop(finished);
    let _ = received.send(Err(error)).await;
}

/// The peer's messages to `received` until the first error; `None` once nobody receives them.
async fn forward_frames<In: DeserializeOwned, R: AsyncRead + Unpin>(
    r: &mut R,
    received: &mpsc::Sender<Result<In, ChannelError>>,
) -> Option<ChannelError> {
    loop {
        let msg = match read_frame(r).await {
            Ok(msg) => msg,
            Err(error) => return Some(error),
        };
        received.send(Ok(msg)).await.ok()?;
    }
}

/// The preamble, then the queued frames until the queue closes or the peer finished (then what
/// is already queued); then the end frame, and it finishes the stream. A failure other than the peer going away is
/// reported to the receiving side (the reader reports that one itself).
async fn write_frames<In, W: AsyncWrite + Unpin>(
    mut w: W,
    mut queued: mpsc::Receiver<Vec<u8>>,
    mut peer_finished: oneshot::Receiver<()>,
    received: mpsc::Sender<Result<In, ChannelError>>,
) {
    let Err(error) = write_queued(&mut w, &mut queued, &mut peer_finished).await else { return };
    let error = failure(error, false);
    if !matches!(error, ChannelError::Closed) {
        let _ = received.send(Err(error)).await;
    }
}

async fn write_queued<W: AsyncWrite + Unpin>(
    w: &mut W,
    queued: &mut mpsc::Receiver<Vec<u8>>,
    peer_finished: &mut oneshot::Receiver<()>,
) -> io::Result<()> {
    w.write_all(&preamble()).await?;
    w.flush().await?;
    let mut peer_reading = true;
    while let Some(frame) = next_frame(queued, peer_finished, &mut peer_reading).await {
        w.write_all(&frame).await?;
        w.flush().await?;
    }
    w.write_all(&END_FRAME).await?;
    w.shutdown().await
}

/// While the peer reads, the next queued frame; once it finished, only what is already queued.
async fn next_frame(
    queued: &mut mpsc::Receiver<Vec<u8>>,
    peer_finished: &mut oneshot::Receiver<()>,
    peer_reading: &mut bool,
) -> Option<Vec<u8>> {
    if *peer_reading {
        tokio::select! {
            biased;
            frame = queued.recv() => return frame,
            _ = peer_finished => *peer_reading = false,
        }
    }
    queued.try_recv().ok()
}

// The OS connection, the channel's one socket adapter.
#[cfg(unix)]
#[path = "ipc_transport_unix.rs"]
mod transport;
#[cfg(windows)]
#[path = "ipc_transport_windows.rs"]
mod transport;
