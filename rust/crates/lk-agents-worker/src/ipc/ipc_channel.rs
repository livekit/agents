//! The connection between the worker and one job process: typed messages over one stream, and
//! the only code that touches its bytes.
//!
//! Each side first writes [`MAGIC`] and its [`VERSION`] (big-endian), then frames of a 4-byte
//! big-endian length and a postcard-encoded message of at most [`MAX_FRAME`] bytes, and last an
//! empty frame: its end, which a named pipe cannot signal by half-closing. A reader task
//! and a writer task own the two halves, so a caller that stops waiting never cuts a frame.
//!
//! Transports: a socketpair on Unix, a named pipe on Windows, a bounded `tokio::io::duplex` in
//! tests. All three run the same codec.

use std::io;
use std::marker::PhantomData;
use std::time::Duration;

use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot};

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
    _out: PhantomData<fn(Out)>,
}

impl<Out: Serialize, In: DeserializeOwned + Send + 'static> Channel<Out, In> {
    /// The channel over a connected stream. Needs a runtime context.
    pub fn new<S: AsyncRead + AsyncWrite + Send + 'static>(stream: S) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        let (outbox, queued) = mpsc::channel(QUEUE);
        let (received, inbox) = mpsc::channel(QUEUE);
        let (reading, finished) = oneshot::channel();
        let spawner = TokioSpawner::current();
        let writer = spawner.spawn(write_frames(writer, queued, finished, received.clone()));
        let reader = spawner.spawn(read_frames(reader, received, reading));
        Self {
            outbox: Mutex::new(Some(outbox)),
            inbox: tokio::sync::Mutex::new(inbox),
            writer: Mutex::new(Some(writer)),
            reader: Mutex::new(Some(reader)),
            _out: PhantomData,
        }
    }

    /// The channel over one end of [`pair`], taking its ownership. Needs a runtime context.
    pub fn from_endpoint(endpoint: i64) -> io::Result<Self> {
        Ok(Self::new(transport::adopt(endpoint)?))
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
            && tokio::time::timeout(CLOSE_TIMEOUT, &mut writer).await.is_err()
        {
            writer.cancel();
            let _ = writer.await;
        }
        let _ = tokio::time::timeout(CLOSE_TIMEOUT, async {
            let mut inbox = self.inbox.lock().await;
            while inbox.recv().await.is_some() {}
        })
        .await;
        let reader = self.reader.lock().take();
        if let Some(reader) = reader {
            reader.cancel();
            let _ = reader.await;
        }
    }
}

/// Two connected channels over a bounded in-memory stream.
pub fn memory<A, B>() -> (Channel<A, B>, Channel<B, A>)
where
    A: Serialize + DeserializeOwned + Send + 'static,
    B: Serialize + DeserializeOwned + Send + 'static,
{
    let (a, b) = tokio::io::duplex(64 << 10);
    (Channel::new(a), Channel::new(b))
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
    let (magic, version) = peer.split_at(MAGIC.len());
    if magic != MAGIC {
        return Err(ChannelError::Malformed("the IPC peer is not a LiveKit agents process".into()));
    }
    if version != VERSION.to_be_bytes() {
        return Err(ChannelError::Version(u32::from_be_bytes(version.try_into().unwrap())));
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

#[cfg(unix)]
mod transport {
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
    use std::os::unix::net::UnixStream;

    pub fn pair() -> io::Result<(i64, i64)> {
        // std creates both ends close-on-exec.
        let (parent, child) = UnixStream::pair()?;
        for end in [&parent, &child] {
            no_sigpipe(end)?;
        }
        Ok((parent.into_raw_fd().into(), child.into_raw_fd().into()))
    }

    pub fn adopt(endpoint: i64) -> io::Result<tokio::net::UnixStream> {
        let fd = i32::try_from(endpoint)
            .ok()
            .filter(|fd| *fd >= 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "bad IPC endpoint"))?;
        // SAFETY: the caller hands the descriptor's ownership over.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // The job process received its end inheritable (multiprocessing's pass_fds or
        // SCM_RIGHTS): a process it spawns must not keep the connection open.
        // SAFETY: plain flag update on an owned descriptor.
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(io::Error::last_os_error());
        }
        let stream = UnixStream::from(fd);
        stream.set_nonblocking(true)?;
        tokio::net::UnixStream::from_std(stream)
    }

    /// macOS has no `MSG_NOSIGNAL`: without this, a write to a dead peer raises SIGPIPE, which
    /// kills a process that does not ignore it. Linux std writes already pass `MSG_NOSIGNAL`.
    fn no_sigpipe(_end: &UnixStream) -> io::Result<()> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let on: libc::c_int = 1;
            // SAFETY: a valid socket and an int option of the declared size.
            let rc = unsafe {
                libc::setsockopt(
                    _end.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_NOSIGPIPE,
                    (&on as *const libc::c_int).cast(),
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                )
            };
            if rc != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

/// Unverified locally (no Windows host); CI runs the channel tests on Windows.
#[cfg(windows)]
mod transport {
    use std::io;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, OwnedHandle};

    use tokio::net::windows::named_pipe::NamedPipeClient;
    use windows_sys::Win32::Foundation::{
        GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, LocalFree,
    };
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
        TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAG_OVERLAPPED, OPEN_EXISTING,
        PIPE_ACCESS_DUPLEX,
    };
    use windows_sys::Win32::System::Pipes::{
        CreateNamedPipeW, GetNamedPipeClientProcessId, PIPE_READMODE_BYTE,
        PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, PIPE_WAIT,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    fn check(ok: i32) -> io::Result<()> {
        if ok == 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }

    /// A security descriptor that grants the current user alone access, freed on drop.
    struct UserOnly(PSECURITY_DESCRIPTOR);

    impl UserOnly {
        fn new() -> io::Result<Self> {
            let mut token: HANDLE = std::ptr::null_mut();
            // SAFETY: the current process pseudo-handle and a writable token slot.
            check(unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) })?;
            // SAFETY: OpenProcessToken returned a new handle we own.
            let token = unsafe { OwnedHandle::from_raw_handle(token) };
            let mut len = 0;
            // SAFETY: a size query: no buffer, a writable length.
            unsafe {
                GetTokenInformation(
                    token.as_raw_handle(),
                    TokenUser,
                    std::ptr::null_mut(),
                    0,
                    &mut len,
                )
            };
            let mut user = vec![0u64; (len as usize).div_ceil(8)];
            // SAFETY: an 8-byte aligned buffer of at least `len` bytes.
            check(unsafe {
                GetTokenInformation(
                    token.as_raw_handle(),
                    TokenUser,
                    user.as_mut_ptr().cast(),
                    len,
                    &mut len,
                )
            })?;
            // SAFETY: the buffer now holds a TOKEN_USER whose SID points into it.
            let sid = unsafe { (*user.as_ptr().cast::<TOKEN_USER>()).User.Sid };
            let mut text = std::ptr::null_mut();
            // SAFETY: a valid SID; the string is LocalAlloc'd and freed below.
            check(unsafe { ConvertSidToStringSidW(sid, &mut text) })?;
            // SAFETY: a NUL-terminated UTF-16 string from ConvertSidToStringSidW.
            let sid = String::from_utf16_lossy(unsafe {
                std::slice::from_raw_parts(text, (0..).take_while(|&i| *text.add(i) != 0).count())
            });
            // SAFETY: allocated by ConvertSidToStringSidW.
            unsafe { LocalFree(text.cast()) };
            // Protected DACL: generic all to this user, nobody else.
            let sddl: Vec<u16> = format!("D:P(A;;GA;;;{sid})").encode_utf16().chain([0]).collect();
            let mut descriptor = std::ptr::null_mut();
            // SAFETY: a NUL-terminated SDDL string; the descriptor is LocalAlloc'd, freed on drop.
            check(unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    sddl.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    std::ptr::null_mut(),
                )
            })?;
            Ok(Self(descriptor))
        }
    }

    impl Drop for UserOnly {
        fn drop(&mut self) {
            // SAFETY: allocated by ConvertStringSecurityDescriptorToSecurityDescriptorW.
            unsafe { LocalFree(self.0) };
        }
    }

    /// A single-instance pipe with an unguessable name, only the current user may open, that
    /// this process connects to itself at once: the connected client must be this process, or
    /// the pair is refused (a process that raced the connect took the one instance).
    pub fn pair() -> io::Result<(i64, i64)> {
        let mut random = [0u8; 16];
        getrandom::fill(&mut random).map_err(|error| io::Error::other(error.to_string()))?;
        let random: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        let name = format!(r"\\.\pipe\livekit-agents-{}-{random}", std::process::id());
        let name: Vec<u16> = name.encode_utf16().chain([0]).collect();
        let user_only = UserOnly::new()?;
        let security = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: user_only.0,
            bInheritHandle: 0,
        };
        // SAFETY: a NUL-terminated name, a valid security descriptor; one instance.
        let server = unsafe {
            CreateNamedPipeW(
                name.as_ptr(),
                PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                1,
                64 << 10,
                64 << 10,
                0,
                &security,
            )
        };
        if server == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: CreateNamedPipeW returned a new handle we own.
        let server = unsafe { OwnedHandle::from_raw_handle(server) };
        // SAFETY: a NUL-terminated name; no sharing; a handle not inherited by children.
        let client = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            )
        };
        if client == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: CreateFileW returned a new handle we own.
        let client = unsafe { OwnedHandle::from_raw_handle(client) };
        let mut pid = 0;
        // SAFETY: a valid server handle and a writable u32.
        check(unsafe { GetNamedPipeClientProcessId(server.as_raw_handle(), &mut pid) })?;
        if pid != std::process::id() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "another process connected to the IPC pipe",
            ));
        }
        Ok((server.into_raw_handle() as i64, client.into_raw_handle() as i64))
    }

    /// Either end: both are overlapped pipe handles, read and written the same way.
    pub fn adopt(endpoint: i64) -> io::Result<NamedPipeClient> {
        if endpoint <= 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "bad IPC endpoint"));
        }
        // SAFETY: the caller hands the handle's ownership over.
        unsafe { NamedPipeClient::from_raw_handle(endpoint as _) }
    }
}
