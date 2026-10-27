//! The job processes: the worker and each job process over one socket.
//!
//! - [`wire`]: the protocol, a typed message per direction in length-prefixed frames, versioned
//!   by its first exchange.
//! - [`ProcSupervisor`] ([`supervisor`]): the worker's side of one job process, its
//!   initialization, pings, memory, shutdown and kill, decided sans I/O.
//! - [`IpcClient`]: the job process's side, with the self health check. The child answers pings
//!   from its own event loop, so a pong still measures that loop's responsiveness.
//! - [`pool`]: the warm processes. Spawning them stays with the host language.

use std::sync::Mutex;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;

pub mod pool;
#[cfg(feature = "python")]
pub mod python;
pub mod supervisor;
#[cfg(test)]
mod tests;
pub mod wire;

pub use supervisor::{
    Extra, ExtraValue, InitError, Level, MemoryError, ProcHost, ProcSupervisor, SupervisorOptions,
};
#[cfg(feature = "python")]
pub(crate) use supervisor::secs;
pub use wire::{Hello, PROTOCOL_VERSION, StartJob, ToChild, ToParent, WireError};

use lk_agents_abi::lock;

/// The child's grace on top of the parent's `ping_timeout` before it calls itself orphaned.
pub const HEALTH_CHECK_GRACE: f64 = 10.0;
/// How long a closing client writes what is still queued for the parent.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);
/// The bound of the client's queues.
const QUEUE: usize = 64;

/// The socket is closed.
#[derive(Debug, thiserror::Error)]
#[error("duplex closed")]
pub struct Closed;

/// The job process's side of the socket: [`connect`](Self::connect) reads the parent's `Hello`,
/// then the parent's messages are [`recv`](Self::recv)ed and the child's [`send`](Self::send)
/// queued, in order. With a health timeout, it prints to stderr and stops reading when the parent
/// sends nothing for that long, whatever the state of the child's event loop.
pub struct IpcClient {
    outbox: Mutex<Option<mpsc::Sender<ToParent>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
    inbox: tokio::sync::Mutex<mpsc::Receiver<ToChild>>,
    reader: Mutex<Option<JoinHandle<()>>>,
}

impl IpcClient {
    /// The parent's `Hello`, then the client. On a version this side does not speak, it answers
    /// `Ready` with the error and fails. Needs a runtime context.
    pub async fn connect<S>(stream: S) -> Result<(Self, Hello), WireError>
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
    {
        let (mut reader, mut writer) = tokio::io::split(stream);
        let hello = match wire::read::<ToChild, _>(&mut reader).await? {
            ToChild::Hello(hello) => hello,
            other => return Err(WireError::Unexpected(format!("{other:?}"))),
        };
        if hello.version != PROTOCOL_VERSION {
            let error = WireError::Version { theirs: hello.version };
            let ready = ToParent::Ready { version: PROTOCOL_VERSION, error: Some(error.to_string()) };
            let _ = wire::write(&mut writer, &ready).await;
            return Err(error);
        }
        let (outbox, mut queued) = mpsc::channel::<ToParent>(QUEUE);
        let writer = tokio::spawn(async move {
            while let Some(msg) = queued.recv().await {
                if wire::write(&mut writer, &msg).await.is_err() {
                    return;
                }
            }
            let _ = writer.shutdown().await;
        });
        let (inbox_tx, inbox) = mpsc::channel(QUEUE);
        let health_timeout = hello.health_timeout_ms.map(Duration::from_millis);
        let reader = tokio::spawn(read_parent(reader, inbox_tx, health_timeout));
        let client = Self {
            outbox: Mutex::new(Some(outbox)),
            writer: Mutex::new(Some(writer)),
            inbox: tokio::sync::Mutex::new(inbox),
            reader: Mutex::new(Some(reader)),
        };
        Ok((client, hello))
    }

    /// Queues a message for the parent, waiting while the queue is full.
    pub async fn send(&self, msg: ToParent) -> Result<(), Closed> {
        let outbox = lock(&self.outbox).clone().ok_or(Closed)?;
        outbox.send(msg).await.map_err(|_| Closed)
    }

    /// The parent's next message; `None` once it is gone.
    pub async fn recv(&self) -> Option<ToChild> {
        self.inbox.lock().await.recv().await
    }

    /// Stops reading, writes what is queued (for up to [`FLUSH_TIMEOUT`]) and closes the socket.
    pub async fn close(&self) {
        if let Some(reader) = lock(&self.reader).take() {
            reader.abort();
        }
        lock(&self.outbox).take();
        let writer = lock(&self.writer).take();
        if let Some(mut writer) = writer
            && tokio::time::timeout(FLUSH_TIMEOUT, &mut writer).await.is_err()
        {
            writer.abort();
        }
    }
}

/// The parent's messages, until it closes the socket, sends what cannot be read (logged), or goes
/// quiet past `health_timeout`.
async fn read_parent<R: AsyncRead + Unpin>(
    mut reader: R,
    inbox: mpsc::Sender<ToChild>,
    health_timeout: Option<Duration>,
) {
    let far = Duration::from_secs(1 << 30);
    loop {
        let deadline = Instant::now() + health_timeout.map_or(far, |t| t.min(far));
        tokio::select! {
            msg = wire::read::<ToChild, _>(&mut reader) => match msg {
                Ok(msg) => {
                    if inbox.send(msg).await.is_err() {
                        return;
                    }
                }
                Err(WireError::Io(_)) => return,
                Err(e) => {
                    log::error!(target: "livekit.agents", "invalid IPC message from the worker: {e}");
                    return;
                }
            },
            () = tokio::time::sleep_until(deadline), if health_timeout.is_some() => {
                eprintln!("worker process is not responding.. worker crashed?");
                return;
            }
        }
    }
}
