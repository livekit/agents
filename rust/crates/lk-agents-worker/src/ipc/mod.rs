//! The job processes: the worker and each job process over one socket.
//!
//! - [`wire`]: the protocol, a typed message per direction in length-prefixed frames, versioned
//!   by its first exchange.
//! - [`ProcSupervisor`] ([`supervisor`]): the worker's side of one job process, its
//!   initialization, pings, memory, shutdown and kill, decided sans I/O.
//! - [`IpcClient`]: the job process's side, with the self health check. The child answers pings
//!   from its own event loop, so a pong still measures that loop's responsiveness.
//! - [`pool`]: the warm processes. Spawning them stays with the host language.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::Mutex;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;

pub mod pool;
mod process;
#[cfg(feature = "python")]
pub mod python;
pub mod supervisor;
mod supervisor_driver;
#[cfg(test)]
mod tests;
pub mod wire;

#[cfg(feature = "python")]
pub(crate) use supervisor::secs;
pub use supervisor::{
    Extra, ExtraValue, InitError, Level, MemoryError, ProcHost, SupervisorOptions,
};
pub use supervisor_driver::ProcSupervisor;
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    outbox: Mutex<Option<mpsc::Sender<ToParent>>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    writer: Mutex<Option<JoinHandle<()>>>,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    inbox: tokio::sync::Mutex<mpsc::Receiver<ToChild>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    reader: Mutex<Option<JoinHandle<()>>>,
}

impl IpcClient {
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The parent's `Hello`, then the client. On a version this side does not speak, it answers
    /// `Ready` with the error and fails. Needs a runtime context.
    pub async fn connect<S>(stream: S) -> Result<(Self, Hello), WireError>
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
    {
        Self::connect_with_health_action(stream, || {}).await
    }

    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    /// A process bootstrap supplies native orphan termination; no host loop is needed.
    pub async fn connect_with_health_action<S, F>(
        stream: S,
        on_orphan: F,
    ) -> Result<(Self, Hello), WireError>
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
        F: FnOnce() + Send + 'static,
    {
        let (mut reader, mut writer) = tokio::io::split(stream);
        let hello = match wire::read::<ToChild, _>(&mut reader).await? {
            ToChild::Hello(hello) => hello,
            other => return Err(WireError::Unexpected(format!("{other:?}"))),
        };
        if hello.version != PROTOCOL_VERSION {
            let error = WireError::Version { theirs: hello.version };
            let ready =
                ToParent::Ready { version: PROTOCOL_VERSION, error: Some(error.to_string()) };
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
        let reader = tokio::spawn(read_parent(reader, inbox_tx, health_timeout, on_orphan));
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// Stops reading, writes what is queued (for up to [`FLUSH_TIMEOUT`]) and closes the socket.
    pub async fn close(&self) {
        let reader = lock(&self.reader).take();
        if let Some(reader) = reader {
            reader.abort();
            let _ = reader.await;
        }
        lock(&self.outbox).take();
        let writer = lock(&self.writer).take();
        if let Some(mut writer) = writer
            && tokio::time::timeout(FLUSH_TIMEOUT, &mut writer).await.is_err()
        {
            writer.abort();
            let _ = writer.await;
        }
    }
}

#[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// The parent's messages, until it closes the socket, sends what cannot be read (logged), or goes
/// quiet past `health_timeout`.
async fn read_parent<R: AsyncRead + Unpin, F: FnOnce()>(
    mut reader: R,
    inbox: mpsc::Sender<ToChild>,
    health_timeout: Option<Duration>,
    on_orphan: F,
) {
    let far = Duration::from_secs(1 << 30);
    let timeout = health_timeout.map(|t| t.min(far));
    let mut deadline = Instant::now() + timeout.unwrap_or(far);
    let mut pending = None;
    loop {
        tokio::select! {
            biased;
            () = tokio::time::sleep_until(deadline), if timeout.is_some() => {
                on_orphan();
                return;
            }
            permit = inbox.reserve(), if pending.is_some() => {
                let Ok(permit) = permit else { return; };
                permit.send(pending.take().unwrap());
            }
            msg = wire::read::<ToChild, _>(&mut reader), if pending.is_none() => match msg {
                Ok(msg) => {
                    if let Some(timeout) = timeout { deadline = Instant::now() + timeout; }
                    pending = Some(msg);
                }
                Err(_) => {
                    on_orphan();
                    return;
                }
            },
        }
    }
}
