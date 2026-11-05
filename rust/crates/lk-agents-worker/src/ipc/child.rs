//! The job process's side of the parent's channel (`_core.IpcClient`): the initialize handshake,
//! then it serves the parent's messages, answers pings once the child's loop ran a callback,
//! starts the one job and acks the parent's shutdown.

use crate::ipc::telemetry;
use std::any::Any;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use lk_agents_abi::worker::ipc_host::{ChildInitialize, JobProcess, WorkerError};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;

use super::ipc_channel::{Channel, ChannelError};
use super::job_run::JobRun;
use super::wire::{ToChild, ToParent, bounded};
use lk_agents_net::env::ProcessEnv;

/// The parent's messages the child holds while its loop is busy.
const INBOX: usize = 64;

pub struct ChildChannel {
    channel: Arc<Channel<ToParent, ToChild>>,
    /// The parent's messages, as [`watch_parent`] reads them.
    inbox: tokio::sync::Mutex<mpsc::Receiver<ToChild>>,
    watching: Mutex<Option<OwnedTask<Result<(), ChannelError>>>>,
    spawner: TokioSpawner,
    /// `stop()`: `serve` returns.
    stop: watch::Sender<bool>,
    /// The job the parent started, once it did.
    run: OnceLock<Arc<JobRun>>,
    asyncio_debug: bool,
}

/// Reads the parent's messages into a bounded inbox until the parent goes or sends what cannot be
/// read, or goes quiet past `health_timeout`, whatever the state of the child's loop; then calls
/// `on_orphan` and ends with why.
pub fn watch_parent<F: FnOnce() + Send + 'static>(
    channel: Arc<Channel<ToParent, ToChild>>,
    health_timeout: Option<Duration>,
    on_orphan: F,
) -> (mpsc::Receiver<ToChild>, OwnedTask<Result<(), ChannelError>>) {
    let (inbox, received) = mpsc::channel(INBOX);
    let timeout = health_timeout.map(|t| t.min(FAR_DEADLINE));
    let mut watch = ParentWatch {
        channel,
        inbox,
        timeout,
        deadline: Instant::now() + timeout.unwrap_or(FAR_DEADLINE),
        pending: None,
    };
    let watching = TokioSpawner::current().spawn(async move {
        let watched = watch.run().await;
        if watched.is_err() {
            on_orphan();
        }
        watched
    });
    (received, watching)
}

/// The deadline that stands for none, kept small enough to add to an `Instant`.
const FAR_DEADLINE: Duration = Duration::from_secs(1 << 30);

/// [`watch_parent`]'s task: one message at a time from the parent into the inbox.
struct ParentWatch {
    channel: Arc<Channel<ToParent, ToChild>>,
    inbox: mpsc::Sender<ToChild>,
    timeout: Option<Duration>,
    deadline: Instant,
    /// The message read but not yet admitted to the inbox.
    pending: Option<ToChild>,
}

impl ParentWatch {
    /// `Ok` once the child dropped its inbox; an error when the parent went, sent what cannot be
    /// read or went quiet. The caller keeps the inbox open until it handled the end.
    async fn run(&mut self) -> Result<(), ChannelError> {
        loop {
            tokio::select! {
                biased;
                () = tokio::time::sleep_until(self.deadline), if self.timeout.is_some() => {
                    return Err(ChannelError::Io {
                        kind: std::io::ErrorKind::TimedOut,
                        os_error: None,
                        message: "the IPC parent exceeded its health timeout".into(),
                    });
                }
                permit = self.inbox.reserve(), if self.pending.is_some() => {
                    let Ok(permit) = permit else { return Ok(()) };
                    permit.send(self.pending.take().unwrap());
                }
                msg = self.channel.recv(), if self.pending.is_none() => {
                    self.pending = Some(msg?);
                    if let Some(timeout) = self.timeout {
                        self.deadline = Instant::now() + timeout;
                    }
                }
            }
        }
    }
}

impl ChildChannel {
    /// One message from the parent; `false` ends the serving.
    async fn on_message(&self, host: &dyn JobProcess, msg: ToChild) -> Result<bool, WorkerError> {
        match msg {
            ToChild::Ping { sent_ms } => {
                if host.ping(&telemetry::Cx::NONE).await.is_err() {
                    return Ok(false);
                }
                Ok(self.channel.send(ToParent::Pong { sent_ms }).await.is_ok())
            }
            // The job starts once: its run ends with the parent's shutdown.
            ToChild::StartJob { request_id, job } => {
                let accepted = self.start(host, *job).await;
                let error = accepted.as_ref().err().map(|error| bounded(error.to_string()));
                let sent = self.channel.send(ToParent::JobAccepted { request_id, error }).await;
                accepted?;
                Ok(sent.is_ok())
            }
            // Acked, then the process ends with its job (at once without one).
            ToChild::Shutdown { reason } => {
                if self.channel.send(ToParent::ShutdownAck).await.is_err() {
                    return Ok(false);
                }
                let Some(run) = self.run.get() else {
                    let _ = self.channel.send(ToParent::ShuttingDown).await;
                    return Ok(false);
                };
                let reason =
                    if reason.is_empty() { "parent process shutdown".into() } else { reason };
                run.end(reason, telemetry::ShutdownCause::ProcessShutdown);
                Ok(true)
            }
            ToChild::Hello(_) => Ok(true),
        }
    }

    async fn start(
        &self,
        host: &dyn JobProcess,
        job: lk_agents_abi::worker::job::StartJobFields,
    ) -> Result<(), WorkerError> {
        let running = || WorkerError::Failed("a job is already running".into());
        if self.run.get().is_some() {
            return Err(running());
        }
        let run: Arc<dyn Any + Send + Sync> = host.start_job(&telemetry::Cx::NONE, job).await?;
        let run = run
            .downcast::<JobRun>()
            .map_err(|_| WorkerError::Failed("the runner started a foreign JobRun".into()))?;
        run.report_to(self.channel.clone());
        self.run.set(run).map_err(|_| running())
    }

    /// The inbox ended: why the parent watch ended, with the parent closing counted as done.
    async fn finish_watching(&self) -> Result<(), WorkerError> {
        let watching = self.watching.lock().take();
        let Some(watching) = watching else { return Ok(()) };
        match watching.await {
            Ok(Ok(())) | Ok(Err(ChannelError::Closed)) => Ok(()),
            Ok(Err(error)) => Err(WorkerError::Failed(error.to_string())),
            Err(error) => Err(WorkerError::Failed(error.to_string())),
        }
    }
}

impl lk_agents_abi::worker::ipc_host::IpcClient for ChildChannel {
    type Env = ProcessEnv;

    /// Once initialized, the child calls itself orphaned when the parent goes quiet past the
    /// health timeout the parent's `Hello` gave.
    fn new(
        env: &ProcessEnv,
        fd: i64,
        initialize: Arc<dyn ChildInitialize>,
        process: bool,
    ) -> Result<Self, WorkerError> {
        let spawner = TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?);
        let runtime = spawner.handle().clone();
        let _runtime = runtime.enter();
        let channel = Arc::new(
            Channel::<ToParent, ToChild>::from_endpoint(fd)
                .map_err(|error| WorkerError::Invalid(error.to_string()))?,
        );
        let hello = match runtime.block_on(channel.recv()) {
            Ok(ToChild::Hello(hello)) => hello,
            Ok(other) => {
                return Err(WorkerError::Failed(format!("expected Hello first, got {other:?}")));
            }
            Err(error) => {
                return Err(WorkerError::Failed(format!("failed to initialize: {error}")));
            }
        };
        let health_timeout = hello.health_timeout_ms.map(Duration::from_millis);
        let (inbox, watching) = watch_parent(channel.clone(), health_timeout, move || {
            if process {
                super::process::terminate_current_process();
            }
        });
        // The host's preloads are ready before readiness.
        let initialized =
            initialize.initialize(hello.http_proxy.clone()).map_err(WorkerError::Host);
        let error = initialized.as_ref().err().map(|error| match error {
            WorkerError::Host(error) => bounded(error.message.clone()),
            error => bounded(error.to_string()),
        });
        let sent = runtime.block_on(channel.send(ToParent::Ready { error }));
        if let Err(error) = initialized {
            drop(watching);
            runtime.block_on(channel.close());
            return Err(error);
        }
        sent.map_err(|_| WorkerError::Closed)?;
        Ok(Self {
            channel,
            inbox: tokio::sync::Mutex::new(inbox),
            watching: Mutex::new(Some(watching)),
            spawner,
            stop: watch::channel(false).0,
            run: OnceLock::new(),
            asyncio_debug: hello.asyncio_debug,
        })
    }

    /// The parent's loop runs in debug mode: so does the child's.
    fn asyncio_debug(&self) -> bool {
        self.asyncio_debug
    }

    async fn serve(&self, host: Arc<dyn JobProcess>) -> Result<(), WorkerError> {
        let mut stop = self.stop.subscribe();
        let served = async {
            loop {
                let Some(msg) = self.inbox.lock().await.recv().await else {
                    return self.finish_watching().await;
                };
                if !self.on_message(host.as_ref(), msg).await? {
                    return Ok(());
                }
            }
        };
        tokio::select! {
            served = served => served,
            _ = stop.wait_for(|stopped| *stopped) => Ok(()),
        }
    }

    fn stop(&self) {
        self.stop.send_replace(true);
    }

    /// Stops watching the parent, then closes the channel: what is queued is flushed and the
    /// parent's last messages are drained, so the parent never sees a reset.
    async fn aclose(&self) {
        let watching = self.watching.lock().take();
        let channel = self.channel.clone();
        self.spawner
            .run(async move {
                if let Some(watching) = watching {
                    watching.cancel();
                    let _ = watching.await;
                }
                channel.close().await;
            })
            .await;
    }
}
