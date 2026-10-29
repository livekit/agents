//! The job process's side of the parent's socket (`_core.IpcClient`): the initialize handshake,
//! then it serves the parent's messages, answers pings once the child's loop ran a callback,
//! starts the one job and acks the parent's shutdown.

use std::any::Any;
use std::sync::{Arc, Mutex};

use lk_agents_abi::lock;
use lk_agents_abi::worker::ipc_host::{ChildInitialize, JobHost, WorkerError, WorkerIpcClient};
use tokio::sync::watch;

use super::job_run::JobRun;
use super::{IpcClient, PROTOCOL_VERSION, ToChild, ToParent};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::blocking_io::BlockingIo;

pub struct ChildChannel {
    client: Arc<IpcClient>,
    io: BlockingIo,
    /// `stop()`: `serve` returns.
    stop: watch::Sender<bool>,
    /// The job the parent started, once it did.
    run: Mutex<Option<Arc<JobRun>>>,
    asyncio_debug: bool,
}

impl ChildChannel {
    /// One message from the parent; `false` ends the serving.
    async fn on_message(&self, host: &dyn JobHost, msg: ToChild) -> Result<bool, WorkerError> {
        match msg {
            ToChild::Ping { sent_ms } => {
                if host.ping().await.is_err() {
                    return Ok(false);
                }
                Ok(self.client.send(ToParent::Pong { sent_ms }).await.is_ok())
            }
            // The job starts once: its run ends with the parent's shutdown.
            ToChild::StartJob { request_id, job } => {
                let accepted = self.start(host, *job).await;
                let error = accepted.as_ref().err().map(ToString::to_string);
                let sent = self.client.send(ToParent::JobAccepted { request_id, error }).await;
                accepted?;
                Ok(sent.is_ok())
            }
            // Acked, then the process ends with its job (at once without one).
            ToChild::Shutdown { reason } => {
                if self.client.send(ToParent::ShutdownAck).await.is_err() {
                    return Ok(false);
                }
                let Some(run) = lock(&self.run).clone() else {
                    let _ = self.client.send(ToParent::ShuttingDown).await;
                    return Ok(false);
                };
                let reason =
                    if reason.is_empty() { "parent process shutdown".into() } else { reason };
                run.end(reason, false);
                Ok(true)
            }
            ToChild::DumpStacks => {
                host.dump_stacks().await?;
                Ok(true)
            }
            ToChild::Hello(_) => Ok(true),
        }
    }

    async fn start(
        &self,
        host: &dyn JobHost,
        job: lk_agents_abi::worker::job::StartJobFields,
    ) -> Result<(), WorkerError> {
        if lock(&self.run).is_some() {
            return Err(WorkerError::Failed("a job is already running".into()));
        }
        let run: Arc<dyn Any + Send + Sync> = host.start_job(job).await?;
        let run = run
            .downcast::<JobRun>()
            .map_err(|_| WorkerError::Failed("the runner started a foreign JobRun".into()))?;
        run.report_to(self.client.clone());
        *lock(&self.run) = Some(run);
        Ok(())
    }
}

impl WorkerIpcClient for ChildChannel {
    type Env = ProcessEnv;

    /// Once initialized, the child calls itself orphaned when the parent goes quiet past the
    /// health timeout the parent's `Hello` gave.
    fn new(
        env: &ProcessEnv,
        fd: i64,
        initialize: Arc<dyn ChildInitialize>,
        process: bool,
    ) -> Result<Self, WorkerError> {
        let io = (env.blocking_io)().map_err(WorkerError::Failed)?;
        let runtime = io.runtime_handle();
        let socket = {
            let _runtime = runtime.enter();
            super::process::socket(fd).map_err(|error| WorkerError::Invalid(error.to_string()))?
        };
        let (client, hello) = runtime
            .block_on(IpcClient::connect_with_health_action(socket, move || {
                if process {
                    super::process::terminate_current_process();
                }
            }))
            .map_err(|error| WorkerError::Failed(format!("failed to initialize: {error}")))?;
        // The host's preloads are ready before readiness.
        let initialized =
            initialize.initialize(hello.http_proxy.clone()).map_err(WorkerError::Host);
        let error = initialized.as_ref().err().map(|error| match error {
            WorkerError::Host(error) => error.message.clone(),
            error => error.to_string(),
        });
        let sent =
            runtime.block_on(client.send(ToParent::Ready { version: PROTOCOL_VERSION, error }));
        if let Err(error) = initialized {
            runtime.block_on(client.close());
            return Err(error);
        }
        sent.map_err(|_| WorkerError::Closed)?;
        Ok(Self {
            client: Arc::new(client),
            io,
            stop: watch::channel(false).0,
            run: Mutex::new(None),
            asyncio_debug: hello.asyncio_debug,
        })
    }

    /// The parent's loop runs in debug mode: so does the child's.
    fn asyncio_debug(&self) -> bool {
        self.asyncio_debug
    }

    async fn serve(&self, host: Arc<dyn JobHost>) -> Result<(), WorkerError> {
        let mut stop = self.stop.subscribe();
        let served = async {
            loop {
                let Some(msg) = self.client.recv().await else {
                    return self
                        .client
                        .wait_reader()
                        .await
                        .map_err(|error| WorkerError::Failed(error.to_string()));
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

    async fn aclose(&self) {
        let client = self.client.clone();
        self.io.run(async move { client.close().await }).await;
    }
}
