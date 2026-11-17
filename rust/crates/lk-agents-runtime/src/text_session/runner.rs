use super::*;
use crate::telemetry;
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::text_session::{TextSessionDelivery, TextSessionRunner};
use lk_agents_abi::{ChatItem, Closed, StreamOutput, Ts};
use lk_agents_core::session::SessionAgent;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use parking_lot::Mutex;

#[derive(Clone)]
struct Control {
    release: mpsc::Sender<Release>,
    progress: mpsc::Sender<(String, u64, String)>,
    usage: signal::Cell<Usage>,
    remote: mpsc::Sender<RemoteCommand>,
}

impl lk_agents_abi::session::SessionCommands for Runner {
    fn input<'a>(
        &'a self,
        command: lk_agents_abi::session::SessionInputCommand,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), lk_agents_abi::storage::StorageError>> {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender =
                self.control.lock().as_ref().map(|control| control.remote.clone()).ok_or_else(
                    || StorageError::Invalid("the conversation owner is not running".into()),
                )?;
            let (done, settled) = signal::once();
            sender
                .send(RemoteCommand::Input { command, done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid(
                    "the conversation owner ended before checkpointing input".into(),
                )
            })?
        })
    }

    fn recover<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        origin: &'a lk_agents_abi::session::ToolCallRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::delegation::remote::RemoteRecovery>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender =
                self.control.lock().as_ref().map(|control| control.remote.clone()).ok_or_else(
                    || StorageError::Invalid("the conversation owner is not running".into()),
                )?;
            let (done, settled) = signal::once();
            sender
                .send(RemoteCommand::Recover {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    origin: origin.clone(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before reading recovery".into())
            })?
        })
    }

    fn default_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<String>, lk_agents_abi::storage::StorageError>>
    {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender =
                self.control.lock().as_ref().map(|control| control.remote.clone()).ok_or_else(
                    || StorageError::Invalid("the conversation owner is not running".into()),
                )?;
            let (done, settled) = signal::once();
            sender
                .send(RemoteCommand::DefaultContext {
                    remote_id: remote_id.into(),
                    endpoint: endpoint.into(),
                    done,
                })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before reading context".into())
            })?
        })
    }

    fn remote<'a>(
        &'a self,
        command: lk_agents_abi::delegation::remote::RemoteCommand,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::delegation::remote::InputRequestRef>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            use lk_agents_abi::storage::StorageError;
            let sender =
                self.control.lock().as_ref().map(|control| control.remote.clone()).ok_or_else(
                    || StorageError::Invalid("the conversation owner is not running".into()),
                )?;
            let (done, settled) = signal::once();
            sender
                .send(RemoteCommand::Apply { command: Box::new(command), done })
                .await
                .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
            settled.wait().await.map_err(|_| {
                StorageError::Invalid("the conversation owner ended before checkpointing".into())
            })?
        })
    }
}

pub struct Runner {
    config: SessionConfig,
    /// Handed to the one request: its host callbacks reach the host's run, which holds this
    /// runner, so they live only as long as that request.
    agents: Mutex<Option<BTreeMap<String, TextSessionAgent>>>,
    control: Mutex<Option<Control>>,
    spawner: fn() -> Result<Arc<dyn Spawner>, lk_agents_net::env::ProcessError>,
    clock: Arc<dyn Clock>,
    tracer: telemetry::Tracer,
}

impl TextSessionRunner for Runner {
    type Updates = Updates;
    type Stream = crate::run_stream::RunStream;
    type Resources = crate::agent_run::resources::RunResources;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        root_agent: String,
        agents: Vec<TextSessionAgent>,
        max_depth: usize,
    ) -> Result<Self, Invalid> {
        // what the model writes that is not for the person never reaches a run
        let agents: Vec<TextSessionAgent> = agents
            .into_iter()
            .map(|mut agent| {
                agent.llm = lk_agents_inference::llm::model_text::ModelText::wrap(agent.llm);
                agent
            })
            .collect();
        let config = SessionConfig::from_agents(
            root_agent,
            agents.iter().map(|agent| SessionAgent {
                definition: agent.definition.clone(),
                llm: Some(LlmRef { id: ResourceId(0), provider: agent.llm.id() }),
                output_type_explicit: agent.output_type_explicit,
                agent_tools: agent.agent_tools.clone(),
            }),
            max_depth,
        )
        .map_err(|e| Invalid(e.to_string()))?;
        let resources = Mutex::new(Some(
            agents.into_iter().map(|agent| (agent.definition.name.clone(), agent)).collect(),
        ));
        let (control, spawner, clock) = (Mutex::new(None), env.spawner, env.clock.clone());
        Ok(Self { config, agents: resources, control, spawner, clock, tracer: env.tracer.clone() })
    }

    fn run(&self, request: TextSessionRequest) -> Result<Updates, Invalid> {
        let spawner = (self.spawner)().map_err(|error| Invalid(error.to_string()))?;
        let mut current = self.control.lock();
        let agents =
            self.agents.lock().take().ok_or_else(|| Invalid("this runner already ran".into()))?;
        let (updates, control) = start(
            spawner,
            self.config.clone(),
            agents,
            request,
            self.clock.clone(),
            self.tracer.root(),
        )?;
        *current = Some(control);
        Ok(updates)
    }

    fn run_stream(
        &self,
        request: TextSessionRequest,
        stream: Arc<Self::Stream>,
        resources: Arc<Self::Resources>,
    ) -> Result<(), Invalid> {
        stream.bind(resources, || {
            let updates = self.run(request)?;
            let control =
                self.control.lock().as_ref().cloned().expect("started session owns control");
            Ok(Box::new(crate::run_stream::input::Session {
                source: Box::new(updates),
                usage: control.usage,
                release: Arc::new(move |delivery| {
                    control
                        .release
                        .try_send(Release {
                            invocation_id: delivery.invocation_id,
                            delivered: delivery.messages,
                        })
                        .map_err(|e| Invalid(e.to_string()))
                }),
            }))
        })
    }

    fn usage(&self) -> Usage {
        let current = self.control.lock();
        current.as_ref().map(|c| c.usage.get()).unwrap_or_default()
    }

    fn release(&self, delivery: TextSessionDelivery) -> Result<(), Invalid> {
        let current = self.control.lock();
        let control =
            current.as_ref().ok_or_else(|| Invalid("session request is not running".into()))?;
        control
            .release
            .try_send(Release {
                invocation_id: delivery.invocation_id,
                delivered: delivery.messages,
            })
            .map_err(|e| Invalid(e.to_string()))
    }

    async fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<(), Invalid> {
        let progress = self
            .control
            .lock()
            .as_ref()
            .map(|control| control.progress.clone())
            .ok_or_else(|| Invalid("session request is not running".into()))?;
        progress
            .send((invocation_id.into(), operation, message))
            .await
            .map_err(|e| Invalid(e.to_string()))
    }
}

fn start(
    spawner: Arc<dyn Spawner>,
    config: SessionConfig,
    agents: BTreeMap<String, TextSessionAgent>,
    request: TextSessionRequest,
    clock: Arc<dyn Clock>,
    cx: telemetry::Cx,
) -> Result<(Updates, Control), Invalid> {
    let ChatItem::Message(message) = request.message.clone() else {
        return Err(Invalid("a session message must be a ChatMessage".into()));
    };
    let state = request
        .session_sink
        .as_ref()
        .map(|binding| binding.restore())
        .transpose()
        .map_err(|error| Invalid(error.to_string()))?
        .flatten();
    let mut session = Session::new(
        config,
        SessionInput {
            session_id: state
                .as_ref()
                .map_or_else(|| request.session_id.clone(), |state| state.session_id.clone()),
            id_seed: request.id_seed.unwrap_or_else(telemetry::random_seed),
            execution_id: request.execution_id.clone(),
            task_id: request.task_id.clone(),
            state,
            wall: WallAnchor::of(&*clock),
            cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
            delegated: request.delegated,
        },
    )
    .map_err(|e| Invalid(e.to_string()))?
    .with_checkpoints();
    if request.session_sink.is_some() {
        session = session.with_persistence();
    }
    let cancel = CancellationToken::new();
    let (updates, receiver) = lk_agents_net::outbox::outbox(UPDATE_QUEUE);
    let (events_tx, events) = mpsc::channel(64);
    let (progress_tx, progress) = mpsc::channel(32);
    let (release_tx, releases) = mpsc::channel(1);
    let (remote_tx, remote) = mpsc::channel(32);
    let (usage_tx, usage) = signal::signal(session.usage());
    let driver = Driver {
        usage: usage_tx,
        published: session.usage_revision(),
        session,
        agents,
        request,
        clock: clock.clone(),
        cx,
        spawner: spawner.clone(),
        events_tx,
        updates,
        work: BTreeMap::new(),
        pending: BTreeMap::new(),
        releases: VecDeque::new(),
        output: None,
        finished: false,
    };
    let driving = Arc::new(TaskSet::default());
    driving.spawn(
        &*spawner,
        "text session",
        driver.drive(message, events, progress, releases, remote, cancel.clone()),
    );
    Ok((
        Updates { cancel, receiver, clock, _driver: driving },
        Control { release: release_tx, progress: progress_tx, usage, remote: remote_tx },
    ))
}

pub struct Updates {
    cancel: CancellationToken,
    receiver: mpsc::Receiver<TextSessionUpdate>,
    clock: Arc<dyn Clock>,
    /// The driver's owner: dropping the updates cancels the session through `cancel`, and the
    /// driver ends after its cleanup.
    _driver: Arc<TaskSet>,
}

impl Drop for Updates {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl StreamOutput<TextSessionUpdate> for Updates {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<TextSessionUpdate>, ProviderError>> {
        Box::pin(async move { Ok(self.receiver.recv().await) })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.cancel.cancel();
            let clock = self.clock.clone();
            Closed::bounded(&*clock, deadline, async move {
                while self.receiver.recv().await.is_some() {}
            })
            .await
        })
    }
}
