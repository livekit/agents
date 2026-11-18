use super::*;
use crate::telemetry;
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::text_session::{TextSessionDelivery, TextSessionRunner};
use lk_agents_abi::{ChatItem, Closed, SpawnerExt, StreamOutput, Ts};
use lk_agents_core::session::SessionAgent;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use parking_lot::Mutex;

#[derive(Clone)]
pub(super) struct Control {
    pub(super) release: mpsc::Sender<Release>,
    pub(super) progress: crate::agent_run::Progress,
    pub(super) usage: signal::Cell<Usage>,
    pub(super) commands: CommandPort,
}

impl Runner {
    fn commands(&self) -> Result<CommandPort, lk_agents_abi::storage::StorageError> {
        self.control.lock().as_ref().map(|control| control.commands.clone()).ok_or_else(|| {
            lk_agents_abi::storage::StorageError::Invalid(
                "the conversation owner is not running".into(),
            )
        })
    }
}

impl lk_agents_abi::session::SessionCommands for Runner {
    fn ended(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        let owner = self.commands();
        Box::pin(async move {
            if let Ok(owner) = owner {
                owner.ended().await;
            }
        })
    }

    fn routing_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        proposed: serde_json::Value,
        selected_task: Option<&'a str>,
        offered: &'a [lk_agents_abi::session::InputCandidate],
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::delegation::routing::CallRoutingRequest,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            self.commands()?
                .routing_context(remote_id, endpoint, proposed, selected_task, offered)
                .await
        })
    }

    fn admit_call<'a>(
        &'a self,
        request: &'a lk_agents_abi::delegation::routing::CallRoutingRequest,
        verdict: Option<
            Result<
                lk_agents_abi::delegation::routing::CallRoute,
                lk_agents_abi::errors::ProviderError,
            >,
        >,
        usage: lk_agents_abi::usage::Usage,
        candidate: std::sync::Weak<dyn lk_agents_abi::a2a::remote::RemoteTaskDriver>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::a2a::remote::RemoteCallAdmission,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(
            async move { self.commands()?.admit_call(request, verdict, usage, candidate).await },
        )
    }

    fn routing_model(&self) -> Option<std::sync::Arc<dyn lk_agents_abi::Llm>> {
        self.commands().ok().and_then(|port| port.routing_model())
    }

    fn register_remote_driver(
        &self,
        remote_id: &str,
        endpoint: &str,
        request_id: &str,
        driver: std::sync::Weak<dyn lk_agents_abi::a2a::remote::RemoteTaskDriver>,
    ) {
        // Registering a weak handle changes no saved state; preparation will refuse an ended owner.
        if let Ok(port) = self.commands() {
            port.register_remote_driver(remote_id, endpoint, request_id, driver);
        }
    }

    fn remote_driver<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task: &'a lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<std::sync::Arc<dyn lk_agents_abi::a2a::remote::RemoteTaskDriver>>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move { self.commands()?.remote_driver(remote_id, endpoint, task).await })
    }

    fn cancel_remote<'a>(
        &'a self,
        target: &'a lk_agents_abi::session::control::RemoteControlRef,
        scope: Option<&'a lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: std::sync::Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
        retry_after: Option<&'a str>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::session::control::RemoteCancellationAdmission,
            lk_agents_abi::a2a::client::AgentClientError,
        >,
    > {
        Box::pin(async move {
            let commands = self.commands().map_err(|error| {
                lk_agents_abi::a2a::client::AgentClientError::InvalidInput(error.to_string())
            })?;
            commands.cancel_remote(target, scope, candidate, retry_after).await
        })
    }

    fn cancel_task<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task_id: &'a str,
        scope: Option<&'a lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: std::sync::Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (
                lk_agents_abi::session::control::RemoteControlRef,
                lk_agents_abi::session::control::RemoteCancellationAdmission,
            ),
            lk_agents_abi::a2a::client::AgentClientError,
        >,
    > {
        Box::pin(async move {
            let commands = self.commands().map_err(|error| {
                lk_agents_abi::a2a::client::AgentClientError::InvalidInput(error.to_string())
            })?;
            commands.cancel_task(remote_id, endpoint, task_id, scope, candidate).await
        })
    }

    fn cancel_outcome<'a>(
        &'a self,
        target: &'a lk_agents_abi::session::control::RemoteControlRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::delegation::remote::RemoteCancelOutcome,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move { self.commands()?.cancel_outcome(target).await })
    }

    fn admit_input(
        &self,
        request: lk_agents_abi::session::InputRoutingRequest,
        verdict: Option<
            Result<lk_agents_abi::session::InputRoute, lk_agents_abi::errors::ProviderError>,
        >,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::session::InputAdmission, lk_agents_abi::storage::StorageError>,
    > {
        Box::pin(async move { self.commands()?.admit_input(request, verdict).await })
    }

    fn checkpoint(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<serde_json::Value, lk_agents_abi::storage::StorageError>>
    {
        Box::pin(async move { self.commands()?.checkpoint().await })
    }

    fn input<'a>(
        &'a self,
        command: lk_agents_abi::session::SessionInputCommand,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), lk_agents_abi::storage::StorageError>> {
        Box::pin(async move { self.commands()?.input(command).await })
    }

    fn recover<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        target: &'a lk_agents_abi::session::SessionRecovery,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<lk_agents_abi::delegation::remote::RemoteRecovery>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move { self.commands()?.recover(remote_id, endpoint, target).await })
    }

    fn default_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<String>, lk_agents_abi::storage::StorageError>>
    {
        Box::pin(async move { self.commands()?.default_context(remote_id, endpoint).await })
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
        Box::pin(async move { self.commands()?.remote(command).await })
    }
}

pub struct Runner {
    config: SessionConfig,
    delegation: Option<lk_agents_abi::delegation::DelegationOptions>,
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
        delegation: Option<lk_agents_abi::delegation::DelegationOptions>,
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
        Ok(Self {
            config,
            delegation,
            agents: resources,
            control,
            spawner,
            clock,
            tracer: env.tracer.clone(),
        })
    }

    fn run(&self, request: TextSessionRequest) -> Result<Updates, Invalid> {
        self.run_with_output(request, None)
    }

    fn run_stream(
        &self,
        request: TextSessionRequest,
        stream: Arc<Self::Stream>,
        resources: Arc<Self::Resources>,
    ) -> Result<(), Invalid> {
        stream.bind(resources, || {
            let updates = self.run_with_output(request, Some(stream.output()))?;
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

    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
        silent: bool,
    ) -> lk_agents_abi::BoxFuture<'static, Result<(), ProviderError>> {
        match self.control.lock().as_ref() {
            Some(control) => control
                .progress
                .for_invocation(invocation_id.into())
                .enqueue(operation, message, silent),
            None => Box::pin(async {
                Err(lk_agents_abi::errors::HostError::new(
                    "the conversation report owner has ended",
                )
                .into())
            }),
        }
    }
    fn say(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<
        lk_agents_abi::BoxFuture<'static, lk_agents_abi::agent::run::SayOutcome>,
        ProviderError,
    > {
        match self.control.lock().as_ref() {
            Some(control) => {
                control.progress.for_invocation(invocation_id.into()).say(operation, message)
            }
            None => Err(lk_agents_abi::errors::HostError::new(
                "the conversation report owner has ended",
            )
            .into()),
        }
    }
}

impl Runner {
    fn run_with_output(
        &self,
        request: TextSessionRequest,
        fallback: Option<crate::agent_run::output::Target>,
    ) -> Result<Updates, Invalid> {
        let output =
            request.output.clone().map(crate::agent_run::output::Target::Handler).or(fallback);
        let spawner = (self.spawner)().map_err(|error| Invalid(error.to_string()))?;
        let mut current = self.control.lock();
        let agents =
            self.agents.lock().take().ok_or_else(|| Invalid("this runner already ran".into()))?;
        let active = request
            .session_sink
            .as_ref()
            .filter(|sink| sink.inbox_policy() == lk_agents_abi::a2a::options::Policy::Live)
            .map(|sink| sink.commands())
            .transpose()
            .map_err(|error| Invalid(error.to_string()))?
            .flatten();
        let (updates, control) = if let Some(active) = active {
            let port =
                (active.as_ref() as &dyn std::any::Any).downcast_ref::<CommandPort>().ok_or_else(
                    || Invalid("the live binding has a different native execution owner".into()),
                )?;
            live::attach(port, request, agents, self.clock.clone(), output)?
        } else {
            start(self, spawner, agents, request, output)?
        };
        *current = Some(control);
        Ok(updates)
    }
}

fn start(
    runner: &Runner,
    spawner: Arc<dyn Spawner>,
    agents: BTreeMap<String, TextSessionAgent>,
    request: TextSessionRequest,
    output_target: Option<crate::agent_run::output::Target>,
) -> Result<(Updates, Control), Invalid> {
    let clock = runner.clock.clone();
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
        runner.config.clone(),
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
    for lost in std::mem::take(&mut session.lost_tools) {
        log::warn!(
            target: lk_agents_abi::LOG_TARGET,
            "tool `{}` (call {}) of agent `{}` was running when its worker stopped; its outcome \
             is unknown (it may have taken effect), it is recorded as an internal error and is \
             not run again",
            lost.tool_name,
            lost.call_id,
            lost.agent_name,
        );
    }
    if request.session_sink.is_some() {
        session = session.with_persistence();
    }
    let live_origin = request
        .session_sink
        .as_ref()
        .filter(|sink| sink.inbox_policy() == lk_agents_abi::a2a::options::Policy::Live)
        .map(|sink| {
            sink.live_origin()
                .ok_or_else(|| Invalid("the live binding has no admitted endpoint request".into()))
        })
        .transpose()?;
    if let Some(origin) = &live_origin {
        session = session.with_live(origin.clone());
    }
    let cancel = CancellationToken::new();
    let (updates, receiver) = lk_agents_net::outbox::outbox(UPDATE_QUEUE);
    let (events_tx, events) = mpsc::channel(64);
    let driving = Arc::new(TaskSet::default());
    // A live binding attaches each later request's own output.
    let progress = crate::agent_run::Progress::new(
        String::new(),
        events_tx.clone(),
        &driving,
        spawner.clone(),
        output_target.is_some() || live_origin.is_some(),
    );
    let (release_tx, releases) = mpsc::channel(1);
    let (owner_tx, owner_commands) = mpsc::channel(2);
    let (mut commands_tx, commands) = CommandPort::channel(
        32,
        runner.delegation.as_ref().and_then(|options| options.routing_model.clone()),
    );
    let (usage_tx, usage) = signal::signal(session.usage());
    if live_origin.is_some() {
        commands_tx.live = Some(Arc::new(live::LiveControl {
            owner: owner_tx,
            cancel: cancel.clone(),
            release: release_tx.clone(),
            progress: progress.clone(),
            usage: usage.clone(),
        }));
    }
    let bound_commands: Arc<dyn lk_agents_abi::session::SessionCommands> =
        Arc::new(commands_tx.clone());
    if let Some(sink) = &request.session_sink {
        sink.bind_commands(Arc::downgrade(&bound_commands))
            .map_err(|error| Invalid(error.to_string()))?;
    }
    let live_requests = live_origin
        .as_ref()
        .map(|origin| {
            (
                origin.request_id.clone(),
                live::LiveRequest {
                    output: output_target.clone(),
                    request: request.clone(),
                    agents: agents.clone(),
                    updates: updates.clone(),
                },
            )
        })
        .into_iter()
        .collect();
    let request_sink = request.session_sink.clone();
    let checkpoint_port = request_sink
        .as_ref()
        .map(|sink| sink.checkpoint_port())
        .transpose()
        .map_err(|error| Invalid(error.to_string()))?
        .flatten();
    let driver = Driver {
        checkpoint_port,
        owner_close: if live_origin.is_some() {
            live::OwnerClose::Open
        } else {
            live::OwnerClose::Closed
        },
        _bound_commands: bound_commands,
        delegation: runner.delegation.clone(),
        routing: None,
        routing_background: lk_agents_net::owned::Background::new(
            Arc::new(TaskSet::default()),
            spawner.clone(),
            clock.clone(),
        ),
        usage: usage_tx,
        published: session.usage_revision(),
        session,
        agents,
        request,
        clock: clock.clone(),
        cx: runner.tracer.root(),
        spawner: spawner.clone(),
        events_tx,
        progress: progress.clone(),
        output_target,
        updates,
        work: BTreeMap::new(),
        live_requests,
        invocation_requests: BTreeMap::new(),
        live_commit: None,
        pending: BTreeMap::new(),
        releases: VecDeque::new(),
        output: None,
        finished: false,
    };
    driving.spawn(
        &*spawner,
        "text session",
        driver.drive(
            message,
            DriverChannels { events, releases, commands, owner_commands },
            cancel.clone(),
        ),
    );
    let driver_owner = if live_origin.is_some() {
        let owned = driving.clone();
        let owner_clock = clock.clone();
        let task = spawner.spawn(async move {
            let settled = owned.wait_empty(&*owner_clock, None).await;
            debug_assert!(settled, "an unbounded task-set join ends only after settlement");
        });
        request_sink
            .as_ref()
            .expect("live endpoint has an explicit binding")
            .retain_session(task)
            .map_err(|error| Invalid(error.to_string()))?;
        None
    } else {
        Some(driving)
    };
    Ok((
        Updates {
            cancel,
            receiver,
            clock,
            _driver: driver_owner,
            cancel_on_drop: live_origin.is_none(),
        },
        Control { release: release_tx, progress, usage, commands: commands_tx },
    ))
}

pub struct Updates {
    pub(super) cancel: CancellationToken,
    pub(super) receiver: mpsc::Receiver<TextSessionUpdate>,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) cancel_on_drop: bool,
    /// The driver's owner: dropping the updates cancels the session through `cancel`, and the
    /// driver ends after its cleanup.
    pub(super) _driver: Option<Arc<TaskSet>>,
}

impl Drop for Updates {
    fn drop(&mut self) {
        if self.cancel_on_drop {
            self.cancel.cancel();
        }
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
            if !self.cancel_on_drop {
                self.receiver.close();
                return Closed::Clean;
            }
            self.cancel.cancel();
            let clock = self.clock.clone();
            Closed::bounded(&*clock, deadline, async move {
                while self.receiver.recv().await.is_some() {}
            })
            .await
        })
    }
}
