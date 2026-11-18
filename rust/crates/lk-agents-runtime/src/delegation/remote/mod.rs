use std::sync::Arc;

use parking_lot::Mutex;

use lk_agents_abi::a2a::client::{
    self as port, AgentClientError, Message, MessageInput, MessagePart,
};
use lk_agents_abi::a2a::remote::{
    DelegationHost, DelegationInput, RemoteEvent, RemoteInteraction as Port, RemoteResponseClaim,
    RemoteTaskDriver,
};
use lk_agents_abi::delegation::remote::{A2ARequest, RemoteRecovery};
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::invoke::{CallContext, Cancellation as _};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ApiConnectOptions, ChatItem, FunctionCall, HOST_TEARDOWN_TIMEOUT};
use lk_agents_core::a2a::caller_history::{CallerHistory, HistorySharing};
use lk_agents_core::delegation::remote::{
    DelegationUpdate, InteractionId, RemoteInteraction as Policy, retained,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use lk_agents_core::a2a::client::{TaskUpdate, message_from_wire};
use lk_agents_net::a2a_client::{AgentClient, Reply};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use lk_agents_net::task::Cancellation;
use lk_agents_telemetry as telemetry;
mod cancel;
mod configuration;
pub mod control;
mod driver;
mod exchange;
pub mod journal;

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;

pub struct RemoteInteraction {
    inner: Interaction,
    /// The running delegation's stop, settlement and the clock bounding its wait.
    running: Mutex<Option<Running>>,
}

struct Running {
    stop: Arc<Cancellation>,
    settled: Settled,
    clock: Arc<dyn lk_agents_abi::Clock>,
}

/// Ends with the delegation's task, however it ends.
type Settled = signal::Cell<Option<()>>;

/// The interaction's decisions, shared with its running delegation.
#[derive(Clone)]
struct Interaction(Arc<Mutex<State>>);

struct State {
    policy: Policy,
    call: FunctionCall,
    livekit: bool,
    project: Project,
    partition_chain: Vec<lk_agents_abi::a2a::remote::DelegationPartition>,
    control_scope: Option<lk_agents_abi::delegation::remote::RemoteControlScope>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextRequest {
    request: String,
}

impl Port for RemoteInteraction {
    type Env = ProcessEnv;

    fn new(
        endpoint: String,
        function_call: ChatItem,
        project_url: String,
        api_key: String,
        api_secret: String,
        remote_id: Option<String>,
    ) -> Result<Self, AgentClientError> {
        let ChatItem::FunctionCall(function_call) = function_call else {
            return Err(AgentClientError::InvalidInput(
                "a remote interaction needs its parent function call".into(),
            ));
        };
        let endpoint = configuration::endpoint(&endpoint)?;
        let remote_id = configuration::remote_id(&endpoint, remote_id)?;
        let id = InteractionId { remote_id, endpoint, parent_call_id: function_call.id.clone() };
        let policy = Policy::new(id).map_err(protocol)?;
        let project = Project { url: project_url, api_key, api_secret };
        let state = State {
            policy,
            call: function_call,
            livekit: false,
            project,
            partition_chain: Vec::new(),
            control_scope: None,
        };
        Ok(Self { inner: Interaction(Arc::new(Mutex::new(state))), running: Mutex::new(None) })
    }

    fn set_partition_chain(
        &self,
        chain: Vec<lk_agents_abi::a2a::remote::DelegationPartition>,
    ) -> Result<(), AgentClientError> {
        let mut metadata = Map::new();
        metadata.insert(
            lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY.into(),
            serde_json::to_value(&chain)
                .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?,
        );
        let chain = lk_agents_core::delegation::remote::partition_chain(&metadata)?;
        self.inner.0.lock().partition_chain = chain;
        Ok(())
    }

    fn message(
        &self,
        arguments: Value,
        structured: bool,
        offers_extension: bool,
        history: Vec<ChatItem>,
        thread_id: Option<String>,
        token: Option<String>,
    ) -> Result<Message, AgentClientError> {
        self.inner.message(
            arguments,
            structured,
            offers_extension,
            &history,
            thread_id.map(|thread| (thread, token)),
            None,
        )
    }

    async fn run(
        &self,
        env: &ProcessEnv,
        host: Arc<dyn DelegationHost>,
        arguments: Value,
        structured: bool,
        history: Vec<ChatItem>,
        token: Option<String>,
        thread_id: Option<String>,
        conn_options: Option<ApiConnectOptions>,
        live: bool,
        owner: Option<Arc<dyn lk_agents_abi::session::SessionCommands>>,
        origin: Option<lk_agents_abi::session::ToolCallRef>,
        context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<RemoteEvent, AgentClientError> {
        let (endpoint, project) = {
            let state = self.inner.0.lock();
            (state.policy.snapshot().id.endpoint, state.project.clone())
        };
        let configured_token = token.clone();
        let token = <Self as lk_agents_abi::a2a::remote::RemoteConfiguration>::_core_remote_token(
            endpoint.clone(),
            project.url,
            project.api_key,
            project.api_secret,
            token,
            thread_id.clone(),
        )?;
        let shared = if configured_token.is_none() && token.is_some() {
            thread_id.map(|thread| (thread, token.clone()))
        } else {
            None
        };
        let conn_options = conn_options.unwrap_or_default();
        let client = AgentClient::new(env, endpoint, token.clone(), conn_options)?;
        let result = self
            .run_with_client(
                env,
                client,
                RemoteInvocation {
                    arguments,
                    structured,
                    history,
                    shared,
                    configured_token,
                    live,
                    host,
                    context,
                    cx,
                    owner,
                    origin,
                    conn_options,
                },
            )
            .await;
        match result {
            Err(AgentClientError::InvalidInput(message)) => {
                Ok(RemoteEvent::Outcome { outcome: ToolOutcome::ToolError(message) })
            }
            result => result,
        }
    }

    async fn cancel_task(
        &self,
        env: &ProcessEnv,
        arguments: Value,
        thread_id: Option<String>,
        owner: Option<Arc<dyn lk_agents_abi::session::SessionCommands>>,
    ) -> Result<ToolOutcome, AgentClientError> {
        use lk_agents_core::delegation::remote::cancel::outcome_text;
        let task_id = match lk_agents_core::a2a::client::cancel_task_id(&arguments) {
            Ok(task_id) => task_id,
            Err(error) => {
                return Ok(ToolOutcome::ToolError(AgentClientError::from(error).to_string()));
            }
        };
        let Some(owner) = owner else {
            return Ok(ToolOutcome::ToolError(
                "There are no running tasks to cancel in this conversation.".into(),
            ));
        };
        let (identity, project) = {
            let state = self.inner.0.lock();
            (state.policy.snapshot().id, state.project.clone())
        };
        let outcome = control::cancel_for_model(
            env,
            owner,
            &identity.remote_id,
            &identity.endpoint,
            &task_id,
            &project,
            thread_id.filter(|thread| !thread.is_empty()),
        )
        .await;
        match outcome {
            Ok(outcome) => Ok(ToolOutcome::Value {
                text: outcome_text(outcome).into(),
                reply_required: true,
                verbatim: false,
                more: None,
            }),
            Err(AgentClientError::InvalidInput(message)) => Ok(ToolOutcome::ToolError(message)),
            Err(error) => Err(error),
        }
    }

    fn cancelled(&self) -> Result<(), AgentClientError> {
        if let Some(running) = &*self.running.lock() {
            running.stop.cancel();
        }
        // The frontend stopped awaiting this call. Its remote task and input occurrence
        // retain their observed state; explicit task cancellation is a separate operation.
        Ok(())
    }

    async fn settled(&self) {
        let running = self.running.lock().as_ref().map(|r| (r.settled.clone(), r.clock.clone()));
        if let Some((settled, clock)) = running
            && lk_agents_abi::timeout(&*clock, HOST_TEARDOWN_TIMEOUT, settled.wait()).await.is_err()
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "delegation cleanup incomplete");
        }
    }

    fn extra(&self) -> Result<Map<String, Value>, AgentClientError> {
        self.inner.extra()
    }
}

/// One native invocation. The host adapter supplies conversation inputs and services; the
/// deterministic tests inject only the endpoint client, keeping the same executor and owner.
pub struct RemoteInvocation {
    pub arguments: Value,
    pub structured: bool,
    pub history: Vec<ChatItem>,
    pub shared: Option<(String, Option<String>)>,
    /// The application's original custom credential; default minting stays native.
    pub configured_token: Option<String>,
    pub live: bool,
    pub host: Arc<dyn DelegationHost>,
    pub context: CallContext,
    pub cx: telemetry::Cx,
    pub owner: Option<Arc<dyn lk_agents_abi::session::SessionCommands>>,
    pub origin: Option<lk_agents_abi::session::ToolCallRef>,
    pub conn_options: ApiConnectOptions,
}

impl RemoteInteraction {
    /// Run over the supplied socket boundary. Production and virtual-time execution share the
    /// same process-owned task, cancellation and cleanup path.
    pub async fn run_with_client(
        &self,
        env: &ProcessEnv,
        client: AgentClient,
        request: RemoteInvocation,
    ) -> Result<RemoteEvent, AgentClientError> {
        let background = env.background().map_err(unavailable)?;
        let stop = Arc::new(Cancellation::default());
        let (done, settled) = signal::once::<()>();
        *self.running.lock() =
            Some(Running { stop: stop.clone(), settled, clock: Arc::clone(&background.clock) });
        let close_clock = background.clock.clone();
        let client = Owned::new(
            &background,
            Arc::new(client),
            Box::new(move |client, deadline| {
                Box::pin(async move {
                    lk_agents_abi::Closed::bounded(&*close_clock, deadline, async move {
                        client.aclose().await;
                    })
                    .await
                })
            }),
        );
        let configuration = {
            let state = self.inner.0.lock();
            configuration::Configuration::new(
                &state.policy.snapshot().id.endpoint,
                &state.project,
                request.configured_token.as_deref(),
                request.shared.as_ref().map(|(thread, _)| thread.as_str()),
            )?
        };
        let (driver, commands) = driver::Driver::channel(&background, configuration);
        let delegation = Delegation {
            interaction: self.inner.clone(),
            client,
            request,
            journal: None,
            background: background.clone(),
            driver,
            commands,
            response_claim: None,
        };
        // The process's task: a dropped `run` leaves it to the host, which stops it in the same
        // turn (`cancelled()`), and `settled()` waits for its cleanup.
        let (outcome, result) = tokio::sync::oneshot::channel();
        background.spawn("delegation", async move {
            let _done = done;
            let _ = outcome.send(delegation.run(&stop).await);
        });
        result.await.unwrap_or_else(|_| {
            Err(AgentClientError::Protocol("the delegation's task was cancelled".into()))
        })
    }
}

/// One running delegation over its own endpoint client.
struct Delegation {
    interaction: Interaction,
    client: Owned<Arc<AgentClient>>,
    request: RemoteInvocation,
    /// The active exchange's durable input receipt, never a lookup of a later task status.
    journal: Option<Arc<journal::Journal>>,
    background: Background,
    driver: Arc<driver::Driver>,
    commands: tokio::sync::mpsc::Receiver<driver::Command>,
    response_claim: Option<Arc<RemoteResponseClaim>>,
}

enum Prepared {
    Forwarded,
    Message { arguments: Value, structured: bool },
}

enum Next {
    Update(Result<Option<TaskUpdate>, AgentClientError>),
    Command(Option<driver::Command>),
    Stop,
}

fn cancelled() -> RemoteEvent {
    RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled }
}

fn forwarded() -> RemoteEvent {
    RemoteEvent::Outcome {
        outcome: ToolOutcome::Value {
            text: "This request is already tracked by the existing remote task; progress, questions and result arrive through its original call.".into(),
            reply_required: true,
            verbatim: false,
            more: None,
        },
    }
}

impl Delegation {
    /// Resolve addressing before connecting a second transport. An active task always keeps
    /// its original observer and result owner, including while it is waiting for input.
    async fn prepare(&mut self) -> Result<Option<Prepared>, AgentClientError> {
        let mut arguments = self.request.arguments.clone();
        let task_id = lk_agents_core::a2a::client::take_task_id(&mut arguments)
            .map_err(AgentClientError::from)?;
        let Some(owner) = self.request.owner.clone() else {
            if task_id.is_some() {
                return Err(AgentClientError::InvalidInput(
                    "task_id needs an active conversation with saved remote tasks".into(),
                ));
            }
            return Ok(Some(Prepared::Message { arguments, structured: self.request.structured }));
        };
        let identity = self.interaction.0.lock().policy.snapshot().id;
        let offered = {
            let state = self.interaction.0.lock();
            lk_agents_core::delegation::remote::routing::captured_offers(&state.call)
                .map_err(input_error)?
        };
        let routing = owner
            .routing_context(
                &identity.remote_id,
                &identity.endpoint,
                arguments.clone(),
                task_id.as_deref(),
                &offered,
            )
            .await
            .map_err(input_error)?;
        let (verdict, usage) = match owner.routing_model() {
            Some(model) if !routing.candidates.is_empty() => {
                let (verdict, usage) = lk_agents_inference::delegation::call::classify(
                    model,
                    &self.background,
                    &self.request.cx,
                    routing.clone(),
                    self.request.conn_options,
                )
                .await;
                (Some(verdict), usage)
            }
            _ => (
                Some(Ok(lk_agents_core::delegation::remote::routing::without_classifier(&routing))),
                Default::default(),
            ),
        };
        use lk_agents_abi::delegation::routing::CallAdmission;
        let candidate: Arc<dyn RemoteTaskDriver> = self.driver.clone();
        let admitted = owner
            .admit_call(&routing, verdict, usage, Arc::downgrade(&candidate))
            .await
            .map_err(input_error)?;
        let (task, selected, structured) = match admitted.admission {
            CallAdmission::New => (None, None, self.request.structured),
            CallAdmission::Continue(task) => (Some(task), None, self.request.structured),
            CallAdmission::Respond { request, message } => {
                let (_, target, _) = request
                    .remote()
                    .ok_or_else(|| protocol("a remote response needs its captured task"))?;
                let target = target.clone();
                // The admitted canonical human words, never the frontend model's rewrite.
                arguments =
                    serde_json::json!({"request": message.raw_text_content().unwrap_or_default()});
                (Some(target), Some(request), false)
            }
            CallAdmission::AlreadyForwarded { .. } => return Ok(Some(Prepared::Forwarded)),
        };
        if let Some(driver) = admitted.driver
            && !Arc::ptr_eq(&driver, &candidate)
        {
            driver
                .input(
                    DelegationInput { arguments, structured, input_request: selected },
                    admitted.response_claim,
                    self.request.origin.clone(),
                )
                .await?;
            return Ok(Some(Prepared::Forwarded));
        }
        self.response_claim = admitted.response_claim;
        let mut state = self.interaction.0.lock();
        if let Some(reference) = selected {
            state.policy.respond_to(reference).map_err(protocol)?;
        } else if let Some(task) = task {
            state.policy.target_task(task).map_err(protocol)?;
        }
        Ok(Some(Prepared::Message { arguments, structured }))
    }

    /// The delegation until its outcome or `stop`; its reply and client are closed either way.
    async fn run(mut self, stop: &Cancellation) -> Result<RemoteEvent, AgentClientError> {
        let mut reply = None;
        let client = Arc::clone(&self.client);
        let result = {
            let driving = self.drive(&mut reply, stop);
            tokio::pin!(driving);
            tokio::select! {
                biased;
                result = &mut driving => result,
                () = stop.cancelled() => {
                    // Close the socket boundary to settle an in-flight exchange, then let the
                    // pinned driver finish any consumed update or checkpoint of its own.
                    client.aclose().await;
                    driving.await
                }
            }
        };
        self.commands.close();
        while let Some(command) = self.commands.recv().await {
            command.done.resolve(Err(AgentClientError::InvalidInput(
                "the remote task observer ended before this command ran".into(),
            )));
        }
        self.driver.close().await;
        if let Some(reply) = reply
            && close_observation(reply).await.is_err()
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "delegation response cleanup incomplete");
        }
        if self.client.close().await != lk_agents_abi::Closed::Clean {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "delegation client cleanup incomplete");
        }
        result
    }

    async fn drive(
        &mut self,
        reply: &mut Option<Owned<Reply>>,
        stop: &Cancellation,
    ) -> Result<RemoteEvent, AgentClientError> {
        let recovery = match (&self.request.owner, &self.request.origin) {
            (Some(owner), Some(origin)) => {
                let identity = self.interaction.0.lock().policy.snapshot().id;
                owner
                    .recover(
                        &identity.remote_id,
                        &identity.endpoint,
                        &lk_agents_abi::session::SessionRecovery::Call(origin.clone()),
                    )
                    .await
                    .map_err(input_error)?
            }
            _ => None,
        };
        if let Some(recovery) = &recovery {
            let restoring = self.recover_live(recovery);
            tokio::pin!(restoring);
            let forwarded = tokio::select! {
                biased;
                restored = &mut restoring => restored?,
                () = stop.cancelled() => return Ok(cancelled()),
            };
            if forwarded {
                return Ok(self::forwarded());
            }
        }
        let prepared = if recovery.is_none() {
            // Before observation starts, cancellation may release classifier reads or a
            // message-admission waiter. Any admitted message stays with its owning driver.
            let preparing = self.prepare();
            tokio::pin!(preparing);
            tokio::select! {
                biased;
                prepared = &mut preparing => prepared?,
                () = stop.cancelled() => return Ok(cancelled()),
            }
        } else {
            None
        };
        if let Some(Prepared::Forwarded) = prepared {
            return Ok(forwarded());
        }
        if stop.is_cancelled() {
            return Ok(cancelled());
        }
        let mut recovered = None;
        if let Some(RemoteRecovery::Observed(response)) = &recovery {
            let response = match response.as_ref() {
                a2a::SendMessageResponse::Task(task) => a2a::StreamResponse::Task(task.clone()),
                a2a::SendMessageResponse::Message(message) => {
                    a2a::StreamResponse::Message(message.clone())
                }
            };
            let mut event = self.interaction.reconciled(response)?;
            if matches!(event, RemoteEvent::Outcome { .. }) {
                self.progress(&mut event).await?;
                return Ok(event);
            }
            // A completed merged task still follows its replacement through the ordinary
            // owner loop; terminal protocol status alone is not a business answer.
            recovered = Some(event);
        }
        port::AgentClient::connect(self.client.as_ref()).await?;
        self.interaction.0.lock().livekit = self.client.offers_extension();
        let mut accepted = true;
        let mut initial = match recovery {
            Some(RemoteRecovery::Dispatch { request_id, request, streaming, input_request }) => {
                let A2ARequest::Send(request) = *request else {
                    return Err(protocol("a recovered tool call does not contain a send"));
                };
                let observer = self
                    .observer_for_input(request_id, None, input_request)
                    .ok_or_else(|| protocol("recovery needs its native owner"))?;
                let resumed = self
                    .client
                    .resume_send(*request, streaming, &self.request.cx, observer)
                    .await?;
                *reply = self.claim_reply(resumed).await?;
                if reply.is_none() {
                    return Ok(forwarded());
                }
                None
            }
            Some(RemoteRecovery::GetTask(task)) => {
                let observer = self.observer();
                let task = self
                    .client
                    .get_task(
                        lk_agents_core::a2a::rpc::TaskRef {
                            id: task.task_id,
                            context_id: task.context_id,
                        },
                        &self.request.cx,
                        observer,
                    )
                    .await?;
                Some(self.interaction.reconciled(a2a::StreamResponse::Task(task))?)
            }
            Some(RemoteRecovery::Observed(_)) => recovered.take(),
            Some(RemoteRecovery::Uncertain) => {
                return Err(protocol(
                    "the remote request may have been accepted, but no task or reply was observed; clarify its outcome before starting new work",
                ));
            }
            None => {
                let Some(Prepared::Message { arguments, structured }) = prepared else {
                    return Err(protocol("a new delegation needs prepared business input"));
                };
                let default_context = match &self.request.owner {
                    Some(owner) => {
                        let identity = self.interaction.0.lock().policy.snapshot().id;
                        owner
                            .default_context(&identity.remote_id, &identity.endpoint)
                            .await
                            .map_err(|error| AgentClientError::Protocol(error.to_string()))?
                    }
                    None => None,
                };
                let message = self.interaction.message(
                    arguments,
                    structured,
                    self.client.offers_extension(),
                    &self.request.history,
                    self.request.shared.clone(),
                    default_context,
                )?;
                let input = self.interaction.0.lock().policy.snapshot().input_request;
                let sent = self.send(message, None, input).await;
                drop(self.response_claim.take());
                match sent {
                    Ok(observed) => *reply = observed,
                    Err(error) => {
                        if self.commands.is_empty() {
                            return Err(error);
                        }
                        // A different admitted input may supersede a stale initial response.
                        // Keep its concrete queued work; never retry this failed message.
                        accepted = false;
                    }
                }
                if reply.is_none() && accepted {
                    return Ok(forwarded());
                }
                None
            }
        };
        if self.request.live && accepted {
            // The peer's acknowledgement names the task; a recovered observation may only
            // have reached the interaction's own record of it.
            let task_id = reply
                .as_ref()
                .map(|reply| reply.task_id.clone())
                .filter(|id| !id.is_empty())
                .or_else(|| self.interaction.0.lock().policy.snapshot().pending_task_id);
            let request =
                lk_agents_core::delegation::remote::facts::request_summary(&self.request.arguments);
            let fact =
                lk_agents_core::delegation::remote::facts::running(task_id.as_deref(), &request);
            self.request
                .host
                .delegated(&self.request.cx, &self.request.context, fact)
                .await
                .map_err(host_error)?;
        }
        loop {
            let mut event = if let Some(event) = initial.take() {
                event
            } else {
                // Only reads race here. A consumed update's host delivery and journal write
                // finish outside the race; a queued terminal result wins over a correction.
                let next = match reply.as_ref().map(|current| &**current) {
                    Some(current) => tokio::select! {
                        biased;
                        update = next_update(current) => Next::Update(update),
                        command = self.commands.recv() => Next::Command(command),
                        () = stop.cancelled() => Next::Stop,
                    },
                    None => tokio::select! {
                        biased;
                        command = self.commands.recv() => Next::Command(command),
                        () = stop.cancelled() => Next::Stop,
                    },
                };
                match next {
                    Next::Update(Ok(Some(update))) => self.interaction.update(update)?,
                    Next::Update(Ok(None)) => {
                        let lost = protocol("the remote interaction ended without an outcome");
                        initial = self.observation_lost(reply, lost).await?;
                        continue;
                    }
                    Next::Update(Err(error)) => {
                        initial = self.observation_lost(reply, error).await?;
                        continue;
                    }
                    Next::Command(Some(command)) => {
                        initial = self.message_command(command, reply).await?;
                        continue;
                    }
                    Next::Command(None) => return Err(protocol("the remote driver ended")),
                    Next::Stop => return Ok(cancelled()),
                }
            };
            self.progress(&mut event).await?;
            match event {
                RemoteEvent::Pending | RemoteEvent::Progress { .. } | RemoteEvent::Exact { .. } => {
                }
                RemoteEvent::Follow { task_id, thread_id } => {
                    if let Some(done) = reply.take() {
                        close_observation(done).await?;
                    }
                    if let Some(driver) = self
                        .driver_elsewhere(&lk_agents_abi::delegation::remote::RemoteTaskRef {
                            task_id: task_id.clone(),
                            context_id: thread_id.clone(),
                        })
                        .await?
                    {
                        driver.reconcile().await;
                        return Ok(forwarded());
                    }
                    let task =
                        lk_agents_core::a2a::rpc::TaskRef { id: task_id, context_id: thread_id };
                    let observer = self.observer();
                    let followed = self
                        .client
                        .subscribe_observed(task, None, &self.request.cx, observer)
                        .await
                        .map_err(AgentClientError::from)?;
                    *reply = self.claim_reply(followed).await?;
                    if reply.is_none() {
                        return Ok(forwarded());
                    }
                }
                RemoteEvent::InputRequired { question, verbatim_full, input_request, .. }
                    if self.request.live =>
                {
                    if let Some(done) = reply.take() {
                        close_observation(done).await?;
                    }
                    self.request
                        .host
                        .question(
                            &self.request.cx,
                            &self.request.context,
                            question,
                            verbatim_full,
                            input_request,
                        )
                        .await
                        .map_err(host_error)?;
                }
                event @ (RemoteEvent::InputRequired { .. } | RemoteEvent::Outcome { .. }) => {
                    return Ok(event);
                }
            }
        }
    }

    async fn progress(&mut self, event: &mut RemoteEvent) -> Result<(), AgentClientError> {
        if let RemoteEvent::InputRequired { input_request, .. } = event {
            let reference = self.journal.as_ref().and_then(|journal| journal.observed_input());
            if self.request.owner.is_some() && reference.is_none() {
                return Err(protocol(
                    "the remote input request is not open; reconcile its earlier response before asking again",
                ));
            }
            self.interaction
                .0
                .lock()
                .policy
                .retain_input_request(reference.clone())
                .map_err(protocol)?;
            *input_request = reference;
        }
        let extra = self.interaction.extra()?;
        if let RemoteEvent::Exact { text } = event {
            return self
                .request
                .host
                .say(&self.request.cx, &self.request.context, text.clone(), extra)
                .await
                .map_err(host_error);
        }
        let text = match event {
            RemoteEvent::Progress { text } => Some(text.clone()),
            _ => None,
        };
        self.request
            .host
            .progress(&self.request.cx, &self.request.context, text, extra, true)
            .await
            .map_err(host_error)
    }
}

/// The native path preserves original SDK messages/artifacts; host normalization is only
/// for its convenience API. The source pull owns a cancellation-safe read position.
async fn next_update(reply: &Reply) -> Result<Option<TaskUpdate>, AgentClientError> {
    let pull = reply.try_pull().ok_or(AgentClientError::Overlap)?;
    Ok(match pull.await? {
        lk_agents_net::source::Pulled::Item(update) => Some(update),
        lk_agents_net::source::Pulled::End | lk_agents_net::source::Pulled::Cancelled => None,
    })
}

async fn close_observation(reply: Owned<Reply>) -> Result<(), AgentClientError> {
    match reply.close().await {
        lk_agents_abi::Closed::Clean => Ok(()),
        lk_agents_abi::Closed::Failed | lk_agents_abi::Closed::TimedOut => {
            Err(protocol("the previous remote observation did not finish cleanup"))
        }
    }
}

impl Interaction {
    fn message(
        &self,
        arguments: Value,
        structured: bool,
        offers_extension: bool,
        history: &[ChatItem],
        shared: Option<(String, Option<String>)>,
        default_context: Option<String>,
    ) -> Result<Message, AgentClientError> {
        let mut state = self.0.lock();
        let (thread_id, token) = shared.unzip();
        let token = token.flatten();
        // sharing the caller's thread needs a grant this project signed for it, verified here
        let authority = match (thread_id.as_deref(), token.as_deref()) {
            (Some(thread), Some(token)) => {
                let endpoint = state.policy.snapshot().id.endpoint;
                let project = state.project.serving(&endpoint)?.ok_or_else(|| {
                    AgentClientError::InvalidInput(
                        "sharing a thread needs the endpoint on the configured project".into(),
                    )
                })?;
                let assertion = lk_agents_net::a2a_client::authority::AuthorityAssertion::Grant {
                    project_url: &project.url,
                    api_key: &project.api_key,
                    api_secret: &project.api_secret,
                    token,
                };
                let now = std::time::SystemTime::now();
                Some(
                    lk_agents_net::a2a_client::authority::verify(
                        &endpoint, thread, DELEGATE, assertion, now,
                    )
                    .map_err(protocol)?,
                )
            }
            _ => None,
        };
        state.control_scope = if authority.is_some() {
            let project = state
                .project
                .serving(&state.policy.snapshot().id.endpoint)?
                .expect("verified authority was restricted to this project");
            Some(lk_agents_abi::delegation::remote::RemoteControlScope {
                project_origin: url::Url::parse(&project.url)
                    .map_err(protocol)?
                    .origin()
                    .ascii_serialization(),
                project_issuer: project.api_key,
                caller_identity: DELEGATE.into(),
                thread_id: thread_id.clone().expect("verified sharing has a caller thread"),
            })
        } else {
            None
        };
        if !arguments.is_object() {
            return Err(AgentClientError::InvalidInput(
                "remote tool arguments must be an object".into(),
            ));
        }
        let text = if structured {
            String::new()
        } else {
            serde_json::from_value::<TextRequest>(arguments.clone())
                .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?
                .request
        };
        let original_call = state.policy.snapshot().id.parent_call_id;
        let caller_history = offers_extension
            .then(|| {
                CallerHistory::from_chat_items(
                    history,
                    Some(&original_call),
                    &HistorySharing::default(),
                )
            })
            .transpose()
            .map_err(protocol)?;
        let mut message = state
            .policy
            .message(
                text,
                lk_agents_net::strings::shortuuid("msg_"),
                thread_id.as_deref().filter(|_| authority.is_some()),
                DELEGATE,
                authority.as_ref(),
                offers_extension,
                caller_history.as_ref(),
                default_context.as_deref(),
            )
            .map_err(protocol)?;
        if structured {
            message.parts[0] = a2a::Part::data(arguments);
        }
        if offers_extension && !state.partition_chain.is_empty() {
            message.metadata.get_or_insert_default().insert(
                lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY.into(),
                serde_json::to_value(&state.partition_chain)
                    .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?,
            );
        }
        state.livekit = offers_extension;
        Ok(message_from_wire(message))
    }

    fn task_message(&self, input: &DelegationInput) -> Result<Message, AgentClientError> {
        if !input.arguments.is_object() {
            return Err(AgentClientError::InvalidInput(
                "remote tool arguments must be an object".into(),
            ));
        }
        let text = if input.structured {
            String::new()
        } else {
            serde_json::from_value::<TextRequest>(input.arguments.clone())
                .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?
                .request
        };
        let state = self.0.lock();
        let mut message = state
            .policy
            .task_message(
                text,
                lk_agents_net::strings::shortuuid("msg_"),
                input.input_request.clone(),
                state.livekit,
            )
            .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?;
        if input.structured {
            message.parts[0] = a2a::Part::data(input.arguments.clone());
        }
        if state.livekit && !state.partition_chain.is_empty() {
            message.metadata.get_or_insert_default().insert(
                lk_agents_abi::a2a::metadata::PARTITION_CHAIN_KEY.into(),
                serde_json::to_value(&state.partition_chain)
                    .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?,
            );
        }
        Ok(message_from_wire(message))
    }

    fn update(&self, update: TaskUpdate) -> Result<RemoteEvent, AgentClientError> {
        let mut state = self.0.lock();
        let event = state.policy.update(&update).map_err(protocol)?;
        Ok(state.event(event))
    }

    fn reconciled(&self, response: a2a::StreamResponse) -> Result<RemoteEvent, AgentClientError> {
        let mut state = self.0.lock();
        let event = state.policy.reconcile(response).map_err(protocol)?;
        Ok(state.event(event))
    }

    fn reconciled_lost(
        &self,
        response: a2a::StreamResponse,
    ) -> Result<Option<RemoteEvent>, AgentClientError> {
        let mut state = self.0.lock();
        let event = state.policy.reconcile_lost(response).map_err(protocol)?;
        Ok(event.map(|event| state.event(event)))
    }

    fn extra(&self) -> Result<Map<String, Value>, AgentClientError> {
        let state = self.0.lock();
        Ok(retained::call_extra(&state.policy, &state.call, state.livekit))
    }
}

impl State {
    fn event(&self, event: DelegationUpdate) -> RemoteEvent {
        match event {
            DelegationUpdate::Pending => RemoteEvent::Pending,
            DelegationUpdate::Progress(text) => RemoteEvent::Progress { text },
            DelegationUpdate::Exact(text) => RemoteEvent::Exact { text },
            DelegationUpdate::InputRequired { question, verbatim } => {
                let verbatim_full = verbatim.as_deref() == Some(question.as_str());
                let outcome = ToolOutcome::InputRequired {
                    question: question.clone(),
                    verbatim: verbatim_full,
                };
                RemoteEvent::InputRequired {
                    input_request: self.policy.snapshot().input_request,
                    question,
                    verbatim,
                    verbatim_full,
                    outcome,
                }
            }
            DelegationUpdate::Completed(outcome) | DelegationUpdate::Failed(outcome) => {
                RemoteEvent::Outcome { outcome }
            }
            DelegationUpdate::Cancelled => RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled },
            DelegationUpdate::Follow(task) => {
                RemoteEvent::Follow { task_id: task.id, thread_id: task.context_id }
            }
        }
    }
}

/// A host callback's failure, raised as it was.
fn host_error(error: HostError) -> AgentClientError {
    AgentClientError::Provider(error.into())
}

fn unavailable(error: lk_agents_net::env::ProcessError) -> AgentClientError {
    AgentClientError::Provider(ProviderError::Local(error.into()))
}

fn protocol(error: impl std::fmt::Display) -> AgentClientError {
    AgentClientError::Protocol(error.to_string())
}

/// Invalid selection is model-facing feedback; storage failures retain their diagnostic kind.
fn input_error(error: lk_agents_abi::storage::StorageError) -> AgentClientError {
    match error {
        lk_agents_abi::storage::StorageError::Invalid(message) => {
            AgentClientError::InvalidInput(message)
        }
        error => protocol(error),
    }
}

impl lk_agents_abi::a2a::client::MessageContent for RemoteInteraction {
    fn _core_a2a_message_text(message: Message) -> String {
        message
            .parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::TextPart { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn _core_a2a_message_data(message: Message) -> Option<Map<String, Value>> {
        message.parts.into_iter().find_map(|part| match part {
            MessagePart::DataPart { data: Value::Object(data), .. } => Some(data),
            _ => None,
        })
    }
}

impl lk_agents_abi::a2a::remote::RemoteConfiguration for RemoteInteraction {
    fn _core_remote_id(endpoint: String, id: Option<String>) -> Result<String, AgentClientError> {
        configuration::remote_id(&configuration::endpoint(&endpoint)?, id)
    }

    fn _core_delegation_chain(
        metadata: Map<String, Value>,
    ) -> Result<Vec<lk_agents_abi::a2a::remote::DelegationPartition>, AgentClientError> {
        lk_agents_core::delegation::remote::partition_chain(&metadata)
    }

    fn _core_delegation_tool(
        name: String,
        description: String,
    ) -> Result<lk_agents_abi::a2a::client::AgentCard, AgentClientError> {
        Ok(lk_agents_core::a2a::client::delegation_tool(name, description)?)
    }

    fn _core_cancel_tool(name: String) -> lk_agents_abi::a2a::client::AgentCard {
        lk_agents_core::a2a::client::cancel_tool(&name)
    }

    fn _core_remote_token(
        endpoint: String,
        project_url: String,
        api_key: String,
        api_secret: String,
        token: Option<String>,
        thread_id: Option<String>,
    ) -> Result<Option<String>, AgentClientError> {
        use lk_agents_abi::a2a::auth::{AuthGrant, Credentials as CredentialsApi};

        if token.is_some() {
            return Ok(token);
        }
        let project = Project { url: project_url, api_key, api_secret };
        let Some(project) = project.serving(&endpoint)? else {
            return Ok(None);
        };
        let credentials =
            lk_agents_net::a2a_auth::Credentials::new(project.api_key, project.api_secret);
        match thread_id.filter(|id| !id.is_empty()) {
            // An endpoint call alone grants no thread access.
            Some(thread_id) => credentials.thread_grant_token(
                AuthGrant {
                    thread_ids: vec![thread_id],
                    read: true,
                    write: true,
                    ..AuthGrant::default()
                },
                DELEGATE.into(),
                3600,
            ),
            // No thread to share: enough to read the card.
            None => credentials.project_token(DELEGATE.into(), 3600),
        }
        .map(Some)
        .map_err(protocol)
    }
}

/// The identity of the tokens a remote tool mints for itself.
const DELEGATE: &str = "delegate";

/// The server's LiveKit project as the host passes it; empty values when there is none.
#[derive(Clone, PartialEq, Eq)]
struct Project {
    url: String,
    api_key: String,
    api_secret: String,
}

impl Project {
    /// The project with its URL as an HTTP origin, when `endpoint` is on that origin.
    fn serving(&self, endpoint: &str) -> Result<Option<Self>, AgentClientError> {
        let Project { url, api_key, api_secret } = self;
        if url.is_empty() || api_key.is_empty() || api_secret.is_empty() {
            return Ok(None);
        }
        let (api_key, api_secret) = (api_key.clone(), api_secret.clone());
        let Ok(mut project) = url::Url::parse(url) else {
            return Ok(None);
        };
        let scheme = match project.scheme() {
            "ws" | "http" => "http",
            "wss" | "https" => "https",
            _ => return Ok(None),
        };
        if project.set_scheme(scheme).is_err()
            || !project.username().is_empty()
            || project.password().is_some()
        {
            return Ok(None);
        }
        let destination = url::Url::parse(endpoint).map_err(protocol)?;
        if project.origin() != destination.origin() {
            return Ok(None);
        }
        Ok(Some(Project { url: project.to_string(), api_key, api_secret }))
    }
}
