mod input;
pub(crate) mod live;
mod runner;
pub use runner::{Runner, Updates};

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use lk_agents_abi::agent::Invocation;
use lk_agents_abi::agent::run::{
    Output, RunError, RunInputRequirement, RunOutcome, RunUpdate, ValidationId,
};
use lk_agents_abi::text_session::{
    FrameId, TextSessionAgent, TextSessionError, TextSessionRequest, TextSessionSettled,
    TextSessionTurn, TextSessionUpdate,
};
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{Clock, HostRef, Spawner, WallAnchor};
use lk_agents_core::agent_run;
use lk_agents_core::session::{Session, SessionConfig, SessionEffect, SessionEvent, SessionInput};
use lk_agents_net::signal;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent_run::RunSetup;
use crate::agent_run::execution::{Execution, Report, Reporter};
use crate::session::{CommandPort, SessionCommand};
use crate::telemetry;

const UPDATE_QUEUE: usize = 16;

pub(super) struct Release {
    invocation_id: String,
    delivered: BTreeMap<String, String>,
}

enum DriverEvent {
    Owner(live::OwnerCommand),
    Core(Option<Box<SessionEvent>>),
    Routed(Box<SessionEvent>),
    Command(Box<SessionCommand>),
    Committed(Box<SessionEvent>),
}

/// What opens an invocation: the core's `InvocationStarted`.
struct InvocationStart {
    origin: Option<lk_agents_abi::text_session::LiveTurnOrigin>,
    frame_id: FrameId,
    agent_name: String,
    invocation_id: String,
    execution_id: String,
    history: Vec<lk_agents_abi::ChatItem>,
}

pub(super) struct DriverChannels {
    events: mpsc::Receiver<(String, Report)>,
    releases: mpsc::Receiver<Release>,
    commands: mpsc::Receiver<SessionCommand>,
    owner_commands: mpsc::Receiver<live::OwnerCommand>,
}

pub(super) struct Driver {
    owner_close: live::OwnerClose,
    /// The execution owns this weak binding's life, independently of the saved view.
    _bound_commands: Arc<dyn lk_agents_abi::session::SessionCommands>,
    delegation: Option<lk_agents_abi::delegation::DelegationOptions>,
    routing: Option<input::RoutingWork>,
    routing_background: lk_agents_net::owned::Background,
    session: Session,
    /// Republished whenever the turn's usage moved.
    usage: signal::Producer<Usage>,
    published: (u64, u64),
    agents: BTreeMap<String, TextSessionAgent>,
    request: TextSessionRequest,
    clock: Arc<dyn Clock>,
    /// Where the invocations' `agent_run` spans open.
    cx: telemetry::Cx,
    spawner: Arc<dyn Spawner>,
    events_tx: mpsc::Sender<(String, Report)>,
    progress: crate::agent_run::Progress,
    output_target: Option<crate::agent_run::output::Target>,
    /// Dropping the receiver cancels the session, so nobody needs what follows.
    updates: lk_agents_net::outbox::Outbox<TextSessionUpdate>,
    work: BTreeMap<String, Execution>,
    live_requests: BTreeMap<String, live::LiveRequest>,
    invocation_requests: BTreeMap<String, String>,
    live_commit: Option<live::CommitWork>,
    /// Settled or parked invocations awaiting release, with their accepted output.
    pending: BTreeMap<String, Option<ValidationId>>,
    releases: VecDeque<Release>,
    output: Option<HostRef>,
    finished: bool,
    checkpoint_port: Option<Arc<dyn lk_agents_abi::text_session::SessionCheckpointPort>>,
}

impl Driver {
    async fn drive(
        mut self,
        message: lk_agents_abi::ChatMessage,
        channels: DriverChannels,
        cancelled: CancellationToken,
    ) {
        let DriverChannels { mut events, mut releases, mut commands, mut owner_commands } =
            channels;
        let mut cancel_seen = false;
        let mut effects: VecDeque<_> =
            self.session.handle(SessionEvent::Start(message), self.clock.now()).into();
        loop {
            // Before the effects: the last ones may end the stream the host reads usage beside.
            self.publish_usage();
            while let Some(effect) = effects.pop_front() {
                if let SessionEffect::Checkpoint(state) = effect {
                    if let Err(error) = self.publish_snapshot(*state).await {
                        effects = self
                            .session
                            .handle(SessionEvent::PersistenceFailed(error), self.clock.now())
                            .into();
                    }
                } else {
                    effects.extend(self.execute(effect).await);
                }
            }
            if !self.session.live_dialogue_pending()
                && let Some(released) = self.take_release()
            {
                effects.extend(released);
                continue;
            }
            // Retain work after a core cleanup deadline: its callbacks and provider requests
            // remain owned until their real settlement, including late prepared resources.
            self.work.retain(|id, work| self.pending.contains_key(id) || !work.idle());
            // A canceled owner (its endpoint failed or lost its lease) settles once its bodies
            // are joined; requests it can no longer answer fail below.
            if (self.live_requests.is_empty() || cancel_seen)
                && self.session.live_idle()
                && !matches!(self.owner_close, live::OwnerClose::Open)
            {
                self.finished = true;
            }
            if self.finished && self.work.is_empty() && self.routing_background.tasks.is_empty() {
                if let live::OwnerClose::Requested(done) =
                    std::mem::replace(&mut self.owner_close, live::OwnerClose::Closed)
                {
                    done.resolve(Ok(()));
                }
                for (_, request) in std::mem::take(&mut self.live_requests) {
                    request
                        .updates
                        .send(TextSessionUpdate::Failed {
                            error: TextSessionError::Run(
                                lk_agents_abi::agent::run::RunError::Cancelled,
                            ),
                            input_requirements: Vec::new(),
                            provider_error: None,
                        })
                        .await;
                }
                return;
            }
            let clock = self.clock.clone();
            let wake = self.session.wake_at();
            // Prepare runs only after preceding writes/effects complete. The priority lane is
            // the same owner's ACK/close path; normal bounded channels retain their inputs.
            let paused = self.session.live_dialogue_pending();
            let ready = tokio::select! {
                biased;
                Some(command) = owner_commands.recv() => DriverEvent::Owner(command),
                () = crate::agent_run::wait(&*clock, wake), if !paused => DriverEvent::Core(Some(Box::new(SessionEvent::Wake))),
                () = cancelled.cancelled(), if !cancel_seen => {
                    cancel_seen = true;
                    self.owner_close = live::OwnerClose::Closed;
                    DriverEvent::Core(Some(Box::new(SessionEvent::Cancel)))
                }
                Some(release) = releases.recv(), if !paused => {
                    self.releases.push_back(release);
                    DriverEvent::Core(None)
                }
                Some(command) = commands.recv(), if !paused => DriverEvent::Command(Box::new(command)),
                Some((invocation_id, report)) = events.recv(), if !paused => {
                    DriverEvent::Core(self.report_event(invocation_id, report).map(Box::new))
                }
                event = input::routed(&self.routing), if !paused => DriverEvent::Routed(Box::new(event)),
                event = live::committed(&self.live_commit), if !paused => DriverEvent::Committed(Box::new(event)),
                settled = settle_work(&mut self.work), if !paused => DriverEvent::Core(settled.map(|(invocation_id, event)| Box::new(SessionEvent::Run { invocation_id, event }))),
                _ = self.routing_background.tasks.wait_empty(&*clock, None), if self.finished && !paused => DriverEvent::Core(None),
            };
            let event = match ready {
                DriverEvent::Owner(command) => {
                    self.owner_command(command, &mut effects);
                    continue;
                }
                DriverEvent::Command(command) => {
                    self.command(*command, &mut effects).await;
                    continue;
                }
                DriverEvent::Core(event) => event.map(|event| *event),
                DriverEvent::Routed(event) => {
                    self.routing.take();
                    Some(*event)
                }
                DriverEvent::Committed(event) => {
                    if let Some(work) = self.live_commit.take()
                        && !work.working
                    {
                        self.live_requests.remove(&work.request_id);
                    }
                    Some(*event)
                }
            };
            let Some(event) = event else { continue };
            effects.extend(self.session.handle(event, self.clock.now()));
            // A canceled owner (its endpoint failed or lost its lease) settles once its bodies
            // are joined; requests it can no longer answer fail below.
            if (self.live_requests.is_empty() || cancel_seen)
                && self.session.live_idle()
                && !matches!(self.owner_close, live::OwnerClose::Open)
            {
                self.finished = true;
            }
        }
    }

    async fn command(&mut self, command: SessionCommand, effects: &mut VecDeque<SessionEffect>) {
        let (command, done) = match command {
            SessionCommand::LiveInput(input) => {
                self.admit_live(*input, effects).await;
                return;
            }
            SessionCommand::RoutingContext {
                remote_id,
                endpoint,
                proposed,
                selected_task,
                offered,
                done,
            } => {
                done.resolve(lk_agents_core::delegation::remote::routing::request(
                    &self.session.snapshot(),
                    &remote_id,
                    &endpoint,
                    proposed,
                    selected_task.as_deref(),
                    &offered,
                ));
                return;
            }
            SessionCommand::AdmitCall { request, verdict, usage, candidate, drivers, done } => {
                self.session.routing_usage(&usage);
                let state = self.session.snapshot();
                done.resolve(
                    lk_agents_core::delegation::remote::routing::admit(&state, &request, verdict)
                        .and_then(|admission| {
                            drivers.admit(
                                &state,
                                &request.remote_id,
                                &request.endpoint,
                                admission,
                                candidate,
                            )
                        }),
                );
                return;
            }
            SessionCommand::RemoteDriver { remote_id, endpoint, task, drivers, done } => {
                done.resolve(drivers.get(&self.session.snapshot(), &remote_id, &endpoint, &task));
                return;
            }
            SessionCommand::CancelRemote {
                target,
                scope,
                candidate,
                retry_after,
                drivers,
                done,
            } => {
                let result = self
                    .session
                    .cancel_required(&target.remote_id, &target.endpoint, &target.task)
                    .and_then(|_| {
                        crate::session::admit_cancellation(
                            &self.session.snapshot(),
                            &drivers,
                            target,
                            scope.as_ref(),
                            candidate,
                            retry_after.as_deref(),
                        )
                    });
                done.resolve(result);
                return;
            }
            SessionCommand::AdmitInput { request, verdict, done } => {
                let result = self.session.admit_input(*request, verdict);
                let saved = self.checkpoint_command(effects).await;
                done.resolve(saved.and(result));
                return;
            }
            SessionCommand::Flush { done } => {
                let result = self.checkpoint_command(effects).await.and_then(|state| {
                    state.ok_or_else(|| {
                        lk_agents_abi::storage::StorageError::Invalid(
                            "explicit checkpoints require a bound conversation".into(),
                        )
                    })
                });
                done.resolve(result);
                return;
            }
            SessionCommand::Input { command, done } => {
                let result = self.session.input(command);
                let persisted = self.checkpoint_command(effects).await;
                done.resolve(persisted.and(result));
                return;
            }
            SessionCommand::Recover { remote_id, endpoint, target, done } => {
                let state = self.session.snapshot();
                done.resolve(
                    lk_agents_core::session::remote::state(&state, &remote_id, &endpoint).and_then(
                        |remote| {
                            remote
                                .map(|remote| {
                                    lk_agents_core::session::remote::recover(remote, &target)
                                })
                                .transpose()
                                .map(Option::flatten)
                        },
                    ),
                );
                return;
            }
            SessionCommand::DefaultContext { remote_id, endpoint, done } => {
                done.resolve(self.session.remote_context(&remote_id, &endpoint));
                return;
            }
            SessionCommand::Apply { command, done } => (command, done),
        };
        let result = self.session.remote(*command);
        // A valid observation can be journaled even when its projection is refused. Persist
        // the evidence before returning either acknowledgement or refusal to the client.
        let persisted = self.checkpoint_command(effects).await;
        done.resolve(persisted.and(result));
    }

    async fn checkpoint_command(
        &mut self,
        effects: &mut VecDeque<SessionEffect>,
    ) -> Result<Option<serde_json::Value>, lk_agents_abi::storage::StorageError> {
        if self.request.session_sink.is_some() {
            return match self.publish_snapshot(self.session.snapshot()).await {
                Ok(state) => Ok(Some(state)),
                Err(error) => {
                    effects.extend(
                        self.session.handle(
                            SessionEvent::PersistenceFailed(error.clone()),
                            self.clock.now(),
                        ),
                    );
                    Err(error)
                }
            };
        }
        Ok(None)
    }

    async fn publish_snapshot(
        &mut self,
        state: lk_agents_abi::text_session::SessionSnapshot,
    ) -> Result<serde_json::Value, lk_agents_abi::storage::StorageError> {
        if let Some(port) = &self.checkpoint_port {
            let committed = port.publish(state).await?;
            for request in self.live_requests.values() {
                if let Some(sink) = &request.request.session_sink {
                    sink.acknowledge_checkpoint(committed.clone())?;
                }
            }
            return Ok(committed);
        }
        match &self.request.session_sink {
            Some(sink) => sink.publish(&self.cx, state).await,
            None => lk_agents_storage::session_state::snapshot_value(state),
        }
    }

    /// Republishes the turn's usage when it moved.
    fn publish_usage(&mut self) {
        if self.session.usage_revision() != self.published {
            self.published = self.session.usage_revision();
            self.usage.set(self.session.usage());
        }
    }

    /// The next release's effects; a release whose invocation still has work in flight waits.
    fn take_release(&mut self) -> Option<Vec<SessionEffect>> {
        let release = self.releases.pop_front()?;
        if !self.session.allows_live_release(&release.invocation_id)
            && self.work.get(&release.invocation_id).is_some_and(|w| !w.idle())
        {
            self.releases.push_front(release);
            return None;
        }
        Some(self.release(release))
    }

    /// An invocation's report as the session's event; a validated output keeps its value.
    fn report_event(&mut self, invocation_id: String, report: Report) -> Option<SessionEvent> {
        let (event, output) = self.work.get_mut(&invocation_id)?.report(report);
        Some(match event {
            agent_run::AgentRunEvent::OutputValidated { validation, verdict } => {
                SessionEvent::OutputValidated { invocation_id, validation, verdict, output }
            }
            event => SessionEvent::Run { invocation_id, event },
        })
    }

    async fn execute(&mut self, effect: SessionEffect) -> Vec<SessionEffect> {
        match effect {
            SessionEffect::LiveTurnFinished { turn, origin, result, working } => {
                return self.live_finished(turn, origin, result, working).await;
            }
            SessionEffect::Checkpoint(_) => {
                unreachable!("checkpoints execute before dependent effects")
            }
            SessionEffect::ToolOutputCheckpoint { invocation_id, output, state } => {
                let result = self.publish_snapshot(*state).await.map(|_| ());
                return self.session.handle(
                    SessionEvent::ToolOutputCommitted { invocation_id, output, result },
                    self.clock.now(),
                );
            }
            SessionEffect::RouteInput(input) => self.route(input),
            SessionEffect::InvocationStarted {
                origin,
                frame_id,
                agent_name,
                invocation_id,
                execution_id,
                history,
            } => self.start_invocation(InvocationStart {
                origin,
                frame_id,
                agent_name,
                invocation_id,
                execution_id,
                history,
            }),
            SessionEffect::Run { invocation_id, effect } => {
                if let Some(work) = self.work.get_mut(&invocation_id)
                    && let Some(RunUpdate::IncompleteCleanup(pending)) = work.execute(effect).await
                {
                    self.updates.send(TextSessionUpdate::IncompleteCleanup(pending)).await;
                }
            }
            SessionEffect::Emit { sequence, frame_id, agent_name, invocation_id, event } => {
                if let lk_agents_abi::agent::run::StreamEvent::Progress { message, silent, .. } =
                    &event
                    && let Some(target) =
                        self.work.get(&invocation_id).and_then(|work| work.output.as_ref())
                {
                    target.progress(message.clone(), *silent).await;
                }
                self.send_invocation(
                    &invocation_id,
                    TextSessionUpdate::Stream {
                        sequence,
                        frame_id,
                        agent_name,
                        invocation_id: invocation_id.clone(),
                        event,
                    },
                )
                .await;
            }
            SessionEffect::InvocationSettled { invocation_id, result, .. } => {
                let output = match result.outcome {
                    RunOutcome::Completed { output: Output::Validated(id) } => Some(id),
                    _ => None,
                };
                self.pending.insert(invocation_id.clone(), output);
                self.send_invocation(
                    &invocation_id,
                    TextSessionUpdate::InvocationSettled(invocation_id.clone()),
                )
                .await;
            }
            SessionEffect::InvocationParked { invocation_id, .. } => {
                self.pending.insert(invocation_id.clone(), None);
                self.send_invocation(
                    &invocation_id,
                    TextSessionUpdate::Parked(invocation_id.clone()),
                )
                .await;
            }
            SessionEffect::RoutingRefused { .. } => {} // The core records the typed refusal in the original call.
            SessionEffect::Finished { invocation_id, result, input_requirements } => {
                self.finish(invocation_id, result, input_requirements).await
            }
        }
        Vec::new()
    }

    /// Opens the invocation's `agent_run` span and its execution.
    fn start_invocation(&mut self, start: InvocationStart) {
        let InvocationStart { origin, frame_id, agent_name, invocation_id, execution_id, history } =
            start;
        if let Some(origin) = &origin {
            self.invocation_requests.insert(invocation_id.clone(), origin.request_id.clone());
        }
        let agent = origin
            .as_ref()
            .and_then(|origin| self.live_requests.get(&origin.request_id))
            .and_then(|request| request.agents.get(&agent_name))
            .unwrap_or(&self.agents[&agent_name]);
        let span = telemetry::start(
            &self.cx,
            telemetry::AgentRun {
                operation: telemetry::GenAiOperation::InvokeAgent,
                agent_name: agent_name.clone(),
            },
        );
        let entry = Invocation {
            operation: 0,
            speech: None,
            invocation_id: invocation_id.clone(),
            agent_name,
            execution_id: Some(execution_id),
            frame_id: Some(frame_id),
            history,
            function_call: None,
            approval: None,
            host_context: Some(agent.host_context.clone()),
        };
        let setup = RunSetup {
            input_answers: Default::default(),
            tools_host: agent.tools_host.clone(),
            llm: agent.llm.clone(),
            host: agent.host.clone(),
            clock: self.clock.clone(),
            conn_options: self.request.conn_options,
            entry,
        };
        let reporter = Reporter { id: invocation_id.clone(), tx: self.events_tx.clone() };
        let mut execution = Execution::new(
            setup,
            self.spawner.clone(),
            reporter,
            span,
            self.progress.for_invocation(invocation_id.clone()),
        );
        execution.output = origin
            .as_ref()
            .and_then(|origin| self.live_requests.get(&origin.request_id))
            .and_then(|request| request.output.clone())
            .or_else(|| self.output_target.clone());
        self.work.insert(invocation_id, execution);
    }

    /// The turn's end: pending releases are dropped before the update is sent.
    async fn finish(
        &mut self,
        invocation_id: Option<String>,
        result: Result<TextSessionTurn, TextSessionError>,
        input_requirements: Vec<RunInputRequirement>,
    ) {
        self.finished = true;
        self.routing.take();
        self.pending.clear();
        self.releases.clear();
        let update = match result {
            Ok(turn) => TextSessionUpdate::Finished(Box::new(TextSessionSettled {
                turn,
                output: self.output.take(),
            })),
            Err(error) => {
                let provider_error = self.provider_failure(invocation_id.as_deref(), &error);
                TextSessionUpdate::Failed { error, input_requirements, provider_error }
            }
        };
        self.updates.send(update).await;
    }

    /// A model failure's original provider error, kept by the invocation that failed.
    pub(super) fn provider_failure(
        &mut self,
        invocation_id: Option<&str>,
        error: &TextSessionError,
    ) -> Option<lk_agents_abi::errors::ProviderError> {
        if !matches!(error, TextSessionError::Run(RunError::Model { .. })) {
            return None;
        }
        // Every request a failed invocation served raises the same provider error.
        self.work.get(invocation_id?)?.failure.clone()
    }

    fn release(&mut self, release: Release) -> Vec<SessionEffect> {
        let Some(output) = self.pending.remove(&release.invocation_id) else {
            return Vec::new();
        };
        self.output = output.and_then(|id| {
            self.work.get_mut(&release.invocation_id).and_then(|w| w.outputs.remove(&id))
        });
        self.session.handle(
            SessionEvent::InvocationReleased {
                invocation_id: release.invocation_id,
                delivered: release.delivered,
            },
            self.clock.now(),
        )
    }
}

async fn settle_work(
    work: &mut BTreeMap<String, Execution>,
) -> Option<(String, agent_run::AgentRunEvent)> {
    let futures: Vec<_> = work
        .iter_mut()
        .filter(|(_, work)| !work.idle())
        .map(|(id, work)| {
            Box::pin(async move { work.settled().await.map(|event| (id.clone(), event)) })
        })
        .collect();
    if futures.is_empty() {
        std::future::pending().await
    } else {
        futures_util::future::select_all(futures).await.0
    }
}
