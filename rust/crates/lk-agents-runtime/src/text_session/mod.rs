//! The request-scoped TextSession driver. The core owns routing; `agent_run::Execution`
//! executes every invocation's effects. A consumer acknowledges presentation before release.

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
use lk_agents_abi::{Clock, HostRef, WallAnchor};
use lk_agents_core::agent_run;
use lk_agents_core::text_session::{
    TextSession, TextSessionConfig, TextSessionEffect, TextSessionEvent, TextSessionInput,
};
use tokio::sync::{mpsc, oneshot, watch};

use crate::agent_run::RunSetup;
use crate::agent_run::execution::{Execution, Report, Reporter};
use crate::telemetry;

const UPDATE_QUEUE: usize = 16;

pub(super) struct Release {
    invocation_id: String,
    delivered: BTreeMap<String, String>,
}

/// What opens an invocation: the core's `InvocationStarted`.
struct InvocationStart {
    frame_id: FrameId,
    agent_name: String,
    invocation_id: String,
    context_scope: String,
    reads_context: bool,
    execution_id: String,
    history: Vec<lk_agents_abi::ChatItem>,
}

pub(super) struct Driver {
    session: TextSession,
    /// Republished whenever the turn's usage moved.
    usage: watch::Sender<Usage>,
    published: (u64, u64),
    agents: BTreeMap<String, TextSessionAgent>,
    request: TextSessionRequest,
    clock: Arc<dyn Clock>,
    /// Where the invocations' `agent_run` spans open.
    cx: telemetry::Cx,
    runtime: tokio::runtime::Handle,
    events_tx: mpsc::Sender<(String, Report)>,
    updates: mpsc::Sender<TextSessionUpdate>,
    work: BTreeMap<String, Execution>,
    /// Settled or parked invocations awaiting release, with their accepted output.
    pending: BTreeMap<String, Option<ValidationId>>,
    releases: VecDeque<Release>,
    output: Option<HostRef>,
    finished: bool,
}

impl Driver {
    async fn drive(
        mut self,
        message: lk_agents_abi::ChatMessage,
        mut events: mpsc::Receiver<(String, Report)>,
        mut progress: mpsc::Receiver<(String, u64, String)>,
        mut releases: mpsc::Receiver<Release>,
        mut cancelled: oneshot::Receiver<()>,
    ) {
        let mut cancel_seen = false;
        let mut effects: VecDeque<_> =
            self.session.handle(TextSessionEvent::Start(message), self.clock.now()).into();
        loop {
            // Before the effects: the last ones may end the stream the host reads usage beside.
            self.publish_usage();
            while let Some(effect) = effects.pop_front() {
                self.execute(effect).await;
            }
            if let Some(released) = self.take_release() {
                effects.extend(released);
                continue;
            }
            // Retain work after a core cleanup deadline: its callbacks and provider requests
            // remain owned until their real settlement, including late prepared resources.
            self.work.retain(|id, work| self.pending.contains_key(id) || !work.idle());
            if self.finished && self.work.is_empty() {
                return;
            }
            let clock = self.clock.clone();
            let wake = self.session.wake_at();
            let event = tokio::select! {
                biased;
                () = crate::agent_run::wait(&*clock, wake) => Some(TextSessionEvent::Wake),
                _ = &mut cancelled, if !cancel_seen => {
                    cancel_seen = true;
                    Some(TextSessionEvent::Cancel)
                }
                Some(release) = releases.recv() => {
                    self.releases.push_back(release);
                    None
                }
                Some((invocation_id, operation, message)) = progress.recv() => {
                    self.progress_event(invocation_id, operation, message)
                }
                Some((invocation_id, report)) = events.recv() => {
                    self.report_event(invocation_id, report)
                }
                _ = settle_work(&mut self.work) => None,
            };
            let Some(event) = event else { continue };
            effects.extend(self.session.handle(event, self.clock.now()));
        }
    }

    /// Republishes the turn's usage when it moved.
    fn publish_usage(&mut self) {
        if self.session.usage_revision() != self.published {
            self.published = self.session.usage_revision();
            self.usage.send_replace(self.session.usage());
        }
    }

    /// The next release's effects; a release whose invocation still has work in flight waits.
    fn take_release(&mut self) -> Option<Vec<TextSessionEffect>> {
        let release = self.releases.pop_front()?;
        if self.work.get(&release.invocation_id).is_some_and(|w| !w.idle()) {
            self.releases.push_front(release);
            return None;
        }
        Some(self.release(release))
    }

    /// A tool body's progress update, while its call is still live.
    fn progress_event(
        &self,
        invocation_id: String,
        operation: u64,
        message: String,
    ) -> Option<TextSessionEvent> {
        let event = self.work.get(&invocation_id)?.progress(operation, message)?;
        Some(TextSessionEvent::Run { invocation_id, event })
    }

    /// An invocation's report as the session's event; a validated output keeps its value.
    fn report_event(&mut self, invocation_id: String, report: Report) -> Option<TextSessionEvent> {
        let (event, output) = self.work.get_mut(&invocation_id)?.report(report);
        Some(match event {
            agent_run::AgentRunEvent::OutputValidated { validation, verdict } => {
                TextSessionEvent::OutputValidated { invocation_id, validation, verdict, output }
            }
            event => TextSessionEvent::Run { invocation_id, event },
        })
    }

    async fn execute(&mut self, effect: TextSessionEffect) {
        match effect {
            TextSessionEffect::InvocationStarted {
                frame_id,
                agent_name,
                invocation_id,
                context_scope,
                reads_context,
                execution_id,
                history,
            } => self.start_invocation(InvocationStart {
                frame_id,
                agent_name,
                invocation_id,
                context_scope,
                reads_context,
                execution_id,
                history,
            }),
            TextSessionEffect::Run { invocation_id, effect } => {
                if let Some(work) = self.work.get_mut(&invocation_id)
                    && let Some(RunUpdate::IncompleteCleanup(pending)) = work.execute(effect).await
                {
                    let _ = self.updates.send(TextSessionUpdate::IncompleteCleanup(pending)).await;
                }
            }
            TextSessionEffect::Emit { sequence, frame_id, agent_name, invocation_id, event } => {
                let _ = self
                    .updates
                    .send(TextSessionUpdate::Stream {
                        sequence,
                        frame_id,
                        agent_name,
                        invocation_id,
                        event,
                    })
                    .await;
            }
            TextSessionEffect::InvocationSettled { invocation_id, result, .. } => {
                let output = match result.outcome {
                    RunOutcome::Completed { output: Output::Validated(id) } => Some(id),
                    _ => None,
                };
                self.pending.insert(invocation_id.clone(), output);
                let _ =
                    self.updates.send(TextSessionUpdate::InvocationSettled(invocation_id)).await;
            }
            TextSessionEffect::InvocationParked { invocation_id, .. } => {
                self.pending.insert(invocation_id.clone(), None);
                let _ = self.updates.send(TextSessionUpdate::Parked(invocation_id)).await;
            }
            TextSessionEffect::RoutingRefused { .. } => {} // The core records the typed refusal in the original call.
            TextSessionEffect::Finished { invocation_id, result, input_requirements } => {
                self.finish(invocation_id, result, input_requirements).await
            }
        }
    }

    /// Opens the invocation's `agent_run` span and its execution.
    fn start_invocation(&mut self, start: InvocationStart) {
        let InvocationStart {
            frame_id,
            agent_name,
            invocation_id,
            context_scope,
            reads_context,
            execution_id,
            history,
        } = start;
        let agent = &self.agents[&agent_name];
        let span = telemetry::start(
            &self.cx,
            telemetry::AgentRun {
                operation: telemetry::GenAiOperation::InvokeAgent,
                agent_name: agent_name.clone(),
            },
        );
        let application_context =
            reads_context.then(|| self.request.contexts[&context_scope].clone());
        let entry = Invocation {
            operation: 0,
            speech: None,
            invocation_id: invocation_id.clone(),
            agent_name,
            execution_id: Some(execution_id),
            frame_id: Some(frame_id),
            application_context,
            context_scope: Some(context_scope),
            history,
            function_call: None,
            approval: None,
            host_context: Some(agent.host_context.clone()),
        };
        let setup = RunSetup {
            tools_host: agent.tools_host.clone(),
            llm: agent.llm.clone(),
            host: agent.host.clone(),
            clock: self.clock.clone(),
            conn_options: self.request.conn_options,
            entry,
        };
        let reporter = Reporter { id: invocation_id.clone(), tx: self.events_tx.clone() };
        let execution = Execution::new(setup, self.runtime.clone(), reporter, span);
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
        let _ = self.updates.send(update).await;
    }

    /// A model failure's original provider error, kept by the invocation that failed.
    fn provider_failure(
        &mut self,
        invocation_id: Option<&str>,
        error: &TextSessionError,
    ) -> Option<lk_agents_abi::errors::ProviderError> {
        if !matches!(error, TextSessionError::Run(RunError::Model(_))) {
            return None;
        }
        self.work.get_mut(invocation_id?)?.failure.take()
    }

    fn release(&mut self, release: Release) -> Vec<TextSessionEffect> {
        let Some(output) = self.pending.remove(&release.invocation_id) else {
            return Vec::new();
        };
        self.output = output.and_then(|id| {
            self.work.get_mut(&release.invocation_id).and_then(|w| w.outputs.remove(&id))
        });
        self.session.handle(
            TextSessionEvent::InvocationReleased {
                invocation_id: release.invocation_id,
                delivered: release.delivered,
            },
            self.clock.now(),
        )
    }
}

async fn settle_work(work: &mut BTreeMap<String, Execution>) {
    let futures: Vec<_> = work
        .values_mut()
        .filter(|w| !w.invoker.is_idle())
        .map(|w| Box::pin(w.invoker.settled()))
        .collect();
    if futures.is_empty() {
        std::future::pending::<()>().await;
    } else {
        futures_util::future::select_all(futures).await;
    }
}
