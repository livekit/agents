//! The request-scoped TextSession driver. The core owns routing; `agent_run::Execution`
//! executes every invocation's effects. A consumer acknowledges presentation before release.

mod runner;
pub use runner::{Runner, Updates};

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::agent::Invocation;
use lk_agents_abi::agent::run::{Output, RunError, RunOutcome, RunUpdate, ValidationId};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::text_session::{
    TextSessionAgent, TextSessionError, TextSessionRequest, TextSessionSettled, TextSessionUpdate,
};
use lk_agents_abi::{Clock, HostRef, WallAnchor};
use lk_agents_core::agent_run;
use lk_agents_core::text_session::{
    Effect, Event, TextSession, TextSessionConfig, TextSessionInput,
};
use tokio::sync::{mpsc, oneshot};

use crate::agent_run::Ports;
use crate::agent_run::execution::{Execution, Report, Reporter};

const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
const UPDATE_QUEUE: usize = 16;

pub(super) struct Release {
    invocation_id: String,
    delivered: BTreeMap<String, String>,
}

struct PendingDelivery {
    history: Vec<lk_agents_abi::ChatItem>,
    output: Option<ValidationId>,
}

pub(super) struct Driver {
    session: TextSession,
    agents: BTreeMap<String, TextSessionAgent>,
    request: TextSessionRequest,
    clock: Arc<dyn Clock>,
    runtime: tokio::runtime::Handle,
    events_tx: mpsc::Sender<(String, Report)>,
    updates: mpsc::Sender<Result<TextSessionUpdate, ProviderError>>,
    work: BTreeMap<String, Execution>,
    pending: BTreeMap<String, PendingDelivery>,
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
        let mut effects: VecDeque<_> = self
            .session
            .handle(Event::Start(message), self.clock.now())
            .into();
        loop {
            while let Some(effect) = effects.pop_front() {
                self.execute(effect).await;
            }
            if let Some(release) = self.releases.pop_front() {
                if self
                    .work
                    .get(&release.invocation_id)
                    .is_some_and(|w| !w.idle())
                {
                    self.releases.push_front(release);
                } else {
                    effects.extend(self.release(release));
                    continue;
                }
            }
            // Retain work after a core cleanup deadline: its callbacks and provider requests
            // remain owned until their real settlement, including late prepared resources.
            self.work
                .retain(|id, work| self.pending.contains_key(id) || !work.idle());
            if self.finished && self.work.is_empty() {
                return;
            }
            let clock = self.clock.clone();
            let wake = self.session.wake_at();
            let event = tokio::select! {
                biased;
                () = crate::agent_run::wait(&*clock, wake) => Event::Wake,
                _ = &mut cancelled, if !cancel_seen => { cancel_seen = true; Event::Cancel }
                Some(release) = releases.recv() => { self.releases.push_back(release); continue; }
                Some((invocation_id, operation, message)) = progress.recv() => {
                    let Some(event) = self.work.get(&invocation_id).and_then(|w| w.progress(operation, message)) else { continue };
                    Event::Run { invocation_id, event }
                }
                Some((invocation_id, report)) = events.recv() => {
                    let Some(work) = self.work.get_mut(&invocation_id) else { continue };
                    let (event, output) = work.report(report);
                    match event {
                        agent_run::Event::OutputValidated { validation, verdict } => {
                            Event::OutputValidated { invocation_id, validation, verdict, output }
                        }
                        event => Event::Run { invocation_id, event },
                    }
                }
                _ = settle_work(&mut self.work) => continue,
            };
            effects.extend(self.session.handle(event, self.clock.now()));
        }
    }

    async fn execute(&mut self, effect: Effect) {
        match effect {
            Effect::InvocationStarted {
                frame_id,
                agent_name,
                invocation_id,
                context_scope,
                execution_id,
                history,
            } => {
                let agent = &self.agents[&agent_name];
                let ports = Ports {
                    llm: agent.llm.clone(),
                    host: agent.host.clone(),
                    clock: self.clock.clone(),
                    conn_options: self.request.conn_options,
                    entry: Invocation {
                        operation: 0,
                        invocation_id: invocation_id.clone(),
                        agent_name,
                        execution_id: Some(execution_id),
                        frame_id: Some(frame_id.clone()),
                        application_context: if self.request.separate_contexts
                            && context_scope == frame_id.value
                        {
                            None
                        } else {
                            Some(self.request.contexts[&context_scope].clone())
                        },
                        context_scope: Some(context_scope),
                        history,
                        function_call: None,
                        approval: None,
                        host_context: Some(agent.host_context.clone()),
                    },
                };
                self.work.insert(
                    invocation_id.clone(),
                    Execution::new(
                        ports,
                        self.runtime.clone(),
                        Reporter {
                            id: invocation_id,
                            tx: self.events_tx.clone(),
                        },
                    ),
                );
            }
            Effect::Run {
                invocation_id,
                effect,
            } => {
                if let Some(work) = self.work.get_mut(&invocation_id)
                    && let Some(RunUpdate::IncompleteCleanup(pending)) = work.execute(effect).await
                {
                    let _ = self
                        .updates
                        .send(Ok(TextSessionUpdate::IncompleteCleanup(pending)))
                        .await;
                }
            }
            Effect::Emit {
                sequence,
                frame_id,
                agent_name,
                invocation_id,
                event,
            } => {
                let _ = self
                    .updates
                    .send(Ok(TextSessionUpdate::Stream {
                        sequence,
                        frame_id,
                        agent_name,
                        invocation_id,
                        event,
                    }))
                    .await;
            }
            Effect::InvocationSettled {
                invocation_id,
                result,
                ..
            } => {
                let usage = result.usage;
                let output = match result.outcome {
                    RunOutcome::Completed {
                        output: Output::Validated(id),
                    } => Some(id),
                    _ => None,
                };
                self.pending.insert(
                    invocation_id.clone(),
                    PendingDelivery {
                        history: result.new_messages,
                        output,
                    },
                );
                let _ = self
                    .updates
                    .send(Ok(TextSessionUpdate::InvocationSettled {
                        invocation_id,
                        usage,
                    }))
                    .await;
            }
            Effect::InvocationParked {
                invocation_id,
                usage,
                history,
                ..
            } => {
                self.pending.insert(
                    invocation_id.clone(),
                    PendingDelivery {
                        history,
                        output: None,
                    },
                );
                let _ = self
                    .updates
                    .send(Ok(TextSessionUpdate::Parked {
                        invocation_id,
                        usage,
                    }))
                    .await;
            }
            Effect::RoutingRefused { .. } => {} // The core records the typed refusal in the original call.
            Effect::Finished(result) => {
                self.finished = true;
                self.pending.clear();
                self.releases.clear();
                if matches!(&result, Err(TextSessionError::Run(RunError::Model(_))))
                    && let Some(error) = self.work.values_mut().find_map(|work| work.failure.take())
                {
                    let _ = self.updates.send(Err(error)).await;
                    return;
                }
                let update = match result {
                    Ok(turn) => TextSessionUpdate::Finished(TextSessionSettled {
                        turn,
                        output: self.output.take(),
                    }),
                    Err(error) => TextSessionUpdate::Failed(error),
                };
                let _ = self.updates.send(Ok(update)).await;
            }
        }
    }

    fn release(&mut self, release: Release) -> Vec<Effect> {
        let Some(pending) = self.pending.remove(&release.invocation_id) else {
            return Vec::new();
        };
        let now = self.clock.now();
        let mut effects = Vec::new();
        for item in &pending.history {
            if let lk_agents_abi::ChatItem::Message(message) = item
                && let Some(text) = release.delivered.get(&message.id)
            {
                let mut message = message.clone();
                message.interrupted |= message.raw_text_content().as_deref() != Some(text.as_str());
                message.content = vec![lk_agents_abi::ChatContent::Text(text.clone())];
                effects.extend(self.session.handle(
                    Event::DialogueDelivered {
                        invocation_id: release.invocation_id.clone(),
                        message,
                    },
                    now,
                ));
            }
        }
        self.output = pending.output.and_then(|id| {
            self.work
                .get_mut(&release.invocation_id)
                .and_then(|w| w.outputs.remove(&id))
        });
        effects.extend(self.session.handle(
            Event::InvocationReleased {
                invocation_id: release.invocation_id,
            },
            now,
        ));
        effects
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
