use super::*;
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::text_session::{LiveTurnOrigin, LiveTurnReport};
use lk_agents_net::task::ScopedTask;

pub(super) struct CommitWork {
    turn: u64,
    pub request_id: String,
    pub working: bool,
    _task: ScopedTask,
    result: signal::Cell<Option<SessionEvent>>,
}

pub(super) async fn committed(work: &Option<CommitWork>) -> SessionEvent {
    let Some(work) = work else {
        return std::future::pending().await;
    };
    work.result.wait().await.unwrap_or_else(|_| SessionEvent::LiveTurnCommitted {
        turn: work.turn,
        result: Err(StorageError::Invalid(
            "the managed commit task ended without its receipt".into(),
        )),
    })
}

pub(crate) enum OwnerCommand {
    PrepareDialogue {
        source: String,
        messages: Vec<lk_agents_abi::ChatMessage>,
        done: crate::session::Receipt<lk_agents_abi::session::snapshot::SessionSnapshot>,
    },
    CommitDialogue {
        source: String,
        done: crate::session::Receipt<()>,
    },
    Close {
        done: crate::session::Receipt<()>,
    },
    CancelTasks {
        tasks: Vec<String>,
        done: crate::session::Receipt<Vec<String>>,
    },
}

pub(super) enum OwnerClose {
    Open,
    Requested(crate::session::Receipt<()>),
    Closed,
}

pub(crate) struct LiveControl {
    pub owner: mpsc::Sender<OwnerCommand>,
    pub cancel: CancellationToken,
    pub release: mpsc::Sender<Release>,
    pub progress: crate::agent_run::Progress,
    pub usage: signal::Cell<Usage>,
}

pub(crate) struct LiveInput {
    pub output: Option<crate::agent_run::output::Target>,
    pub request: TextSessionRequest,
    pub agents: BTreeMap<String, TextSessionAgent>,
    pub updates: lk_agents_net::outbox::Outbox<TextSessionUpdate>,
}

pub(super) struct LiveRequest {
    pub output: Option<crate::agent_run::output::Target>,
    pub request: TextSessionRequest,
    pub agents: BTreeMap<String, TextSessionAgent>,
    pub updates: lk_agents_net::outbox::Outbox<TextSessionUpdate>,
}

impl Driver {
    pub(super) fn owner_command(
        &mut self,
        command: OwnerCommand,
        effects: &mut VecDeque<SessionEffect>,
    ) {
        match command {
            OwnerCommand::CancelTasks { tasks, done } => {
                if self.session.live_dialogue_pending() {
                    done.resolve(Err(StorageError::Unavailable(
                        "an opaque write is pending".into(),
                    )));
                } else {
                    let unowned = self.session.unowned_live_tasks(&tasks);
                    effects.extend(self.session.cancel_live_tasks(&tasks, self.clock.now()));
                    done.resolve(Ok(unowned));
                }
            }
            OwnerCommand::Close { done } => match &self.owner_close {
                OwnerClose::Open => self.owner_close = OwnerClose::Requested(done),
                OwnerClose::Requested(_) | OwnerClose::Closed => done.resolve(Err(
                    StorageError::Invalid("the endpoint owner already requested close".into()),
                )),
            },
            OwnerCommand::PrepareDialogue { source, messages, done } => {
                done.resolve(self.session.prepare_live_dialogue(source, messages));
            }
            OwnerCommand::CommitDialogue { source, done } => {
                match self.session.commit_live_dialogue(&source, self.clock.now()) {
                    Ok(next) => {
                        effects.extend(next);
                        done.resolve(Ok(()));
                    }
                    Err(error) => done.resolve(Err(error)),
                }
            }
        }
    }

    pub(super) async fn admit_live(
        &mut self,
        input: LiveInput,
        effects: &mut VecDeque<SessionEffect>,
    ) {
        let LiveInput { request, agents, updates, output } = input;
        let result = (|| {
            let sink = request.session_sink.as_ref().ok_or_else(|| {
                StorageError::Invalid("a live turn requires its explicit session binding".into())
            })?;
            let origin = sink.live_origin().ok_or_else(|| {
                StorageError::Invalid("a live turn has no admitted endpoint identity".into())
            })?;
            let lk_agents_abi::ChatItem::Message(message) = &request.message else {
                return Err(StorageError::Invalid("a live turn requires a human message".into()));
            };
            // New handlers may lend fresh host context, but may not silently substitute a
            // different configured agent for an already running conversation.
            if agents.len() != self.agents.len()
                || agents.iter().any(|(name, agent)| {
                    self.agents
                        .get(name)
                        .is_none_or(|existing| existing.definition != agent.definition)
                })
            {
                return Err(StorageError::Invalid(
                    "the active live agent configuration differs".into(),
                ));
            }
            let next =
                self.session.admit_live(origin.clone(), message.clone(), self.clock.now())?;
            Ok((origin, next))
        })();
        match result {
            Ok((origin, next)) => {
                self.live_requests
                    .insert(origin.request_id, LiveRequest { request, agents, updates, output });
                effects.extend(next);
            }
            Err(error) => {
                updates
                    .send(TextSessionUpdate::Failed {
                        error: match error {
                            StorageError::Cancelled(_) => TextSessionError::Run(
                                lk_agents_abi::agent::run::RunError::Cancelled,
                            ),
                            error => TextSessionError::Storage(error),
                        },
                        input_requirements: Vec::new(),
                        provider_error: None,
                    })
                    .await;
            }
        }
    }

    pub(super) async fn send_invocation(&mut self, invocation: &str, update: TextSessionUpdate) {
        let updates = self
            .invocation_requests
            .get(invocation)
            .and_then(|request| self.live_requests.get(request))
            .map_or(&self.updates, |request| &request.updates);
        updates.send(update).await;
    }

    pub(super) async fn live_finished(
        &mut self,
        turn: u64,
        origin: LiveTurnOrigin,
        result: Box<Result<TextSessionTurn, TextSessionError>>,
        working: bool,
    ) -> Vec<SessionEffect> {
        // The failed invocation keeps its provider's original error, as outside live. A request
        // served by several invocations takes the one that kept a failure.
        let invocation = self
            .invocation_requests
            .iter()
            .filter(|(_, request)| **request == origin.request_id)
            .map(|(invocation, _)| invocation)
            .find(|invocation| self.work.get(*invocation).is_some_and(|w| w.failure.is_some()))
            .cloned();
        let provider_error = match result.as_ref() {
            Err(error) => self.provider_failure(invocation.as_deref(), error),
            Ok(_) => None,
        };
        let Some(request) = self.live_requests.get(&origin.request_id) else {
            return self.session.handle(
                SessionEvent::PersistenceFailed(StorageError::Invalid(
                    "a live result lost its admitted request owner".into(),
                )),
                self.clock.now(),
            );
        };
        let sink =
            request.request.session_sink.clone().expect("admitted live request has a binding");
        if let Ok(turn) = result.as_ref()
            && let Err(error) = sink.stage_voice(turn.state.clone())
        {
            return self.session.handle(SessionEvent::PersistenceFailed(error), self.clock.now());
        }
        let update = if !working {
            Some(match result.as_ref() {
                Ok(turn) => TextSessionUpdate::Finished(Box::new(TextSessionSettled {
                    turn: turn.clone(),
                    output: self.output.take(),
                })),
                Err(error) => TextSessionUpdate::Failed {
                    error: error.clone(),
                    input_requirements: Vec::new(),
                    provider_error,
                },
            })
        } else {
            drop(self.output.take());
            None
        };
        let request_id = origin.request_id.clone();
        // Admission captures this source's authority and queues its native report before the
        // handler can observe Finished, close its binding, or return an older state snapshot.
        let committed = match sink
            .live_turn(&self.cx, LiveTurnReport { turn, origin, result, working })
            .await
        {
            Ok(committed) => committed,
            Err(error) => {
                return self
                    .session
                    .handle(SessionEvent::PersistenceFailed(error), self.clock.now());
            }
        };
        if let Some(update) = update {
            request.updates.send(update).await;
        }
        let (done, receipt) = signal::once();
        let task = self.routing_background.tasks.scoped(
            &*self.spawner,
            "managed turn commit",
            async move {
                done.resolve(SessionEvent::LiveTurnCommitted { turn, result: committed.await });
            },
        );
        self.live_commit =
            Some(CommitWork { turn, request_id, working, _task: task, result: receipt });
        Vec::new()
    }
}

pub(super) fn attach(
    port: &CommandPort,
    request: TextSessionRequest,
    agents: BTreeMap<String, TextSessionAgent>,
    clock: Arc<dyn Clock>,
    output: Option<crate::agent_run::output::Target>,
) -> Result<(runner::Updates, runner::Control), Invalid> {
    let live = port.live.as_ref().ok_or_else(|| {
        Invalid("this active conversation does not accept live endpoint turns".into())
    })?;
    let (updates, receiver) = lk_agents_net::outbox::outbox(UPDATE_QUEUE);
    port.enqueue_live(LiveInput { request, agents, updates, output })
        .map_err(|error| Invalid(error.to_string()))?;
    Ok((
        runner::Updates {
            cancel: CancellationToken::new(),
            receiver,
            clock,
            _driver: None,
            cancel_on_drop: false,
        },
        runner::Control {
            release: live.release.clone(),
            progress: live.progress.clone(),
            usage: live.usage.clone(),
            commands: port.clone(),
        },
    ))
}
