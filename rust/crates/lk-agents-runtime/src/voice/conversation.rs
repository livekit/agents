//! Execute shared conversation commands under a fresh scope while audio keeps running.

use std::sync::Arc;

use super::{Now, VoiceEvent, VoiceRuntime};
use crate::session::{Receipt, SessionCommand};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::session::{
    InputAdmission, InputRequestRef, InputRoute, InputRoutingRequest, SessionCheckpoint,
    SessionInputCommand, snapshot::SessionSnapshot,
};
use lk_agents_abi::storage::StorageError;
use lk_agents_core::session::{input, remote};
use lk_agents_net::{signal, task::ScopedTask};
use serde_json::Value;

/// The owner keeps the caller's receipt until the admitted write settles. Acquiring a scope
/// precedes every read or mutation, including when the previous speech released its lease.
pub(super) enum Work {
    Opening {
        command: SessionCommand,
        read: Vec<lk_agents_abi::ChatItem>,
        _task: ScopedTask,
        result: signal::Cell<Option<Result<Arc<dyn SessionCheckpoint>, StorageError>>>,
    },
    Writing {
        write: ConversationWrite,
        _task: ScopedTask,
        result: signal::Cell<Option<Result<Value, StorageError>>>,
    },
    /// The caller has ended; the admitted write and its scope still join session cleanup.
    Closing { _task: ScopedTask },
}

/// Each write retains its original command and correctly typed caller until the receipt.
pub(super) enum ConversationWrite {
    Flush {
        done: Receipt<Value>,
    },
    Input {
        command: SessionInputCommand,
        done: Receipt<()>,
        outcome: Result<(), StorageError>,
    },
    Remote {
        command: Box<lk_agents_abi::delegation::remote::RemoteCommand>,
        done: Receipt<Option<InputRequestRef>>,
        outcome: Result<Option<InputRequestRef>, StorageError>,
    },
    Admission {
        request: Box<InputRoutingRequest>,
        verdict: Option<Result<InputRoute, ProviderError>>,
        done: Receipt<InputAdmission>,
        outcome: Result<InputAdmission, StorageError>,
    },
}

impl ConversationWrite {
    fn into_command(self) -> SessionCommand {
        match self {
            Self::Flush { done } => SessionCommand::Flush { done },
            Self::Input { command, done, .. } => SessionCommand::Input { command, done },
            Self::Remote { command, done, .. } => SessionCommand::Apply { command, done },
            Self::Admission { request, verdict, done, .. } => {
                SessionCommand::AdmitInput { request, verdict, done }
            }
        }
    }

    fn resolve(self, saved: Result<Value, StorageError>) {
        match self {
            Self::Flush { done } => done.resolve(saved),
            Self::Input { done, outcome, .. } => done.resolve(saved.and(outcome)),
            Self::Remote { done, outcome, .. } => done.resolve(saved.and(outcome)),
            Self::Admission { done, outcome, .. } => done.resolve(saved.and(outcome)),
        }
    }
}

pub(super) enum Advance {
    Opened(Result<Arc<dyn SessionCheckpoint>, StorageError>),
    Written(Result<Value, StorageError>),
}

pub(super) async fn checkpointed(work: &Option<Work>) -> Advance {
    match work {
        Some(Work::Opening { result, .. }) => {
            Advance::Opened(result.wait().await.unwrap_or_else(|_| {
                Err(StorageError::Unavailable("the checkpoint acquisition task ended".into()))
            }))
        }
        Some(Work::Writing { result, .. }) => {
            Advance::Written(result.wait().await.unwrap_or_else(|_| {
                Err(StorageError::Unavailable("the conversation checkpoint task ended".into()))
            }))
        }
        None | Some(Work::Closing { .. }) => std::future::pending().await,
    }
}

impl VoiceRuntime {
    /// View effects run first, so the delivery's canonical message is already present.
    pub(super) fn input_presented(&mut self, requests: Vec<InputRequestRef>, message_id: String) {
        let result = (|| {
            let mut state = self.conversation_snapshot()?;
            input::apply(
                &mut state,
                vec![lk_agents_abi::session::SessionInputCommand::Presented {
                    requests,
                    message_id,
                }],
            )?;
            self.session_sink.as_ref().expect("snapshot has a binding").stage_voice(state)
        })();
        if let Err(error) = result {
            let _ = self.reports.send(VoiceEvent::ConversationStateFailed(error));
        }
    }

    pub(super) fn conversation_snapshot(&self) -> Result<SessionSnapshot, StorageError> {
        self.driver.admit_conversation()?;
        let state = self
            .session_sink
            .as_ref()
            .ok_or_else(|| StorageError::Invalid("the voice conversation has no binding".into()))?
            .restore()?
            .ok_or_else(|| StorageError::Invalid("the voice conversation has not opened".into()))?;
        self.driver.validate_conversation(&state)?;
        Ok(state)
    }

    pub(super) fn conversation_command(&mut self, command: SessionCommand) {
        if let Err(error) = self.driver.admit_conversation() {
            command.fail(error);
            return;
        }
        let Some(sink) = self.session_sink.clone() else {
            command.fail(StorageError::Invalid("the voice conversation has no binding".into()));
            return;
        };
        let history = self.driver.checkpoint_history();
        let read = history.items.clone();
        let (complete, result) = signal::once();
        let task = self.shared.tasks.scoped(
            &*self.shared.spawner,
            "acquire voice conversation checkpoint",
            async move {
                complete.resolve(sink.checkpoint_voice(history).await);
            },
        );
        self.conversation_work = Some(Work::Opening { command, read, _task: task, result });
    }

    pub(super) fn conversation_advanced(&mut self, advance: Advance, now: Now) {
        let work = self.conversation_work.take().expect("a receipt has an owning operation");
        match (work, advance) {
            (Work::Opening { command, read, .. }, Advance::Opened(opened)) => {
                let scope = match opened {
                    Ok(scope) => scope,
                    Err(error) => {
                        command.fail(error.clone());
                        self.handle(VoiceEvent::ConversationStateFailed(error).into(), now);
                        return;
                    }
                };
                // Refuse a text-local continuation before projecting live voice history over it.
                if let Err(error) = self.conversation_snapshot() {
                    command.fail(error.clone());
                    self.handle(VoiceEvent::ConversationStateFailed(error).into(), now);
                    return;
                }
                if let Some(history) = scope.history() {
                    self.handle(VoiceEvent::StorageReloaded { read, history }.into(), now);
                }
                // A lease reload can replace the binding even when the core's chat revision
                // stays equal. Re-publish the accepted words against that fresh native state.
                let effect = self.driver.conversation_history();
                self.perform_voice(vec![effect]);
                self.apply_conversation_command(command, scope);
            }
            (Work::Writing { write, .. }, Advance::Written(saved)) => {
                if let Err(error) = &saved
                    && self.driver.retry_conversation_checkpoint(error)
                {
                    self.conversation_command(write.into_command());
                    return;
                }
                let failed = saved.as_ref().err().cloned();
                if saved.is_ok()
                    && let ConversationWrite::Admission { request, outcome: Ok(admission), .. } =
                        &write
                {
                    self.handle(
                        VoiceEvent::InputReserved {
                            message: request.message.clone(),
                            admission: admission.clone(),
                        }
                        .into(),
                        now,
                    );
                }
                if saved.is_ok()
                    && let ConversationWrite::Remote { command, outcome: Ok(_), .. } = &write
                    && let lk_agents_abi::delegation::remote::RemoteInput::Prepare {
                        input_request: Some(request),
                        ..
                    } = &command.input
                {
                    self.handle(
                        VoiceEvent::RemoteResponseQueued { request: request.clone() }.into(),
                        now,
                    );
                }
                write.resolve(saved);
                if let Some(error) = failed {
                    self.handle(VoiceEvent::ConversationStateFailed(error).into(), now);
                }
            }
            _ => unreachable!("each receipt matches its owner's phase"),
        }
    }

    fn apply_conversation_command(
        &mut self,
        command: SessionCommand,
        scope: Arc<dyn SessionCheckpoint>,
    ) {
        let mut state = match self.conversation_snapshot() {
            Ok(state) => state,
            Err(error) => {
                command.fail(error.clone());
                let _ = self.reports.send(VoiceEvent::ConversationStateFailed(error));
                return;
            }
        };
        let write = match command {
            SessionCommand::RoutingContext {
                remote_id,
                endpoint,
                proposed,
                selected_task,
                offered,
                done,
            } => {
                done.resolve(lk_agents_core::delegation::remote::routing::request(
                    &state,
                    &remote_id,
                    &endpoint,
                    proposed,
                    selected_task.as_deref(),
                    &offered,
                ));
                return;
            }
            SessionCommand::AdmitCall { request, verdict, usage, candidate, drivers, done } => {
                self.driver.routing_usage(&usage);
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
                done.resolve(drivers.get(&state, &remote_id, &endpoint, &task));
                return;
            }
            SessionCommand::HeldInput { remote_id, endpoint, task_id, done } => {
                let result = remote::state(&state, &remote_id, &endpoint).and_then(|remote| {
                    remote
                        .map(|remote| {
                            input::held_remote_input(&remote_id, remote, task_id.as_deref())
                        })
                        .transpose()
                        .map(Option::flatten)
                });
                done.resolve(result);
                return;
            }
            SessionCommand::Task { remote_id, endpoint, task_id, done } => {
                let result = remote::state(&state, &remote_id, &endpoint).and_then(|saved| {
                    let saved = saved.ok_or_else(|| {
                        lk_agents_abi::storage::StorageError::Invalid(
                            "this remote has no known tasks".into(),
                        )
                    })?;
                    remote::task(saved, &task_id)
                });
                done.resolve(result);
                return;
            }
            SessionCommand::Response { reference, done } => {
                done.resolve(input::response(&state, &reference));
                return;
            }
            SessionCommand::DefaultContext { remote_id, endpoint, done } => {
                let result = remote::state(&state, &remote_id, &endpoint)
                    .map(|remote| remote.and_then(|remote| remote.default_context_id.clone()));
                done.resolve(result);
                return;
            }
            SessionCommand::Recover { remote_id, endpoint, origin, done } => {
                let result = remote::state(&state, &remote_id, &endpoint).and_then(|remote| {
                    remote
                        .map(|remote| remote::recovery(remote, &origin))
                        .transpose()
                        .map(Option::flatten)
                });
                done.resolve(result);
                return;
            }
            SessionCommand::Input { command, done } => {
                let outcome = input::apply(&mut state, vec![command.clone()]);
                ConversationWrite::Input { command, done, outcome }
            }
            SessionCommand::Apply { command, done } => {
                let outcome = self
                    .driver
                    .admit_remote(&command.input)
                    .and_then(|()| remote::apply(&mut state, *command.clone()));
                ConversationWrite::Remote { command, done, outcome }
            }
            SessionCommand::AdmitInput { request, verdict, done } => {
                let outcome = input::route(
                    &mut state,
                    &request.message,
                    &request.candidates,
                    verdict.clone(),
                )
                .map_err(|error| StorageError::Invalid(error.0));
                ConversationWrite::Admission { request, verdict, done, outcome }
            }
            SessionCommand::Flush { done } => ConversationWrite::Flush { done },
        };
        // A refused projection can retain valid protocol evidence. Persist that evidence
        // before acknowledging either outcome, as the text owner does.
        if let Err(error) =
            self.session_sink.as_ref().expect("snapshot has a binding").stage_voice(state)
        {
            write.into_command().fail(error.clone());
            let _ = self.reports.send(VoiceEvent::ConversationStateFailed(error));
            return;
        }
        let (complete, result) = signal::once();
        let task = self.shared.tasks.scoped(
            &*self.shared.spawner,
            "voice conversation command checkpoint",
            async move {
                complete.resolve(scope.commit().await);
            },
        );
        self.conversation_work = Some(Work::Writing { write, _task: task, result });
    }

    pub(super) fn close_conversation(&mut self) {
        self.conversation.close();
        while let Ok(command) = self.conversation.try_recv() {
            command.fail(StorageError::Cancelled("the voice conversation ended".into()));
        }
        match self.conversation_work.take() {
            Some(Work::Writing { write, _task, .. }) => {
                write.into_command().fail(StorageError::Cancelled(
                    "the voice conversation ended during its write".into(),
                ));
                self.conversation_work = Some(Work::Closing { _task });
            }
            Some(Work::Opening { command, .. }) => command.fail(StorageError::Cancelled(
                "the voice conversation ended before its checkpoint".into(),
            )),
            None | Some(Work::Closing { .. }) => {}
        }
    }
}
