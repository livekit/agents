//! Execute shared conversation commands under a fresh scope while audio keeps running.

use std::sync::Arc;

use super::{Now, VoiceEvent, VoiceRuntime};
use crate::session::SessionCommand;
use lk_agents_abi::session::{InputRequestRef, SessionCheckpoint, snapshot::SessionSnapshot};
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
        command: SessionCommand,
        outcome: Result<Option<InputRequestRef>, StorageError>,
        _task: ScopedTask,
        result: signal::Cell<Option<Result<Value, StorageError>>>,
    },
    /// The caller has ended; the admitted write and its scope still join session cleanup.
    Closing { _task: ScopedTask },
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

    fn conversation_snapshot(&self) -> Result<SessionSnapshot, StorageError> {
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
            (Work::Writing { command, outcome, .. }, Advance::Written(saved)) => {
                if let Err(error) = &saved
                    && self.driver.retry_conversation_checkpoint(error)
                {
                    self.conversation_command(command);
                    return;
                }
                let failed = saved.as_ref().err().cloned();
                match command {
                    SessionCommand::Flush { done } => done.resolve(saved),
                    SessionCommand::Input { done, .. } => {
                        done.resolve(saved.map(drop).and(outcome.map(drop)));
                    }
                    SessionCommand::Apply { done, .. } => {
                        done.resolve(saved.map(drop).and(outcome));
                    }
                    command => command.fail(StorageError::Invalid(
                        "a read-only command cannot own a write".into(),
                    )),
                }
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
        let outcome = match &command {
            SessionCommand::HeldInput { remote_id, endpoint, .. } => {
                let result = remote::state(&state, remote_id, endpoint).and_then(|remote| {
                    remote
                        .map(|remote| input::held_remote_input(remote_id, remote))
                        .transpose()
                        .map(Option::flatten)
                });
                if let SessionCommand::HeldInput { done, .. } = command {
                    done.resolve(result);
                }
                return;
            }
            SessionCommand::Response { reference, .. } => {
                let result = input::response(&state, reference);
                if let SessionCommand::Response { done, .. } = command {
                    done.resolve(result);
                }
                return;
            }
            SessionCommand::DefaultContext { remote_id, endpoint, .. } => {
                let result = remote::state(&state, remote_id, endpoint)
                    .map(|remote| remote.and_then(|remote| remote.default_context_id.clone()));
                if let SessionCommand::DefaultContext { done, .. } = command {
                    done.resolve(result);
                }
                return;
            }
            SessionCommand::Recover { remote_id, endpoint, origin, .. } => {
                let result = remote::state(&state, remote_id, endpoint).and_then(|remote| {
                    remote
                        .map(|remote| remote::recovery(remote, origin))
                        .transpose()
                        .map(Option::flatten)
                });
                if let SessionCommand::Recover { done, .. } = command {
                    done.resolve(result);
                }
                return;
            }
            SessionCommand::Input { command, .. } => {
                input::apply(&mut state, vec![command.clone()]).map(|()| None)
            }
            SessionCommand::Apply { command, .. } => self
                .driver
                .admit_remote(&command.input)
                .and_then(|()| remote::apply(&mut state, *command.clone())),
            SessionCommand::Flush { .. } => Ok(None),
        };
        // A refused projection can retain valid protocol evidence. Persist that evidence
        // before acknowledging either outcome, as the text owner does.
        if let Err(error) =
            self.session_sink.as_ref().expect("snapshot has a binding").stage_voice(state)
        {
            command.fail(error.clone());
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
        self.conversation_work = Some(Work::Writing { command, outcome, _task: task, result });
    }

    pub(super) fn close_conversation(&mut self) {
        self.conversation.close();
        while let Ok(command) = self.conversation.try_recv() {
            command.fail(StorageError::Cancelled("the voice conversation ended".into()));
        }
        match self.conversation_work.take() {
            Some(Work::Writing { command, _task, .. }) => {
                command.fail(StorageError::Cancelled(
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
