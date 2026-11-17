//! Execute the shared conversation commands without holding up voice events or audio.

use super::VoiceRuntime;
use crate::session::SessionCommand;
use lk_agents_abi::session::snapshot::SessionSnapshot;
use lk_agents_abi::storage::StorageError;
use lk_agents_core::session::{input, remote};
use lk_agents_net::{signal, task::ScopedTask};

pub(super) struct Checkpoint {
    _task: ScopedTask,
    result: signal::Cell<Option<Result<(), StorageError>>>,
}

pub(super) async fn checkpointed(work: &Option<Checkpoint>) -> Result<(), StorageError> {
    let Some(work) = work else { return std::future::pending().await };
    work.result.wait().await.unwrap_or_else(|_| {
        Err(StorageError::Unavailable("the conversation checkpoint task ended".into()))
    })
}

impl VoiceRuntime {
    fn conversation_snapshot(&self) -> Result<SessionSnapshot, StorageError> {
        self.driver.admit_conversation()?;
        self.session_sink
            .as_ref()
            .ok_or_else(|| StorageError::Invalid("the voice conversation has no binding".into()))?
            .restore()?
            .ok_or_else(|| StorageError::Invalid("the voice conversation has not opened".into()))
    }

    pub(super) fn conversation_command(&mut self, command: SessionCommand) {
        match command {
            SessionCommand::HeldInput { remote_id, endpoint, done } => {
                let result = self.conversation_snapshot().and_then(|state| {
                    remote::state(&state, &remote_id, &endpoint)?
                        .map(|state| input::held_remote_input(&remote_id, state))
                        .transpose()
                        .map(Option::flatten)
                });
                done.resolve(result);
            }
            SessionCommand::Response { reference, done } => {
                done.resolve(
                    self.conversation_snapshot()
                        .and_then(|state| input::response(&state, &reference)),
                );
            }
            SessionCommand::DefaultContext { remote_id, endpoint, done } => {
                done.resolve(self.conversation_snapshot().and_then(|state| {
                    Ok(remote::state(&state, &remote_id, &endpoint)?
                        .and_then(|remote| remote.default_context_id.clone()))
                }));
            }
            SessionCommand::Recover { remote_id, endpoint, origin, done } => {
                done.resolve(self.conversation_snapshot().and_then(|state| {
                    remote::state(&state, &remote_id, &endpoint)?
                        .map(|remote| remote::recovery(remote, &origin))
                        .transpose()
                        .map(Option::flatten)
                }));
            }
            SessionCommand::Input { command, done } => {
                let mut state = match self.conversation_snapshot() {
                    Ok(state) => state,
                    Err(error) => {
                        done.resolve(Err(error));
                        return;
                    }
                };
                let result = input::apply(&mut state, vec![command]);
                self.checkpoint_conversation(state, done, result);
            }
            SessionCommand::Apply { command, done } => {
                let mut state = match self
                    .driver
                    .admit_remote(&command.input)
                    .and_then(|()| self.conversation_snapshot())
                {
                    Ok(state) => state,
                    Err(error) => {
                        done.resolve(Err(error));
                        return;
                    }
                };
                // A refused projection may still retain a valid protocol observation. As in
                // text, persist that evidence before acknowledging either outcome.
                let result = remote::apply(&mut state, *command);
                self.checkpoint_conversation(state, done, result);
            }
        }
    }

    fn checkpoint_conversation<T: Send + Sync + 'static>(
        &mut self,
        state: SessionSnapshot,
        done: signal::Producer<Option<Result<T, StorageError>>>,
        result: Result<T, StorageError>,
    ) {
        let Some(sink) = self.session_sink.clone() else {
            done.resolve(Err(StorageError::Invalid(
                "the voice conversation has no binding".into(),
            )));
            return;
        };
        if let Err(error) = sink.stage_voice(state) {
            done.resolve(Err(error.clone()));
            let _ = self
                .reports
                .send(lk_agents_voice::agent_session::VoiceEvent::ConversationStateFailed(error));
            return;
        }
        let cx = self.peer.session.clone();
        let (complete, receipt) = signal::once();
        let task = self.shared.tasks.scoped(
            &*self.shared.spawner,
            "voice conversation command checkpoint",
            async move {
                let saved = sink.flush_voice(&cx).await;
                done.resolve(saved.clone().and(result));
                complete.resolve(saved);
            },
        );
        self.conversation_write = Some(Checkpoint { _task: task, result: receipt });
    }
}
