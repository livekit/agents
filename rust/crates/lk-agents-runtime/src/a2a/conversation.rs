use super::*;
use std::future::Future;

enum OwnerRead<T> {
    Checkpoint(super::checkpoint::Write),
    Receipt(Result<T, StorageError>),
    Handler(String, Option<Read>),
}

impl Turn {
    pub(super) async fn cancel_live_tasks(
        &self,
        call: &mut Call,
        tasks: Vec<String>,
    ) -> Result<Completion, StorageError> {
        let unowned = match call.managed.commands() {
            Some(owner) => self.owner_receipt(call, owner.cancel_live_tasks(tasks)).await?,
            None => tasks,
        };
        Ok(Completion::CancelledLive(unowned))
    }

    pub(super) async fn prepare_dialogue(
        &self,
        call: &mut Call,
        source: String,
        messages: Vec<lk_agents_abi::ChatMessage>,
    ) -> Result<Completion, StorageError> {
        let owner = call.managed.commands().ok_or_else(|| {
            StorageError::Invalid("the native conversation owner ended before opaque output".into())
        })?;
        let snapshot = self.owner_receipt(call, owner.prepare_dialogue(source, messages)).await?;
        Ok(Completion::DialoguePrepared(lk_agents_storage::session_state::snapshot_value(
            snapshot,
        )?))
    }

    pub(super) async fn commit_dialogue(
        &self,
        call: &mut Call,
        source: String,
    ) -> Result<Completion, StorageError> {
        let owner = call.managed.commands().ok_or_else(|| {
            StorageError::Invalid("the native owner ended before dialogue activation".into())
        })?;
        // This is the paused owner's priority ACK. It performs no I/O before resolving;
        // effects resume after the receipt, when this A2A owner can reopen its kept run.
        owner.commit_dialogue(source).await?;
        Ok(Completion::Done)
    }

    async fn owner_receipt<T>(
        &self,
        call: &mut Call,
        future: impl Future<Output = Result<T, StorageError>>,
    ) -> Result<T, StorageError> {
        let lease = call.run.as_ref().and_then(Run::lease).cloned().ok_or_else(|| {
            StorageError::Invalid("native dialogue preparation lost its fenced run".into())
        })?;
        let mut future = std::pin::pin!(future);
        loop {
            lease.check()?;
            // Race only receipt/read readiness. Once a report owns a write, execute it to
            // completion before inspecting another input; never cut a write off on an ACK.
            let read = tokio::select! {
                biased;
                Some(write) = call.checkpoints.recv() => OwnerRead::Checkpoint(write),
                result = &mut future => OwnerRead::Receipt(result),
                (id, read) = live::next_handler(&mut call.handlers) => OwnerRead::Handler(id, read),
                error = lease.lost() => return Err(error),
            };
            let (id, read) = match read {
                OwnerRead::Checkpoint(write) => {
                    self.framework_checkpoint(call, write).await?;
                    continue;
                }
                OwnerRead::Receipt(result) => return result,
                OwnerRead::Handler(id, read) => (id, read),
            };
            if let Some(completion) = self.live_report(call, id, read).await? {
                // Each retained source can owe a checkpoint and one final report/output.
                // The producer stays with its Handler while the current operation settles.
                call.deferred.push_back(completion);
            }
            lease.check()?;
        }
    }
}
