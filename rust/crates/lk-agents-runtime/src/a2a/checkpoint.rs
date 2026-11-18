//! Framework checkpoints use the endpoint's existing lease after any individual handler closes.

use super::*;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::storage::voice::VoiceHistory;
use lk_agents_abi::text_session::{SessionCheckpointPort, SessionSnapshot};
use lk_agents_net::signal;
use parking_lot::Mutex;
use tokio::sync::mpsc;

pub(super) struct Port {
    sender: mpsc::Sender<Write>,
    baseline: Mutex<Option<VoiceHistory>>,
}

pub(super) struct Write {
    state: Value,
    history: VoiceHistory,
    done: signal::Producer<
        Option<Result<lk_agents_storage::thread::SessionCheckpointData, StorageError>>,
    >,
}

pub(super) fn channel() -> (Arc<Port>, mpsc::Receiver<Write>) {
    let (sender, receiver) = mpsc::channel(handler::REPORT_QUEUE);
    (Arc::new(Port { sender, baseline: Mutex::new(None) }), receiver)
}

impl Port {
    pub(super) fn initialize(
        &self,
        history: &[lk_agents_abi::storage::ItemRow],
    ) -> Result<(), StorageError> {
        let mut baseline = self.baseline.lock();
        if baseline.is_none() {
            let stored = history.iter().map(|row| row.payload.clone()).collect();
            *baseline = Some(VoiceHistory { stored, ..Default::default() });
        }
        Ok(())
    }
}

impl SessionCheckpointPort for Port {
    fn publish(&self, state: SessionSnapshot) -> BoxFuture<'_, Result<Value, StorageError>> {
        Box::pin(async move {
            let mut history = self.baseline.lock().clone().ok_or_else(|| {
                StorageError::Invalid("the endpoint checkpoint has no observed baseline".into())
            })?;
            history.items = state
                .frames
                .get(&state.root_frame_id)
                .ok_or_else(|| {
                    StorageError::Invalid("the checkpoint lost its root context".into())
                })?
                .history
                .clone();
            let state = lk_agents_storage::session_state::snapshot_value(state)?;
            let (done, receipt) = signal::once();
            self.sender.send(Write { state, history, done }).await.map_err(|_| {
                StorageError::Invalid("the leased endpoint checkpoint owner ended".into())
            })?;
            let committed = receipt.wait().await.map_err(|_| {
                StorageError::Invalid("the endpoint ended before its checkpoint receipt".into())
            })??;
            *self.baseline.lock() = Some(committed.history);
            Ok(committed.state)
        })
    }
}

impl Turn {
    /// The leased Call owns every admitted write until its exact storage acknowledgment.
    /// The receipt is independent of the handler and no second partition lease is acquired.
    pub(super) async fn framework_checkpoint(
        &self,
        call: &mut Call,
        write: Write,
    ) -> Result<(), StorageError> {
        let result = match call.run() {
            Ok(run) => match run.reconcile_session_view(write.state, write.history) {
                Ok(prepared) => prepared.commit(run).await,
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };
        let outcome = result.as_ref().map(|_| ()).map_err(Clone::clone);
        write.done.resolve(result);
        outcome
    }
}
