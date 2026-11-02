//! Typed storage commands and receipts, over the session's existing event queue.

use super::VoiceRuntime;
use crate::voice::session::Tx;
use lk_agents_abi::StreamInput;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::storage::voice::{VoiceStorageCommand, VoiceStorageEvent};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::VoiceEvent;
use std::sync::Arc;

pub(super) struct StorageReporter(pub Tx);

impl StreamInput<VoiceStorageEvent> for StorageReporter {
    fn push(&self, event: VoiceStorageEvent) -> Result<(), ProviderError> {
        let input = match event {
            VoiceStorageEvent::Settled { speech, result, lease_lost } => {
                if let Err(error) = result {
                    log::warn!(target: "livekit.agents", "the voice turn could not commit: {error}");
                }
                if lease_lost {
                    let _ = self.0.send(VoiceEvent::StorageLost(SpeechId(speech)));
                }
                VoiceEvent::StorageSettled(SpeechId(speech))
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                if let Err(error) = result {
                    log::warn!(target: "livekit.agents", "the voice turn could not abandon: {error}");
                }
                VoiceEvent::StorageAbandoned(SpeechId(speech))
            }
            VoiceStorageEvent::LeaseLost { speech } => VoiceEvent::StorageLost(SpeechId(speech)),
        };
        self.0.send(input).map_err(|_| {
            ProviderError::Local(ApiError::connection(Some("the voice session has closed".into())))
        })
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl VoiceRuntime {
    pub(super) fn storage_command(&self, command: VoiceStorageCommand) {
        let event = match &command {
            VoiceStorageCommand::Settle { speech, .. } => Some(VoiceStorageEvent::Settled {
                speech: *speech,
                result: Ok(()),
                lease_lost: false,
            }),
            VoiceStorageCommand::Abort { speech } => {
                Some(VoiceStorageEvent::Abandoned { speech: *speech, result: Ok(()) })
            }
            VoiceStorageCommand::ObserveLoss { .. } => None,
        };
        let reporter = Arc::new(StorageReporter(self.reports.clone()));
        if let Some(storage) = &self.storage {
            if let Err(error) = storage.clone().dispatch(command, reporter.clone())
                && let Some(mut event) = event
            {
                match &mut event {
                    VoiceStorageEvent::Settled { result, .. }
                    | VoiceStorageEvent::Abandoned { result, .. } => *result = Err(error),
                    VoiceStorageEvent::LeaseLost { .. } => {}
                }
                let _ = reporter.push(event);
            }
        } else if let Some(event) = event {
            let _ = reporter.push(event);
        }
    }
}
