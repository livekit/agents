//! Storage receipts precede model reads and reply settlement; I/O stays in the executor.

use std::collections::BTreeMap;

use lk_agents_abi::{ChatItem, Ts};

use super::Driver;
use super::effects::{Effect, Out};
use crate::agent_activity::scheduling::SpeechId;
use crate::events::SessionEvent;
use crate::generation::TurnOutcome;

#[derive(Default)]
pub(crate) struct StorageTurns {
    pub enabled: bool,
    pub lost: bool,
    failures: BTreeMap<SpeechId, Box<lk_agents_abi::errors::ProviderError>>,
    pending: BTreeMap<SpeechId, (TurnOutcome, bool)>,
}

impl StorageTurns {
    pub(super) fn new(enabled: bool) -> Self {
        Self { enabled, ..Default::default() }
    }
}

impl Driver {
    pub(crate) fn storage_reloaded(&mut self, history: Vec<ChatItem>, out: &mut Out) {
        self.session.chat_ctx.replace_transport(history, &self.activity.agent);
        self.publish_activity();
        out.machine().emit(SessionEvent::HistoryReloaded {
            items: self.session.chat_ctx.items().cloned().collect(),
        });
    }

    pub(crate) fn storage_settle(
        &mut self,
        speech: SpeechId,
        outcome: TurnOutcome,
        interrupted: bool,
        out: &mut Out,
    ) {
        self.session.storage.pending.insert(speech, (outcome, interrupted));
        out.machine().push(Effect::StorageCommit {
            speech,
            history: self.session.chat_ctx.items().cloned().collect(),
        });
    }

    pub(crate) fn storage_settled(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        if let Some((outcome, interrupted)) = self.session.storage.pending.remove(&speech) {
            if self.session.storage.lost {
                return;
            }
            self.activity.set_tools_in_flight(speech, false);
            self.finish_reply(now, speech, outcome, interrupted, out);
        }
    }

    pub(crate) fn storage_abandon(
        &mut self,
        speech: SpeechId,
        error: Box<lk_agents_abi::errors::ProviderError>,
        out: &mut Out,
    ) {
        self.session.storage.failures.insert(speech, error);
        out.machine().push(Effect::StorageAbort(speech));
    }

    pub(crate) fn storage_abandoned(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        if let Some(error) = self.session.storage.failures.remove(&speech) {
            self.generation_failed(now, speech, error, out);
        }
    }

    pub(crate) fn storage_lost(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        self.session.storage.lost = true;
        self.end_live_tools(now, speech, out);
        if let Some(handle) = self.activity.speech_scheduler.handle(speech) {
            out.machine().push(Effect::CancelSpeech(handle.clone()));
        }
        self.release_speech(now, speech, out);
    }
}
