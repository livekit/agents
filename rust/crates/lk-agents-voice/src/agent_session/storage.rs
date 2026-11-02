//! Storage receipts precede model reads and reply settlement; I/O stays in the executor.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use lk_agents_abi::{ChatItem, Ts};

use super::VoiceDriver;
use super::effects::{Timer, VoiceEffect};
use crate::agent_activity::scheduling::SpeechId;
use crate::events::SessionEvent;
use crate::generation::TurnOutcome;

/// How long a speech that ended without reporting its turn (its tasks aborted by the
/// interruption watchdog or `cancel_tasks`) may still report it before its lane is released
/// (voice-session.md: "within 30 seconds").
pub(crate) const LANE_RELEASE_BOUND: Duration = Duration::from_secs(30);

#[derive(Default)]
pub(crate) struct StorageTurns {
    pub enabled: bool,
    failures: BTreeMap<SpeechId, Box<lk_agents_abi::errors::ProviderError>>,
    pending: BTreeMap<SpeechId, (TurnOutcome, bool)>,
    /// The started speeches holding the lane until their turn is committed or abandoned.
    holding: BTreeSet<SpeechId>,
}

impl StorageTurns {
    pub(super) fn new(enabled: bool) -> Self {
        Self { enabled, ..Default::default() }
    }
}

impl VoiceDriver {
    pub(crate) fn storage_reloaded(&mut self, read: Vec<ChatItem>, history: Vec<ChatItem>) {
        if !self.session.chat_ctx.reload_transport(&read, history, self.activity.agent_name()) {
            return;
        }
        self.publish_activity();
        self.publish_event(SessionEvent::HistoryReloaded {
            items: self.session.chat_ctx.items().cloned().collect(),
        });
    }

    /// A started speech takes the lane: it holds it until its turn is committed or abandoned.
    pub(crate) fn storage_started(&mut self, speech: SpeechId) {
        if self.session.storage.enabled {
            self.session.storage.holding.insert(speech);
        }
    }

    /// The speech reported its turn, or the lane is lost: it no longer holds the lane.
    fn storage_reported(&mut self, speech: SpeechId) -> bool {
        self.clear_deadline_if_set(Timer::LaneRelease(speech));
        self.session.storage.holding.remove(&speech)
    }

    /// The speech left the loop: one that never reported its turn releases the lane within
    /// [`LANE_RELEASE_BOUND`].
    pub(crate) fn storage_speech_released(&mut self, now: Ts, speech: SpeechId) {
        if self.session.storage.holding.contains(&speech) {
            self.set_deadline(Timer::LaneRelease(speech), now.saturating_add(LANE_RELEASE_BOUND));
        }
    }

    /// The bound passed with the turn still unreported: it is abandoned.
    pub(crate) fn lane_release_due(&mut self, speech: SpeechId) {
        if self.session.storage.holding.remove(&speech) {
            self.record_effect(VoiceEffect::StorageAbort(speech));
        }
    }

    pub(crate) fn storage_settle(
        &mut self,
        speech: SpeechId,
        outcome: TurnOutcome,
        interrupted: bool,
    ) {
        self.storage_reported(speech);
        self.session.storage.pending.insert(speech, (outcome, interrupted));
        self.record_effect(VoiceEffect::StorageCommit {
            speech,
            history: self.session.chat_ctx.items().cloned().collect(),
        });
    }

    pub(crate) fn storage_settled(&mut self, now: Ts, speech: SpeechId) {
        if let Some((outcome, interrupted)) = self.session.storage.pending.remove(&speech) {
            self.activity.set_tools_in_flight(speech, false);
            self.finish_reply(now, speech, outcome, interrupted);
        }
    }

    pub(crate) fn storage_abandon(
        &mut self,
        speech: SpeechId,
        error: Box<lk_agents_abi::errors::ProviderError>,
    ) {
        self.storage_reported(speech);
        self.session.storage.failures.insert(speech, error);
        self.record_effect(VoiceEffect::StorageAbort(speech));
    }

    pub(crate) fn storage_abandoned(&mut self, now: Ts, speech: SpeechId) {
        if let Some(error) = self.session.storage.failures.remove(&speech) {
            self.generation_failed(now, speech, error);
        }
    }

    /// The lane's lease is lost: the call goes on in memory, as one whose thread could not be
    /// loaded does (voice-session.md: storage failures never end the call). What its speeches
    /// say is still heard; no turn from here on takes the lane or commits.
    pub(crate) fn storage_lost(&mut self, speech: SpeechId) {
        if self.session.storage.enabled {
            log::warn!(target: "livekit.agents", "the voice lane lease was lost: the call continues without storage");
            self.record_trace(crate::trace::SpeechStep::LaneLost { speech });
        }
        self.session.storage.enabled = false;
        for speech in std::mem::take(&mut self.session.storage.holding) {
            self.clear_deadline_if_set(Timer::LaneRelease(speech));
        }
    }
}
