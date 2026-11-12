use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use lk_agents_abi::{ChatItem, Ts};

use super::VoiceDriver;
use super::effects::{Timer, VoiceEffect};
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::events::SessionEvent;
use crate::generation::TurnOutcome;

// Storage receipts precede model reads and reply settlement.

/// How long a speech that ended without reporting its turn (its tasks aborted by the
/// interruption watchdog or `cancel_tasks`) may still report it before its partition is released.
pub(crate) const PARTITION_RELEASE_BOUND: Duration = Duration::from_secs(30);

#[derive(Default)]
pub(crate) struct StorageTurns {
    pub enabled: bool,
    next_checkpoint: u64,
    checkpoints: BTreeMap<u64, (SpeechId, Option<lk_agents_core::agent_run::AgentRunEffect>)>,
    /// Each holding turn's message so far, by turn.
    progress: BTreeMap<u64, lk_agents_abi::ChatMessage>,
    progress_dirty: BTreeSet<SpeechId>,
    failures: BTreeMap<SpeechId, Box<lk_agents_abi::errors::ProviderError>>,
    pending: BTreeMap<SpeechId, (TurnOutcome, bool)>,
    /// The started speeches holding the partition until their turn is committed or abandoned.
    holding: BTreeSet<SpeechId>,
    /// Turns the person cut while their tools ran, until their commit or abort is answered: a
    /// later turn commits only after them, so the thread's history stays in turn order.
    outliving: BTreeSet<u64>,
    /// Later turns whose commit waits behind an outliving one; their speeches are already done.
    waiting: BTreeMap<u64, SpeechId>,
    /// Each holding speech's turn: its place in the thread, the order the speeches started.
    /// Speech ids are random, so the collections above are keyed by turn to keep that order.
    turns: BTreeMap<SpeechId, u64>,
    next_turn: u64,
}

impl StorageTurns {
    pub(super) fn new(enabled: bool) -> Self {
        Self { enabled, ..Default::default() }
    }

    /// The speech's turn ended in the thread (committed or abandoned): it is forgotten with
    /// everything keyed by it, so no turn outlives its speech and blocks the turns after it.
    fn forget(&mut self, speech: SpeechId) {
        if let Some(turn) = self.turns.remove(&speech) {
            self.progress.remove(&turn);
            self.outliving.remove(&turn);
            self.waiting.remove(&turn);
        }
    }

    /// `speech`'s turn, numbered when the speech first reaches storage.
    fn turn(&mut self, speech: SpeechId) -> u64 {
        let next = &mut self.next_turn;
        *self.turns.entry(speech).or_insert_with(|| {
            *next += 1;
            *next
        })
    }
}

impl VoiceDriver {
    pub(crate) fn checkpoint_reply_effect(
        &mut self,
        speech: SpeechId,
        effect: lk_agents_core::agent_run::AgentRunEffect,
    ) {
        self.checkpoint_speech(speech, Some(effect));
    }

    pub(crate) fn storage_progress(&mut self, speech: SpeechId, item: lk_agents_abi::ChatMessage) {
        if self.session.storage.enabled && self.session.storage.holding.contains(&speech) {
            let turn = self.session.storage.turn(speech);
            self.session.storage.progress.insert(turn, item);
            self.session.storage.progress_dirty.insert(speech);
            self.flush_progress(speech);
        }
    }

    fn flush_progress(&mut self, speech: SpeechId) {
        if self.session.storage.checkpoints.values().any(|(owner, _)| *owner == speech) {
            return;
        }
        if self.session.storage.progress_dirty.remove(&speech) {
            self.checkpoint_speech(speech, None);
        }
    }

    pub(crate) fn storage_step_spoken(&mut self, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if let Some(turn) = storage.turns.get(&speech) {
            storage.progress.remove(turn);
        }
        storage.progress_dirty.remove(&speech);
    }

    fn checkpoint_speech(
        &mut self,
        speech: SpeechId,
        effect: Option<lk_agents_core::agent_run::AgentRunEffect>,
    ) {
        self.session.storage.progress_dirty.remove(&speech);
        let checkpoint = self.session.storage.next_checkpoint;
        self.session.storage.next_checkpoint =
            checkpoint.checked_add(1).expect("checkpoint ids fit u64");
        self.session.storage.checkpoints.insert(checkpoint, (speech, effect));
        let mut history = self.saved_history();
        // While a cut turn's call is open, what follows it waits: the thread keeps rows in the order
        // they are first written.
        let settled = history.len() == self.session.chat_ctx.items().count();
        for partial in self.session.storage.progress.values().filter(|_| settled) {
            let item = ChatItem::Message(partial.clone());
            if let Some(existing) = history.iter_mut().find(|existing| existing.id() == item.id()) {
                *existing = item;
            } else {
                history.push(item);
            }
        }
        for active in self.activity.speeches().filter(|_| settled) {
            let Some(items) = self.checkpoint_reply_history(active) else { continue };
            // Speech text comes from the delivery ledger above. AgentRun may already hold
            // generated text that has not been heard; only tool records may lead delivery.
            for item in items.iter().filter(|item| item.as_message().is_none()) {
                if let Some(existing) =
                    history.iter_mut().find(|existing| existing.id() == item.id())
                {
                    *existing = item.clone();
                } else {
                    history.push(item.clone());
                }
            }
        }
        self.record_effect(VoiceEffect::StorageCheckpoint {
            checkpoint,
            speech,
            agent: self.activity.agent_name().to_owned(),
            history,
        });
    }

    pub(crate) fn storage_checkpointed(
        &mut self,
        now: Ts,
        checkpoint: u64,
        result: Result<(), lk_agents_abi::storage::StorageError>,
    ) {
        let Some((speech, effect)) = self.session.storage.checkpoints.remove(&checkpoint) else {
            return;
        };
        match result {
            Ok(())
                if self.session.closing.is_none()
                    && self.activity.speech_can_bind_output(speech) =>
            {
                if let Some(effect) = effect {
                    self.reply_effects_ready(speech, vec![effect]);
                }
            }
            Ok(()) => {}
            Err(error) => self.storage_failed(now, speech, error),
        }
        if self.session.closing.is_none() {
            self.flush_progress(speech);
            self.commit_after_checkpoints(now, speech);
            self.abort_after_checkpoints(speech);
        }
    }

    pub(crate) fn storage_reloaded(&mut self, read: Vec<ChatItem>, history: Vec<ChatItem>) {
        if !self.session.chat_ctx.reload_transport(&read, history, self.activity.agent_name()) {
            return;
        }
        self.publish_event(SessionEvent::HistoryReloaded {
            items: self.session.chat_ctx.items().cloned().collect(),
        });
    }

    /// A started speech takes the partition: it holds it until its turn is committed or abandoned.
    pub(crate) fn storage_started(&mut self, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if storage.enabled && storage.holding.insert(speech) {
            storage.turn(speech);
        }
    }

    /// The speech's tasks reported its turn: it no longer holds the partition, and the interruption
    /// watchdog, there for tasks that never report, no longer ends it: its receipt does.
    fn turn_reported(&mut self, speech: SpeechId) {
        self.clear_deadline_if_set(Timer::PartitionRelease(speech));
        self.session.storage.holding.remove(&speech);
        if self.clear_deadline_if_set(Timer::InterruptionTimeout(speech)) {
            self.record_trace(crate::trace::SpeechStep::WatchdogCleared { speech });
        }
    }

    /// A speech the person cut is done while its tools still run: its turn keeps the partition
    /// until it reports, and later turns commit after it.
    pub(crate) fn storage_outlive(&mut self, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if storage.holding.contains(&speech) {
            let turn = storage.turn(speech);
            storage.outliving.insert(turn);
        }
    }

    /// The speech left the loop: one that never reported its turn releases the partition within
    /// [`PARTITION_RELEASE_BOUND`]. Not while its tools still run: they report it.
    pub(crate) fn storage_speech_released(&mut self, now: Ts, speech: SpeechId) {
        if self.session.storage.holding.contains(&speech)
            && !self.activity.outliving_tools.contains(&speech)
        {
            self.set_deadline(
                Timer::PartitionRelease(speech),
                now.saturating_add(PARTITION_RELEASE_BOUND),
            );
        }
    }

    /// The bound passed with the turn still unreported: it is abandoned.
    pub(crate) fn partition_release_due(&mut self, speech: SpeechId) {
        if self.session.storage.holding.remove(&speech) {
            self.record_effect(VoiceEffect::StorageAbort(speech));
        }
    }

    pub(crate) fn storage_settle(
        &mut self,
        now: Ts,
        speech: SpeechId,
        outcome: TurnOutcome,
        interrupted: bool,
    ) {
        self.turn_reported(speech);
        self.storage_step_spoken(speech);
        self.session.storage.pending.insert(speech, (outcome, interrupted));
        self.commit_after_checkpoints(now, speech);
    }

    /// Commits a reported turn once its checkpoints landed and every earlier turn still
    /// outliving its interruption has ended. A turn that waits is done now: only persistence
    /// waits.
    fn commit_after_checkpoints(&mut self, now: Ts, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if !storage.pending.contains_key(&speech)
            || storage.checkpoints.values().any(|(owner, _)| *owner == speech)
        {
            return;
        }
        let turn = storage.turn(speech);
        if storage.outliving.range(..turn).next().is_some() {
            if storage.waiting.insert(turn, speech).is_none()
                && let Some((outcome, interrupted)) = storage.pending.get(&speech).cloned()
            {
                self.record_trace(crate::trace::SpeechStep::CommitWaits { speech });
                self.finish_reply(now, speech, outcome, interrupted);
            }
            return;
        }
        storage.waiting.remove(&turn);
        let history = self.saved_history();
        self.record_effect(VoiceEffect::StorageCommit { speech, history });
    }

    /// The record as the thread may save it now: up to a turn cut while its tools run, whose
    /// calls and late outputs will land right after what was said before them.
    fn saved_history(&self) -> Vec<ChatItem> {
        let outliving = &self.activity.outliving_tools;
        let record = self.session.chat_ctx.items();
        let Some(open_since) = self.activity.outliving_since() else {
            return record.cloned().collect();
        };
        let open = |source: &crate::agent::Source| {
            source.speech().is_some_and(|speech| outliving.contains(&speech))
        };
        let settled = self.session.chat_ctx.settled_prefix(open_since, open);
        record.take(settled).cloned().collect()
    }

    /// An outliving turn's commit or abort was answered: the turns waiting behind it commit, in
    /// order, up to the next one still outliving.
    fn outliving_turn_ended(&mut self, now: Ts, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if !storage.turns.get(&speech).is_some_and(|turn| storage.outliving.remove(turn)) {
            return;
        }
        let waiting: Vec<SpeechId> = self.session.storage.waiting.values().copied().collect();
        for later in waiting {
            self.commit_after_checkpoints(now, later);
        }
    }

    pub(crate) fn storage_settled(&mut self, now: Ts, speech: SpeechId) {
        if let Some((outcome, interrupted)) = self.session.storage.pending.remove(&speech) {
            self.activity.tools_in_flight.remove(&speech);
            self.finish_reply(now, speech, outcome, interrupted);
        }
        self.outliving_turn_ended(now, speech);
        self.session.storage.forget(speech);
    }

    pub(crate) fn storage_abandon(
        &mut self,
        speech: SpeechId,
        error: Box<lk_agents_abi::errors::ProviderError>,
    ) {
        self.turn_reported(speech);
        self.session.storage.failures.insert(speech, error);
        self.abort_after_checkpoints(speech);
    }

    fn abort_after_checkpoints(&mut self, speech: SpeechId) {
        if !self.session.storage.failures.contains_key(&speech)
            || self.session.storage.checkpoints.values().any(|(owner, _)| *owner == speech)
        {
            return;
        }
        self.record_effect(VoiceEffect::StorageAbort(speech));
    }

    pub(crate) fn storage_abandoned(&mut self, now: Ts, speech: SpeechId) {
        self.storage_step_spoken(speech);
        if let Some(error) = self.session.storage.failures.remove(&speech) {
            self.generation_failed(now, speech, error);
        }
        self.outliving_turn_ended(now, speech);
        self.session.storage.forget(speech);
    }

    /// A bound call cannot continue making effects after it loses durable ownership.
    pub(crate) fn storage_lost(&mut self, now: Ts, speech: SpeechId) {
        if self.session.storage.enabled {
            self.record_trace(crate::trace::SpeechStep::PartitionLost { speech });
            self.storage_failed(
                now,
                speech,
                lk_agents_abi::storage::StorageError::LeaseLost(
                    "the voice partition lease was lost".into(),
                ),
            );
        }
    }

    pub(crate) fn storage_failed(
        &mut self,
        now: Ts,
        speech: SpeechId,
        error: lk_agents_abi::storage::StorageError,
    ) {
        self.session.storage.checkpoints.clear();
        self.session.storage.progress_dirty.clear();
        let event = crate::events::ErrorEvent::provider(
            lk_agents_abi::voice::events::ErrorSource::Storage,
            false,
            lk_agents_abi::errors::ProviderError::Local(error.into()),
            Some(speech),
        );
        self.begin_closing(
            now,
            super::commands::Close {
                reason: lk_agents_abi::voice::events::CloseReason::Error,
                drain: false,
                error: Some(event),
            },
        );
    }
}
