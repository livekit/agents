use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{ChatItem, Ts};
use lk_agents_core::agent_run::AgentRunEffect;

use super::VoiceDriver;
use super::effects::{Timer, VoiceEffect};
use super::machine::StorageRequest;
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::events::SessionEvent;
use crate::generation::TurnOutcome;

// Storage receipts precede model reads and reply settlement.

/// How long a speech that ended without reporting its turn (its tasks aborted by the
/// interruption watchdog or `cancel_tasks`) may still report it before its partition is released.
pub(crate) const PARTITION_RELEASE_BOUND: Duration = Duration::from_secs(30);

/// A started speech's turn in the thread. A speech waits on its storage only through its
/// [`Save`], whose every request is answered exactly once (success, failure, or the death of
/// the task that carried it): no answer, and no failure, can leave the speech waiting.
struct Turn {
    speech: SpeechId,
    save: Save,
}

/// Checkpoints in flight, each with the reply's next step when it waits on that receipt.
type Checkpoints = BTreeMap<u64, Option<AgentRunEffect>>;

/// Where a turn's save stands. The reply's report of how its turn ends and the store's answers
/// may arrive in either order; this one state holds both.
enum Save {
    /// The partition is being taken: the words heard so far and the reply's next steps wait for
    /// it, as nothing can be saved before.
    Taking { held: Vec<AgentRunEffect> },
    /// The reply runs; its checkpoints are in flight.
    Running { checkpoints: Checkpoints },
    /// The reply reported how its turn ends; the save is asked once its checkpoints are answered.
    Reported { end: TurnEnd, checkpoints: Checkpoints },
    /// The commit or abort is asked: its answer ends the turn.
    Asked { end: TurnEnd },
}

/// How a turn ends in the thread, and what its answer tells the speech.
enum TurnEnd {
    /// Committed, then the speech ends with its reply's outcome.
    Commit { outcome: Box<TurnOutcome>, interrupted: bool },
    /// Abandoned, then the speech ends with its reply's failure.
    Abort { error: Box<ProviderError> },
    /// Abandoned; the speech already ended without reporting its turn.
    AbortEnded,
}

/// Which save a turn asks.
enum TurnSave {
    Commit,
    Abort,
}

impl Save {
    /// A reported turn whose checkpoints are all answered asks its save.
    fn ask(self) -> (Save, Option<TurnSave>) {
        match self {
            Save::Reported { end, checkpoints } if checkpoints.is_empty() => {
                let save = match end {
                    TurnEnd::Commit { .. } => TurnSave::Commit,
                    TurnEnd::Abort { .. } | TurnEnd::AbortEnded => TurnSave::Abort,
                };
                (Save::Asked { end }, Some(save))
            }
            save => (save, None),
        }
    }
}

impl TurnEnd {
    /// The speech has had its end: the save's answer only frees the partition.
    fn told(&self) -> bool {
        matches!(self, TurnEnd::AbortEnded)
    }
}

#[derive(Default)]
pub(crate) struct StorageTurns {
    pub enabled: bool,
    next_checkpoint: u64,
    next_turn: u64,
    /// Each started speech's turn until its save is answered, numbered in the order the speeches
    /// started: their order in the thread.
    turns: BTreeMap<u64, Turn>,
    /// Each turn's message heard so far.
    progress: BTreeMap<u64, lk_agents_abi::ChatMessage>,
    progress_dirty: BTreeSet<SpeechId>,
}

impl StorageTurns {
    pub(super) fn new(enabled: bool) -> Self {
        Self { enabled, ..Default::default() }
    }

    fn turn_of(&self, speech: SpeechId) -> Option<u64> {
        self.turns.iter().find(|(_, turn)| turn.speech == speech).map(|(n, _)| *n)
    }

    /// The speech has not reported how its turn ends.
    fn unreported(&self, speech: SpeechId) -> bool {
        self.turn_of(speech).is_some_and(|n| {
            matches!(self.turns[&n].save, Save::Taking { .. } | Save::Running { .. })
        })
    }

    fn running(&mut self, speech: SpeechId) -> Option<&mut Checkpoints> {
        let n = self.turn_of(speech)?;
        match &mut self.turns.get_mut(&n)?.save {
            Save::Running { checkpoints } => Some(checkpoints),
            _ => None,
        }
    }

    /// The turn's save was answered or failed: it is forgotten with what is keyed by it.
    fn remove(&mut self, n: u64) -> Option<Turn> {
        let turn = self.turns.remove(&n)?;
        self.progress.remove(&n);
        self.progress_dirty.remove(&turn.speech);
        Some(turn)
    }
}

impl VoiceDriver {
    pub(crate) fn checkpoint_reply_effect(&mut self, speech: SpeechId, effect: AgentRunEffect) {
        self.checkpoint_speech(speech, Some(effect));
    }

    pub(crate) fn storage_progress(&mut self, speech: SpeechId, item: lk_agents_abi::ChatMessage) {
        let storage = &mut self.session.storage;
        if let Some(n) = storage.turn_of(speech)
            && storage.unreported(speech)
        {
            storage.progress.insert(n, item);
            storage.progress_dirty.insert(speech);
            self.flush_progress(speech);
        }
    }

    /// One progress checkpoint at a time: the latest words go once the last one is answered.
    fn flush_progress(&mut self, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if storage.running(speech).is_some_and(|checkpoints| checkpoints.is_empty())
            && storage.progress_dirty.remove(&speech)
        {
            self.checkpoint_speech(speech, None);
        }
    }

    pub(crate) fn storage_step_spoken(&mut self, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if let Some(n) = storage.turn_of(speech) {
            storage.progress.remove(&n);
        }
        storage.progress_dirty.remove(&speech);
    }

    /// Saves the conversation so far for a running turn; `effect`, the reply's next step, runs
    /// once the checkpoint is answered. A turn still taking the partition holds the step until
    /// it is taken; one that has nothing left to save hands it on at once.
    fn checkpoint_speech(&mut self, speech: SpeechId, effect: Option<AgentRunEffect>) {
        let storage = &mut self.session.storage;
        if let Some(n) = storage.turn_of(speech)
            && let Save::Taking { held } = &mut storage.turns.get_mut(&n).expect("held").save
        {
            held.extend(effect);
            return;
        }
        storage.progress_dirty.remove(&speech);
        let checkpoint = storage.next_checkpoint;
        let Some(checkpoints) = storage.running(speech) else {
            return self.hand_on(speech, effect.into_iter().collect());
        };
        checkpoints.insert(checkpoint, effect);
        storage.next_checkpoint = checkpoint.checked_add(1).expect("checkpoint ids fit u64");
        let history = self.checkpoint_history();
        self.record_effect(VoiceEffect::StorageCheckpoint {
            checkpoint,
            speech,
            agent: self.activity.agent_name().to_owned(),
            history,
        });
    }

    /// The conversation as the session will hold it with what the turns have in flight: the
    /// words heard so far and the replies' tool records, each where a commit puts it.
    fn checkpoint_history(&self) -> Vec<ChatItem> {
        let progress = self.session.storage.progress.values().map(|m| ChatItem::Message(m.clone()));
        // Speech text comes from the delivery ledger: AgentRun may already hold generated text
        // that has not been heard; only tool records may lead delivery.
        let replies: Vec<(SpeechId, Vec<ChatItem>)> = self
            .checkpoint_reply_history()
            .map(|(speech, items)| {
                (speech, items.into_iter().filter(|i| i.as_message().is_none()).collect())
            })
            .collect();
        let calls = (replies.iter())
            .flat_map(|(_, items)| items)
            .filter(|item| !matches!(item, ChatItem::FunctionCallOutput(_)))
            .cloned();
        let held = self.session.chat_ctx.projected(progress.chain(calls));
        // An output lands as its commit stamps it: a cut turn's right after its call's group, any
        // other once its step's results are handled, after everything stamped before.
        let outputs = replies.into_iter().flat_map(|(speech, items)| {
            let cut = self.activity.outliving_tools.contains(&speech);
            let held = &held;
            items.into_iter().filter_map(move |item| {
                let ChatItem::FunctionCallOutput(mut output) = item else { return None };
                if !held.iter().any(|h| h.id() == output.id) {
                    output.created_at = if cut {
                        crate::agent::call_group_end(held, &output.call_id)
                            .unwrap_or(output.created_at)
                    } else {
                        output.created_at.max(self.step.wall())
                    };
                }
                Some(ChatItem::FunctionCallOutput(output))
            })
        });
        let outputs: Vec<ChatItem> = outputs.collect();
        let mut history = held;
        for output in outputs {
            crate::agent::commit_by_instant(&mut history, output);
        }
        history
    }

    /// The store answered one of a turn's requests. A failure ends the turn, and its speech with
    /// the storage error unless it already ended, then closes the call: a bound call does not go
    /// on without its thread.
    pub(crate) fn storage_answered(
        &mut self,
        now: Ts,
        speech: SpeechId,
        request: StorageRequest,
        result: Result<(), StorageError>,
    ) {
        let Some(n) = self.session.storage.turn_of(speech) else { return };
        if let Err(StorageError::Cancelled(_)) = &result
            && request == StorageRequest::Acquire
        {
            // The speech was cut before it took the partition: nothing was taken, nothing is
            // saved, and its reply goes on without its thread.
            // A turn that already reported keeps waiting on its save's answer.
            if matches!(self.session.storage.turns[&n].save, Save::Taking { .. })
                && let Some(Turn { save: Save::Taking { held }, .. }) =
                    self.session.storage.remove(n)
            {
                self.hand_on(speech, held);
            }
            return;
        }
        if let Err(error) = result {
            // A turn whose save is asked ends with that save's answer, which still comes; any
            // other failure ends the turn now.
            let asked = matches!(self.session.storage.turns[&n].save, Save::Asked { .. });
            if asked && !matches!(request, StorageRequest::Commit | StorageRequest::Abort) {
                return self.storage_failed(now, speech, error);
            }
            return self.turn_failed(now, n, error);
        }
        match request {
            StorageRequest::Acquire => {
                let turn = self.session.storage.turns.get_mut(&n).expect("the turn is held");
                if let Save::Taking { held } = &mut turn.save {
                    let held = std::mem::take(held);
                    turn.save = Save::Running { checkpoints: Checkpoints::new() };
                    self.flush_progress(speech);
                    for effect in held {
                        self.checkpoint_speech(speech, Some(effect));
                    }
                }
            }
            StorageRequest::Checkpoint(checkpoint) => {
                let turn = self.session.storage.turns.get_mut(&n).expect("the turn is held");
                let (Save::Running { checkpoints } | Save::Reported { checkpoints, .. }) =
                    &mut turn.save
                else {
                    return;
                };
                let step = checkpoints.remove(&checkpoint).flatten();
                self.hand_on(speech, step.into_iter().collect());
                self.flush_progress(speech);
            }
            StorageRequest::Commit | StorageRequest::Abort => {
                let turn = self.session.storage.remove(n).expect("the turn is held");
                if let Save::Asked { end } = turn.save {
                    self.speech_end_saved(now, speech, end);
                }
            }
        }
        self.ask_saves();
    }

    /// The reply's next steps go to the reply, which knows whether its speech was cut: dropped,
    /// the reply would wait on them forever. A closing call starts nothing new; the close ends
    /// its speeches at its bound.
    fn hand_on(&mut self, speech: SpeechId, steps: Vec<AgentRunEffect>) {
        if self.session.closing.is_none() && !steps.is_empty() {
            self.reply_effects_ready(speech, steps);
        }
    }

    /// The speech ends as its turn's save tells it.
    fn speech_end_saved(&mut self, now: Ts, speech: SpeechId, end: TurnEnd) {
        match end {
            TurnEnd::Commit { outcome, interrupted } => {
                self.activity.tools_in_flight.remove(&speech);
                self.finish_reply(now, speech, *outcome, interrupted);
            }
            TurnEnd::Abort { error } => self.generation_failed(now, speech, error),
            TurnEnd::AbortEnded => {}
        }
    }

    fn turn_failed(&mut self, now: Ts, n: u64, error: StorageError) {
        let turn = self.session.storage.remove(n).expect("the turn is held");
        let told = match &turn.save {
            Save::Reported { end, .. } | Save::Asked { end } => end.told(),
            Save::Taking { .. } | Save::Running { .. } => false,
        };
        if !told {
            let failure = ProviderError::Local(error.clone().into());
            self.generation_failed(now, turn.speech, Box::new(failure));
        }
        self.storage_failed(now, turn.speech, error);
        self.ask_saves();
    }

    /// Asks each reported turn's save whose checkpoints are answered.
    fn ask_saves(&mut self) {
        let numbers: Vec<u64> = self.session.storage.turns.keys().copied().collect();
        for n in numbers {
            let turns = &mut self.session.storage.turns;
            let Some(Turn { speech, save }) = turns.remove(&n) else { continue };
            let (save, asked) = save.ask();
            turns.insert(n, Turn { speech, save });
            match asked {
                Some(TurnSave::Commit) => {
                    let history = self.session.chat_ctx.items().cloned().collect();
                    self.record_effect(VoiceEffect::StorageCommit { speech, history });
                }
                Some(TurnSave::Abort) => self.record_effect(VoiceEffect::StorageAbort(speech)),
                None => {}
            }
        }
    }

    /// The reply reported how its turn ends. A speech with no turn held (its save already failed,
    /// or storage is off) ends at once.
    fn turn_ends(&mut self, now: Ts, speech: SpeechId, end: TurnEnd) {
        let storage = &mut self.session.storage;
        let Some(n) = storage.turn_of(speech) else {
            return self.speech_end_saved(now, speech, end);
        };
        let turn = storage.turns.get_mut(&n).expect("the turn is held");
        // A turn already ended (abandoned at its partition's bound) keeps that end.
        match &mut turn.save {
            Save::Running { checkpoints } => {
                let checkpoints = std::mem::take(checkpoints);
                turn.save = Save::Reported { end, checkpoints };
            }
            // The reply ended before the partition was taken: its held steps go with it.
            Save::Taking { .. } => {
                turn.save = Save::Reported { end, checkpoints: Checkpoints::new() };
            }
            Save::Reported { .. } | Save::Asked { .. } => {}
        }
        self.turn_reported(speech);
        self.storage_step_spoken(speech);
        self.ask_saves();
    }

    pub(crate) fn storage_settle(
        &mut self,
        now: Ts,
        speech: SpeechId,
        outcome: TurnOutcome,
        interrupted: bool,
    ) {
        self.turn_ends(now, speech, TurnEnd::Commit { outcome: Box::new(outcome), interrupted });
    }

    pub(crate) fn storage_abandon(&mut self, now: Ts, speech: SpeechId, error: Box<ProviderError>) {
        self.turn_ends(now, speech, TurnEnd::Abort { error });
    }

    pub(crate) fn storage_reloaded(&mut self, read: Vec<ChatItem>, history: Vec<ChatItem>) {
        if !self.session.chat_ctx.reload_transport(&read, history, self.activity.agent_name()) {
            return;
        }
        self.publish_event(SessionEvent::HistoryReloaded {
            items: self.session.chat_ctx.items().cloned().collect(),
        });
    }

    /// A started speech takes the partition: it holds it until its turn's save is answered.
    pub(crate) fn storage_started(&mut self, speech: SpeechId) {
        let storage = &mut self.session.storage;
        if storage.enabled && storage.turn_of(speech).is_none() {
            storage.next_turn += 1;
            let save = Save::Taking { held: Vec::new() };
            storage.turns.insert(storage.next_turn, Turn { speech, save });
        }
    }

    /// The speech's tasks reported its turn: the interruption watchdog, there for tasks that never
    /// report, no longer ends it: its save's answer does.
    fn turn_reported(&mut self, speech: SpeechId) {
        self.clear_deadline_if_set(Timer::PartitionRelease(speech));
        if self.clear_deadline_if_set(Timer::InterruptionTimeout(speech)) {
            self.record_trace(crate::trace::SpeechStep::WatchdogCleared { speech });
        }
    }

    /// The speech left the loop: one that never reported its turn releases the partition within
    /// [`PARTITION_RELEASE_BOUND`]. Not while its tools still run: they report it.
    pub(crate) fn storage_speech_released(&mut self, now: Ts, speech: SpeechId) {
        if self.session.storage.unreported(speech)
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
        let storage = &mut self.session.storage;
        let Some(n) = storage.turn_of(speech) else { return };
        let turn = storage.turns.get_mut(&n).expect("the turn is held");
        let checkpoints = match &mut turn.save {
            Save::Taking { .. } => Checkpoints::new(),
            Save::Running { checkpoints } => std::mem::take(checkpoints),
            Save::Reported { .. } | Save::Asked { .. } => return,
        };
        turn.save = Save::Reported { end: TurnEnd::AbortEnded, checkpoints };
        self.ask_saves();
    }

    /// A bound call cannot continue making effects after it loses durable ownership.
    pub(crate) fn storage_lost(&mut self, now: Ts, speech: SpeechId) {
        if self.session.storage.enabled {
            self.record_trace(crate::trace::SpeechStep::PartitionLost { speech });
            let lost = StorageError::LeaseLost("the voice partition lease was lost".into());
            self.storage_failed(now, speech, lost);
        }
    }

    /// The bound call closes with the storage error. Each turn still waits only on answers its
    /// store owes it, which still come.
    fn storage_failed(&mut self, now: Ts, speech: SpeechId, error: StorageError) {
        let event = crate::events::ErrorEvent::provider(
            lk_agents_abi::voice::events::ErrorSource::Storage,
            false,
            ProviderError::Local(error.into()),
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

#[cfg(test)]
#[path = "storage_tests.rs"]
mod tests;
