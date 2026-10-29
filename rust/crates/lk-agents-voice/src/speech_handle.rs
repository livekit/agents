//! A speech's accepted state. The voice core owns it; the runtime's live `SpeechHandle` mirrors
//! what the core publishes ([`SpeechUpdate`]) and turns the host's requests into queued inputs.

use crate::agent_activity::scheduling::SpeechId;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{ChatItem, MetricsReport};

/// The `extra` key marking a call that detached (`RunContext.update()`): its speech no longer
/// waits on it.
pub const TOOL_NON_BLOCKING_EXTRA_KEY: &str = "__livekit_agents_tool_non_blocking";

/// A [`SpeechId`] on the wire, `speech_{n}`: `Display` mints it and `FromStr` reads it back.
impl std::fmt::Display for SpeechId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "speech_{}", self.0)
    }
}

/// A string that is not a `speech_{n}`. `Display` is the binding's `ValueError` text, verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotASpeechId(pub String);

impl std::fmt::Display for NotASpeechId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not a driver speech id: {:?}", self.0)
    }
}

impl std::error::Error for NotASpeechId {}

impl std::str::FromStr for SpeechId {
    type Err = NotASpeechId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.strip_prefix("speech_")
            .and_then(|n| n.parse::<u64>().ok())
            .map(SpeechId)
            .ok_or_else(|| NotASpeechId(s.to_owned()))
    }
}

#[cfg(test)]
mod wire_id_tests {
    use super::*;

    // Catches: the mint and the parse drifting apart; a non-round-tripping id is a filler that
    // never fires.
    #[test]
    fn a_wire_id_round_trips_and_a_foreign_one_is_named() {
        let id = SpeechId(42);
        assert_eq!(id.to_string(), "speech_42");
        assert_eq!("speech_42".parse::<SpeechId>(), Ok(id));
        for bad in ["42", "speech_", "speech_x1", "speech_ab12cd"] {
            let err = bad.parse::<SpeechId>().unwrap_err();
            assert_eq!(err.to_string(), format!("not a driver speech id: {bad:?}"));
        }
    }
}

/// Every way an operation on a speech can refuse. `Display` is the reference exception's
/// message verbatim (public API, re-raised by the binding).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeechError {
    /// `RuntimeError` from `interrupt()` (`speech_handle.SpeechHandle.interrupt`).
    #[error("This generation handle does not allow interruptions")]
    InterruptionsNotAllowed,

    /// `RuntimeError` from the `allow_interruptions` setter; also what `_hold_interruptions` propagates.
    #[error("Cannot set allow_interruptions to False, the SpeechHandle is already interrupted")]
    AlreadyInterrupted,

    /// `RuntimeError` from `_wait_for_generation` (`speech_handle.SpeechHandle._wait_for_generation`).
    #[error("cannot use wait_for_generation: no active generation is running.")]
    NoActiveGeneration,

    /// `RuntimeError` from `_mark_generation_done` (`speech_handle.SpeechHandle._mark_generation_done`).
    #[error("cannot use mark_generation_done: no active generation is running.")]
    NoGenerationToMark,

    /// `IndexError`, as `self._generations[step_idx]`; see [`SpeechView::generation`].
    #[error("generation step {step} out of range (the speech has {len} generations)")]
    NoSuchGeneration { step: usize, len: usize },

    /// `asyncio.InvalidStateError` from `exception()` (`speech_handle.SpeechHandle.exception`).
    #[error("SpeechHandle is not done yet")]
    NotDoneYet,
}

/// What the core publishes about a speech, in order; the runtime's handle applies each.
#[derive(Debug, Clone)]
pub enum SpeechUpdate {
    /// The loop created the speech: over the host's handle for its id, else a new one.
    Created,
    View(SpeechView),
    /// `_item_added`.
    ItemAdded(ChatItem),
    /// `preemptive.user_message.metrics = metrics_report`: the turn a preemptive reply answers.
    UserMetrics(MetricsReport),
    /// The loop refused the host's `interrupt()`: the request it recorded is dropped.
    InterruptRefused,
    /// The loop let the speech go; `say()`'s unread stream goes with it.
    Released,
}

/// The speech as the core accepted it: what every view of it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechView {
    pub allow_interruptions: bool,
    pub interrupted: bool,
    pub done: bool,
    pub error: Option<ProviderError>,
    pub scheduled: bool,
    /// `_num_steps`: 1 until the tool loop's first back edge.
    pub num_steps: u32,
    /// One per authorized step, addressed absolutely: `true` once its generation is done.
    pub generations: Vec<bool>,
}

impl SpeechView {
    pub fn new(allow_interruptions: bool) -> Self {
        SpeechView {
            allow_interruptions,
            interrupted: false,
            done: false,
            error: None,
            scheduled: false,
            num_steps: 1,
            generations: Vec::new(),
        }
    }

    /// The host's `interrupt(force)` against this view, `requested` already or not: whether it
    /// records a new request for the core, in the reference's order (`speech_handle.py:185`).
    /// Interrupted or done already, it is a no-op; disallowed, it raises unless forced.
    pub fn host_interrupt(&self, requested: bool, force: bool) -> Result<bool, SpeechError> {
        if requested || self.interrupted || self.done {
            return Ok(false);
        }
        if !force && !self.allow_interruptions {
            return Err(SpeechError::InterruptionsNotAllowed);
        }
        Ok(true)
    }

    /// The host's `allow_interruptions = allow`: refused once the speech is (being) interrupted.
    pub fn host_set_allow_interruptions(
        &self,
        allow: bool,
        requested: bool,
    ) -> Result<(), SpeechError> {
        if !allow && (requested || self.interrupted) {
            return Err(SpeechError::AlreadyInterrupted);
        }
        Ok(())
    }

    /// The host's `_hold_interruptions`: refused once the speech is interrupted.
    pub fn host_hold_interruptions(&self) -> Result<(), SpeechError> {
        if self.interrupted {
            return Err(SpeechError::AlreadyInterrupted);
        }
        Ok(())
    }

    /// `interrupted` as the host reads it: from its own request on, and once the core interrupted
    /// the speech. Forward-only, like the reference's `_interrupt_fut`: a request is only dropped
    /// when the core refuses it.
    pub fn host_interrupted(&self, requested: bool) -> bool {
        self.interrupted || requested
    }

    /// The user may cut it now: not interrupted already, and interruptions allowed.
    pub fn interruptible(&self) -> bool {
        !self.interrupted && self.allow_interruptions
    }

    /// Whether step `step` (counting as `num_steps` does) was authorized to play: each
    /// authorization opens one generation.
    pub fn step_authorized(&self, step: u32) -> bool {
        self.generations.len() >= step as usize
    }

    /// Step `step`'s generation (absolute, from zero), as `_wait_for_generation(step_idx)` reads it.
    pub fn generation(&self, step: usize) -> Result<usize, SpeechError> {
        let len = self.generations.len();
        if len == 0 {
            return Err(SpeechError::NoActiveGeneration);
        }
        if step >= len {
            return Err(SpeechError::NoSuchGeneration { step, len });
        }
        Ok(step)
    }

    /// `_wait_for_generation()` with `step_idx=-1`: the step currently running.
    pub fn latest_generation(&self) -> Result<usize, SpeechError> {
        self.generations.len().checked_sub(1).ok_or(SpeechError::NoActiveGeneration)
    }

    /// `exception()`. [`SpeechError::NotDoneYet`] stands in for `asyncio.InvalidStateError`.
    pub fn exception(&self) -> Result<Option<ProviderError>, SpeechError> {
        if self.done { Ok(self.error.clone()) } else { Err(SpeechError::NotDoneYet) }
    }

    /// How a done speech ended, as `SpeechDone` reports it.
    pub fn outcome(&self) -> crate::events::SpeechOutcome {
        use crate::events::SpeechOutcome;
        if self.error.is_some() && self.done {
            SpeechOutcome::Failed
        } else if self.interrupted {
            SpeechOutcome::Interrupted
        } else {
            SpeechOutcome::Completed
        }
    }
}

/// One speech's interruption, authorization, step and completion state, owned by the core. Every
/// change is published: [`take_updates`](Self::take_updates) hands the runtime what changed.
#[derive(Debug, Clone)]
pub struct SpeechState {
    view: SpeechView,
    holds: u32,
    /// The first holder's `allow_interruptions`, restored by the last release.
    restore: bool,
    /// An interruption landed and its watchdog has not fired or been disarmed.
    watchdog_armed: bool,
    /// The ids of what this speech committed to the conversation, in order.
    item_ids: Vec<String>,
    events: Vec<SpeechUpdate>,
    changed: bool,
}

impl SpeechState {
    /// The loop creates the speech with its accepted interruptibility.
    pub fn new(allow_interruptions: bool) -> Self {
        SpeechState {
            view: SpeechView::new(allow_interruptions),
            holds: 0,
            restore: allow_interruptions,
            watchdog_armed: false,
            item_ids: Vec::new(),
            events: vec![SpeechUpdate::Created],
            changed: true,
        }
    }

    pub fn view(&self) -> &SpeechView {
        &self.view
    }

    pub fn interrupted(&self) -> bool {
        self.view.interrupted
    }

    pub fn done(&self) -> bool {
        self.view.done
    }

    pub fn allow_interruptions(&self) -> bool {
        self.view.allow_interruptions
    }

    pub fn interruptible(&self) -> bool {
        self.view.interruptible()
    }

    pub fn scheduled(&self) -> bool {
        self.view.scheduled
    }

    pub fn num_steps(&self) -> u32 {
        self.view.num_steps
    }

    pub fn step_authorized(&self, step: u32) -> bool {
        self.view.step_authorized(step)
    }

    pub fn exception(&self) -> Result<Option<ProviderError>, SpeechError> {
        self.view.exception()
    }

    pub fn outcome(&self) -> crate::events::SpeechOutcome {
        self.view.outcome()
    }

    /// What changed since the last call, in order: the events, then the view.
    pub fn take_updates(&mut self) -> Vec<SpeechUpdate> {
        let mut updates = std::mem::take(&mut self.events);
        if std::mem::take(&mut self.changed) {
            updates.push(SpeechUpdate::View(self.view.clone()));
        }
        updates
    }

    fn change(&mut self) -> &mut SpeechView {
        self.changed = true;
        &mut self.view
    }

    /// `speech_handle._num_steps += 1` on the tool-loop back edge, which the reply reports.
    pub fn advance_step(&mut self) -> u32 {
        let view = self.change();
        view.num_steps += 1;
        view.num_steps
    }

    /// `_authorize_generation`: the next step may play, with a fresh generation.
    pub fn authorize_generation(&mut self) {
        self.change().generations.push(false);
    }

    /// `_mark_generation_done`: always the newest generation (only it is in flight).
    pub fn mark_generation_done(&mut self) -> Result<(), SpeechError> {
        let last = self.change().generations.last_mut().ok_or(SpeechError::NoGenerationToMark)?;
        *last = true;
        Ok(())
    }

    /// `SpeechHandle._mark_scheduled`. Idempotent, like the `suppress(InvalidStateError)`.
    pub fn mark_scheduled(&mut self) {
        if !self.view.scheduled {
            self.change().scheduled = true;
        }
    }

    /// `interrupt(force=...)`: the guards, then `_cancel`. Returns whether it landed now; on
    /// `true` the driver arms the interruption watchdog.
    pub fn interrupt(&mut self, force: bool) -> Result<bool, SpeechError> {
        if self.view.interrupted || self.view.done {
            return Ok(false);
        }
        if !force && !self.view.allow_interruptions {
            return Err(SpeechError::InterruptionsNotAllowed);
        }
        self.change().interrupted = true;
        self.watchdog_armed = true;
        Ok(true)
    }

    /// The loop refused the host's `interrupt()` (the speech disallowed it by then).
    pub fn interrupt_refused(&mut self) {
        self.events.push(SpeechUpdate::InterruptRefused);
    }

    /// The watchdog an interruption armed fired: `true` once, if it was still armed.
    pub fn take_interruption_timeout(&mut self) -> bool {
        std::mem::take(&mut self.watchdog_armed)
    }

    /// The `allow_interruptions` setter. Ignores the hold count, so a set during a hold is undone
    /// by the last release, as in the reference.
    pub fn set_allow_interruptions(&mut self, allow: bool) -> Result<(), SpeechError> {
        if self.view.interrupted && !allow {
            return Err(SpeechError::AlreadyInterrupted);
        }
        self.change().allow_interruptions = allow;
        Ok(())
    }

    /// Disallow interruptions until every hold taken is released: counted, so one holder's
    /// release cannot lift another's. The first holder's value is restored.
    pub fn hold_interruptions(&mut self) -> Result<(), SpeechError> {
        if self.holds == 0 {
            if self.view.interrupted {
                return Err(SpeechError::AlreadyInterrupted);
            }
            self.restore = self.view.allow_interruptions;
            self.change().allow_interruptions = false;
        }
        self.holds += 1;
        Ok(())
    }

    /// `SpeechHandle._release_interruptions`.
    pub fn release_interruptions(&mut self) {
        if self.holds == 0 {
            return;
        }
        self.holds -= 1;
        if self.holds == 0 && !(self.view.interrupted && !self.restore) {
            let restore = self.restore;
            self.change().allow_interruptions = restore;
        }
    }

    /// `_mark_done`: the first error stays. The newest generation is done with the speech.
    pub fn mark_done(&mut self, error: Option<ProviderError>) {
        let view = self.change();
        if !view.done {
            if error.is_some() {
                view.error = error;
            }
            view.done = true;
        }
        if let Some(last) = view.generations.last_mut() {
            *last = true;
        }
        self.watchdog_armed = false;
    }

    pub fn item_ids(&self) -> &[String] {
        &self.item_ids
    }

    /// `_item_added`.
    pub fn item_added(&mut self, item: &ChatItem) {
        self.item_ids.push(item.id().to_owned());
        self.events.push(SpeechUpdate::ItemAdded(item.clone()));
    }

    pub fn set_user_metrics(&mut self, metrics: MetricsReport) {
        self.events.push(SpeechUpdate::UserMetrics(metrics));
    }
}

#[cfg(test)]
#[path = "speech_handle_tests.rs"]
mod tests;
