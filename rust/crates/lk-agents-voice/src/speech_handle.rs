use crate::agent_activity::scheduling::SpeechId;

pub mod state;
use state::{
    Creation, CreationInterrupt, Effect as StateEffect, Input as StateInput, Reply, SpeechState,
};

use lk_agents_abi::ChatItem;
use lk_agents_abi::errors::ProviderError;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::watch;

/// The `extra` key marking a call that detached (`RunContext.update()`): its speech no longer
/// waits on it.
pub const TOOL_NON_BLOCKING_EXTRA_KEY: &str = "__livekit_agents_tool_non_blocking";
type Api = crate::agent_session::Tx;

/// A refusal on a host's speech handle; each `Display` is the `RuntimeError` it raises.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeechRefused {
    #[error(transparent)]
    Speech(#[from] SpeechError),
    #[error(transparent)]
    Creation(#[from] CreationError),
    #[error(
        "this SpeechHandle was constructed directly, so no speech is behind it. In this port the \
         driver creates the speech; use `session.say()` or `session.generate_reply()`, which return \
         an attached handle."
    )]
    Detached,
    #[error(
        "cannot call `SpeechHandle.wait_for_playout()` from inside the function tool `{name}` that \
         owns this SpeechHandle. This creates a circular wait: the speech handle is waiting for \
         the function tool to complete, while the function tool is simultaneously waiting for the \
         speech handle.\nTo wait for the assistant’s spoken response prior to running this tool, \
         use `RunContext.wait_for_playout()` instead."
    )]
    SelfWait { name: String },
}

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

/// Every way an operation on a [`SpeechHandle`] can refuse. `Display` is the reference exception's
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

    /// `IndexError`, as `self._generations[step_idx]`; see [`SpeechHandle::wait_for_generation`].
    #[error("generation step {step} out of range (the speech has {len} generations)")]
    NoSuchGeneration { step: usize, len: usize },

    /// `asyncio.InvalidStateError` from `exception()` (`speech_handle.SpeechHandle.exception`).
    #[error("SpeechHandle is not done yet")]
    NotDoneYet,
}

/// Why a handle has no speech to wait on. Each `Display` is the message a host raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CreationError {
    /// The session closed before the loop handled the command.
    #[error("the session ended before this speech was created, so it will never run")]
    Ended,
    /// A handle constructed directly: no loop will ever create its speech.
    #[error("this speech handle's promise was consumed without being fulfilled")]
    Consumed,
}

enum HandleInput {
    Accepted(StateInput),
    Complete(Option<ProviderError>),
    HostRequest { force: bool, created: bool },
    ValidateHostSetter { allow: bool },
}

struct Inner {
    state: SpeechState,
    chat_items: Vec<ChatItem>,
    /// A host `interrupt(force)` accepted before the loop created the speech: sent at creation.
    pending_interrupt: Option<bool>,
    /// The metrics of the user turn a preemptive reply was adopted for.
    user_metrics: Option<lk_agents_abi::MetricsReport>,
    /// The host's own object for the speech (its `SpeechHandle`), so every event, tool and hook
    /// that names the speech hands back the object `say()` / `generate_reply()` returned.
    host: Option<lk_agents_abi::HostRef>,
    /// The tools this speech waits on, as `(call_id, name)`: running and not yet detached.
    blocking_tools: Vec<(String, String)>,
}

/// `SpeechHandle`: one speech's id, state and outcome, shared by the loop that runs the speech and
/// the host that holds it. A host's `say()` gets it before the loop has created the speech; until
/// then every read answers as the reference's freshly constructed handle would.
pub struct SpeechHandle {
    /// Set by the loop as it creates the speech.
    id: OnceLock<SpeechId>,
    /// The id a host call's handle was given as the call returned; the loop creates it under it.
    reserved: Option<SpeechId>,
    /// `SpeechHandle.id`: `speech_{n}`, or what a detached handle was constructed with.
    speech_id: OnceLock<Arc<str>>,
    creation: watch::Sender<Creation>,
    /// The loop, for what the host asks of the speech.
    api: Api,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `interrupt()` was called from the host; `interrupted` reads `true` from this tick on, even
    /// before the loop has acted on it.
    interrupt_requested: AtomicBool,

    scheduled: Monotone<bool>,
    /// `_interrupt_fut`: one-shot, the whole speech (its tools and any follow-up step).
    interrupted: Monotone<bool>,
    done: Monotone<bool>,

    // Level-triggered, because `SpeechHandle._clear_authorization` clears it.
    authorized: watch::Sender<bool>,
    // `watch` so the hold/release read-modify-write runs under one write lock.
    interruptions: watch::Sender<bool>,

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// One cell per tool-loop step, addressed absolutely.
    generations: Mutex<Vec<Arc<Monotone<bool>>>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    num_steps: AtomicU32,

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    inner: Mutex<Inner>,
}

/// By id once created, so a speech is itself across handles; by identity before.
impl PartialEq for SpeechHandle {
    fn eq(&self, other: &Self) -> bool {
        match (self.id.get(), other.id.get()) {
            (Some(a), Some(b)) => a == b,
            _ => std::ptr::eq(self, other),
        }
    }
}

impl Eq for SpeechHandle {}

/// A level-triggered flag that only moves forward: `false` to `true`, never back.
struct Monotone<T>(watch::Sender<T>);

impl Monotone<bool> {
    fn new() -> Self {
        Monotone(watch::Sender::new(false))
    }

    fn get(&self) -> bool {
        *self.0.borrow()
    }

    /// Debug-panics on a regression: `authorized` and `allow_interruptions` go backwards and are
    /// plain watch cells.
    fn set(&self, v: bool) {
        self.0.send_modify(|cur| {
            debug_assert!(v >= *cur, "a speech handle's forward-only flag would regress");
            *cur = v;
        });
    }

    /// Resolves once set; a late waiter does not hang.
    async fn raised(&self) {
        // The handle holds the sender for the whole wait, so the channel never closes under it.
        let _ = self.0.subscribe().wait_for(|set| *set).await;
    }
}

use lk_agents_abi::lock;

impl SpeechHandle {
    /// Capture live request observations under the same lock as the accepted transition.
    /// This interpreter remains here until the concrete host handle moves to runtime.
    fn transition(&self, input: HandleInput) -> Result<Reply, SpeechError> {
        let (reply, generation_done) = {
            let mut inner = lock(&self.inner);
            let input = match input {
                HandleInput::Accepted(input) => input,
                HandleInput::Complete(error) => StateInput::Complete {
                    error,
                    interrupt_requested: self.interrupt_requested.load(Ordering::Acquire),
                },
                HandleInput::HostRequest { force, created } => StateInput::HostRequest {
                    force,
                    created,
                    already_requested: self.interrupt_requested.load(Ordering::Acquire),
                },
                HandleInput::ValidateHostSetter { allow } => StateInput::ValidateHostSetter {
                    allow,
                    interrupt_requested: self.interrupt_requested.load(Ordering::Acquire),
                },
            };
            let transition = inner.state.handle(input)?;
            let mut generation_done = None;
            for effect in transition.effects {
                match effect {
                    StateEffect::PublishAcceptedView(view) => {
                        self.interruptions.send_replace(view.allow_interruptions);
                        self.interrupted.set(view.interrupted);
                        self.done.set(view.done);
                    }
                    StateEffect::PublishRequestFlag(requested) => {
                        self.interrupt_requested.store(requested, Ordering::Release);
                    }
                    StateEffect::NewGeneration => {
                        self.generations.lock().unwrap().push(Arc::new(Monotone::new()));
                    }
                    StateEffect::PublishAuthorization(authorized) => {
                        self.authorized.send_replace(authorized);
                    }
                    StateEffect::PublishStep(step) => self.num_steps.store(step, Ordering::Release),
                    StateEffect::GenerationDone { step } => generation_done = Some(step),
                }
            }
            (transition.reply, generation_done)
        };
        if let Some(step) = generation_done {
            self.generation_cell(step).set(true);
        }
        Ok(reply)
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn base(
        creation: Creation,
        reserved: Option<SpeechId>,
        allow_interruptions: bool,
        api: Api,
    ) -> Self {
        SpeechHandle {
            id: OnceLock::new(),
            reserved,
            speech_id: OnceLock::new(),
            creation: watch::Sender::new(creation),
            api,
            interrupt_requested: AtomicBool::new(false),
            scheduled: Monotone::new(),
            interrupted: Monotone::new(),
            done: Monotone::new(),
            authorized: watch::Sender::new(false),
            interruptions: watch::Sender::new(allow_interruptions),
            generations: Mutex::new(Vec::new()),
            // `_num_steps = 1`, so `RunContext`'s `_initial_step_idx` is 0.
            num_steps: AtomicU32::new(1),
            inner: Mutex::new(Inner {
                state: SpeechState::new(allow_interruptions),
                chat_items: Vec::new(),
                pending_interrupt: None,
                user_metrics: None,
                host: None,
                blocking_tools: Vec::new(),
            }),
        }
    }

    /// A speech the loop creates on its own (a reply, a tool's follow-up).
    pub fn new(id: SpeechId, allow_interruptions: bool, api: Api) -> Self {
        let handle = SpeechHandle::base(Creation::Created, None, allow_interruptions, api);
        handle.create(id, allow_interruptions);
        handle
    }

    /// What a host's `say()` / `generate_reply()` answers, before the loop creates the speech:
    /// its id, and the interruptibility it was asked with or the session's as the call returned.
    pub fn pending(api: Api, id: SpeechId, allow_interruptions: bool) -> Arc<Self> {
        let handle = SpeechHandle::base(Creation::Pending, Some(id), allow_interruptions, api);
        let _ = handle.speech_id.set(id.to_string().into());
        Arc::new(handle)
    }

    /// The id the loop creates a host call's speech under.
    pub(crate) fn reserved(&self) -> Option<SpeechId> {
        self.reserved
    }

    /// The loop creates the speech behind this handle with its accepted interruptibility.
    pub fn create(&self, id: SpeechId, allow_interruptions: bool) {
        self.transition(HandleInput::Accepted(StateInput::ResetInterruptions {
            allow: allow_interruptions,
        }))
        .expect("resetting interruptions cannot refuse");
        let _ = self.speech_id.set(id.to_string().into());
        let _ = self.id.set(id);
    }

    /// The loop answered the host's call: with the speech it created over this handle, or, when
    /// the session ended first, not at all. Wakes whoever awaits the creation; an interrupt the
    /// host asked for meanwhile reaches the created speech now.
    pub(crate) fn answered(&self, created: bool) {
        let pending = {
            let mut inner = lock(&self.inner);
            self.creation.send_if_modified(|creation| creation.answered(created));
            inner.pending_interrupt.take()
        };
        if let (Some(force), Some(speech)) = (pending, self.created()) {
            self.send_interrupt(speech, force);
        }
    }

    fn send_interrupt(&self, speech: SpeechId, force: bool) {
        // A closed loop has nothing left to interrupt.
        let _ = self.api.send(crate::agent_session::Command::from(
            crate::agent_session::SpeechCommand::InterruptSpeech { speech, force },
        ));
    }

    pub fn id(&self) -> SpeechId {
        *self.id.get().expect("a speech the loop has created")
    }

    /// The handle as the host receives it in an event, a tool or a hook: this same object.
    pub fn object(self: &Arc<Self>) -> Arc<dyn lk_agents_abi::voice::speech_handle::Speech> {
        Arc::clone(self) as Arc<dyn lk_agents_abi::voice::speech_handle::Speech>
    }

    /// The id once the loop has created the speech.
    pub fn created(&self) -> Option<SpeechId> {
        self.id.get().copied()
    }

    /// `SpeechHandle.id`: `speech_{n}`.
    pub fn speech_id(&self) -> &str {
        self.speech_id.get().map_or("", |s| s)
    }

    /// The created speech, or why there is none to wait on.
    async fn wait_created(&self) -> Result<(), CreationError> {
        let mut rx = self.creation.subscribe();
        let creation = *rx.wait_for(Creation::settled).await.expect("held by self");
        creation.result()
    }

    fn attached(&self) -> Result<SpeechId, SpeechRefused> {
        self.created().ok_or(SpeechRefused::Detached)
    }

    /// `speech_handle._num_steps += 1` on the tool-loop back edge.
    pub fn step(&self) -> u32 {
        self.transition(HandleInput::Accepted(StateInput::AdvanceStep))
            .expect("advancing a speech step cannot refuse")
            .step()
    }

    /// The loop interrupted the speech: its playout, its tools and any follow-up step.
    pub fn interrupted(&self) -> bool {
        self.interrupted.get()
    }

    /// The loop refused the host's `interrupt()` (the speech disallowed it by then): the request
    /// is dropped, so the speech is not reported interrupted for it.
    pub(crate) fn interrupt_refused(&self) {
        self.transition(HandleInput::Accepted(StateInput::InterruptRefused))
            .expect("clearing a refused request cannot refuse");
    }

    /// A host `interrupt()` landed as the speech was marked done, once: the loop journals it.
    pub(crate) fn take_interrupted_at_done(&self) -> bool {
        self.transition(HandleInput::Accepted(StateInput::TakeInterruptedAtDone))
            .expect("reading the completion indication cannot refuse")
            .flag()
    }

    /// `SpeechHandle._mark_scheduled`. Idempotent, like the `suppress(InvalidStateError)`.
    pub fn mark_scheduled(&self) {
        self.scheduled.set(true);
    }

    /// `_wait_for_scheduled`. The caller must re-check [`interrupted`](Self::interrupted) after:
    /// a cancelled speech is never marked scheduled, so the driver races this against the interrupt.
    pub async fn wait_for_scheduled(&self) {
        self.scheduled.raised().await;
    }

    pub async fn wait_for_interrupted(&self) {
        self.interrupted.raised().await;
    }

    pub async fn wait_for_done(&self) {
        self.done.raised().await;
    }

    /// `allow_interruptions` as the loop last wrote it.
    pub fn allow_interruptions(&self) -> bool {
        *self.interruptions.borrow()
    }

    /// The user may cut it now: not interrupted already, and interruptions allowed.
    pub fn interruptible(&self) -> bool {
        lock(&self.inner).state.interruptible()
    }

    /// The `allow_interruptions` setter. Ignores the hold count, so a set during a hold is undone by
    /// the last release, as in the reference.
    pub fn set_allow_interruptions(&self, value: bool) -> Result<(), SpeechError> {
        self.transition(HandleInput::Accepted(StateInput::SetAllowInterruptions { allow: value }))?;
        Ok(())
    }

    /// Disallow interruptions until every hold taken here is released: counted, so one holder's
    /// release cannot lift another's. The first holder's value is restored.
    pub fn hold_interruptions(&self) -> Result<(), SpeechError> {
        self.transition(HandleInput::Accepted(StateInput::HoldInterruptions))?;
        Ok(())
    }

    /// `SpeechHandle._release_interruptions`.
    pub fn release_interruptions(&self) {
        self.transition(HandleInput::Accepted(StateInput::ReleaseInterruptions))
            .expect("releasing an interruption hold cannot refuse");
    }

    // No `subscribe_allow_interruptions`: nothing in the reference awaits `allow_interruptions`.

    /// `interrupt(force=...)`. Returns whether it landed now; on `true` the driver arms
    /// [`HOST_TEARDOWN_TIMEOUT`], after which the runtime cancels the speech's tasks.
    /// The guards, then `_cancel` (`speech_handle.SpeechHandle._cancel`).
    pub fn interrupt(&self, force: bool) -> Result<bool, SpeechError> {
        Ok(self.transition(HandleInput::Accepted(StateInput::Interrupt { force }))?.flag())
    }

    pub(crate) fn take_interruption_timeout(&self) -> bool {
        self.transition(HandleInput::Accepted(StateInput::TakeWatchdog))
            .expect("taking the interruption timeout cannot refuse")
            .flag()
    }

    pub fn authorized(&self) -> bool {
        lock(&self.inner).state.authorized()
    }

    /// `_authorize_generation`: append a fresh generation cell, then set the authorize event.
    pub fn authorize_generation(&self) {
        self.transition(HandleInput::Accepted(StateInput::AuthorizeGeneration))
            .expect("authorizing a generation cannot refuse");
    }

    /// `_clear_authorization`, right after the gate: the one flag that goes back to false.
    pub fn clear_authorization(&self) {
        self.transition(HandleInput::Accepted(StateInput::ClearAuthorization))
            .expect("clearing authorization cannot refuse");
    }

    /// `SpeechHandle._wait_for_authorization`.
    pub async fn wait_for_authorization(&self) {
        let mut rx = self.authorized.subscribe();
        // `wait_for` checks the current value first, so an earlier authorization isn't missed.
        let _ = rx.wait_for(|a| *a).await;
    }

    /// Whether step `step` (counting as [`num_steps`](Self::num_steps) does) was authorized to
    /// play: each authorization opens one generation.
    pub fn step_authorized(&self, step: u32) -> bool {
        lock(&self.inner).state.step_authorized(step)
    }

    /// Until [`step_authorized`](Self::step_authorized). Subscribed before the read: an
    /// authorization opens its generation before it is sent.
    pub async fn wait_for_step_authorized(&self, step: u32) {
        let mut rx = self.authorized.subscribe();
        while !self.step_authorized(step) {
            if rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Wait for step `step`'s generation (absolute, from zero). `RunContext.wait_for_playout()`
    /// passes the step captured at tool-call time, so a long tool waits for its own step's playout.
    pub async fn wait_for_generation(&self, step: usize) -> Result<(), SpeechError> {
        let step = lock(&self.inner).state.generation(step)?;
        self.generation_cell(step).raised().await;
        Ok(())
    }

    /// `_wait_for_generation()` with `step_idx=-1`: the step currently running.
    pub async fn wait_for_latest_generation(&self) -> Result<(), SpeechError> {
        let step = lock(&self.inner).state.latest_generation()?;
        self.generation_cell(step).raised().await;
        Ok(())
    }

    /// `_mark_generation_done`: always the last cell (only the newest generation is in flight).
    pub fn mark_generation_done(&self) -> Result<(), SpeechError> {
        self.transition(HandleInput::Accepted(StateInput::MarkGenerationDone))?;
        Ok(())
    }

    /// The lock is released before any await.
    fn generation_cell(&self, step: usize) -> Arc<Monotone<bool>> {
        self.generations
            .lock()
            .unwrap()
            .get(step)
            .cloned()
            .expect("the accepted generation was published")
    }

    /// `_mark_done`. The error is kept out of the completion signal so awaiting never raises; see
    /// [`exception`](Self::exception).
    ///
    /// A host `interrupt()` the loop has not acted on yet lands here, as it would have on the
    /// reference's handle (`interrupt()` before done is an interruption); see
    /// [`take_interrupted_at_done`](Self::take_interrupted_at_done).
    pub fn mark_done(&self, error: Option<ProviderError>) {
        self.transition(HandleInput::Complete(error)).expect("completing a speech cannot refuse");
    }

    /// `exception()`. [`SpeechError::NotDoneYet`] stands in for `asyncio.InvalidStateError`.
    pub fn exception(&self) -> Result<Option<ProviderError>, SpeechError> {
        lock(&self.inner).state.exception()
    }

    /// How a done speech ended, as `SpeechDone` reports it.
    pub fn outcome(&self) -> crate::events::SpeechOutcome {
        lock(&self.inner).state.outcome()
    }

    /// `wait_for_playout()`, and so `__await__`; the host's call adds the self-wait guard.
    pub async fn wait_for_playout(&self) {
        self.wait_for_done().await;
    }

    /// The speech waits on tool call `call_id` (`name`) until [`tool_returned`]: from its start
    /// until it answers, or detaches with its first update.
    ///
    /// [`tool_returned`]: Self::tool_returned
    pub fn tool_blocking(&self, call_id: &str, name: &str) {
        lock(&self.inner).blocking_tools.push((call_id.to_owned(), name.to_owned()));
    }

    pub fn tool_returned(&self, call_id: &str) {
        let mut inner = lock(&self.inner);
        if let Some(i) = inner.blocking_tools.iter().position(|(c, _)| c == call_id) {
            inner.blocking_tools.remove(i);
        }
    }

    /// The tool `origin` names, when this speech is waiting on it: awaiting the speech from inside
    /// it would wait forever.
    fn self_wait(&self, origin: Option<&str>) -> Result<(), SpeechRefused> {
        let Some(call_id) = origin.and_then(|o| o.strip_prefix("tool:")) else { return Ok(()) };
        match lock(&self.inner).blocking_tools.iter().find(|(c, _)| c == call_id) {
            Some((_, name)) => Err(SpeechRefused::SelfWait { name: name.clone() }),
            None => Ok(()),
        }
    }

    /// `_item_added`.
    pub fn item_added(&self, item: &ChatItem) {
        self.inner.lock().unwrap().chat_items.push(item.clone());
    }

    /// `preemptive.user_message.metrics = metrics_report`: the turn a preemptive reply answers.
    pub(crate) fn set_user_metrics(&self, metrics: lk_agents_abi::MetricsReport) {
        self.inner.lock().unwrap().user_metrics = Some(metrics);
    }

    pub fn user_metrics(&self) -> Option<lk_agents_abi::MetricsReport> {
        self.inner.lock().unwrap().user_metrics.clone()
    }
}

impl SpeechHandle {
    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    pub fn detached(speech_id: String, allow_interruptions: bool) -> Self {
        let handle = SpeechHandle::base(
            Creation::Detached,
            None,
            allow_interruptions,
            crate::agent_session::Tx::disconnected(),
        );
        let _ = handle.speech_id.set(speech_id.into());
        handle
    }

    /// The host's object for this speech, as it set it; `None` until then.
    pub fn host(&self) -> Option<lk_agents_abi::HostRef> {
        lock(&self.inner).host.clone()
    }

    /// Set by the host when it creates its handle over this one: from then on every event, tool
    /// and hook naming the speech hands back this object.
    pub fn set_host(&self, host: lk_agents_abi::HostRef) {
        lock(&self.inner).host = Some(host);
    }

    pub fn num_steps(&self) -> u32 {
        self.num_steps.load(Ordering::Acquire)
    }

    /// `chat_items` (`speech_handle.py:156`): what this speech committed to the conversation.
    /// A copy, so it does not grow under a caller.
    pub fn chat_items(&self) -> Vec<ChatItem> {
        self.inner.lock().unwrap().chat_items.clone()
    }

    pub fn scheduled(&self) -> bool {
        self.scheduled.get()
    }

    pub fn done(&self) -> bool {
        self.done.get()
    }
}

impl From<SpeechRefused> for lk_agents_abi::voice::speech_handle::SpeechRefused {
    fn from(refused: SpeechRefused) -> Self {
        Self(refused.to_string())
    }
}

impl lk_agents_abi::voice::speech_handle::Speech for SpeechHandle {}

/// `_core.SpeechHandle`: what `voice/speech_handle.py`'s `SpeechHandle` wraps. Before the loop has
/// created the speech every read is a fresh handle's.
impl lk_agents_abi::voice::speech_handle::SpeechHandle for SpeechHandle {
    fn new(speech_id: String, allow_interruptions: bool) -> Self {
        Self::detached(speech_id, allow_interruptions)
    }

    fn host_object(&self) -> Option<lk_agents_abi::HostRef> {
        self.host()
    }

    fn set_host_object(&self, host: lk_agents_abi::HostRef) {
        self.set_host(host);
    }

    /// Empty until the loop has minted `speech_{n}`; a detached handle answers the id it was
    /// constructed with.
    fn id(&self) -> String {
        self.speech_id().to_owned()
    }

    fn num_steps(&self) -> u32 {
        SpeechHandle::num_steps(self)
    }

    fn chat_items(&self) -> Vec<ChatItem> {
        SpeechHandle::chat_items(self)
    }

    fn scheduled(&self) -> bool {
        SpeechHandle::scheduled(self)
    }

    /// From the tick `interrupt()` was called, and once the loop has interrupted the speech for
    /// any reason (an interruption, a supersede, the close).
    fn interrupted(&self) -> bool {
        self.interrupt_requested.load(Ordering::Acquire) || SpeechHandle::interrupted(self)
    }

    fn done(&self) -> bool {
        SpeechHandle::done(self)
    }

    fn allow_interruptions(&self) -> bool {
        SpeechHandle::allow_interruptions(self)
    }

    fn set_allow_interruptions(&self, value: bool) -> Result<(), HostRefused> {
        self.transition(HandleInput::ValidateHostSetter { allow: value })?;
        if *self.creation.borrow() == Creation::Detached {
            return Ok(SpeechHandle::set_allow_interruptions(self, value)?);
        }
        let Some(speech) = self.created().or(self.reserved) else {
            return Err(SpeechRefused::Detached.into());
        };
        let _ = self.api.send(crate::agent_session::Command::from(
            crate::agent_session::SpeechCommand::SetAllowInterruptions { speech, allow: value },
        ));
        Ok(())
    }

    fn exception(&self) -> Result<Option<ProviderError>, HostRefused> {
        if self.created().is_none() {
            return Ok(None);
        }
        Ok(SpeechHandle::exception(self)?)
    }

    /// Synchronous and in the reference's order (`speech_handle.py:185`).
    fn interrupt(&self, force: bool) -> Result<(), HostRefused> {
        let created = self.created();
        if !self.transition(HandleInput::HostRequest { force, created: created.is_some() })?.flag()
        {
            return Ok(());
        }
        // Read under the lock `answered` takes, so a creation racing this call sends it once.
        let mut inner = lock(&self.inner);
        match self.creation.borrow().interrupt(self.created()) {
            CreationInterrupt::Send(speech) => self.send_interrupt(speech, force),
            CreationInterrupt::Ignore => {}
            CreationInterrupt::Wait => inner.pending_interrupt = Some(force),
        }
        Ok(())
    }

    fn hold_interruptions(&self) -> Result<(), HostRefused> {
        let speech = self.attached()?;
        self.transition(HandleInput::Accepted(StateInput::ValidateHostHold))?;
        let _ = self.api.send(crate::agent_session::Command::from(
            crate::agent_session::SpeechCommand::HoldInterruptions { speech, hold: true },
        ));
        Ok(())
    }

    fn release_interruptions(&self) -> Result<(), HostRefused> {
        let speech = self.attached()?;
        let _ = self.api.send(crate::agent_session::Command::from(
            crate::agent_session::SpeechCommand::HoldInterruptions { speech, hold: false },
        ));
        Ok(())
    }

    async fn wait_for_playout(&self, origin: Option<String>) -> Result<(), HostRefused> {
        self.self_wait(origin.as_deref())?;
        if self.wait_created().await.is_ok() {
            SpeechHandle::wait_for_playout(self).await;
        }
        Ok(())
    }

    async fn wait_for_generation(&self, step: Option<usize>) -> Result<(), HostRefused> {
        self.wait_created().await?;
        match step {
            Some(step) => SpeechHandle::wait_for_generation(self, step).await?,
            None => self.wait_for_latest_generation().await?,
        }
        Ok(())
    }

    async fn wait_for_interrupted(&self) -> Result<(), HostRefused> {
        self.wait_created().await?;
        SpeechHandle::wait_for_interrupted(self).await;
        Ok(())
    }

    async fn wait_until_created(&self) -> Result<(), HostRefused> {
        Ok(self.wait_created().await?)
    }
}

impl From<SpeechError> for HostRefused {
    fn from(error: SpeechError) -> Self {
        SpeechRefused::from(error).into()
    }
}

impl From<CreationError> for HostRefused {
    fn from(error: CreationError) -> Self {
        SpeechRefused::from(error).into()
    }
}

type HostRefused = lk_agents_abi::voice::speech_handle::SpeechRefused;

impl std::fmt::Debug for SpeechHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechHandle")
            .field("id", &self.created())
            .field("speech_id", &self.speech_id())
            .field("creation", &*self.creation.borrow())
            .field("scheduled", &self.scheduled())
            .field("interrupted", &self.interrupted())
            .field("done", &self.done())
            .field("allow_interruptions", &self.allow_interruptions())
            .field("num_steps", &self.num_steps())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "speech_handle_tests.rs"]
mod tests;
