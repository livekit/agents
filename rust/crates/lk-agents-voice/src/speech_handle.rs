use crate::agent_activity::scheduling::SpeechId;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{ChatItem, Modality, Monotone};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

/// How long after an interruption the handle is force-completed (`SpeechHandle._cancel`). Without
/// it a wedged generation leaves the handle un-done and every `wait_for_playout()` hangs.
pub const INTERRUPTION_TIMEOUT: Duration = Duration::from_secs(5);

/// The `extra` key a tool sets to opt out of the self-wait guard.
pub const TOOL_NON_BLOCKING_EXTRA_KEY: &str = "__livekit_agents_tool_non_blocking";
/// `SpeechHandle.SPEECH_PRIORITY_{LOW,NORMAL,HIGH}`, as `(low, normal, high)`. Minted beside the
/// scheduler's heap key that reads them, so a host copy cannot drift and silently reorder.
/// No fork guard: `voice/speech_handle.py` reads these at import.
#[lk_agents_abi::host(export(pid = false))]
pub fn speech_priorities() -> (i32, i32, i32) {
    use crate::agent_activity::scheduling as s;
    (s::SPEECH_PRIORITY_LOW, s::SPEECH_PRIORITY_NORMAL, s::SPEECH_PRIORITY_HIGH)
}

/// `INTERRUPTION_TIMEOUT`, in seconds.
#[lk_agents_abi::host(export(pid = false))]
pub fn interruption_timeout() -> Duration {
    INTERRUPTION_TIMEOUT
}

/// `DEFAULT_INPUT_DETAILS.modality`: `InputDetails::default()`'s modality, as its wire string.
#[lk_agents_abi::host(export(pid = false))]
pub fn default_input_modality() -> &'static str {
    InputDetails::default().modality.as_str()
}

type Api = crate::agent_session::Tx<crate::agent_session::Command>;

/// `say()` and `generate_reply()` hand one back before the loop has created the speech; every sync
/// read answers as the reference's freshly constructed handle would.
pub struct HostSpeech {
    promise: Arc<SpeechPromise>,
    api: Api,
    /// A detached handle's id, given at construction; a loop-created speech's id is its own.
    detached_id: String,
}

/// A refusal on a host's speech handle; each `Display` is the `RuntimeError` it raises.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeechRefused {
    #[error(transparent)]
    Speech(#[from] SpeechError),
    #[error(transparent)]
    Promise(#[from] PromiseError),
    #[error(
        "this SpeechHandle was constructed directly, so no speech is behind it. In this port the \
         driver creates the speech; use `session.say()` or `session.generate_reply()`, which return \
         an attached handle."
    )]
    Detached,
    #[error(
        "this speech has not been created yet, so its interruptibility cannot be changed; pass \
         allow_interruptions= to say()/generate_reply() instead"
    )]
    NotCreated,
}

/// The exception a speech failed with: the host's own object when a provider raised one (read,
/// not released: a second `exception()` must still find it), else a `RuntimeError` of the message.
pub struct SpeechException(pub ProviderError);

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for SpeechException {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        let runtime_error = |msg: String| {
            pyo3::exceptions::PyRuntimeError::new_err(msg).value(py).clone().into_any()
        };
        Ok(match &self.0 {
            ProviderError::Host { error, .. } => match error
                .host_ref
                .clone()
                .and_then(|r| lk_agents_abi::python::resolve_captured(py, r))
            {
                Some(obj) => obj.into_bound(py),
                None => runtime_error(error.to_string()),
            },
            other => runtime_error(other.to_string()),
        })
    }
}

impl HostSpeech {
    pub fn pending(
        api: Api,
        rx: tokio::sync::oneshot::Receiver<Arc<SpeechHandle>>,
        requested_allow: Option<bool>,
    ) -> Self {
        let promise = SpeechPromise::pending(rx, requested_allow);
        HostSpeech { promise: Arc::new(promise), api, detached_id: String::new() }
    }

    pub fn from_core(handle: Arc<SpeechHandle>, api: Api) -> Self {
        HostSpeech {
            promise: Arc::new(SpeechPromise::ready(handle)),
            api,
            detached_id: String::new(),
        }
    }

    pub fn object(handle: &Arc<SpeechHandle>, api: &Api) -> lk_agents_abi::CoreObject {
        lk_agents_abi::CoreObject(Arc::new(HostSpeech::from_core(Arc::clone(handle), api.clone())))
    }

    pub fn peek(&self) -> Option<Arc<SpeechHandle>> {
        self.promise.peek()
    }

    fn attached(&self) -> Result<Arc<SpeechHandle>, SpeechRefused> {
        self.peek().ok_or(SpeechRefused::Detached)
    }
}

#[lk_agents_abi::host(object(name = "SpeechHandle", cell = "arc"))]
impl HostSpeech {
    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    #[host(new(kw_only, defaults(speech_id = String::new(), allow_interruptions = true)))]
    pub fn detached(speech_id: String, allow_interruptions: bool) -> Self {
        HostSpeech {
            promise: Arc::new(SpeechPromise::detached(allow_interruptions)),
            api: crate::agent_session::Tx::disconnected(),
            detached_id: speech_id,
        }
    }

    /// `speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
    /// the id it was constructed with.
    #[host(getter)]
    pub fn id(&self) -> String {
        self.peek().map_or_else(|| self.detached_id.clone(), |h| h.speech_id().to_owned())
    }

    #[host(getter)]
    pub fn num_steps(&self) -> u32 {
        self.peek().map_or(1, |h| h.num_steps())
    }

    /// `chat_items` (`speech_handle.py:156`): what this speech committed to the conversation.
    /// Empty before the speech is created; a copy, so it does not grow under a caller.
    #[host(getter)]
    pub fn chat_items(&self) -> Vec<ChatItem> {
        self.peek().map(|h| h.chat_items()).unwrap_or_default()
    }

    #[host(getter)]
    pub fn scheduled(&self) -> bool {
        self.peek().is_some_and(|h| h.scheduled())
    }

    /// `True` from the tick `interrupt()` was called, and once the loop has interrupted the speech
    /// for any reason (a interruption, a supersede, the close).
    #[host(getter)]
    pub fn interrupted(&self) -> bool {
        self.promise.interrupted()
    }

    #[host(method)]
    pub fn done(&self) -> bool {
        self.peek().is_some_and(|h| h.done())
    }

    /// `allow_interruptions`. While pending: the value the caller passed, or `True` when it passed
    /// none (the session default is resolved by the loop).
    #[host(getter)]
    pub fn allow_interruptions(&self) -> bool {
        self.promise.allow_interruptions()
    }

    /// The setter (`speech_handle.py:117`): sent to the loop, the one writer of a speech's
    /// interruption state, which journals it. Raises the reference's `RuntimeError` on an
    /// already-interrupted speech; the getter follows the loop's write by one wake.
    #[host(setter)]
    pub fn set_allow_interruptions(&self, value: bool) -> Result<(), SpeechRefused> {
        if !value {
            self.promise.check_can_disallow()?;
        }
        match self.peek() {
            Some(h) => {
                if !value && h.interrupted() {
                    return Err(SpeechError::AlreadyInterrupted.into());
                }
                // A closed loop has no speech left to protect.
                let _ = self.api.send(
                    crate::agent_session::SpeechCommand::SetAllowInterruptions {
                        speech: h.id(),
                        allow: value,
                    }
                    .into(),
                );
                Ok(())
            }
            None if self.promise.is_detached() => {
                self.promise.set_requested_allow(value);
                Ok(())
            }
            None => Err(SpeechRefused::NotCreated),
        }
    }

    /// The exception the speech failed with, as the original Python object.
    #[host(method)]
    pub fn exception(&self) -> Result<Option<SpeechException>, SpeechRefused> {
        let Some(h) = self.peek() else { return Ok(None) };
        Ok(h.exception()?.map(SpeechException))
    }

    /// `interrupt(force=)` (`speech_handle.py:185`). Synchronous; if the speech does not exist yet,
    /// a task sends the command once it does, so the interrupt is not lost.
    #[host(method(kw_only, defaults(force = false)))]
    pub fn interrupt(
        &self,
        force: bool,
        #[host(ambient)] spawner: Arc<dyn lk_agents_abi::Spawner>,
    ) -> Result<(), SpeechRefused> {
        match self.promise.interrupt(force)? {
            InterruptRequest::Nothing => {}
            InterruptRequest::Now(speech) => {
                // A closed loop has nothing left to interrupt.
                let _ = self.api.send(
                    crate::agent_session::SpeechCommand::InterruptSpeech { speech, force }.into(),
                );
            }
            InterruptRequest::WhenCreated => {
                let promise = Arc::clone(&self.promise);
                let api = self.api.clone();
                spawner.spawn(Box::pin(async move {
                    promise.interrupt_once_created(&api, force).await;
                }));
            }
        }
        Ok(())
    }

    /// `_hold_interruptions`: disallow interruptions until every hold taken is released.
    ///
    /// Counted, not a flag: an inline `AgentTask` and a filler may hold the same speech, and an
    /// inner release must not re-arm the outer. Raises the reference's `RuntimeError` when the
    /// speech is already interrupted, without counting the attempt.
    #[host(method)]
    pub fn hold_interruptions(&self) -> Result<(), SpeechRefused> {
        Ok(self.attached()?.hold_interruptions()?)
    }

    /// `_release_interruptions`. Idempotent past zero, like the reference's.
    #[host(method)]
    pub fn release_interruptions(&self) -> Result<(), SpeechRefused> {
        self.attached()?.release_interruptions();
        Ok(())
    }

    /// `wait_for_playout()`, and therefore `__await__`. The self-wait guard is not here: it reads
    /// the ambient tool frame, so it belongs on the shim's `__await__`.
    #[host(method)]
    pub async fn wait_for_playout(&self) -> Result<(), SpeechRefused> {
        self.promise.resolve().await?.wait_for_playout().await;
        Ok(())
    }

    /// `_wait_for_generation(step_idx)`: wait for one step of the turn, not the whole thing.
    /// `step=None` is the reference's `step_idx=-1`. No generation, no such step, and a
    /// never-created speech are distinct errors.
    #[host(method)]
    pub async fn wait_for_generation(&self, step: Option<usize>) -> Result<(), SpeechRefused> {
        let handle = self.promise.resolve().await?;
        match step {
            Some(step) => handle.wait_for_generation(step).await?,
            None => handle.wait_for_latest_generation().await?,
        }
        Ok(())
    }

    /// Resolve when the speech is interrupted, and only then: a speech that finishes normally never
    /// resolves this, so `wait_if_not_interrupted` can race it.
    #[host(method)]
    pub async fn wait_for_interrupted(&self) -> Result<(), SpeechRefused> {
        self.promise.resolve().await?.wait_for_interrupted().await;
        Ok(())
    }

    /// Resolve once the loop has created the speech and put it in the queue.
    #[host(method)]
    pub async fn wait_until_created(&self) -> Result<(), SpeechRefused> {
        self.promise.resolve().await?;
        Ok(())
    }
}

/// A [`SpeechId`] on the wire: `speech_{n}`. `Display` mints it and `FromStr` reads it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeechWireId(pub SpeechId);

impl std::fmt::Display for SpeechWireId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "speech_{}", self.0.0)
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

impl std::str::FromStr for SpeechWireId {
    type Err = NotASpeechId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.strip_prefix("speech_")
            .and_then(|n| n.parse::<u64>().ok())
            .map(|n| SpeechWireId(SpeechId(n)))
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
        let id = SpeechWireId(SpeechId(42));
        assert_eq!(id.to_string(), "speech_42");
        assert_eq!("speech_42".parse::<SpeechWireId>(), Ok(id));
        for bad in ["42", "speech_", "speech_x1", "speech_ab12cd"] {
            let err = bad.parse::<SpeechWireId>().unwrap_err();
            assert_eq!(err.to_string(), format!("not a driver speech id: {bad:?}"));
        }
    }
}

/// `DEFAULT_INPUT_DETAILS` is `InputDetails(modality="audio")`, which is [`Default`] here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InputDetails {
    pub modality: Modality,
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

/// How wide an interruption is. A tool's own `interrupt()` (or a session `Interrupt` while its tools
/// run) stops only the playing step, cleared at [`SpeechHandle::step`]; everything else is the
/// whole speech. Not [`Monotone`]: `Step` regresses to `None` at the step boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interruption {
    None,
    /// The step whose playout it cut, counting as [`SpeechHandle::num_steps`] does.
    Step(u32),
    Speech,
}

/// Proof that this speech has the floor, returned only by [`SpeechHandle::wait_for_authorization`].
/// Audio pushes take it by reference, so only the words leg can run ahead (preemptive generation).
#[derive(Debug)]
pub struct Authorized(());

/// Something the [`INTERRUPTION_TIMEOUT`] watchdog force-cancels (`SpeechHandle._cancel`).
pub trait Cancellable: Send + Sync {
    fn cancel(&self);
}

/// The real task object behind `speech_handle._tasks.append(task)`.
impl Cancellable for lk_agents_abi::AbortHandle {
    fn cancel(&self) {
        self.abort();
    }
}

/// `_allow_interruptions` and its hold count and restore value, in one cell so holds are atomic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interruptions {
    allow: bool,
    holds: u32,
    restore: bool,
}

struct Inner {
    error: Option<ProviderError>,
    chat_items: Vec<ChatItem>,
    tasks: Vec<Arc<dyn Cancellable>>,
    watchdog_armed: bool,
}

pub struct SpeechHandle {
    id: SpeechId,
    speech_id: Arc<str>,

    scheduled: Monotone<bool>,
    /// Level-triggered: a step-scoped interruption is cleared at the step boundary.
    interrupted: watch::Sender<Interruption>,
    done: Monotone<bool>,

    // Level-triggered, because `SpeechHandle._clear_authorization` clears it.
    authorized: watch::Sender<bool>,
    // `watch` so the hold/release read-modify-write runs under one write lock.
    interruptions: watch::Sender<Interruptions>,

    /// One cell per tool-loop step, addressed absolutely.
    generations: Mutex<Vec<Arc<Monotone<bool>>>>,
    num_steps: AtomicU32,

    inner: Mutex<Inner>,
}

/// By id, not pointer: ids are per-session and monotone.
impl PartialEq for SpeechHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for SpeechHandle {}

impl SpeechHandle {
    /// `id` is minted by the driver; the public `speech_<uuid>` comes from the host. Nothing reads
    /// `input_details` (always the default); the parameter stays for the fuzz crate's callers.
    pub fn new(
        id: SpeechId,
        speech_id: impl Into<Arc<str>>,
        allow_interruptions: bool,
        _input_details: InputDetails,
    ) -> Self {
        SpeechHandle {
            id,
            speech_id: speech_id.into(),
            scheduled: Monotone::new(false),
            interrupted: watch::Sender::new(Interruption::None),
            done: Monotone::new(false),
            authorized: watch::Sender::new(false),
            interruptions: watch::Sender::new(Interruptions {
                allow: allow_interruptions,
                holds: 0,
                // `_interruption_holds_restore = allow_interruptions` (`SpeechHandle.__init__`).
                restore: allow_interruptions,
            }),
            generations: Mutex::new(Vec::new()),
            // `_num_steps = 1`, so `RunContext`'s `_initial_step_idx` is 0.
            num_steps: AtomicU32::new(1),
            inner: Mutex::new(Inner {
                error: None,
                chat_items: Vec::new(),
                tasks: Vec::new(),
                watchdog_armed: false,
            }),
        }
    }

    pub fn id(&self) -> SpeechId {
        self.id
    }

    /// `SpeechHandle.id` -- the Python-visible `speech_<shortuuid>`.
    pub fn speech_id(&self) -> &str {
        &self.speech_id
    }

    pub fn num_steps(&self) -> u32 {
        self.num_steps.load(Ordering::Acquire)
    }

    /// `speech_handle._num_steps += 1` on the tool-loop back edge. Ends a step-scoped interruption
    /// and disarms its watchdog; a whole-speech interruption is untouched.
    pub fn step(&self) -> u32 {
        let n = self.num_steps.fetch_add(1, Ordering::AcqRel) + 1;
        let cleared = self.interrupted.send_if_modified(|i| match i {
            Interruption::Step(_) => {
                *i = Interruption::None;
                true
            }
            Interruption::None | Interruption::Speech => false,
        });
        if cleared {
            self.inner.lock().unwrap().watchdog_armed = false;
        }
        n
    }

    // ---------------------------------------------------------------- flags

    pub fn scheduled(&self) -> bool {
        self.scheduled.get()
    }

    /// Whether the speech or its playing step is interrupted right now.
    pub fn interrupted(&self) -> bool {
        !matches!(*self.interrupted.borrow(), Interruption::None)
    }

    /// Whether the whole speech was interrupted: decides whether a tool's follow-up reply is owed.
    pub fn interrupted_speech(&self) -> bool {
        matches!(*self.interrupted.borrow(), Interruption::Speech)
    }

    pub fn done(&self) -> bool {
        self.done.get()
    }

    /// `SpeechHandle._mark_scheduled`. Idempotent, like the `suppress(InvalidStateError)`.
    pub fn mark_scheduled(&self) {
        self.scheduled.set(true);
    }

    /// `_wait_for_scheduled`. The caller must re-check [`interrupted`](Self::interrupted) after:
    /// a cancelled speech is never marked scheduled, so the driver races this against the interrupt.
    pub async fn wait_for_scheduled(&self) {
        self.scheduled.wait_for(|s| *s).await;
    }

    pub async fn wait_for_interrupted(&self) {
        let mut rx = self.interrupted.subscribe();
        let _ = rx.wait_for(|i| !matches!(i, Interruption::None)).await;
    }

    pub async fn wait_for_done(&self) {
        self.done.wait_for(|d| *d).await;
    }

    // ------------------------------------------------------- interruptions

    pub fn allow_interruptions(&self) -> bool {
        self.interruptions.borrow().allow
    }

    /// The user may cut it now: not interrupted already, and interruptions allowed.
    pub fn interruptible(&self) -> bool {
        !self.interrupted() && self.allow_interruptions()
    }

    /// The `allow_interruptions` setter. Ignores the hold count, so a set during a hold is undone by
    /// the last release, as in the reference.
    pub fn set_allow_interruptions(&self, value: bool) -> Result<(), SpeechError> {
        if self.interrupted() && !value {
            return Err(SpeechError::AlreadyInterrupted);
        }
        self.interruptions.send_if_modified(|s| {
            // `!=` vs `==` is an equivalent mutant: nothing subscribes to `interruptions`.
            let changed = s.allow != value;
            s.allow = value;
            changed
        });
        Ok(())
    }

    /// Disallow interruptions until every hold taken here is released. Counted: parallel tool
    /// calls' inline tasks each hold, and an early release would let one task's sub-conversation
    /// interrupt the speech the rest are anchored to. The first holder's value is restored.
    pub fn hold_interruptions(&self) -> Result<(), SpeechError> {
        // Read before writing, as the setter's guard does.
        let interrupted = self.interrupted();
        let mut err = None;
        self.interruptions.send_if_modified(|s| {
            if s.holds == 0 {
                if interrupted {
                    // `self.allow_interruptions = False` raises, so the count is NOT incremented.
                    err = Some(SpeechError::AlreadyInterrupted);
                    return false;
                }
                s.restore = s.allow;
                s.allow = false;
            }
            s.holds += 1;
            true
        });
        err.map_or(Ok(()), Err)
    }

    /// `SpeechHandle._release_interruptions`.
    pub fn release_interruptions(&self) {
        let interrupted = self.interrupted();
        self.interruptions.send_if_modified(|s| {
            debug_assert!(s.holds > 0, "release_interruptions without a matching hold");
            s.holds = s.holds.saturating_sub(1);
            if s.holds != 0 {
                return false;
            }
            // `contextlib.suppress(RuntimeError)`: a forced interrupt leaves nothing to restore.
            if interrupted && !s.restore {
                return false;
            }
            // Equivalent mutant, as in the setter.
            let changed = s.allow != s.restore;
            s.allow = s.restore;
            changed
        });
    }

    // No `subscribe_allow_interruptions`: nothing in the reference awaits `allow_interruptions`.

    /// `interrupt(force=...)`. Returns whether it landed now; on `true` the driver arms
    /// [`INTERRUPTION_TIMEOUT`] and calls [`on_interruption_timeout`](Self::on_interruption_timeout).
    pub fn interrupt(&self, force: bool) -> Result<bool, SpeechError> {
        self.stop(force, Interruption::Speech)
    }

    /// `interrupt()` scoped to the playing step; [`step`](Self::step) clears it so the tool's
    /// follow-up plays on the same handle. Never forced.
    pub fn interrupt_step(&self) -> Result<bool, SpeechError> {
        self.stop(false, Interruption::Step(self.num_steps()))
    }

    /// The guards, then `_cancel` (`speech_handle.SpeechHandle._cancel`, `.wait_if_not_interrupted`).
    fn stop(&self, force: bool, how: Interruption) -> Result<bool, SpeechError> {
        // Before the `allow_interruptions` guard: a done non-interruptible speech does not raise.
        if self.interrupted() || self.done() {
            return Ok(false);
        }
        if !force && !self.allow_interruptions() {
            return Err(SpeechError::InterruptionsNotAllowed);
        }
        self.interrupted.send_replace(how);
        self.inner.lock().unwrap().watchdog_armed = true;
        Ok(true)
    }

    // ---------------------------------------------------------- watchdog

    /// The watchdog body (`SpeechHandle._cancel`): cancel every linked task, then mark done.
    /// Returns whether it fired.
    pub fn on_interruption_timeout(&self) -> bool {
        let tasks = {
            let mut inner = self.inner.lock().unwrap();
            if !inner.watchdog_armed {
                return false;
            }
            inner.watchdog_armed = false;
            std::mem::take(&mut inner.tasks)
        };
        for task in &tasks {
            task.cancel();
        }
        self.mark_done(None);
        true
    }

    /// `speech_handle._tasks.append(task)`. The handle only cancels these; done-when-all-done is
    /// the driver's.
    pub fn add_task(&self, task: Arc<dyn Cancellable>) {
        self.inner.lock().unwrap().tasks.push(task);
    }

    // -------------------------------------------------------- generations

    pub fn authorized(&self) -> bool {
        *self.authorized.borrow()
    }

    /// `_authorize_generation`: append a fresh generation cell, then set the authorize event.
    pub fn authorize_generation(&self) {
        self.generations.lock().unwrap().push(Arc::new(Monotone::new(false)));
        self.authorized.send_replace(true);
    }

    /// `_clear_authorization`, right after the gate; must not be a [`Monotone`].
    pub fn clear_authorization(&self) {
        self.authorized.send_replace(false);
    }

    /// `SpeechHandle._wait_for_authorization`.
    pub async fn wait_for_authorization(&self) -> Authorized {
        let mut rx = self.authorized.subscribe();
        // `wait_for` checks the current value first, so an earlier authorization isn't missed.
        let _ = rx.wait_for(|a| *a).await;
        Authorized(())
    }

    /// Wait for step `step`'s generation (absolute, from zero). `RunContext.wait_for_playout()`
    /// passes the step captured at tool-call time, so a long tool waits for its own step's playout.
    pub async fn wait_for_generation(&self, step: usize) -> Result<(), SpeechError> {
        let cell = {
            let gens = self.generations.lock().unwrap();
            if gens.is_empty() {
                return Err(SpeechError::NoActiveGeneration);
            }
            gens.get(step)
                .cloned()
                .ok_or(SpeechError::NoSuchGeneration { step, len: gens.len() })?
        };
        // The lock is released above, on purpose: never hold a lock across an await.
        cell.wait_for(|d| *d).await;
        Ok(())
    }

    /// `_wait_for_generation()` with `step_idx=-1`: the step currently running.
    pub async fn wait_for_latest_generation(&self) -> Result<(), SpeechError> {
        let cell = self.last_generation().ok_or(SpeechError::NoActiveGeneration)?;
        cell.wait_for(|d| *d).await;
        Ok(())
    }

    /// `_mark_generation_done`: always the last cell (only the newest generation is in flight).
    pub fn mark_generation_done(&self) -> Result<(), SpeechError> {
        self.last_generation().ok_or(SpeechError::NoGenerationToMark)?.set(true);
        Ok(())
    }

    /// The lock is released before any await.
    fn last_generation(&self) -> Option<Arc<Monotone<bool>>> {
        self.generations.lock().unwrap().last().cloned()
    }

    // ------------------------------------------------------------- done

    /// `_mark_done`. The error is kept out of the completion signal so awaiting never raises; see
    /// [`exception`](Self::exception).
    pub fn mark_done(&self, error: Option<ProviderError>) {
        let newly_done = !self.done();
        if newly_done {
            if let Some(err) = error {
                self.inner.lock().unwrap().error = Some(err);
            }
            self.done.set(true);
        }

        // Both run on every call; with no generation the mark refuses, which is fine here.
        let _ = self.mark_generation_done();
        self.inner.lock().unwrap().watchdog_armed = false;
    }

    /// `exception()`. [`SpeechError::NotDoneYet`] stands in for `asyncio.InvalidStateError`.
    pub fn exception(&self) -> Result<Option<ProviderError>, SpeechError> {
        if !self.done() {
            return Err(SpeechError::NotDoneYet);
        }
        Ok(self.inner.lock().unwrap().error.clone())
    }

    /// `wait_for_playout()`, and so `__await__`. The self-wait guard is the binding shim's.
    pub async fn wait_for_playout(&self) {
        self.wait_for_done().await;
    }

    /// `_item_added`.
    pub fn item_added(&self, item: &ChatItem) {
        self.inner.lock().unwrap().chat_items.push(item.clone());
    }

    /// `chat_items`, as a clone.
    pub fn chat_items(&self) -> Vec<ChatItem> {
        self.inner.lock().unwrap().chat_items.clone()
    }
}

// --- the promise of a speech ---

/// Why a [`SpeechPromise`] could not be resolved. Each `Display` is the message a host raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PromiseError {
    /// The session closed before the loop handled the command.
    #[error("the session ended before this speech was created, so it will never run")]
    Ended,
    /// A caller constructed a handle over no channel at all.
    #[error("this speech handle's promise was consumed without being fulfilled")]
    Consumed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptRequest {
    Nothing,
    /// The speech exists: send `InterruptSpeech` for it now.
    Now(SpeechId),
    /// Not created yet: send it once it exists ([`SpeechPromise::interrupt_once_created`]).
    WhenCreated,
}

/// `stash` is where `peek` leaves a speech taken while `resolve` holds the cell; `rx` is put back
/// if a `resolve` is dropped mid-await; `lost` is sticky. Otherwise the speech could be lost for
/// good (`done() == False` forever). `interrupt()` is synchronous (`speech_handle.py:185`):
/// recorded here.
pub struct SpeechPromise {
    cell: tokio::sync::OnceCell<Arc<SpeechHandle>>,
    rx: Mutex<Option<tokio::sync::oneshot::Receiver<Arc<SpeechHandle>>>>,
    stash: Mutex<Option<Arc<SpeechHandle>>>,
    lost: std::sync::atomic::AtomicBool,
    /// `allow_interruptions=` as passed; `None` is the session default.
    requested_allow: Mutex<Option<bool>>,
    /// Constructed directly (`SpeechHandle.create()`): no loop will ever fulfil it.
    detached: bool,
    /// `interrupt()` was called; `interrupted` reads `true` from this tick on.
    interrupt_requested: std::sync::atomic::AtomicBool,
}

/// `rx`, on loan to one `resolve`: given back on drop unless the await consumed it.
struct Borrowed<'a> {
    rx: Option<tokio::sync::oneshot::Receiver<Arc<SpeechHandle>>>,
    slot: &'a Mutex<Option<tokio::sync::oneshot::Receiver<Arc<SpeechHandle>>>>,
}

impl Drop for Borrowed<'_> {
    fn drop(&mut self) {
        if let Some(rx) = self.rx.take() {
            *lock(self.slot) = Some(rx);
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl SpeechPromise {
    fn base(
        cell: tokio::sync::OnceCell<Arc<SpeechHandle>>,
        rx: Option<tokio::sync::oneshot::Receiver<Arc<SpeechHandle>>>,
        requested_allow: Option<bool>,
        detached: bool,
    ) -> Self {
        SpeechPromise {
            cell,
            rx: Mutex::new(rx),
            stash: Mutex::new(None),
            lost: std::sync::atomic::AtomicBool::new(false),
            requested_allow: Mutex::new(requested_allow),
            detached,
            interrupt_requested: Default::default(),
        }
    }

    pub fn pending(
        rx: tokio::sync::oneshot::Receiver<Arc<SpeechHandle>>,
        requested_allow: Option<bool>,
    ) -> Self {
        Self::base(Default::default(), Some(rx), requested_allow, false)
    }

    pub fn detached(allow_interruptions: bool) -> Self {
        Self::base(Default::default(), None, Some(allow_interruptions), true)
    }

    pub fn is_detached(&self) -> bool {
        self.detached
    }

    /// `allow_interruptions = value` on a detached promise; the guard is the caller's.
    pub fn set_requested_allow(&self, value: bool) {
        *lock(&self.requested_allow) = Some(value);
    }

    /// A promise already kept: a speech the loop announced (`speech_created`, `run_settled`).
    pub fn ready(handle: Arc<SpeechHandle>) -> Self {
        Self::base(tokio::sync::OnceCell::new_with(Some(handle)), None, None, false)
    }

    fn mark_lost(&self) {
        self.lost.store(true, Ordering::Release);
    }

    fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Whether `interrupt()` was called on this promise, whatever the loop has done since.
    pub fn interrupt_requested(&self) -> bool {
        self.interrupt_requested.load(Ordering::Acquire)
    }

    pub fn interrupted(&self) -> bool {
        self.interrupt_requested() || self.peek().is_some_and(|h| h.interrupted())
    }

    /// `allow_interruptions`; while pending, the passed value or `true`.
    pub fn allow_interruptions(&self) -> bool {
        match self.peek() {
            Some(h) => h.allow_interruptions(),
            None => lock(&self.requested_allow).unwrap_or(true),
        }
    }

    /// The speech if the loop already produced it, without awaiting. Also fulfils the promise on
    /// the sync path; `resolve` reports a closed channel.
    pub fn peek(&self) -> Option<Arc<SpeechHandle>> {
        if let Some(handle) = self.cell.get() {
            return Some(Arc::clone(handle));
        }
        if let Some(handle) = lock(&self.stash).as_ref() {
            return Some(Arc::clone(handle));
        }
        let mut guard = lock(&self.rx);
        if let Some(rx) = guard.as_mut() {
            match rx.try_recv() {
                Ok(handle) => {
                    *guard = None;
                    if let Err(tokio::sync::SetError::InitializingError(handle)) =
                        self.cell.set(Arc::clone(&handle))
                    {
                        // Left before `guard` is released, so the initialiser finds it.
                        *lock(&self.stash) = Some(handle);
                    }
                    return Some(handle);
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    *guard = None;
                    self.mark_lost();
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }
        self.cell.get().cloned()
    }

    /// Wait for the loop to create the speech. One awaiter borrows the receiver; it is returned if
    /// the future is dropped.
    pub async fn resolve(&self) -> Result<Arc<SpeechHandle>, PromiseError> {
        self.cell
            .get_or_try_init(|| async {
                let taken = lock(&self.rx).take();
                let Some(rx) = taken else {
                    // `peek` got there first, or the channel closed, or a directly constructed handle.
                    if let Some(handle) = lock(&self.stash).take() {
                        return Ok(handle);
                    }
                    if self.is_lost() {
                        return Err(PromiseError::Ended);
                    }
                    return Err(PromiseError::Consumed);
                };
                let mut borrowed = Borrowed { rx: Some(rx), slot: &self.rx };
                // `&mut Receiver` is itself a future; a dropped poll leaves the receiver usable.
                let outcome = borrowed.rx.as_mut().expect("just set").await;
                borrowed.rx = None;
                match outcome {
                    Ok(handle) => Ok(handle),
                    Err(_) => {
                        self.mark_lost();
                        Err(PromiseError::Ended)
                    }
                }
            })
            .await
            .cloned()
    }

    /// The `allow_interruptions = false` guard, counting an interrupt requested from this promise.
    pub fn check_can_disallow(&self) -> Result<(), SpeechError> {
        if self.interrupt_requested() {
            return Err(SpeechError::AlreadyInterrupted);
        }
        Ok(())
    }

    /// `interrupt(force=)` (`speech_handle.py:185`), synchronous and in the reference's order. The
    /// caller sends `InterruptSpeech`; a pending speech is interrupted once it exists.
    pub fn interrupt(&self, force: bool) -> Result<InterruptRequest, SpeechError> {
        let attached = self.peek();
        if self.interrupt_requested()
            || attached.as_ref().is_some_and(|h| h.interrupted() || h.done())
        {
            return Ok(InterruptRequest::Nothing);
        }
        let allow = match &attached {
            Some(h) => h.allow_interruptions(),
            None => lock(&self.requested_allow).unwrap_or(true),
        };
        if !force && !allow {
            return Err(SpeechError::InterruptionsNotAllowed);
        }
        self.interrupt_requested.store(true, Ordering::Release);
        Ok(match attached {
            Some(h) => InterruptRequest::Now(h.id()),
            // Recorded above; a detached handle has no speech to interrupt, now or later.
            None if self.detached => InterruptRequest::Nothing,
            None => InterruptRequest::WhenCreated,
        })
    }

    /// Wait for the speech, then send its interruption through `api`. Run from a caller's task.
    pub async fn interrupt_once_created(
        &self,
        api: &crate::agent_session::Tx<crate::agent_session::Command>,
        force: bool,
    ) {
        if let Ok(h) = self.resolve().await {
            let _ = api.send(
                crate::agent_session::SpeechCommand::InterruptSpeech { speech: h.id(), force }
                    .into(),
            );
        }
    }
}

impl std::fmt::Debug for SpeechPromise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechPromise")
            .field("speech", &self.peek().map(|h| h.id()))
            .field("lost", &self.is_lost())
            .field("interrupt_requested", &self.interrupt_requested())
            .finish()
    }
}

impl std::fmt::Debug for SpeechHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechHandle")
            .field("id", &self.id)
            .field("speech_id", &self.speech_id)
            .field("scheduled", &self.scheduled())
            .field("interrupted", &self.interrupted())
            .field("done", &self.done())
            .field("allow_interruptions", &self.allow_interruptions())
            .field("num_steps", &self.num_steps())
            .finish_non_exhaustive()
    }
}
