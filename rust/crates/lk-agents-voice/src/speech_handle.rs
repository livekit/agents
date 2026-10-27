use crate::agent_activity::scheduling::SpeechId;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{ChatItem, Modality, Monotone};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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

/// `DEFAULT_INPUT_DETAILS.modality`, as its wire string.
#[lk_agents_abi::host(export(pid = false))]
pub fn default_input_modality() -> &'static str {
    Modality::default().as_str()
}

type Api = crate::agent_session::Tx;

/// A refusal on a host's speech handle; each `Display` is the `RuntimeError` it raises.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
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

/// `_allow_interruptions` and its hold count and restore value, in one cell so holds are atomic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interruptions {
    allow: bool,
    holds: u32,
    restore: bool,
}

/// Whether the host's call was answered: `say()` and `generate_reply()` return before the loop
/// creates the speech, and a handle constructed directly never gets one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Creation {
    Pending,
    Created,
    /// The session ended before the loop created the speech.
    Lost,
    Detached,
}

struct Inner {
    error: Option<ProviderError>,
    chat_items: Vec<ChatItem>,
    /// What the [`INTERRUPTION_TIMEOUT`] watchdog cancels (`speech_handle._tasks`).
    tasks: Vec<lk_agents_abi::AbortHandle>,
    watchdog_armed: bool,
    /// The metrics of the user turn a preemptive reply was adopted for.
    user_metrics: Option<lk_agents_abi::MetricsReport>,
    /// The host's own object for the speech (its `SpeechHandle`), so every event, tool and hook
    /// that names the speech hands back the object `say()` / `generate_reply()` returned.
    host: Option<lk_agents_abi::HostRef>,
}

/// `SpeechHandle`: one speech's id, state and outcome, shared by the loop that runs the speech and
/// the host that holds it. A host's `say()` gets it before the loop has created the speech; until
/// then every read answers as the reference's freshly constructed handle would.
pub struct SpeechHandle {
    /// Minted by the loop as it creates the speech.
    id: OnceLock<SpeechId>,
    /// `SpeechHandle.id`: `speech_{n}`, or what a detached handle was constructed with.
    speech_id: OnceLock<Arc<str>>,
    creation: watch::Sender<Creation>,
    /// The loop, for what the host asks of the speech.
    api: Api,
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
    interruptions: watch::Sender<Interruptions>,

    /// One cell per tool-loop step, addressed absolutely.
    generations: Mutex<Vec<Arc<Monotone<bool>>>>,
    num_steps: AtomicU32,

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

use lk_agents_abi::lock;

impl SpeechHandle {
    fn base(creation: Creation, allow_interruptions: bool, api: Api) -> Self {
        SpeechHandle {
            id: OnceLock::new(),
            speech_id: OnceLock::new(),
            creation: watch::Sender::new(creation),
            api,
            interrupt_requested: AtomicBool::new(false),
            scheduled: Monotone::new(false),
            interrupted: Monotone::new(false),
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
                user_metrics: None,
                host: None,
            }),
        }
    }

    /// A speech the loop creates on its own (a reply, a tool's follow-up).
    pub fn new(id: SpeechId, allow_interruptions: bool, api: Api) -> Self {
        let handle = SpeechHandle::base(Creation::Created, allow_interruptions, api);
        handle.create(id, allow_interruptions);
        handle
    }

    /// What a host's `say()` / `generate_reply()` answers, before the loop creates the speech.
    /// `allow_interruptions` is what was passed; `None` reads `True` until the loop resolves it.
    pub fn pending(api: Api, allow_interruptions: Option<bool>) -> Arc<Self> {
        Arc::new(SpeechHandle::base(Creation::Pending, allow_interruptions.unwrap_or(true), api))
    }

    /// The loop creates the speech behind this handle: its id, and its resolved interruptibility.
    pub fn create(&self, id: SpeechId, allow_interruptions: bool) {
        // Untouched since `pending`: the host can neither set nor hold it before this.
        self.interruptions.send_replace(Interruptions {
            allow: allow_interruptions,
            holds: 0,
            restore: allow_interruptions,
        });
        let _ = self.speech_id.set(id.to_string().into());
        let _ = self.id.set(id);
    }

    /// The loop answered the host's call: with the speech it created over this handle, or, when
    /// the session ended first, not at all. Wakes whoever awaits the creation.
    pub(crate) fn answered(&self, created: bool) {
        self.creation.send_if_modified(|c| {
            let pending = *c == Creation::Pending;
            if pending {
                *c = if created { Creation::Created } else { Creation::Lost };
            }
            pending
        });
    }

    pub fn id(&self) -> SpeechId {
        *self.id.get().expect("a speech the loop has created")
    }

    /// The handle as the host receives it in an event, a tool or a hook: this same object.
    pub fn object(self: &Arc<Self>) -> lk_agents_abi::CoreObject {
        lk_agents_abi::CoreObject(Arc::clone(self) as Arc<dyn std::any::Any + Send + Sync>)
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
        let creation = *rx.wait_for(|c| *c != Creation::Pending).await.expect("held by self");
        match creation {
            Creation::Created => Ok(()),
            Creation::Lost => Err(CreationError::Ended),
            Creation::Pending | Creation::Detached => Err(CreationError::Consumed),
        }
    }

    fn attached(&self) -> Result<SpeechId, SpeechRefused> {
        self.created().ok_or(SpeechRefused::Detached)
    }

    /// `speech_handle._num_steps += 1` on the tool-loop back edge.
    pub fn step(&self) -> u32 {
        self.num_steps.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// The loop interrupted the speech: its playout, its tools and any follow-up step.
    pub fn interrupted(&self) -> bool {
        self.interrupted.get()
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
        self.interrupted.wait_for(|i| *i).await;
    }

    pub async fn wait_for_done(&self) {
        self.done.wait_for(|d| *d).await;
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

    /// Disallow interruptions until every hold taken here is released: counted, so one holder's
    /// release cannot lift another's. The first holder's value is restored.
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
    /// The guards, then `_cancel` (`speech_handle.SpeechHandle._cancel`).
    pub fn interrupt(&self, force: bool) -> Result<bool, SpeechError> {
        // Before the `allow_interruptions` guard: a done non-interruptible speech does not raise.
        if self.interrupted() || self.done() {
            return Ok(false);
        }
        if !force && !self.allow_interruptions() {
            return Err(SpeechError::InterruptionsNotAllowed);
        }
        self.interrupted.set(true);
        self.inner.lock().unwrap().watchdog_armed = true;
        Ok(true)
    }

    /// The watchdog body (`SpeechHandle._cancel`): cancel every linked task, then mark done.
    /// Returns whether it fired.
    pub fn on_interruption_timeout(&self) -> bool {
        if !std::mem::replace(&mut self.inner.lock().unwrap().watchdog_armed, false) {
            return false;
        }
        self.cancel_tasks();
        self.mark_done(None);
        true
    }

    /// Cancel every linked task, as `_cancel` does, without marking the speech done.
    pub(crate) fn cancel_tasks(&self) {
        let tasks = std::mem::take(&mut self.inner.lock().unwrap().tasks);
        for task in &tasks {
            task.abort();
        }
    }

    /// `speech_handle._tasks.append(task)`. The handle only cancels these; done-when-all-done is
    /// the driver's.
    pub fn add_task(&self, task: lk_agents_abi::AbortHandle) {
        self.inner.lock().unwrap().tasks.push(task);
    }

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
    pub async fn wait_for_authorization(&self) {
        let mut rx = self.authorized.subscribe();
        // `wait_for` checks the current value first, so an earlier authorization isn't missed.
        let _ = rx.wait_for(|a| *a).await;
    }

    /// Whether step `step` (counting as [`num_steps`](Self::num_steps) does) was authorized to
    /// play: each authorization opens one generation.
    pub fn step_authorized(&self, step: u32) -> bool {
        self.generations.lock().unwrap().len() >= step as usize
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

    /// How a done speech ended, as `SpeechDone` reports it: as the host reads the handle, so an
    /// `interrupt()` that raced the speech's end is not reported completed.
    pub fn outcome(&self) -> crate::events::SpeechOutcome {
        use crate::events::SpeechOutcome;
        if matches!(self.exception(), Ok(Some(_))) {
            SpeechOutcome::Failed
        } else if self.interrupted_for_host() {
            SpeechOutcome::Interrupted
        } else {
            SpeechOutcome::Completed
        }
    }

    /// `wait_for_playout()`, and so `__await__`. The self-wait guard is the binding shim's.
    pub async fn wait_for_playout(&self) {
        self.wait_for_done().await;
    }

    /// `_item_added`.
    pub fn item_added(&self, item: &ChatItem) {
        self.inner.lock().unwrap().chat_items.push(item.clone());
    }

    /// `preemptive.user_message.metrics = metrics_report`: the turn a preemptive reply answers.
    pub(crate) fn set_user_metrics(&self, metrics: lk_agents_abi::MetricsReport) {
        self.inner.lock().unwrap().user_metrics = Some(metrics);
    }

    pub(crate) fn user_metrics(&self) -> Option<lk_agents_abi::MetricsReport> {
        self.inner.lock().unwrap().user_metrics.clone()
    }
}

// What `voice/speech_handle.py`'s `SpeechHandle` wraps. Before the loop has created the speech
// every read is a fresh handle's.
#[lk_agents_abi::host(object(name = "SpeechHandle", cell = "arc"))]
impl SpeechHandle {
    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    #[host(new(kw_only, defaults(speech_id = String::new(), allow_interruptions = true)))]
    pub fn detached(speech_id: String, allow_interruptions: bool) -> Self {
        let handle = SpeechHandle::base(
            Creation::Detached,
            allow_interruptions,
            crate::agent_session::Tx::disconnected(),
        );
        let _ = handle.speech_id.set(speech_id.into());
        handle
    }

    /// The host's object for this speech, as it set it; `None` until then.
    #[host(getter)]
    pub fn host(&self) -> Option<lk_agents_abi::HostRef> {
        lock(&self.inner).host.clone()
    }

    /// Set by the host when it creates its handle over this one: from then on every event, tool
    /// and hook naming the speech hands back this object.
    #[host(setter)]
    pub fn set_host(&self, host: lk_agents_abi::HostRef) {
        lock(&self.inner).host = Some(host);
    }

    /// `speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
    /// the id it was constructed with.
    #[host(getter(name = "id"))]
    pub fn id_for_host(&self) -> String {
        self.speech_id().to_owned()
    }

    #[host(getter)]
    pub fn num_steps(&self) -> u32 {
        self.num_steps.load(Ordering::Acquire)
    }

    /// `chat_items` (`speech_handle.py:156`): what this speech committed to the conversation.
    /// A copy, so it does not grow under a caller.
    #[host(getter)]
    pub fn chat_items(&self) -> Vec<ChatItem> {
        self.inner.lock().unwrap().chat_items.clone()
    }

    #[host(getter)]
    pub fn scheduled(&self) -> bool {
        self.scheduled.get()
    }

    /// `True` from the tick `interrupt()` was called, and once the loop has interrupted the speech
    /// for any reason (a interruption, a supersede, the close).
    #[host(getter(name = "interrupted"))]
    pub fn interrupted_for_host(&self) -> bool {
        self.interrupt_requested.load(Ordering::Acquire) || self.interrupted()
    }

    #[host(method)]
    pub fn done(&self) -> bool {
        self.done.get()
    }

    /// `allow_interruptions`. Before the loop creates the speech: the value the caller passed, or
    /// `True` when it passed none (the session default is resolved by the loop).
    #[host(getter)]
    pub fn allow_interruptions(&self) -> bool {
        self.interruptions.borrow().allow
    }

    /// The setter (`speech_handle.py:117`): sent to the loop, the one writer of a speech's
    /// interruption state, which journals it. Raises the reference's `RuntimeError` on an
    /// already-interrupted speech; the getter follows the loop's write by one wake.
    #[host(setter(name = "allow_interruptions"))]
    pub fn set_allow_interruptions_for_host(&self, value: bool) -> Result<(), SpeechRefused> {
        if !value && self.interrupt_requested.load(Ordering::Acquire) {
            return Err(SpeechError::AlreadyInterrupted.into());
        }
        if *self.creation.borrow() == Creation::Detached {
            return Ok(self.set_allow_interruptions(value)?);
        }
        let Some(speech) = self.created() else { return Err(SpeechRefused::NotCreated) };
        if !value && self.interrupted() {
            return Err(SpeechError::AlreadyInterrupted.into());
        }
        // A closed loop has no speech left to protect.
        let _ = self.api.send(crate::agent_session::Command::from(
            crate::agent_session::SpeechCommand::SetAllowInterruptions { speech, allow: value },
        ));
        Ok(())
    }

    /// The exception the speech failed with, as the original Python object.
    #[host(method(name = "exception"))]
    pub fn exception_for_host(&self) -> Result<Option<SpeechException>, SpeechRefused> {
        if self.created().is_none() {
            return Ok(None);
        }
        Ok(self.exception()?.map(SpeechException))
    }

    /// `interrupt(force=)` (`speech_handle.py:185`), synchronous and in the reference's order. If
    /// the speech does not exist yet, a task sends the command once it does, so the interrupt is
    /// not lost.
    #[host(method(name = "interrupt", kw_only, defaults(force = false)))]
    pub fn interrupt_for_host(
        self: &Arc<Self>,
        force: bool,
        #[host(ambient)] spawner: Arc<dyn lk_agents_abi::Spawner>,
    ) -> Result<(), SpeechRefused> {
        let created = self.created();
        if self.interrupt_requested.load(Ordering::Acquire)
            || (created.is_some() && (self.interrupted() || self.done()))
        {
            return Ok(());
        }
        if !force && !self.allow_interruptions() {
            return Err(SpeechError::InterruptionsNotAllowed.into());
        }
        self.interrupt_requested.store(true, Ordering::Release);
        let interrupt = move |api: &Api, speech| {
            // A closed loop has nothing left to interrupt.
            let _ = api.send(crate::agent_session::Command::from(
                crate::agent_session::SpeechCommand::InterruptSpeech { speech, force },
            ));
        };
        let creation = *self.creation.borrow();
        match (created, creation) {
            (Some(speech), _) => interrupt(&self.api, speech),
            // Recorded above; a detached handle has no speech to interrupt, now or later.
            (None, Creation::Detached) => {}
            (None, _) => {
                let handle = Arc::clone(self);
                spawner.spawn(Box::pin(async move {
                    if handle.wait_created().await.is_ok() {
                        interrupt(&handle.api, handle.id());
                    }
                }));
            }
        }
        Ok(())
    }

    /// `_hold_interruptions`: disallow interruptions until every hold taken is released. Raises
    /// the reference's `RuntimeError` when the speech is already interrupted, without counting the
    /// attempt.
    #[host(method(name = "hold_interruptions"))]
    pub fn hold_interruptions_for_host(&self) -> Result<(), SpeechRefused> {
        self.attached()?;
        Ok(self.hold_interruptions()?)
    }

    /// `_release_interruptions`. Idempotent past zero, like the reference's.
    #[host(method(name = "release_interruptions"))]
    pub fn release_interruptions_for_host(&self) -> Result<(), SpeechRefused> {
        self.attached()?;
        self.release_interruptions();
        Ok(())
    }

    /// `wait_for_playout()`, and therefore `__await__`. The self-wait guard is not here: it reads
    /// the ambient tool frame, so it belongs on the shim's `__await__`.
    #[host(method(name = "wait_for_playout"))]
    pub async fn wait_for_playout_for_host(&self) -> Result<(), SpeechRefused> {
        self.wait_created().await?;
        self.wait_for_playout().await;
        Ok(())
    }

    /// `_wait_for_generation(step_idx)`: wait for one step of the turn, not the whole thing.
    /// `step=None` is the reference's `step_idx=-1`. No generation, no such step, and a
    /// never-created speech are distinct errors.
    #[host(method(name = "wait_for_generation"))]
    pub async fn wait_for_generation_for_host(
        &self,
        step: Option<usize>,
    ) -> Result<(), SpeechRefused> {
        self.wait_created().await?;
        match step {
            Some(step) => self.wait_for_generation(step).await?,
            None => self.wait_for_latest_generation().await?,
        }
        Ok(())
    }

    /// Resolve when the speech is interrupted, and only then: a speech that finishes normally never
    /// resolves this, so `wait_if_not_interrupted` can race it.
    #[host(method(name = "wait_for_interrupted"))]
    pub async fn wait_for_interrupted_for_host(&self) -> Result<(), SpeechRefused> {
        self.wait_created().await?;
        self.wait_for_interrupted().await;
        Ok(())
    }

    /// Resolve once the loop has created the speech and put it in the queue.
    #[host(method)]
    pub async fn wait_until_created(&self) -> Result<(), SpeechRefused> {
        Ok(self.wait_created().await?)
    }
}

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
