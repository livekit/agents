//! `SpeechHandle`: the live object a host and the speech's tasks hold. The voice core owns the
//! speech's state; this handle mirrors what the core publishes ([`SpeechUpdate`]), lets tasks and
//! the host wait on it, and sends the host's requests to the core as queued inputs.

use super::session::Tx;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{AudioFrame, ChatItem, HostRef, MetricsReport, lock};
use lk_agents_net::monotone::Monotone;
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::{Command, SpeechCommand};
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::speech_handle::{SpeechError, SpeechUpdate, SpeechView};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::{mpsc, watch};

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

impl Creation {
    fn result(self) -> Result<(), CreationError> {
        match self {
            Self::Created => Ok(()),
            Self::Lost => Err(CreationError::Ended),
            Self::Pending | Self::Detached => Err(CreationError::Consumed),
        }
    }
}

/// What `say()` streams to its speech, held here until the speech's task takes it.
pub enum SayStream {
    Text(mpsc::Receiver<String>),
    Audio(mpsc::Receiver<AudioFrame>),
}

struct Inner {
    /// The core's last published view.
    view: SpeechView,
    /// `interrupt()` was called from the host and the loop has not refused it.
    requested: bool,
    /// The speech's task reported its end: a host `interrupt()` from here on is queued behind the
    /// input that ends the speech, so it lands on a done speech and changes nothing.
    ended: bool,
    chat_items: Vec<ChatItem>,
    user_metrics: Option<MetricsReport>,
    /// The host's own object for the speech (its `SpeechHandle`), so every event, tool and hook
    /// that names the speech hands back the object `say()` / `generate_reply()` returned.
    host: Option<HostRef>,
    /// The tools this speech waits on, as `(call_id, name)`: running and not yet detached.
    blocking_tools: Vec<(String, String)>,
    /// A host `interrupt(force)` accepted before the loop created the speech: sent at creation.
    pending_interrupt: Option<bool>,
    say: Option<SayStream>,
}

pub struct SpeechHandle {
    /// Set as the core publishes the speech's creation.
    id: OnceLock<SpeechId>,
    /// The id a host call's handle was given as the call returned; the loop creates it under it.
    reserved: Option<SpeechId>,
    /// `SpeechHandle.id`: `speech_{n}`, or what a detached handle was constructed with.
    speech_id: OnceLock<Arc<str>>,
    creation: watch::Sender<Creation>,
    /// The loop, for what the host asks of the speech.
    api: Tx,
    scheduled: Monotone<bool>,
    /// `_interrupt_fut`: one-shot, the whole speech (its tools and any follow-up step).
    interrupted: Monotone<bool>,
    done: Monotone<bool>,
    /// How many steps were authorized to play: each authorization opens one generation.
    authorized: Monotone<usize>,
    /// One cell per authorized step, addressed absolutely: set once its generation is done.
    generations: Mutex<Vec<Arc<Monotone<bool>>>>,
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

impl SpeechHandle {
    fn base(creation: Creation, reserved: Option<SpeechId>, allow: bool, api: Tx) -> Self {
        SpeechHandle {
            id: OnceLock::new(),
            reserved,
            speech_id: OnceLock::new(),
            creation: watch::Sender::new(creation),
            api,
            scheduled: Monotone::new(false),
            interrupted: Monotone::new(false),
            done: Monotone::new(false),
            authorized: Monotone::new(0),
            generations: Mutex::new(Vec::new()),
            inner: Mutex::new(Inner {
                view: SpeechView::new(allow),
                requested: false,
                ended: false,
                chat_items: Vec::new(),
                user_metrics: None,
                host: None,
                blocking_tools: Vec::new(),
                pending_interrupt: None,
                say: None,
            }),
        }
    }

    /// A speech the loop created on its own (a reply, a tool's follow-up).
    pub fn new(id: SpeechId, allow_interruptions: bool, api: Tx) -> Self {
        let handle = SpeechHandle::base(Creation::Created, None, allow_interruptions, api);
        let _ = handle.speech_id.set(id.to_string().into());
        let _ = handle.id.set(id);
        handle
    }

    /// What a host's `say()` / `generate_reply()` answers, before the loop creates the speech:
    /// its id, and the interruptibility it was asked with or the session's as the call returned.
    pub fn pending(api: Tx, id: SpeechId, allow_interruptions: bool) -> Arc<Self> {
        let handle = SpeechHandle::base(Creation::Pending, Some(id), allow_interruptions, api);
        let _ = handle.speech_id.set(id.to_string().into());
        Arc::new(handle)
    }

    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    pub fn detached(speech_id: String, allow_interruptions: bool) -> Self {
        let handle =
            SpeechHandle::base(Creation::Detached, None, allow_interruptions, Tx::disconnected());
        let _ = handle.speech_id.set(speech_id.into());
        handle
    }

    /// What `say()` streams, for the speech's task to take.
    pub fn hold_say(&self, stream: SayStream) {
        lock(&self.inner).say = Some(stream);
    }

    /// `say()`'s text chunks, once.
    pub fn take_text_stream(&self) -> Option<mpsc::Receiver<String>> {
        match lock(&self.inner).say.take() {
            Some(SayStream::Text(rx)) => Some(rx),
            _ => None,
        }
    }

    /// `say(audio=)`'s frames, once.
    pub fn take_audio(&self) -> Option<mpsc::Receiver<AudioFrame>> {
        match lock(&self.inner).say.take() {
            Some(SayStream::Audio(rx)) => Some(rx),
            _ => None,
        }
    }

    /// One update the core published, applied in order.
    fn apply(&self, id: SpeechId, update: SpeechUpdate) {
        match update {
            SpeechUpdate::Created => {
                let _ = self.speech_id.set(id.to_string().into());
                let _ = self.id.set(id);
            }
            SpeechUpdate::View(view) => self.publish(view),
            SpeechUpdate::ItemAdded(item) => lock(&self.inner).chat_items.push(item),
            SpeechUpdate::UserMetrics(metrics) => lock(&self.inner).user_metrics = Some(metrics),
            SpeechUpdate::InterruptRefused => lock(&self.inner).requested = false,
            SpeechUpdate::Released => lock(&self.inner).say = None,
        }
    }

    /// The view and the cells waiters watch, in the order a waiter must see them: a step's
    /// generation exists before its authorization is announced.
    fn publish(&self, view: SpeechView) {
        {
            let mut generations = lock(&self.generations);
            for (step, done) in view.generations.iter().enumerate() {
                if step == generations.len() {
                    generations.push(Arc::new(Monotone::new(false)));
                }
                if *done {
                    generations[step].set(true);
                }
            }
        }
        self.scheduled.set(view.scheduled);
        self.interrupted.set(view.interrupted);
        self.done.set(view.done);
        self.authorized.set(view.generations.len());
        lock(&self.inner).view = view;
    }

    /// The loop answered the host's call: with the speech it created over this handle, or, when
    /// the session ended first, not at all. Wakes whoever awaits the creation; an interrupt the
    /// host asked for meanwhile reaches the created speech now.
    pub fn answered(&self, created: bool) {
        let pending = {
            let mut inner = lock(&self.inner);
            let settled = self.creation.send_if_modified(|creation| {
                if *creation != Creation::Pending {
                    return false;
                }
                *creation = if created { Creation::Created } else { Creation::Lost };
                true
            });
            // Lost: nothing will read what `say()` streams.
            if settled && !created {
                inner.say = None;
            }
            inner.pending_interrupt.take()
        };
        if let (Some(force), Some(speech)) = (pending, self.created()) {
            self.send_interrupt(speech, force);
        }
    }

    /// The speech's task reports its end, `report(asked)` naming whether the host asked to
    /// interrupt it first: under this handle's lock, so a request lands with the end however the
    /// queue orders the two, and one after it is a no-op the host sees as such.
    pub fn report_end(&self, queue: &Tx, report: impl FnOnce(bool) -> GenEvent) {
        let mut inner = lock(&self.inner);
        inner.ended = true;
        let _ = queue.send(report(inner.requested));
    }

    fn send_interrupt(&self, speech: SpeechId, force: bool) {
        // A closed loop has nothing left to interrupt.
        let _ = self.api.send(Command::from(SpeechCommand::InterruptSpeech { speech, force }));
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
        let creation = *rx.wait_for(|c| *c != Creation::Pending).await.expect("held by self");
        creation.result()
    }

    fn attached(&self) -> Result<SpeechId, SpeechRefused> {
        self.created().ok_or(SpeechRefused::Detached)
    }

    fn view<T>(&self, read: impl FnOnce(&SpeechView) -> T) -> T {
        read(&lock(&self.inner).view)
    }

    /// The loop interrupted the speech: its playout, its tools and any follow-up step.
    pub fn interrupted(&self) -> bool {
        self.interrupted.get()
    }

    pub fn done(&self) -> bool {
        self.done.get()
    }

    pub fn scheduled(&self) -> bool {
        self.scheduled.get()
    }

    /// `allow_interruptions` as the loop last accepted it.
    pub fn allow_interruptions(&self) -> bool {
        self.view(|v| v.allow_interruptions)
    }

    pub fn num_steps(&self) -> u32 {
        self.view(|v| v.num_steps)
    }

    /// `exception()`. [`SpeechError::NotDoneYet`] stands in for `asyncio.InvalidStateError`.
    pub fn exception(&self) -> Result<Option<ProviderError>, SpeechError> {
        self.view(SpeechView::exception)
    }

    /// The metrics of the user turn a preemptive reply was adopted for.
    pub fn user_metrics(&self) -> Option<MetricsReport> {
        lock(&self.inner).user_metrics.clone()
    }

    /// `chat_items` (`speech_handle.py:156`): what this speech committed to the conversation.
    /// A copy, so it does not grow under a caller.
    pub fn chat_items(&self) -> Vec<ChatItem> {
        lock(&self.inner).chat_items.clone()
    }

    /// `_wait_for_scheduled`. The caller must re-check [`interrupted`](Self::interrupted) after:
    /// a cancelled speech is never marked scheduled, so a task races this against the interrupt.
    pub async fn wait_for_scheduled(&self) {
        self.scheduled.wait_for(|set| *set).await;
    }

    pub async fn wait_for_interrupted(&self) {
        self.interrupted.wait_for(|set| *set).await;
    }

    /// `wait_for_playout()`, and so `__await__`; the host's call adds the self-wait guard.
    pub async fn wait_for_playout(&self) {
        self.done.wait_for(|set| *set).await;
    }

    /// Whether step `step` (counting as [`num_steps`](Self::num_steps) does) was authorized to
    /// play.
    pub fn step_authorized(&self, step: u32) -> bool {
        self.authorized.get() >= step as usize
    }

    /// `_wait_for_authorization` for step `step`: each step waits for its own grant, so a step
    /// never plays on the one before it.
    pub async fn wait_for_step_authorized(&self, step: u32) {
        self.authorized.wait_for(|n| *n >= step as usize).await;
    }

    /// Wait for step `step`'s generation (absolute, from zero). `RunContext.wait_for_playout()`
    /// passes the step captured at tool-call time, so a long tool waits for its own step's playout.
    pub async fn wait_for_generation(&self, step: usize) -> Result<(), SpeechError> {
        let step = self.view(|v| v.generation(step))?;
        self.generation_cell(step).wait_for(|done| *done).await;
        Ok(())
    }

    /// `_wait_for_generation()` with `step_idx=-1`: the step currently running.
    pub async fn wait_for_latest_generation(&self) -> Result<(), SpeechError> {
        let step = self.view(SpeechView::latest_generation)?;
        self.generation_cell(step).wait_for(|done| *done).await;
        Ok(())
    }

    /// The lock is released before any await.
    fn generation_cell(&self, step: usize) -> Arc<Monotone<bool>> {
        lock(&self.generations).get(step).cloned().expect("the accepted generation was published")
    }

    /// The speech waits on tool call `call_id` (`name`) until [`tool_returned`]: from its start
    /// until it answers, or detaches with its first update.
    ///
    /// [`tool_returned`]: Self::tool_returned
    pub fn tool_blocking(&self, call_id: &str, name: &str) {
        lock(&self.inner).blocking_tools.push((call_id.to_owned(), name.to_owned()));
    }

    pub fn tool_returned(&self, call_id: &str) {
        lock(&self.inner).blocking_tools.retain(|(c, _)| c != call_id);
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

    pub fn host(&self) -> Option<HostRef> {
        lock(&self.inner).host.clone()
    }
}

/// A session's speeches by id, from the host's call (or the core's creation) to the session's
/// end: events and runs name a speech after the loop let it go.
#[derive(Default)]
pub struct Speeches(Mutex<BTreeMap<SpeechId, Arc<SpeechHandle>>>);

impl Speeches {
    /// The handle a host call returns, under the id the loop will create its speech with.
    pub fn register(&self, handle: Arc<SpeechHandle>) {
        let id = handle.reserved.expect("a host call's handle has its id");
        lock(&self.0).insert(id, handle);
    }

    pub fn get(&self, id: SpeechId) -> Option<Arc<SpeechHandle>> {
        lock(&self.0).get(&id).cloned()
    }

    /// The host's object for `id`; a speech the session never held reads as a fresh one.
    pub fn object(&self, id: SpeechId) -> Arc<dyn lk_agents_abi::voice::speech_handle::Speech> {
        let handle = self.get(id);
        handle.unwrap_or_else(|| Arc::new(SpeechHandle::detached(id.to_string(), true))).object()
    }

    /// What the core published about `id`: its creation makes the handle a loop speech's (over
    /// the host's handle, when a call reserved the id).
    pub fn apply(&self, id: SpeechId, update: SpeechUpdate, api: &Tx) {
        let handle = {
            let mut speeches = lock(&self.0);
            let entry = speeches
                .entry(id)
                .or_insert_with(|| Arc::new(SpeechHandle::new(id, true, api.clone())));
            Arc::clone(entry)
        };
        handle.apply(id, update);
    }

    /// `invocation` as a callback of `speech` runs it: with the host's object for the speech.
    pub fn invocation(
        &self,
        hosted: &lk_agents_voice::tool_executor::HostedTool,
    ) -> lk_agents_abi::agent::Invocation {
        let mut invocation = hosted.invocation.clone();
        invocation.speech = Some(self.object(hosted.speech));
        invocation
    }
}

impl From<SpeechRefused> for lk_agents_abi::voice::speech_handle::SpeechRefused {
    fn from(refused: SpeechRefused) -> Self {
        Self(refused.to_string())
    }
}

/// The host's refusal for `error`.
fn refused(error: impl Into<SpeechRefused>) -> HostRefused {
    error.into().into()
}

type HostRefused = lk_agents_abi::voice::speech_handle::SpeechRefused;

impl lk_agents_abi::voice::speech_handle::Speech for SpeechHandle {}

/// `_core.SpeechHandle`: what `voice/speech_handle.py`'s `SpeechHandle` wraps. Before the loop has
/// created the speech every read is a fresh handle's.
impl lk_agents_abi::voice::speech_handle::SpeechHandle for SpeechHandle {
    fn new(speech_id: String, allow_interruptions: bool) -> Self {
        Self::detached(speech_id, allow_interruptions)
    }

    fn host_object(&self) -> Option<HostRef> {
        self.host()
    }

    fn set_host_object(&self, host: HostRef) {
        lock(&self.inner).host = Some(host);
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
        let inner = lock(&self.inner);
        inner.view.host_interrupted(inner.requested)
    }

    fn done(&self) -> bool {
        SpeechHandle::done(self)
    }

    fn allow_interruptions(&self) -> bool {
        SpeechHandle::allow_interruptions(self)
    }

    fn set_allow_interruptions(&self, value: bool) -> Result<(), HostRefused> {
        let mut inner = lock(&self.inner);
        inner.view.host_set_allow_interruptions(value, inner.requested).map_err(refused)?;
        // A detached handle's setter is its own, with no loop to tell.
        if *self.creation.borrow() == Creation::Detached {
            inner.view.allow_interruptions = value;
            return Ok(());
        }
        drop(inner);
        let Some(speech) = self.created().or(self.reserved) else {
            return Err(refused(SpeechRefused::Detached));
        };
        let _ = self
            .api
            .send(Command::from(SpeechCommand::SetAllowInterruptions { speech, allow: value }));
        Ok(())
    }

    fn exception(&self) -> Result<Option<ProviderError>, HostRefused> {
        if self.created().is_none() {
            return Ok(None);
        }
        SpeechHandle::exception(self).map_err(refused)
    }

    /// Synchronous and in the reference's order (`speech_handle.py:185`).
    fn interrupt(&self, force: bool) -> Result<(), HostRefused> {
        // Under the lock `answered` takes, so a creation racing this call sends it once.
        let mut inner = lock(&self.inner);
        // Behind the speech's end: it lands on a done speech.
        if inner.ended || !inner.view.host_interrupt(inner.requested, force).map_err(refused)? {
            return Ok(());
        }
        let created = self.created();
        inner.requested = true;
        match (created, *self.creation.borrow()) {
            (Some(speech), _) => self.send_interrupt(speech, force),
            (None, Creation::Detached) => {}
            (None, _) => inner.pending_interrupt = Some(force),
        }
        Ok(())
    }

    fn hold_interruptions(&self) -> Result<(), HostRefused> {
        let speech = self.attached().map_err(refused)?;
        lock(&self.inner).view.host_hold_interruptions().map_err(refused)?;
        let _ =
            self.api.send(Command::from(SpeechCommand::HoldInterruptions { speech, hold: true }));
        Ok(())
    }

    fn release_interruptions(&self) -> Result<(), HostRefused> {
        let speech = self.attached().map_err(refused)?;
        let _ =
            self.api.send(Command::from(SpeechCommand::HoldInterruptions { speech, hold: false }));
        Ok(())
    }

    async fn wait_for_playout(&self, origin: Option<String>) -> Result<(), HostRefused> {
        self.self_wait(origin.as_deref()).map_err(refused)?;
        if self.wait_created().await.is_ok() {
            SpeechHandle::wait_for_playout(self).await;
        }
        Ok(())
    }

    async fn wait_for_generation(&self, step: Option<usize>) -> Result<(), HostRefused> {
        self.wait_created().await.map_err(refused)?;
        match step {
            Some(step) => SpeechHandle::wait_for_generation(self, step).await,
            None => self.wait_for_latest_generation().await,
        }
        .map_err(refused)
    }

    async fn wait_for_interrupted(&self) -> Result<(), HostRefused> {
        self.wait_created().await.map_err(refused)?;
        SpeechHandle::wait_for_interrupted(self).await;
        Ok(())
    }

    async fn wait_until_created(&self) -> Result<(), HostRefused> {
        self.wait_created().await.map_err(refused)
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

#[cfg(test)]
#[path = "speech_handle_tests.rs"]
mod tests;
