use super::session::Tx;
use crate::voice::VoiceRuntimeEvent;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::session::VoiceCommandOrigin;
use lk_agents_abi::{AudioFrame, ChatItem, HostRef, MetricsReport};
use lk_agents_net::signal;
use lk_agents_voice::agent_session::SpeechCommand;
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::speech_handle::{SpeechError, SpeechUpdate, SpeechView};
use parking_lot::{Mutex, MutexGuard};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

/// A refusal on a host's speech handle; each `Display` is the `RuntimeError` it raises.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeechRefused {
    #[error(transparent)]
    Speech(#[from] SpeechError),
    #[error("the session ended before this speech was created, so it will never run")]
    Ended,
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

/// Whether the host's call was answered: `say()` and `generate_reply()` return before the loop
/// creates the speech, and a handle constructed directly never gets one. A pending speech whose
/// signals ended was never created: its session ended first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Creation {
    #[default]
    Pending,
    Created,
    Detached,
}

/// A speech's lifecycle as its waiters watch it, in one cell. Each mark only ever turns on, so no
/// waiter misses a phase.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Lifecycle {
    creation: Creation,
    scheduled: bool,
    /// `_interrupt_fut`: one-shot, the whole speech (its tools and any follow-up step).
    interrupted: bool,
    done: bool,
    /// How many steps were authorized to play: each authorization opens one generation.
    authorized: usize,
    /// One mark per authorized step, addressed absolutely: set once its generation is done.
    generations: Vec<bool>,
}

impl Lifecycle {
    /// As read once its producer is gone: a speech the session never finished ends interrupted,
    /// with its generations. A detached handle never had a speech to end.
    fn ended(mut self) -> Lifecycle {
        if self.creation != Creation::Detached {
            self.interrupted |= !self.done;
            self.done = true;
            self.generations.iter_mut().for_each(|done| *done = true);
        }
        self
    }

    fn publish(&mut self, view: &SpeechView) {
        self.scheduled |= view.scheduled();
        self.interrupted |= view.interrupted();
        self.done |= view.done();
        self.authorized = self.authorized.max(view.generations.len());
        for (step, done) in view.generations.iter().enumerate() {
            if step == self.generations.len() {
                self.generations.push(false);
            }
            self.generations[step] |= *done;
        }
    }
}

/// The writers of one speech's cells. The voice loop holds them while the speech lives
/// ([`SpeechProducers`]), so the loop's end, however it comes, resolves every wait on the speech.
pub struct SpeechSignals {
    lifecycle: signal::Producer<Lifecycle>,
    /// Travels with the speech's `speech_done` event; finished once the host handled it.
    reported: Option<signal::Producer<bool>>,
}

fn signals(creation: Creation) -> (SpeechSignals, signal::Cell<Lifecycle>, signal::Cell<bool>) {
    let (lifecycle, cell) = signal::signal(Lifecycle { creation, ..Lifecycle::default() });
    let (reported, reported_cell) = signal::signal(false);
    (SpeechSignals { lifecycle, reported: Some(reported) }, cell, reported_cell)
}

/// The voice loop's [`SpeechSignals`], by speech: dropped with the loop.
#[derive(Default)]
pub struct SpeechProducers(BTreeMap<SpeechId, SpeechSignals>);

impl SpeechProducers {
    pub fn adopt(&mut self, id: SpeechId, signals: SpeechSignals) {
        self.0.insert(id, signals);
    }

    pub fn get(&self, id: SpeechId) -> Option<&SpeechSignals> {
        self.0.get(&id)
    }

    /// The producer that travels with `id`'s `speech_done` event.
    pub fn take_reported(&mut self, id: SpeechId) -> Option<signal::Producer<bool>> {
        self.0.get_mut(&id).and_then(|signals| signals.reported.take())
    }

    /// Nothing will update `id` again: its waits resolve on what it reached.
    pub fn retire(&mut self, id: SpeechId) {
        self.0.remove(&id);
    }
}

/// [`SpeechHandle::wait_scheduled`] on a speech interrupted, or done without being scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WillNotPlay;

/// What `say()` streams to its speech, held here until the speech's task takes it.
pub enum SayStream {
    Text(mpsc::Receiver<String>),
    Audio(mpsc::Receiver<AudioFrame>),
}

struct Inner {
    view: SpeechView,
    /// `interrupt()` was called from the host and the loop has not refused it.
    requested: bool,
    chat_items: Vec<ChatItem>,
    user_metrics: Option<MetricsReport>,
    /// The host's own object for the speech (its `SpeechHandle`), so every event, tool and hook
    /// that names the speech hands back the object `say()` / `generate_reply()` returned.
    host: Option<HostRef>,
    /// The tools this speech waits on, as `(call_id, name)`: running and not yet detached.
    blocking_tools: Vec<(VoiceCommandOrigin, String)>,
    /// What the host asked before the loop created the speech, in order: sent at creation.
    pending: Vec<SpeechCommand>,
    say: Option<SayStream>,
}

pub struct SpeechHandle {
    /// Set as the core publishes the speech's creation.
    id: OnceLock<SpeechId>,
    /// The id a host call's handle was given as the call returned; the loop creates it under it.
    reserved: Option<SpeechId>,
    /// `SpeechHandle.id`: `speech_{n}`, or what a detached handle was constructed with.
    speech_id: OnceLock<Arc<str>>,
    /// The loop, for what the host asks of the speech.
    requests: Requests,
    lifecycle: signal::Cell<Lifecycle>,
    /// The session handed the host this speech's `speech_done` event (or never will, once ended).
    reported: signal::Cell<bool>,
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
    fn base(
        (lifecycle, reported): (signal::Cell<Lifecycle>, signal::Cell<bool>),
        reserved: Option<SpeechId>,
        allow: bool,
        requests: Requests,
    ) -> Self {
        SpeechHandle {
            id: OnceLock::new(),
            reserved,
            speech_id: OnceLock::new(),
            requests,
            lifecycle,
            reported,
            inner: Mutex::new(Inner {
                view: SpeechView::new(allow),
                requested: false,
                chat_items: Vec::new(),
                user_metrics: None,
                host: None,
                blocking_tools: Vec::new(),
                pending: Vec::new(),
                say: None,
            }),
        }
    }

    /// A speech the loop created on its own (a reply, a tool's follow-up), and its writers.
    pub fn new(
        id: SpeechId,
        allow_interruptions: bool,
        requests: Requests,
    ) -> (Self, SpeechSignals) {
        let (signals, lifecycle, reported) = signals(Creation::Created);
        let handle = SpeechHandle::base((lifecycle, reported), None, allow_interruptions, requests);
        let _ = handle.speech_id.set(id.to_string().into());
        let _ = handle.id.set(id);
        (handle, signals)
    }

    /// What a host's `say()` / `generate_reply()` answers, before the loop creates the speech:
    /// its id, and the interruptibility it was asked with or the session's as the call returned.
    /// The writers travel to the loop with the call.
    pub fn pending(
        requests: Requests,
        id: SpeechId,
        allow_interruptions: bool,
    ) -> (Arc<Self>, SpeechSignals) {
        let (signals, lifecycle, reported) = signals(Creation::Pending);
        let handle =
            SpeechHandle::base((lifecycle, reported), Some(id), allow_interruptions, requests);
        let _ = handle.speech_id.set(id.to_string().into());
        (Arc::new(handle), signals)
    }

    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    pub fn detached(speech_id: String, allow_interruptions: bool) -> Self {
        let requests = Requests::disconnected();
        let lifecycle = Lifecycle { creation: Creation::Detached, ..Lifecycle::default() };
        let cells = (signal::Cell::ended_with(lifecycle), signal::Cell::ended_with(false));
        let handle = SpeechHandle::base(cells, None, allow_interruptions, requests);
        let _ = handle.speech_id.set(speech_id.into());
        handle
    }

    /// What `say()` streams, for the speech's task to take.
    pub fn hold_say(&self, stream: SayStream) {
        self.inner.lock().say = Some(stream);
    }

    /// `say()`'s text chunks, once.
    pub fn take_text_stream(&self) -> Option<mpsc::Receiver<String>> {
        match self.inner.lock().say.take() {
            Some(SayStream::Text(rx)) => Some(rx),
            _ => None,
        }
    }

    /// `say(audio=)`'s frames, once.
    pub fn take_audio(&self) -> Option<mpsc::Receiver<AudioFrame>> {
        match self.inner.lock().say.take() {
            Some(SayStream::Audio(rx)) => Some(rx),
            _ => None,
        }
    }

    /// One update the core published, applied in order.
    fn apply(&self, signals: Option<&SpeechSignals>, id: SpeechId, update: SpeechUpdate) {
        match update {
            SpeechUpdate::Created => {
                let _ = self.speech_id.set(id.to_string().into());
                let _ = self.id.set(id);
            }
            SpeechUpdate::View(view) => self.publish(signals, view),
            SpeechUpdate::ItemAdded(item) => self.inner.lock().chat_items.push(item),
            SpeechUpdate::UserMetrics(metrics) => self.inner.lock().user_metrics = Some(metrics),
            SpeechUpdate::InterruptRefused => self.inner.lock().requested = false,
            SpeechUpdate::Released => self.inner.lock().say = None,
        }
    }

    /// The cell waiters watch, then the view.
    fn publish(&self, signals: Option<&SpeechSignals>, view: SpeechView) {
        if let Some(signals) = signals {
            signals.lifecycle.modify(|lifecycle| lifecycle.publish(&view));
        }
        self.inner.lock().view = view;
    }

    /// The loop answered the host's call: with the speech it created over this handle, or, when
    /// the session ended first, not at all. Wakes whoever awaits the creation; what the host asked
    /// meanwhile reaches the created speech now. A speech created done (refused, or answered as
    /// the session closed) never ran: an interrupt asked of it lands nowhere.
    /// `signals` are the speech's writers, in the loop's table once it received the call; a
    /// lost speech's are dropped by the caller, which ends every wait on it.
    pub fn answered(&self, signals: Option<&SpeechSignals>, created: bool) {
        let mut requests = self.requests.lock();
        let mut inner = self.inner.lock();
        if let (true, Some(signals)) = (created, signals) {
            signals.lifecycle.modify(|lifecycle| {
                if lifecycle.creation == Creation::Pending {
                    lifecycle.creation = Creation::Created;
                }
            });
        }
        // Lost: nothing will read what `say()` streams.
        if !created {
            inner.say = None;
        }
        let pending = std::mem::take(&mut inner.pending);
        if created && inner.view.done() {
            inner.requested &= inner.view.interrupted();
            // The setters asked meanwhile are the handle's own, as on any ended speech.
            for command in pending {
                if let SpeechCommand::SetAllowInterruptions { allow, .. } = command {
                    inner.view.allow_interruptions = allow;
                }
            }
        } else if created {
            for command in pending {
                self.requests.send(&mut requests, command);
            }
        }
    }

    /// The id a host call's handle was given as the call returned.
    pub fn reserved(&self) -> Option<SpeechId> {
        self.reserved
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
    async fn wait_created(&self) -> Result<(), SpeechRefused> {
        let creation =
            self.lifecycle.wait_map(|l| (l.creation != Creation::Pending).then_some(l.creation));
        match creation.await {
            Ok(Creation::Created) => Ok(()),
            Ok(_) => Err(SpeechRefused::Detached),
            Err(signal::Ended) => Err(SpeechRefused::Ended),
        }
    }

    fn creation(&self) -> Creation {
        self.lifecycle.with(|l| l.creation)
    }

    fn attached(&self) -> Result<SpeechId, SpeechRefused> {
        self.created().ok_or(SpeechRefused::Detached)
    }

    fn request(&self, command: SpeechCommand) {
        let mut requests = self.requests.lock();
        let mut inner = self.inner.lock();
        self.ask(&mut requests, &mut inner, command);
    }

    /// The created speech's loop is asked now; an uncreated one's at its creation, in order.
    fn ask(&self, requests: &mut Vec<SpeechCommand>, inner: &mut Inner, command: SpeechCommand) {
        if self.creation() == Creation::Created {
            self.requests.send(requests, command);
        } else {
            inner.pending.push(command);
        }
    }

    fn view<T>(&self, read: impl FnOnce(&SpeechView) -> T) -> T {
        read(&self.inner.lock().view)
    }

    fn lifecycle(&self) -> Lifecycle {
        let lifecycle = self.lifecycle.get();
        if self.lifecycle.ended() { lifecycle.ended() } else { lifecycle }
    }

    /// The lifecycle once `ready` holds, or as it ended; cancel-safe.
    async fn wait(&self, ready: impl Fn(&Lifecycle) -> bool + Send) -> Lifecycle {
        match self.lifecycle.wait_for(ready).await {
            Ok(lifecycle) => lifecycle,
            Err(signal::Ended) => self.lifecycle(),
        }
    }

    /// The loop interrupted the speech: its playout, its tools and any follow-up step.
    pub fn interrupted(&self) -> bool {
        self.lifecycle().interrupted
    }

    pub fn done(&self) -> bool {
        self.lifecycle().done
    }

    pub fn scheduled(&self) -> bool {
        self.lifecycle().scheduled
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
        self.inner.lock().user_metrics.clone()
    }

    /// `chat_items` (`speech_handle.py:156`): what this speech committed to the conversation.
    /// A copy, so it does not grow under a caller.
    pub fn chat_items(&self) -> Vec<ChatItem> {
        self.inner.lock().chat_items.clone()
    }

    /// `_wait_for_scheduled`: resolves for every speech, `Err` once it was interrupted or ended
    /// unscheduled.
    pub async fn wait_scheduled(&self) -> Result<(), WillNotPlay> {
        let lifecycle = self.wait(|l| l.scheduled || l.interrupted || l.done).await;
        if lifecycle.interrupted || !lifecycle.scheduled { Err(WillNotPlay) } else { Ok(()) }
    }

    /// Resolves too once the speech can no longer be interrupted: its session ended.
    pub async fn wait_for_interrupted(&self) {
        self.wait(|l| l.interrupted).await;
    }

    /// `wait_for_playout()`, and so `__await__`; the host's call adds the self-wait guard.
    pub async fn wait_for_playout(&self) {
        self.wait(|l| l.done).await;
    }

    /// Whether step `step` (counting as [`num_steps`](Self::num_steps) does) was authorized to
    /// play.
    pub fn step_authorized(&self, step: u32) -> bool {
        self.lifecycle.with(|l| l.authorized >= step as usize)
    }

    /// `_wait_for_authorization` for step `step`: each step waits for its own grant, so a step
    /// never plays on the one before it. Also resolves once the speech is interrupted or done,
    /// as it never will be granted; callers race the interruption first.
    pub async fn wait_for_step_authorized(&self, step: u32) {
        self.wait(|l| l.authorized >= step as usize || l.interrupted || l.done).await;
    }

    /// Wait for step `step`'s generation (absolute, from zero). `RunContext.wait_for_playout()`
    /// passes the step captured at tool-call time, so a long tool waits for its own step's playout.
    pub async fn wait_for_generation(&self, step: usize) -> Result<(), SpeechError> {
        let step = self.view(|v| v.generation(step))?;
        self.wait_generation(step).await;
        Ok(())
    }

    /// `_wait_for_generation()` with `step_idx=-1`: the step currently running.
    pub async fn wait_for_latest_generation(&self) -> Result<(), SpeechError> {
        let step = self.view(SpeechView::latest_generation)?;
        self.wait_generation(step).await;
        Ok(())
    }

    /// The accepted generation `step` is done, or the speech ended.
    async fn wait_generation(&self, step: usize) {
        self.wait(|l| l.generations.get(step).copied().unwrap_or(false)).await;
    }

    /// The speech waits on tool call `call_id` (`name`) until [`tool_returned`]: from its start
    /// until it answers, or detaches with its first update.
    ///
    /// [`tool_returned`]: Self::tool_returned
    pub fn tool_blocking(
        &self,
        invocation: &lk_agents_abi::agent::Invocation,
        call_id: &str,
        name: &str,
    ) {
        let origin = VoiceCommandOrigin::Tool {
            invocation_id: invocation.invocation_id.clone(),
            operation: invocation.operation,
            call_id: call_id.to_owned(),
        };
        self.inner.lock().blocking_tools.push((origin, name.to_owned()));
    }

    pub fn tool_returned(&self, invocation: &lk_agents_abi::agent::Invocation, call_id: &str) {
        let origin = VoiceCommandOrigin::Tool {
            invocation_id: invocation.invocation_id.clone(),
            operation: invocation.operation,
            call_id: call_id.to_owned(),
        };
        self.inner.lock().blocking_tools.retain(|(owned, _)| *owned != origin);
    }

    /// Awaiting a speech from its own blocking callback would wait forever. Reused call IDs
    /// from a different invocation do not own this speech's wait.
    fn self_wait(&self, origin: Option<&VoiceCommandOrigin>) -> Result<(), SpeechRefused> {
        let Some(origin @ VoiceCommandOrigin::Tool { .. }) = origin else { return Ok(()) };
        match self.inner.lock().blocking_tools.iter().find(|(owned, _)| owned == origin) {
            Some((_, name)) => Err(SpeechRefused::SelfWait { name: name.clone() }),
            None => Ok(()),
        }
    }

    pub fn host(&self) -> Option<HostRef> {
        self.inner.lock().host.clone()
    }
}

/// Where the host's requests on its speeches go (`interrupt()` and the interruption setters):
/// the session's list, which the loop steps before anything else it takes, and the queue, which
/// wakes it. Each request is so ordered at its call (`VoiceRuntime::step`).
#[derive(Clone)]
pub struct Requests {
    list: Arc<Mutex<Vec<SpeechCommand>>>,
    wake: Tx,
}

impl Requests {
    /// Requests nobody reads: a handle made outside a session.
    pub fn disconnected() -> Self {
        Requests { list: Arc::default(), wake: Tx::disconnected() }
    }

    /// The list, held by a host call while it checks the speech's view and asks, and by the loop
    /// from its last request through the views its step publishes.
    pub fn lock(&self) -> MutexGuard<'_, Vec<SpeechCommand>> {
        self.list.lock()
    }

    fn send(&self, list: &mut Vec<SpeechCommand>, command: SpeechCommand) {
        // The list is locked until append, so a successful wake cannot observe it too early.
        // A refused wake means nobody can consume this command; never retain it in the mailbox.
        if self.wake.send(VoiceRuntimeEvent::HostRequests).is_ok() {
            list.push(command);
        }
    }
}

/// A session's speeches by id, from the host's call (or the core's creation) until the core
/// retires the speech: events, runs and running tools name a speech after the loop let it go.
/// Their writers are the loop's ([`SpeechProducers`]), never this shared table's.
#[derive(Default)]
pub struct Speeches {
    handles: Mutex<BTreeMap<SpeechId, Arc<SpeechHandle>>>,
    requests: Arc<Mutex<Vec<SpeechCommand>>>,
}

impl Speeches {
    /// Where this session's handles send their requests, waking the loop on `wake`.
    pub fn requests(&self, wake: Tx) -> Requests {
        Requests { list: Arc::clone(&self.requests), wake }
    }

    /// The handle a host call returns, under the id the loop will create its speech with.
    pub fn register(&self, handle: Arc<SpeechHandle>) {
        let id = handle.reserved.expect("a host call's handle has its id");
        self.handles.lock().insert(id, handle);
    }

    pub fn get(&self, id: SpeechId) -> Option<Arc<SpeechHandle>> {
        self.handles.lock().get(&id).cloned()
    }

    /// Nothing names `id` any more: the session lets its handle go (the host keeps its own).
    pub fn retire(&self, id: SpeechId) {
        self.handles.lock().remove(&id);
    }

    /// How many speeches the session holds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.handles.lock().len()
    }

    /// The host's object for `id`; a speech the session never held reads as a fresh one.
    pub fn object(&self, id: SpeechId) -> Arc<dyn lk_agents_abi::voice::speech_handle::Speech> {
        let handle = self.get(id);
        handle.unwrap_or_else(|| Arc::new(SpeechHandle::detached(id.to_string(), true))).object()
    }

    /// What the core published about `id`: its creation makes the handle a loop speech's (over
    /// the host's handle, when a call reserved the id), whose writers join `producers`.
    pub fn apply(
        &self,
        producers: &mut SpeechProducers,
        id: SpeechId,
        update: SpeechUpdate,
        wake: &Tx,
    ) {
        let handle = {
            let mut speeches = self.handles.lock();
            let entry = speeches.entry(id).or_insert_with(|| {
                let (handle, signals) = SpeechHandle::new(id, true, self.requests(wake.clone()));
                producers.adopt(id, signals);
                Arc::new(handle)
            });
            Arc::clone(entry)
        };
        handle.apply(producers.get(id), id, update);
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

fn refused(error: impl Into<SpeechRefused>) -> HostRefused {
    error.into().into()
}

type HostRefused = lk_agents_abi::voice::speech_handle::SpeechRefused;
use lk_agents_abi::voice::speech_handle::ExceptionRefused;

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
        self.inner.lock().host = Some(host);
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
        let inner = self.inner.lock();
        inner.view.host_interrupted(inner.requested)
    }

    fn done(&self) -> bool {
        SpeechHandle::done(self)
    }

    fn allow_interruptions(&self) -> bool {
        SpeechHandle::allow_interruptions(self)
    }

    /// Checked and asked under the list's lock, as `interrupt()` is: a speech not done here is one
    /// the request reaches before anything ends it.
    fn set_allow_interruptions(&self, value: bool) -> Result<(), HostRefused> {
        let mut requests = self.requests.lock();
        let mut inner = self.inner.lock();
        inner.view.host_set_allow_interruptions(value, inner.requested).map_err(refused)?;
        // A detached handle, or a speech that ended: nothing is left to tell, the value is the
        // handle's own.
        if self.creation() == Creation::Detached || inner.view.done() {
            inner.view.allow_interruptions = value;
            return Ok(());
        }
        let Some(speech) = self.created().or(self.reserved) else {
            return Err(refused(SpeechRefused::Detached));
        };
        self.ask(
            &mut requests,
            &mut inner,
            SpeechCommand::SetAllowInterruptions { speech, allow: value },
        );
        Ok(())
    }

    fn exception(&self) -> Result<Option<ProviderError>, ExceptionRefused> {
        if !SpeechHandle::done(self) {
            return Err(ExceptionRefused::NotDone);
        }
        if self.created().is_none() {
            return Ok(None);
        }
        Ok(SpeechHandle::exception(self).map_err(refused)?)
    }

    /// Synchronous, in the reference's order.
    fn interrupt(&self, force: bool) -> Result<(), HostRefused> {
        // Under the list's lock: the loop has published every step it took before this request,
        // so a speech not done here is one the request reaches before anything ends it.
        let mut requests = self.requests.lock();
        let mut inner = self.inner.lock();
        if !inner.view.host_interrupt(inner.requested, force).map_err(refused)? {
            return Ok(());
        }
        inner.requested = true;
        if self.creation() != Creation::Detached {
            self.ask(
                &mut requests,
                &mut inner,
                SpeechCommand::InterruptSpeech {
                    speech: self
                        .created()
                        .or(self.reserved)
                        .expect("an attached handle has its id"),
                    force,
                },
            );
        }
        Ok(())
    }

    fn end_call(&self) -> Result<(), HostRefused> {
        let speech = self.attached().map_err(refused)?;
        self.request(SpeechCommand::EndCall { speech });
        Ok(())
    }

    fn hold_interruptions(&self) -> Result<(), HostRefused> {
        let speech = self.attached().map_err(refused)?;
        self.inner.lock().view.host_hold_interruptions().map_err(refused)?;
        self.request(SpeechCommand::HoldInterruptions { speech, hold: true });
        Ok(())
    }

    fn release_interruptions(&self) -> Result<(), HostRefused> {
        let speech = self.attached().map_err(refused)?;
        self.request(SpeechCommand::HoldInterruptions { speech, hold: false });
        Ok(())
    }

    async fn wait_for_playout(
        &self,
        origin: Option<VoiceCommandOrigin>,
    ) -> Result<(), HostRefused> {
        self.self_wait(origin.as_ref()).map_err(refused)?;
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

    async fn wait_reported(&self) {
        // Ended: the host will never be handed its `speech_done`.
        if self.wait_created().await.is_ok() {
            self.reported.until(|reported| *reported).await;
        }
    }
}

impl std::fmt::Debug for SpeechHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechHandle")
            .field("id", &self.created())
            .field("speech_id", &self.speech_id())
            .field("creation", &self.creation())
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
