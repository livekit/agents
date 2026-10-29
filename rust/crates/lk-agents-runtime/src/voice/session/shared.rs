//! What a session's spawned tasks and the host share with its queue owner: the clock, the
//! published views and outputs, the queue and the replies of the calls awaiting the core, the
//! host event stream and the taps a system watches.

use super::calls::{Awaited, Call, Calls};
use super::time::Timebase;
use lk_agents_abi::Spawner;
use lk_agents_abi::voice::io::{AudioOutput, TextOutput};
use lk_agents_voice::agent_session::{
    Answer, Command, Event, HostEvent, Input, Origin, Outputs, QueueClosed, RequestId, Sent,
    SessionView, SystemTap,
};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::trace::{Journal, Step};
use parking_lot::Mutex;
use std::sync::Arc;

/// The loop's one queue: every source sends on it, so the loop receives one [`Event`] at a time
/// in arrival order and merges it with its timers by a stated rule; the trace is a function of the
/// event sequence.
#[derive(Clone)]
pub struct Tx(tokio::sync::mpsc::UnboundedSender<Event>);

impl Tx {
    pub fn new(inner: tokio::sync::mpsc::UnboundedSender<Event>) -> Self {
        Tx(inner)
    }

    /// A queue nobody reads: what a handle made outside a session sends on.
    pub fn disconnected() -> Self {
        Tx(tokio::sync::mpsc::unbounded_channel().0)
    }

    pub fn send(&self, event: impl Into<Event>) -> Result<(), QueueClosed> {
        self.0.send(event.into()).map_err(|_| QueueClosed)
    }

    pub fn send_opened(
        &self,
        opened: lk_agents_voice::room_io::StreamOpened,
    ) -> Result<(), lk_agents_voice::room_io::StreamOpened> {
        self.0.send(Event::LegOpened(opened)).map_err(|error| {
            let Event::LegOpened(opened) = error.0 else { unreachable!() };
            opened
        })
    }

    pub fn send_from(&self, origin: Origin, command: Command) -> Result<(), QueueClosed> {
        self.send(Input::Host(Sent { origin, command }))
    }

    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Resolves when the loop has dropped its receiver.
    pub async fn closed(&self) {
        self.0.closed().await
    }
}

impl std::fmt::Debug for Tx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tx")
    }
}

/// One unbounded channel per subscriber; a departed one is forgotten on publish.
#[derive(Clone, Default)]
pub struct WatchHub {
    taps: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<SystemTap>>>>,
}

impl WatchHub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<SystemTap> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.taps.lock().push(tx);
        rx
    }

    /// Cheap with none: the common session runs no system.
    pub fn publish(&self, item: SystemTap) {
        let mut taps = self.taps.lock();
        taps.retain(|tap| tap.send(item.clone()).is_ok());
    }

    pub fn is_empty(&self) -> bool {
        self.taps.lock().is_empty()
    }
}

impl std::fmt::Debug for WatchHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatchHub").field("taps", &self.taps.lock().len()).finish()
    }
}

/// Everything a peer of the host may touch, and nothing else.
#[derive(Clone)]
pub struct PeerHandle {
    /// The loop's queue: commands, and the audio taps' installs.
    pub queue: Tx,
    pub(crate) events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub journal: Journal,
    pub time: Timebase,
    pub spawner: Arc<dyn lk_agents_abi::Spawner>,
    pub(crate) watch: WatchHub,
    pub(crate) resources: lk_agents_abi::resources::Resources,
    /// Flips at `begin_closing`: a system's own drain ends on it like the driver's do.
    pub closing: tokio::sync::watch::Receiver<bool>,
}

impl PeerHandle {
    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<SystemTap> {
        self.watch.subscribe()
    }

    pub fn send(&self, origin: Origin, cmd: impl Into<Command>) -> Result<(), QueueClosed> {
        self.queue.send_from(origin, cmd.into())
    }

    pub fn journal(&self, step: Step) {
        self.journal.record(step);
    }

    /// A host event to the host's drain; handed back when the drain has gone.
    pub fn deliver(&self, event: HostEvent) -> Result<(), HostEvent> {
        self.events.send(event).map_err(|refused| refused.0)
    }

    /// A tap to every watcher.
    pub fn publish(&self, tap: SystemTap) {
        self.watch.publish(tap);
    }

    pub fn emit(&self, ev: SessionEvent) {
        if self.journal.is_on() {
            self.journal(Step::Emitted(ev.journal_form()));
        }
        let _ = self.events.send(HostEvent::Session(ev));
    }

    /// The microphone's own frames, never the gate's silence. A second tap replaces the first.
    pub fn install_audio_tap(
        &self,
        sink: Arc<dyn lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame>>,
    ) -> lk_agents_abi::resources::ResourceId {
        let sink = self.resources.register_audio_sink(sink);
        let channel = lk_agents_voice::audio::AudioChannel::Amd;
        let _ = self
            .queue
            .send(lk_agents_voice::data_plane::ForwarderCommand::Install { channel, sink });
        sink
    }

    /// Only if this tap is still the one in place.
    pub fn remove_audio_tap(&self, tap: lk_agents_abi::resources::ResourceId) {
        let _ = self.queue.send(lk_agents_voice::data_plane::ForwarderCommand::Remove(tap));
    }
}

impl std::fmt::Debug for PeerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerHandle").field("time", &self.time).field("watch", &self.watch).finish()
    }
}

pub struct SessionHandle {
    pub api: Tx,
    /// Every event, in the order the driver emitted it.
    pub events: Rx<HostEvent>,
    /// The last keyterm list pushed to the recogniser; `None` before any push.
    pub keyterms: tokio::sync::watch::Receiver<Option<Vec<String>>>,
    /// The driving agent's context and state, as last published.
    pub chat_ctx: tokio::sync::watch::Receiver<Arc<lk_agents_abi::ChatContext>>,
    /// What a system runs on; the host starts the systems it configures on this after `start()`.
    pub peer: PeerHandle,
    pub ivr_detection: bool,
    /// The session's id counter: a host speech call takes its speech's id from it.
    pub speech_ids: Arc<std::sync::atomic::AtomicU64>,
    /// The replies of the calls the host awaits, until the core answers them.
    pub calls: Arc<Calls>,
}

pub type Rx<T> = tokio::sync::mpsc::UnboundedReceiver<T>;

pub struct SessionShared {
    pub time: Timebase,
    pub view: tokio::sync::watch::Sender<Arc<SessionView>>,
    /// `conn_options=`: the budgets the engine retries the LLM and the TTS on.
    pub conn: lk_agents_abi::voice::session_options::SessionConnOptions,
    /// `session.output` as the core last published it.
    pub output: tokio::sync::watch::Sender<Outputs>,
    pub recorder: Arc<std::sync::RwLock<Option<lk_agents_voice::recorder_io::RecorderTap>>>,
    /// A pipeline is built per segment, so a half-open `**` cannot leak into the next utterance.
    pub tts_transforms: Vec<lk_agents_voice::transcription::TransformSpec>,
    /// The loop's queue: a task's reports, and the commands of the speech handle a host tool
    /// receives. Unbounded: a task must never block on the loop draining it.
    pub queue: Tx,
    /// The replies of the calls a task awaits, until the core answers them.
    pub calls: Arc<Calls>,
    /// Every speech's live handle, which the core's updates keep current.
    pub speeches: Arc<crate::voice::speech_handle::Speeches>,
    /// Flips once, when the legs close or the runtime is dropped: every forwarder, drain, pump and
    /// reader selects on it, or one parked on an idle source would outlive the session.
    pub closing: tokio::sync::watch::Sender<bool>,
    /// The host runs hooks (`hook_bound`): a step waits for `function_tools_executed`'s handlers.
    pub hook_host: bool,
    /// Never `tokio::spawn`.
    pub spawner: Arc<dyn Spawner>,
    /// The task enters its own frame; it never reads the driver's `current`.
    pub frames: Arc<lk_agents_abi::utils::frame::FrameTable>,
    /// A task cannot reach `AgentSession::mint`; the `g` prefix keeps the id spaces apart.
    pub gen_ids: std::sync::atomic::AtomicU64,
    pub session_id: String,
    pub chat_ctx: tokio::sync::watch::Sender<Arc<lk_agents_abi::ChatContext>>,
    /// The providers by id.
    pub resources: lk_agents_abi::resources::Resources,
    /// `session.keyterms`: the last list pushed to the recogniser.
    pub keyterms: tokio::sync::watch::Sender<Option<Vec<String>>>,
    /// A task's steps reach the loop's journal only when one is observed.
    pub journal: lk_agents_voice::trace::Journal,
}

impl SessionShared {
    pub fn mint_item_id(&self) -> String {
        let n = self.gen_ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("{}_item_g{n:04}", self.session_id)
    }

    /// A closed channel means the session is ending; not an error.
    pub fn report(&self, ev: GenEvent) {
        let _ = self.queue.send(ev);
    }

    /// A call a task awaits, answered with what `pick` takes.
    pub fn open_call<T>(&self, pick: fn(Answer) -> Option<T>) -> (Call, Awaited<T>) {
        let n = self.gen_ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::calls::open(RequestId::task(n), pick)
    }

    /// `ev`, naming `call`, with its reply.
    pub fn report_awaited(&self, ev: GenEvent, call: Call) {
        let _ = self.calls.send(&self.queue, Input::Gen(ev), call);
    }

    /// Through the queue, so the step is journalled in the loop's order.
    pub fn trace(&self, step: impl Into<lk_agents_voice::trace::Step>) {
        if self.journal.is_on() {
            self.report(GenEvent::Trace(step.into()));
        }
    }

    pub fn recorder_tap(&self) -> Option<lk_agents_voice::recorder_io::RecorderTap> {
        self.recorder.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The audio sink the core installed, switched on or not.
    pub fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        let id = self.output.borrow().audio.as_ref()?.id;
        self.resources.audio_output(id)
    }

    /// The transcription sink the core installed, switched on or not.
    pub fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        let id = self.output.borrow().transcription.as_ref()?.id;
        self.resources.text_output(id)
    }

    /// `output.audio_enabled`: the switch, which a speech freezes at its voice leg.
    pub fn audio_enabled(&self) -> bool {
        self.output.borrow().audio_enabled
    }

    /// `output.transcription_enabled`.
    pub fn transcription_enabled(&self) -> bool {
        self.output.borrow().transcription_enabled
    }
}
