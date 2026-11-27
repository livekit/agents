use super::calls::{Awaited, Call};
use super::time::Timebase;
use crate::telemetry;
use crate::voice::VoiceRuntimeEvent;
use lk_agents_abi::Spawner;
use lk_agents_abi::voice::io::AudioOutput;
use lk_agents_voice::agent_session::{
    Answer, Command, Outputs, QueueClosed, RequestId, Sent, SessionView, VoiceCommandOrigin,
    VoiceEvent, VoiceHostEvent,
};
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::trace::Step;
use parking_lot::Mutex;
use std::sync::Arc;

type JournalObserver = dyn FnMut(&Step) + Send;

/// Where the driver's steps go: to the observer the simulation or a test installed, or nowhere.
/// Production installs none, so a session keeps no record of its steps.
#[derive(Clone, Default)]
pub struct Journal(Option<Arc<Mutex<JournalObserver>>>);

impl std::fmt::Debug for Journal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Journal").field(&self.is_on()).finish()
    }
}

impl Journal {
    pub fn observed(observer: impl FnMut(&Step) + Send + 'static) -> Self {
        Journal(Some(Arc::new(Mutex::new(observer))))
    }

    /// Whether anyone reads the steps: a step that is costly to build is built only then.
    pub fn is_on(&self) -> bool {
        self.0.is_some()
    }

    pub fn record(&self, step: impl Into<Step>) {
        if let Some(observer) = &self.0 {
            (observer.lock())(&step.into());
        }
    }
}

/// The loop's one queue: every source sends on it, so the loop receives one [`VoiceRuntimeEvent`] at a time
/// in arrival order and merges it with its timers by a stated rule; the trace is a function of the
/// event sequence.
#[derive(Clone)]
pub struct Tx(tokio::sync::mpsc::UnboundedSender<VoiceRuntimeEvent>);

impl Tx {
    pub fn new(inner: tokio::sync::mpsc::UnboundedSender<VoiceRuntimeEvent>) -> Self {
        Tx(inner)
    }

    /// A queue nobody reads: what a handle made outside a session sends on.
    pub fn disconnected() -> Self {
        Tx(tokio::sync::mpsc::unbounded_channel().0)
    }

    pub fn send(&self, event: impl Into<VoiceRuntimeEvent>) -> Result<(), QueueClosed> {
        self.0.send(event.into()).map_err(|_| QueueClosed)
    }

    /// A closed queue hands the result back, for its stream to be retired.
    pub fn send_opened(
        &self,
        opened: crate::voice::events::StreamOpened,
    ) -> Result<(), Box<crate::voice::events::StreamOpened>> {
        self.0.send(VoiceRuntimeEvent::LegOpened(opened)).map_err(|error| {
            let VoiceRuntimeEvent::LegOpened(opened) = error.0 else { unreachable!() };
            Box::new(opened)
        })
    }

    pub fn send_from(
        &self,
        origin: VoiceCommandOrigin,
        command: Command,
    ) -> Result<(), QueueClosed> {
        self.send(VoiceEvent::Host(Sent { origin, command }))
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

/// A host event and, on a speech's `speech_done`, that speech's `reported` writer: the drain
/// finishes it once the host handled the event, and dropping it unhandled ends the wait.
pub(crate) type HostEvent = (VoiceHostEvent, Option<lk_agents_net::signal::Producer<bool>>);

/// The host-event stream. Its end is the runtime's: once the runtime is gone, dropped or
/// panicked, the stream ends after what was queued, whatever other senders remain.
pub(crate) struct HostEvents {
    events: Rx<HostEvent>,
    runtime: lk_agents_net::signal::Cell<()>,
}

impl HostEvents {
    pub(crate) fn new(events: Rx<HostEvent>) -> (Self, lk_agents_net::signal::Producer<()>) {
        let (end, runtime) = lk_agents_net::signal::signal(());
        (HostEvents { events, runtime }, end)
    }

    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }

    #[cfg(test)]
    pub(crate) fn try_recv(&mut self) -> Option<HostEvent> {
        self.events.try_recv().ok()
    }

    /// `None` once every sender is gone, or the runtime is and nothing is queued.
    pub(crate) async fn recv(&mut self) -> Option<HostEvent> {
        tokio::select! {
            biased;
            event = self.events.recv() => event,
            _ = self.runtime.wait_for(|()| false) => self.events.try_recv().ok(),
        }
    }
}

/// Everything a peer of the host may touch, and nothing else.
#[derive(Clone)]
pub struct PeerHandle {
    /// The loop's queue.
    pub queue: Tx,
    pub(crate) events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub journal: Journal,
    pub time: Timebase,
    /// The session's context, under its `agent_session`.
    pub session: telemetry::Cx,
}

impl PeerHandle {
    /// A host event to the host's drain; handed back when the drain has gone.
    pub fn deliver(
        &self,
        event: VoiceHostEvent,
        reported: Option<lk_agents_net::signal::Producer<bool>>,
    ) -> Result<(), Box<VoiceHostEvent>> {
        self.events.send((event, reported)).map_err(|refused| Box::new(refused.0.0))
    }
}

impl std::fmt::Debug for PeerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerHandle").field("time", &self.time).finish()
    }
}

pub struct SessionHandle {
    pub api: Tx,
    pub(crate) conversation: crate::session::CommandPort,
    /// Every event, in the order the driver emitted it.
    pub(crate) events: HostEvents,
    /// The last keyterm list pushed to the recogniser; `None` before any push.
    pub keyterms: lk_agents_net::signal::Cell<Option<Vec<String>>>,
    /// The driving agent's context and state, as last published.
    pub chat_ctx: lk_agents_net::signal::Cell<Arc<lk_agents_abi::ChatContext>>,
    /// What a system runs on; the host starts the systems it configures on this after `start()`.
    pub peer: PeerHandle,
    /// The session's id counter: a host speech call takes its speech's id from it.
    pub speech_ids: Arc<std::sync::atomic::AtomicU64>,
    /// With the counter, a host speech's id (`SpeechId::nth`).
    pub id_seed: u64,
}

pub type Rx<T> = tokio::sync::mpsc::UnboundedReceiver<T>;

pub struct SessionShared {
    pub(crate) tool_bodies: super::calls::ToolBodies,
    /// The session's report as the loop closed.
    pub(in crate::voice) report: super::FinalReport,
    /// The session's span recording: its records reach the exporter, and the work it runs opens
    /// its spans under the core's.
    pub(in crate::voice) tracer: telemetry::Tracer,
    /// The session's span context, outside any span: its job and conversation.
    pub(in crate::voice) trace: telemetry::Cx,
    pub(in crate::voice) realtime_spans:
        crate::voice::generation::realtime_generation::RealtimeSpans,
    /// The host callable answering the current agent's tools, which every realtime session the
    /// session opens offers.
    pub(in crate::voice) realtime_tools: Option<lk_agents_abi::HostRef>,
    pub(in crate::voice) time: Timebase,
    pub(in crate::voice) view: lk_agents_net::signal::Cell<Arc<SessionView>>,
    /// `conn_options=`: the budgets the engine retries the LLM and the TTS on.
    pub(in crate::voice) conn: lk_agents_abi::voice::session_options::SessionConnOptions,
    /// `session.output` as the core last published it.
    pub(in crate::voice) output: lk_agents_net::signal::Cell<Outputs>,
    /// The recorder, fed what the outputs play and still have to play.
    pub(in crate::voice) recorder: Arc<crate::voice::recorder::RecorderFeed>,
    /// The loop's queue: a task's reports, and the commands of the speech handle a host tool
    /// receives. Unbounded: a task must never block on the loop draining it.
    pub(in crate::voice) queue: Tx,
    /// Every speech's live handle, which the core's updates keep current.
    pub(in crate::voice) speeches: Arc<crate::voice::speech_handle::Speeches>,
    /// Cancelled once, when the legs close or the runtime is dropped: every forwarder, drain,
    /// pump and reader selects on it, or one parked on an idle source would outlive the session.
    pub(in crate::voice) closing: tokio_util::sync::CancellationToken,
    /// Never `tokio::spawn`.
    pub(in crate::voice) spawner: Arc<dyn Spawner>,
    /// The executor's tasks: readers, drains, pumps, provider calls and closes, drained by the
    /// runtime's close.
    pub(in crate::voice) tasks: Arc<lk_agents_net::task::TaskSet>,
    /// The requests a task opens (`open_call`).
    pub(in crate::voice) gen_ids: std::sync::atomic::AtomicU64,
    pub(in crate::voice) chat_ctx: lk_agents_net::signal::Cell<Arc<lk_agents_abi::ChatContext>>,
    pub(in crate::voice) resources: crate::voice::resources::Resources,
    /// `session.keyterms`: the last list pushed to the recogniser.
    pub(in crate::voice) keyterms: lk_agents_net::signal::Cell<Option<Vec<String>>>,
    /// A task's steps reach the loop's journal only when one is observed.
    pub(in crate::voice) journal: Journal,
}

impl SessionShared {
    /// Where work that outlives its caller runs, and an abandoned resource's close: the
    /// session's tasks.
    pub(in crate::voice) fn background(&self) -> lk_agents_net::owned::Background {
        lk_agents_net::owned::Background::new(
            Arc::clone(&self.tasks),
            Arc::clone(&self.spawner),
            self.time.clock(),
        )
    }

    /// What a speech's host calls run under: its turn, else the session.
    pub fn turn_cx(&self, turn: Option<telemetry::SpanContext>) -> telemetry::Cx {
        match turn {
            Some(turn) => self.trace.clone().under::<telemetry::AgentTurn>(turn),
            None => self.trace.clone(),
        }
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
        let _ = super::calls::send(&self.queue, VoiceEvent::Gen(ev), call);
    }

    /// Through the queue, so the step is journalled in the loop's order.
    pub fn trace(&self, step: impl Into<lk_agents_voice::trace::Step>) {
        if self.journal.is_on() {
            self.report(GenEvent::Trace(step.into()));
        }
    }

    pub fn recorder_tap(&self) -> Option<crate::voice::recorder::RecorderTap> {
        self.recorder.tap()
    }

    /// The audio sink the core installed, switched on or not.
    pub fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        let id = self.output.with(|o| Some(o.audio.as_ref()?.id))?;
        self.resources.audio_output(id)
    }

    /// `output.audio_enabled`: the switch, which a speech freezes at its voice leg.
    pub fn audio_enabled(&self) -> bool {
        self.output.with(|o| o.audio_enabled)
    }
}

/// Only the queue owner publishes accepted decision state. Spawned work receives readers.
pub(crate) struct Publications {
    /// Dropping the publications cancels it too.
    pub closing: tokio_util::sync::DropGuard,
    pub view: lk_agents_net::signal::Producer<Arc<SessionView>>,
    pub output: lk_agents_net::signal::Producer<Outputs>,
    pub chat_ctx: lk_agents_net::signal::Producer<Arc<lk_agents_abi::ChatContext>>,
    pub keyterms: lk_agents_net::signal::Producer<Option<Vec<String>>>,
}
