use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use lk_agents_abi::{
    AudioFrame, EotPrediction, LlmChunk, SpeechEvent, Ts, VadEvent, errors::ProviderError,
    utils::stream::StreamInput,
};
use lk_agents_runtime::voice::VoiceSession;
use tokio::sync::{mpsc, oneshot};

use crate::{
    input::Input,
    record::{Detail, Kind, Observed, Sink, Stamps},
};

// Invocation controls belong to the harness: a fake sends its records and controls over one
// channel, so a delivery holds no lock and a fake recording mid-delivery cannot re-enter one.
// A delivery to a closed receiver is a recorded, rejected attempt, never a panic.

mod avatar;
mod console;
pub use console::{FakeCli, FakeConsolePort};
mod eot;
pub mod hazard;
mod interruption;
mod llm;
mod nodes;
mod output;
pub mod realtime;
pub mod room;
pub(crate) mod storage;
mod stt;
pub(crate) mod tools;
mod tts;
mod vad;
pub mod vocabulary;

pub use avatar::FakeAvatarWorker;
pub use eot::{FakeEot, THRESHOLD as EOT_THRESHOLD};
pub use interruption::{FakeInterruption, FakeInterruptionSource};
pub use llm::{COMPLETION_TOKENS, FakeLlm};
pub use nodes::FakeNodes;
pub use output::{
    FakeAudioInput, FakeAudioOutput, FakeRecorder, FakeSoundStream, FakeTextOutput, FakeVideoInput,
    HALF, room_camera,
};
pub use realtime::{FakeDuplex, FakeRealtime, RealtimeDelays, ServerSetup, Vad as RealtimeVad};
pub use storage::{FakeStore, death_resends, storage_faults, worker_deaths};
pub use stt::FakeStt;
pub use tools::{FakeTools, TEMPLATED_REPLY, TEMPLATED_TOOL, completion_value, frame};
pub use tts::{FakeTts, MS_PER_WORD, TtsOptions};
pub use vad::FakeVad;

pub const SAMPLE_RATE: u32 = 24_000;
/// One fixture chunk is 10 ms of mono audio.
pub const SAMPLES_PER_CHUNK: usize = (SAMPLE_RATE / 100) as usize;

/// A provider quantity by its measured deciles: min, p10 to p90, max (from `calibration.json`).
pub type Measured = [u64; 11];

/// The share of provider delays that stray from the measured spread, and the deadlines armed in
/// the run, in ms.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timing {
    pub stray: f64,
    pub deadlines: Vec<u64>,
}

/// A value across the whole measured span, distributed as measured.
pub fn measured(rng: &mut fastrand::Rng, m: &Measured) -> u64 {
    let at = rng.f64() * 10.0;
    let i = (at as usize).min(9);
    let (low, high) = (m[i] as f64, m[i + 1] as f64);
    (low + (high - low) * (at - i as f64)).round() as u64
}

/// A provider delay in ms. Most draws are [`measured`]; a stray one is either a boundary (none,
/// a tick, a frame, or a tick around a deadline in play) or a stall ten to a hundred times the
/// most ever measured.
pub fn delay(rng: &mut fastrand::Rng, m: &Measured, timing: &Timing) -> u64 {
    if rng.f64() >= timing.stray {
        return measured(rng, m);
    }
    if rng.bool() {
        return match rng.usize(0..3 + 3 * timing.deadlines.len()) {
            0 => 0,
            1 => 1,
            2 => 10,
            k => (timing.deadlines[(k - 3) / 3] + (k - 3) as u64 % 3).saturating_sub(1),
        };
    }
    (m[10].max(1) as f64 * 10f64.powf(1.0 + rng.f64())).round() as u64
}

/// One thing the fakes opened. `nth` is the number a case's `Ref` carries, so the n-th chat
/// registered is the n-th a case can point at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Slot {
    pub kind: Kind,
    pub nth: u32,
}

/// The real input side of one invocation, owned by the harness until consumed.
pub enum Control {
    Chat(mpsc::UnboundedSender<Result<LlmChunk, ProviderError>>),
    Tts(Arc<tts::Synthesis>),
    /// A recognition stream, when it opened, and when its utterance began: the instants a
    /// transcript's `start_time`/`end_time` are measured from.
    Stt(mpsc::UnboundedSender<Result<SpeechEvent, ProviderError>>, Ts, Option<Ts>),
    Vad(mpsc::UnboundedSender<Result<VadEvent, ProviderError>>),
    Prediction(oneshot::Sender<Result<EotPrediction, ProviderError>>),
    Interruption(Arc<interruption::FakeInterruption>),
    Tool(oneshot::Sender<tools::Body>),
    /// A tool that answered `Detached`; its end comes as a `ToolUpdate` command.
    Running,
    /// One audio segment, which takes the case's playout reports.
    Segment(Arc<output::Playout>),
    /// An audio output as a whole, for a reset.
    Output(Arc<output::Playout>),
    Video(mpsc::UnboundedSender<u64>),
    RoomVideo(Arc<crate::fakes::room::FakeTrackPublication>),
    Realtime(Arc<realtime::Socket>),
    /// A realtime turn in progress; its end comes through the socket.
    Generation,
}

impl Control {
    /// A synthesis stream the product finished reading is spent though its control remains.
    pub(crate) fn open(&self) -> bool {
        match self {
            Control::Tts(synthesis) => synthesis.open(),
            _ => true,
        }
    }
}

/// One thing the fakes did. Each record carries the `Observed` stamp taken at the event, from the
/// run's one counter, so the harness may drain them later and still build the happens-before
/// order.
pub enum Record {
    Boundary {
        at: Observed,
        outcome: crate::record::BoundaryOutcome,
    },
    Storage {
        at: Observed,
        step: crate::record::StorageStep,
    },
    /// An invocation a fake opened, with its control for the harness. Outputs carry none.
    Opened {
        at: Observed,
        slot: Slot,
        detail: Detail,
        control: Option<Control>,
    },
    Closed {
        at: Observed,
        slot: Slot,
    },
    /// The product physically attached or detached this object. Attachments may overlap.
    OutputAttachment {
        at: Observed,
        output: u32,
        attached: bool,
    },
    /// A synthesis chunk, its text, and its word stamps (start seconds from the first audio).
    Produced {
        at: Observed,
        tts: u32,
        chunk: u16,
        text: String,
        stamps: Vec<(String, f64)>,
    },
    Generated {
        at: Observed,
        generation: u32,
        chunk: u16,
    },
    /// A fake panicked or never answered here instead (`hazard`).
    Fault {
        at: Observed,
        site: hazard::Site,
        fault: hazard::Fault,
        /// The provider stream it struck, if it struck one.
        on: Option<Slot>,
    },
    /// A tool body refused the host's cancellation and runs on.
    CancelRefused {
        at: Observed,
        call_id: String,
    },
    /// What the n-th realtime socket's server refused, or answered from the wrong state.
    Breach {
        at: Observed,
        socket: u32,
        breach: crate::record::Breach,
    },
    /// How much of a realtime generation was heard; `whole` when all of it captured so far has
    /// played or been cut.
    Heard {
        at: Observed,
        generation: u32,
        ms: u64,
        whole: bool,
    },
    /// What a realtime socket's server held as a connection closed.
    Held {
        at: Observed,
        socket: u32,
        items: Vec<(u32, u64)>,
    },
    /// The driver ended a provider stream's input: it is owed no audio from here.
    InputEnded {
        at: Observed,
        slot: Slot,
    },
    Forwarded {
        at: Observed,
        slot: Slot,
        tags: Vec<crate::record::Tag>,
    },
    Sink {
        at: Observed,
        sink: Sink,
    },
    /// The product was handed a synthesis stream's first audio.
    Delivered {
        at: Observed,
        slot: Slot,
    },
    /// A synthesis stream opened whose first audio the voice produces this long after.
    FirstAudioDue {
        at: Observed,
        slot: Slot,
        after_ms: u64,
    },
    /// An input processor's failing span opened, by the processor's host object and label.
    ProcessorSpan {
        at: Observed,
        processor: u64,
        label: String,
    },
    /// An SDK object was bound to the room's host loop, or on the core's thread.
    HostBinding {
        at: Observed,
        object: &'static str,
        thread: room::Thread,
    },
    /// The session opened one of an agent's nodes.
    Node {
        at: Observed,
        agent: String,
        node: crate::record::NodeUse,
    },
    /// Which output a flushed segment belongs to: a fact kept for the run, not a stamped event.
    Segment {
        segment: u32,
        output: u32,
    },
}

/// The boundary an input is delivered to, or `None` when it is not a delivery.
pub fn boundary_of(input: &Input) -> Option<Kind> {
    Some(match input {
        Input::LlmText { .. }
        | Input::LlmToolCall { .. }
        | Input::LlmEnd { .. }
        | Input::LlmFail { .. } => Kind::Chat,
        Input::TtsHold { .. } | Input::TtsRelease { .. } | Input::TtsFail { .. } => Kind::Tts,
        Input::SttStart { .. }
        | Input::SttInterim { .. }
        | Input::SttPreflight { .. }
        | Input::SttFinal { .. }
        | Input::SttEnd { .. }
        | Input::SttUsage { .. }
        | Input::SttFail { .. } => Kind::Stt,
        Input::VadInference { .. } | Input::VadStart { .. } | Input::VadEnd { .. } => Kind::Vad,
        Input::EotAnswer { .. } | Input::EotFail { .. } => Kind::Prediction,
        Input::OverlapVerdict { .. } | Input::OverlapFail { .. } => Kind::Interruption,
        Input::ToolAnswered { .. } => Kind::Tool,
        Input::PlayoutStarted { .. }
        | Input::PlayoutFinished { .. }
        | Input::PlayoutFinishLost { .. }
        | Input::PlayoutProgressed { .. } => Kind::Segment,
        Input::SinkReset { .. } => Kind::Output,
        Input::VideoFrames { .. } => Kind::Video,
        Input::RealtimeEvent { .. } => Kind::Realtime,
        _ => return None,
    })
}

/// The kinds the fakes mint identities for; the rest the harness opens from the journal.
const MINTED: [Kind; 14] = [
    Kind::Chat,
    Kind::Tts,
    Kind::Stt,
    Kind::Vad,
    Kind::Eot,
    Kind::Prediction,
    Kind::Interruption,
    Kind::Tool,
    Kind::Segment,
    Kind::Output,
    Kind::Mic,
    Kind::Video,
    Kind::Realtime,
    Kind::Generation,
];

struct StoragePending(Arc<Registry>);

impl Drop for StoragePending {
    fn drop(&mut self) {
        self.0.storage_pending.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What every fake holds: the stamp counter, the identity counters, and the way back to the
/// harness.
pub struct Registry {
    stamps: Stamps,
    /// The run's spawner for what the fakes run beside the session (`drive::World`).
    pub world: Arc<dyn lk_agents_abi::Spawner>,
    records: mpsc::UnboundedSender<Record>,
    /// Atomic so a fake can mint an identity from inside a product call without blocking.
    next: BTreeMap<Kind, AtomicU32>,
    tool_serial: AtomicU32,
    /// Every synthesis fails at the open while set.
    voice_down: std::sync::atomic::AtomicBool,
    refuse_cancel: std::sync::atomic::AtomicBool,
    /// What a tool body calls, as `RunContext` does.
    session: std::sync::OnceLock<Arc<VoiceSession>>,
    /// First audio each synthesis still owes on its own clock, which no case answer releases.
    voice_due: std::sync::Mutex<BTreeMap<u32, tokio::time::Instant>>,
    /// When the last storage answer owed on the fake's own clock is due.
    storage_due: std::sync::Mutex<lk_agents_abi::Ts>,
    /// Acquires of the voice lane still awaited.
    storage_pending: AtomicU32,
    chat_responses: std::sync::Mutex<BTreeMap<String, u32>>,
    /// Armed once the session started, in a stressed run only.
    hazards: std::sync::OnceLock<hazard::Hazards>,
}

impl Registry {
    pub fn new(
        stamps: Stamps,
        world: Arc<dyn lk_agents_abi::Spawner>,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<Record>) {
        let (records, receiver) = mpsc::unbounded_channel();
        let next = MINTED.into_iter().map(|kind| (kind, AtomicU32::new(0))).collect();
        let registry = Registry {
            stamps,
            world,
            records,
            next,
            tool_serial: AtomicU32::new(0),
            voice_down: std::sync::atomic::AtomicBool::new(false),
            refuse_cancel: std::sync::atomic::AtomicBool::new(false),
            session: std::sync::OnceLock::new(),
            voice_due: std::sync::Mutex::default(),
            storage_due: std::sync::Mutex::new(lk_agents_abi::Ts::ZERO),
            storage_pending: AtomicU32::new(0),
            chat_responses: std::sync::Mutex::default(),
            hazards: std::sync::OnceLock::new(),
        };
        (Arc::new(registry), receiver)
    }

    /// From now on every site draws its fault from `seed`.
    pub fn arm_hazards(&self, seed: u64) {
        let _ = self.hazards.set(hazard::Hazards::new(seed));
    }

    /// The fault to play out at `site` instead of answering, recorded; `None` almost always.
    pub fn hazard(&self, site: hazard::Site) -> Option<hazard::Fault> {
        let fault = self.hazards.get()?.draw()?;
        self.struck(site, fault, None);
        Some(fault)
    }

    /// Whether the host never answers at `site`: a call with no future of the fake's to panic in.
    pub fn hang(&self, site: hazard::Site) -> bool {
        let hung = self.hazards.get().and_then(hazard::Hazards::draw).is_some();
        if hung {
            self.struck(site, hazard::Fault::Hang, None);
        }
        hung
    }

    /// A provider stream's fault, drawn at its open (`hazard::StreamHazard`).
    pub fn stream_hazard(&self, next: hazard::Site, close: hazard::Site) -> hazard::StreamHazard {
        self.hazards.get().map(|h| h.draw_stream(next, close)).unwrap_or_default()
    }

    /// A stream's close fault as its port reports it: a panic at the poll, or a teardown that
    /// never answers, abandoned at `deadline` on the session's clock, as the binding bounds a
    /// host's `aclose()` (`py_streams`). No port's close outlives its deadline.
    pub(super) async fn strike_close(
        &self,
        fault: hazard::Fault,
        site: hazard::Site,
        deadline: Ts,
    ) -> lk_agents_abi::Closed {
        let clock = self.stamps.clock();
        lk_agents_abi::Closed::bounded(clock.as_ref(), deadline, hazard::strike(fault, site)).await
    }

    /// Stream `on`'s fault at `site`, now due: recorded.
    pub fn stream_fault(
        &self,
        hazard: &mut hazard::StreamHazard,
        site: hazard::Site,
        on: Slot,
    ) -> Option<hazard::Fault> {
        let fault = hazard.due(site)?;
        self.struck(site, fault, Some(on));
        Some(fault)
    }

    fn struck(&self, site: hazard::Site, fault: hazard::Fault, on: Option<Slot>) {
        hazard::count(fault);
        self.write(Record::Fault { at: self.stamps.next(), site, fault, on });
    }

    pub fn set_voice_down(&self, down: bool) {
        self.voice_down.store(down, Ordering::SeqCst);
    }

    pub(super) fn voice_down(&self) -> bool {
        self.voice_down.load(Ordering::SeqCst)
    }

    pub fn set_refuse_cancel(&self, refuse: bool) {
        self.refuse_cancel.store(refuse, Ordering::SeqCst);
    }

    pub(super) fn refuses_cancel(&self) -> bool {
        self.refuse_cancel.load(Ordering::SeqCst)
    }

    pub(super) fn record_cancel_refused(&self, call_id: String) {
        let at = self.stamps.next();
        self.write(Record::CancelRefused { at, call_id });
    }

    fn record_storage(&self, step: crate::record::StorageStep) {
        self.write(Record::Storage { at: self.stamps.next(), step });
    }

    pub(super) fn record_boundary(
        &self,
        operation: crate::record::BoundaryOperation,
        error: Option<String>,
    ) {
        self.write(Record::Boundary {
            at: self.stamps.next(),
            outcome: crate::record::BoundaryOutcome { operation, error },
        });
    }

    pub fn attach_session(&self, session: Arc<VoiceSession>) {
        assert!(self.session.set(session).is_ok(), "session attached twice");
    }

    pub fn request_of(&self, response_id: &str) -> Option<u32> {
        self.chat_responses.lock().unwrap().get(response_id).copied()
    }

    fn response_id(&self, nth: u32) -> String {
        let mut responses = self.chat_responses.lock().unwrap();
        if let Some((id, _)) = responses.iter().find(|(_, chat)| **chat == nth) {
            return id.clone();
        }
        // An opaque wire id; correlation uses the registry, never its spelling.
        let id = format!("{:016x}", u64::from(nth).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        responses.insert(id.clone(), nth);
        id
    }

    /// The session's instant, for what a fake stamps on the driver's clock.
    pub(super) fn now(&self) -> Ts {
        self.stamps.ts()
    }

    pub(super) fn session(&self) -> Option<&Arc<VoiceSession>> {
        self.session.get()
    }

    fn mint(&self, kind: Kind) -> Slot {
        let counter = self.next.get(&kind).expect("a kind the fakes mint");
        Slot { kind, nth: counter.fetch_add(1, Ordering::SeqCst) }
    }

    /// A fake on a detached task may record after the run dropped the receiver; that is fine.
    fn write(&self, record: Record) {
        let _ = self.records.send(record);
    }

    pub(super) fn node_used(&self, agent: &str, node: crate::record::NodeUse) {
        self.write(Record::Node { at: self.stamps.next(), agent: agent.to_owned(), node });
    }

    pub(super) fn new_audio_output(&self, playout: Arc<output::Playout>) -> u32 {
        self.register(Kind::Output, Detail::Output { audio: true }, Control::Output(playout)).nth
    }

    pub fn new_output(&self, audio: bool) -> u32 {
        let at = self.stamps.next();
        let slot = self.mint(Kind::Output);
        self.write(Record::Opened { at, slot, detail: Detail::Output { audio }, control: None });
        slot.nth
    }

    /// Minted here so its `nth` is what a `Ref<Mic>` names.
    pub fn new_mic(&self) -> u32 {
        let at = self.stamps.next();
        let slot = self.mint(Kind::Mic);
        self.write(Record::Opened { at, slot, detail: Detail::None, control: None });
        slot.nth
    }

    /// Minted so the frames pushed into the end-of-turn leg are recorded against a nameable slot.
    pub fn new_eot(&self) -> u32 {
        let at = self.stamps.next();
        let slot = self.mint(Kind::Eot);
        self.write(Record::Opened { at, slot, detail: Detail::None, control: None });
        slot.nth
    }

    pub(super) fn register(&self, kind: Kind, detail: Detail, control: Control) -> Slot {
        let at = self.stamps.next();
        let slot = self.mint(kind);
        self.write(Record::Opened { at, slot, detail, control: Some(control) });
        slot
    }

    pub(super) fn attach_segment(&self, segment: u32, output: u32) {
        self.write(Record::Segment { segment, output });
    }

    /// A room stream handed its reader the failure that opens `processor`'s failing span.
    pub fn record_processor_span(&self, processor: u64, label: &str) {
        let at = self.stamps.next();
        self.write(Record::ProcessorSpan { at, processor, label: label.to_owned() });
    }

    /// The room bound an SDK object (`room::HostLoop`).
    pub fn record_host_binding(&self, object: &'static str, thread: room::Thread) {
        let at = self.stamps.next();
        self.write(Record::HostBinding { at, object, thread });
    }

    pub(super) fn record_input_ended(&self, slot: Slot) {
        let at = self.stamps.next();
        self.write(Record::InputEnded { at, slot });
    }

    pub(super) fn record_forwarded(&self, slot: Slot, tags: Vec<crate::record::Tag>) {
        let at = self.stamps.next();
        self.write(Record::Forwarded { at, slot, tags });
    }

    /// A synthesis owes its first audio at `at`; `None` once it handed it over or closed.
    pub(super) fn voice_due(&self, tts: u32, at: Option<tokio::time::Instant>) {
        let mut due = self.voice_due.lock().unwrap();
        match at {
            Some(at) => due.insert(tts, at),
            None => due.remove(&tts),
        };
    }

    /// How long until the last first audio a synthesis still owes, if one is due after now.
    pub fn voice_due_in(&self) -> Option<std::time::Duration> {
        let now = tokio::time::Instant::now();
        let due = self.voice_due.lock().unwrap();
        due.values().filter(|at| **at > now).max().map(|at| *at - now)
    }

    pub(super) fn storage_due(&self, at: lk_agents_abi::Ts) {
        let mut due = self.storage_due.lock().unwrap();
        *due = (*due).max(at);
    }

    /// Counts an acquire of the voice lane until the guard drops: answered or cancelled.
    fn storage_pending(self: &Arc<Self>) -> StoragePending {
        self.storage_pending.fetch_add(1, Ordering::SeqCst);
        StoragePending(self.clone())
    }

    /// Whether an acquire of the voice lane is in flight: where a cancelled caller can leave the
    /// database ahead of it.
    pub fn storage_in_flight(&self) -> bool {
        self.storage_pending.load(Ordering::SeqCst) > 0
    }

    /// When the last storage answer still owed is due, if it is after `now`.
    pub fn storage_due_after(&self, now: lk_agents_abi::Ts) -> Option<lk_agents_abi::Ts> {
        let due = *self.storage_due.lock().unwrap();
        (due > now).then_some(due)
    }

    pub(super) fn output_attachment(&self, output: u32, attached: bool) {
        self.write(Record::OutputAttachment { at: self.stamps.next(), output, attached });
    }

    pub(super) fn mark_closed(&self, slot: Slot) {
        let at = self.stamps.next();
        self.write(Record::Closed { at, slot });
    }

    fn record_sink(&self, sink: Sink) {
        let at = self.stamps.next();
        self.write(Record::Sink { at, sink });
    }

    pub(super) fn record_generated(&self, generation: u32, chunk: u16) {
        let at = self.stamps.next();
        self.write(Record::Generated { at, generation, chunk });
    }

    fn record_first_audio_due(&self, slot: Slot, after_ms: u64) {
        let at = self.stamps.next();
        self.write(Record::FirstAudioDue { at, slot, after_ms });
    }

    fn record_delivered(&self, slot: Slot) {
        let at = self.stamps.next();
        self.write(Record::Delivered { at, slot });
    }

    pub(super) fn record_breach(&self, socket: u32, breach: crate::record::Breach) {
        let at = self.stamps.next();
        self.write(Record::Breach { at, socket, breach });
    }

    pub(super) fn record_heard(&self, generation: u32, ms: u64, whole: bool) {
        let at = self.stamps.next();
        self.write(Record::Heard { at, generation, ms, whole });
    }

    pub(super) fn record_held(&self, socket: u32, items: Vec<(u32, u64)>) {
        let at = self.stamps.next();
        self.write(Record::Held { at, socket, items });
    }

    fn record_produced(&self, tts: u32, chunk: u16, text: String, stamps: Vec<(String, f64)>) {
        let at = self.stamps.next();
        self.write(Record::Produced { at, tts, chunk, text, stamps });
    }

    /// One delivery against a control the harness took out of its table. Returns whether the
    /// receiver took it, and the control back unless the delivery spent it.
    pub fn deliver(&self, slot: Slot, control: Control, input: &Input) -> (bool, Option<Control>) {
        // Taken for every delivery, so a generated `call_id` depends only on how many preceded it.
        let serial = self.tool_serial.fetch_add(1, Ordering::SeqCst);
        match control {
            Control::Chat(sender) => {
                let accepted =
                    llm::deliver(&sender, slot.nth, self.response_id(slot.nth), input, serial);
                let done = matches!(input, Input::LlmEnd { .. } | Input::LlmFail { .. });
                (accepted, (!done).then_some(Control::Chat(sender)))
            }
            Control::Tts(synthesis) => {
                let accepted = synthesis.deliver(input);
                let done = matches!(input, Input::TtsFail { .. });
                (accepted, (!done).then_some(Control::Tts(synthesis)))
            }
            Control::Stt(sender, opened, started) => {
                let now = self.stamps.ts();
                let started = match input {
                    Input::SttStart { .. } => Some(now),
                    _ => started,
                };
                let clock = stt::Clock { opened, started, now };
                let accepted = stt::deliver(&sender, input, clock);
                // A failed recogniser is spent, exactly as a failed model request is.
                let done = matches!(input, Input::SttFail { .. });
                (accepted, (!done).then_some(Control::Stt(sender, opened, started)))
            }
            Control::Vad(sender) => (vad::deliver(&sender, input), Some(Control::Vad(sender))),
            Control::Prediction(sender) => (eot::deliver(sender, input), None),
            Control::Interruption(port) => {
                (interruption::deliver(&port, input), Some(Control::Interruption(port)))
            }
            Control::Tool(sender) => {
                let accepted = tools::deliver(sender, input);
                let detached = matches!(
                    input,
                    Input::ToolAnswered { answer: crate::input::ToolAnswer::Detached, .. }
                );
                (accepted, (accepted && detached).then_some(Control::Running))
            }
            Control::Running => (false, Some(Control::Running)),
            Control::Output(playout) => {
                let segments = playout.reset();
                self.record_sink(Sink::Reset { output: slot.nth, segments });
                (true, Some(Control::Output(playout)))
            }
            Control::Video(sender) => {
                (output::feed_video(&sender, slot.nth, input), Some(Control::Video(sender)))
            }
            Control::RoomVideo(camera) => {
                let accepted = output::feed_room_video(&camera, slot.nth, input);
                (accepted, Some(Control::RoomVideo(camera)))
            }
            Control::Realtime(socket) => {
                (realtime::deliver(&socket, input), Some(Control::Realtime(socket)))
            }
            Control::Generation => (false, Some(Control::Generation)),
            Control::Segment(playout) => {
                if let Input::PlayoutStarted { .. } = input {
                    let accepted = playout.started(slot.nth, self.stamps.ts());
                    self.record_sink(Sink::Started { segment: slot.nth, accepted });
                    return (accepted, Some(Control::Segment(playout)));
                }
                if let Input::PlayoutProgressed { ms, .. } = input {
                    let played = std::time::Duration::from_millis(ms.0);
                    let accepted = playout.progressed(slot.nth, played, self.stamps.ts());
                    self.record_sink(Sink::Progressed { segment: slot.nth, accepted });
                    return (accepted, Some(Control::Segment(playout)));
                }
                if let Input::PlayoutFinishLost { .. } = input {
                    let accepted = playout.lose(slot.nth, self.stamps.ts());
                    self.record_sink(Sink::FinishLost { segment: slot.nth, accepted });
                    return (accepted, (!accepted).then_some(Control::Segment(playout)));
                }
                let accepted = playout.report(slot.nth);
                self.record_sink(Sink::Played { segment: slot.nth, accepted });
                // A refused finish (still taking frames) stays reportable after the flush.
                (accepted, (!accepted).then_some(Control::Segment(playout)))
            }
        }
    }
}

/// The audio half of a provider stream: frames are recorded as their tags, never turned into
/// events, so the audio-to-event link stays the case's to make or break.
pub(super) struct Forward {
    pub(super) slot: Slot,
    pub(super) registry: Arc<Registry>,
    /// Told on `end_input`, for a stream whose events end with its input.
    pub(super) ended: Option<Arc<tokio::sync::Notify>>,
}

pub(super) fn tags_of(frame: &AudioFrame) -> Vec<crate::record::Tag> {
    let (pairs, _) = frame.data.as_chunks::<2>();
    pairs.iter().map(|b| crate::record::Tag::decode(i16::from_le_bytes(*b))).collect()
}

impl StreamInput<AudioFrame> for Forward {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.registry.record_forwarded(self.slot, tags_of(&frame));
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {
        self.registry.record_input_ended(self.slot);
        if let Some(ended) = &self.ended {
            ended.notify_one();
        }
    }
}
