//! What a run leaves behind, as plain stamped data, and what it still has open.
//!
//! The harness appends; nothing here is reconstructed. Rules never read this file: they read the
//! four views in [`views`], which are the only account of a run a rule is given.
//!
//! The product's own journal is kept verbatim. The rest is the other side of each boundary it
//! talks across, which the product cannot record: what the case sent and whether it was refused,
//! what the fakes produced, what left the session, what the host received. A rule comparing the
//! driver's account against what arrived needs both sides.
//!
//! One log holds every resource the run opened, and [`Live`] is the part still open, because
//! "what can an input point at" and "what is open" are one question asked twice.
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use lk_agents_abi::Ts;
use lk_agents_voice::{
    agent_activity::pause::InterruptRefusal,
    agent_activity::scheduling::{SpeechId, SpeechKind},
    agent_session::{HandoffRefused, HoldId, HookCall, HookId, IdleRefused, Timebase},
    events::{SessionEvent, ToolLifecycle},
    filler_scheduler::FillerId,
    room_io::RoomIoEvent,
    trace::{Step, step_names},
};

use crate::input::Input;

mod views;
pub use views::*;

// ==============================================================================================
// When
// ==============================================================================================

/// When something happened: an occurrence stamp and the session clock in milliseconds.
///
/// Order by `seq`. It comes from one counter taken synchronously at each boundary, so on the
/// current-thread runtime it is a real happens-before order across host events, fakes, the sink
/// and the driver's own journal. `at_ms` is for deadlines and for reading.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Observed {
    pub seq: u64,
    pub at_ms: u64,
}

impl Observed {
    /// A stamp for hand-written recordings.
    pub const fn at(seq: u64, at_ms: u64) -> Self {
        Observed { seq, at_ms }
    }

    /// Later than anything a run records: the end of the run, for open-ended stretches.
    pub const END: Observed = Observed { seq: u64::MAX, at_ms: u64::MAX };

    /// How many milliseconds of the session's clock passed from `earlier` to this instant.
    pub fn ms_since(&self, earlier: Observed) -> u64 {
        self.at_ms.saturating_sub(earlier.at_ms)
    }
}

/// The shared counter and the session clock. Cheap to clone; every fake holds one.
#[derive(Clone)]
pub struct Stamps {
    counter: Arc<AtomicU64>,
    time: Timebase,
}

impl Stamps {
    pub fn new(time: Timebase) -> Self {
        Stamps { counter: Arc::default(), time }
    }

    /// Take the next stamp, at the boundary, before anything else runs.
    pub fn next(&self) -> Observed {
        Observed { seq: self.counter.fetch_add(1, Ordering::SeqCst), at_ms: self.now_ms() }
    }

    /// The current instant without consuming a stamp.
    pub fn now(&self) -> Observed {
        Observed { seq: self.counter.load(Ordering::SeqCst), at_ms: self.now_ms() }
    }

    fn now_ms(&self) -> u64 {
        self.time.now().saturating_duration_since(Ts::ZERO).as_millis() as u64
    }

    /// The current instant on the session's own clock, for events the fakes stamp for the driver.
    pub fn ts(&self) -> Ts {
        self.time.now()
    }
}

// ==============================================================================================
// What a run opens
// ==============================================================================================

/// A kind of thing a run can open and point at.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Kind {
    /// A speech, whether the case asked for it or the driver started it.
    Speech,
    /// A floor hold.
    Hold,
    /// A filler the case started on a running tool.
    Filler,
    /// A hook call the driver is waiting on.
    Hook,
    /// One model request.
    Chat,
    /// One synthesis stream.
    Tts,
    /// One recognition stream.
    Stt,
    /// One utterance on a recognition stream, from start of speech to final.
    Utterance,
    /// One tool body the runner was handed.
    Tool,
    /// One flushed audio segment awaiting a playout report.
    Segment,
    /// A text or audio stream a `say` opened for the case to feed.
    Stream,
    /// An output instance the case attached.
    Output,
    /// An answer the driver owes a caller.
    Answer,
    /// The microphone the case speaks into.
    Mic,
    /// One voice-activity detection stream.
    Vad,
    /// The end-of-turn predictor: the third observer of the person's audio.
    Eot,
    /// One prediction the driver asked the predictor for and is waiting on.
    Prediction,
    /// The adaptive interruption detector: the fourth observer, judging overlaps.
    Interruption,
    /// A camera the case attached and feeds.
    Video,
    /// The realtime model's socket.
    Realtime,
    /// A turn the realtime server is in the middle of: open until the case says it finished.
    Generation,
}

/// The production identity behind a resource, once the run knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Id {
    Speech(SpeechId),
    Hold(HoldId),
    Filler(FillerId),
    Hook(HookId),
    /// A fake's invocation, or a stream, output or answer the harness minted.
    Slot(u32),
    /// Nothing yet: the resource exists but its identity has not arrived.
    Pending,
}

/// One resource, as the run opened it.
#[derive(Clone, Debug, PartialEq)]
pub struct Opened {
    pub at: Observed,
    pub kind: Kind,
    /// Which one of its kind, counting from zero: what a `Ref` in a case names.
    pub nth: u32,
    pub id: Id,
    pub detail: Detail,
}

/// What is worth knowing about a resource beyond its identity.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Detail {
    #[default]
    None,
    /// The item ids the request was planned with, the speech the driver made it for, which the
    /// driver's own metrics supply, and the tools it offered the model -- what tells a
    /// classifier's request from a reply's.
    Chat {
        ctx_ids: Vec<String>,
        speech: Option<u64>,
        tools: Vec<String>,
    },
    Tool {
        call_id: String,
        name: String,
        speech: u64,
    },
    Speech {
        kind: SpeechKind,
        user_initiated: bool,
    },
    Hook {
        kind: lk_agents_voice::agent_session::HookKind,
    },
    /// Which of the two output kinds this instance is.
    Output {
        audio: bool,
    },
    /// Which stream a `say` opened, and whether the case feeds it audio rather than text.
    Stream {
        audio: bool,
    },
}

/// What the run still holds: the resources an input may point at right now.
///
/// The counterpart of the `opened` log, which is everything the run ever opened. This is only
/// what is still live, which is the one question the sampler and replay both ask.
#[derive(Debug, Default)]
pub struct Live {
    open: BTreeMap<Kind, BTreeSet<u32>>,
    opened: BTreeMap<Kind, u32>,
}

impl Live {
    /// The resources of this kind a case may still point at.
    pub fn of(&self, kind: Kind) -> Vec<u32> {
        self.open.get(&kind).map(|open| open.iter().copied().collect()).unwrap_or_default()
    }

    /// How many of this kind the run has opened, live or not: the next one's index.
    pub fn opened(&self, kind: Kind) -> u32 {
        self.opened.get(&kind).copied().unwrap_or(0)
    }

    /// Whether this reference still resolves. Replay asks exactly this before applying a recorded
    /// input, which is the same question the sampler asked when it drew one.
    pub fn has(&self, kind: Kind, nth: u32) -> bool {
        self.open.get(&kind).is_some_and(|open| open.contains(&nth))
    }

    fn open(&mut self, kind: Kind) -> u32 {
        let next = self.opened.entry(kind).or_default();
        let nth = *next;
        *next += 1;
        self.open.entry(kind).or_default().insert(nth);
        nth
    }

    fn close(&mut self, kind: Kind, nth: u32) {
        if let Some(open) = self.open.get_mut(&kind) {
            open.remove(&nth);
        }
    }
}

// ==============================================================================================
// What a run records
// ==============================================================================================

/// One input the case sent, and whether the thing it was sent to took it. A delivery to a closed
/// receiver is a recorded, rejected attempt, never a panic and never a reason to reach into the
/// driver.
#[derive(Clone, Debug, PartialEq)]
pub struct Applied {
    pub at: Observed,
    pub input: Input,
    pub accepted: bool,
}

/// What the host received, as the production types themselves.
#[derive(Clone, Debug, PartialEq)]
pub enum Host {
    Session(SessionEvent),
    /// A hook call, projected because `HookCall` is not `Clone`; its id is what an answer names.
    Hook {
        id: HookId,
        kind: lk_agents_voice::agent_session::HookKind,
        speech: Option<u64>,
    },
    Room(RoomIoEvent),
}

/// What left the session, and what the recorder was told about it.
#[derive(Clone, Debug, PartialEq)]
pub enum Sink {
    /// Audio on an output, decoded to the sources it came from.
    Audio {
        output: u32,
        tags: Vec<Tag>,
    },
    /// A segment flushed on an output, awaiting its playout report.
    Flush {
        output: u32,
        segment: u32,
    },
    /// The product cleared an output's buffer; these pending segments were reported interrupted.
    Clear {
        output: u32,
        segments: Vec<u32>,
    },
    /// A playout report reached the tracker, and whether the tracker accepted it.
    Played {
        segment: u32,
        accepted: bool,
    },
    /// The case reported a segment's first audio leaving the speaker; refused for a segment gone
    /// or already started.
    Started {
        segment: u32,
        accepted: bool,
    },
    /// The product paused, or resumed, a pausable output.
    Paused {
        output: u32,
    },
    Resumed {
        output: u32,
    },
    /// The case reported a stretch of a segment as played; refused before its start.
    Progressed {
        segment: u32,
        accepted: bool,
    },
    /// The case reset an output's playback count; the segments it still owed, now cut.
    Reset {
        output: u32,
        segments: Vec<u32>,
    },
    Text {
        output: u32,
        text: String,
    },
    TextFlush {
        output: u32,
    },
    Recorder(RecorderCall),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecorderCall {
    InputFrame,
    CaptureFrame,
    PlaybackStarted,
    PlaybackProgressed,
    PlaybackFinished,
    Flush,
}

/// What one audio sample says about where it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tag {
    /// The n-th synthesis stream, and which chunk of it.
    Synthesized { tts: u32, chunk: u16 },
    /// The n-th stream a `say` opened, and which chunk the case fed.
    Fed { stream: u32, chunk: u16 },
    /// The n-th microphone, and which frame the case spoke into it.
    Mic { mic: u32, chunk: u16 },
    /// A sample nobody in this run produced.
    Unknown(i16),
}

impl Tag {
    pub const MAX_CHUNK: u16 = 63;
    /// Set on user-side samples that came from the microphone rather than a fed stream.
    const MIC: i16 = 0x4000;

    pub fn synthesized(tts: u32, chunk: u16) -> i16 {
        (((tts as i32 + 1) << 6) | chunk.min(Self::MAX_CHUNK) as i32) as i16
    }

    pub fn fed(stream: u32, chunk: u16) -> i16 {
        -Self::synthesized(stream, chunk)
    }

    pub fn mic(mic: u32, chunk: u16) -> i16 {
        -(Self::synthesized(mic, chunk) | Self::MIC)
    }

    pub fn decode(sample: i16) -> Self {
        if sample == 0 || sample == i16::MIN {
            return Tag::Unknown(sample);
        }
        let user_side = sample < 0;
        let magnitude = sample.unsigned_abs() as i16;
        let from_mic = user_side && magnitude & Self::MIC != 0;
        let value = (magnitude & !Self::MIC) as u16;
        let (id, chunk) = ((value >> 6) as u32 - 1, value & 0x3f);
        match (user_side, from_mic) {
            (false, _) => Tag::Synthesized { tts: id, chunk },
            (true, false) => Tag::Fed { stream: id, chunk },
            (true, true) => Tag::Mic { mic: id, chunk },
        }
    }
}

/// An answer the driver gave a caller.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Speech(u64),
    InterruptOk,
    InterruptRefused(InterruptRefusal),
    IdleOk,
    IdleRefused(IdleRefused),
    Hold(u64),
    Committed(String),
    CommitSuperseded,
    HandoffOk,
    HandoffRefused(HandoffRefused),
    DrainOk,
    DrainRefused(lk_agents_voice::agent_session::DrainRefused),
    /// What a driver-run DTMF collection answered its `Start` with.
    Dtmf(lk_agents_voice::workflows::builtins::dtmf::DtmfOutcome),
    /// The channel closed without an answer.
    Lost,
}

/// A speech handle as the public API reports it, snapshotted after every input so a rule and its
/// counterexample work on the same plain data.
#[derive(Clone, Debug, PartialEq)]
pub struct Handle {
    pub kind: Option<SpeechKind>,
    pub done: bool,
    /// The first stamp at which `done` was seen.
    pub done_since: Option<Observed>,
    pub interrupted: bool,
    pub scheduled: bool,
    pub allow_interruptions: bool,
    pub error: Option<String>,
}

impl Handle {
    /// Nothing known yet.
    pub fn unknown() -> Self {
        Handle {
            kind: None,
            done: false,
            done_since: None,
            interrupted: false,
            scheduled: false,
            allow_interruptions: true,
            error: None,
        }
    }
    /// Scheduled, still running.
    pub fn live() -> Self {
        Handle { scheduled: true, ..Self::unknown() }
    }
    /// Ran to the end.
    pub fn finished() -> Self {
        Handle { done: true, ..Self::live() }
    }
    pub fn interrupted(self) -> Self {
        Handle { interrupted: true, ..self }
    }
    pub fn failed(self, error: &str) -> Self {
        Handle { error: Some(error.into()), ..self }
    }
}

/// Where the runner has got to. A progress deadline starts at `Settling`, not before: releasing
/// advances virtual time to let the driver's own timers fire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    #[default]
    Exploring,
    Releasing,
    Settling,
    Closing,
    Closed,
}

#[derive(Debug, Default)]
pub struct Recording {
    now: Observed,
    phase: Phase,
    released: Option<Observed>,
    settling: Option<Observed>,
    applied: Vec<Applied>,
    opened: Vec<Opened>,
    closed: Vec<(Observed, Kind, u32)>,
    results: Vec<(Observed, u32, Answer)>,
    produced: Vec<(Observed, u32, u16, String)>,
    /// Frames the driver forwarded into a provider stream, tag-decoded: the microphone-to-provider
    /// link, made visible so a rule can hold injected VAD and STT events against the audio.
    forwarded: Vec<(Observed, Kind, u32, Vec<Tag>)>,
    sink: Vec<(Observed, Sink)>,
    host: Vec<(Observed, Host)>,
    journal: Vec<(Observed, Step)>,
    handles: BTreeMap<u64, Handle>,
    driver_finished: bool,
    driver_panic: Option<String>,
    live: Live,
    index: Index,
}

/// Lookups by identity, maintained as facts arrive.
///
/// Most rules ask "what happened to this speech" or "to this call" once per subject, and the
/// views answer from these positions rather than by scanning the whole run per subject. The logs
/// above stay the facts.
#[derive(Debug, Default)]
struct Index {
    journal_by_speech: BTreeMap<u64, Vec<usize>>,
    host_by_speech: BTreeMap<u64, Vec<usize>>,
    host_by_tool: BTreeMap<String, Vec<usize>>,
    produced_chunks: BTreeSet<(u32, u16)>,
    /// When each listening stream heard each microphone frame, ascending: `(stream kind, stream,
    /// microphone, chunk)`. A run forwards tens of thousands of frames, and the rules ask about
    /// each one.
    heard: HashMap<(Kind, u32, u32, u16), Vec<Observed>>,
}

// ==============================================================================================
// Reading a recording
// ==============================================================================================

impl Recording {
    /// The last stamp taken, so a rule can measure how long something has been pending.
    pub fn now(&self) -> Observed {
        self.now
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// When the runner stopped adding work, and when it ran out of things to release.
    pub fn released(&self) -> Option<Observed> {
        self.released
    }

    /// When the release phase found nothing more to answer, twice in a row. The runner's own
    /// mark: the rules read the host's close instead, which the runner asks for only after this.
    pub fn settling(&self) -> Option<Observed> {
        self.settling
    }

    /// What the run still holds: the resources an input may point at right now.
    pub fn live(&self) -> &Live {
        &self.live
    }

    pub fn driver_finished(&self) -> bool {
        self.driver_finished
    }

    pub fn driver_panic(&self) -> Option<&str> {
        self.driver_panic.as_deref()
    }

    /// Every input the case sent, taken or refused, in order.
    pub fn applied(&self) -> &[Applied] {
        &self.applied
    }

    /// The driver's own journal, verbatim.
    pub fn journal(&self) -> &[(Observed, Step)] {
        &self.journal
    }

    /// What the host received, as the production types themselves.
    pub fn host(&self) -> &[(Observed, Host)] {
        &self.host
    }

    /// What left the session.
    pub fn sink(&self) -> &[(Observed, Sink)] {
        &self.sink
    }

    /// The answers the driver gave, by the call each was owed to.
    pub fn results(&self) -> &[(Observed, u32, Answer)] {
        &self.results
    }

    /// Each chunk of audio a synthesis stream produced, with the text that caused it.
    pub fn produced(&self) -> &[(Observed, u32, u16, String)] {
        &self.produced
    }

    /// What reached each provider stream from the microphone, in order.
    pub fn forwarded(&self) -> &[(Observed, Kind, u32, Vec<Tag>)] {
        &self.forwarded
    }

    /// Every resource the run opened, live or not, in the order it opened them.
    pub fn opened(&self) -> &[Opened] {
        &self.opened
    }

    pub fn closed(&self) -> &[(Observed, Kind, u32)] {
        &self.closed
    }

    /// Every speech handle the run has seen, as the public API reports it.
    pub fn handles(&self) -> &BTreeMap<u64, Handle> {
        &self.handles
    }
}

// ==============================================================================================
// Writing a recording
// ==============================================================================================

/// Only the harness and the authored recordings write here. Each mutator maintains whatever
/// lookup it feeds, so there is no separate indexing step to forget.
impl Recording {
    /// Open a resource, returning its index. The live set and the log move together.
    pub fn open(&mut self, at: Observed, kind: Kind, id: Id, detail: Detail) -> u32 {
        let nth = self.live.open(kind);
        self.opened.push(Opened { at, kind, nth, id, detail });
        nth
    }

    /// Close a resource, so nothing new can point at it. What it did stays in the log.
    pub fn close(&mut self, at: Observed, kind: Kind, nth: u32) {
        self.live.close(kind, nth);
        self.closed.push((at, kind, nth));
    }

    /// The case sent this input, and the thing it was sent to took it or refused it.
    pub fn record_applied(&mut self, at: Observed, input: Input, accepted: bool) {
        self.applied.push(Applied { at, input, accepted });
    }

    pub fn record_journal(&mut self, at: Observed, step: Step) {
        if let Some(speech) = step.speech() {
            let position = self.journal.len();
            self.index.journal_by_speech.entry(speech.0).or_default().push(position);
        }
        self.journal.push((at, step));
    }

    pub fn record_host(&mut self, at: Observed, event: Host) {
        let position = self.host.len();
        if let Host::Session(event) = &event {
            match event {
                SessionEvent::SpeechCreated { speech, .. } => {
                    self.index.host_by_speech.entry(speech.id().0).or_default().push(position)
                }
                SessionEvent::SpeechDone { speech, .. } => {
                    self.index.host_by_speech.entry(speech.0).or_default().push(position)
                }
                SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started { call }) => {
                    self.index.host_by_tool.entry(call.call_id.clone()).or_default().push(position)
                }
                SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended { call_id, .. }) => {
                    self.index.host_by_tool.entry(call_id.clone()).or_default().push(position)
                }
                _ => {}
            }
        }
        self.host.push((at, event));
    }

    pub fn record_sink(&mut self, at: Observed, sink: Sink) {
        self.sink.push((at, sink));
    }

    /// The driver answered the n-th call it owed.
    pub fn record_result(&mut self, at: Observed, answer: u32, result: Answer) {
        self.results.push((at, answer, result));
    }

    /// One chunk of audio the n-th synthesis stream produced.
    pub fn record_forwarded(&mut self, at: Observed, kind: Kind, nth: u32, tags: Vec<Tag>) {
        for tag in &tags {
            if let Tag::Mic { mic, chunk } = *tag {
                self.index.heard.entry((kind, nth, mic, chunk)).or_default().push(at);
            }
        }
        self.forwarded.push((at, kind, nth, tags));
    }

    pub fn record_produced(&mut self, at: Observed, tts: u32, chunk: u16, text: String) {
        self.index.produced_chunks.insert((tts, chunk));
        self.produced.push((at, tts, chunk, text));
    }

    /// Snapshot one speech handle. The earliest stamp at which it was seen done is kept, because
    /// a rule measures how long a handle has been done from it -- and so is how it ended. A speech
    /// ends once; what the handle says afterwards, when the session's close marks everything it
    /// releases as interrupted, is the close's doing, not the speech's, and a rule asking whether
    /// a speech completed normally is asking about the moment it completed.
    pub fn record_handle(&mut self, speech: u64, handle: Handle) {
        let entry = self.handles.entry(speech).or_insert_with(Handle::unknown);
        *entry = if entry.done {
            Handle {
                done_since: entry.done_since,
                interrupted: entry.interrupted,
                error: entry.error.take(),
                ..handle
            }
        } else {
            Handle { done_since: entry.done_since.or(handle.done_since), ..handle }
        };
    }

    /// Which speech a chat request was planned for, once the driver has said so.
    pub fn attribute_chat(&mut self, chat: u32, speech: u64) {
        for opened in &mut self.opened {
            if opened.kind == Kind::Chat
                && opened.nth == chat
                && let Detail::Chat { speech: slot, .. } = &mut opened.detail
            {
                *slot = Some(speech);
            }
        }
    }

    /// Enter a phase. A run only ever moves forward, so this never goes back.
    pub fn enter_phase(&mut self, phase: Phase) {
        self.phase = self.phase.max(phase);
    }

    /// The runner stopped adding work here. Only the first such mark counts.
    pub fn mark_released(&mut self, at: Observed) {
        self.released.get_or_insert(at);
    }

    /// Nothing was left to release here; from now on outcomes are only awaited.
    pub fn mark_settling(&mut self, at: Observed) {
        self.released.get_or_insert(at);
        self.settling.get_or_insert(at);
    }

    pub fn set_driver_finished(&mut self, finished: bool) {
        self.driver_finished = finished;
    }

    pub fn set_driver_panic(&mut self, panic: String) {
        self.driver_panic = Some(panic);
    }

    pub fn set_now(&mut self, now: Observed) {
        self.now = now;
    }
}

// ==============================================================================================
// Finding things
// ==============================================================================================
// What the harness asks of a recording while a run is going: which resource a case's `Ref`
// names, and what the driver has reached. Rules read none of this; they read the views.

/// One resource the run opened, with the input behind it.
#[derive(Clone, Copy)]
pub struct Resource<'a> {
    record: &'a Recording,
    opened: &'a Opened,
}

impl<'a> Resource<'a> {
    pub fn kind(&self) -> Kind {
        self.opened.kind
    }

    /// Which one of its kind, counting from zero: what a `Ref` in a case names.
    pub fn nth(&self) -> u32 {
        self.opened.nth
    }

    pub fn id(&self) -> Id {
        self.opened.id
    }

    pub fn detail(&self) -> &'a Detail {
        &self.opened.detail
    }

    pub fn opened_at(&self) -> Observed {
        self.opened.at
    }

    pub fn closed_at(&self) -> Option<Observed> {
        self.record
            .closed
            .iter()
            .find(|(_, kind, nth)| *kind == self.opened.kind && *nth == self.opened.nth)
            .map(|(at, ..)| *at)
    }

    /// Whether an input may still point at it.
    pub fn is_live(&self) -> bool {
        self.record.live.has(self.opened.kind, self.opened.nth)
    }

    /// The input that opened this, if the case opened it at all.
    ///
    /// Exact, not nearest: an input and everything it opens share one stamp. A resource the
    /// driver opened itself matches nothing and gives `None`.
    pub fn opened_by(&self) -> Option<&'a Applied> {
        self.record.applied.iter().find(|a| a.at == self.opened.at)
    }
}

impl Recording {
    /// The inputs the case applied that `pick` matches, with their stamps. The release phase
    /// asks this to know what the case has already released.
    pub fn sent<'a, T>(
        &'a self,
        pick: impl Fn(&'a Input) -> Option<T> + 'a,
    ) -> impl Iterator<Item = (Observed, T, bool)> + 'a {
        self.applied.iter().filter_map(move |a| pick(&a.input).map(|t| (a.at, t, a.accepted)))
    }

    /// Steps anywhere in the journal that `pick` matches. The release phase asks this for the
    /// driver's armed deadlines and the navigator's triggers.
    pub fn journal_where<'a, T>(
        &'a self,
        pick: impl Fn(&'a Step) -> Option<T> + 'a,
    ) -> impl Iterator<Item = (Observed, T)> + 'a {
        self.journal.iter().filter_map(move |(at, step)| pick(step).map(|t| (*at, t)))
    }

    /// The n-th resource of a kind, named the way a case names one.
    pub fn resource(&self, kind: Kind, nth: u32) -> Option<Resource<'_>> {
        self.opened
            .iter()
            .find(|o| o.kind == kind && o.nth == nth)
            .map(|opened| Resource { record: self, opened })
    }

    /// Every resource of a kind the run opened, live or not.
    pub fn resources(&self, kind: Kind) -> impl Iterator<Item = Resource<'_>> {
        self.opened
            .iter()
            .filter(move |o| o.kind == kind)
            .map(move |opened| Resource { record: self, opened })
    }

    /// Journal step variants this run produced, as `Family::Variant`.
    pub fn journal_names(&self) -> BTreeSet<String> {
        // The folded view is part of the vocabulary: `ByAudioActivityRepeated` is only ever produced by
        // folding a run of identical barge steps, so a journal that folds to it reached it. The
        // rules keep reading the raw steps; only the reach count reads the fold.
        let steps: Vec<Step> = self.journal.iter().map(|(_, step)| step.clone()).collect();
        lk_agents_voice::trace::folded(&steps).iter().map(step_name).collect()
    }

    /// Journal step variants the driver can produce that this run did not.
    ///
    /// The search steers by this, so it must not overstate itself: a name the vocabulary does not
    /// know means the two spellings have drifted, and it panics rather than quietly shrinking.
    pub fn journal_gaps(&self) -> Vec<String> {
        let known = step_names();
        let reached = self.journal_names();
        if let Some(stray) = reached.iter().find(|name| !known.contains(name)) {
            panic!(
                "step_name produced `{stray}`, which trace::step_names does not list: the two \
                 spellings of a journal variant have drifted apart"
            );
        }
        known.into_iter().filter(|name| !reached.contains(name)).collect()
    }
}

/// A journal step's variant path, spelled `Family::Variant`.
///
/// Read off `Debug`, because a `Step` is thirteen nested enums with no other way to ask which
/// variant it is. [`Recording::journal_gaps`] checks the spelling matches [`step_names`].
pub fn step_name(step: &Step) -> String {
    let text = format!("{step:?}");
    let cut = |s: &str| s.find(['(', '{', ' ', ')']).unwrap_or(s.len());
    let family = &text[..cut(&text)];
    let rest = text[family.len()..].trim_start_matches('(');
    match &rest[..cut(rest)] {
        "" => family.to_string(),
        variant => format!("{family}::{variant}"),
    }
}

/// The hook a call names, for the projection the recording keeps.
pub fn hook_of(call: &HookCall) -> Host {
    Host::Hook {
        id: call.id(),
        kind: call.kind(),
        speech: match call {
            HookCall::DescribeAgent { speech, .. } => Some(speech.0),
            _ => None,
        },
    }
}
