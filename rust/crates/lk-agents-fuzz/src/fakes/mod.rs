//! Controlled dependencies. Each fake implements a provider trait, mints an identity for every
//! invocation the product makes, and reads its answers from the control the harness holds.
//!
//! **The fakes hold no shared state.** A fake writes down what it did by sending a [`Record`], and
//! hands the control side of an invocation over the same channel to the harness, which owns it
//! alone. So a delivery is the harness calling a fake while holding nothing, and a fake recording
//! from inside that call is a `send`. No lock on the path means nothing to re-enter, which is what
//! deadlocked the old registry whenever a fake wrote back mid-delivery.
//!
//! A slot is a `(Kind, nth)` pair, not an opaque serial, because that is what a case's `Ref<Chat>`
//! names: the n-th chat the fakes registered is the n-th a case can point at.
//!
//! A delivery to a closed receiver is a recorded, rejected attempt, never a panic.
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use lk_agents_abi::{
    AudioFrame, EotPrediction, LlmChunk, SpeechEvent, Ts, VadEvent, errors::ProviderError,
    utils::stream::StreamSink,
};
use tokio::sync::{mpsc, oneshot};

use crate::{
    input::Input,
    record::{Detail, Kind, Observed, Sink, Stamps},
};

mod eot;
mod interruption;
mod llm;
mod output;
mod realtime;
mod stt;
mod tools;
mod tts;
mod vad;

pub use eot::FakeEot;
pub use interruption::FakeInterruption;
pub use llm::{FakeLlm, request_of};
pub use output::{FakeAudioInput, FakeAudioOutput, FakeRecorder, FakeTextOutput, FakeVideoInput};
pub use realtime::FakeRealtime;
pub use stt::FakeStt;
pub use tools::FakeTools;
pub use tts::FakeTts;
pub use vad::FakeVad;

pub const SAMPLE_RATE: u32 = 24_000;
/// One fixture chunk is 10 ms of mono audio.
pub const SAMPLES_PER_CHUNK: usize = (SAMPLE_RATE / 100) as usize;

/// One thing the fakes opened: which boundary it sits at, and which one of its kind it is.
///
/// The `nth` is the same number a `Ref` in a case carries, so nothing has to translate between
/// the fakes' identities and the recording's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Slot {
    pub kind: Kind,
    pub nth: u32,
}

/// The real input side of one invocation, owned by the harness until consumed.
pub enum Control {
    Chat(mpsc::UnboundedSender<Result<LlmChunk, ProviderError>>),
    Tts(Arc<tts::Synthesis>),
    /// A recognition stream: its event channel, when it opened, and when the utterance it is in
    /// began -- the two instants a transcript's `start_time`/`end_time` are measured from.
    Stt(mpsc::UnboundedSender<Result<SpeechEvent, ProviderError>>, Ts, Option<Ts>),
    Vad(mpsc::UnboundedSender<Result<VadEvent, ProviderError>>),
    /// One pending end-of-turn prediction; answered once, like a tool call.
    Prediction(oneshot::Sender<Result<EotPrediction, ProviderError>>),
    /// The interruption detector, which stays for the session and keeps its own verdict channel.
    Interruption(Arc<interruption::FakeInterruption>),
    Tool(oneshot::Sender<tools::Body>),
    /// A tool that answered `Detached`: still running as far as the driver knows, nothing left
    /// for the case to deliver to it -- its end comes as a `ToolUpdate` command.
    Running,
    /// One audio segment, open at its first frame: the playout side of the output it belongs to,
    /// which takes the case's reports on it -- started, then finished.
    Segment(Arc<output::Playout>),
    /// An audio output as a whole: the playout side, for what the case does to the output rather
    /// than to one segment -- today, a reset.
    Output(Arc<output::Playout>),
    /// A camera: the frames the case feeds it, as ordinals it mints frames from.
    Video(mpsc::UnboundedSender<u64>),
    /// The realtime socket: the server-side events the case injects.
    Realtime(Arc<realtime::Session>),
    /// A realtime turn in progress: nothing to deliver to it, its end comes through the socket.
    Generation,
}

impl Control {
    /// Whether the case still has something to deliver here. A synthesis stream the product has
    /// finished reading is spent even though its control object is still in the map.
    pub(crate) fn open(&self) -> bool {
        match self {
            Control::Tts(synthesis) => synthesis.open(),
            _ => true,
        }
    }
}

/// One thing the fakes did, stamped at the moment it happened and sent to the harness.
///
/// **Draining later preserves the true order**, which is the property this whole design rests on:
/// every record carries the `Observed` stamp taken at the instant of the event, at the same
/// boundary and from the same counter as everything else the run observes. So the harness is free
/// to read these whenever it likes -- after each input, never during one -- and the recording it
/// builds is still the happens-before order. Nothing has to be observed while it happens, and so
/// nothing has to share a lock with the thing observing it.
pub enum Record {
    /// An invocation a fake opened, with the control side for the harness to keep. Outputs carry
    /// no control: nothing is ever delivered to one.
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
    /// One chunk of audio a synthesis stream produced, with the text that caused it.
    Produced {
        at: Observed,
        tts: u32,
        chunk: u16,
        text: String,
    },
    /// Frames the driver pushed into a provider stream, as the tags they carried.
    Forwarded {
        at: Observed,
        slot: Slot,
        tags: Vec<crate::record::Tag>,
    },
    Sink {
        at: Observed,
        sink: Sink,
    },
    /// Which output a flushed segment belongs to. Not a stamped event: it is a fact about the
    /// segment the harness keeps for as long as the run lasts.
    Segment {
        segment: u32,
        output: u32,
    },
}

/// The boundary an input is a delivery to, or `None` when it is not a delivery at all. The one
/// place the mapping lives, so the harness can refuse a delivery aimed at the wrong kind
/// without every caller restating it.
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
        | Input::SttFail { .. } => Kind::Stt,
        Input::VadInference { .. } | Input::VadStart { .. } | Input::VadEnd { .. } => Kind::Vad,
        Input::EotAnswer { .. } | Input::EotFail { .. } => Kind::Prediction,
        Input::OverlapVerdict { .. } | Input::OverlapFail { .. } => Kind::Interruption,
        Input::ToolAnswered { .. } => Kind::Tool,
        Input::PlayoutStarted { .. }
        | Input::PlayoutFinished { .. }
        | Input::PlayoutProgressed { .. } => Kind::Segment,
        Input::SinkReset { .. } => Kind::Output,
        Input::VideoFrames { .. } => Kind::Video,
        Input::RealtimeEvent { .. } => Kind::Realtime,
        _ => return None,
    })
}

/// The kinds the fakes mint identities for. Speeches, holds and the rest are the driver's, and
/// the harness opens those from the journal.
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

/// What every fake holds: the stamp counter, the identity counters, and the way back to the
/// harness. Cheap to clone into each fake, and nothing here is ever mutated behind a lock.
pub struct Registry {
    stamps: Stamps,
    records: mpsc::UnboundedSender<Record>,
    /// One counter per kind, because that is what a `Ref` counts. Atomic rather than locked so a
    /// fake can mint its own identity from inside a product call without blocking anything.
    next: BTreeMap<Kind, AtomicU32>,
    /// Distinct `call_id`s for generated tool calls.
    tool_serial: AtomicU32,
    /// The voice provider is down: every synthesis fails at the open until it comes back.
    voice_down: std::sync::atomic::AtomicBool,
    /// The session's command queue, once the session exists: what a tool body speaks to the
    /// driver through from its own origin, as `RunContext` does.
    api: std::sync::OnceLock<
        lk_agents_voice::agent_session::Tx<lk_agents_voice::agent_session::Command>,
    >,
    /// The session's open calls, the run's and the tool bodies' alike; the run settles them.
    calls: lk_agents_voice::agent_session::Calls,
}

impl Registry {
    pub fn new(stamps: Stamps) -> (Arc<Self>, mpsc::UnboundedReceiver<Record>) {
        let (records, receiver) = mpsc::unbounded_channel();
        let next = MINTED.into_iter().map(|kind| (kind, AtomicU32::new(0))).collect();
        let registry = Registry {
            stamps,
            records,
            next,
            tool_serial: AtomicU32::new(0),
            voice_down: std::sync::atomic::AtomicBool::new(false),
            api: std::sync::OnceLock::new(),
            calls: Default::default(),
        };
        (Arc::new(registry), receiver)
    }

    /// The voice provider's health, as the case sets it.
    pub fn set_voice_down(&self, down: bool) {
        self.voice_down.store(down, Ordering::SeqCst);
    }

    pub(super) fn voice_down(&self) -> bool {
        self.voice_down.load(Ordering::SeqCst)
    }

    /// The session started: its queue is what running tool bodies command the driver through.
    pub fn attach_api(
        &self,
        api: lk_agents_voice::agent_session::Tx<lk_agents_voice::agent_session::Command>,
    ) {
        let _ = self.api.set(api);
    }

    /// The session's instant, for what a fake stamps on the driver's clock.
    pub(super) fn now(&self) -> Ts {
        self.stamps.ts()
    }

    pub(super) fn api(
        &self,
    ) -> Option<&lk_agents_voice::agent_session::Tx<lk_agents_voice::agent_session::Command>> {
        self.api.get()
    }

    pub(crate) fn calls(&self) -> &lk_agents_voice::agent_session::Calls {
        &self.calls
    }

    /// The next slot of a kind.
    fn mint(&self, kind: Kind) -> Slot {
        let counter = self.next.get(&kind).expect("a kind the fakes mint");
        Slot { kind, nth: counter.fetch_add(1, Ordering::SeqCst) }
    }

    /// Write one record down. A run that has ended has dropped the receiver, and a fake still
    /// running on a detached task may record after that; it is not an error, there is simply no
    /// one left to tell.
    fn write(&self, record: Record) {
        let _ = self.records.send(record);
    }

    /// A new output instance the case is attaching. Recorded like an invocation so the harness has
    /// one place to read everything the fakes opened.
    /// An audio output, with its playout side as the control the case reaches the output by.
    pub(super) fn new_audio_output(&self, playout: Arc<output::Playout>) -> u32 {
        self.register(Kind::Output, Detail::Output { audio: true }, Control::Output(playout)).nth
    }

    pub fn new_output(&self, audio: bool) -> u32 {
        let at = self.stamps.next();
        let slot = self.mint(Kind::Output);
        self.write(Record::Opened { at, slot, detail: Detail::Output { audio }, control: None });
        slot.nth
    }

    /// The case attached a microphone. Not a fake's invocation, but minted here so its `nth` is
    /// what a `Ref<Mic>` names, like every other resource.
    pub fn new_mic(&self) -> u32 {
        let at = self.stamps.next();
        let slot = self.mint(Kind::Mic);
        self.write(Record::Opened { at, slot, detail: Detail::None, control: None });
        slot.nth
    }

    /// The end-of-turn predictor leg, one per session, minted so the frames the driver pushes into
    /// it can be recorded against something a rule can name.
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

    /// Remember which output a flushed segment belongs to. Separate from `register` because only
    /// the audio output knows the pairing, and it learns the segment's identity from the call.
    pub(super) fn attach_segment(&self, segment: u32, output: u32) {
        self.write(Record::Segment { segment, output });
    }

    pub(super) fn record_forwarded(&self, slot: Slot, tags: Vec<crate::record::Tag>) {
        let at = self.stamps.next();
        self.write(Record::Forwarded { at, slot, tags });
    }

    fn mark_closed(&self, slot: Slot) {
        let at = self.stamps.next();
        self.write(Record::Closed { at, slot });
    }

    fn record_sink(&self, sink: Sink) {
        let at = self.stamps.next();
        self.write(Record::Sink { at, sink });
    }

    fn record_produced(&self, tts: u32, chunk: u16, text: String) {
        let at = self.stamps.next();
        self.write(Record::Produced { at, tts, chunk, text });
    }

    /// Run one delivery through the real channel, against a control the harness took out of its
    /// own table and hands in here.
    ///
    /// Nothing is locked on this path, so nothing can be re-entered: the control belongs to the
    /// caller alone, and a fake that records from inside this call -- the synthesis logging the
    /// audio a release produced, a tracker notifying an attached recorder -- only sends. The
    /// answer is whether the receiver took it; `End`, `Fail`, tool answers and playout reports
    /// spend the control either way. Each fake owns the conversion from an `Input` to its wire
    /// type.
    pub fn deliver(&self, slot: Slot, control: Control, input: &Input) -> (bool, Option<Control>) {
        // Taken for every delivery, not only the ones that mint a call id, so a generated
        // `call_id` stays a function of how many deliveries preceded it.
        let serial = self.tool_serial.fetch_add(1, Ordering::SeqCst);
        match control {
            Control::Chat(sender) => {
                let accepted = llm::deliver(&sender, slot.nth, input, serial);
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
                let accepted = playout.report(slot.nth);
                self.record_sink(Sink::Played { segment: slot.nth, accepted });
                // A finish the output refused -- the segment is still taking frames -- leaves it
                // outstanding; the case can report it once the flush has come.
                (accepted, (!accepted).then_some(Control::Segment(playout)))
            }
        }
    }
}

/// The audio half of a provider stream: the driver pushes frames here, and each one is written
/// down as the tags it carried. The fake never derives events from these -- that link is the
/// case's to make or break -- but recording them is what lets a rule check whether it did.
pub(super) struct Forward {
    pub(super) slot: Slot,
    pub(super) registry: Arc<Registry>,
}

/// The origin tags a frame carries, one per sample.
pub(super) fn tags_of(frame: &AudioFrame) -> Vec<crate::record::Tag> {
    let (pairs, _) = frame.data.as_chunks::<2>();
    pairs.iter().map(|b| crate::record::Tag::decode(i16::from_le_bytes(*b))).collect()
}

impl StreamSink<AudioFrame> for Forward {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.registry.record_forwarded(self.slot, tags_of(&frame));
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}
