//! A real session with controlled dependencies, and one function that applies an [`Input`] to it.
//!
//! The only place that touches the product. It builds the driver with the fakes attached, turns
//! each input into the production command, feed or delivery it mirrors, and writes down what
//! happened. It decides nothing: rules read [`Recording`] and nothing else.
//!
//! **Each resource is opened by whoever mints it.** Speeches, holds, fillers and hook calls come
//! from the driver's own journal, the one account of them that cannot drift from what it did. The
//! rest have no journal to read -- a chat exists because a fake was called, an answer because a
//! command carried a channel -- and are opened here, from the side that knows.
//!
//! **Applying an input does not yield.** Only [`Input::Advance`] and the awaits reach the
//! executor, so a case can build a burst the driver must untangle in one cycle.
//! [`Recording::driver_ran_since`] is written against that, and breaking it would quietly weaken
//! every rule that allows the driver one input of grace.//!
//! The module is split by what a run does, in order: [`start`] builds the session, [`apply`]
//! turns one input into what the product sees, [`collect`] writes down what became true, and
//! [`finish`] releases what a run owes and reports what it found. This file holds what they
//! share: the setup, the simulation, and the run itself.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use lk_agents_abi::{
    AudioFrame, BoxFuture, ChatItem, ChatMessage, HostRef, ImageContent, ImageSource, Maybe,
    MediaRef, Realtime, Spawner, Ts,
    errors::RefRegistry,
    metrics::base::AgentMetrics,
    tokenize::language::ThresholdOverride,
    voice::room_io::{ParticipantLinked, ParticipantUnlinked},
};
use lk_agents_rtc::RoomEvent;
use lk_agents_voice::{
    agent_activity::lifecycle::RoomLinkConfig,
    agent_activity::options::{ConfigStack, LegSet, SessionPatch, TurnDetection},
    agent_activity::options::{
        Deployment, InterruptionGate, InterruptionGateInputs, RealtimeGate, SttGate,
        resolve_interruption_detector,
    },
    agent_activity::pause::InterruptRefusal,
    agent_activity::scheduling::SpeechId,
    DriverBuilder, SessionHandle,
    agent::TurnEdit,
    agent_activity::DriverConfig,
    agent_session::{
        AgentUpdate, Close, Command, DrainRefused, DtmfCommand, FillerSource, FrameCommand,
        HandoffRefused, HoldId, HookId, HookOutcome, HostAudio, HostAudioOut, HostEvent,
        HostRecorder, HostTextOut, HostVideo, IdleRefused, IvrCommand, NewActivity,
        PreviousActivity, RecorderCommand, SayText, SpeechCommand, TaskCommand, TaskEntry,
        Timebase, ToolUpdate, TurnCommand,
        cycle::Decision,
    },
    amd::{
        classifier::{AmdConfig, In as AmdIn, Start as AmdStart},
        detector::{AmdSetup, AmdSystem},
    },
    events::SessionEvent,
    filler_scheduler::FillerId,
    generation::Instructions,
    io::{AudioOutput, TextOutput},
    keyterm_detection::{KeytermDetection, KeytermSetup},
    room_io::RoomIoEvent,
    run_result::{RunOutput, RunSpec},
    speech_handle::SpeechHandle,
    tool_executor::{ToolAnswer as Answered, Toolset},
    trace::{
        CoreStep, DetectorStep, GenerationStep, HandoffStep, PlayoutStep, SpeechStep, Step,
        ToolStep, TurnStep,
    },
    turn::Nullable,
    workflows::TaskHandle,
    workflows::builtins::dtmf::{DtmfCollector, DtmfEvent, DtmfOutcome},
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use crate::{
    check::{Failure, Subject, check_all},
    fakes::{self, Control, Record, Registry, SAMPLE_RATE, SAMPLES_PER_CHUNK, Slot},
    input::{
        AgentText, Confidence, Count, DtmfInput, HookAnswer, Input, IvrInput, Knob, Legs, Millis,
        Ref, Role, RoomIn, RtEvent, SayInput, ToolAnswer, TurnMode, UserText,
    },
    record::step_name,
    record::{
        Answer, Detail, Handle, Host, Id, Kind, Observed, Phase, Recording, Sink, Stamps, Tag,
        hook_of,
    },
};
use lk_agents_abi::{
    voice::session_options::{DetectorKind, EndpointingMode, InterruptionMode},
    turn::interruption::InterruptionPort,
};

mod apply;
mod collect;
mod finish;
mod start;

pub use apply::BUDGET_MS;
pub(crate) use finish::RELEASE_CAP_MS;
pub use finish::{Outcome, finish, guard, run};

use self::apply::{Feeding, Pending};
// ==============================================================================================
// What a session is built with
// ==============================================================================================

/// What the session is built with: everything fixed for the life of a run. Everything a case can
/// change afterwards is an [`Input`] instead, so a saved case says what it did rather than how the
/// session was configured to let it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Setup {
    /// A recognition leg listening to the microphone, so recognition events can be delivered. The
    /// microphone itself is always there: a voice call has one whether or not anything transcribes
    /// it.
    pub stt: bool,
    /// A voice-activity detector on the microphone leg: the second observer of the person.
    #[serde(default)]
    pub vad: bool,
    /// An end-of-turn predictor on the microphone leg: the third observer of the person.
    #[serde(default)]
    pub eot: bool,
    /// The adaptive interruption detector: judges whether an overlap is a barge-in.
    #[serde(default)]
    pub interruption: bool,
    /// Learn the endpointing delay from the conversation instead of a fixed one.
    #[serde(default)]
    pub dynamic_endpointing: bool,
    pub turn: TurnMode,
    /// Preemptive generation on a preflight transcript, with preemptive TTS.
    pub preemptive: bool,
    /// Tools advertised to the model; each is run by the controlled tool runner.
    pub tools: Vec<String>,
    pub max_tool_steps: Option<u32>,
    pub instructions: String,
    pub max_unrecoverable_errors: u32,
    /// Host hooks: the driver issues `HookCall`s on the event stream and waits up to this many
    /// milliseconds for the case to answer with `HookDone`. `None` runs without host hooks.
    pub hook_bound_ms: Option<u64>,
    /// A realtime model instead of the LLM+TTS pair: the socket is the voice, and the whole
    /// `Realtime` journal family exists only under one. `Some` describes the server the session
    /// is built against.
    #[serde(default)]
    pub realtime: Option<RealtimeSetup>,
    /// The IVR navigator: `None` starts none, `Some(detect)` starts one and navigates from the
    /// session's start when `detect`. Without one an `Ivr` command is relayed to nobody.
    #[serde(default)]
    pub ivr: Option<bool>,
    /// Attach the answering-machine detector, as `await amd.execute()` does.
    #[serde(default)]
    pub amd: bool,
    /// Static keyterms, which is also what makes the session have a keyterm pass at all.
    #[serde(default)]
    pub keyterms: Vec<String>,
    /// Room legs: the room's events, the room io machine's, and the link's two switches.
    #[serde(default)]
    pub room: Option<RoomSetup>,
    /// An audio output that can pause: what makes a false interruption resumable at all, and so
    /// the whole pause machine reachable. The fake pauses by recording that it was told to.
    #[serde(default)]
    pub pausable_output: bool,
    /// `min_consecutive_speech_delay`: the floor the scheduler holds between two of the agent's
    /// speeches. `None` is the default of none.
    #[serde(default)]
    pub speech_gap_ms: Option<u64>,
    /// `user_turn_limit`: the words and the duration after which a user turn is declared
    /// exceeded. `None` is no limit, the default.
    #[serde(default)]
    pub user_turn_max_words: Option<u32>,
    #[serde(default)]
    pub user_turn_max_ms: Option<u64>,
    /// `transcription_timeout`: how long after the detector's end of speech the driver waits for
    /// a transcript before giving up on one. `None` is the default of never.
    #[serde(default)]
    pub transcription_timeout_ms: Option<u64>,
    /// The end-of-turn model's backchannel threshold, in percent: with one, a prediction whose
    /// backchannel probability clears it while the turn is not over is an opportunity to
    /// backchannel. `None` is no threshold, the default.
    #[serde(default)]
    pub backchannel_threshold: Option<u8>,
    /// A TTS leg. Off, a pipeline session has an LLM and an audio output but no voice, which is
    /// the one configuration in which `say(words)` has nothing to speak with.
    #[serde(default = "yes")]
    pub tts: bool,
    /// How long every provider stream takes to open, on the session's clock. `None` is at once.
    #[serde(default)]
    pub connect_ms: Option<u64>,
    /// A camera attached from the start, for the case to feed frames into.
    #[serde(default)]
    pub video: bool,
    /// The transcript synchroniser between the two outputs: the agent's words are shown paced
    /// against its audio rather than as they are generated, which is the default with a room.
    #[serde(default)]
    pub sync_transcription: bool,
    /// The speaker filter from the start: learn who the primary speaker is from diarised
    /// transcripts and suppress the others.
    #[serde(default)]
    pub speaker_filter: bool,
}

fn yes() -> bool {
    true
}

/// The realtime server a session is built against: what it can do, as its capabilities say.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RealtimeSetup {
    /// The server detects the user's turns itself. The driver then has a choice the fake
    /// otherwise never offers it: to leave detection to the server or to switch it off.
    pub turn_detection: bool,
}

/// `RoomOptions`' two session-level switches. `Some` is a session with room io attached, which
/// is what gives the `Room` family anything to journal.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RoomSetup {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
}

/// One run, whole: how the session is built and what is sent to it.
///
/// The setup is not optional detail. Some of what a rule needs cannot be reached by any input,
/// because it is fixed when the session is constructed -- speculative generation, the turn mode,
/// whether host hooks are bound. A probe that carried only inputs could never reach those rules,
/// so a case carries both halves and a saved case replays both.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Case {
    pub setup: Setup,
    pub inputs: Vec<Input>,
    /// The scheduler's seed: which interleaving of the driver's concurrent work this case ran
    /// under. Part of the case because a finding may exist under one interleaving only, and a
    /// replay that used another would not be a replay. Zero is the schedule of every case saved
    /// before this was a dimension.
    #[serde(default)]
    pub schedule: u64,
}

impl Default for Setup {
    fn default() -> Self {
        Setup {
            stt: false,
            vad: false,
            eot: false,
            interruption: false,
            dynamic_endpointing: false,
            turn: TurnMode::Default,
            preemptive: false,
            tools: vec!["lookup".into()],
            max_tool_steps: None,
            instructions: "You are a test agent.".into(),
            max_unrecoverable_errors: 3,
            hook_bound_ms: None,
            realtime: None,
            pausable_output: false,
            speech_gap_ms: None,
            user_turn_max_words: None,
            user_turn_max_ms: None,
            transcription_timeout_ms: None,
            backchannel_threshold: None,
            tts: true,
            connect_ms: None,
            video: false,
            sync_transcription: false,
            speaker_filter: false,
            ivr: None,
            amd: false,
            keyterms: Vec::new(),
            room: None,
        }
    }
}

// ==============================================================================================
// The simulation
// ==============================================================================================

/// The scheduler seed a run uses when its case names none: the one every case was run with
/// before schedules were part of a case, so every saved case still reproduces. The bytes are the
/// crate's former name; changing them would change every interleaving the corpus was saved under.
const SCHEDULER_SEED: &[u8] = b"lk-agents-voice-tests";

/// Run one simulation under the schedule every case ran with before schedules were a dimension.
/// Tests with no case go through here.
pub fn simulate<T>(body: impl std::future::Future<Output = T>) -> T {
    simulate_with(0, body)
}

/// Run one simulation: paused time, a single thread, and a scheduler whose choices are fixed by
/// `schedule`.
///
/// Tokio randomises unbiased `select!` branch order and task polling order per call, which is the
/// interleaving of the driver's concurrent work: two events in one instant, a verdict against an
/// overlap's close, a timer against a command. Pinning the seed is what makes a saved case
/// reproduce; varying it across cases is what makes the campaign explore interleavings rather
/// than walk one of them a thousand times. Zero is the schedule of every case saved before this
/// was a dimension.
pub fn simulate_with<T>(schedule: u64, body: impl std::future::Future<Output = T>) -> T {
    let mut seed = SCHEDULER_SEED.to_vec();
    if schedule != 0 {
        seed.extend_from_slice(&schedule.to_le_bytes());
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .rng_seed(tokio::runtime::RngSeed::from_bytes(&seed))
        .build()
        .expect("simulation runtime")
        .block_on(body)
}

/// Retains real Tokio joins, so a task the driver detached cannot disappear from the test.
#[derive(Default)]
struct Tasks(Mutex<Vec<JoinHandle<()>>>);

impl Spawner for Tasks {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> lk_agents_abi::AbortHandle {
        let task = tokio::spawn(future);
        let abort = task.abort_handle().into();
        self.0.lock().unwrap().push(task);
        abort
    }
}

impl Drop for Tasks {
    fn drop(&mut self) {
        for task in self.0.get_mut().unwrap().drain(..) {
            task.abort();
        }
    }
}

/// The side table a host ref points into. Releases are no-ops: no host holds these objects.
struct NoopRefs;

impl RefRegistry for NoopRefs {
    fn release(&self, _id: u64) {}
}

// ==============================================================================================
// The run
// ==============================================================================================

/// A real session, the fakes behind it, and everything the run has learned.
pub struct Run {
    /// The transcript synchroniser, when the session has one: closed with the session, as the
    /// session layer closes its own.
    sync: Option<Arc<lk_agents_voice::transcription::synchronizer::TranscriptSynchronizer>>,
    session: SessionHandle,
    driver: Option<JoinHandle<()>>,
    tasks: Arc<Tasks>,
    stamps: Stamps,
    pub registry: Arc<Registry>,
    /// What the fakes have done and not yet been read. Drained after every input; see [`Record`]
    /// for why reading it later still yields the order things happened in.
    records: mpsc::UnboundedReceiver<Record>,
    /// The way to reach a live fake, owned outright and shared with no one. A delivery is a
    /// lookup in this map and a call, so there is no lock on that path to re-enter.
    controls: BTreeMap<Slot, Control>,
    /// Which output each flushed segment sits on, kept after the segment is gone.
    segments: BTreeMap<u32, u32>,
    /// Journal steps the observer stamped, waiting to be absorbed. The observer runs under the
    /// driver's own lock and cannot touch the recording, so it parks them here.
    steps: Arc<Mutex<Vec<(Observed, Step)>>>,
    pending: BTreeMap<u32, Pending>,
    streams: BTreeMap<u32, Feeding>,
    handles: BTreeMap<u64, Arc<SpeechHandle>>,
    /// Which recognition stream each utterance is on: an `SttInterim` names the utterance, and the
    /// fake behind it is reached through its stream.
    utterances: BTreeMap<u32, u32>,
    /// Each attached microphone's frame channel, and how many frames it has carried.
    mics: BTreeMap<u32, (mpsc::UnboundedSender<AudioFrame>, u16)>,
    /// The next filler id to mint. The driver is told the id rather than choosing it, so a case can
    /// name a filler in the same burst that started it.
    next_filler: u64,
    /// The next inline task id to mint. Per-`Run` and not process-global, so two runs of one case
    /// journal the same `task_0001`, `task_0002`, ... -- the determinism suite reads those ids.
    next_task: u64,
    refs: Arc<dyn RefRegistry>,
    /// The room's own event channel, and the room io machine's. The harness holds the sending
    /// end of both: a room event is something that happens to the session, so it is an input.
    room: tokio::sync::mpsc::UnboundedSender<RoomEvent>,
    room_io: tokio::sync::mpsc::UnboundedSender<RoomIoEvent>,
    pub setup: Setup,
    pub record: Recording,
}

impl Drop for Run {
    fn drop(&mut self) {
        if let Some(driver) = &self.driver {
            driver.abort();
        }
    }
}
