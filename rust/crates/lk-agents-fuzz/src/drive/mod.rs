//! A real session with controlled dependencies, and one function that applies an [`Input`] to it.
//!
//! The only place that touches the product, and only as a host does: the session is assembled
//! with `LiveSession(...)` (`SessionApi::host_new`) over the fake room and the fakes, started,
//! and every call is the host method a Python `AgentSession` makes, with its argument shapes;
//! people join and leave through the fake room, which the real room io watches. What the host
//! cannot do it does not do: the test-only probe (the `dst` feature) only watches the journal and
//! the host events, and hands the session's cores to the run, which steps them as its own queue.
//! It decides nothing: rules read [`Recording`] and nothing else.
//!
//! **Each resource is opened by whoever mints it.** Speeches, holds, fillers and hook calls come
//! from the driver's own journal, the one account of them that cannot drift from what it did. The
//! rest have no journal to read -- a chat exists because a fake was called, an answer because a
//! call returned a future -- and are opened here, from the side that knows.
//!
//! **Applying an input does not yield.** Only [`Input::Advance`] and the awaits reach the
//! executor, so a case can build a burst the driver must untangle in one cycle.
//! [`Recording::driver_ran_since`] is written against that, and breaking it would quietly weaken
//! every rule that allows the driver one input of grace.
//!
//! The module is split by what a run does, in order: [`start`] builds the session, [`apply`]
//! turns one input into what the product sees, [`collect`] writes down what became true, and
//! [`finish`] releases what a run owes and reports what it found. This file holds what they
//! share: the setup, the simulation, and the run itself.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use lk_agents_abi::{
    AudioFrame, BoxFuture, ChatItem, ChatMessage, HostRef, ImageContent, ImageSource, Maybe,
    MediaRef, Realtime, Spawner, Ts,
    errors::RefRegistry,
    metrics::base::AgentMetrics,
    turn::interruption::InterruptionSource,
    turn::keyterms::{DetectionLlm, KeytermDetectionConfig, SttContextConfig},
    voice::room_io::{AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig},
    voice::session_options::{
        DeprecatedTurnKwargs, EndpointingKwargs, HostValue, InterruptionKwargs, PreemptiveKwargs, SessionKwargs,
        TurnDetectionKwarg, TurnHandlingKwargs, UserTurnLimitKwargs,
    },
};
use lk_agents_rtc::{FakeRoom, ParticipantKind};
use lk_agents_abi::TokioSpawner;
use lk_agents_voice::background_audio::BackgroundAudioPlayer;
use lk_agents_voice::remote_session::driver::HostRuntime as RemoteSessionHost;
use lk_agents_voice::remote_session::{
    WireEvent, proto,
    proto::agent_session_message::{Message, console_io},
};
use lk_agents_voice::room_io::types::PRE_CONNECT_AUDIO_BUFFER_STREAM;
use lk_agents_voice::{
    Runtime,
    agent_activity::options::{
        Deployment, HostOptions, InterruptionGate, InterruptionGateInputs, RealtimeGate,
        SessionPatch, SttGate, TurnDetection, resolve_interruption_detector,
    },
    agent_activity::scheduling::SpeechId,
    agent_session::{
        Event, FixedRoom, FramePusher, HoldId, HookId, HostEvent, IoKind, Origin, Probe,
        SessionApi, SessionEvents, SpeechCommand, TextPusher, Timebase, TurnCommand,
        UpdateAgentArgs, api::NodeFactory, effects::Decision,
    },
    events::SessionEvent,
    filler_scheduler::FillerId,
    generation::Instructions,
    speech_handle::{HostSpeech, SpeechHandle},
    tool_executor::{HostToolRunner, ToolAnswer as Answered},
    trace::{
        CoreStep, DetectorStep, GenerationStep, HandoffStep, PlayoutStep, SpeechStep, Step,
        ToolStep, TurnStep,
    },
    turn::Nullable,
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use crate::{
    check::{Failure, Subject, check_all},
    fakes::{self, Control, Record, Registry, SAMPLE_RATE, SAMPLES_PER_CHUNK, Slot},
    input::{
        AgentText, AmdInput, Confidence, ConsoleAsk, Count, DtmfInput, HookAnswer, Input, IvrInput,
        Knob, Millis, Models, Ref, Role, RoomIn, RtEvent, SayInput, ToolAnswer, TurnMode, UserText,
    },
    record::step_name,
    record::{
        Answer, Detail, Handle, Host, Id, Kind, Observed, Phase, Recording, Sink, Stamps, Tag,
        hook_of,
    },
};
use lk_agents_abi::voice::session_options::{EndpointingMode, InterruptionMode, TextTransformSet};

mod apply;
mod collect;
mod finish;
mod queue;
mod start;

pub use apply::BUDGET_MS;
pub use finish::{Outcome, PHASES, finish, guard, run};
pub use queue::Policy;

use self::apply::{Feeding, Pending};
use self::queue::Next;
// ==============================================================================================
// What a session is built with
// ==============================================================================================

/// What the session is built with: everything fixed for the life of a run. Everything a case can
/// change afterwards is an [`Input`] instead, so a case says what it did rather than how the
/// session was configured to let it.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    /// A recognition leg listening to the microphone, so recognition events can be delivered. The
    /// microphone itself is always there: a voice call has one whether or not anything transcribes
    /// it.
    pub stt: bool,
    /// A voice-activity detector on the microphone leg: the second observer of the person.
    pub vad: bool,
    /// An end-of-turn predictor on the microphone leg: the third observer of the person.
    pub eot: bool,
    /// The adaptive interruption detector: judges whether an overlap is a interruption.
    pub interruption: bool,
    /// Learn the endpointing delay from the conversation instead of a fixed one.
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
    pub realtime: Option<RealtimeSetup>,
    /// `ivr_detection`: `Some(true)` navigates from the session's start. The navigator itself
    /// runs in every session, as `start()` spawns it.
    pub ivr: Option<bool>,
    /// The answering-machine detector, attached as `AMD(session, ...)` attaches it.
    pub amd: Option<AmdSetup>,
    /// Static keyterms, which is also what makes the session have a keyterm pass at all.
    pub keyterms: Vec<String>,
    /// The agent overrides `llm_node`, `tts_node` and `stt_node` (each as the default does), on
    /// the first agent and on every agent a handoff installs.
    pub nodes: Option<NodeSetup>,
    /// `stt_context_options["keyterm_detection"]` on, with the detection LLM the host resolves.
    pub keyterm_detection: Option<KeytermDetectionSetup>,
    /// What happens in the room: `None` is a room nothing happens in (a `Room` input is refused),
    /// `Some` a room people join and leave, with `RoomOptions`' two switches.
    pub room: Option<RoomSetup>,
    /// An audio output that can pause: what makes a false interruption resumable at all, and so
    /// the whole pause machine reachable. The fake pauses by recording that it was told to.
    pub pausable_output: bool,
    /// `min_consecutive_speech_delay`: the floor the scheduler holds between two of the agent's
    /// speeches. `None` is the default of none.
    pub speech_gap_ms: Option<u64>,
    /// `user_turn_limit`: the words and the duration after which a user turn is declared
    /// exceeded. `None` is no limit, the default.
    pub user_turn_max_words: Option<u32>,
    pub user_turn_max_ms: Option<u64>,
    /// `transcription_timeout`: how long after the detector's end of speech the driver waits for
    /// a transcript before giving up on one. `None` is the default of never.
    pub transcription_timeout_ms: Option<u64>,
    /// A TTS leg. Off, a pipeline session has an LLM and an audio output but no voice, which is
    /// the one configuration in which `say(words)` has nothing to speak with.
    pub tts: bool,
    /// How long every provider stream takes to open, on the session's clock. `None` is at once.
    pub connect_ms: Option<u64>,
    /// A camera attached from the start, for the case to feed frames into.
    pub video: bool,
    /// `RoomOptions(video_input=True)` on the room's legs: the person publishes a camera with
    /// their microphone, and the room's video input reads it.
    pub room_camera: bool,
    /// The transcript synchroniser between the two outputs: the agent's words are shown paced
    /// against its audio rather than as they are generated, which is the default with a room.
    pub sync_transcription: bool,
    /// An audio output that reports a cut a few scheduler hops late, as a host source does.
    pub lagging_output: bool,
    /// `tts_text_transforms`: `None` is not given (markdown and emoji filtered), `Some` the set
    /// written, `Some(None)` none.
    pub text_transforms: Option<Option<TextTransformSet>>,
    /// The room's own audio and transcription legs, which a host that hands none of its own
    /// runs on (`agent_session.py`'s default): the person is in the room from the start and
    /// speaks into a microphone track, and the agent's voice plays out on the room's clock.
    /// `None` is the host's own legs, the fakes.
    pub room_legs: Option<RoomLegs>,
    /// `lk agent console`: the CLI's socket is the session's microphone and speaker, and a
    /// `RemoteSessionHost` on it forwards the session's events and answers the CLI's requests,
    /// as `voice/console.py` runs it. Never with the room's legs.
    pub console: bool,
    /// The turn keywords a host still writes the pre-`turn_handling` way (`min_endpointing_delay`,
    /// `allow_interruptions`, ...), which the core migrates.
    pub deprecated_turn: bool,
    /// `VoiceActivityVideoSampler()`, which `agent_session.py` hands over by default: camera
    /// frames reach the model at one rate while the person speaks and another while silent.
    pub video_sampler: bool,
    /// `BackgroundAudioPlayer` on the room: an ambient sound from the start and a thinking sound
    /// while the agent thinks, mixed on its own published track. Only with the room's legs.
    pub background_audio: bool,
    /// An avatar worker in the room: the agent's voice streams to it (`DataStreamAudioOutput`,
    /// what an avatar plugin sets as `session.output.audio`) and it reports each segment played.
    /// Only with the room's legs.
    pub avatar: bool,
    /// The avatar reports when each segment starts playing (`wait_playback_start=True`).
    pub avatar_reports_start: bool,
}

/// The room's legs, and whether the person's app sends the audio it buffered before the agent
/// joined (`pre_connect_audio`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomLegs {
    pub pre_connect: Option<PreConnect>,
}

/// What the person's app does with its pre-connect buffer: sends it, never sends it (the
/// session waits out its timeout), or sends one the session cannot read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreConnect {
    Sent,
    Never,
    Malformed,
}

/// Which of the three nodes the agents override.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSetup {
    pub llm: bool,
    pub tts: bool,
    pub stt: bool,
}

/// Keyterm detection as `KeytermDetectionOptions` sets it up, and whether the session's
/// recogniser takes keyterms at all (`STTCapabilities.keyterms`): without it no pass runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeytermDetectionSetup {
    pub stt_takes_keyterms: bool,
    pub turn_interval: u32,
    pub max_keyterms: Option<usize>,
    pub timeout_ms: Option<u64>,
}

/// `AMD(session, ...)`: its switches, its recogniser, and what the room says once it is attached.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmdSetup {
    pub interrupt_on_machine: bool,
    pub ivr_detection: bool,
    pub wait_until_finished: bool,
    /// `AMD(stt=..)`: a recogniser of AMD's own; otherwise the session's finals.
    pub own_stt: bool,
    pub answer: AmdAnswer,
}

/// What `AMD._setup` finds in the room, as `amd_decide_setup_step` turns it into a step: the
/// callee's track (listening opens at once), a SIP call answered later (the case sends
/// `Amd { Listen }` when `sip.callStatus` goes active), or nobody (`participant_missing`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AmdAnswer {
    #[default]
    Track,
    Sip,
    Missing,
}

/// The realtime server a session is built against: what it can do, as its capabilities say.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealtimeSetup {
    /// The server detects the user's turns itself. The driver then has a choice the fake
    /// otherwise never offers it: to leave detection to the server or to switch it off.
    pub turn_detection: bool,
}

/// `RoomOptions`' two session-level switches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomSetup {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
}

/// One run, whole: how the session is built and what is sent to it.
///
/// The setup is not optional detail. Some of what a rule needs cannot be reached by any input,
/// because it is fixed when the session is constructed -- speculative generation, the turn mode,
/// whether host hooks are bound. A probe that carried only inputs could never reach those rules,
/// so a case carries both halves.
#[derive(Clone, Debug, Default)]
pub struct Case {
    pub setup: Setup,
    pub inputs: Vec<Input>,
    /// The scheduler's seed: which interleaving of the driver's concurrent work this case ran
    /// under. Part of the case because a finding may exist under one interleaving only. Zero is
    /// the schedule of a hand-written case.
    pub schedule: u64,
    /// Indices into `inputs` of the uniform moves: rows no actor had a reason to send. A failure
    /// in a case with none is one a real call produces with correct providers.
    pub random: Vec<usize>,
    /// How the run's queue picked the next pending event. Arrival order by default.
    pub policy: Policy,
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
            tts: true,
            connect_ms: None,
            video: false,
            room_camera: false,
            sync_transcription: false,
            lagging_output: false,
            ivr: None,
            amd: None,
            keyterms: Vec::new(),
            keyterm_detection: None,
            nodes: None,
            room: None,
            text_transforms: None,
            room_legs: None,
            console: false,
            deprecated_turn: false,
            video_sampler: false,
            background_audio: false,
            avatar: false,
            avatar_reports_start: false,
        }
    }
}

// ==============================================================================================
// The simulation
// ==============================================================================================

/// The scheduler seed a run uses when its case names none. The bytes are the crate's former name;
/// changing them would change every interleaving the hand-written regression tests run under.
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
/// overlap's close, a timer against a command. Pinning the seed is what makes a seed reproduce
/// its run; varying it across cases is what makes the campaign explore interleavings rather than
/// walk one of them a thousand times. Zero is the schedule of a hand-written case.
pub fn simulate_with<T>(schedule: u64, body: impl std::future::Future<Output = T>) -> T {
    let mut seed = SCHEDULER_SEED.to_vec();
    if schedule != 0 {
        seed.extend_from_slice(&schedule.to_le_bytes());
    }
    let _watchdog = watchdog(schedule);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        // The run's own future is polled after every task, so the queue owner can step the driver
        // between any two polls of the tasks it spawned.
        .event_interval(1)
        .rng_seed(tokio::runtime::RngSeed::from_bytes(&seed))
        .build()
        .expect("simulation runtime")
        .block_on(body)
}

/// Real seconds one simulation may take before the watchdog ends the process; `VOICE_HANG_S`
/// overrides it. A soak call, the longest run, takes about a second optimised and sixteen times
/// that unoptimised: this must never fire on a run that is merely slow.
const HANG_S: u64 = 600;

/// What no step count can see: a product task that never yields, or a lock that never comes back,
/// blocks the run's only thread, so nothing on it can notice. A thread beside it can, and the only
/// thing it can do about it is end the process loudly, naming the schedule, rather than leave a
/// campaign parked forever. Disarmed when the returned sender is dropped, with the simulation.
fn watchdog(schedule: u64) -> std::sync::mpsc::Sender<()> {
    use std::io::Write;
    let limit = std::env::var("VOICE_HANG_S").ok().and_then(|s| s.parse().ok()).unwrap_or(HANG_S);
    let (disarm, armed) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
            armed.recv_timeout(Duration::from_secs(limit))
        {
            // Straight to the stream: the test harness captures `eprintln!` and loses it on abort.
            let _ = writeln!(
                std::io::stderr(),
                "HANG: the simulation of schedule {schedule} (the survey's seed) is still running \
                 after {limit}s of real time: a step or a task is blocked or never yields. \
                 Rerun it with VOICE_SEED={schedule} VOICE_TRACE=1 to see the input it stopped on."
            );
            std::process::abort();
        }
    });
    disarm
}

/// Steps one virtual instant may hold, in the journal or in the run's own loop, before the run
/// counts as a livelock. The busiest instant measured (1024-seed survey and soak, 2026-10-25) took
/// 503 journal steps and 127 turns of the loop: twenty times that is a session feeding itself,
/// and waiting longer only spends real time before the same verdict.
const NO_PROGRESS_STEPS: u32 = 10_000;

/// Retains real Tokio joins, so a task the driver detached cannot disappear from the test, each
/// with when and from which line it was spawned, so a task that never finishes can be placed in
/// the run and in the source.
struct Tasks {
    stamps: Stamps,
    spawned: Mutex<Vec<Spawned>>,
}

struct Spawned {
    at: Observed,
    site: &'static std::panic::Location<'static>,
    task: JoinHandle<()>,
}

impl Spawner for Tasks {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> lk_agents_abi::AbortHandle {
        let site = std::panic::Location::caller();
        let task = tokio::spawn(future);
        let abort = task.abort_handle().into();
        self.spawned.lock().unwrap().push(Spawned { at: self.stamps.now(), site, task });
        abort
    }
}

impl Drop for Tasks {
    fn drop(&mut self) {
        for spawned in self.spawned.get_mut().unwrap().drain(..) {
            spawned.task.abort();
        }
    }
}

/// Where a microphone's frames go: the host's own leg, or the person's track in the room.
enum Mic {
    Host(mpsc::UnboundedSender<AudioFrame>),
    /// The track, and how many frames went into it at which instant.
    Room { track: Arc<lk_agents_rtc::FakeTrackPublication>, burst: (u64, u32) },
    /// The console's socket, as the CLI sends its microphone.
    Console(fakes::FakeCli),
}

/// The most frames pushed into the person's track in one instant: half of what each of the
/// track's two queues holds (`AUDIO_QUEUE_FRAMES`), which drain only as time moves.
const ROOM_BURST_FRAMES: u32 = (lk_agents_abi::AUDIO_QUEUE_FRAMES / 2) as u32;

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
    /// The session, as a host holds it.
    api: Arc<SessionApi>,
    /// The room it is in: people join and leave it as a case says.
    room: Arc<FakeRoom>,
    /// `aclose()` then `close_io()`, once a case asked for the close, until they resolve.
    closing: Option<BoxFuture<'static, ()>>,
    /// The session's cores, stepped by the run itself: the run owns its queue. `None` once it
    /// finished, or panicked.
    runtime: Option<Runtime>,
    queue: queue::Queue,
    time: Timebase,
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
    /// Journal steps the probe stamped, waiting to be absorbed. The probe runs under the
    /// driver's own lock and cannot touch the recording, so it parks them here.
    steps: Arc<Mutex<Vec<(Observed, Step)>>>,
    /// Host events the probe stamped as the session's drain handed them over.
    host: Arc<Mutex<Vec<(Observed, Host)>>>,
    /// Set by the observer when it stopped a step for journaling without bound at one instant.
    livelock: Arc<Mutex<Option<String>>>,
    pending: BTreeMap<u32, Pending>,
    streams: BTreeMap<u32, Feeding>,
    handles: BTreeMap<u64, Arc<SpeechHandle>>,
    /// Which recognition stream each utterance is on: an `SttInterim` names the utterance, and the
    /// fake behind it is reached through its stream.
    utterances: BTreeMap<u32, u32>,
    /// Each attached microphone, and how many frames it has carried.
    mics: BTreeMap<u32, (Mic, u16)>,
    /// The host replaced the room's microphone: the person's track is no longer what the session
    /// hears, so a rejoin publishes one nobody listens to.
    host_mic: bool,
    /// How many times the person joined the room, which names their microphone track.
    joins: u32,
    /// The console's session host and the CLI on the other end of its socket.
    console: Option<(Arc<RemoteSessionHost>, fakes::FakeCli)>,
    /// The next console request's id.
    next_request: u64,
    /// The background audio player, as `background_audio.py` runs it beside the session.
    background: Option<Arc<BackgroundAudioPlayer>>,
    /// The avatar worker the agent's voice streams to, in its own room.
    avatar: Option<Arc<fakes::FakeAvatarWorker>>,
    /// The next filler id to mint. The host tells the driver the id rather than letting it
    /// choose, so a case can name a filler in the same burst that started it.
    next_filler: u64,
    /// The next inline task id to mint. Per-`Run` and not process-global, so two runs of one case
    /// journal the same `task_0001`, `task_0002`, ... -- the determinism suite reads those ids.
    refs: Arc<dyn RefRegistry>,
    /// The session's providers, given again by a handoff whose agent overrides nodes.
    legs: start::Legs,
    pub setup: Setup,
    pub record: Recording,
}
