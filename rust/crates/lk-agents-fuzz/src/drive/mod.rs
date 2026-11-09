use lk_agents_voice::audio_recognition::observation::RecognitionObservation;

type HostJournal = Arc<Mutex<Vec<(Observed, Host, Option<Arc<SpeechHandle>>)>>>;

use lk_agents_runtime::voice::SpeechHandle;
use lk_agents_runtime::voice::session::Timebase;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::fakes::room::FakeRoom;
pub use crate::fakes::room::Faults;
use lk_agents_abi::voice::background_audio::BackgroundAudioPlayer as _;
use lk_agents_abi::voice::nodes::NodeFactory;
use lk_agents_abi::voice::rtc::ParticipantKind;
use lk_agents_abi::voice::session::UpdateAgentArgs;
use lk_agents_abi::voice::session_effects::SessionEvents;
use lk_agents_abi::{
    AudioFrame, BoxFuture, ChatItem, ChatMessage, ImageContent, ImageSource, Maybe, Realtime,
    Spawner, Ts,
    errors::RefRegistry,
    metrics::base::AgentMetrics,
    turn::interruption::InterruptionSource,
    turn::keyterms::{DetectionLlm, KeytermDetectionConfig, SttContextConfig},
    utils::video::FakeFrame,
    voice::room_io::{AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig},
    voice::session_options::{
        EndpointingKwargs, InterruptionKwargs, PreemptiveKwargs, SessionConnOptions, SessionKwargs,
        TurnDetectionKwarg, TurnHandlingKwargs, UserTurnLimitKwargs,
    },
};
use lk_agents_abi::{Launch, OwnedTask};
use lk_agents_runtime::voice::background_audio::{BackgroundAudio, BackgroundAudioPlayer};
use lk_agents_runtime::voice::session::{Probe, Pusher, SessionOptions, VoiceSession};
use lk_agents_voice::remote_session::{
    proto,
    proto::agent_session_message::{Message, console_io},
};
use lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM;
use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    agent_session::options::{
        AgentSessionOptions, Deployment, InterruptionGate, InterruptionGateInputs, RealtimeGate,
        SttGate, TurnDetection, resolve_interruption_detector,
    },
    agent_session::{
        HoldId, HookId, SpeechCommand, TurnCommand, VoiceCommandOrigin, VoiceHostEvent,
    },
    events::SessionEvent,
    filler_scheduler::FillerId,
    trace::{
        CoreStep, DetectorStep, GenerationStep, HandoffStep, PlayoutStep, ProviderKind, SpeechStep,
        Step, ToolStep, TurnStep,
    },
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
        TranscriptSetup, hook_of,
    },
};
use lk_agents_abi::voice::session_options::{EndpointingMode, InterruptionMode, TextTransformSet};

mod apply;
mod background;
mod collect;
mod executor;
mod finish;
mod probe;
pub mod queue;
mod start;

pub use apply::BUDGET_MS;
pub use finish::{Ending, Outcome, PHASES, finish, guard, run};
pub use probe::WAIT_BOUND_MS;
pub use queue::Policy;

use self::apply::{Feeding, Pending};
use self::queue::Next;

// The session is driven only as a host drives it, through the host API and the fake room.
// Speeches, holds, fillers and hook calls are opened from the driver's journal; everything else
// is opened here by the side that mints it. Applying an input never yields: only
// `Input::Advance` and the awaits reach the executor, which `Recording::driver_ran_since` and
// every rule granting one input of grace rely on.

/// What the session is built with: everything fixed for the life of a run. Everything a case can
/// change afterwards is an [`Input`] instead.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    /// A recognition leg listening to the microphone, so recognition events can be delivered. The
    /// microphone itself is always there: a voice call has one whether or not anything transcribes
    /// it.
    pub stt: bool,
    /// A voice-activity detector on the microphone leg: the second observer of the person.
    pub vad: bool,
    /// The detector is a plugin's: its audio waits on the core's `Now` queue for a pump that
    /// awaits each frame, and it takes none for this long after a stream opens (a stuck consumer).
    pub vad_stall_ms: Option<u64>,
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
    /// The agents override nodes (each as the default does): the opening agent these, every agent
    /// a handoff installs its own.
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
    /// The room's microphone input runs input processors that reject some of the person's audio:
    /// each failing span is the session's error, never its silence.
    pub failing_processors: Option<ProcessorFaults>,
    /// The transcript synchroniser between the two outputs: the agent's words are shown paced
    /// against its audio rather than as they are generated, which is the default with a room.
    pub sync_transcription: bool,
    /// The host's transcription output raises on every capture, as a Python `io.TextOutput` can.
    pub text_output_raises: bool,
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
    /// `lk agent console`: the CLI's socket is the session's microphone and speaker, and
    /// the session's remote session host on it forwards its events and answers the CLI's requests,
    /// as `voice/console.py` runs it. Never with the room's legs.
    pub console: bool,
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
    /// The avatar generator's frame rate and its latency per frame (under one frame period).
    pub avatar_fps: u64,
    pub avatar_latency_ms: u64,
    /// Which recorded recogniser the person's transcripts behave like, by its index in
    /// `search::STT_MODELS`: its timing, its cuts, its quirks.
    pub stt_model: usize,
    /// The voice takes as long to its first audio as a real one does, drawn from this seed;
    /// `None` answers at once.
    pub tts_latency: Option<u64>,
    /// `conn_options=SessionConnectOptions(..)`: each leg's retry budget. `None` is the defaults.
    pub conn: Option<ConnSetup>,
    /// A voice that does not stream: the session speaks it sentence by sentence through its
    /// stream adapter, each sentence a synthesis of its own, retried on its own.
    pub chunked_tts: bool,
    /// Whether the voice says it aligns its words to its audio, and whether it then does.
    pub tts_alignment: TtsAlignment,
    /// `VoiceSession(use_tts_aligned_transcript=)`; `None` is not given.
    pub aligned_transcript: Option<bool>,
    /// `Agent(use_tts_aligned_transcript=)` on the first agent and on every agent a handoff
    /// installs; `None` defers to the session's.
    pub agent_aligned_transcript: Option<bool>,
    /// Every tool body catches its cancellation and runs on (`except CancelledError:` in a
    /// Python tool): the host's cancel is refused, and the body still waits for its answer.
    pub tools_refuse_cancel: bool,
    /// The share of provider delays that stray from the measured ones to a deadline or a stall
    /// (`fakes::delay`): the search's randomness, so a pure conversation keeps real timings.
    pub stray: f64,
    /// The voice lane is stored, by storage's own code over its SQLite backend; `None` runs
    /// the session without storage.
    pub storage: Option<StorageSetup>,
    /// Panic and never-answer faults at the fakes (`fakes::hazard`), drawn from this seed, and
    /// host waits probed at random points (`probe`). `None` runs without either.
    pub stress: Option<u64>,
}

/// The thread database behind the voice lane. Every answer takes a seeded latency.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StorageSetup {
    /// Seeded faults on top: answers lost after the write applied, writes lost before it, lease
    /// takes answered after the interruption watchdog, and outages, a quarter of which outlast
    /// the lease.
    pub faults: bool,
    /// The database unreachable over `[from, from + len)` ms of the call, as `(from, len)`.
    pub outage: Option<(u64, u64)>,
    /// Every lease take answers this long after it applied: a slow acknowledgment.
    pub take_answer_ms: Option<u64>,
}

/// The pre-connect buffer wait the room's legs run with: shorter than the half second of frames
/// the track's queue holds, so the frames spoken while the session waits for a buffer that never
/// comes are not dropped.
pub const PRE_CONNECT_TIMEOUT_MS: u64 = 200;

impl Setup {
    /// What the run's provider delays are drawn against: the deadlines a session built from this
    /// setup arms, at their defaults where the setup leaves them.
    pub fn timing(&self) -> fakes::Timing {
        let conn = self.conn.unwrap_or(ConnSetup::DEFAULT);
        let mut deadlines = vec![
            PRE_CONNECT_TIMEOUT_MS,
            // Endpointing's default min and max delays.
            500,
            3_000,
            conn.interval_ms,
            // The connect options' limits, the close's transcript wait, the away timer.
            10_000,
            2_000,
            15_000,
        ];
        deadlines.extend(
            [
                self.speech_gap_ms,
                self.transcription_timeout_ms,
                self.user_turn_max_ms,
                self.hook_bound_ms,
                self.connect_ms,
                self.keyterm_detection.and_then(|k| k.timeout_ms),
            ]
            .into_iter()
            .flatten(),
        );
        deadlines.sort_unstable();
        deadlines.dedup();
        fakes::Timing { stray: self.stray, deadlines }
    }
}

/// Each leg's `max_retry`, and the interval between retries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConnSetup {
    pub stt: u32,
    pub llm: u32,
    pub tts: u32,
    pub interval_ms: u64,
}

impl ConnSetup {
    /// `SessionConnectOptions()`: three retries on every leg, two seconds apart.
    pub const DEFAULT: ConnSetup = ConnSetup { stt: 3, llm: 3, tts: 3, interval_ms: 2_000 };
}

/// `TTSCapabilities.aligned_transcript`, and what the voice then sends.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TtsAlignment {
    /// Not advertised: the voice sends audio only.
    #[default]
    None,
    /// Advertised, and every word comes back stamped, as Cartesia's `word_timestamps` do.
    Words,
    /// Advertised, and nothing comes back: a model and language pair with no timings.
    Missing,
}

/// The room's legs, and whether the person's app sends the audio it buffered before the agent
/// joined (`pre_connect_audio`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomLegs {
    pub pre_connect: Option<PreConnect>,
}

/// How the room microphone's processors reject audio: noise cancellation always runs, gain
/// control only with `gain_control` (the session turns it off otherwise).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProcessorFaults {
    pub denoiser: Faults,
    pub gain_control: Option<Faults>,
}

/// What the person's app does with its pre-connect buffer: sends it, never sends it (the
/// session waits out its timeout), or sends one the session cannot read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreConnect {
    Sent,
    Never,
    Malformed,
}

/// The agent the session opens with; a handoff names another.
pub const OPENING_AGENT: &str = "agent";

/// Which nodes the opening agent overrides; every other agent flips some of them
/// (`start::overrides`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSetup {
    pub llm: bool,
    pub tts: bool,
    pub stt: bool,
    /// The `llm_node` answers this text itself (a node returning a `str`), never calling the model.
    pub llm_text: Option<&'static str>,
    /// `transcription_node`, opened per segment over the session's text output.
    pub transcription: bool,
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

/// What the callee does in the room AMD awaits them in: joins with their audio (listening opens
/// at once), joins as a SIP call the case answers later (`Amd { Answer }` sets `sip.callStatus`
/// to `active`), or has not joined (`participant_missing` once the track wait runs out, unless
/// they join in time).
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
    /// Which server-side detector: `semantic_vad` (the default a session gets) or `server_vad`.
    pub semantic: bool,
    /// What the server's delays and choices are drawn from.
    pub seed: u64,
    /// How often the model calls a tool when it has some, in percent: the developer's prompt
    /// decides it. `None` is the measured rate.
    pub tool_calls: Option<u8>,
    /// `RealtimeModel(modalities=["text"])`: the server answers in text, which the session's TTS
    /// speaks when the setup has one.
    pub text_only: bool,
    /// A `RealtimeModelFallbackAdapter`'s model: a fatal error swaps to a fresh model.
    pub fallback: bool,
    /// A full-duplex speech model (`DuplexRealtimeAdapter`) instead of the server: it detects
    /// turns and speaks on its own; the server's switches above do not apply.
    pub duplex: bool,
    /// Delays the case sets instead of the seed drawing them.
    pub delays: crate::fakes::RealtimeDelays,
}

impl Default for RealtimeSetup {
    fn default() -> Self {
        RealtimeSetup {
            turn_detection: true,
            semantic: true,
            seed: 0,
            tool_calls: None,
            text_only: false,
            fallback: false,
            duplex: false,
            delays: Default::default(),
        }
    }
}

/// `RoomOptions`' two session-level switches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoomSetup {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
}

/// One run, whole: how the session is built and what is sent to it. Some prerequisites are fixed
/// at construction, so a case carries both halves.
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
            vad_stall_ms: None,
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
            failing_processors: None,
            sync_transcription: false,
            text_output_raises: false,
            lagging_output: false,
            ivr: None,
            amd: None,
            stt_model: 0,
            tts_latency: None,
            tts_alignment: TtsAlignment::None,
            aligned_transcript: None,
            agent_aligned_transcript: None,
            keyterms: Vec::new(),
            keyterm_detection: None,
            nodes: None,
            room: None,
            text_transforms: None,
            room_legs: None,
            console: false,
            video_sampler: false,
            background_audio: false,
            avatar: false,
            avatar_reports_start: false,
            avatar_fps: 25,
            avatar_latency_ms: 20,
            conn: None,
            chunked_tts: false,
            tools_refuse_cancel: false,
            stray: 0.0,
            storage: None,
            stress: None,
        }
    }
}

/// The scheduler seed a run uses when its case names none. Changing the bytes would change every
/// interleaving the hand-written regression tests run under.
const SCHEDULER_SEED: &[u8] = b"lk-agents-voice-tests";

/// Run one simulation under schedule zero, the schedule of a hand-written case.
pub fn simulate<T>(body: impl std::future::Future<Output = T>) -> T {
    simulate_with(0, body)
}

/// Run one simulation: paused time, a single thread, and a scheduler whose choices are fixed by
/// `schedule`. Tokio's unbiased `select!` and task polling order follow the seed, so pinning it
/// reproduces a run and varying it explores interleavings.
pub fn simulate_with<T>(schedule: u64, body: impl std::future::Future<Output = T>) -> T {
    let mut seed = SCHEDULER_SEED.to_vec();
    if schedule != 0 {
        seed.extend_from_slice(&schedule.to_le_bytes());
    }
    let _watchdog = watchdog(schedule, queue::Trail::install());
    tokio::runtime::Builder::new_current_thread()
        // Timers only: every leg is in memory, and an I/O driver would poll the OS on every turn
        // of the scheduler, which `event_interval(1)` makes every task poll.
        .enable_time()
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

/// A task that never yields blocks the run's only thread, so only a thread beside it can notice:
/// it aborts the process naming the schedule and dumping the run's trail. Disarmed when the
/// returned sender is dropped.
fn watchdog(schedule: u64, trail: Arc<Mutex<queue::Trail>>) -> std::sync::mpsc::Sender<()> {
    use std::io::Write;
    let caller = std::thread::current().name().unwrap_or("campaign worker").to_owned();
    let limit = std::env::var("VOICE_HANG_S").ok().and_then(|s| s.parse().ok()).unwrap_or(HANG_S);
    let (disarm, armed) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
            armed.recv_timeout(Duration::from_secs(limit))
        {
            // Straight to the stream: the test harness captures `eprintln!` and loses it on abort.
            let _ = writeln!(
                std::io::stderr(),
                "HANG: {caller}, schedule {schedule}, is still running \
                 after {limit}s of real time: a step or a task is blocked or never yields. \
                 A schedule is not necessarily a campaign seed. Rerun the named test, or use \
                 the survey's printed seed/budget/policy command, adding VOICE_TRACE=1."
            );
            // The run's thread may hold the trail's lock: never wait on it here.
            let dump = trail.try_lock().map_or_else(|_| "trail locked".into(), |t| t.dump());
            let _ = writeln!(std::io::stderr(), "{dump}");
            std::process::abort();
        }
    });
    disarm
}

/// Steps one virtual instant may hold, in the journal or in the run's own loop, before the run
/// counts as a livelock: twenty times the busiest instant measured (`Outcome::busiest_instant`,
/// 6,262 steps at seed 30341).
const NO_PROGRESS_STEPS: u32 = 20 * 6_262;

/// The run's spawner. Product tasks are retained with their spawn instant and site, so a detached
/// task that never finishes can be placed (`every_task_finishes`); the fakes' tasks ([`World`])
/// are scheduled the same way and aborted with the run.
struct Tasks {
    /// Teardown can drop a returned run outside `simulate_with`'s entered runtime. Keep
    /// launches tied to the original scheduler rather than consulting ambient context.
    runtime: tokio::runtime::Handle,
    stamps: Stamps,
    spawned: Mutex<Vec<Spawned>>,
    /// The fakes' tasks: the world's, not the product's, so never judged.
    world: Mutex<Vec<JoinHandle<()>>>,
    /// The task bridge, under `Policy::TaskBridge`.
    gate: Option<Arc<queue::Gate>>,
}

struct Spawned {
    at: Observed,
    site: &'static std::panic::Location<'static>,
    task: JoinHandle<()>,
}

impl Spawner for Tasks {
    fn launch(&self, task: Launch) -> OwnedTask<Box<dyn std::any::Any + Send>> {
        let site = task.site;
        let (run, owned) = lk_agents_runtime::net::task::launch(task);
        let run = match &self.gate {
            Some(gate) => gate.wrap(run),
            None => run,
        };
        let task = self.runtime.spawn(run);
        self.spawned.lock().unwrap().push(Spawned { at: self.stamps.now(), site, task });
        owned
    }
}

impl Drop for Tasks {
    fn drop(&mut self) {
        for spawned in self.spawned.get_mut().unwrap().drain(..) {
            spawned.task.abort();
        }
        for task in self.world.get_mut().unwrap().drain(..) {
            task.abort();
        }
    }
}

/// The run's spawner as the fakes hold it: a peer's or provider's own loop (a CLI, an avatar
/// worker, a host source's late report) runs under the run's scheduling like the product's
/// tasks, and ends with the run rather than being judged by it.
pub struct World(Arc<Tasks>);

impl Spawner for World {
    fn launch(&self, task: Launch) -> OwnedTask<Box<dyn std::any::Any + Send>> {
        let (run, owned) = lk_agents_runtime::net::task::launch(task);
        let run = match &self.0.gate {
            Some(gate) => gate.wrap(run),
            None => run,
        };
        self.0.world.lock().unwrap().push(self.0.runtime.spawn(run));
        owned
    }
}

/// Where a microphone's frames go: the host's own leg, or the person's track in the room.
enum Mic {
    Host(mpsc::UnboundedSender<AudioFrame>),
    /// The track, how many frames went into it at which instant, and, once the room io has read
    /// it, since when it is subscribed: past that instant the room io had its turn to read it
    /// again, so words spoken into it are owed to the session even when nothing reads them.
    Room {
        track: Arc<crate::fakes::room::FakeTrackPublication>,
        burst: (u64, u32),
        read: Option<u64>,
    },
    /// The console's socket, as the CLI sends its microphone.
    Console(fakes::FakeCli),
}

/// The most frames pushed into the person's track in one instant: half of the audio each of the
/// track's two queues holds (`AUDIO_QUEUE`), which drain only as time moves.
const ROOM_BURST_FRAMES: u32 = (lk_agents_abi::AUDIO_QUEUE.as_millis() as usize / 2
    * SAMPLE_RATE as usize
    / 1000
    / SAMPLES_PER_CHUNK) as u32;

/// The side table a host ref points into. Releases are no-ops: no host holds these objects.
pub(crate) struct NoopRefs;

impl RefRegistry for NoopRefs {
    fn release(&self, _id: u64) {}
}

/// A real session, the fakes behind it, and everything the run has learned.
pub struct Run {
    /// The session, as a host holds it.
    api: Arc<VoiceSession>,
    /// The room it is in: people join and leave it as a case says.
    room: Arc<FakeRoom>,
    /// `aclose()` then `close_io()`, once a case asked for the close, until they resolve.
    closing: Option<BoxFuture<'static, ()>>,
    /// The session's real loop over fake ports, fed by the run's deterministic queue. `None` once
    /// it closed, or panicked.
    session: Option<executor::Session>,
    queue: queue::Queue,
    time: Timebase,
    tasks: Arc<Tasks>,
    stamps: Stamps,
    pub registry: Arc<Registry>,
    /// The voice lane's store and database, closed as the host closes it after the session.
    store: Option<fakes::FakeStore>,
    /// What the fakes have done and not yet been read. Drained after every input; see [`Record`]
    /// for why reading it later still yields the order things happened in.
    records: mpsc::UnboundedReceiver<Record>,
    /// The way to reach a live fake, owned outright and shared with no one. A delivery is a
    /// lookup in this map and a call, so there is no lock on that path to re-enter.
    controls: BTreeMap<Slot, Control>,
    /// The `override_options` layers the host holds, oldest first, by the id the session minted.
    overrides: Vec<u64>,
    /// Which output each flushed segment sits on, kept after the segment is gone.
    segments: BTreeMap<u32, u32>,
    /// Journal steps the probe stamped, waiting to be absorbed. The probe runs under the
    /// driver's own lock and cannot touch the recording, so it parks them here.
    steps: Arc<Mutex<Vec<(Observed, Step)>>>,
    /// What the host received, with the live handle of a speech it was told of, taken as the
    /// event was delivered (the session lets it go once nothing names it).
    host: HostJournal,
    /// Set by the observer when it stopped a step for journaling without bound at one instant.
    livelock: Arc<Mutex<Option<String>>>,
    pending: BTreeMap<u32, Pending>,
    /// The `VoiceTask.run()` answer each tool body awaits, by call id: answering the tool first
    /// drops the await, as a cancelled body does.
    task_awaits: BTreeMap<String, u32>,
    /// The DTMF collection's answer the host awaits, if any.
    dtmf_await: Option<u32>,
    /// Hooks the host never answers (`Site::Hook`), by `nth`.
    hung_hooks: BTreeSet<u32>,
    /// The host waits armed, as tasks of the world's that stamp when they resolve.
    probes: probe::Probes,
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
    /// The CLI on the other end of the console's socket.
    console: Option<fakes::FakeCli>,
    /// The next console request's id.
    next_request: u64,
    /// The background audio player, as `background_audio.py` runs it beside the session.
    background: Option<background::Background>,
    /// The avatar worker the agent's voice streams to, in its own room.
    avatar: Option<Arc<fakes::FakeAvatarWorker>>,
    /// The agent's sink streaming to it, whose give-up timer after a clear is a deadline.
    avatar_sink: Option<Arc<lk_agents_runtime::avatar::output::DataStreamAudioOutput>>,
    /// The next inline task id to mint. Per-`Run` and not process-global, so two runs of one case
    /// journal the same `task_0001`, `task_0002`, ... -- the determinism suite reads those ids.
    refs: Arc<dyn RefRegistry>,
    /// The session's providers, given again by a handoff whose agent overrides nodes.
    legs: start::Legs,
    pub setup: Setup,
    pub record: Recording,
    /// The most steps the run took at one virtual instant: what `NO_PROGRESS_STEPS` is derived
    /// from.
    pub busiest_instant: u32,
    /// A task the run awaited could never finish: nothing was pending, no wake was armed and no
    /// task could run when its deadline passed. The evidence of a deadlock.
    deadlock: Option<String>,
    /// Retries the request stream armed for a speech's chat request, with the context the request
    /// is reopened with and when its backoff ends (ms), until the reopened request is seen or the
    /// speech's request finishes.
    retried_chats: Vec<(u64, Vec<String>, u64)>,
}

impl Run {
    /// The order of every step dispatched so far, folded: a seed replays it exactly.
    pub fn dispatch_order(&self) -> u64 {
        self.queue.order()
    }

    /// An acquire's lease take applied with its answer still owed (`FakeStore::take_unanswered`).
    pub fn lease_take_unanswered(&self) -> bool {
        self.store.as_ref().is_some_and(fakes::FakeStore::take_unanswered)
    }
}

use lk_agents_runtime::room_io::FixedRoom;
