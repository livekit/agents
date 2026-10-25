//! Everything the test can send the driver, declared once.
//!
//! One table, one line per input, saying what it carries, what tends to come of it, and which
//! production command it becomes. The `Input` enum, availability, how one is made, and the proof
//! that no production leaf was forgotten are all generated from it.
//!
//! **Dependencies live in the field types.** A field typed `Ref<Filler>` needs a filler something
//! earlier opened, and comes out `None` when none is live, so the input is simply unavailable and
//! no availability rule is written twice. `Ref` counts from the run's own history, so a recorded
//! case resolves the same way it was made and a stale reference is exactly a replay divergence.
use std::marker::PhantomData;

use fastrand::Rng;

use lk_agents_abi::{
    Vocabulary,
    utils::vocabulary::{Coverage, Covers},
};
use lk_agents_voice::agent_session::{
    IoKind, activity_command_vocabulary as activity, assert_command_is_covered,
    command_vocabulary as command, speech_command_vocabulary as speech,
    turn_command_vocabulary as turn,
};

use crate::record::{Kind, Live};

// ==============================================================================================
// References: the only way an input names something the run already has
// ==============================================================================================

/// The n-th resource of kind `K` this run opened, counting from zero.
///
/// An index rather than an identity so that a case is replayable: the driver mints a different
/// `SpeechId` every run, but "the third speech this run opened" is the same thing both times.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(transparent, bound = "")]
pub struct Ref<K> {
    pub nth: u32,
    #[serde(skip)]
    kind: PhantomData<fn() -> K>,
}

impl<K> Ref<K> {
    pub fn new(nth: u32) -> Self {
        Ref { nth, kind: PhantomData }
    }
}

impl<K> Clone for Ref<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for Ref<K> {}
impl<K> PartialEq for Ref<K> {
    fn eq(&self, other: &Self) -> bool {
        self.nth == other.nth
    }
}
impl<K: Resource> std::fmt::Debug for Ref<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}#{}", K::KIND, self.nth)
    }
}

/// The n-th resource of kind `K` this run opened and has since closed: what a late answer names.
///
/// A host answers a hook after the driver stopped waiting for it -- the bound ran out, a newer
/// turn superseded it, the close abandoned it -- and the driver has to take that in its stride.
/// Drawn from what is gone, never from what is live, so the row means what it says.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(transparent, bound = "")]
pub struct Gone<K> {
    pub nth: u32,
    #[serde(skip)]
    kind: PhantomData<fn() -> K>,
}

impl<K> Gone<K> {
    pub fn new(nth: u32) -> Self {
        Gone { nth, kind: PhantomData }
    }
}

impl<K> Clone for Gone<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for Gone<K> {}
impl<K> PartialEq for Gone<K> {
    fn eq(&self, other: &Self) -> bool {
        self.nth == other.nth
    }
}
impl<K: Resource> std::fmt::Debug for Gone<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gone {:?}#{}", K::KIND, self.nth)
    }
}

impl<K: Resource> Make for Gone<K> {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        let live = s.live().of(K::KIND);
        let gone: Vec<u32> = (0..s.live().opened(K::KIND)).filter(|n| !live.contains(n)).collect();
        (!gone.is_empty()).then(|| Gone::new(*s.pick(&gone)))
    }
}

/// Names nothing live, by construction.
impl<K> Referenced for Gone<K> {}

/// A kind of thing a run can open. The marker types are what `Ref` is parameterised by.
pub trait Resource {
    const KIND: Kind;
}

macro_rules! resources {
    ($($name:ident),* $(,)?) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        pub struct $name;
        impl Resource for $name {
            const KIND: Kind = Kind::$name;
        }
    )*};
}

resources!(
    Speech,
    Answer,
    Hold,
    Filler,
    Hook,
    Chat,
    Tts,
    Stt,
    Vad,
    Eot,
    Prediction,
    Interruption,
    Mic,
    Utterance,
    Tool,
    Segment,
    Stream,
    Output,
    Video,
    Realtime,
    Generation
);

// ==============================================================================================
// Making a value
// ==============================================================================================
//
// The words, the randomness and the making all live here rather than with the search,
// because they are what an input is made of. The search decides which runs to do; it should not
// be the thing this table has to import from.

// The words and durations a case is built from.
//
// Realistic on purpose: sentence-length agent turns, disfluent user turns, and unnormalised text,
// because normalisation, tokenisation and chunk boundaries are where a driver goes wrong.

/// What the agent says.
pub const AGENT_TURNS: &[&str] = &[
    "Sure, I can help with that.",
    "Let me pull up your account -- one moment.",
    "I found three appointments: Tuesday at 9, Wednesday at 2:30, and Friday at 11 a.m. Which works best?",
    "Okay. I've cancelled the 2:30 and moved you to Friday. You'll get a confirmation by text.",
    "I'm sorry, I didn't catch that. Could you say the last four digits again?",
    "Your balance is $1,204.55 as of today, and the next payment of $87.20 is due on 03/14.",
    "That's outside what I can do here, so I'll pass you to someone who can.",
    // Unnormalised on purpose: abbreviations, symbols, numbers and case the driver must
    // decide how to chunk and speak.
    "Dr. Alvarez is at 1600 Pennsylvania Ave. NW, Ste. 200 -- call (202) 555-0147 ext. 12.",
    "Ref #A1B2-C3D4; ETA ~15 min; cost is EUR 12.50 (approx. USD 13.60).",
    "Yes... no, wait -- actually, let me check that again.",
    "",
];

/// What the caller says.
pub const USER_TURNS: &[&str] = &[
    "yeah hi um I'm calling about my order",
    "can you hear me",
    "no no not that one, the other one",
    "uh let me think... okay so it was last tuesday I think? maybe wednesday",
    "WAIT stop stop stop",
    "my number is five five five, oh one four seven",
    "what",
    "okay thanks bye",
    "",
];

/// Delays a case can ask for, in milliseconds. Spread across the scales the driver reasons at:
/// inside one audio frame, around the end-of-turn and interruption thresholds, and long enough
/// that a deadline fires.
pub const DELAYS_MS: &[u64] = &[0, 10, 20, 50, 120, 200, 500, 700, 1_000, 2_500, 5_000, 15_000];

/// Names for tools, agents and participants.
pub const NAMES: &[&str] =
    &["lookup_order", "transfer", "billing_agent", "caller", "sip_1", "agent-2"];

/// Samples values, knowing what the run still holds.
///
/// The two halves belong together: almost every choice here is a choice *among what is live*, and
/// the ones that are not still need the same randomness. The seed lives outside, in the campaign,
/// because it spans a whole run while the live set changes with every input.
pub struct Sampler<'a> {
    live: &'a Live,
    rng: &'a mut Rng,
}

impl<'a> Sampler<'a> {
    pub fn new(live: &'a Live, rng: &'a mut Rng) -> Self {
        Sampler { live, rng }
    }

    /// What the run still holds, which is what a reference can point at.
    pub fn live(&self) -> &Live {
        self.live
    }

    /// An index into `0..n`. `n == 0` gives 0, which no caller indexes with.
    pub fn index(&mut self, n: usize) -> usize {
        if n <= 1 { 0 } else { self.rng.usize(0..n) }
    }

    /// True this percent of the time.
    pub fn chance(&mut self, percent: u8) -> bool {
        self.rng.u8(0..100) < percent
    }

    /// One of these. Panics on an empty slice, which no caller passes.
    pub fn pick<'b, T>(&mut self, of: &'b [T]) -> &'b T {
        &of[self.index(of.len())]
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        self.rng.shuffle(items);
    }
}

/// How one value of this type is made, when an input that carries it is being sampled.
///
/// `None` means there is nothing to point at, which is how an input becomes unavailable without
/// anyone writing an availability rule for it.
pub trait Make: Sized {
    fn make(s: &mut Sampler<'_>) -> Option<Self>;
}

impl<K: Resource> Make for Ref<K> {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        let live = s.live().of(K::KIND);
        (!live.is_empty()).then(|| Ref::new(*s.pick(&live)))
    }
}

impl Make for bool {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(s.chance(50))
    }
}

impl<T: Make> Make for Option<T> {
    /// Given as often as not, because a `None` argument is its own case: the reference treats it
    /// as "not given", which is not the same as a default.
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        if s.chance(50) { T::make(s).map(Some) } else { Some(None) }
    }
}

impl<T: Make> Make for Vec<T> {
    /// Nothing, one, or a few: an empty list is a case of its own.
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        let count = s.index(4);
        (0..count).map(|_| T::make(s)).collect()
    }
}

/// Text an agent would say: replies, instructions, what a `say` speaks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AgentText(pub String);

/// Text a caller would say: transcripts, `user_input`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UserText(pub String);

/// A duration in milliseconds, from the domain a case is allowed to ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Millis(pub u64);

/// A small count: how many frames, how many retries, how many words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Count(pub u32);

impl Make for AgentText {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(AgentText(s.pick(AGENT_TURNS).to_string()))
    }
}
impl Make for UserText {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(UserText(s.pick(USER_TURNS).to_string()))
    }
}
impl Make for Millis {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(Millis(*s.pick(DELAYS_MS)))
    }
}
impl Make for Count {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(Count(s.index(4) as u32))
    }
}

/// A detector's confidence, in percent. Sampled from a spread that straddles the thresholds a
/// driver decides on, so hysteresis and flicker are reached rather than hoped for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Confidence(pub u8);

/// The arguments a model hands a tool, as the JSON text on the wire. Drawn from the shapes the
/// driver's own tools take -- the answering-machine classifier's `save_prediction(label)` and
/// `postpone_termination(seconds)` -- and the empty object every host tool accepts here.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolArgs(pub String);

impl Make for ToolArgs {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(ToolArgs(
            (*s.pick(&[
                "{}",
                r#"{"label":"human"}"#,
                r#"{"label":"machine-vm"}"#,
                r#"{"seconds":3}"#,
            ]))
            .to_owned(),
        ))
    }
}

/// Who a recogniser says was speaking, when it diarises: a small id, the same person keeping the
/// same one. `None` on an event is a recogniser that does not say.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Speaker(pub u8);

impl Make for Speaker {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(Speaker(*s.pick(&[0, 1, 2])))
    }
}

impl Make for Confidence {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(Confidence(*s.pick(&[0, 5, 20, 45, 50, 55, 70, 90, 100])))
    }
}
impl Make for String {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(s.pick(NAMES).to_string())
    }
}
impl Make for IoKind {
    fn make(s: &mut Sampler<'_>) -> Option<Self> {
        Some(*s.pick(&[IoKind::AudioInput, IoKind::AudioOutput, IoKind::Transcription]))
    }
}

/// An enum whose variants are all payload-free and equally interesting.
macro_rules! plain_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Vocabulary)]
        pub enum $name { $($variant),* }
        impl Make for $name {
            fn make(s: &mut Sampler<'_>) -> Option<Self> {
                Some(*s.pick(&[$($name::$variant),*]))
            }
        }
    };
}

plain_enum! {
    /// What a `say` speaks: words given up front, a stream the case feeds, or caller audio.
    SayInput { Words, TextStream, Audio }
}
plain_enum! {
    /// What a tool body answers with. `Detached` is `ctx.update()`: the step is answered with a
    /// progress message and the tool keeps running, its real end coming later as a deferred reply.
    /// `AfterIdle` is `ctx.wait_for_idle()` inside the body: the tool asks the agent to go idle,
    /// from its own origin, and returns once it has.
    ToolAnswer { Return, Value, Error, StopResponse, Cancelled, Detached, AfterIdle }
}
plain_enum! {
    /// Something the realtime server did on its own: dropped the socket, came back, sent a delta
    /// for no generation the driver knows, or opened a generation over the one in progress.
    RtEvent { Disconnected, Reconnected, StrayDelta, SecondGeneration, GenerationDone }
}
plain_enum! {
    /// What the host answers a hook call with. Any answer may go to any hook; the driver decides
    /// what a mismatched one means and the journal says so.
    HookAnswer { Returned, Failed, TurnInstructions, TurnStop, Rendered, Filler }
}
plain_enum! {
    /// One field of the live session configuration, changed alone so a finding names it.
    Knob { Detection, MinEndpointing, MaxEndpointing, Interruptions, MinInterruptionWords, UserAway, MaxToolSteps }
}
plain_enum! {
    /// Which turn-taking mode the session runs in.
    TurnMode { Default, Stt, Manual }
}
plain_enum! {
    /// What the IVR navigator is told.
    IvrInput { Start, SeedTranscript, Close }
}
plain_enum! {
    /// What happens in the room the session is in: the link the room io machine reports, and the
    /// room's own two events that name no participant.
    RoomIn { Linked, Unlinked, StateChanged, Closed }
}
plain_enum! {
    /// What `GetDtmfTask`'s event half is told: start collecting, a SIP digit, stop.
    DtmfInput { Start, Digit, Stop }
}
plain_enum! {
    /// A SIP key as the room reports it: digits, the two symbols (`#` is the stop key), and a
    /// string that is no DTMF key at all.
    DtmfKey { One, Two, Nine, Star, Pound, Unknown }
}

impl DtmfKey {
    /// `ev.digit`, raw.
    pub fn digit(self) -> &'static str {
        match self {
            DtmfKey::One => "1",
            DtmfKey::Two => "2",
            DtmfKey::Nine => "9",
            DtmfKey::Star => "*",
            DtmfKey::Pound => "#",
            DtmfKey::Unknown => "E",
        }
    }
}

plain_enum! {
    /// Which models a swap replaces.
    Models { Stt, Llm, Tts, LlmAndTts, All }
}
plain_enum! {
    /// Who is speaking in a history item the host supplies.
    Role { System, User, Assistant }
}

// ==============================================================================================
// The table
// ==============================================================================================

/// Something a field can be a reference to. Only `Ref` is; everything else says so by saying
/// nothing, which is what lets the table find an input's references without inspecting types.
pub trait Referenced {
    fn reference(&self) -> Option<(Kind, u32)> {
        None
    }
}

macro_rules! unreferenced {
    ($($ty:ty),* $(,)?) => {$( impl Referenced for $ty {} )*};
}

unreferenced!(
    bool, AgentText, UserText, Millis, Count, String, IoKind, SayInput, ToolAnswer, HookAnswer,
    Knob, TurnMode, IvrInput, Models, Role, RoomIn, Confidence, ToolArgs, Speaker, RtEvent,
    DtmfInput, DtmfKey
);

impl<K: Resource> Referenced for Ref<K> {
    fn reference(&self) -> Option<(Kind, u32)> {
        Some((K::KIND, self.nth))
    }
}
impl<T: Referenced> Referenced for Option<T> {
    fn reference(&self) -> Option<(Kind, u32)> {
        self.as_ref()?.reference()
    }
}
impl<T: Referenced> Referenced for Vec<T> {
    fn reference(&self) -> Option<(Kind, u32)> {
        self.first()?.reference()
    }
}

/// Declares every input once.
///
/// `Name { fields } opens [kinds] leaf production::Variant;`
///
/// * the fields become the variant's payload, and a `Ref` among them is what the input needs;
/// * `opens` is what tends to appear when it is applied, which is how the sampler reaches a
///   chain rather than waiting to hit its second half by luck. It is a hint, not a promise: the
///   run records what actually opened;
/// * `leaf` names the production command this becomes, and every named leaf is checked against
///   the production vocabulary below, so a new driver command does not compile until it is here;
/// * `never` marks an input the runner emits itself and the sampler must not make.
macro_rules! inputs {
    ($(
        $(#[$meta:meta])*
        $name:ident { $($field:ident : $ty:ty),* $(,)? }
        $(opens [$($opens:ident),*])?
        $(leaf $vocab:ident :: $leaf:ident)?
        $(never $never:literal)?
        ;
    )*) => {
        /// Everything the test can send the driver.
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, Vocabulary)]
        pub enum Input {
            $($(#[$meta])* $name { $($field: $ty),* },)*
        }

        impl Input {
            /// The variant's name, for reports and for the sampler's own bookkeeping.
            pub fn name(&self) -> &'static str {
                match self { $(Input::$name { .. } => stringify!($name),)* }
            }

            /// A call the host program makes: the row becomes a production command (`leaf`).
            /// Everything else is a person, a model, an output, the room or the clock.
            pub fn is_command(&self) -> bool {
                match self { $(Input::$name { .. } => false $(|| stringify!($leaf) != "")?,)* }
            }

            /// What tends to come of sending this input.
            ///
            /// Asserted in the table, so it can be wrong; contrast [`Input::needs`], which is
            /// read off the value and cannot be. The survey reports a row that opens something
            /// it never named.
            pub fn outcomes(&self) -> &'static [Kind] {
                match self { $(Input::$name { .. } => &[$($(Kind::$opens),*)?],)* }
            }

            /// Every resource this input names. Empty means it depends on nothing, so it is
            /// always available; a reference that no longer resolves is a replay divergence.
            pub fn needs(&self) -> Vec<(Kind, u32)> {
                match self {
                    $(Input::$name { $($field),* } => {
                        #[allow(unused_mut)]
                        let mut out = Vec::new();
                        $(if let Some(r) = Referenced::reference($field) { out.push(r); })*
                        out
                    })*
                }
            }

            /// One of each input the run could send right now, payloads filled in. An input whose
            /// references cannot be resolved is simply absent.
            pub fn available(s: &mut Sampler<'_>) -> Vec<Input> {
                let mut out = Vec::new();
                $(
                    let sendable = true $(&& !$never)?;
                    if sendable && let ($(Some($field),)*) = ($(<$ty as Make>::make(s),)*) {
                        out.push(Input::$name { $($field),* });
                    }
                )*
                out
            }
        }

        /// Proof that every production command leaf is in the table. Implementing `Covers` for
        /// each one is what makes a new driver command a compile error here.
        pub struct Table;
        $($(impl Covers<$vocab::$leaf> for Table {
            type With = ();
            fn coverage() -> Coverage<()> {
                Coverage::Covered(())
            }
        })?)*
    };
}

inputs! {
    // --- what the host program calls ----------------------------------------------------------
    /// `say()`: speak these words, no model.
    Say { text: AgentText, input: SayInput, allow: Option<bool>, to_ctx: bool }
        opens [Speech, Stream, Answer] leaf speech::Say;
    /// `generate_reply()`: reply now, with no end of turn behind it.
    GenerateReply { user_input: Option<UserText>, instructions: Option<AgentText>, allow: Option<bool> }
        opens [Speech, Answer] leaf speech::GenerateReply;
    /// `interrupt()`: stop everything being said. `wait` asks for the settle answer.
    Interrupt { force: bool, wait: bool } opens [Answer] leaf speech::Interrupt;
    /// `SpeechHandle.interrupt()`: stop one speech, done or not.
    InterruptSpeech { speech: Ref<Speech>, force: bool } leaf speech::InterruptSpeech;
    SetAllowInterruptions { speech: Ref<Speech>, allow: bool } leaf speech::SetAllowInterruptions;
    /// `commit_user_turn()`: flush the recogniser and commit whatever the turn holds.
    CommitUserTurn { transcript_timeout: Millis, stt_flush: Millis, skip_reply: bool }
        opens [Answer] leaf speech::CommitUserTurn;
    /// `wait_for_idle()`. Naming a hold is how its own holder passes through.
    WaitForIdle { hold: Option<Ref<Hold>> } opens [Answer] leaf speech::WaitForIdle;
    WaitForIdleAndHold { } opens [Hold, Answer] leaf speech::WaitForIdleAndHold;
    ReleaseIdleHold { hold: Ref<Hold> } leaf speech::ReleaseIdleHold;
    ClaimUserTurn { } leaf speech::ClaimUserTurn;
    ReleaseUserTurn { } leaf speech::ReleaseUserTurn;
    /// `AgentTask.complete(result)` from inside a speech's tool.
    SetRunOutput { speech: Ref<Speech>, ok: bool } leaf speech::SetRunOutput;
    /// A background tool speaking again after its first `ctx.update()`.
    ToolUpdate { tool: Ref<Tool>, progress: Option<AgentText> } leaf speech::ToolUpdate;
    /// `with_filler()` entered: the filler speaks while the tool runs.
    FillerStart { tool: Ref<Tool>, speech: Ref<Speech>, delay: Millis, interval: Option<Millis>, text: Option<AgentText> }
        opens [Filler] leaf speech::FillerStart;
    FillerStop { filler: Ref<Filler> } leaf speech::FillerStop;
    PauseAuthorization { } leaf speech::PauseAuthorization;
    ResumeAuthorization { } leaf speech::ResumeAuthorization;
    CommitTurn { skip_reply: bool } leaf turn::Commit;
    ClearTurn { } leaf turn::Clear;
    /// One live option changed, alone, so a finding names the option that mattered.
    Configure { knob: Knob, mode: TurnMode, ms: Millis, on: bool, words: Count } leaf turn::Configure;
    /// A handoff to another agent, with the same fakes behind it.
    Handoff { agent: String, instructions: AgentText, tools: Vec<String>, pause_previous: bool, resume: bool }
        opens [Answer] leaf activity::Update;
    SetSpeakerFilter { enabled: bool } leaf activity::SetSpeakerFilter;
    UpdateInstructions { text: AgentText } leaf activity::UpdateInstructions;
    UpdateTools { tools: Vec<String> } leaf activity::UpdateTools;
    SetAudioInput { attach: bool } leaf activity::SetAudioInput;
    SetVideoInput { attach: bool } leaf activity::SetVideoInput;
    SetIoEnabled { leg: IoKind, enabled: bool } leaf activity::SetIoEnabled;
    UpdateChatCtx { items: Vec<Role>, text: AgentText } leaf activity::UpdateChatCtx;
    SetAudioOutput { attach: bool } opens [Output] leaf activity::SetAudioOutput;
    SetTranscriptionOutput { attach: bool } opens [Output] leaf activity::SetTranscriptionOutput;
    UpdateModels { models: Models } leaf activity::UpdateModels;
    /// `drain()`: the frame's `on_exit`, then scheduling pauses and the session stays open, idle.
    /// `wait` asks for the answer.
    Drain { wait: bool } opens [Answer] leaf activity::Drain;
    /// One camera frame, as the host-minted key the driver moves without inspecting.
    PushVideo { } leaf command::Video;
    Close { drain: bool } leaf command::Close;
    /// The answer to a hook call the driver is waiting on.
    HookDone { hook: Ref<Hook>, answer: HookAnswer, text: AgentText } leaf command::HookDone;
    /// The host answers a hook the driver has stopped waiting for.
    HookDoneLate { hook: Gone<Hook>, answer: HookAnswer, text: AgentText };
    HostMetrics { } leaf command::HostMetrics;
    Ivr { input: IvrInput, silence: Millis, text: UserText } leaf command::Ivr;
    SetStaticKeyterms { terms: Vec<String> } leaf command::SetStaticKeyterms;
    /// `GetDtmfTask`'s event half: `Start` a collection of `digits + 1` keys (`#` stops,
    /// `confirm` reads the sequence back, `timeout` is the per-digit debounce), forward the SIP
    /// `key`, or `Stop`. `Start` opens the collection's answer.
    Dtmf { input: DtmfInput, key: DtmfKey, digits: Count, confirm: bool, timeout: Millis }
        opens [Answer] leaf command::Dtmf;
    /// An inline `AgentTask` entered from a tool body or a hook.
    Task { agent: String, speech: Option<Ref<Speech>> } leaf command::Task;
    Recorder { attach: bool } leaf command::Recorder;
    Run { user_input: Option<UserText>, expects_output: bool, retries: Count } leaf command::Run;

    // --- what the case feeds into a stream a call opened --------------------------------------
    FeedText { stream: Ref<Stream>, text: AgentText };
    FeedAudio { stream: Ref<Stream>, chunks: Count };
    FeedEnd { stream: Ref<Stream> };

    // --- what a controlled dependency answers -------------------------------------------------
    LlmText { chat: Ref<Chat>, text: AgentText };
    LlmToolCall { chat: Ref<Chat>, name: String, args: ToolArgs } opens [Tool];
    LlmEnd { chat: Ref<Chat> };
    LlmFail { chat: Ref<Chat> };
    /// Queue what the product pushes to a synthesis stream instead of producing it.
    TtsHold { tts: Ref<Tts> };
    TtsRelease { tts: Ref<Tts> };
    TtsFail { tts: Ref<Tts> };
    /// The recogniser fails mid-stream; the stream is spent.
    SttFail { stt: Ref<Stt> };
    /// The person speaks into the microphone: this many 10 ms frames, forwarded by the driver to
    /// every provider listening. What they contained is for VAD and STT to report.
    MicFrames { mic: Ref<Mic>, chunks: Count };
    /// One detector tick: confidence, verdict, and the duration of what it is reporting -- of
    /// the speech when `speaking`, of the silence otherwise. A real detector restates that total on
    /// every inference, and the driver decides a interruption from exactly this number.
    VadInference { vad: Ref<Vad>, probability: Confidence, speaking: bool, duration: Millis };
    VadStart { vad: Ref<Vad>, speech_ms: Millis };
    VadEnd { vad: Ref<Vad>, silence_ms: Millis };
    /// The predictor's verdict on a prediction the driver asked for: how likely the turn is over.
    /// The predictor's answer: how likely the turn is over, and -- from a model that scores it --
    /// how fitting a backchannel would be at this pause.
    EotAnswer { prediction: Ref<Prediction>, probability: Confidence, backchannel: Option<Confidence> };
    /// The predictor fails that request instead.
    EotFail { prediction: Ref<Prediction> };
    /// The interruption detector's judgement of the current overlap: a real interruption, or not.
    OverlapVerdict { port: Ref<Interruption>, is_interruption: bool, agent_ended: bool };
    OverlapFail { port: Ref<Interruption> };
    /// The caller starts speaking on a recognition stream.
    SttStart { stt: Ref<Stt> } opens [Utterance];
    SttInterim { utterance: Ref<Utterance>, text: UserText, speaker: Option<Speaker> };
    SttPreflight { utterance: Ref<Utterance>, text: UserText, speaker: Option<Speaker> };
    SttFinal { utterance: Ref<Utterance>, text: UserText, speaker: Option<Speaker> };
    SttEnd { utterance: Ref<Utterance> };
    ToolAnswered { tool: Ref<Tool>, answer: ToolAnswer, text: AgentText };
    /// The speaker began emitting this segment: the first of its audio is leaving for the
    /// listener. Legal from the segment's first frame; the driver marks the agent as speaking here.
    PlayoutStarted { segment: Ref<Segment> };
    /// The voice provider goes down, or comes back: while down, every synthesis the driver opens
    /// fails at the open. A synthesis lives inside one input's application -- text in, audio out,
    /// closed -- so a failure has to be standing to land on one; this is that standing failure.
    TtsOutage { down: bool };
    /// The speaker finished this segment. Refused by the output while the segment is still taking
    /// frames: nothing has finished playing before it was flushed.
    PlayoutFinished { segment: Ref<Segment> };
    /// The speaker reports a stretch of this segment as played: what a real output says at a
    /// discontinuity, and what the driver's played position moves on. Refused before the start.
    PlayoutProgressed { segment: Ref<Segment>, ms: Millis };
    /// The output threw its playback count away, as a republished track does: every segment it
    /// owed is reported cut, and the driver hears a reset it did not ask for.
    SinkReset { output: Ref<Output> };
    /// Frames from a camera the case attached.
    VideoFrames { video: Ref<Video>, frames: Count };
    /// The realtime server did something of its own accord on this socket. `GenerationDone`
    /// ends its oldest open turn: a server takes time to finish speaking, and until the case says
    /// it has, the turn is open -- which is when a second generation over it is a real event.
    RealtimeEvent { socket: Ref<Realtime>, event: RtEvent };
    /// Something the room did. Refused by a session with no room legs, which is most of them.
    Room { event: RoomIn };

    // --- the clock, and waiting for something the driver owes ---------------------------------
    /// Virtual time passes with nobody acting.
    Advance { ms: Millis };
    /// Wait for one answer the driver owes a caller.
    AwaitAnswer { answer: Ref<Answer> };
    /// Wait for one speech to finish.
    AwaitSpeech { speech: Ref<Speech> };
    /// Wait for the driver loop to end.
    AwaitExit { };

    // --- the runner's own phase boundaries, never sampled --------------------------------------
    /// The case stops adding work; from here the runner releases what is pending.
    Release { } never true;
    /// Nothing is left to release; from here outcomes are only awaited.
    Settle { } never true;
}

/// Production leaves the table deliberately does not carry, each with the reason.
macro_rules! not_sent {
    ($($vocab:ident :: $leaf:ident => $why:literal;)*) => {$(
        impl Covers<$vocab::$leaf> for Table {
            type With = ();
            fn coverage() -> Coverage<()> {
                Coverage::Skipped($why)
            }
        }
    )*};
}

not_sent! {
    command::Speech => "a family, not a leaf: its leaves are in the table";
    command::Turn => "a family, not a leaf: its leaves are in the table";
    command::Activity => "a family, not a leaf: its leaves are in the table";
}

/// Compiles only while every production command leaf is either in the table or excused above.
pub fn every_command_is_sendable() {
    assert_command_is_covered::<Table, ()>();
}
