//! Answering-machine detection (AMD): the protocol between the session's reply gate and an AMD
//! run (`docs/proposals/amd.md`, "The protocol").
//!
//! The session and AMD talk only through effects and inputs, each direction FIFO:
//!
//! - [`AmdObservation`]: produced by the session's reply gate
//!   (`agent_session::machine::activity::amd_gate`) while a run is attached, as
//!   [`VoiceEffect::Amd`](crate::agent_session::effects::VoiceEffect::Amd). Handled by the AMD
//!   executor in `lk-agents-runtime`.
//! - [`AmdAnswer`]: produced by the AMD machine, delivered to the session as
//!   [`VoiceEvent::Amd`](crate::agent_session::VoiceEvent::Amd). The gate owns the state it
//!   changes.
//!
//! Attaching a run is a session command, [`AmdCommand::Attach`].

use crate::agent_session::RequestId;
use crate::agent_session::machine::activity::scheduling::{SpeechId, SpeechKind};
use crate::audio_recognition::TurnId;
use lk_agents_abi::Ts;
use std::time::Duration;

/// One AMD run. A run attaches once; only one is attached to a session at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AmdRunId(pub u64);

/// What the session tells the attached run. Each carries the time it describes, so AMD does not
/// depend on when it receives them.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum AmdObservation {
    /// A turn committed. The session already holds authorization for it.
    TurnCommitted {
        turn: TurnId,
        transcript: String,
        eot_delay: Option<Duration>,
        skip_reply: bool,
        /// The participant was still speaking at commit (spec §7.1 speech duration).
        speaking: bool,
        /// The agent that was running when the turn committed.
        agent: String,
        at: Ts,
    },
    /// The turn left the activity by any path, including local supersession.
    TurnEnded { turn: TurnId, how: TurnEnd },
    /// A reply exists for the turn: created, adopted from a preemptive generation, or rebuilt
    /// after a correction.
    ReplyCreated { turn: TurnId, speech: SpeechId },
    /// A speech's life. At attach the session also reports each speech already queued or
    /// playing.
    Speech { speech: SpeechId, event: SpeechObservation, at: Ts },
    /// Whether the agent is busy for the idle timer: a held or playing speech, a paused speech,
    /// thinking, a pending false interruption, or a turn waiting on its hook or decision. Sent at
    /// attach, then on change.
    AgentBusy { busy: bool, at: Ts },
    /// The session entered or left `Hold::Silence` for a turn.
    SilenceHold { turn: TurnId, held: bool, at: Ts },
    /// Participant speech boundaries and session STT finals.
    UserSpeech(UserSpeechObservation),
    /// Executed tool calls, for DTMF in classifier history (spec §11.1).
    ToolsExecuted { calls: Vec<ToolCallRecord> },
    /// The session applied `Detach` and removed the gate.
    Detached { run: AmdRunId },
}

/// How a turn committed under the gate left the activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum TurnEnd {
    /// An allowed decision made (or adopted) its reply; `ReplyCreated` came before.
    Replied,
    /// The decision denied it: its words were committed with no reply.
    Denied,
    /// `on_user_turn_completed` raised `StopResponse` or failed.
    Stopped,
    /// The turn asked for no reply.
    SkipReply,
    /// Kept for the arriving agent inside a handoff's window.
    HeldForHandoff,
    /// A newer turn committed first: by `supersede_turn` behind a hook, or by the gate's local
    /// supersession of a parked turn or a held reply.
    Superseded,
    /// Dropped behind an uninterruptible speech, or its reply was refused by scheduling.
    Dropped,
    /// The session closed before the turn finished.
    Abandoned,
}

/// One speech's life, as AMD's classifier history and voicemail record need it.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum SpeechObservation {
    /// Created, or found queued or playing at attach (`playing`).
    Created { kind: SpeechKind, add_to_chat_ctx: bool, playing: bool },
    /// An assistant message of the speech: its words and whether it was cut.
    ItemAdded { text: Option<String>, interrupted: bool },
    /// A step's first frame went out: playout advanced.
    PlayoutAdvanced,
    /// The speech ended.
    Done { interrupted: bool, failed: bool },
}

/// Participant speech, dated to its boundary rather than to when the session saw it.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum UserSpeechObservation {
    /// The VAD (or STT) onset, back-dated as the session's user state is.
    Started { at: Ts },
    /// The end of speech, back-dated by the silence already elapsed.
    Ended { at: Ts },
    /// A non-empty session STT final.
    Final { text: String, at: Ts },
}

/// One executed tool call and its output, if it produced one.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallRecord {
    pub call: lk_agents_abi::FunctionCall,
    pub output: Option<lk_agents_abi::FunctionCallOutput>,
}

/// What an AMD run tells the session. Every answer names its run; the session drops one for
/// another run.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum AmdAnswer {
    /// Listening starts (spec §5.2). Applying it is the activation instant: the session stops
    /// discarding participant audio there. The opening, if any, comes with it and is checked at
    /// the same point.
    Activated { run: AmdRunId, opening: Option<Opening> },
    /// The decision for a committed turn. It may arrive before or after the customer hook
    /// returns.
    ReplyDecided { run: AmdRunId, turn: TurnId, decision: ReplyDecision },
    /// The transcript for a realtime model without user transcription (spec §7.2).
    InsertUserItem { run: AmdRunId, turn: TurnId, text: String },
    /// AMD completed. `pending` settles every turn committed under the gate whose decision is
    /// not applied yet.
    Detach { run: AmdRunId, pending: ReplyDecision, cancel: DetachCancel },
}

/// Whether and how the agent replies to one turn. Policy only: live facts (speaking, silence,
/// the latest turn) stay in the session.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReplyDecision {
    pub allow: bool,
    /// Stage instructions for this reply and its tool follow-ups. Never saved in history.
    pub instructions: Option<String>,
    /// AMD may add the built-in `send_dtmf_events` to this reply (`machine-ivr`, spec §11.1).
    pub dtmf: bool,
    /// Silence the participant must keep before the reply plays (spec §10.3). The session times
    /// it.
    pub silence: Option<Duration>,
}

impl ReplyDecision {
    /// No reply.
    pub fn deny() -> Self {
        ReplyDecision::default()
    }

    /// A reply under the agent's own instructions, at once.
    pub fn allow() -> Self {
        ReplyDecision { allow: true, ..ReplyDecision::default() }
    }

    /// Whether the reply's request differs from the one a plain reply would send: then it never
    /// adopts the preemptive generation.
    pub fn changes_request(&self) -> bool {
        self.instructions.is_some() || self.dtmf
    }
}

/// What the agent says first, before any prediction allows a reply (spec §10.6). It bypasses the
/// hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Opening {
    /// Said as written (`opening_text`).
    Text(String),
    /// Generated with no tools (`opening_instructions`).
    Instructions(String),
}

/// What `Detach` cancels (spec §12.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum DetachCancel {
    /// Replies stay allowed: keep every speech, including the preemptive generation.
    Keep,
    /// `machine-unavailable` or a stage the application handles: cancel every speech with no
    /// playout yet.
    ForbidReplies,
}

/// Session commands for AMD. Producer: the host (step 5) or a test; handler: the session's gate.
#[derive(Debug, lk_agents_abi::Vocabulary)]
#[vocabulary(kind = AmdCommandKind)]
pub enum AmdCommand {
    /// Install the gate for `run`, hold authorization, and set the attached rules. Answered with
    /// [`Answer::AmdAttach`](crate::agent_session::Answer::AmdAttach).
    Attach { run: AmdRunId, call: Option<RequestId> },
}

/// Why the session refused to attach a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, lk_agents_abi::Vocabulary)]
pub enum AmdAttachRefused {
    #[error("amd is already active")]
    AlreadyAttached,
    #[error("wait for the agent handoff to finish before entering AMD")]
    HandoffInFlight,
    #[error("finish the AgentTask before entering AMD")]
    TaskRunning,
    /// Realtime commit paths are not gated yet (implementation step 4).
    #[error("amd does not support realtime models yet")]
    Realtime,
    #[error("the session is closing")]
    Closing,
}
