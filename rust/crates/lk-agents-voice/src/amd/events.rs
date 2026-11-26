//! AMD's categories and public notifications (`voice/amd/events.py`).
//!
//! Producer: the AMD machine ([`Amd`](super::machine::Amd)), as
//! [`AmdEffect::Emit`](super::effects::AmdEffect::Emit). Handler: the AMD executor, which hands
//! them to the host (`amd_prediction`, `amd_menu_observed`, `amd_completed`).

use std::time::Duration;

/// A per-turn prediction or a stage (spec §4.4). `uncertain` and `wait` keep the stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    Human,
    MachineIvr,
    MachineScreening,
    MachineVm,
    MachineUnavailable,
    /// Speech that asks nothing; the stage stays and the reply is skipped.
    Wait,
    /// Not enough evidence; the stage stays and a reply is allowed.
    Uncertain,
}

impl Category {
    /// Every category, in declaration order.
    pub const ALL: [Category; 7] = [
        Category::Human,
        Category::MachineIvr,
        Category::MachineScreening,
        Category::MachineVm,
        Category::MachineUnavailable,
        Category::Wait,
        Category::Uncertain,
    ];

    /// The wire value (`AMDCategory.value`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Category::Human => "human",
            Category::MachineIvr => "machine-ivr",
            Category::MachineScreening => "machine-screening",
            Category::MachineVm => "machine-vm",
            Category::MachineUnavailable => "machine-unavailable",
            Category::Wait => "wait",
            Category::Uncertain => "uncertain",
        }
    }

    /// The category a classifier answered, or `None` for any other value.
    pub fn from_wire(value: &str) -> Option<Category> {
        Category::ALL.into_iter().find(|category| category.as_str() == value)
    }

    /// The stage this category names; `None` for `wait`, which is never a stage.
    pub const fn stage(self) -> Option<Stage> {
        Some(match self {
            Category::Human => Stage::Human,
            Category::MachineIvr => Stage::MachineIvr,
            Category::MachineScreening => Stage::MachineScreening,
            Category::MachineVm => Stage::MachineVm,
            Category::MachineUnavailable => Stage::MachineUnavailable,
            Category::Uncertain => Stage::Uncertain,
            Category::Wait => return None,
        })
    }

    /// `AMDPredictionEvent.is_machine`: the four machine categories.
    pub const fn is_machine(self) -> bool {
        matches!(
            self,
            Category::MachineScreening
                | Category::MachineVm
                | Category::MachineIvr
                | Category::MachineUnavailable
        )
    }
}

/// The call category AMD keeps across turns. Never `wait`; the initial stage is `uncertain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Stage {
    #[default]
    Uncertain,
    Human,
    MachineScreening,
    MachineVm,
    MachineIvr,
    MachineUnavailable,
}

impl Stage {
    pub const fn category(self) -> Category {
        match self {
            Stage::Uncertain => Category::Uncertain,
            Stage::Human => Category::Human,
            Stage::MachineScreening => Category::MachineScreening,
            Stage::MachineVm => Category::MachineVm,
            Stage::MachineIvr => Category::MachineIvr,
            Stage::MachineUnavailable => Category::MachineUnavailable,
        }
    }

    pub const fn as_str(self) -> &'static str {
        self.category().as_str()
    }

    /// `_MACHINE_CATEGORIES`: the machine stages and `machine-unavailable`.
    pub const fn is_machine(self) -> bool {
        self.category().is_machine()
    }
}

/// Why a prediction was published or why the run completed (`AMDReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdReason {
    /// A model prediction was released normally.
    Prediction,
    /// An empty turn reused the latest prediction without a new request.
    Reused,
    /// The turn's inference deadline passed: the prediction is uncertain in the current stage.
    /// Repeated timeouts also complete the run.
    InferenceTimeout,
    /// Classification failed or returned unusable output: the prediction is uncertain in the
    /// current stage.
    InferenceError,
    /// A leg or a host listener failed, ending the run.
    InternalError,
    /// AMD reached `human`, `machine-unavailable` or a stage the application handles.
    Finished,
    /// The limit of consecutive uncertain predictions was reached.
    MaxUncertainTurns,
    /// The time limit since listening started was reached.
    Timeout,
    /// The call stayed silent and inactive for the current stage's idle interval.
    IdleTimeout,
    /// The caller closed AMD or its session before detection completed.
    Cancelled,
    /// The participant or audio track was unavailable during listening setup.
    ParticipantMissing,
    /// The participant disconnected before detection completed.
    ParticipantDisconnected,
    /// The session switched agents during AMD.
    AgentChanged,
}

impl AmdReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            AmdReason::Prediction => "prediction",
            AmdReason::Reused => "reused",
            AmdReason::InferenceTimeout => "inference_timeout",
            AmdReason::InferenceError => "inference_error",
            AmdReason::InternalError => "internal_error",
            AmdReason::Finished => "finished",
            AmdReason::MaxUncertainTurns => "max_uncertain_turns",
            AmdReason::Timeout => "timeout",
            AmdReason::IdleTimeout => "idle_timeout",
            AmdReason::Cancelled => "cancelled",
            AmdReason::ParticipantMissing => "participant_missing",
            AmdReason::ParticipantDisconnected => "participant_disconnected",
            AmdReason::AgentChanged => "agent_changed",
        }
    }
}

/// `amd_prediction` (`AMDPredictionEvent`, spec §4.3): at most one per committed turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// AMD's turn id: local to the run, from 1, and it may skip a turn whose request a newer turn
    /// cancelled.
    pub turn_id: u64,
    /// The prediction for this turn.
    pub category: Category,
    /// The stage after this prediction.
    pub stage: Stage,
    /// `prediction`, `reused`, `inference_timeout` or `inference_error`.
    pub reason: AmdReason,
    /// The session transcript, else AMD's.
    pub transcript: String,
    /// Participant speech in this turn.
    pub speech_duration: Duration,
    /// From commit to this prediction.
    pub delay: Duration,
    /// Model time, for `prediction` events only.
    pub inference_duration: Option<Duration>,
}

impl Prediction {
    pub fn is_human(&self) -> bool {
        self.category == Category::Human
    }

    pub fn is_machine(&self) -> bool {
        self.category.is_machine()
    }
}

/// `amd_completed` (`AMDCompletedEvent`, spec §4.3, §12): fixed when the run completes, emitted
/// once the session detached and the legs closed.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The stage at completion, except that after a played voicemail message a
    /// `machine-unavailable` stage reports the stage before it (spec §12.2).
    pub category: Stage,
    pub reason: AmdReason,
    /// The voicemail message played to its end while AMD was active (spec §10.4).
    pub voicemail_message_played: bool,
    /// The last classified turn's prediction; `None` when the run ended before one.
    pub last_prediction: Option<Prediction>,
}

impl Outcome {
    /// AMD stopped without detecting the call. Timeouts, idle completion and a participant
    /// disconnect are normal outcomes.
    pub fn is_error(&self) -> bool {
        matches!(
            self.reason,
            AmdReason::InternalError
                | AmdReason::ParticipantMissing
                | AmdReason::AgentChanged
                | AmdReason::Cancelled
        )
    }
}

/// One option of an observed IVR menu (`IVRMenuOption`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IvrMenuOption {
    pub label: String,
    pub dtmf: String,
    pub spoken_response: String,
}

/// `amd_menu_observed` (`AMDMenuObservedEvent`, spec §11.2). Informational: it never authorizes
/// an action.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuObserved {
    pub session_id: String,
    pub turn_id: u64,
    pub menu: String,
    pub options: Vec<IvrMenuOption>,
    pub extraction_duration: Duration,
}

/// What AMD tells the host.
#[derive(Debug, Clone, PartialEq)]
pub enum AmdEvent {
    Prediction(Prediction),
    MenuObserved(MenuObserved),
    Completed(Outcome),
}
