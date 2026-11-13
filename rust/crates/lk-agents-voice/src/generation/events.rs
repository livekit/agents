use lk_agents_abi::errors::ProviderError;

use super::{ReplyPlan, TurnOutcome};
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::transcription::Visible;

/// A segment is one or more chunks followed by `End`. Raw text retains synthesis markup;
/// visible text is suitable for the conversation item. The same event crosses the runtime
/// channel without a second enum or an adapter conversion.
#[derive(Debug)]
pub enum SpeechSegmentEvent {
    Chunk(Visible),
    End,
}

/// Chunks never travel here; only the moments the session machines must hear about, each a
/// cross-machine write the task may not make itself.
#[derive(Debug)]
pub enum GenEvent {
    /// Reservation through release, handled by the session's output-binding owner.
    OutputBinding(crate::agent_session::machine::output::OutputBindingEvent),
    /// Answered with [`Answer::ReplyPlan`](crate::agent_session::Answer::ReplyPlan) under `call`.
    Plan {
        call: crate::agent_session::RequestId,
        plan: Box<ReplyPlan>,
    },
    /// With an audio output the edge is [`GenEvent::PlaybackStarted`]'s. Between them the only
    /// setters of `_agent_speaking`.
    FirstFrame {
        speech: SpeechId,
    },

    Done {
        speech: SpeechId,
        outcome: Box<TurnOutcome>,
        interrupted: bool,
    },
    DeliveredProgress {
        speech: SpeechId,
        item: Box<lk_agents_abi::ChatMessage>,
    },

    /// One of the speech's audio segments stopped playing at `position`: the sink's
    /// `PlaybackFinished`, or the progress it confirmed before its finish report was lost (a lower
    /// bound). `agent_turn` records where an interrupted speech stopped.
    PlayedTo {
        speech: SpeechId,
        position: std::time::Duration,
    },

    /// Forwarded to the bus, ordered against every other `GenEvent`.
    Emit(Box<crate::events::SessionEvent>),

    /// `None` when no call, failure or timeout.
    KeytermAnswer {
        req: u64,
        arguments: Option<String>,
        failed: bool,
        metrics: Option<Box<lk_agents_abi::metrics::base::LlmMetrics>>,
    },

    /// A model step's calls are over (it may have made none). Awaited (`answer`) with the reply's
    /// [`StepNext`](crate::tool_executor::StepNext) when the task goes on from it.
    ToolsFinished {
        speech: SpeechId,
        /// `Source::{ToolCall, ToolOutput}` carry it.
        step: u32,
        end: Box<crate::tool_executor::StepEnd>,
        answer: Option<crate::agent_session::RequestId>,
    },

    ToolStep {
        speech: SpeechId,
        step: u32,
    },

    /// Reported **before** the step's tools run, so the loop can end the current speech per step.
    StepSpoken {
        speech: SpeechId,
        step: u32,
        interrupted: bool,
        has_tool_output: bool,
        tools_running: bool,
        /// Up to the played position; `None` for a tool-only step.
        item: Option<Box<lk_agents_abi::ChatMessage>>,
        /// Where an interrupted step's playout stopped; `None` when nothing was cut while heard.
        cut_at: Option<std::time::Duration>,
    },

    /// The error is **moved**, never formatted: a [`ProviderError::Host`] owns a side-table entry
    /// only the binding can release.
    Failed {
        speech: SpeechId,
        error: Box<ProviderError>,
    },

    Trace(crate::trace::Step),
    /// As the sink reported it: the loop's `speaking` edge (`_on_first_frame`).
    PlaybackStarted {
        speech: SpeechId,
    },
    /// Not reported when the hold ended with the speech's interruption.
    LaunchReleased {
        speech: SpeechId,
    },
    /// A `kind` request of `speech` was sent, to `provider`'s `model`; it counts from here.
    RequestStarted {
        speech: SpeechId,
        kind: lk_agents_core::usage::ModelKind,
        provider: String,
        model: String,
    },
    LlmRequestFinished {
        speech: SpeechId,
        step: u32,
        error: Option<crate::trace::FailureClass>,
    },
    /// The model made a call. Awaited (`go`) with the call's start, or with `Gone` when the
    /// driver drops it unrun.
    ToolCallMade {
        speech: SpeechId,
        step: u32,
        made: Box<crate::tool_executor::MadeCall>,
        go: crate::agent_session::RequestId,
    },
    /// The host bound and validated the call's arguments, or did not prepare them. Awaited with
    /// the call's next move.
    ToolPrepared {
        speech: SpeechId,
        call: crate::tool_executor::CallId,
        identity: Result<
            lk_agents_abi::agent::ArgumentIdentity,
            lk_agents_core::tool_calls::PrepareFailure,
        >,
        go: crate::agent_session::RequestId,
    },
    /// The tool's read-back function answered. Awaited with the call's settlement.
    ReadBackRendered {
        speech: SpeechId,
        call: crate::tool_executor::CallId,
        answer: Result<lk_agents_abi::agent::ReadBackAnswer, lk_agents_abi::errors::HostError>,
        go: crate::agent_session::RequestId,
    },
    /// As the call's answer arrives; `message` is `None` when there is nothing to say. `call`:
    /// its id in the reply's run, once it started.
    ToolCallEnded {
        speech: SpeechId,
        call: Option<crate::tool_executor::CallId>,
        call_id: String,
        /// `None`: detached by `ctx.update()`.
        end: Option<crate::events::ToolEndStatus>,
        message: Option<String>,
        outcome: crate::tool_executor::ToolOutcome,
    },
    /// The reply's agent validated an output candidate; `value` is what it accepted.
    OutputValidated {
        speech: SpeechId,
        validation: lk_agents_abi::agent::run::ValidationId,
        verdict: lk_agents_core::agent_run::OutputVerdict,
        value: Option<lk_agents_abi::HostRef>,
    },
    /// The host rendered a `tool_handling` template; `Ok(None)` keeps the framework's text.
    ToolHandlingTemplateRendered {
        rendering: crate::tool_executor::Rendering,
        text: Result<Option<String>, lk_agents_abi::errors::HostError>,
    },
    /// The speech's audio output raised on a frame: the forwarding stops there, and the speech
    /// is cut and done with `error`.
    OutputFailed {
        speech: SpeechId,
        error: Box<ProviderError>,
    },
    /// `recoverable` when the budget retries it. Then the close policy.
    ProviderFailed {
        speech: SpeechId,
        leg: crate::trace::ProviderKind,
        recoverable: bool,
        error: Box<ProviderError>,
    },
}
