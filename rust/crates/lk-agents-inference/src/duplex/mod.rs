//! `DuplexRealtimeAdapter` (`llm/duplex_adapter.py`): a full-duplex speech model (it listens and
//! speaks at once, handling interruptions itself) run as the session's realtime model. The model's
//! continuous output is gated, cut into generations and given its transcript here.

pub mod gate;
mod host;
pub mod segmenter;
pub mod session;
#[cfg(test)]
mod tests;

use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use lk_agents_abi::{AudioFrame, ChatItem, ToolChoice};

pub use gate::{AdaptiveNoiseGate, AudioGate, FixedGate};
pub use host::GateClass;
pub use lk_agents_abi::llm::duplex::DuplexCapabilities;
pub use segmenter::{AUDIO_TIMEOUT, Fragment, Segmenter, Step};
pub use session::NativeDuplexModel;

/// The realtime capabilities a duplex model runs under: it detects turns and answers them itself,
/// speaks only audio, and owns its output timeline.
pub fn realtime_capabilities(caps: DuplexCapabilities) -> RealtimeCapabilities {
    RealtimeCapabilities {
        message_truncation: false,
        turn_detection: true,
        auto_turn_reply_generation: true,
        user_transcription: caps.user_transcription,
        auto_tool_reply_generation: caps.auto_tool_reply_generation,
        audio_output: true,
        manual_function_calls: false,
        can_disable_turn_detection: false,
        mutable_chat_context: caps.mutable_chat_context,
        mutable_instructions: caps.mutable_instructions,
        mutable_tools: caps.mutable_tools,
        per_response_tool_choice: false,
        supports_say: false,
    }
}

/// What the adapter tells the plugin's session.
#[derive(Debug, Clone)]
pub enum DuplexCommand {
    PushAudio(AudioFrame),
    UpdateInstructions(String),
    /// New chat items only: the adapter owns the context.
    AppendItems(Vec<ChatItem>),
    /// Tool names: the host owns the tool objects.
    UpdateTools(Vec<String>),
    GenerateReply {
        req_id: String,
        instructions: Option<String>,
        tool_choice: Option<ToolChoice>,
    },
    /// The whole startup configuration at once (`_update_session`): a model whose configuration
    /// is immutable once started waits on it before connecting.
    Configure {
        instructions: Option<String>,
        items: Option<Vec<ChatItem>>,
        tools: Vec<String>,
    },
}

/// What the plugin's session reports.
#[derive(Debug, Clone)]
pub enum DuplexEvent {
    /// One frame of the model's output, its own silence included.
    Audio {
        frame: AudioFrame,
        start_ms: Option<i64>,
    },
    Transcript(Fragment),
    /// The output stream ended, or failed with `error`; it cannot be resumed.
    OutputEnded {
        error: Option<String>,
    },
    /// Everything else, as a host realtime session reports it (the caller's turns, transcriptions,
    /// function calls, reconnects, errors, a refused reply).
    Session(Event),
}
