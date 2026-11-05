//! Typed realtime inputs decoded by the runtime before dispatch.

/// The `previous_item_id` that places an item first in the conversation.
pub const ROOT_ITEM: &str = "root";

/// How the message of a cancel that raced a finished response starts.
pub const CANCELLATION_FAILED: &str = "Cancellation failed";

/// Server error codes the session acts on.
pub mod error_code {
    /// A create for an item the server already holds.
    pub const DUPLICATE_ITEM_ID: &str = "item_create_duplicate_item_id";
    /// A commit of an input buffer the server's own VAD already emptied.
    pub const COMMIT_EMPTY: &str = "input_audio_buffer_commit_empty";
}

/// What a `response.output_item.added` announces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputItemKind {
    Message,
    FunctionCall,
    Other,
}

/// One decoded server message. Covers xAI's extra types for every dialect.
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum Incoming {
    SpeechStarted {
        item_id: Option<String>,
    },
    SpeechStopped,
    ResponseCreated {
        response_id: String,
        client_event_id: Option<String>,
    },
    OutputItemAdded {
        item_id: String,
        kind: OutputItemKind,
    },
    /// The part's modalities: a text part is text-only, any other carries audio.
    ContentPartAdded {
        item_id: String,
        modalities: lk_agents_abi::llm::realtime::Modalities,
    },
    ItemAdded {
        previous_item_id: Option<String>,
        item: lk_agents_abi::ChatItem,
    },
    ItemDeleted {
        item_id: String,
    },
    TranscriptionDelta {
        item_id: String,
        content_index: usize,
        delta: String,
    },
    TranscriptionCompleted(TranscriptionCompleted),
    TranscriptionFailed {
        item_id: String,
        content_index: usize,
    },
    /// xAI only: an interim transcript -- the whole transcript so far, re-sent.
    TranscriptionUpdated {
        item_id: String,
        transcript: String,
    },
    TextDelta {
        item_id: String,
        delta: String,
    },
    /// The transcript of audio output. `start_time` only when the provider sent it.
    AudioTranscriptDelta {
        item_id: String,
        delta: String,
        start_time: Option<f64>,
    },
    AudioDelta {
        item_id: String,
        frame: lk_agents_abi::AudioFrame,
    },
    OutputItemDone {
        item: lk_agents_abi::ChatItem,
    },
    ResponseDone(ResponseDone),
    Error(ServerError),
    /// xAI only. The server is about to take a proactive turn.
    IdleTimeoutTriggered,
    SessionCreated {
        model: Option<String>,
    },
    /// An unhandled event, carried so the raw-event passthrough can forward it.
    Unhandled {
        event_type: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionCompleted {
    pub item_id: String,
    pub content_index: usize,
    pub transcript: String,
    /// Geometric mean of `exp(logprob)`, or `None` when the provider sent no logprobs.
    pub confidence: Option<f64>,
    /// xAI sends `status: "in_progress"` on a transcript it will revise. Absent on OpenAI.
    pub in_progress: bool,
    pub event_id: String,
    /// The transcription's billed usage, when the provider reports it.
    pub usage: Option<TranscriptionUsage>,
}

/// `UsageTranscriptTextUsageTokens | UsageTranscriptTextUsageDuration`, told apart by `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionUsage {
    Tokens(lk_agents_abi::usage::TokenUsage),
    Duration { seconds: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponseDone {
    pub response_id: String,
    pub status: lk_agents_abi::llm::realtime::GenerationStatus,
    pub error_type: Option<String>,
    pub error_code: Option<String>,
    pub usage: lk_agents_abi::usage::TokenUsage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServerError {
    pub event_id: Option<String>,
    pub code: Option<String>,
    pub type_: Option<String>,
    pub message: String,
}
