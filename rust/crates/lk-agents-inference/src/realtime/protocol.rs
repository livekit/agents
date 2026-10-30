//! Typed realtime inputs decoded by the runtime before dispatch.

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
        item_type: String,
    },
    ContentPartAdded {
        item_id: String,
        part_type: String,
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
