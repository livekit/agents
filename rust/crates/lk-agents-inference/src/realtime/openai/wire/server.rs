//! Typed server events. Unknown events retain their envelope identity; unknown items are ignored.

use super::*;
use crate::realtime::protocol::OutputItemKind;
use lk_agents_abi::llm::realtime::{GenerationStatus, Modalities};
use lk_agents_abi::usage::TokenUsage;
use serde::Deserialize;

#[derive(Deserialize)]
struct Envelope {
    #[serde(rename = "type")]
    kind: String,
    event_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ServerEvent {
    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted { item_id: Option<String> },
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped,
    #[serde(rename = "response.created")]
    ResponseCreated { response: ResponseRef },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { item: ItemHead },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded { item_id: String, part: PartHead },
    #[serde(rename = "conversation.item.added")]
    ItemAdded { previous_item_id: Option<String>, item: ServerItem },
    #[serde(rename = "conversation.item.created")]
    BetaItemAdded { previous_item_id: Option<String>, item: ServerItem },
    #[serde(rename = "conversation.item.deleted")]
    ItemDeleted { item_id: String },
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    TranscriptionDelta {
        item_id: String,
        #[serde(default)]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted {
        item_id: String,
        #[serde(default)]
        content_index: usize,
        transcript: String,
        logprobs: Option<Vec<LogProb>>,
        status: Option<TranscriptStatus>,
        event_id: Option<String>,
        usage: Option<TranscriptUsage>,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    TranscriptionFailed {
        item_id: String,
        #[serde(default)]
        content_index: usize,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.updated")]
    TranscriptionUpdated { item_id: String, transcript: String },
    #[serde(rename = "response.output_text.delta")]
    TextDelta { item_id: String, delta: String },
    #[serde(rename = "response.text.delta")]
    BetaTextDelta { item_id: String, delta: String },
    #[serde(rename = "response.output_audio_transcript.delta")]
    AudioTranscriptDelta { item_id: String, delta: String, start_time: Option<f64> },
    #[serde(rename = "response.audio_transcript.delta")]
    BetaAudioTranscriptDelta { item_id: String, delta: String, start_time: Option<f64> },
    #[serde(rename = "response.output_audio.delta")]
    AudioDelta { item_id: String, delta: String },
    #[serde(rename = "response.audio.delta")]
    BetaAudioDelta { item_id: String, delta: String },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { item: ServerItem },
    #[serde(rename = "response.done")]
    ResponseDone { response: FinalResponse },
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        error: ErrorBody,
    },
    #[serde(rename = "input_audio_buffer.timeout_triggered")]
    IdleTimeoutTriggered,
    #[serde(rename = "session.created")]
    SessionCreated { session: Option<SessionHead> },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct ResponseRef {
    id: String,
    metadata: Option<std::collections::BTreeMap<String, String>>,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ItemHead {
    Message {
        id: String,
    },
    FunctionCall {
        id: String,
    },
    #[serde(other)]
    Other,
}
#[derive(Deserialize)]
struct PartHead {
    #[serde(rename = "type")]
    kind: PartKind,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum PartKind {
    Text,
    OutputText,
    #[serde(other)]
    Audio,
}
#[derive(Deserialize)]
struct SessionHead {
    model: Option<String>,
}
#[derive(Deserialize)]
struct LogProb {
    logprob: f64,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum TranscriptStatus {
    InProgress,
    #[serde(other)]
    Final,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerItem {
    Message {
        id: String,
        #[serde(with = "role")]
        role: Role,
        #[serde(default)]
        content: Vec<ContentPart>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        id: String,
        call_id: String,
        output: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: Option<ContentKind>,
    text: Option<String>,
    transcript: Option<String>,
    image_url: Option<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ContentKind {
    InputAudio,
    InputImage,
    #[serde(other)]
    Text,
}

impl ServerItem {
    fn into_chat(self, created_at: SystemTime) -> Option<ChatItem> {
        Some(match self {
            Self::Unknown => return None,
            Self::FunctionCall { id, call_id, name, arguments } => {
                ChatItem::FunctionCall(FunctionCall {
                    id,
                    call_id,
                    name,
                    arguments,
                    created_at,
                    extra: Default::default(),
                    group_id: None,
                })
            }
            Self::FunctionCallOutput { id, call_id, output } => ChatItem::FunctionCallOutput(
                FunctionCallOutput::new(id, call_id, output, false, created_at),
            ),
            Self::Message { id, role, content } => {
                let mut parts = Vec::new();
                for part in content {
                    parts.extend(part.into_chat(&id, parts.len()));
                }
                ChatItem::Message(ChatMessage::new(id, role, parts, created_at))
            }
        })
    }
}

impl ContentPart {
    fn into_chat(self, id: &str, kept: usize) -> Option<ChatContent> {
        match self.kind {
            Some(ContentKind::InputAudio) => self.transcript.map(ChatContent::Text),
            Some(ContentKind::InputImage) => self.image_url.map(|url| {
                ChatContent::Image(lk_agents_abi::ImageContent {
                    id: format!("img_{id}_{kept}"),
                    image: lk_agents_abi::ImageSource::Url(url),
                    inference_width: None,
                    inference_height: None,
                    inference_detail: Default::default(),
                    mime_type: None,
                })
            }),
            _ => self.text.filter(|text| !text.is_empty()).map(ChatContent::Text),
        }
    }
}

#[derive(Default, Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    input_token_details: Option<InputDetails>,
    output_token_details: Option<TokenDetails>,
}
#[derive(Default, Deserialize)]
struct InputDetails {
    cached_tokens: Option<u64>,
    text_tokens: Option<u64>,
    audio_tokens: Option<u64>,
    image_tokens: Option<u64>,
    cached_tokens_details: Option<TokenDetails>,
}
#[derive(Default, Deserialize)]
struct TokenDetails {
    text_tokens: Option<u64>,
    audio_tokens: Option<u64>,
    image_tokens: Option<u64>,
}
impl ResponseUsage {
    fn into_tokens(self) -> TokenUsage {
        let input = self.input_token_details.unwrap_or_default();
        let cached = input.cached_tokens_details.unwrap_or_default();
        let output = self.output_token_details.unwrap_or_default();
        TokenUsage {
            input_tokens: self.input_tokens.unwrap_or(0),
            output_tokens: self.output_tokens.unwrap_or(0),
            cache_read_tokens: input.cached_tokens,
            input_text_tokens: input.text_tokens,
            input_audio_tokens: input.audio_tokens,
            input_image_tokens: input.image_tokens,
            output_text_tokens: output.text_tokens,
            output_audio_tokens: output.audio_tokens,
            output_image_tokens: output.image_tokens,
            cache_read_text_tokens: cached.text_tokens,
            cache_read_audio_tokens: cached.audio_tokens,
            cache_read_image_tokens: cached.image_tokens,
            ..TokenUsage::default()
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TranscriptUsage {
    Tokens {
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        input_token_details: Option<TokenDetails>,
    },
    Duration {
        seconds: f64,
    },
    #[serde(other)]
    Unknown,
}
impl TranscriptUsage {
    fn into_usage(self) -> Result<Option<TranscriptionUsage>, DecodeCause> {
        Ok(match self {
            Self::Unknown => None,
            Self::Tokens { input_tokens, output_tokens, input_token_details } => {
                Some(TranscriptionUsage::Tokens(TokenUsage {
                    input_tokens: input_tokens.unwrap_or(0),
                    output_tokens: output_tokens.unwrap_or(0),
                    input_audio_tokens: input_token_details
                        .and_then(|details| details.audio_tokens),
                    ..TokenUsage::default()
                }))
            }
            Self::Duration { seconds } if seconds >= 0.0 => {
                Some(TranscriptionUsage::Duration { seconds })
            }
            Self::Duration { .. } => {
                return Err(DecodeCause(
                    "transcription usage seconds must be a non-negative number".into(),
                ));
            }
        })
    }
}

#[derive(Deserialize)]
struct FinalResponse {
    id: String,
    #[serde(deserialize_with = "generation_status")]
    status: GenerationStatus,
    status_details: Option<StatusDetails>,
    usage: Option<ResponseUsage>,
}

fn generation_status<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<GenerationStatus, D::Error> {
    let value = String::deserialize(deserializer)?;
    GenerationStatus::from_wire(&value)
        .ok_or_else(|| serde::de::Error::custom(format!("unsupported response status: {value}")))
}
#[derive(Deserialize)]
struct StatusDetails {
    error: Option<ErrorBody>,
}
#[derive(Default, Deserialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    kind: Option<String>,
    code: Option<String>,
    message: Option<String>,
    event_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct DecodeCause(String);

/// A malformed provider frame retains the operation and correlated event identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid realtime {operation} ({event_id:?}): {cause}")]
pub struct DecodeError {
    pub operation: String,
    pub event_id: Option<String>,
    pub cause: String,
}

pub fn decode_text(
    text: &str,
    shape: &WireShape,
    created_at: SystemTime,
) -> Result<Incoming, DecodeError> {
    let envelope: Envelope = serde_json::from_str(text).map_err(|error| DecodeError {
        operation: "message".into(),
        event_id: None,
        cause: error.to_string(),
    })?;
    let fail = |cause: String| DecodeError {
        operation: envelope.kind.clone(),
        event_id: envelope.event_id.clone(),
        cause,
    };
    // xAI lists both text names. Only the beta/Azure dialect consumes the legacy name.
    if !shape.beta_event_names
        && matches!(
            envelope.kind.as_str(),
            "response.text.delta"
                | "response.audio.delta"
                | "response.audio_transcript.delta"
                | "conversation.item.created"
        )
    {
        return Ok(Incoming::Unhandled { event_type: envelope.kind });
    }
    let event: ServerEvent = serde_json::from_str(text).map_err(|error| fail(error.to_string()))?;
    event.into_incoming(created_at, envelope.kind.clone()).map_err(|error| fail(error.to_string()))
}

impl ServerEvent {
    fn into_incoming(
        self,
        created_at: SystemTime,
        event_type: String,
    ) -> Result<Incoming, DecodeCause> {
        use ServerEvent::*;
        let unhandled = || Incoming::Unhandled { event_type: event_type.clone() };
        Ok(match self {
            SpeechStarted { item_id } => Incoming::SpeechStarted { item_id },
            SpeechStopped => Incoming::SpeechStopped,
            ResponseCreated { response } => Incoming::ResponseCreated {
                response_id: response.id,
                client_event_id: response
                    .metadata
                    .and_then(|mut metadata| metadata.remove("client_event_id")),
            },
            OutputItemAdded { item } => match item {
                ItemHead::Message { id } => {
                    Incoming::OutputItemAdded { item_id: id, kind: OutputItemKind::Message }
                }
                ItemHead::FunctionCall { id } => {
                    Incoming::OutputItemAdded { item_id: id, kind: OutputItemKind::FunctionCall }
                }
                ItemHead::Other => unhandled(),
            },
            ContentPartAdded { item_id, part } => Incoming::ContentPartAdded {
                item_id,
                modalities: match part.kind {
                    PartKind::Text | PartKind::OutputText => Modalities::TextOnly,
                    PartKind::Audio => Modalities::AudioAndText,
                },
            },
            ItemAdded { previous_item_id, item } | BetaItemAdded { previous_item_id, item } => {
                match item.into_chat(created_at) {
                    Some(item) => Incoming::ItemAdded { previous_item_id, item },
                    None => unhandled(),
                }
            }
            ItemDeleted { item_id } => Incoming::ItemDeleted { item_id },
            TranscriptionDelta { item_id, content_index, delta } => {
                Incoming::TranscriptionDelta { item_id, content_index, delta }
            }
            TranscriptionCompleted {
                item_id,
                content_index,
                transcript,
                logprobs,
                status,
                event_id,
                usage,
            } => {
                let confidence = logprobs.filter(|lps| !lps.is_empty()).map(|lps| {
                    (lps.iter().map(|lp| lp.logprob).sum::<f64>() / lps.len() as f64).exp()
                });
                Incoming::TranscriptionCompleted(super::TranscriptionCompleted {
                    item_id,
                    content_index,
                    transcript,
                    confidence,
                    in_progress: matches!(status, Some(TranscriptStatus::InProgress)),
                    event_id: event_id.unwrap_or_default(),
                    usage: usage.map(TranscriptUsage::into_usage).transpose()?.flatten(),
                })
            }
            TranscriptionFailed { item_id, content_index } => {
                Incoming::TranscriptionFailed { item_id, content_index }
            }
            TranscriptionUpdated { item_id, transcript } => {
                Incoming::TranscriptionUpdated { item_id, transcript }
            }
            TextDelta { item_id, delta } | BetaTextDelta { item_id, delta } => {
                Incoming::TextDelta { item_id, delta }
            }
            AudioTranscriptDelta { item_id, delta, start_time }
            | BetaAudioTranscriptDelta { item_id, delta, start_time } => {
                Incoming::AudioTranscriptDelta { item_id, delta, start_time }
            }
            AudioDelta { item_id, delta } | BetaAudioDelta { item_id, delta } => {
                let data = base64::engine::general_purpose::STANDARD
                    .decode(delta)
                    .map_err(|error| DecodeCause(format!("invalid PCM base64: {error}")))?;
                if data.len() % PCM_SAMPLE_BYTES != 0 {
                    return Err(DecodeCause("PCM data ends in a partial i16 sample".into()));
                }
                Incoming::AudioDelta { item_id, frame: audio_frame(data) }
            }
            OutputItemDone { item } => match item.into_chat(created_at) {
                Some(item) => Incoming::OutputItemDone { item },
                None => unhandled(),
            },
            ResponseDone { response } => {
                let error =
                    response.status_details.and_then(|details| details.error).unwrap_or_default();
                Incoming::ResponseDone(super::ResponseDone {
                    response_id: response.id,
                    status: response.status,
                    error_type: error.kind,
                    error_code: error.code,
                    usage: response.usage.unwrap_or_default().into_tokens(),
                })
            }
            Error { error } => Incoming::Error(ServerError {
                event_id: error.event_id,
                code: error.code,
                type_: error.kind,
                message: error.message.unwrap_or_default(),
            }),
            IdleTimeoutTriggered => Incoming::IdleTimeoutTriggered,
            SessionCreated { session } => {
                Incoming::SessionCreated { model: session.and_then(|s| s.model) }
            }
            Unknown => unhandled(),
        })
    }
}

#[cfg(test)]
pub fn decode_at(event: &Value, created_at: SystemTime) -> Result<Incoming, DecodeError> {
    decode_text(&event.to_string(), &WireShape::default(), created_at)
}
#[cfg(test)]
pub fn decode(event: &Value) -> Incoming {
    decode_at(event, SystemTime::UNIX_EPOCH).expect("valid fixture")
}
