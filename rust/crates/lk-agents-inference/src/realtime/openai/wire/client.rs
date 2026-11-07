//! Client frames and the GA, beta and xAI session layouts.

use super::*;
use crate::realtime::dialect::TurnDetection;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    #[serde(rename = "session.update")]
    SessionUpdate { event_id: String, session: Box<Session> },
    #[serde(rename = "conversation.item.create")]
    ItemCreate {
        event_id: String,
        item: Item,
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
    },
    #[serde(rename = "conversation.item.delete")]
    ItemDelete { event_id: String, item_id: String },
    #[serde(rename = "conversation.item.truncate")]
    ItemTruncate { item_id: String, content_index: u32, audio_end_ms: u64 },
    #[serde(rename = "input_audio_buffer.append")]
    AudioAppend { audio: String },
    #[serde(rename = "input_audio_buffer.commit")]
    AudioCommit,
    #[serde(rename = "input_audio_buffer.clear")]
    AudioClear,
    #[serde(rename = "response.create")]
    ResponseCreate { event_id: String, response: ResponseParams },
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseParams {
    pub instructions: Option<String>,
    pub metadata: Metadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Choice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Metadata {
    pub client_event_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Choice {
    Mode(ChoiceMode),
    Function(NamedFunction),
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChoiceMode {
    Auto,
    Required,
    None,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename = "function")]
pub struct NamedFunction {
    pub name: String,
}

pub fn tool_choice(choice: Option<&lk_agents_abi::ToolChoice>) -> Choice {
    use lk_agents_abi::ToolChoice;
    match choice {
        None | Some(ToolChoice::Auto) => Choice::Mode(ChoiceMode::Auto),
        Some(ToolChoice::Required) => Choice::Mode(ChoiceMode::Required),
        Some(ToolChoice::Forbidden) => Choice::Mode(ChoiceMode::None),
        Some(ToolChoice::Named(name)) => Choice::Function(NamedFunction { name: name.clone() }),
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Tool {
    Function(FunctionTool),
    Provider(Value),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename = "function")]
pub struct FunctionTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub fn tool_spec(tool: &lk_agents_abi::ToolSpec) -> Tool {
    if let Some(declaration) = &tool.provider_tool {
        return Tool::Provider(declaration.payload.clone());
    }
    Tool::Function(FunctionTool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.json_schema.clone(),
    })
}

/// An absent field leaves the remote option alone; Some(Value::Null) clears it.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(tag = "type", rename = "realtime")]
pub struct SessionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<Modality>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracing: Option<Value>,
    #[serde(skip_serializing_if = "AudioConfig::is_empty")]
    pub audio: AudioConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Audio,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AudioConfig {
    #[serde(skip_serializing_if = "AudioInput::is_empty")]
    pub input: AudioInput,
    #[serde(skip_serializing_if = "AudioOutput::is_empty")]
    pub output: AudioOutput,
}

impl AudioConfig {
    fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AudioInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcription: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<Option<TurnDetection>>,
}
impl AudioInput {
    fn is_empty(&self) -> bool {
        self.format.is_none()
            && self.noise_reduction.is_none()
            && self.transcription.is_none()
            && self.turn_detection.is_none()
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AudioOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<Value>,
}
impl AudioOutput {
    fn is_empty(&self) -> bool {
        self.format.is_none() && self.speed.is_none() && self.voice.is_none()
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "type", rename = "audio/pcm")]
pub struct AudioFormat {
    pub rate: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Session {
    Ga(SessionConfig),
    Beta(BetaSession),
    Xai(XaiSession),
}

impl Session {
    pub fn new(config: SessionConfig, provider: RealtimeProvider, protocol: Protocol) -> Self {
        if protocol == Protocol::Beta {
            Self::Beta(config.into())
        } else if provider == RealtimeProvider::Xai {
            Self::Xai(config.into())
        } else {
            Self::Ga(config)
        }
    }

    pub(super) fn drop_model(&mut self) {
        match self {
            Self::Ga(s) => s.model = None,
            Self::Beta(s) => s.model = None,
            Self::Xai(s) => s.model = None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BetaSession {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<Vec<Modality>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_response_output_tokens: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tracing: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_audio_format: Option<BetaAudioFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_audio_format: Option<BetaAudioFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_audio_transcription: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_audio_noise_reduction: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_detection: Option<Option<TurnDetection>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum BetaAudioFormat {
    #[serde(rename = "pcm16")]
    Pcm16,
}

impl From<SessionConfig> for BetaSession {
    fn from(s: SessionConfig) -> Self {
        Self {
            model: s.model,
            modalities: s.output_modalities.map(|m| {
                if m.iter().any(|m| matches!(m, Modality::Audio)) {
                    vec![Modality::Text, Modality::Audio]
                } else {
                    vec![Modality::Text]
                }
            }),
            max_response_output_tokens: s.max_output_tokens,
            tool_choice: s.tool_choice,
            tracing: s.tracing.filter(|v| !v.is_null()),
            instructions: s.instructions,
            tools: s.tools,
            input_audio_format: s.audio.input.format.map(|_| BetaAudioFormat::Pcm16),
            output_audio_format: s.audio.output.format.map(|_| BetaAudioFormat::Pcm16),
            input_audio_transcription: s.audio.input.transcription,
            input_audio_noise_reduction: s.audio.input.noise_reduction,
            turn_detection: s.audio.input.turn_detection,
            voice: s.audio.output.voice,
            speed: s.audio.output.speed,
        }
    }
}

/// xAI has top-level voice and turn detection, and no GA session type or output limits.
#[derive(Debug, Clone, Serialize)]
pub struct XaiSession {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_detection: Option<Option<TurnDetection>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Value>,
    #[serde(skip_serializing_if = "AudioConfig::is_empty")]
    audio: AudioConfig,
}
impl From<SessionConfig> for XaiSession {
    fn from(mut s: SessionConfig) -> Self {
        let voice = s.audio.output.voice.take();
        let turn_detection = s.audio.input.turn_detection.take();
        // xAI's schema has no input noise reduction.
        s.audio.input.noise_reduction = None;
        Self {
            model: s.model,
            instructions: s.instructions,
            voice,
            turn_detection,
            tools: s.tools,
            reasoning: s.reasoning,
            audio: s.audio,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    Message {
        id: String,
        #[serde(with = "role")]
        role: Role,
        content: Vec<Content>,
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
    ForceMessage {
        #[serde(with = "role")]
        role: Role,
        content: Vec<Content>,
    },
}

impl Item {
    pub(super) fn beta(&mut self) {
        if let Self::Message { content, .. } = self {
            for part in content {
                if let Content::OutputText { text } = part {
                    *part = Content::Text { text: std::mem::take(text) };
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    InputText { text: String },
    OutputText { text: String },
    Text { text: String },
    InputImage { image_url: String },
}

pub fn to_wire_item(item: &ChatItem, images: &dyn ImageResolver) -> Result<Item, EncodeError> {
    Ok(match item {
        ChatItem::FunctionCall(fc) => Item::FunctionCall {
            id: fc.id.clone(),
            call_id: shorten_call_id(&fc.call_id),
            name: fc.name.clone(),
            arguments: fc.arguments.clone(),
        },
        ChatItem::FunctionCallOutput(fc) => Item::FunctionCallOutput {
            id: fc.id.clone(),
            call_id: shorten_call_id(&fc.call_id),
            output: fc.output.clone(),
        },
        ChatItem::Message(m) => Item::Message {
            id: m.id.clone(),
            role: if m.role == Role::Developer { Role::System } else { m.role },
            content: wire_content(m, images)?,
        },
        other => return Err(EncodeError(format!("unsupported item type: {}", other.kind()))),
    })
}

fn wire_content(m: &ChatMessage, images: &dyn ImageResolver) -> Result<Vec<Content>, EncodeError> {
    let mut content = Vec::new();
    for part in &m.content {
        if let ChatContent::Text(text) = part {
            content.push(if m.role == Role::Assistant {
                Content::OutputText { text: text.clone() }
            } else {
                Content::InputText { text: text.clone() }
            });
            continue;
        }
        let ChatContent::Image(image) = part else { continue };
        if m.role != Role::User {
            continue;
        }
        let resolved = images.resolve(image).map_err(|error| EncodeError(error.to_string()))?;
        if resolved.external_url.is_some() {
            return Err(EncodeError(
                "realtime images require an inline data URL; external URLs are unsupported".into(),
            ));
        }
        let Some(bytes) = resolved.data_bytes else { continue };
        let mime = resolved.mime_type.as_deref().unwrap_or("image/jpeg");
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        content.push(Content::InputImage { image_url: format!("data:{mime};base64,{b64}") });
    }
    Ok(content)
}

pub fn input_audio_buffer_append(pcm: &[u8]) -> ClientEvent {
    ClientEvent::AudioAppend { audio: base64::engine::general_purpose::STANDARD.encode(pcm) }
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EncodeError(pub String);
