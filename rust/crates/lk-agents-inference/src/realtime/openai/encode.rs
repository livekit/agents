//! Serialize typed realtime effects at the transport boundary.

use super::wire::{
    self, AudioConfig, AudioFormat, AudioInput, AudioOutput, ClientEvent, Session, SessionConfig,
};
use crate::realtime::dialect::{Dialect, SAMPLE_RATE};
use crate::realtime::machine::{OptionsPatch, Outgoing, SessionOptions};
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{Maybe, ToolSpec};
use serde_json::Value;

pub fn encode(
    effect: Outgoing,
    images: &dyn ImageResolver,
) -> Result<ClientEvent, wire::EncodeError> {
    Ok(match effect {
        Outgoing::SessionUpdate { event_id, opts, dialect } => {
            session_options(&event_id, &opts, &dialect)
        }
        Outgoing::OptionsUpdate { event_id, patch, dialect } => {
            options_update(&event_id, patch, &dialect)
        }
        Outgoing::InstructionsUpdate { event_id, instructions, dialect } => session_update(
            &event_id,
            SessionConfig { instructions: Some(instructions), ..Default::default() },
            &dialect,
        ),
        Outgoing::ToolsUpdate { event_id, tools, model, dialect } => {
            tools_update(&event_id, &tools, &model, &dialect)
        }
        Outgoing::ItemCreate { event_id, item, previous_item_id } => ClientEvent::ItemCreate {
            event_id,
            item: wire::to_wire_item(&item, images)?,
            previous_item_id,
        },
        Outgoing::ItemDelete { event_id, item_id } => ClientEvent::ItemDelete { event_id, item_id },
        Outgoing::ItemTruncate { item_id, audio_end_ms } => {
            ClientEvent::ItemTruncate { item_id, audio_end_ms, content_index: 0 }
        }
        Outgoing::AudioAppend { frame } => wire::input_audio_buffer_append(&frame.data),
        Outgoing::AudioCommit => ClientEvent::AudioCommit,
        Outgoing::AudioClear => ClientEvent::AudioClear,
        Outgoing::ResponseCreate { req_id, instructions, tool_choice } => {
            let metadata = wire::Metadata { client_event_id: req_id.clone() };
            ClientEvent::ResponseCreate {
                event_id: req_id,
                response: wire::ResponseParams {
                    instructions,
                    metadata,
                    tool_choice: tool_choice.as_ref().map(|choice| wire::tool_choice(Some(choice))),
                },
            }
        }
        Outgoing::ResponseCancel { response_id } => ClientEvent::ResponseCancel { response_id },
        Outgoing::ForceMessage { req_id, text } => ClientEvent::ItemCreate {
            event_id: req_id,
            previous_item_id: None,
            item: wire::Item::ForceMessage {
                role: lk_agents_abi::Role::Assistant,
                content: vec![wire::Content::OutputText { text }],
            },
        },
    })
}

fn session_update(event_id: &str, config: SessionConfig, dialect: &Dialect) -> ClientEvent {
    ClientEvent::SessionUpdate {
        event_id: event_id.to_owned(),
        session: Box::new(Session::new(config, dialect.provider, dialect.protocol)),
    }
}

pub fn tools_update(
    event_id: &str,
    tools: &[ToolSpec],
    model: &str,
    dialect: &Dialect,
) -> ClientEvent {
    session_update(
        event_id,
        SessionConfig {
            model: Some(model.to_owned()),
            tools: Some(tools.iter().map(|tool| wire::tool_spec(tool, dialect.provider)).collect()),
            ..Default::default()
        },
        dialect,
    )
}

pub fn session_options(event_id: &str, opts: &SessionOptions, dialect: &Dialect) -> ClientEvent {
    session_update(
        event_id,
        SessionConfig {
            model: Some(opts.model.clone()),
            output_modalities: Some(vec![if opts.audio_output {
                wire::Modality::Audio
            } else {
                wire::Modality::Text
            }]),
            max_output_tokens: Some(opts.max_output_tokens.clone()),
            tool_choice: Some(wire::tool_choice(opts.tool_choice.as_ref())),
            tracing: Some(opts.tracing.clone().unwrap_or(Value::Null)),
            audio: AudioConfig {
                input: AudioInput {
                    format: Some(AudioFormat { rate: SAMPLE_RATE }),
                    noise_reduction: Some(opts.noise_reduction.clone().unwrap_or(Value::Null)),
                    transcription: Some(opts.transcription.clone().unwrap_or(Value::Null)),
                    turn_detection: Some(opts.turn_detection.clone()),
                },
                output: AudioOutput {
                    format: Some(AudioFormat { rate: SAMPLE_RATE }),
                    speed: Some(opts.speed),
                    voice: Some(opts.voice.clone()),
                },
            },
            instructions: opts.instructions.clone(),
            truncation: opts.truncation.clone(),
            reasoning: opts.reasoning.clone(),
            tools: None,
        },
        dialect,
    )
}

pub fn options_update(event_id: &str, patch: OptionsPatch, dialect: &Dialect) -> ClientEvent {
    let config = SessionConfig {
        tool_choice: given(patch.tool_choice).map(|choice| wire::tool_choice(choice.as_ref())),
        max_output_tokens: given(patch.max_output_tokens),
        tracing: nullable(patch.tracing),
        truncation: nullable(patch.truncation),
        reasoning: nullable(patch.reasoning),
        audio: AudioConfig {
            input: AudioInput {
                transcription: nullable(patch.transcription),
                noise_reduction: nullable(patch.noise_reduction),
                turn_detection: given(patch.turn_detection),
                ..Default::default()
            },
            output: AudioOutput {
                voice: given(patch.voice),
                speed: given(patch.speed),
                ..Default::default()
            },
        },
        ..Default::default()
    };
    session_update(event_id, config, dialect)
}

fn given<T>(value: Maybe<T>) -> Option<T> {
    match value {
        Maybe::NotGiven => None,
        Maybe::Given(value) => Some(value),
    }
}
fn nullable(value: Maybe<Option<Value>>) -> Option<Value> {
    given(value).map(|value| value.unwrap_or(Value::Null))
}
