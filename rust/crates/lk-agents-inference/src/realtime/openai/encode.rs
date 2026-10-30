//! Serialize typed realtime effects at the transport boundary.

use super::wire::{self, SessionPatch};
use crate::realtime::dialect::{Dialect, TurnDetection};
use crate::realtime::machine::Outgoing;
use crate::realtime::machine::{OptionsPatch, SessionOptions};
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{Maybe, ToolSpec};
use serde_json::{Map, Value, json};

pub fn encode(effect: Outgoing, images: &dyn ImageResolver) -> Result<Value, String> {
    Ok(match effect {
        Outgoing::SessionUpdate { event_id, opts, dialect } => {
            session_options(&event_id, &opts, &dialect)
        }
        Outgoing::OptionsUpdate { event_id, patch, dialect } => {
            options_update(&event_id, patch, &dialect)
        }
        Outgoing::InstructionsUpdate { event_id, instructions, dialect } => {
            let mut session = Map::new();
            session.insert("type".into(), json!("realtime"));
            session.insert("instructions".into(), json!(instructions));
            dialect.rewrite_session(&mut session);
            wire::session_update(&event_id, session)
        }
        Outgoing::ToolsUpdate { event_id, tools, model, dialect } => {
            tools_update(&event_id, &tools, &model, &dialect)
        }
        Outgoing::ItemCreate { event_id, item, previous_item_id } => {
            wire::conversation_item_create(
                &event_id,
                wire::to_wire_item(&item, images)?,
                previous_item_id.as_deref(),
            )
        }
        Outgoing::ItemDelete { event_id, item_id } => {
            wire::conversation_item_delete(&event_id, &item_id)
        }
        Outgoing::ItemTruncate { item_id, audio_end_ms } => {
            wire::conversation_item_truncate(&item_id, audio_end_ms)
        }
        Outgoing::AudioAppend { frame } => wire::input_audio_buffer_append(&frame.data),
        Outgoing::AudioCommit => wire::input_audio_buffer_commit(),
        Outgoing::AudioClear => wire::input_audio_buffer_clear(),
        Outgoing::ResponseCreate { req_id, instructions, tool_choice } => wire::response_create(
            &req_id,
            instructions.as_deref(),
            tool_choice.as_ref().map(|choice| wire::tool_choice(Some(choice))),
        ),
        Outgoing::ResponseCancel { response_id } => wire::response_cancel(response_id.as_deref()),
        Outgoing::ForceMessage { req_id, text } => wire::force_message(&req_id, &text),
    })
}

pub fn tools_update(event_id: &str, tools: &[ToolSpec], model: &str, dialect: &Dialect) -> Value {
    let mut session = Map::new();
    session.insert("type".into(), json!("realtime"));
    session.insert("model".into(), json!(model));
    session.insert("tools".into(), Value::Array(tools.iter().map(wire::tool_spec).collect()));
    dialect.rewrite_session(&mut session);
    wire::session_update(event_id, session)
}

pub fn session_options(event_id: &str, opts: &SessionOptions, dialect: &Dialect) -> Value {
    let mut p = SessionPatch::default();
    p.top.insert("model".into(), json!(opts.model));
    // The reference: "it'll respond in audio + transcript" -- ONE modality, not the list.
    p.top.insert(
        "output_modalities".into(),
        json!([if opts.audio_output { "audio" } else { "text" }]),
    );
    p.audio_in.insert("format".into(), wire::audio_format());
    // Explicit nulls (see `wire`): "off" is said, not left to the server's default.
    p.audio_in
        .insert("noise_reduction".into(), opts.noise_reduction.clone().unwrap_or(Value::Null));
    p.audio_in.insert("transcription".into(), opts.transcription.clone().unwrap_or(Value::Null));
    p.audio_in.insert(
        "turn_detection".into(),
        opts.turn_detection.as_ref().map_or(Value::Null, TurnDetection::to_json),
    );
    p.audio_out.insert("format".into(), wire::audio_format());
    p.audio_out.insert("speed".into(), json!(opts.speed));
    p.audio_out.insert("voice".into(), json!(opts.voice));
    p.top.insert("max_output_tokens".into(), opts.max_output_tokens.clone());
    p.top.insert("tool_choice".into(), wire::tool_choice(opts.tool_choice.as_ref()));
    p.top.insert("tracing".into(), opts.tracing.clone().unwrap_or(Value::Null));

    let mut session = p.into_session();
    if let Some(i) = &opts.instructions {
        session.insert("instructions".into(), json!(i));
    }
    if let Some(t) = &opts.truncation {
        session.insert("truncation".into(), t.clone());
    }
    if let Some(r) = &opts.reasoning {
        session.insert("reasoning".into(), r.clone());
    }
    dialect.rewrite_session(&mut session);
    wire::session_update(event_id, session)
}

pub fn options_update(event_id: &str, p: OptionsPatch, dialect: &Dialect) -> Value {
    let mut patch = SessionPatch::default();
    if let Maybe::Given(value) = p.tool_choice {
        patch.top.insert("tool_choice".into(), wire::tool_choice(value.as_ref()));
    }
    if let Maybe::Given(value) = p.max_output_tokens {
        patch.top.insert("max_output_tokens".into(), value);
    }
    for (key, given) in
        [("tracing", p.tracing), ("truncation", p.truncation), ("reasoning", p.reasoning)]
    {
        if let Maybe::Given(value) = given {
            patch.top.insert(key.into(), value.unwrap_or(Value::Null));
        }
    }
    if let Maybe::Given(value) = p.voice {
        patch.audio_out.insert("voice".into(), json!(value));
    }
    if let Maybe::Given(value) = p.speed {
        patch.audio_out.insert("speed".into(), json!(value));
    }
    if let Maybe::Given(value) = p.turn_detection {
        patch.audio_in.insert(
            "turn_detection".into(),
            value.as_ref().map_or(Value::Null, TurnDetection::to_json),
        );
    }
    for (key, given) in [("transcription", p.transcription), ("noise_reduction", p.noise_reduction)]
    {
        if let Maybe::Given(value) = given {
            patch.audio_in.insert(key.into(), value.unwrap_or(Value::Null));
        }
    }
    let mut session = patch.into_session();
    dialect.rewrite_session(&mut session);
    wire::session_update(event_id, session)
}
