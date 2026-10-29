//! The simulated realtime server's typed records against the recorded wire: a recorded client frame
//! as the frame the realtime core sends, a delivered event as the JSON it went out as.
use std::sync::Arc;
use std::time::SystemTime;

use lk_agents_abi::{
    AudioFrame, ChatContent, ChatItem, ChatMessage, FunctionCallOutput, Maybe, Role, ToolSpec,
    llm::realtime::GenerationStatus,
};
use lk_agents_inference::realtime::{
    INFERENCE, Incoming, OptionsPatch, Outgoing, SessionOptions, TurnDetection,
};
use serde_json::{Value, json};

use super::super::server::Delivered;

/// The model a session names, as a LiveKit Inference realtime model's tools update carries it.
const MODEL: &str = "openai/gpt-realtime";

/// A recorded client frame as the realtime core sends it: the frame its encoder turns into this
/// JSON. A `session.update` naming tools is the core's two updates; a kind the core never sends
/// (`conversation.item.retrieve`, `output_audio_buffer.clear`, an unknown type) is none. An item
/// created without an id gets the one the core always gives it.
pub fn outgoing(frame: &Value, minted: &mut u32) -> Vec<Outgoing> {
    let event_id = frame["event_id"].as_str().unwrap_or("e").to_owned();
    let s = |v: &Value| v.as_str().map(str::to_owned);
    match frame["type"].as_str().unwrap() {
        // A session update naming no modalities is the core's patch of what it names.
        "session.update" if frame["session"].get("output_modalities").is_none() => {
            let session = &frame["session"];
            let input = &session["audio"]["input"];
            let mut frames = Vec::new();
            if let Some(instructions) = s(&session["instructions"]) {
                let event_id = event_id.clone();
                frames.push(Outgoing::InstructionsUpdate {
                    event_id,
                    instructions,
                    dialect: INFERENCE,
                });
            }
            let mut patch = OptionsPatch::default();
            if let Some(td) = input.get("turn_detection") {
                patch.turn_detection =
                    Maybe::Given(Some(TurnDetection::Raw(td.clone())).filter(|_| !td.is_null()));
            }
            if let Some(t) = input.get("transcription") {
                patch.transcription = Maybe::Given(Some(t.clone()).filter(|t| !t.is_null()));
            }
            if let Some(voice) = s(&session["audio"]["output"]["voice"]) {
                patch.voice = Maybe::Given(voice);
            }
            if !patch.is_empty() || frames.is_empty() {
                frames.push(Outgoing::OptionsUpdate { event_id, patch, dialect: INFERENCE });
            }
            frames
        }
        "session.update" => {
            let session = &frame["session"];
            let input = &session["audio"]["input"];
            let mut opts = SessionOptions::from_dialect(&INFERENCE, false);
            opts.turn_detection = match &input["turn_detection"] {
                Value::Null => None,
                td => Some(TurnDetection::Raw(td.clone())),
            };
            opts.transcription = Some(input["transcription"].clone()).filter(|t| !t.is_null());
            if let Some(m) = session["output_modalities"].as_array() {
                opts.audio_output = m.iter().any(|v| v == "audio");
            }
            if let Some(voice) = s(&session["audio"]["output"]["voice"]) {
                opts.voice = voice;
            }
            opts.instructions = s(&session["instructions"]);
            let mut frames = vec![Outgoing::SessionUpdate {
                event_id: event_id.clone(),
                opts,
                dialect: INFERENCE,
            }];
            if let Some(tools) = session["tools"].as_array() {
                let tool = |t: &Value| ToolSpec {
                    provider_tool: None,
                    name: t["name"].as_str().unwrap().to_owned(),
                    description: String::new(),
                    json_schema: json!({}),
                    flags: Default::default(),
                };
                let tools = Arc::new(tools.iter().map(tool).collect());
                frames.push(Outgoing::ToolsUpdate {
                    event_id,
                    tools,
                    model: MODEL.into(),
                    dialect: INFERENCE,
                });
            }
            frames
        }
        "input_audio_buffer.append" => {
            let audio = frame["audio"].as_str().unwrap();
            let pcm =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, audio).unwrap();
            let samples = (pcm.len() / 2) as u32;
            vec![Outgoing::AudioAppend { frame: AudioFrame::new(pcm, 24_000, 1, samples) }]
        }
        "input_audio_buffer.commit" => vec![Outgoing::AudioCommit],
        "input_audio_buffer.clear" => vec![Outgoing::AudioClear],
        "response.create" => vec![Outgoing::ResponseCreate {
            req_id: event_id,
            instructions: s(&frame["response"]["instructions"]),
            tool_choice: None,
        }],
        "response.cancel" => {
            vec![Outgoing::ResponseCancel { response_id: s(&frame["response_id"]) }]
        }
        "conversation.item.create" => {
            let item = &frame["item"];
            *minted += 1;
            let id = s(&item["id"]).unwrap_or_else(|| format!("item_client_{minted}"));
            // An assistant message's content is its text: one with a part of none (audio) is
            // nothing the core sends.
            let parts = item["content"].as_array().into_iter().flatten();
            if item["role"] == "assistant" && parts.clone().any(|p| p["text"].is_null()) {
                return Vec::new();
            }
            let item = if item["type"] == "function_call_output" {
                let (call, output) =
                    (s(&item["call_id"]).unwrap(), s(&item["output"]).unwrap_or_default());
                ChatItem::FunctionCallOutput(FunctionCallOutput::new(
                    id,
                    call,
                    output,
                    false,
                    SystemTime::UNIX_EPOCH,
                ))
            } else {
                let role = match item["role"].as_str() {
                    Some("assistant") => Role::Assistant,
                    Some("system") => Role::System,
                    _ => Role::User,
                };
                let parts = item["content"].as_array().into_iter().flatten();
                let content = parts.filter_map(|p| s(&p["text"])).map(ChatContent::Text).collect();
                ChatItem::Message(ChatMessage::new(id, role, content, SystemTime::UNIX_EPOCH))
            };
            let previous_item_id = s(&frame["previous_item_id"]);
            vec![Outgoing::ItemCreate { event_id, item, previous_item_id }]
        }
        "conversation.item.delete" => {
            vec![Outgoing::ItemDelete { event_id, item_id: s(&frame["item_id"]).unwrap() }]
        }
        "conversation.item.truncate" => vec![Outgoing::ItemTruncate {
            item_id: s(&frame["item_id"]).unwrap(),
            audio_end_ms: frame["audio_end_ms"].as_u64().unwrap(),
        }],
        _ => Vec::new(),
    }
}

/// A delivered event as the wire carried it: the fields these checks read.
pub fn wire(d: &Delivered) -> Value {
    let item = |item: &ChatItem| match item {
        ChatItem::Message(m) => {
            let role = match m.role {
                Role::Assistant => "assistant",
                Role::User => "user",
                Role::System | Role::Developer => "system",
            };
            json!({"id": m.id, "type": "message", "role": role})
        }
        ChatItem::FunctionCall(f) => {
            json!({"id": f.id, "type": "function_call", "call_id": f.call_id})
        }
        ChatItem::FunctionCallOutput(f) => {
            json!({"id": f.id, "type": "function_call_output", "call_id": f.call_id})
        }
        other => json!({"id": other.id()}),
    };
    let mut v = match &d.event {
        Incoming::SessionCreated { .. } => json!({"type": "session.created"}),
        Incoming::SpeechStarted { item_id } => {
            json!({"type": "input_audio_buffer.speech_started", "item_id": item_id})
        }
        Incoming::SpeechStopped => {
            json!({"type": "input_audio_buffer.speech_stopped", "item_id": d.item})
        }
        Incoming::ResponseCreated { response_id, client_event_id } => {
            let metadata = client_event_id.as_ref().map(|c| json!({"client_event_id": c}));
            json!({"type": "response.created", "response": {"id": response_id, "status": "in_progress", "metadata": metadata}})
        }
        Incoming::OutputItemAdded { item_id, item_type } => {
            json!({"type": "response.output_item.added", "item": {"id": item_id, "type": item_type}})
        }
        Incoming::ContentPartAdded { item_id, part_type } => {
            json!({"type": "response.content_part.added", "item_id": item_id, "part": {"type": part_type}})
        }
        Incoming::ItemAdded { previous_item_id, item: added } => {
            json!({"type": "conversation.item.added", "previous_item_id": previous_item_id, "item": item(added)})
        }
        Incoming::ItemDeleted { item_id } => {
            json!({"type": "conversation.item.deleted", "item_id": item_id})
        }
        Incoming::TranscriptionDelta { item_id, .. } => {
            json!({"type": "conversation.item.input_audio_transcription.delta", "item_id": item_id})
        }
        Incoming::TranscriptionCompleted(t) => {
            json!({"type": "conversation.item.input_audio_transcription.completed", "item_id": t.item_id})
        }
        Incoming::TranscriptionFailed { item_id, .. } => {
            json!({"type": "conversation.item.input_audio_transcription.failed", "item_id": item_id})
        }
        Incoming::TextDelta { item_id, .. } => {
            json!({"type": "response.output_text.delta", "item_id": item_id})
        }
        Incoming::AudioTranscriptDelta { item_id, .. } => {
            json!({"type": "response.output_audio_transcript.delta", "item_id": item_id})
        }
        Incoming::AudioDelta { item_id, .. } => {
            json!({"type": "response.output_audio.delta", "item_id": item_id})
        }
        Incoming::OutputItemDone { item: done } => {
            json!({"type": "response.output_item.done", "item": item(done)})
        }
        Incoming::ResponseDone(done) => {
            let status = match done.status {
                GenerationStatus::Completed => "completed",
                GenerationStatus::Cancelled => "cancelled",
                GenerationStatus::Incomplete => "incomplete",
                GenerationStatus::Failed => "failed",
            };
            let details = d.reason.map(|r| json!({"reason": r}));
            json!({"type": "response.done", "response": {"id": done.response_id, "status": status, "status_details": details}})
        }
        Incoming::Error(e) => {
            json!({"type": "error", "error": {"code": e.code, "event_id": e.event_id}})
        }
        Incoming::Unhandled { event_type } => json!({"type": event_type, "item_id": d.item}),
        Incoming::TranscriptionUpdated { .. } | Incoming::IdleTimeoutTriggered => {
            unreachable!("xAI only")
        }
    };
    if let Some(response) = &d.response
        && v.get("response").is_none()
    {
        v["response_id"] = json!(response);
    }
    v
}
