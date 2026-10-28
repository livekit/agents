//! Retained native allocations counted before entering the shared event queue.

use std::mem::{size_of, size_of_val};

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::chat_context::{Extra, MetricsMetadata, MetricsReport};
use lk_agents_abi::llm::realtime::{Event, RealtimeModelError};
use lk_agents_abi::metrics::base::{AgentMetrics, Metadata};
use lk_agents_abi::{ChatContent, ChatItem, FunctionCall, ImageSource};

// Two maximum wire frames fit with decoded metadata; count admission still caps small events.
pub(super) const EVENT_BYTES: usize = 32 * 1024 * 1024;

pub(super) fn retained_bytes(event: &mut Event) -> Result<usize, ApiError> {
    let mut bytes = Retention { bytes: size_of::<Event>(), metadata_nodes: 0 };
    match event {
        Event::InputSpeechStarted(_)
        | Event::InputSpeechStopped(_)
        | Event::SessionReconnected(_)
        | Event::SessionSwapped { .. } => {}
        Event::InputTranscription(value) => bytes.strings(&[&value.item_id, &value.transcript])?,
        Event::GenerationCreated(value) => {
            bytes.string(&value.response_id)?;
            bytes.optional(&value.req_id)?;
        }
        Event::GenerateReplyFailed(value) => {
            bytes.strings(&[&value.req_id, &value.error.message])?;
            bytes.optional(&value.error.code)?;
        }
        Event::ServerGeneration(_) => return Err(opaque()),
        Event::MessageStarted(value) => {
            bytes.string(&value.message_id)?;
            bytes.optional(&value.response_id)?;
        }
        Event::TextDelta(value) => {
            bytes.strings(&[&value.message_id, &value.delta])?;
            bytes.optional(&value.response_id)?;
        }
        Event::TranscriptAttribution(value) => {
            bytes.strings(&[&value.response_id, &value.source_response_id])?;
        }
        Event::AudioDelta(value) => {
            bytes.string(&value.message_id)?;
            bytes.optional(&value.response_id)?;
            bytes.add(value.frame.data.len())?;
        }
        Event::MessageModalities(value) => {
            bytes.string(&value.message_id)?;
            bytes.optional(&value.response_id)?;
        }
        Event::MessageDone(value) => {
            bytes.string(&value.message_id)?;
            bytes.optional(&value.response_id)?;
        }
        Event::FunctionCall(value) => {
            bytes.add(size_of_val(value.call.as_ref()))?;
            bytes.function(&mut value.call)?;
        }
        Event::GenerationDone(value) => bytes.string(&value.response_id)?,
        Event::RemoteItemAdded(value) => {
            bytes.optional(&value.previous_item_id)?;
            bytes.add(size_of_val(value.item.as_ref()))?;
            bytes.item(&mut value.item)?;
        }
        Event::Metrics(value) => {
            bytes.add(size_of_val(value.metrics.as_ref()))?;
            bytes.metrics(&value.metrics)?;
        }
        Event::Error(value) => bytes.error(value)?,
        Event::HostError(value) => bytes.error(&value.error)?,
    }
    Ok(bytes.bytes)
}

fn overload() -> ApiError {
    ApiError::new("realtime event retained-byte limit exceeded").with_retryable(false)
}

fn opaque() -> ApiError {
    ApiError::new("unresolved opaque payload cannot enter the realtime event queue")
        .with_retryable(false)
}

struct Retention {
    bytes: usize,
    metadata_nodes: usize,
}

impl Retention {
    fn add(&mut self, bytes: usize) -> Result<(), ApiError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|total| *total <= EVENT_BYTES)
            .ok_or_else(overload)?;
        Ok(())
    }

    fn nodes(&mut self, count: usize) -> Result<(), ApiError> {
        self.metadata_nodes = self
            .metadata_nodes
            .checked_add(count)
            .filter(|total| *total <= 4096)
            .ok_or_else(overload)?;
        Ok(())
    }

    fn string(&mut self, value: &String) -> Result<(), ApiError> {
        self.add(value.capacity())
    }

    fn strings(&mut self, values: &[&String]) -> Result<(), ApiError> {
        for value in values {
            self.string(value)?;
        }
        Ok(())
    }

    fn optional(&mut self, value: &Option<String>) -> Result<(), ApiError> {
        if let Some(value) = value {
            self.string(value)?;
        }
        Ok(())
    }

    fn string_list(&mut self, values: &Option<Vec<String>>) -> Result<(), ApiError> {
        if let Some(values) = values {
            self.nodes(values.len())?;
            self.add(values.capacity().saturating_mul(size_of::<String>()))?;
            for value in values {
                self.string(value)?;
            }
        }
        Ok(())
    }

    fn extra(&mut self, values: &mut Extra, depth: usize) -> Result<(), ApiError> {
        self.nodes(values.len())?;
        self.add(values.len().saturating_mul(size_of::<(String, serde_json::Value)>() + 64))?;
        // serde_json hides map capacity. Moving entries removes unmeasurable spare allocation;
        // strings keep their original capacities, and declaration order is preserved.
        *values = std::mem::take(values).into_iter().collect();
        for (key, value) in values.iter_mut() {
            self.string(key)?;
            self.json(value, depth)?;
        }
        Ok(())
    }

    fn json(&mut self, value: &mut serde_json::Value, depth: usize) -> Result<(), ApiError> {
        if depth == 64 {
            return Err(overload());
        }
        match value {
            serde_json::Value::String(value) => self.string(value)?,
            serde_json::Value::Array(values) => {
                self.nodes(values.len())?;
                self.add(values.capacity().saturating_mul(size_of::<serde_json::Value>()))?;
                for value in values {
                    self.json(value, depth + 1)?;
                }
            }
            serde_json::Value::Object(values) => self.extra(values, depth + 1)?,
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            }
        }
        Ok(())
    }

    fn function(&mut self, call: &mut FunctionCall) -> Result<(), ApiError> {
        self.strings(&[&call.id, &call.call_id, &call.arguments, &call.name])?;
        self.optional(&call.group_id)?;
        self.extra(&mut call.extra, 0)
    }

    fn item(&mut self, item: &mut ChatItem) -> Result<(), ApiError> {
        match item {
            ChatItem::Message(message) => {
                self.string(&message.id)?;
                self.nodes(message.content.len())?;
                self.add(message.content.capacity().saturating_mul(size_of::<ChatContent>()))?;
                for content in &mut message.content {
                    match content {
                        ChatContent::Text(text) => self.string(text)?,
                        ChatContent::Image(image) => {
                            self.string(&image.id)?;
                            self.optional(&image.mime_type)?;
                            match &image.image {
                                ImageSource::Url(url) => self.string(url)?,
                                ImageSource::Frame(_) => return Err(opaque()),
                            }
                        }
                        ChatContent::Audio(_) => return Err(opaque()),
                    }
                }
                self.extra(&mut message.extra, 0)?;
                self.report(&message.metrics)?;
            }
            ChatItem::FunctionCall(call) => self.function(call)?,
            ChatItem::FunctionCallOutput(output) => {
                self.strings(&[&output.id, &output.name, &output.call_id, &output.output])?;
            }
            ChatItem::AgentHandoff(handoff) => {
                self.strings(&[&handoff.id, &handoff.new_agent_id])?;
                self.optional(&handoff.old_agent_id)?;
            }
            ChatItem::AgentConfigUpdate(config) => {
                self.string(&config.id)?;
                self.optional(&config.instructions)?;
                self.string_list(&config.tools_added)?;
                self.string_list(&config.tools_removed)?;
                if config.tools.is_some() {
                    return Err(opaque());
                }
            }
        }
        Ok(())
    }

    fn report(&mut self, report: &MetricsReport) -> Result<(), ApiError> {
        self.string_list(&report.provider_request_ids)?;
        for metadata in
            [&report.llm_metadata, &report.tts_metadata, &report.stt_metadata].into_iter().flatten()
        {
            let MetricsMetadata { model_name, model_provider } = metadata;
            self.optional(model_name)?;
            self.optional(model_provider)?;
        }
        Ok(())
    }

    fn metadata(&mut self, metadata: Option<&Metadata>) -> Result<(), ApiError> {
        if let Some(Metadata { model_name, model_provider }) = metadata {
            self.optional(model_name)?;
            self.optional(model_provider)?;
        }
        Ok(())
    }

    fn metrics(&mut self, metrics: &AgentMetrics) -> Result<(), ApiError> {
        match metrics {
            AgentMetrics::Stt(m) => self.strings(&[&m.label, &m.request_id])?,
            AgentMetrics::Llm(m) => {
                self.strings(&[&m.label, &m.request_id])?;
                self.optional(&m.speech_id)?;
            }
            AgentMetrics::Tts(m) => {
                self.strings(&[&m.label, &m.request_id])?;
                self.optional(&m.segment_id)?;
                self.optional(&m.speech_id)?;
            }
            AgentMetrics::Vad(m) => self.string(&m.label)?,
            AgentMetrics::Eou(m) => self.optional(&m.speech_id)?,
            AgentMetrics::RealtimeModel(m) => self.strings(&[&m.label, &m.request_id])?,
            AgentMetrics::EotInference(_)
            | AgentMetrics::Interruption(_)
            | AgentMetrics::Avatar(_) => {}
        }
        self.metadata(metrics.metadata())
    }

    fn error(&mut self, error: &RealtimeModelError) -> Result<(), ApiError> {
        self.strings(&[&error.label, &error.message])?;
        self.optional(&error.req_id)?;
        self.optional(&error.code)?;
        if let Some(provider) = &error.provider_error {
            self.string(&provider.message)?;
            self.optional(&provider.request_id)?;
        }
        Ok(())
    }
}
