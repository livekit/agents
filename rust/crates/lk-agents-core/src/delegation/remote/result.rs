//! Model-facing results use SDK content; binary data stays in the retained observation.

use std::io::{self, Write};

use lk_agents_abi::a2a_protocol::{Artifact, Message, Part, PartContent};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use serde::Serialize;
use serde_json::Value;

use crate::a2a::client::TaskUpdate;

// Bound model material independently of the durable SDK payload. Exceeding the bound cannot
// turn completed business work into a retry or imply that omitted content was read.
const MODEL_RESULT_BYTES: usize = 32 * 1024;
const RETAINED_RESULT: &str = "The remote task completed. Its result is retained in this \
    conversation's remote observations, but is too large to include in this model turn.";
const NO_RESULT: &str = "The remote task completed. No result content was supplied or could \
    be recovered; inspect its retained remote observations for details.";

pub(super) fn completed(update: &TaskUpdate, remote_id: &str) -> ToolOutcome {
    let exact = update.metadata.get("livekit/verbatim").and_then(Value::as_str);
    let parts = || {
        update.artifacts.iter().flat_map(|artifact| &artifact.parts).chain(
            update.message.iter().filter(|_| update.artifacts.is_empty()).flat_map(|m| &m.parts),
        )
    };
    if parts().all(|part| matches!(part.content, PartContent::Text(_))) {
        let text = if update.text.is_empty() { NO_RESULT } else { &update.text };
        if text.len() <= MODEL_RESULT_BYTES {
            return super::answer(text, exact);
        }
        // Exact delivery is an independent promise. A bound on supporting model material
        // cannot silently truncate or discard explicitly required words.
        return match exact.filter(|exact| !exact.is_empty() && text.starts_with(exact)) {
            Some(exact) => ToolOutcome::Value {
                text: exact.into(),
                verbatim: true,
                reply_required: text.len() > exact.len(),
                more: Some(RETAINED_RESULT.into()),
            },
            None => super::answer(RETAINED_RESULT, None),
        };
    }
    let mut content = parts();
    let sole_data = match (content.next(), content.next()) {
        (Some(Part { content: PartContent::Data(value), .. }), None) => Some(value),
        _ => None,
    };
    let rendered = match sole_data {
        Some(value) => render(value),
        None => payload(update, remote_id),
    };
    let text = rendered.unwrap_or_else(|| RETAINED_RESULT.into());
    if let Some(exact) = exact.filter(|text| !text.is_empty() && update.text.starts_with(text)) {
        ToolOutcome::Value {
            text: exact.into(),
            verbatim: true,
            reply_required: true,
            more: Some(text),
        }
    } else {
        ToolOutcome::Value { text, verbatim: false, reply_required: true, more: None }
    }
}

fn payload(update: &TaskUpdate, remote_id: &str) -> Option<String> {
    let mut budget = Buffer::default();
    if !budget.measure(&remote_id) || !budget.measure(&update.task_id) {
        return None;
    }
    if !update.artifacts.is_empty() {
        let mut artifacts = Vec::new();
        for artifact in &update.artifacts {
            if !budget.measure(&artifact.artifact_id)
                || !budget.measure(&artifact.name)
                || !budget.measure(&artifact.description)
                || !budget.measure(&artifact.metadata)
                || !budget.measure(&artifact.extensions)
            {
                return None;
            }
            artifacts.push(Artifact {
                artifact_id: artifact.artifact_id.clone(),
                name: artifact.name.clone(),
                description: artifact.description.clone(),
                parts: model_parts(
                    &artifact.parts,
                    &format!(
                        "remote {remote_id:?}, task {:?}, artifact {:?}",
                        update.task_id, artifact.artifact_id
                    ),
                    &mut budget,
                )?,
                metadata: artifact.metadata.clone(),
                extensions: artifact.extensions.clone(),
            });
        }
        return render(&artifacts);
    }
    let message = update.message.as_ref()?;
    if !budget.measure(&message.message_id)
        || !budget.measure(&message.context_id)
        || !budget.measure(&message.task_id)
        || !budget.measure(&message.metadata)
        || !budget.measure(&message.extensions)
        || !budget.measure(&message.reference_task_ids)
    {
        return None;
    }
    render(&Message {
        message_id: message.message_id.clone(),
        context_id: message.context_id.clone(),
        task_id: message.task_id.clone(),
        role: message.role.clone(),
        parts: model_parts(
            &message.parts,
            &format!("remote {remote_id:?}, message {:?}", message.message_id),
            &mut budget,
        )?,
        metadata: message.metadata.clone(),
        extensions: message.extensions.clone(),
        reference_task_ids: message.reference_task_ids.clone(),
    })
}

fn model_parts(parts: &[Part], location: &str, budget: &mut Buffer) -> Option<Vec<Part>> {
    let mut model = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let content = match &part.content {
            PartContent::Raw(bytes) => {
                if !budget.measure(&part.filename)
                    || !budget.measure(&part.media_type)
                    || !budget.measure(&part.metadata)
                {
                    return None;
                }
                let text = format!(
                    "Inline file: {} bytes retained at {location}, part {index}. \
                     Its contents have not been read or included in this model turn.",
                    bytes.len()
                );
                if !budget.measure(&text) {
                    return None;
                }
                PartContent::Text(text)
            }
            content => {
                // Check before cloning data/metadata; an oversized result must not allocate
                // another unbounded model representation merely to reject it afterward.
                if !budget.measure(part) {
                    return None;
                }
                content.clone()
            }
        };
        model.push(Part {
            content,
            filename: part.filename.clone(),
            media_type: part.media_type.clone(),
            metadata: part.metadata.clone(),
        });
    }
    Some(model)
}

fn render(value: &impl Serialize) -> Option<String> {
    let mut output = Buffer::default();
    serde_json::to_writer(&mut output, value).ok()?;
    let mut value: Value = serde_json::from_slice(&output.bytes).ok()?;
    sort(&mut value);
    serde_json::to_string(&value).ok()
}

fn sort(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            fields.sort_keys();
            for value in fields.values_mut() {
                sort(value);
            }
        }
        Value::Array(values) => values.iter_mut().for_each(sort),
        _ => {}
    }
}

#[derive(Default)]
struct Buffer {
    bytes: Vec<u8>,
}

impl Buffer {
    fn measure(&mut self, value: &impl Serialize) -> bool {
        serde_json::to_writer(self, value).is_ok()
    }
}

impl Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MODEL_RESULT_BYTES.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("the remote result exceeds this model turn's budget"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "result_tests.rs"]
mod tests;
