//! Conversion between the upstream A2A wire and the one generated host message schema.

use lk_agents_abi::a2a::client::{
    AgentClientError, Message, MessageInput, MessageMetadata, MessagePart,
};
use lk_agents_storage::run::EXTENSION_URI;
use serde_json::{Map, Value};

pub fn message_to_wire(
    input: MessageInput,
    message_id: String,
) -> Result<a2a::Message, AgentClientError> {
    let input = match input {
        MessageInput::Text(text) => Message {
            parts: vec![MessagePart::TextPart { text, metadata: None }],
            metadata: None,
            message_id: None,
            task_id: None,
            context_id: None,
            role: "user".into(),
            channel: None,
            received_at: None,
            raw: None,
        },
        MessageInput::Message(message) => message,
    };
    let role = match input.role.as_str() {
        "user" => a2a::Role::User,
        "agent" => a2a::Role::Agent,
        _ => {
            return Err(AgentClientError::InvalidInput(
                "message.role must be user or agent".into(),
            ));
        }
    };
    Ok(a2a::Message {
        message_id: input.message_id.unwrap_or(message_id),
        role,
        parts: input.parts.into_iter().map(part_to_wire).collect::<Result<_, _>>()?,
        metadata: input
            .metadata
            .map(metadata_to_wire)
            .transpose()?
            .map(|metadata| metadata.into_iter().collect()),
        task_id: input.task_id,
        context_id: input.context_id,
        extensions: None,
        reference_task_ids: None,
    })
}

pub fn message_from_wire(message: a2a::Message) -> Message {
    let mut application: Map<String, Value> =
        message.metadata.unwrap_or_default().into_iter().collect();
    let mut take = |key: &str| {
        let key = format!("{EXTENSION_URI}/{key}");
        let value = application.get(&key).and_then(Value::as_str).map(str::to_owned);
        if value.is_some() {
            application.remove(&key);
        }
        value
    };
    let mut metadata = MessageMetadata {
        kind: take("kind"),
        channel: take("channel"),
        verbatim: take("verbatim"),
        answer_tool: take("answerTool"),
        merged_into: take("mergedInto"),
        reason: take("reason"),
        interrupted: None,
        interruption_reason: take("interruptionReason"),
        application: None,
    };
    let interrupted_key = format!("{EXTENSION_URI}/interrupted");
    metadata.interrupted = application.get(&interrupted_key).and_then(Value::as_bool);
    if metadata.interrupted.is_some() {
        application.remove(&interrupted_key);
    }
    metadata.application = Some(application);
    Message {
        parts: message.parts.into_iter().map(part_from_wire).collect(),
        metadata: Some(metadata),
        message_id: Some(message.message_id),
        task_id: message.task_id,
        context_id: message.context_id,
        channel: None,
        received_at: None,
        raw: None,
        role: match message.role {
            a2a::Role::User => "user",
            a2a::Role::Agent => "agent",
            a2a::Role::Unspecified => "user",
        }
        .into(),
    }
}

fn part_to_wire(part: MessagePart) -> Result<a2a::Part, AgentClientError> {
    let (mut part, metadata) = match part {
        MessagePart::TextPart { text, metadata } => (a2a::Part::text(text), metadata),
        MessagePart::DataPart { data, metadata } => (a2a::Part::data(data), metadata),
        MessagePart::FilePart { url, data, filename, media_type, metadata } => {
            let mut part = match (url, data) {
                (Some(url), None) if !url.is_empty() => a2a::Part::url(url),
                (None, Some(data)) => a2a::Part::raw(data),
                _ => {
                    return Err(AgentClientError::InvalidInput(
                        "a file part needs exactly one of url or data".into(),
                    ));
                }
            };
            part.filename = filename;
            part.media_type = media_type;
            (part, metadata)
        }
    };
    part.metadata = metadata.map(|metadata| metadata.into_iter().collect());
    Ok(part)
}

pub fn part_from_wire(part: a2a::Part) -> MessagePart {
    let metadata = part.metadata.map(|metadata| metadata.into_iter().collect());
    match part.content {
        a2a::PartContent::Text(text) => MessagePart::TextPart { text, metadata },
        a2a::PartContent::Data(data) => MessagePart::DataPart { data, metadata },
        a2a::PartContent::Url(url) => MessagePart::FilePart {
            url: Some(url),
            data: None,
            filename: part.filename,
            media_type: part.media_type,
            metadata,
        },
        a2a::PartContent::Raw(data) => MessagePart::FilePart {
            url: None,
            data: Some(data),
            filename: part.filename,
            media_type: part.media_type,
            metadata,
        },
    }
}

pub fn metadata_to_wire(metadata: MessageMetadata) -> Result<Map<String, Value>, AgentClientError> {
    let mut output = metadata.application.unwrap_or_default();
    if output.keys().any(|key| {
        [
            "kind",
            "channel",
            "verbatim",
            "answerTool",
            "mergedInto",
            "reason",
            "interrupted",
            "interruptionReason",
        ]
        .iter()
        .any(|field| key == &format!("{EXTENSION_URI}/{field}"))
    }) {
        return Err(AgentClientError::InvalidInput(
            "application metadata cannot define typed LiveKit keys".into(),
        ));
    }
    for (key, value) in [
        ("kind", metadata.kind),
        ("channel", metadata.channel),
        ("verbatim", metadata.verbatim),
        ("answerTool", metadata.answer_tool),
        ("mergedInto", metadata.merged_into),
        ("reason", metadata.reason),
        ("interruptionReason", metadata.interruption_reason),
    ] {
        if let Some(value) = value {
            output.insert(format!("{EXTENSION_URI}/{key}"), Value::String(value));
        }
    }
    if let Some(interrupted) = metadata.interrupted {
        output.insert(format!("{EXTENSION_URI}/interrupted"), Value::Bool(interrupted));
    }
    Ok(output)
}
