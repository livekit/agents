//! Managed endpoint attachment boundary. A URL part carries an opaque application reference;
//! resolving or authorizing that reference belongs to the application. Inline uploads have no
//! durable reference and cannot enter the thread's task or inbox records.

use serde_json::Value;

use lk_agents_abi::storage::StorageError;

/// Check declared part shapes before the upstream protobuf-JSON adapter can discard fields.
/// In particular, that adapter chooses the first content field when several are supplied.
/// This does not inspect a reference's scheme, existence, availability or authorization.
pub fn validate_wire_parts(message: &Value) -> Result<(), String> {
    let Some(parts) = message.get("parts") else { return Ok(()) };
    let parts = parts.as_array().ok_or("message.parts must be an array")?;
    for part in parts {
        let part = part.as_object().ok_or("each message part must be an object")?;
        let contents = ["text", "raw", "url", "data"];
        let declared: Vec<_> = contents.into_iter().filter(|key| part.contains_key(*key)).collect();
        if declared.len() != 1 {
            return Err("a part must declare exactly one of text, raw, url, data".into());
        }
        let key = declared[0];
        match key {
            "data" if !part[key].is_object() => return Err("part.data must be an object".into()),
            "text" | "url" | "raw" if !part[key].is_string() => {
                return Err(format!("part.{key} must be a string"));
            }
            _ => {}
        }
        for key in ["filename", "mediaType"] {
            if part.get(key).is_some_and(|v| !v.is_null() && !v.is_string()) {
                return Err(format!("part.{key} must be a string"));
            }
        }
        if part.get("metadata").is_some_and(|v| !v.is_null() && !v.is_object()) {
            return Err("part.metadata must be an object".into());
        }
        if key == "raw" {
            return Err(inline_upload_error());
        }
    }
    Ok(())
}

/// Native callers use the same no-inline boundary as JSON-RPC before durable receipt.
pub fn validate_received_parts(message: &a2a::Message) -> Result<(), StorageError> {
    if message.parts.iter().any(|part| matches!(part.content, a2a::PartContent::Raw(_))) {
        return Err(StorageError::Invalid(inline_upload_error()));
    }
    Ok(())
}

fn inline_upload_error() -> String {
    "inline attachment bytes cannot be stored with a thread; supply an application reference in a URL part".into()
}
