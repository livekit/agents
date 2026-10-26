//! Decisions for gateway requests.
use serde_json::{Map, Value};

/// `"provider"` or `"provider/<id>"`, trimmed; anything else is refused naming the string.
#[crate::host(export(name = "avatar_model", pid = false))]
pub fn avatar_model_for_host(model: &str) -> Result<(String, Option<String>), String> {
    let (provider, id) = model.split_once('/').unwrap_or((model, ""));
    let provider = provider.trim_matches(crate::tokenize::is_py_space);
    let id = id.trim_matches(crate::tokenize::is_py_space);
    if provider.is_empty() {
        return Err(format!(
            "invalid avatar model string: {} (expected 'provider' or 'provider/<id>')",
            crate::evals::transcript::py_repr(model)
        ));
    }
    Ok((provider.to_owned(), (!id.is_empty()).then(|| id.to_owned())))
}

#[crate::host(export)]
pub fn avatar_payload(
    mut fields: Map<String, Value>,
    avatar_id: Option<&str>,
    options: Map<String, Value>,
) -> Value {
    if let Some(id) = avatar_id.filter(|id| !id.is_empty()) {
        fields.insert("avatar_id".into(), id.into());
    }
    let mut extra = Map::new();
    for (key, value) in options {
        let target = match key.as_str() {
            "image_url" => Some("image_url"),
            "prompt" => Some("prompt"),
            "idle_prompt" => Some("idle_prompt"),
            "idle_timeout" => Some("idle_timeout_s"),
            _ => None,
        };
        if let Some(target) = target {
            fields.insert(target.into(), value);
        } else {
            extra.insert(key, value);
        }
    }
    if !extra.is_empty() {
        fields.insert("extra_kwargs".into(), extra.into());
    }
    fields.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn avatar_request_options() {
        assert_eq!(
            avatar_model_for_host(" lemonslice / id/x "),
            Ok(("lemonslice".into(), Some("id/x".into())))
        );
        assert!(avatar_model_for_host(" /id").unwrap_err().starts_with("invalid avatar model string: "));
        assert_eq!(avatar_model_for_host("lemonslice/"), Ok(("lemonslice".into(), None)));
        let payload = avatar_payload(
            Map::new(),
            Some("id"),
            json!({"idle_timeout": 0, "prompt": "", "other": {"x": true}})
                .as_object()
                .unwrap()
                .clone(),
        );
        assert_eq!(
            payload,
            json!({"avatar_id":"id", "idle_timeout_s":0, "prompt":"", "extra_kwargs":{"other":{"x":true}}})
        );
        assert_eq!(avatar_payload(Map::new(), Some(""), Map::new()), json!({}));
    }
}
