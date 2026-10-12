//! Host-independent tool argument decoding and bounded recovery of incomplete model output.
use serde_json::{Map, Value};
use std::sync::LazyLock;

const MAX_DEPTH: usize = 64;
static TOKENS: LazyLock<Vec<regex::Regex>> = LazyLock::new(|| {
    [
        r"<\|[^<>|]{0,40}\|>",
        r"<\|[^<>a-zA-Z0-9_]{0,10}",
        r"[^<>a-zA-Z0-9_]{0,10}\|>",
        r"<(?:start|end)_of_turn>",
    ]
    .iter()
    .map(|p| regex::Regex::new(p).expect("constant pattern"))
    .collect()
});

pub fn strip_template_tokens(value: Value) -> Value {
    match value {
        Value::String(mut text) => {
            for pattern in TOKENS.iter() {
                text = pattern.replace_all(&text, "").into_owned();
            }
            Value::String(crate::tokenize::py_strip(&text).to_owned())
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(strip_template_tokens)
                .filter(|v| !v.is_null() && v.as_str() != Some(""))
                .collect(),
        ),
        Value::Object(values) => {
            Value::Object(values.into_iter().map(|(k, v)| (k, strip_template_tokens(v))).collect())
        }
        value => value,
    }
}

#[derive(Debug)]
pub struct ParsedArguments {
    pub arguments: Map<String, Value>,
    /// Hosts may log this diagnostic; it never changes parsing behavior.
    pub repair: Option<Repair>,
}

/// A malformed payload's repair: the strict parser's error and the cleaned value it became.
#[derive(Debug)]
pub struct Repair {
    pub error: String,
    pub repaired: Value,
}
fn prefix(text: &str) -> String {
    text.chars().take(200).collect()
}

/// `parse_function_arguments`: the model's JSON arguments as an object, repaired when malformed
/// (the repair is logged, never raised).
#[crate::host(export(pid = false, name = "parse_function_arguments"))]
pub fn parse_logged(raw: &str) -> Result<Map<String, Value>, String> {
    let parsed = parse(raw)?;
    if let Some(Repair { error, repaired }) = &parsed.repair {
        // The reference's `extra`, keyed `lk.pii.*` so the collector redacts it.
        let raw_arguments: String = raw.chars().take(500).collect();
        let repaired = match repaired {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        log::warn!(
            target: "livekit.agents",
            "lk.pii.raw_arguments" = raw_arguments.as_str(),
            "lk.pii.repaired" = repaired.as_str(),
            "lk.pii.error" = error.as_str();
            "repaired malformed function-call JSON arguments"
        );
    }
    Ok(parsed.arguments)
}

pub fn parse(raw: &str) -> Result<ParsedArguments, String> {
    let (mut value, mut repair) = match serde_json::from_str::<Value>(raw) {
        Ok(value) => (value, None),
        Err(error) => {
            let value = crate::llm::json_repair::repair(raw)?;
            if value.as_str() == Some("") {
                return Err(format!(
                    "could not parse function arguments as JSON: {error}: {}",
                    prefix(raw)
                ));
            }
            let value = strip_template_tokens(value);
            (value.clone(), Some(Repair { error: error.to_string(), repaired: value }))
        }
    };
    for _ in 0..MAX_DEPTH {
        match value {
            Value::String(ref nested) => {
                value = serde_json::from_str(nested).map_err(|_| {
                    format!("function arguments decoded to a non-JSON string: {}", prefix(nested))
                })?;
            }
            Value::Null => return Ok(ParsedArguments { arguments: Map::new(), repair: repair.take() }),
            Value::Object(arguments) => return Ok(ParsedArguments { arguments, repair: repair.take() }),
            ref other => {
                let kind = match other {
                    Value::Array(_) => "list",
                    Value::Bool(_) => "bool",
                    Value::Number(n) if n.is_f64() => "float",
                    _ => "int",
                };
                return Err(format!(
                    "expected dict from function arguments, got {kind}: {}",
                    prefix(raw)
                ));
            }
        }
    }
    Err("function arguments exceed maximum encoded nesting depth".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn strict_and_nested_preserve_legitimate_markup() {
        let raw = r#"{"x":"<|im_start|>keep", "n":1}"#;
        for text in [raw.to_owned(), serde_json::to_string(raw).unwrap()] {
            let result = parse(&text).unwrap();
            assert_eq!(
                result.arguments,
                json!({"x":"<|im_start|>keep","n":1}).as_object().unwrap().clone()
            );
            assert!(result.repair.is_none());
        }
        assert!(parse("null").unwrap().arguments.is_empty());
    }
    #[test]
    fn recover_common_model_errors_without_a_host() {
        for raw in [
            "{'x':1}",
            "{x:1}",
            "{\"x\":1,}",
            "{\"x\":1",
            "```json\n{\"x\":1}\n```",
            "{/* comment */\"x\":1}",
        ] {
            let result = parse(raw).unwrap();
            assert_eq!(result.arguments["x"], json!(1));
            assert!(result.repair.is_some());
        }
        assert_eq!(
            parse(r#"{"a":1 "b":[true false,]}"#).unwrap().arguments["b"],
            json!([true, false])
        );
        assert_eq!(
            parse("{'x':'<|im_start|>value', 'a':['<|im_end|>',null,0,false]}").unwrap().arguments
                ["a"],
            json!([0, false])
        );
        assert_eq!(parse("{'x':'\\uD83D\\uDE00'}").unwrap().arguments["x"], json!("😀"));
    }
    // Catches: the repair warning losing the reference's `lk.pii.*` extras (the raw payload, the
    // repaired value, the strict error), which is all a reader of that warning has to go on.
    #[test]
    fn a_repair_is_logged_with_the_raw_payload_the_repair_and_the_error() {
        struct Keys(std::sync::Mutex<Vec<(String, String)>>);
        impl log::Log for Keys {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &log::Record<'_>) {
                struct Collect<'a>(&'a mut Vec<(String, String)>);
                impl<'kvs> log::kv::VisitSource<'kvs> for Collect<'_> {
                    fn visit_pair(&mut self, k: log::kv::Key<'kvs>, v: log::kv::Value<'kvs>) -> Result<(), log::kv::Error> {
                        self.0.push((k.to_string(), v.to_string()));
                        Ok(())
                    }
                }
                record.key_values().visit(&mut Collect(&mut self.0.lock().unwrap())).unwrap();
            }
            fn flush(&self) {}
        }
        let keys: &'static Keys = Box::leak(Box::new(Keys(Default::default())));
        log::set_logger(keys).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
        parse_logged("{'x':1}").unwrap();
        let seen = keys.0.lock().unwrap().clone();
        let get = |k: &str| seen.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
        assert_eq!(get("lk.pii.raw_arguments"), Some("{'x':1}"));
        assert_eq!(get("lk.pii.repaired"), Some(r#"{"x":1}"#));
        assert!(get("lk.pii.error").is_some_and(|e| !e.is_empty()), "{seen:?}");
    }
    #[test]
    fn invalid_and_deep_input_fails_boundedly() {
        assert!(parse(&format!("{{\"x\":{}", "[".repeat(1000))).is_err());
        assert!(parse("{}{}").unwrap().arguments.is_empty());
        for raw in ["", "garbage", "[1]", "42", "\"not json\""] {
            assert!(parse(raw).is_err(), "{raw}");
        }
    }
}
