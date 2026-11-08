//! Every string the intent judge's model sees, the reading of its answer, and the scoring of
//! judgments: a wording change moves verdict rates. Pure: data in, data out.

use lk_agents_abi::errors::Invalid;
use lk_agents_abi::evals::{EvaluationScore, JudgeVerdict};
use lk_agents_abi::{ToolFlags, ToolSpec};
use serde_json::{Value, json};

pub const INTENT_SYSTEM_PROMPT: &str = "You are a test evaluator for conversational agents.\n\
You will be shown a message and a target intent. Determine whether the message accomplishes the intent.\n\
Only respond by calling the `check_intent(success: bool, reason: str)` function with your final judgment.\n\
Be strict: if the message does not clearly fulfill the intent, return `success = False` and explain why.";

pub fn intent_prompt(intent: &str, message: &str) -> String {
    format!(
        "Check if the following message fulfills the given intent.\n\nIntent:\n{intent}\n\nMessage:\n{message}"
    )
}

pub fn check_intent_tool() -> ToolSpec {
    tool(
        "check_intent",
        "Determines whether the message correctly fulfills the given intent.",
        json!({
            "properties": {
                "success": {
                    "description": "Whether the message satisfies the intent.",
                    "title": "Success",
                    "type": "boolean",
                },
                "reason": {
                    "description": "A concise explanation justifying the result.",
                    "title": "Reason",
                    "type": "string",
                },
            },
            "required": ["success", "reason"],
            "title": "CheckIntentArgs",
            "type": "object",
        }),
    )
}

fn tool(name: &str, description: &str, json_schema: Value) -> ToolSpec {
    ToolSpec {
        provider_tool: None,
        name: name.into(),
        description: description.into(),
        json_schema,
        flags: ToolFlags::default(),
    }
}

/// `temperature` is pinned to 0 for every model that accepts it.
pub fn request_extra(model: &str) -> lk_agents_abi::Extra {
    let mut extra = lk_agents_abi::Extra::new();
    if !crate::llm::models::unsupported_params(model, false, None, None).contains(&"temperature") {
        extra.insert("temperature".into(), json!(0.0));
    }
    extra
}

/// `check_intent`'s arguments: repaired JSON, then a lax boolean.
pub fn parse_intent(arguments: &str) -> Result<(bool, String), Invalid> {
    let value = lk_agents_core::llm::json_repair::repair(arguments)
        .map_err(|too_deep| Invalid(too_deep.to_string()))?;
    let success = match value.get("success") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) if n.as_i64() == Some(1) => true,
        Some(Value::Number(n)) if n.as_i64() == Some(0) => false,
        Some(Value::String(s)) => match s.to_ascii_lowercase().as_str() {
            "t" | "true" | "on" | "y" | "yes" | "1" => true,
            "f" | "false" | "off" | "n" | "no" | "0" => false,
            _ => return Err(Invalid(format!("success is not a boolean: {s:?}"))),
        },
        other => {
            return Err(Invalid(format!("success is not a boolean: {}", shown(other))));
        }
    };
    let reason = match value.get("reason") {
        Some(Value::String(s)) => s.clone(),
        other => {
            return Err(Invalid(format!("reason must be a string, got {}", shown(other))));
        }
    };
    Ok((success, reason))
}

fn shown(value: Option<&Value>) -> String {
    value.map_or_else(|| "nothing".to_owned(), Value::to_string)
}

/// Pass scores 1, maybe 0.5, fail 0. Empty: score 0 while the vacuous aggregates hold.
pub fn score(verdicts: &[JudgeVerdict]) -> EvaluationScore {
    let mut points = 0.0;
    let (mut passed, mut failed) = (0usize, 0usize);
    for verdict in verdicts {
        match verdict {
            JudgeVerdict::Pass => {
                points += 1.0;
                passed += 1;
            }
            JudgeVerdict::Maybe => points += 0.5,
            JudgeVerdict::Fail => failed += 1,
        }
    }
    let n = verdicts.len();
    EvaluationScore {
        score: if n == 0 { 0.0 } else { points / n as f64 },
        all_passed: passed == n,
        any_passed: passed > 0,
        majority_passed: n == 0 || passed * 2 > n,
        none_failed: failed == 0,
    }
}
