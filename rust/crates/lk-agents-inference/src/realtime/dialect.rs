//! What a provider changes about the OpenAI Realtime wire protocol. Most of xAI's deltas are
//! [`Dialect`] fields; the held-transcript machine and `say()` via `force_message` are real state
//! in [`crate::realtime::machine`]. xAI's extra inbound events are decoded for every dialect:
//! OpenAI never sends them, and gating would make the decoder depend on config.

use std::time::Duration;

use serde_json::{Map, Value, json};

use lk_agents_abi::llm::realtime::RealtimeCapabilities;

pub const SAMPLE_RATE: u32 = 24_000;
pub const NUM_CHANNELS: u16 = 1;
/// A string: `f64::INFINITY` would serialise as `null`.
pub const DEFAULT_MAX_OUTPUT_TOKENS: &str = "inf";
/// `_FATAL_ERROR_CODES`: the socket reconnects but every generation fails, so these stop the
/// reconnect loop.
pub const FATAL_ERROR_CODES: &[&str] =
    &["insufficient_quota", "invalid_api_key", "account_deactivated", "billing_hard_limit_reached"];

/// `_is_fatal_error`: `code`, falling back to `type`.
pub fn is_fatal_error(code: Option<&str>, type_: Option<&str>) -> bool {
    let c = code.or(type_);
    c.is_some_and(|c| FATAL_ERROR_CODES.contains(&c))
}

/// Server-side turn detection, in the two shapes the API accepts; `Raw` passes any other shape.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnDetection {
    ServerVad {
        threshold: f64,
        prefix_padding_ms: u32,
        silence_duration_ms: u32,
        create_response: bool,
        interrupt_response: bool,
    },
    SemanticVad {
        eagerness: &'static str,
        create_response: bool,
        interrupt_response: bool,
    },
    Raw(Value),
}

impl TurnDetection {
    pub fn to_json(&self) -> Value {
        match self {
            TurnDetection::ServerVad {
                threshold,
                prefix_padding_ms,
                silence_duration_ms,
                create_response,
                interrupt_response,
            } => json!({
                "type": "server_vad",
                "threshold": threshold,
                "prefix_padding_ms": prefix_padding_ms,
                "silence_duration_ms": silence_duration_ms,
                "create_response": create_response,
                "interrupt_response": interrupt_response,
            }),
            TurnDetection::SemanticVad { eagerness, create_response, interrupt_response } => {
                json!({
                    "type": "semantic_vad",
                    "create_response": create_response,
                    "eagerness": eagerness,
                    "interrupt_response": interrupt_response,
                })
            }
            TurnDetection::Raw(v) => v.clone(),
        }
    }

    /// `_server_turn_taking_enabled`: detects turns and answers them. `create_response is not
    /// False`, so an omitted key counts as enabled.
    pub fn server_turn_taking(td: Option<&TurnDetection>) -> bool {
        match td {
            None => false,
            Some(TurnDetection::ServerVad { create_response, .. })
            | Some(TurnDetection::SemanticVad { create_response, .. }) => *create_response,
            Some(TurnDetection::Raw(v)) => v.get("create_response") != Some(&Value::Bool(false)),
        }
    }
}

/// The three capability bits a dialect flips after the shared computation; `None` keeps the
/// shared rule's answer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapabilityOverrides {
    pub per_response_tool_choice: Option<bool>,
    pub can_disable_turn_detection: Option<bool>,
    pub supports_say: Option<bool>,
}

/// One `(nested path, top-level name)` promotion in a `session.update` payload.
pub type Promotion = (&'static [&'static str], &'static str);

#[derive(Debug, Clone)]
#[lk_agents_abi::host(string)]
pub struct Dialect {
    /// `self._provider_label`. Appears in every error message the session emits.
    pub provider_label: &'static str,
    /// `base_url` before [`crate::realtime::wire::process_base_url`] turns it into a socket URL.
    pub default_base_url: &'static str,
    /// The env var `api_key=None` falls back to; read by the host, named in its error message.
    pub api_key_env: &'static str,
    pub default_model: &'static str,
    pub default_voice: &'static str,
    /// `None` means the dialect has no transcription default, not "transcription off".
    pub default_transcription_model: Option<&'static str>,
    pub default_turn_detection: Option<TurnDetection>,
    /// `DEFAULT_MAX_SESSION_DURATION`. `None` = never recycle (xAI); OpenAI recycles at 20 min.
    pub default_max_session_duration: Option<Duration>,
    /// `Some` when the provider does not let the caller choose (xAI: `["audio", "text"]`).
    pub forced_modalities: Option<&'static [&'static str]>,
    pub caps: CapabilityOverrides,

    // --- payload shape ---
    /// Fields lifted out of the nested `audio` block to the top level of `session`.
    pub promote: &'static [Promotion],
    /// Drop `session.audio` entirely when the promotions emptied both of its halves.
    pub prune_empty_audio: bool,

    // --- behaviour switches, each one an `if` in `machine.rs` ---
    /// `session(turn_detection_disabled=True)` is honoured. xAI ignores the argument.
    pub supports_turn_detection_disabled: bool,
    /// A `function_call` naming no known tool is dropped with a warning (xAI's server-side tools).
    pub ignore_unknown_function_calls: bool,
    /// The provider re-sends a whole transcript on every commit, so `...transcription.completed`
    /// is not final. Hold it and finalise on the next turn boundary.
    pub defer_transcript_finalization: bool,
    /// A response the server created and then left silent is discarded when the next one arrives.
    pub discard_silent_responses: bool,
    /// The delete ack arrives with `item_id: ""`; resolve it against the oldest in-flight delete.
    pub repair_empty_delete_item_id: bool,
    /// `previous_item_id: null` on `conversation.item.added` means "at the tail", not "at the
    /// root" (xAI only).
    pub anchor_root_at_tail: bool,
    /// `say()` is xAI's `force_message`, answered with no `metadata.client_event_id` (hence the
    /// FIFO tag queue in [`crate::realtime::machine`]). Off: `say()` is refused.
    pub force_message_say: bool,
}

/// `livekit-plugins-openai/realtime`, the protocol everything else here is a delta on.
pub const OPENAI: Dialect = Dialect {
    provider_label: "OpenAI Realtime API",
    default_base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    default_model: "gpt-realtime",
    default_voice: "marin",
    default_transcription_model: Some("gpt-4o-mini-transcribe"),
    // `DEFAULT_TURN_DETECTION`, "got from a default session from their API".
    default_turn_detection: Some(TurnDetection::SemanticVad {
        eagerness: "medium",
        create_response: true,
        interrupt_response: true,
    }),
    default_max_session_duration: Some(Duration::from_secs(20 * 60)),
    forced_modalities: None,
    caps: CapabilityOverrides {
        per_response_tool_choice: None,
        can_disable_turn_detection: None,
        supports_say: None,
    },
    promote: &[],
    prune_empty_audio: false,
    supports_turn_detection_disabled: true,
    ignore_unknown_function_calls: false,
    defer_transcript_finalization: false,
    discard_silent_responses: false,
    repair_empty_delete_item_id: false,
    anchor_root_at_tail: false,
    force_message_say: false,
};

/// `livekit-plugins-xai/realtime`, as a value.
pub const XAI: Dialect = Dialect {
    provider_label: "xAI Realtime API",
    // Already `wss://` with a path; `process_base_url` only appends `?model=`.
    default_base_url: "wss://api.x.ai/v1/realtime",
    api_key_env: "XAI_API_KEY",
    default_model: "grok-voice-latest",
    // Capitalised as the reference ships it.
    default_voice: "Ara",
    default_transcription_model: Some("grok-transcribe"),
    // Server VAD: OpenAI's Azure defaults plus an explicit `interrupt_response`.
    default_turn_detection: Some(TurnDetection::ServerVad {
        threshold: 0.5,
        prefix_padding_ms: 300,
        silence_duration_ms: 200,
        create_response: true,
        interrupt_response: true,
    }),
    // Never recycles. A 20-minute call on xAI holds one socket; on OpenAI it holds two.
    default_max_session_duration: None,
    forced_modalities: Some(&["audio", "text"]),
    caps: CapabilityOverrides {
        per_response_tool_choice: Some(false),
        // "client turn-taking is not stable during testing"
        can_disable_turn_detection: Some(false),
        // "force_message drives scripted TTS without a follow-up response.create"
        supports_say: Some(true),
    },
    promote: &[
        (&["audio", "output", "voice"], "voice"),
        (&["audio", "input", "turn_detection"], "turn_detection"),
    ],
    prune_empty_audio: true,
    supports_turn_detection_disabled: false,
    ignore_unknown_function_calls: true,
    defer_transcript_finalization: true,
    discard_silent_responses: true,
    repair_empty_delete_item_id: true,
    anchor_root_at_tail: true,
    force_message_say: true,
};

/// A dialect name this build does not speak; the message lists the ones it does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownDialect(pub String);

impl std::fmt::Display for UnknownDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown realtime dialect {:?}; this port speaks \"openai\" and \"xai\"", self.0)
    }
}

impl std::error::Error for UnknownDialect {}

/// `"openai"` or `"xai"`.
impl std::fmt::Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.provider_label == XAI.provider_label { "xai" } else { "openai" })
    }
}

impl std::str::FromStr for Dialect {
    type Err = UnknownDialect;

    fn from_str(name: &str) -> Result<Self, UnknownDialect> {
        match name {
            "openai" => Ok(OPENAI),
            "xai" => Ok(XAI),
            other => Err(UnknownDialect(other.to_owned())),
        }
    }
}

impl Dialect {
    /// The map already holds only what was given (pydantic's `model_fields_set`).
    pub fn rewrite_session(&self, session: &mut Map<String, Value>) {
        for (path, dest) in self.promote {
            if let Some(v) = remove_path(session, path) {
                session.insert((*dest).to_owned(), v);
            }
        }

        if self.prune_empty_audio
            && let Some(Value::Object(audio)) = session.get("audio")
            && audio.values().all(|half| matches!(half, Value::Object(m) if m.is_empty()))
        {
            session.remove("audio");
        }
    }

    /// The capability record for a session. `turn_detection_given` is `is_given(turn_detection)`
    /// at construction, before the default is applied.
    pub fn capabilities(
        &self,
        turn_detection: Option<&TurnDetection>,
        turn_detection_given: bool,
        audio_output: bool,
        user_transcription: bool,
    ) -> RealtimeCapabilities {
        let c = self.caps;
        RealtimeCapabilities {
            message_truncation: true,
            turn_detection: TurnDetection::server_turn_taking(turn_detection),
            // Same as `turn_detection` until the dialects say otherwise (R-11).
            auto_turn_reply_generation: TurnDetection::server_turn_taking(turn_detection),
            user_transcription,
            auto_tool_reply_generation: false,
            audio_output,
            manual_function_calls: true,
            can_disable_turn_detection: c
                .can_disable_turn_detection
                .unwrap_or(!turn_detection_given),
            mutable_chat_context: true,
            mutable_instructions: true,
            mutable_tools: true,
            per_response_tool_choice: c.per_response_tool_choice.unwrap_or(true),
            supports_say: c.supports_say.unwrap_or(false),
        }
    }
}

/// Remove `path` from a nested map, leaving the (possibly empty) parents for `prune_empty_audio`.
fn remove_path(root: &mut Map<String, Value>, path: &[&str]) -> Option<Value> {
    let (last, parents) = path.split_last()?;
    let mut cur = root;
    for step in parents {
        cur = match cur.get_mut(*step) {
            Some(Value::Object(m)) => m,
            _ => return None,
        };
    }
    cur.remove(*last)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_session() -> Map<String, Value> {
        let audio_format = json!({"rate": SAMPLE_RATE, "type": "audio/pcm"});
        let Value::Object(m) = json!({
            "type": "realtime",
            "model": "grok-voice-latest",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": audio_format,
                    "noise_reduction": null,
                    "transcription": {"model": "grok-transcribe"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {"format": audio_format, "speed": 1.0, "voice": "Ara"},
            },
            "max_output_tokens": "inf",
        }) else {
            unreachable!()
        };
        m
    }

    // Catches: the xAI promotion implemented as a copy rather than a move.
    #[test]
    fn xai_moves_voice_and_turn_detection_to_the_top_level() {
        let mut s = full_session();
        XAI.rewrite_session(&mut s);

        assert_eq!(s["voice"], json!("Ara"));
        assert_eq!(s["turn_detection"], json!({"type": "server_vad"}));
        let audio = &s["audio"];
        assert!(audio["output"].get("voice").is_none(), "voice must be gone from audio.output");
        assert!(audio["input"].get("turn_detection").is_none());
        // and what the promotion did NOT touch is still nested
        assert_eq!(audio["output"]["speed"], json!(1.0));
        assert_eq!(audio["input"]["transcription"], json!({"model": "grok-transcribe"}));
    }

    // Catches: shipping an empty `audio` block for a voice-only update.
    #[test]
    fn a_voice_only_update_ships_no_audio_block_at_all() {
        let Value::Object(mut s) =
            json!({"type": "realtime", "audio": {"input": {}, "output": {"voice": "Eve"}}})
        else {
            unreachable!()
        };
        XAI.rewrite_session(&mut s);
        assert_eq!(
            Value::Object(s),
            json!({"type": "realtime", "voice": "Eve"}),
            "the whole audio block goes once the promotion empties it"
        );
    }

    // Catches: promoting `voice` for OpenAI, which rejects a top-level `voice`.
    #[test]
    fn openai_promotes_nothing() {
        let mut s = full_session();
        let before = s.clone();
        OPENAI.rewrite_session(&mut s);
        assert_eq!(s, before);
    }

    // Catches: computing xAI's `can_disable_turn_detection` instead of forcing `false`.
    #[test]
    fn xai_pins_three_capabilities_that_openai_derives() {
        let td = XAI.default_turn_detection.clone();
        let xai = XAI.capabilities(td.as_ref(), false, true, true);
        assert!(!xai.can_disable_turn_detection);
        assert!(!xai.per_response_tool_choice);
        assert!(xai.supports_say);

        let td = OPENAI.default_turn_detection.clone();
        let oai = OPENAI.capabilities(td.as_ref(), false, true, true);
        assert!(oai.can_disable_turn_detection, "nothing was given, so it can still be disabled");
        assert!(oai.per_response_tool_choice);
        assert!(!oai.supports_say);
        // the shared half agrees
        assert_eq!(xai.turn_detection, oai.turn_detection);
        assert!(xai.message_truncation && xai.manual_function_calls);
    }

    // Catches: `_server_turn_taking_enabled` written as `create_response == Some(true)`.
    #[test]
    fn an_omitted_create_response_still_counts_as_server_turn_taking() {
        let raw = TurnDetection::Raw(json!({"type": "server_vad"}));
        assert!(TurnDetection::server_turn_taking(Some(&raw)));
        let off = TurnDetection::Raw(json!({"type": "server_vad", "create_response": false}));
        assert!(!TurnDetection::server_turn_taking(Some(&off)));
        assert!(!TurnDetection::server_turn_taking(None));
    }

    // Catches: reading only `error.code` when classifying a fatal error.
    #[test]
    fn a_fatal_code_is_recognised_in_either_field() {
        assert!(is_fatal_error(Some("insufficient_quota"), None));
        assert!(is_fatal_error(None, Some("invalid_api_key")));
        assert!(!is_fatal_error(Some("server_error"), Some("invalid_request_error")));
        assert!(!is_fatal_error(None, None));
    }
}
