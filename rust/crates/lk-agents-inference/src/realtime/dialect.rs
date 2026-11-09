use std::time::Duration;

use lk_agents_abi::llm::realtime::{RealtimeConfigurationError, RealtimeProvider};
use serde_json::{Value, json};

use lk_agents_abi::llm::realtime::RealtimeCapabilities;

pub use lk_agents_abi::llm::realtime::{NUM_CHANNELS, SAMPLE_RATE};

// xAI's extra inbound events are decoded for every dialect: OpenAI never sends them, and gating
// would make the decoder depend on config.

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
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
    #[serde(untagged)]
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

/// The capability bits a dialect flips after the shared computation; `None` keeps the
/// shared rule's answer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CapabilityOverrides {
    pub message_truncation: Option<bool>,
    pub per_response_tool_choice: Option<bool>,
    pub can_disable_turn_detection: Option<bool>,
    pub supports_say: Option<bool>,
}

/// The OpenAI Realtime protocol version a provider speaks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Protocol {
    #[default]
    Ga,
    /// The pre-GA protocol: a flat `session` and the beta event names.
    Beta,
}

#[derive(Debug, Clone)]
pub struct Dialect {
    pub provider: RealtimeProvider,
    /// `self._provider_label`. Appears in every error message the session emits.
    pub provider_label: &'static str,
    /// `base_url` before the runtime turns it into a socket URL.
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
    /// Applied on the socket by [`crate::realtime::openai::wire::WireShape`].
    pub protocol: Protocol,

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
    /// Codes terminal on this provider beyond [`FATAL_ERROR_CODES`].
    pub extra_fatal_codes: &'static [&'static str],
}

#[derive(serde::Deserialize)]
#[serde(tag = "type")]
enum XaiProviderTool {
    #[serde(rename = "web_search")]
    Web,
    #[serde(rename = "x_search")]
    X {
        #[serde(default)]
        #[serde(rename = "allowed_x_handles")]
        _allowed_x_handles: Option<Vec<String>>,
    },
    #[serde(rename = "file_search")]
    File {
        #[serde(rename = "vector_store_ids")]
        _vector_store_ids: Vec<String>,
        #[serde(default)]
        #[serde(rename = "max_num_results")]
        _max_num_results: Option<i64>,
    },
}

impl XaiProviderTool {
    fn reserved_names(self) -> &'static [&'static str] {
        match self {
            Self::Web => &["web_search", "browse_page"],
            Self::X { .. } => {
                &["x_keyword_search", "x_semantic_search", "x_user_search", "x_thread_fetch"]
            }
            Self::File { .. } => &["collections_search", "file_search"],
        }
    }
}

impl Dialect {
    /// Validate external provider declarations before they replace the current tool set.
    pub fn prepare_tools(
        &self,
        tools: Vec<lk_agents_abi::ToolSpec>,
    ) -> Result<Vec<lk_agents_abi::ToolSpec>, RealtimeConfigurationError> {
        use serde::Deserialize as _;
        // Current OpenAI/Azure realtime dialects support client functions. xAI additionally
        // supports its own server-side declarations; foreign SDK tools stay in the host pool.
        let tools = tools
            .into_iter()
            .filter(|tool| {
                tool.provider_tool.as_ref().is_none_or(|declaration| {
                    self.provider == RealtimeProvider::Xai
                        && declaration.provider.as_deref() == Some("xai")
                })
            })
            .collect::<Vec<_>>();
        let function_names = tools
            .iter()
            .filter(|tool| tool.provider_tool.is_none())
            .map(|tool| tool.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for tool in &tools {
            let Some(declaration) = &tool.provider_tool else { continue };
            let provider = XaiProviderTool::deserialize(&declaration.payload).map_err(|_| {
                RealtimeConfigurationError::Invalid(format!(
                    "xAI provider tool {:?} has an invalid declaration",
                    tool.name
                ))
            })?;
            let conflicts = provider
                .reserved_names()
                .iter()
                .copied()
                .filter(|name| function_names.contains(name))
                .collect::<std::collections::BTreeSet<_>>();
            if !conflicts.is_empty() {
                return Err(RealtimeConfigurationError::Invalid(format!(
                    "xAI provider tool {:?} already uses function name(s) {}. Rename or remove the client functions.",
                    tool.name,
                    conflicts.into_iter().collect::<Vec<_>>().join(", "),
                )));
            }
        }
        Ok(tools)
    }
}

/// `livekit-plugins-openai/realtime`, the protocol everything else here is a delta on.
pub const OPENAI: Dialect = Dialect {
    provider: RealtimeProvider::Openai,
    provider_label: "OpenAI Realtime API",
    default_base_url: lk_agents_abi::llm::realtime::OPENAI_BASE_URL,
    api_key_env: "OPENAI_API_KEY",
    default_model: "gpt-realtime",
    default_voice: lk_agents_abi::llm::realtime::DEFAULT_VOICE,
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
        message_truncation: None,
        per_response_tool_choice: None,
        can_disable_turn_detection: None,
        supports_say: None,
    },
    protocol: Protocol::Ga,
    supports_turn_detection_disabled: true,
    ignore_unknown_function_calls: false,
    defer_transcript_finalization: false,
    discard_silent_responses: false,
    repair_empty_delete_item_id: false,
    anchor_root_at_tail: false,
    force_message_say: false,
    extra_fatal_codes: &[],
};

/// LiveKit Inference: the OpenAI protocol through the gateway, which refuses a session it cannot
/// route with one of three codes no retry fixes.
pub const INFERENCE: Dialect = Dialect {
    provider: RealtimeProvider::Inference,
    provider_label: "LiveKit Inference Realtime",
    extra_fatal_codes: &[
        "unsupported_transcription_model",
        "unsupported_audio_transport",
        "unsupported_audio_format",
    ],
    ..OPENAI
};

/// `livekit-plugins-xai/realtime`, as a value.
pub const XAI: Dialect = Dialect {
    provider: RealtimeProvider::Xai,
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
        message_truncation: None,
        per_response_tool_choice: Some(false),
        // "client turn-taking is not stable during testing"
        can_disable_turn_detection: Some(false),
        // "force_message drives scripted TTS without a follow-up response.create"
        supports_say: Some(true),
    },
    protocol: Protocol::Ga,
    supports_turn_detection_disabled: false,
    ignore_unknown_function_calls: true,
    defer_transcript_finalization: true,
    discard_silent_responses: true,
    repair_empty_delete_item_id: true,
    anchor_root_at_tail: true,
    force_message_say: true,
    extra_fatal_codes: &[],
};

/// `livekit-plugins-azure/realtime`: Azure Voice Live, the beta protocol behind an Azure
/// endpoint. The plugin supplies the endpoint and the voice object.
pub const VOICE_LIVE: Dialect = Dialect {
    provider: RealtimeProvider::VoiceLive,
    provider_label: "Azure Voice Live",
    default_base_url: "",
    api_key_env: "AZURE_VOICE_LIVE_API_KEY",
    default_voice: "en-US-AvaMultilingualNeural",
    // Chosen per model by the plugin: `azure-speech` is the only one text models accept.
    default_transcription_model: None,
    default_turn_detection: Some(TurnDetection::ServerVad {
        threshold: 0.5,
        prefix_padding_ms: 300,
        silence_duration_ms: 500,
        create_response: true,
        interrupt_response: true,
    }),
    default_max_session_duration: None,
    // Voice Live has no `conversation.item.truncate`.
    caps: CapabilityOverrides { message_truncation: Some(false), ..OPENAI.caps },
    protocol: Protocol::Beta,
    ..OPENAI
};

/// The provider's name.
impl std::fmt::Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.provider, f)
    }
}

impl Dialect {
    /// The dialect a provider speaks.
    pub fn of(provider: RealtimeProvider) -> Dialect {
        match provider {
            RealtimeProvider::Openai => OPENAI,
            RealtimeProvider::Xai => XAI,
            RealtimeProvider::Inference => INFERENCE,
            RealtimeProvider::VoiceLive => VOICE_LIVE,
        }
    }
}

impl Dialect {
    /// `_is_fatal_error`, with this provider's own terminal codes.
    pub fn is_fatal(&self, code: Option<&str>, type_: Option<&str>) -> bool {
        is_fatal_error(code, type_)
            || code.or(type_).is_some_and(|c| self.extra_fatal_codes.contains(&c))
    }

    /// The dialect a `base_url` speaks: xAI's host is xAI's protocol, anything else OpenAI's.
    pub fn for_base_url(base_url: &str) -> Dialect {
        let host = base_url.split_once("://").map_or(base_url, |(_, r)| r);
        let host = host.split(['/', '?']).next().unwrap_or(host);
        if host == "api.x.ai" { XAI } else { OPENAI }
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
            message_truncation: c.message_truncation.unwrap_or(true),
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
            supports_overlapping_speech: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_names_use_the_provider_vocabulary() {
        for (name, provider) in [
            ("openai", RealtimeProvider::Openai),
            ("xai", RealtimeProvider::Xai),
            ("inference", RealtimeProvider::Inference),
            ("voice_live", RealtimeProvider::VoiceLive),
        ] {
            let dialect = Dialect::of(RealtimeProvider::from_wire(name).unwrap());
            assert_eq!(dialect.provider, provider);
            assert_eq!(dialect.to_string(), name);
        }
        assert!(RealtimeProvider::from_wire("xai-other").is_none());
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
