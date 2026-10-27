//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.

#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]

use napi::Env;
use napi_derive::napi;

/// `CamelCase` -> `snake_case`, matching the reference's two-regex form exactly:
/// `([A-Z]+)([A-Z][a-z])` then `([a-z0-9])([A-Z])`, lowercased.
#[napi(js_name = "camelToSnakeCase")]
pub fn camel_to_snake_case(name: String) -> String {
    lk_agents_abi::camel_to_snake_case(&name)
}

/// Is this a LiveKit Cloud URL? Hostname suffix only, so a path or query cannot fake it.
#[napi(js_name = "isCloud")]
pub fn is_cloud(url: String) -> bool {
    lk_agents_abi::is_cloud_url(&url)
}

/// `<expression value="X"/>` and `<sound value="X"/>` (wrapping form included) -> `[X]`.
#[napi(js_name = "convertExpressionTags")]
pub fn convert_expression_tags(text: String) -> String {
    lk_agents_abi::tts::markup::convert_expression_tags(&text)
}

/// Strip XML markup tags and collect `(type, value)` pairs in one pass. `value` is a wrapping
/// tag's inner text, else its first quoted attribute, else `""`. Square brackets are left alone.
#[napi(js_name = "extractAndStrip")]
pub fn extract_and_strip(text: String, xml_tags: Vec<String>) -> (String, Vec<(String, String)>) {
    let xml_tags: Vec<&str> = xml_tags.iter().map(String::as_str).collect();
    lk_agents_abi::tts::markup::extract_and_strip(&text, &xml_tags)
}

/// Strip the union of every provider's expressive XML markup (provider-agnostic): the sinks no
/// longer know the provider. Square brackets are prose here and are left alone.
#[napi(js_name = "splitAllMarkup")]
pub fn split_all_markup(text: String) -> (String, Vec<crate::generated::records::ExpressiveTag>) {
    { let (t0, t1,) = lk_agents_abi::tts::markup::split_all_markup(&text); (t0, t1.into_iter().map(|v| crate::generated::records::ExpressiveTag::from(&v)).collect::<Vec<_>>()) }
}

/// `split_all_markup` returning only the clean text (tags discarded).
#[napi(js_name = "stripAllMarkup")]
pub fn strip_all_markup(text: String) -> String {
    lk_agents_abi::tts::markup::strip_all_markup(&text)
}

/// Strip only the `<expr/>` dialect, leaving all other markup untouched.
#[napi(js_name = "stripExprMarkup")]
pub fn strip_expr_markup(text: String) -> String {
    lk_agents_abi::tts::markup::strip_expr_markup(&text)
}

/// `expression_attribute` as the one-entry `{ATTRIBUTE_TRANSCRIPTION_EXPRESSION: json}` dict the
/// reference returns.
#[napi(js_name = "expressionAttribute")]
pub fn expression_attribute(env: Env, tags: Vec<crate::generated::records::ExpressiveTag>) -> napi::Result<Option<std::collections::HashMap<String, String>>> {
    crate::host_error::run(&env, || {
        let tags = tags.into_iter().map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).collect::<Result<Vec<_>, _>>()?;
        Ok(lk_agents_abi::tts::markup::expression_attribute_map(tags).map(|v| v.into_iter().map(|(k, v)| (k, v)).collect::<std::collections::HashMap<_, _>>()))
    })
}

/// `drop_bracket_cues` as a plan over token texts, for a host whose `TimedString` cannot cross.
/// Returns `(emit, keep)`; `None` text means reuse the host's token object.
#[napi(js_name = "dropBracketCues")]
pub fn drop_bracket_cues(texts: Vec<String>, end_of_stream: bool) -> (Vec<(i64, Option<String>)>, Vec<(i64, Option<String>)>) {
    { let (t0, t1,) = lk_agents_abi::tts::markup::drop_bracket_cues_plan(&texts, end_of_stream); (t0.into_iter().map(|v| { let (t0, t1,) = v; (i64::try_from(t0).unwrap_or(i64::MAX), t1) }).collect::<Vec<_>>(), t1.into_iter().map(|v| { let (t0, t1,) = v; (i64::try_from(t0).unwrap_or(i64::MAX), t1) }).collect::<Vec<_>>()) }
}

/// `to_xml` over owned pairs; `attrs` is a list of pairs because `_summarize`'s two renderings
/// order `name` and `call_id` differently.
#[napi(js_name = "toXml")]
pub fn to_xml(tag_name: String, content: Option<String>, attrs: Option<Vec<(String, String)>>) -> String {
    lk_agents_abi::tts::markup::to_xml_of(&tag_name, content.as_deref(), attrs)
}

/// The `assistant` message body the summary is stored as.
#[napi(js_name = "chatHistorySummaryXml")]
pub fn chat_history_summary_xml(summary: String) -> String {
    lk_agents_abi::tts::markup::summary_content(&summary)
}

/// `build_legacy_openai_schema` (`utils.py:203`) — non-strict tool description.
#[napi(js_name = "buildLegacyOpenaiSchema")]
pub fn build_legacy_openai_schema(name: String, description: Option<String>, parameters: serde_json::Value, internally_tagged: Option<bool>) -> serde_json::Value {
    let internally_tagged = internally_tagged.unwrap_or(false);
    lk_agents_abi::llm::schema::build_legacy_openai_schema(&name, description.as_deref(), parameters, internally_tagged)
}

/// `build_strict_openai_schema` (`utils.py:229`) — strict-mode tool description. Defaulted fields
/// become nullable so the model can decline them; `inject_schema_defaults`(crate::llm::schema::inject_schema_defaults)
/// swaps the null back.
#[napi(js_name = "buildStrictOpenaiSchema")]
pub fn build_strict_openai_schema(env: Env, name: String, description: Option<String>, parameters: serde_json::Value) -> napi::Result<serde_json::Value> {
    crate::host_error::run(&env, || {
        lk_agents_abi::llm::schema::build_strict_openai_schema(&name, description.as_deref(), &parameters).map_err(crate::host_error::schema_error_to_node)
    })
}

/// `to_openai_response_format` (`utils.py:288`) — structured-output response format. No null
/// sentinel: nothing resolves it on the decode side, so defaulted fields stay required.
#[napi(js_name = "toOpenaiResponseFormat")]
pub fn to_openai_response_format(env: Env, name: String, schema: serde_json::Value) -> napi::Result<serde_json::Value> {
    crate::host_error::run(&env, || {
        lk_agents_abi::llm::schema::to_openai_response_format(&name, &schema).map_err(crate::host_error::schema_error_to_node)
    })
}

/// Rewrite a JSON schema into the strict dialect.
#[napi(js_name = "toStrictJsonSchema")]
pub fn to_strict_json_schema(env: Env, schema: serde_json::Value, null_sentinel_for_defaults: Option<bool>) -> napi::Result<serde_json::Value> {
    crate::host_error::run(&env, || {
        let null_sentinel_for_defaults = null_sentinel_for_defaults.unwrap_or(false);
        lk_agents_abi::llm::provider_format::to_strict_json_schema(&schema, null_sentinel_for_defaults).map_err(crate::generated::errors::strict_error_to_node)
    })
}

/// Resolve a local `$ref` against the schema root (`_strict.resolve_ref`, shared with
/// `llm/utils.py`). `#/`-rooted pointers only, with no `~0`/`~1` unescaping, as the reference.
#[napi(js_name = "resolveRef")]
pub fn resolve_ref(env: Env, root: serde_json::Map<String, serde_json::Value>, ref_: String) -> napi::Result<serde_json::Map<String, serde_json::Value>> {
    crate::host_error::run(&env, || {
        lk_agents_abi::llm::provider_format::resolve_ref(&root, &ref_).map_err(crate::generated::errors::strict_error_to_node)
    })
}

/// Whether recording is enabled for the job.
#[napi(js_name = "recordingEnabled")]
pub fn recording_enabled(options: &crate::generated::objects::RecordingOptions) -> bool {
    lk_agents_abi::telemetry::report::recording_enabled(&options.inner)
}

/// `validate_card_number`: the Luhn checksum over an all-digit string.
#[napi(js_name = "validateCardNumber")]
pub fn validate_card_number(card_number: String) -> bool {
    lk_agents_voice::workflows::validate_card_number(&card_number)
}

/// `normalize_email_address`: stripped, and refused unless it reads `local@label.….tld` (the
/// reference's `EMAIL_REGEX`: a local part starting alphanumeric over `[A-Za-z0-9._%+-]`, labels
/// alphanumeric at both ends with inner hyphens, and an alphabetic TLD of two or more).
#[napi(js_name = "normalizeEmailAddress")]
pub fn normalize_email_address(env: Env, email: String) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_voice::workflows::normalize_email_address(&email).map_err(|e| crate::host_error::JsError::new("livekit.agents.llm.tool_context.ToolError", e.to_string()))
    })
}

/// `normalize_phone_number`: whitespace, hyphens, parentheses and dots dropped, and refused unless
/// it reads `+?[1-9]` followed by six to fourteen digits (the reference's `PHONE_REGEX`).
#[napi(js_name = "normalizePhoneNumber")]
pub fn normalize_phone_number(env: Env, phone_number: String) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_voice::workflows::normalize_phone_number(&phone_number).map_err(|e| crate::host_error::JsError::new("livekit.agents.llm.tool_context.ToolError", e.to_string()))
    })
}

/// The answer to a call no runnable tool has (`generation._execute_tools_task`), in the reference's
/// words so the model corrects itself.
#[napi(js_name = "unknownToolMessage")]
pub fn unknown_tool_message(name: String, available: Vec<String>) -> String {
    lk_agents_voice::tool_executor::unknown_tool_message(&name, &available)
}

/// The join token a simulated job's agent participant uses.
#[napi(js_name = "agentJoinToken")]
pub fn agent_join_token(env: Env, api_key: String, api_secret: String, identity: String, room: String) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_worker::agent_join_token(&api_key, &api_secret, &identity, &room).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// Verify an SFU-issued join token and refresh its lifetime to `RELOAD_EXTENSION`.
/// Preserves all modelled claims and checks the issuer and expiration.
#[napi(js_name = "extendJobToken")]
pub fn extend_job_token(env: Env, token: String, api_key: String, api_secret: String) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_worker::extend_job_token(&token, &api_key, &api_secret).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// The `sub` of a token without verifying it. Only for `simulate_job`, where the caller supplied
/// the token and is asking what identity it names.
#[napi(js_name = "tokenIdentity")]
pub fn token_identity(env: Env, token: String) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_worker::token_identity(&token).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// How often the load function is re-invoked. Lives for the whole worker, connected or not.
#[napi(js_name = "updateLoadInterval")]
pub fn update_load_interval() -> f64 {
    lk_agents_worker::UPDATE_LOAD_INTERVAL.as_secs_f64()
}

/// The command line `lk agent` passes, as the dict the host dispatches on: `command` (`start`,
/// `console` or `download-files`), `entrypoint`, and for `start` `log_level`, `log_format`
/// (`json` or `colored`), `dev`, `cli_addr` (also read as the legacy `--reload-addr`),
/// `simulation` and `options` (the server's `ws_url`/`api_key`/`api_secret` given on the line);
/// for `console` `connect_addr` and `record`.
#[napi(js_name = "parseCli")]
pub fn parse_cli(env: Env, argv: Vec<String>) -> napi::Result<serde_json::Value> {
    crate::host_error::run(&env, || {
        lk_agents_worker::cli::parse_cli(argv).map_err(crate::generated::errors::cli_error_to_node)
    })
}

/// The entrypoint's module, as `(import string, directory to put on sys.path)`: `path`, or the
/// first default file in the working directory; a file inside packages imports through them.
#[napi(js_name = "entrypointModule")]
pub fn entrypoint_module(env: Env, path: Option<String>) -> napi::Result<(String, String)> {
    crate::host_error::run(&env, || {
        lk_agents_worker::cli::entrypoint_module(path).map_err(crate::generated::errors::cli_error_to_node)
    })
}

/// `is_evaluated_model` against the `"llm"` or `"stt"` table; any other kind is refused.
#[napi(js_name = "amdIsEvaluatedModel")]
pub fn amd_is_evaluated_model(env: Env, model: String, kind: String) -> napi::Result<bool> {
    crate::host_error::run(&env, || {
        lk_agents_inference::amd::is_evaluated_model_of_kind(&model, &kind).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// `AMDPredictionEvent.is_human`, over `AmdCategory::is_human`.
#[napi(js_name = "amdCategoryIsHuman")]
pub fn amd_category_is_human(env: Env, category: String) -> napi::Result<bool> {
    crate::host_error::run(&env, || {
        lk_agents_inference::amd::category_is_human(&category).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// `AMDPredictionEvent.is_machine`, over `AmdCategory::is_machine`.
#[napi(js_name = "amdCategoryIsMachine")]
pub fn amd_category_is_machine(env: Env, category: String) -> napi::Result<bool> {
    crate::host_error::run(&env, || {
        lk_agents_inference::amd::category_is_machine(&category).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`).
#[napi(js_name = "amdAutoSelectModels")]
pub fn amd_auto_select_models(livekit_url: String, api_key: String, api_secret: String) -> bool {
    lk_agents_inference::amd::auto_select_models(&livekit_url, &api_key, &api_secret)
}

/// `_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's, else default.
#[napi(js_name = "amdResolveMaxEndpointingDelay")]
pub fn amd_resolve_max_endpointing_delay(env: Env, provided: Option<f64>, activity: Option<f64>) -> napi::Result<f64> {
    crate::host_error::run(&env, || {
        let provided = provided.map(|v| Ok::<_, crate::host_error::JsError>(std::time::Duration::try_from_secs_f64(v).map_err(|e| crate::host_error::JsError::new("ValueError", format!("provided: {e}")))?)).transpose()?;
        let activity = activity.map(|v| Ok::<_, crate::host_error::JsError>(std::time::Duration::try_from_secs_f64(v).map_err(|e| crate::host_error::JsError::new("ValueError", format!("activity: {e}")))?)).transpose()?;
        Ok(lk_agents_voice::amd::classifier::resolve_max_endpointing_delay_for_host(provided, activity).as_secs_f64())
    })
}

/// The detector's constants, as one dict so a host cannot pick up half of them.
#[napi(js_name = "amdConstants")]
pub fn amd_constants() -> crate::generated::records::AmdConstants {
    crate::generated::records::AmdConstants::from(&lk_agents_voice::amd::classifier::amd_constants())
}

/// `AgentActivity._rt_turn_detection_enabled`: `server_turn_detection_enabled` from the host's
/// reads. `turn_detection` is the explicit mode string (agent over session), `None` when unsaid or
/// a detector object; `streaming_detector` says it was a streaming detector.
#[napi(js_name = "rtServerTurnDetection")]
pub fn rt_server_turn_detection(caps_turn_detection: bool, can_disable_turn_detection: bool, vad: bool, turn_detection: Option<String>, streaming_detector: bool, interruption_given: bool) -> bool {
    lk_agents_voice::agent_activity::options::rt_server_turn_detection_for_host(caps_turn_detection, can_disable_turn_detection, vad, turn_detection.as_deref(), streaming_detector, interruption_given)
}

/// `SpeechHandle.SPEECH_PRIORITY_{LOW,NORMAL,HIGH}`, as `(low, normal, high)`. Minted beside the
/// scheduler's heap key that reads them, so a host copy cannot drift and silently reorder.
/// No fork guard: `voice/speech_handle.py` reads these at import.
#[napi(js_name = "speechPriorities")]
pub fn speech_priorities() -> (i32, i32, i32) {
    lk_agents_voice::speech_handle::speech_priorities()
}

/// `INTERRUPTION_TIMEOUT`, in seconds.
#[napi(js_name = "interruptionTimeout")]
pub fn interruption_timeout() -> f64 {
    lk_agents_voice::speech_handle::interruption_timeout().as_secs_f64()
}

/// `DEFAULT_INPUT_DETAILS.modality`, as its wire string.
#[napi(js_name = "defaultInputModality")]
pub fn default_input_modality() -> String {
    lk_agents_voice::speech_handle::default_input_modality().to_string()
}

/// `dtmf_event_to_code` (`beta/workflows/utils.py:30`): digits are themselves, `*` is 10, `#` is 11
/// and `A`-`D` are 12-15.
#[napi(js_name = "dtmfCode")]
pub fn dtmf_code(env: Env, symbol: String) -> napi::Result<u8> {
    crate::host_error::run(&env, || {
        lk_agents_voice::workflows::dtmf::dtmf_code(&symbol).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// `format_dtmf` (`beta/workflows/utils.py:44`): the space-joined digits that fill
/// `prompts::dtmf`'s `{digits}` hole.
#[napi(js_name = "formatDtmf")]
pub fn format_dtmf(env: Env, symbols: Vec<String>) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_voice::workflows::dtmf::format_dtmf_symbols(symbols).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// The SIP participant's identity, the second session's `participant_identity`, and what
/// `move_participant` names.
#[napi(js_name = "WARM_TRANSFER_IDENTITY")]
pub fn WARM_TRANSFER_IDENTITY() -> String {
    lk_agents_voice::workflows::warm_transfer::HUMAN_AGENT_IDENTITY.to_string()
}

/// Twilio reports no-answer and failure only through status webhooks, which nothing here reads,
/// so an unanswered call is given up after this long unless `ringing_timeout` says otherwise.
#[napi(js_name = "TWILIO_RINGING_TIMEOUT")]
pub fn TWILIO_RINGING_TIMEOUT() -> f64 {
    lk_agents_voice::workflows::warm_transfer::TWILIO_RINGING_TIMEOUT
}

/// `EndCallTool`'s description, before its `extra_description`.
#[napi(js_name = "END_CALL_DESCRIPTION")]
pub fn END_CALL_DESCRIPTION() -> String {
    lk_agents_voice::workflows::prompts::END_CALL_DESCRIPTION.to_string()
}

/// The parameters the given model refuses. `has_tools` is `bool(tools)`; `reasoning_effort` is
/// the request's `reasoning_effort` and `reasoning` the effort of its Responses `reasoning`.
#[napi(js_name = "llmUnsupportedParams")]
pub fn llm_unsupported_params(model: String, has_tools: Option<bool>, reasoning_effort: Option<String>, reasoning: Option<String>) -> Vec<String> {
    let has_tools = has_tools.unwrap_or(false);
    lk_agents_inference::llm::models::unsupported_params(&model, has_tools, reasoning_effort.as_deref(), reasoning.as_deref()).into_iter().map(|v| v.to_string()).collect::<Vec<_>>()
}

/// The lowest reasoning effort the model supports, or `None` if it has no such control.
#[napi(js_name = "minReasoningEffort")]
pub fn min_reasoning_effort(model: String) -> Option<String> {
    lk_agents_inference::llm::models::min_reasoning_effort(&model).map(|v| v.to_string())
}

/// `_DEFAULT_AEC_WARMUP_DURATION`, as a `Duration`.
#[napi(js_name = "defaultAecWarmup")]
pub fn default_aec_warmup() -> f64 {
    lk_agents_voice::agent_session::DEFAULT_AEC_WARMUP.as_secs_f64()
}

/// One entry of a `tool_calls` attribute, as `json.dumps` writes it (`traces.py:551`, `llm.py:450`).
#[napi(js_name = "genAiToolCallJson")]
pub fn gen_ai_tool_call_json(name: String, arguments: String, call_id: String) -> String {
    lk_agents_abi::gen_ai::tool_call_json(&name, &arguments, &call_id)
}

/// The inference class every judgment must run under (`_judge_chat_kwargs`, `judge.py:19`).
#[napi(js_name = "judgeInferenceClass")]
pub fn judge_inference_class() -> String {
    lk_agents_abi::evals::JUDGE_INFERENCE_CLASS.to_string()
}

/// Is there an incomplete or unclosed tag in `text`? A `<` after the last `>` is tag-shaped when
/// followed by `/`, a letter, or nothing.
#[napi(js_name = "xmlHasUnclosedTags")]
pub fn xml_has_unclosed_tags(text: String) -> bool {
    lk_agents_abi::tokenize::xml::has_unclosed_tags(&text)
}

/// `replace_words`, case-insensitively, over one string. Returns the text and `processed_index`.
#[napi(js_name = "replaceWords")]
pub fn replace_words(text: String, replacements: std::collections::HashMap<String, String>, hold_last_word: Option<bool>) -> Option<(String, i64)> {
    let replacements = replacements.into_iter().map(|(k, v)| (k, v)).collect();
    let hold_last_word = hold_last_word.unwrap_or(false);
    lk_agents_abi::tokenize::replace_words(&text, replacements, hold_last_word).map(|v| { let (t0, t1,) = v; (t0, i64::try_from(t1).unwrap_or(i64::MAX)) })
}

/// `split_paragraphs`: the text between blank lines, each with its span in the original.
#[napi(js_name = "splitParagraphs")]
pub fn split_paragraphs(text: String) -> Vec<(String, i64, i64)> {
    lk_agents_abi::tokenize::split_paragraphs(&text).into_iter().map(|v| { let (t0, t1, t2,) = v; (t0, i64::try_from(t1).unwrap_or(i64::MAX), i64::try_from(t2).unwrap_or(i64::MAX)) }).collect::<Vec<_>>()
}

/// The turn detector a host constructs: `(model, cloud, warnings)`.
#[napi(js_name = "eotSelect")]
pub fn eot_select(env: Env, version: Option<String>, hosted: bool, dev: bool, url: String, key: String, secret: String, local_fallback: bool) -> napi::Result<(String, bool, Vec<String>)> {
    crate::host_error::run(&env, || {
        Ok({ let (t0, t1, t2,) = lk_agents_inference::eot::options::select_wire(version.as_deref(), hosted, dev, &url, &key, &secret, local_fallback).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?; (t0.to_string(), t1, t2) })
    })
}

/// `_observability_url(livekit_url)` (`job.py:58`). The env override wins unvalidated; otherwise
/// only LiveKit Cloud URLs get one, at `https://` + the ws url's hostname.
#[napi(js_name = "observabilityUrl")]
pub fn observability_url(livekit_url: String, env_override: Option<String>) -> Option<String> {
    lk_agents_abi::worker::job::observability_url(&livekit_url, env_override.as_deref())
}

/// The option keys a `SessionState` row's `options` carries, in the reference's order, and whether each is
/// a record the host expands first. Only the key set is pinned: values are Python `str()` output,
/// which stays with the host rather than a second `repr` implementation here.
#[napi(js_name = "sessionStateOptionFields")]
pub fn session_state_option_fields() -> Vec<(String, bool)> {
    lk_agents_voice::remote_session::STATE_OPTION_FIELDS.iter().map(|v| { let (t0, t1,) = v; (t0.to_string(), *t1) }).collect::<Vec<_>>()
}

/// `TOPIC_SESSION_MESSAGES` (`remote_session.py:57`): the byte-stream topic both halves register on.
#[napi(js_name = "sessionMessagesTopic")]
pub fn session_messages_topic() -> String {
    lk_agents_voice::remote_session::TOPIC_SESSION_MESSAGES.to_string()
}

/// `struct.pack(">I", len(data)) + data`, as one buffer so no sender can splice between header and body.
#[napi(js_name = "frame")]
pub fn frame(payload: napi::bindgen_prelude::Buffer) -> napi::bindgen_prelude::Buffer {
    napi::bindgen_prelude::Buffer::from(lk_agents_voice::remote_session::frame(&payload))
}

/// The rule `APIStatusError.__init__` delegates: is this status retryable? A non-transient 4xx is
/// forced non-retryable even when the caller passed `retryable=True` (#5831).
#[napi(js_name = "statusRetryable")]
pub fn status_retryable(status_code: i64, retryable: Option<bool>) -> bool {
    lk_agents_abi::errors::retryable_for_status(status_code, retryable)
}

/// What `create_api_error_from_http` shows the user: message and reason when the message adds
/// detail, else just the reason (never `"Not Found (404 Not Found)"`).
#[napi(js_name = "httpErrorDisplay")]
pub fn http_error_display(message: String, status: i64, reason: Option<String>) -> String {
    lk_agents_abi::errors::http_error_message(&message, status, reason.as_deref())
}

/// `MarkupInfo.nonverbals` for a `_provider_key()`: the field -> labels matrix, in insertion order.
#[napi(js_name = "supportedNonverbals")]
pub fn supported_nonverbals(provider: String) -> serde_json::Map<String, serde_json::Value> {
    lk_agents_abi::tts::markup::nonverbals_of(&provider)
}

/// The max text chunk length for a `_provider_key()`, or `None` if unlimited.
#[napi(js_name = "maxInputLen")]
pub fn max_input_len(provider: String) -> Option<i64> {
    lk_agents_abi::tts::markup::max_input_len_of(&provider).map(|v| i64::try_from(v).unwrap_or(i64::MAX))
}

/// `(max_token_len, min_token_len, xml_aware)` for `sentence_tokenizer(provider, expressive=)`; the
/// shim builds the blingfire tokenizer from it.
#[napi(js_name = "sentenceTokenizerConfig")]
pub fn sentence_tokenizer_config(provider: String, expressive: bool) -> (Option<i64>, Option<i64>, bool) {
    { let (t0, t1, t2,) = lk_agents_abi::tts::markup::sentence_tokenizer_of(&provider, expressive); (t0.map(|v| i64::try_from(v).unwrap_or(i64::MAX)), t1.map(|v| i64::try_from(v).unwrap_or(i64::MAX)), t2) }
}

/// Match a raw delivery label to a mood; an unrecognized label resolves to `fallback`.
#[napi(js_name = "matchMood", ts_return_type = "'excited' | 'happy' | 'playful' | 'curious' | 'surprised' | 'hopeful' | 'empathetic' | 'sad' | 'angry' | 'anxious' | 'calm' | null")]
pub fn match_mood(env: Env, label: String, #[napi(ts_arg_type = "'excited' | 'happy' | 'playful' | 'curious' | 'surprised' | 'hopeful' | 'empathetic' | 'sad' | 'angry' | 'anxious' | 'calm' | null")] fallback: Option<String>) -> napi::Result<Option<String>> {
    crate::host_error::run(&env, || {
        let fallback = fallback.map(|v| Ok::<_, crate::host_error::JsError>(v.parse().map_err(|e| crate::host_error::JsError::new("ValueError", format!("fallback: {e}")))?)).transpose()?;
        Ok(lk_agents_abi::tts::markup::match_mood(&label, fallback).map(|v| v.as_str().to_owned()))
    })
}

/// `ChatMessage.raw_text_content` over the message's text parts.
#[napi(js_name = "chatMessageRawTextContent")]
pub fn chat_message_raw_text_content(parts: Vec<String>) -> Option<String> {
    lk_agents_abi::chat_message_raw_text_content(parts)
}

/// `ChatMessage.text_content` over the message's text parts: an assistant's loses its markup.
#[napi(js_name = "chatMessageTextContent")]
pub fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String> {
    lk_agents_abi::chat_message_text_content(parts, assistant)
}

/// `ChatContext.copy` as an index plan over the host's own items.
#[napi(js_name = "chatCtxCopyPlan")]
pub fn chat_ctx_copy_plan(env: Env, facts: Vec<(String, String, Option<String>, i64, Option<String>, f64, Option<String>)>, exclude_function_call: Option<bool>, exclude_instructions: Option<bool>, exclude_empty_message: Option<bool>, exclude_handoff: Option<bool>, exclude_config_update: Option<bool>, tools: Option<Vec<String>>) -> napi::Result<Vec<i64>> {
    crate::host_error::run(&env, || {
        let facts = facts.into_iter().map(|v| Ok::<_, crate::host_error::JsError>({ let (t0, t1, t2, t3, t4, t5, t6,) = v; (t0, t1, t2, <usize>::try_from(t3).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("facts: {e}")))?, t4, t5, t6) })).collect::<Result<Vec<_>, _>>()?;
        let exclude_function_call = exclude_function_call.unwrap_or(false);
        let exclude_instructions = exclude_instructions.unwrap_or(false);
        let exclude_empty_message = exclude_empty_message.unwrap_or(false);
        let exclude_handoff = exclude_handoff.unwrap_or(false);
        let exclude_config_update = exclude_config_update.unwrap_or(false);
        Ok(lk_agents_abi::copy_plan_of(facts, exclude_function_call, exclude_instructions, exclude_empty_message, exclude_handoff, exclude_config_update, tools).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?.into_iter().map(|v| i64::try_from(v).unwrap_or(i64::MAX)).collect::<Vec<_>>())
    })
}

/// `ChatContext.truncate` as an index plan; `None` when under the limit.
#[napi(js_name = "chatCtxTruncatePlan")]
pub fn chat_ctx_truncate_plan(env: Env, facts: Vec<(String, String, Option<String>, i64, Option<String>, f64, Option<String>)>, max_items: i64) -> napi::Result<Option<Vec<i64>>> {
    crate::host_error::run(&env, || {
        let facts = facts.into_iter().map(|v| Ok::<_, crate::host_error::JsError>({ let (t0, t1, t2, t3, t4, t5, t6,) = v; (t0, t1, t2, <usize>::try_from(t3).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("facts: {e}")))?, t4, t5, t6) })).collect::<Result<Vec<_>, _>>()?;
        Ok(lk_agents_abi::truncate_plan_of(facts, max_items).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?.map(|v| v.into_iter().map(|v| i64::try_from(v).unwrap_or(i64::MAX)).collect::<Vec<_>>()))
    })
}

/// `ChatContext.merge` as `(index, j)` insertions.
#[napi(js_name = "chatCtxMergePlan")]
pub fn chat_ctx_merge_plan(env: Env, mine: Vec<(String, String, Option<String>, i64, Option<String>, f64, Option<String>)>, theirs: Vec<(String, String, Option<String>, i64, Option<String>, f64, Option<String>)>, exclude_function_call: Option<bool>, exclude_instructions: Option<bool>, exclude_config_update: Option<bool>) -> napi::Result<Vec<(i64, i64)>> {
    crate::host_error::run(&env, || {
        let mine = mine.into_iter().map(|v| Ok::<_, crate::host_error::JsError>({ let (t0, t1, t2, t3, t4, t5, t6,) = v; (t0, t1, t2, <usize>::try_from(t3).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("mine: {e}")))?, t4, t5, t6) })).collect::<Result<Vec<_>, _>>()?;
        let theirs = theirs.into_iter().map(|v| Ok::<_, crate::host_error::JsError>({ let (t0, t1, t2, t3, t4, t5, t6,) = v; (t0, t1, t2, <usize>::try_from(t3).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("theirs: {e}")))?, t4, t5, t6) })).collect::<Result<Vec<_>, _>>()?;
        let exclude_function_call = exclude_function_call.unwrap_or(false);
        let exclude_instructions = exclude_instructions.unwrap_or(false);
        let exclude_config_update = exclude_config_update.unwrap_or(false);
        Ok(lk_agents_abi::merge_plan_of(mine, theirs, exclude_function_call, exclude_instructions, exclude_config_update).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?.into_iter().map(|v| { let (t0, t1,) = v; (i64::try_from(t0).unwrap_or(i64::MAX), i64::try_from(t1).unwrap_or(i64::MAX)) }).collect::<Vec<_>>())
    })
}

/// `_record_turn_metrics` (`otel_metrics.py:124`), as the list of histogram observations.
#[napi(js_name = "otelTurnRecords")]
pub fn otel_turn_records(env: Env, report: Option<crate::generated::records::MetricsReport>) -> napi::Result<Vec<(String, String, serde_json::Value, serde_json::Map<String, serde_json::Value>)>> {
    crate::host_error::run(&env, || {
        let report = report.map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).transpose()?;
        Ok(lk_agents_abi::metrics::turn_records_wire(report).into_iter().map(|v| { let (t0, t1, t2, t3,) = v; (t0.to_string(), t1.to_string(), t2, t3) }).collect::<Vec<_>>())
    })
}

/// `APIConnectOptions._interval_for_retry` (`types.py:138`) for the host's retry loops; only
/// `retry_interval` affects the answer.
#[napi(js_name = "apiRetryInterval")]
pub fn api_retry_interval(retry_interval: f64, num_retries: u32) -> f64 {
    lk_agents_abi::api_retry_interval(retry_interval, num_retries)
}

/// The id of the rendered prompt in every request. Must not change (persisted contexts carry it).
#[napi(js_name = "instructionsMessageId")]
pub fn instructions_message_id() -> String {
    lk_agents_voice::generation::INSTRUCTIONS_MESSAGE_ID.to_string()
}

/// `remove_instructions(chat_ctx)` as a plan: the positions of every
/// `INSTRUCTIONS_MESSAGE_ID` item, ascending, given the items' ids.
#[napi(js_name = "removeInstructions")]
pub fn remove_instructions(item_ids: Vec<String>) -> Vec<i64> {
    lk_agents_voice::generation::instructions_positions(item_ids).into_iter().map(|v| i64::try_from(v).unwrap_or(i64::MAX)).collect::<Vec<_>>()
}

/// Build the key a call is grouped under.
#[napi(js_name = "duplicateKey")]
pub fn duplicate_key(env: Env, name: String, args: Option<serde_json::Value>, #[napi(ts_arg_type = "'name' | 'name_and_args'")] scope: String) -> napi::Result<(String, Option<String>)> {
    crate::host_error::run(&env, || {
        let scope = scope.parse().map_err(|e| crate::host_error::JsError::new("ValueError", format!("scope: {e}")))?;
        Ok(lk_agents_abi::llm::provider_format::tools::duplicate_key(&name, args.as_ref(), scope))
    })
}

/// `_check_duplicate`'s decision (`tool_executor.py:649`) for the host, without its effects.
/// `in_flight` is `[(call_id, allow_cancellation)]` for calls with the same duplicate key. Returns
/// `("proceed", None)`, `("cancel_then_proceed", [call_id, ..])`, `("reject", None)`,
/// `("confirm", None)` or `("not_cancellable", name)`.
#[napi(js_name = "decideDuplicate")]
pub fn decide_duplicate(env: Env, name: String, #[napi(ts_arg_type = "'allow' | 'reject' | 'replace' | 'confirm'")] mode: String, confirm_duplicate: Option<bool>, in_flight: Vec<(String, bool)>) -> napi::Result<(String, Option<serde_json::Value>)> {
    crate::host_error::run(&env, || {
        let mode = mode.parse().map_err(|e| crate::host_error::JsError::new("ValueError", format!("mode: {e}")))?;
        Ok({ let (t0, t1,) = lk_agents_abi::llm::provider_format::tools::decide_duplicate_wire(&name, mode, confirm_duplicate, in_flight); (t0.to_string(), t1) })
    })
}

/// The description the model reads for that parameter. A prompt, so rewording it changes
/// behaviour: it must say the user asked for the repeat.
#[napi(js_name = "confirmDuplicateDescription")]
pub fn confirm_duplicate_description() -> String {
    lk_agents_abi::llm::provider_format::tools::CONFIRM_DUPLICATE_DESCRIPTION.to_string()
}

/// `_inject_confirm_duplicate` (`llm/tool_context.py:499`): add the confirmation flag to a **raw**
/// JSON schema, strict-mode conformant.
#[napi(js_name = "injectConfirmDuplicate")]
pub fn inject_confirm_duplicate(parameters: serde_json::Value) -> serde_json::Value {
    lk_agents_abi::llm::provider_format::tools::inject_confirm_duplicate(&parameters)
}

/// `to_fnc_ctx(tools)` for a dialect; each tool is the host's `[kind, schema, name]` (`name` only for
/// `raw`). `tool_behavior` and `use_parameters_json_schema` are Gemini-only (Gemini Live rejects
/// `parameters_json_schema`, googleapis/python-genai#1147).
#[napi(js_name = "toFncCtx")]
pub fn to_fnc_ctx(env: Env, format: String, tools: Vec<serde_json::Value>, tool_behavior: Option<String>, use_parameters_json_schema: Option<bool>) -> napi::Result<Vec<serde_json::Value>> {
    crate::host_error::run(&env, || {
        let use_parameters_json_schema = use_parameters_json_schema.unwrap_or(false);
        lk_agents_abi::llm::provider_format::host::to_fnc_ctx(&format, tools, tool_behavior.as_deref(), use_parameters_json_schema).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// `strip_thinking_tokens` over the filter's fields: `(visible, buffer, end_marker)`.
/// `content=None` (no field) and `""` (empty field) are distinct, and only `""` can answer `""`:
/// collapsing them would make a content-filtered chunk look like pure reasoning.
#[napi(js_name = "stripThinking")]
pub fn strip_thinking(content: Option<String>, start_tag: String, end_tag: String, buffer: String, end_marker: Option<String>, is_final: bool) -> (Option<String>, String, Option<String>) {
    lk_agents_abi::tokenize::thinking::strip_thinking(content.as_deref(), &start_tag, &end_tag, &buffer, end_marker.as_deref(), is_final)
}

/// An empty list means no text; a present empty part is an empty delta.
#[napi(js_name = "flattenDeltaText")]
pub fn flatten_delta_text(parts: Vec<String>) -> Option<String> {
    lk_agents_abi::tokenize::thinking::flatten_text_parts(&parts)
}

/// Host-independent search validation and model-facing result policy.
#[napi(js_name = "toolSearchValidate")]
pub fn tool_search_validate(env: Env, query: String) -> napi::Result<()> {
    crate::host_error::run(&env, || {
        lk_agents_abi::tokenize::search::validate_query(&query).map_err(|e| crate::host_error::JsError::new("livekit.agents.llm.tool_context.ToolError", e.to_string()))?;
        Ok(())
    })
}

#[napi(js_name = "toolSearchLoaded")]
pub fn tool_search_loaded(env: Env, query: String, count: i64) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        let count = <usize>::try_from(count).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("count: {e}")))?;
        Ok(lk_agents_abi::tokenize::search::loaded_result(&query, count).map_err(|e| crate::host_error::JsError::new("livekit.agents.llm.tool_context.ToolError", e.to_string()))?.to_string())
    })
}

/// `ToolProxyToolset`'s `call_tool` arguments: the tool's name and its parameters (an object, or
/// the JSON string of one), refused as the model's error when missing, mistyped, or naming no tool
/// in `known`.
#[napi(js_name = "toolProxyArguments")]
pub fn tool_proxy_arguments(env: Env, arguments: serde_json::Value, known: Vec<String>) -> napi::Result<(String, serde_json::Value)> {
    crate::host_error::run(&env, || {
        lk_agents_abi::tokenize::search::proxy_arguments(arguments, known).map_err(|e| crate::host_error::JsError::new("livekit.agents.llm.tool_context.ToolError", e.to_string()))
    })
}

/// Each `ALL` array's names, indexed by wire value, for the host adapter's reads of SDK ints:
/// `(connection states, disconnect reasons, participant kinds, track sources, audio features)`.
#[napi(js_name = "rtcWireNames")]
pub fn rtc_wire_names() -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    { let (t0, t1, t2, t3, t4,) = lk_agents_abi::voice::rtc::rtc_wire_names(); (t0.into_iter().map(|v| v.to_string()).collect::<Vec<_>>(), t1.into_iter().map(|v| v.to_string()).collect::<Vec<_>>(), t2.into_iter().map(|v| v.to_string()).collect::<Vec<_>>(), t3.into_iter().map(|v| v.to_string()).collect::<Vec<_>>(), t4.into_iter().map(|v| v.to_string()).collect::<Vec<_>>()) }
}

/// How many frames a `video_stream` queue holds before it evicts the oldest. Tiny on purpose: each
/// held frame pins a decoded host frame, and the newest frame is the one worth keeping. A host
/// video stream is opened with the same capacity, so both sides drop the oldest alike.
#[napi(js_name = "videoQueueFrames")]
pub fn video_queue_frames() -> i64 {
    i64::try_from(lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES).unwrap_or(i64::MAX)
}

#[napi(js_name = "validateSynthesis")]
pub fn validate_synthesis(env: Env, text: String, duration: f64, segments: Option<(i64, i64)>) -> napi::Result<Option<String>> {
    crate::host_error::run(&env, || {
        let segments = segments.map(|v| Ok::<_, crate::host_error::JsError>({ let (t0, t1,) = v; (<usize>::try_from(t0).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("segments: {e}")))?, <usize>::try_from(t1).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("segments: {e}")))?) })).transpose()?;
        Ok(lk_agents_abi::types::retry::validate_synthesis(&text, duration, segments))
    })
}

/// `_merge_capabilities` (`:70-86`). `audio_output` and `turn_detection` must agree (they shape
/// the pipeline at activity start); the rest are ANDed. Raises `ValueError` naming the capability
/// and both values, as the reference does.
#[napi(js_name = "mergeRealtimeCapabilities")]
pub fn merge_realtime_capabilities(env: Env, caps: Vec<crate::generated::records::RealtimeCapabilities>) -> napi::Result<crate::generated::records::RealtimeCapabilities> {
    crate::host_error::run(&env, || {
        let caps = caps.into_iter().map(|v| Ok::<_, crate::host_error::JsError>(v.try_into()?)).collect::<Result<Vec<_>, _>>()?;
        Ok(crate::generated::records::RealtimeCapabilities::from(&lk_agents_inference::fallback::merge_capabilities_of(caps).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?))
    })
}

/// `"provider/model:language"` -> `("provider/model", Some("language"))`, split at the last colon.
#[napi(js_name = "sttParseModelString")]
pub fn stt_parse_model_string(model: String) -> (String, Option<String>) {
    { let (t0, t1,) = lk_agents_inference::stt::options::parse_model_string(&model); (t0.to_string(), t1.map(|v| v.to_string())) }
}

/// True if any known provider diarization flag is truthy.
#[napi(js_name = "sttDiarizationEnabled")]
pub fn stt_diarization_enabled(extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>) -> bool {
    lk_agents_inference::stt::options::diarization_enabled(extra_kwargs.as_ref())
}

/// True if the model natively carries `agent_context`.
#[napi(js_name = "sttSupportsAgentContextCarryover")]
pub fn stt_supports_agent_context_carryover(model: Option<String>) -> bool {
    lk_agents_inference::stt::options::supports_agent_context_carryover(model.as_deref())
}

/// Word-level alignment for the model; `false` for `"auto"`, since claiming alignment is the costlier error.
#[napi(js_name = "sttAlignedTranscriptForModel")]
pub fn stt_aligned_transcript_for_model(model: Option<String>) -> bool {
    lk_agents_inference::stt::options::aligned_transcript_for_model(model.as_deref())
}

/// `_normalize_fallback`: a bare string becomes `FallbackModel(model=<name>)`, with the
/// `:language` suffix parsed off and discarded.
#[napi(js_name = "sttNormalizeFallbackStr")]
pub fn stt_normalize_fallback_str(model: String) -> crate::generated::records::SttFallbackModel {
    crate::generated::records::SttFallbackModel::from(&lk_agents_inference::stt::options::normalize_fallback_str(&model))
}

#[napi(js_name = "sttResolveVadForModel", ts_return_type = "'keep' | 'drop_and_warn' | 'create_default' | 'none'")]
pub fn stt_resolve_vad_for_model(model: Option<String>, has_vad: Option<bool>) -> String {
    let has_vad = has_vad.unwrap_or(false);
    lk_agents_inference::stt::options::resolve_vad_for_model(model.as_deref(), has_vad).as_str().to_owned()
}

/// `"provider/model:voice"` -> `("provider/model", Some("voice"))`.
#[napi(js_name = "ttsParseModelString")]
pub fn tts_parse_model_string(model: String) -> (String, Option<String>) {
    { let (t0, t1,) = lk_agents_inference::tts::wire::parse_model_string(&model); (t0.to_string(), t1.map(|v| v.to_string())) }
}

/// `_normalize_fallback`'s string arm.
#[napi(js_name = "ttsNormalizeFallbackStr")]
pub fn tts_normalize_fallback_str(model: String) -> crate::generated::records::TtsFallbackModel {
    crate::generated::records::TtsFallbackModel::from(&lk_agents_inference::tts::wire::normalize_fallback_str(&model))
}

/// `_has_aligned_transcript`: whether this model and these options produce timings (recomputed
/// on `update_options`).
#[napi(js_name = "ttsHasAlignedTranscript")]
pub fn tts_has_aligned_transcript(model: String, extra_kwargs: serde_json::Map<String, serde_json::Value>) -> bool {
    lk_agents_inference::tts::wire::has_aligned_transcript(&model, &extra_kwargs)
}

/// `TTS.Markup._provider_key()`: the markup dialect; "" means the model has no markup.
#[napi(js_name = "ttsMarkupProviderKey")]
pub fn tts_markup_provider_key(model: String) -> String {
    lk_agents_inference::tts::wire::markup_provider_key(&model).to_string()
}

/// The 2nd-largest per-frame probability (the reference's body, not its docstring); 0.0 for fewer than two frames.
#[napi(js_name = "interruptionEstimateProbability")]
pub fn interruption_estimate_probability(probabilities: Vec<f64>) -> f64 {
    let probabilities = probabilities.into_iter().map(|v| v as f32).collect::<Vec<_>>();
    lk_agents_inference::interruption::estimate_probability(&probabilities)
}

/// `math.ceil(min_interruption_duration * _FRAMES_PER_SECOND)`, saturating (never a silent 0).
#[napi(js_name = "interruptionMinFrames")]
pub fn interruption_min_frames(min_interruption_duration: f64) -> u32 {
    lk_agents_inference::interruption::min_frames_for(min_interruption_duration)
}

/// Normalise a language code or name to BCP-47.
#[napi(js_name = "normalizeLanguage")]
pub fn normalize_language(code: String) -> String {
    lk_agents_abi::tokenize::language::normalize(&code)
}

/// Base language, ISO 639-1 where one exists: `"en"` from `"en-US"`, `"zh"` from `"cmn-Hans-CN"`.
#[napi(js_name = "baseLanguage")]
pub fn base_language(code: String) -> String {
    lk_agents_abi::tokenize::language::base_language(&code).to_string()
}

/// Region subtag: the first two-character subtag after the primary (`None` for `"es-419"`).
#[napi(js_name = "languageRegion")]
pub fn language_region(code: String) -> Option<String> {
    lk_agents_abi::tokenize::language::region(&code).map(|v| v.to_string())
}

/// The 639-1 base plus the region: `"zh-CN"` from `"cmn-Hans-CN"` (`turn_detector/base.py:241`).
#[napi(js_name = "languageIso")]
pub fn language_iso(code: String) -> String {
    lk_agents_abi::tokenize::language::iso(&code)
}

/// Lowercase English (NLTK punkt) name of the base language, or `None` if unknown.
#[napi(js_name = "languageName")]
pub fn language_name(code: String) -> Option<String> {
    lk_agents_abi::tokenize::language::language_name(&code).map(|v| v.to_string())
}

/// `utils.images.encode(frame, options)` over tight RGBA, the options as the reference's
/// `Literal` strings. Unknown values raise rather than default.
#[napi(js_name = "encodeImage")]
pub fn encode_image(env: Env, data: napi::bindgen_prelude::Buffer, width: u32, height: u32, format: Option<String>, quality: Option<u8>, resize_width: Option<u32>, resize_height: Option<u32>, resize_strategy: Option<String>) -> napi::Result<napi::bindgen_prelude::Buffer> {
    crate::host_error::run(&env, || {
        let format = format.as_deref().unwrap_or("JPEG");
        Ok(napi::bindgen_prelude::Buffer::from(lk_agents_media::images::encode_image(&data, width, height, &format, quality, resize_width, resize_height, resize_strategy.as_deref()).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?))
    })
}

/// `split_words` for the host: `(word, start_char, end_char)`, with character offsets
/// (`replace_words` splices with them).
#[napi(js_name = "splitWords")]
pub fn split_words(text: String, ignore_punctuation: Option<bool>, split_character: Option<bool>, retain_format: Option<bool>) -> Vec<(String, i64, i64)> {
    let ignore_punctuation = ignore_punctuation.unwrap_or(true);
    let split_character = split_character.unwrap_or(false);
    let retain_format = retain_format.unwrap_or(false);
    lk_agents_abi::tokenize::split_words_of(&text, ignore_punctuation, split_character, retain_format).into_iter().map(|v| { let (t0, t1, t2,) = v; (t0, i64::try_from(t1).unwrap_or(i64::MAX), i64::try_from(t2).unwrap_or(i64::MAX)) }).collect::<Vec<_>>()
}

/// Sentences of more than `min_sentence_len` characters (shorter ones join the next), each with
/// its span in the text.
#[napi(js_name = "basicSplitSentences")]
pub fn basic_split_sentences(env: Env, text: String, min_sentence_len: Option<i64>, retain_format: Option<bool>) -> napi::Result<Vec<(String, i64, i64)>> {
    crate::host_error::run(&env, || {
        let min_sentence_len = match min_sentence_len { Some(v) => <usize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("min_sentence_len: {e}")))?, None => 20 };
        let retain_format = retain_format.unwrap_or(false);
        Ok(lk_agents_abi::tokenize::basic_split_sentences(&text, min_sentence_len, retain_format).into_iter().map(|v| { let (t0, t1, t2,) = v; (t0, i64::try_from(t1).unwrap_or(i64::MAX), i64::try_from(t2).unwrap_or(i64::MAX)) }).collect::<Vec<_>>())
    })
}

/// Five samples at the load-update interval cover 2.5 seconds.
#[napi(js_name = "defaultLoadWindow")]
pub fn default_load_window() -> i64 {
    i64::try_from(lk_agents_worker::DefaultLoadCalc::WINDOW).unwrap_or(i64::MAX)
}

/// `parse_function_arguments`: the model's JSON arguments as an object, repaired when malformed
/// (the repair is logged, never raised).
#[napi(js_name = "parseFunctionArguments")]
pub fn parse_function_arguments(env: Env, raw: String) -> napi::Result<serde_json::Map<String, serde_json::Value>> {
    crate::host_error::run(&env, || {
        lk_agents_abi::llm::tool_arguments::parse_logged(&raw).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
    })
}

/// `tokenize/blingfire.py::_split_sentences`, exactly.
#[napi(js_name = "splitSentences")]
pub fn split_sentences(env: Env, text: String, min_sentence_len: Option<i64>, retain_format: Option<bool>) -> napi::Result<Vec<(String, i64, i64)>> {
    crate::host_error::run(&env, || {
        let min_sentence_len = match min_sentence_len { Some(v) => <usize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("min_sentence_len: {e}")))?, None => 20 };
        let retain_format = retain_format.unwrap_or(false);
        Ok(lk_agents_media::blingfire::split_sentences(&text, min_sentence_len, retain_format).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?.into_iter().map(|v| { let (t0, t1, t2,) = v; (t0, i64::try_from(t1).unwrap_or(i64::MAX), i64::try_from(t2).unwrap_or(i64::MAX)) }).collect::<Vec<_>>())
    })
}

/// `filter_emoji`. Stateless: nothing here spans a chunk boundary.
#[napi(js_name = "filterEmoji")]
pub fn filter_emoji(chunk: String) -> String {
    lk_agents_voice::transcription::filters::filter_emoji(&chunk)
}

/// `hyphenate_word(word)`: the pieces, broken at the possible hyphenation points.
#[napi(js_name = "hyphenateWord")]
pub fn hyphenate_word(word: String) -> Vec<String> {
    lk_agents_voice::transcription::synchronizer::hyphenate_word(&word)
}

/// `inject_schema_defaults` for the host, where `root` defaults to `schema`.
#[napi(js_name = "injectSchemaDefaults")]
pub fn inject_schema_defaults(env: Env, value: serde_json::Value, schema: serde_json::Value, root: Option<serde_json::Value>) -> napi::Result<serde_json::Value> {
    crate::host_error::run(&env, || {
        lk_agents_abi::llm::schema::inject_schema_defaults_of(&value, &schema, root.as_ref()).map_err(crate::host_error::schema_error_to_node)
    })
}

/// `serialize_image`'s `str` arm, for a host that owns `ImageContent`:
/// `(data_bytes, mime_type, external_url, header_mime_if_overridden)`.
#[napi(js_name = "serializeImageUrl")]
pub fn serialize_image_url(env: Env, url: String, mime_type: Option<String>) -> napi::Result<(Option<napi::bindgen_prelude::Buffer>, Option<String>, Option<String>, Option<String>)> {
    crate::host_error::run(&env, || {
        Ok({ let (t0, t1, t2, t3,) = lk_agents_media::images::serialize_image_url(&url, mime_type.as_deref()).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?; (t0.map(|v| napi::bindgen_prelude::Buffer::from(v)), t1, t2, t3) })
    })
}

/// `serialize_image`'s `rtc.VideoFrame` arm over tight RGBA: `(data_bytes, mime_type)`.
#[napi(js_name = "serializeImageFrame")]
pub fn serialize_image_frame(env: Env, data: napi::bindgen_prelude::Buffer, width: u32, height: u32, inference_width: Option<u32>, inference_height: Option<u32>) -> napi::Result<(napi::bindgen_prelude::Buffer, String)> {
    crate::host_error::run(&env, || {
        Ok({ let (t0, t1,) = lk_agents_media::images::serialize_image_frame(&data, width, height, inference_width, inference_height).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?; (napi::bindgen_prelude::Buffer::from(t0), t1) })
    })
}

/// `_make_function_tool`'s `raw_schema`: `description` always present (null when absent), `meta`
/// only when truthy.
#[napi(js_name = "mcpRawSchema")]
pub fn mcp_raw_schema(name: String, description: Option<String>, input_schema: serde_json::Map<String, serde_json::Value>, meta: Option<serde_json::Map<String, serde_json::Value>>) -> serde_json::Value {
    lk_agents_inference::mcp::raw_schema(&name, description.as_deref(), &input_schema, meta.as_ref())
}

#[napi(js_name = "mcpPlanSetup", ts_return_type = "'skip' | 'connect_then_list' | 'invalidate_then_list' | 'list_only'")]
pub fn mcp_plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> String {
    lk_agents_inference::mcp::plan_setup(reload, toolset_initialized, server_connected).as_str().to_owned()
}

/// `transport_type` when given, else streamable HTTP only for a `urlparse` path ending in `/mcp`
/// (case- and trailing-slash-insensitive); everything else is SSE.
#[napi(js_name = "mcpHttpTransport", ts_return_type = "'sse' | 'streamable_http'")]
pub fn mcp_http_transport(env: Env, url: String, transport_type: Option<String>) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        Ok(lk_agents_inference::mcp::http_transport(&url, transport_type.as_deref()).map_err(crate::generated::errors::mcp_error_to_node)?.as_str().to_owned())
    })
}

/// `_resolve` then the default resolver.
#[napi(js_name = "mcpRenderResult")]
pub fn mcp_render_result(env: Env, tool_name: String, result: serde_json::Value) -> napi::Result<String> {
    crate::host_error::run(&env, || {
        lk_agents_inference::mcp::render::render_result(&tool_name, &result).map_err(crate::generated::errors::mcp_error_to_node)
    })
}

/// `_resolve`'s `isError` arm, checked before the content count.
#[napi(js_name = "mcpCheckResult")]
pub fn mcp_check_result(env: Env, result: serde_json::Value) -> napi::Result<()> {
    crate::host_error::run(&env, || {
        lk_agents_inference::mcp::render::check_result(&result).map_err(crate::generated::errors::mcp_error_to_node)?;
        Ok(())
    })
}
