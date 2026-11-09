//! The ABI-declared OpenAI-protocol realtime facade. Provider defaults and option resolution stay here.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::realtime::{
    OpenAIRealtimeModel, RealtimeConfigurationError, RealtimeProvider,
};
use lk_agents_abi::llm::realtime_options::{
    AudioTranscription, InferenceGateway, NoiseReduction, Reasoning, Tracing, Truncation,
    TurnDetection as TurnDetectionOption, Voice,
};
use lk_agents_abi::{ApiConnectOptions, Maybe, Modalities, ToolChoice};
use lk_agents_net::task::TokioSpawner;
use serde_json::{Value, json};

use super::session::{Auth, Connect, OpenAIRealtime};
use super::wire::Azure;
use crate::realtime::dialect::{DEFAULT_MAX_OUTPUT_TOKENS, Dialect, OPENAI, TurnDetection};
use crate::realtime::machine::{OptionsPatch, SessionOptions};
use lk_agents_net::env::ProcessEnv;

/// A session update: an omitted option keeps its setting, an explicit `None` clears it.
#[derive(Clone, Default)]
struct RealtimeOptions {
    voice: Maybe<Voice>,
    tool_choice: Maybe<Option<ToolChoice>>,
    input_audio_transcription: Maybe<Option<AudioTranscription>>,
    input_audio_noise_reduction: Maybe<Option<NoiseReduction>>,
    turn_detection: Maybe<Option<TurnDetectionOption>>,
    speed: Maybe<f64>,
    tracing: Maybe<Option<Tracing>>,
    truncation: Maybe<Option<Truncation>>,
    reasoning: Maybe<Option<Reasoning>>,
    max_response_output_tokens: Maybe<Option<u32>>,
}

/// The constructor's arguments.
#[derive(Clone, Default)]
struct Config {
    model: Option<String>,
    voice: Option<Voice>,
    modalities: Option<Vec<Modalities>>,
    options: RealtimeOptions,
    base_url: Option<String>,
    api_key: Option<String>,
    azure_deployment: Option<String>,
    azure_endpoint: Option<String>,
    entra_token: Option<String>,
    max_session_duration: Maybe<Option<Duration>>,
    conn_options: ApiConnectOptions,
    dialect: Option<RealtimeProvider>,
    label: Option<String>,
    gateway: Option<InferenceGateway>,
}

/// An option record as the session field it sets.
fn wire(option: impl serde::Serialize) -> Value {
    serde_json::to_value(option).expect("a realtime option is plain JSON")
}

fn patch(options: RealtimeOptions) -> OptionsPatch {
    OptionsPatch {
        voice: options.voice.map(wire),
        tool_choice: options.tool_choice,
        turn_detection: options
            .turn_detection
            .map(|value| value.map(|td| TurnDetection::Raw(wire(td)))),
        max_output_tokens: options
            .max_response_output_tokens
            .map(|tokens| tokens.map_or(json!(DEFAULT_MAX_OUTPUT_TOKENS), |n| json!(n))),
        transcription: options.input_audio_transcription.map(|v| v.map(wire)),
        noise_reduction: options.input_audio_noise_reduction.map(|v| v.map(wire)),
        speed: options.speed,
        tracing: options.tracing.map(|v| v.map(wire)),
        truncation: options.truncation.map(|v| v.map(wire)),
        reasoning: options.reasoning.map(|v| v.map(wire)),
    }
}

/// The caller's key and base URL, with the environment's defaults: Azure OpenAI's when Azure is
/// selected on the OpenAI dialect, else the dialect's. Errors are the messages a caller sees.
fn endpoint(
    config: &Config,
    named: Option<&Dialect>,
    env: impl Fn(&str) -> Option<String>,
) -> Result<(Option<String>, String), String> {
    let nonempty = |value: &Option<String>| value.clone().filter(|value| !value.is_empty());
    let mut api_key = nonempty(&config.api_key);
    let mut base_url = nonempty(&config.base_url);
    let is_azure = config.entra_token.is_some() || config.azure_deployment.is_some();
    if is_azure && named.is_none() {
        api_key = api_key.or_else(|| env("AZURE_OPENAI_API_KEY"));
        if api_key.is_none() && config.entra_token.is_none() {
            return Err(
                "Missing credentials. Please pass one of `api_key`, `entra_token`, or the \
                 `AZURE_OPENAI_API_KEY` environment variable."
                    .into(),
            );
        }
        let azure_endpoint = nonempty(&config.azure_endpoint);
        if base_url.is_some() && azure_endpoint.is_some() {
            return Err("base_url and azure_endpoint are mutually exclusive".into());
        }
        if base_url.is_none() {
            let endpoint = azure_endpoint.or_else(|| env("AZURE_OPENAI_ENDPOINT")).ok_or(
                "Missing Azure endpoint. Please pass the `azure_endpoint` parameter or set the \
                 `AZURE_OPENAI_ENDPOINT` environment variable.",
            )?;
            base_url = Some(format!("{}/openai", endpoint.trim_end_matches('/')));
        }
    }
    if !is_azure {
        let key_env = named.unwrap_or(&OPENAI).api_key_env;
        api_key = api_key.or_else(|| env(key_env));
        if api_key.is_none() {
            return Err(format!(
                "The api_key client option must be set either by passing api_key to the client \
                 or by setting the {key_env} environment variable"
            ));
        }
    }
    let base_url = base_url.unwrap_or_else(|| match named {
        Some(dialect) => dialect.default_base_url.to_owned(),
        None => env("OPENAI_BASE_URL").unwrap_or_else(|| OPENAI.default_base_url.to_owned()),
    });
    Ok((api_key, base_url))
}

fn build(env: &ProcessEnv, config: Config) -> Result<OpenAIRealtime, RealtimeConfigurationError> {
    let named = config.dialect.map(Dialect::of);
    // the gateway's caller resolved its own credentials and URL
    let (api_key, base_url) = match &named {
        Some(dialect) if dialect.provider == RealtimeProvider::Inference => {
            (config.api_key.clone(), config.base_url.clone().unwrap_or_default())
        }
        _ => endpoint(&config, named.as_ref(), |name| std::env::var(name).ok())
            .map_err(RealtimeConfigurationError::Invalid)?,
    };
    let is_azure = config.azure_deployment.is_some() || config.entra_token.is_some();
    let azure_openai = is_azure && named.is_none();
    let dialect = named.unwrap_or_else(|| Dialect::for_base_url(&base_url));
    let is_inference = dialect.provider == RealtimeProvider::Inference;
    let mut opts = SessionOptions::from_dialect(&dialect, false);
    if let Some(model) = config.model {
        opts.model = model;
    }
    let model_provider = opts
        .model
        .split_once('/')
        .and_then(|(provider, _)| RealtimeProvider::from_wire(provider))
        .unwrap_or(dialect.provider);
    if is_inference && model_provider == RealtimeProvider::Xai {
        let xai = Dialect::of(RealtimeProvider::Xai);
        opts.transcription = xai.default_transcription_model.map(|m| json!({"model": m}));
        opts.voice = json!(xai.default_voice);
        opts.turn_detection = xai.default_turn_detection;
    }
    if let Some(voice) = config.voice {
        opts.voice = wire(voice);
    }
    if dialect.forced_modalities.is_none() {
        opts.audio_output = config
            .modalities
            .is_none_or(|modalities| modalities.contains(&Modalities::AudioAndText));
    }
    if azure_openai {
        opts.transcription = Some(json!({"model": "whisper-1"}));
        opts.turn_detection = Some(TurnDetection::Raw(json!({
            "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
            "silence_duration_ms": 200, "create_response": true,
        })));
    }
    let turn_detection_given = config.options.turn_detection.is_given();
    let options = patch(config.options);
    super::session::warn_on_half_disabled_turn_taking(&options.turn_detection);
    opts.apply(&options);
    if let Maybe::Given(duration) = config.max_session_duration {
        opts.max_session_duration = duration;
    }
    let auth = if is_inference {
        let gateway = config.gateway.unwrap_or_default();
        Auth::Inference {
            api_key: api_key.unwrap_or_default(),
            api_secret: gateway.api_secret.unwrap_or_default(),
            provider: gateway.provider,
            inference_class: gateway.inference_class,
            identity: Default::default(),
        }
    } else if is_azure {
        Auth::Azure { api_key, entra_token: config.entra_token }
    } else {
        Auth::Bearer(api_key.unwrap_or_default())
    };
    let connect = Connect {
        base_url,
        azure: is_azure.then_some(Azure { deployment: config.azure_deployment }),
        auth,
        conn: config.conn_options,
    };
    let runtime = (env.runtime)().map_err(ApiError::from)?;
    let mut native = OpenAIRealtime::new(
        dialect,
        "",
        Arc::new(lk_agents_net::websocket::TungsteniteTransport),
        Arc::new(TokioSpawner::new(runtime)),
        Arc::new(lk_agents_net::clock::RealClock::default()),
        crate::images::resolver(),
    )
    .with_options(opts, turn_detection_given)
    .with_connect(connect);
    if let Some(label) = config.label {
        native = native.with_label(label);
    }
    Ok(native)
}

impl OpenAIRealtimeModel for OpenAIRealtime {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        model: Option<String>,
        voice: Option<Voice>,
        modalities: Option<Vec<Modalities>>,
        tool_choice: Maybe<Option<ToolChoice>>,
        base_url: Option<String>,
        input_audio_transcription: Maybe<Option<AudioTranscription>>,
        input_audio_noise_reduction: Maybe<Option<NoiseReduction>>,
        turn_detection: Maybe<Option<TurnDetectionOption>>,
        speed: Maybe<f64>,
        tracing: Maybe<Option<Tracing>>,
        truncation: Maybe<Option<Truncation>>,
        reasoning: Maybe<Option<Reasoning>>,
        max_response_output_tokens: Maybe<Option<u32>>,
        api_key: Option<String>,
        azure_deployment: Option<String>,
        azure_endpoint: Option<String>,
        entra_token: Option<String>,
        max_session_duration: Maybe<Option<Duration>>,
        conn_options: Option<ApiConnectOptions>,
        dialect: Option<RealtimeProvider>,
        label: Option<String>,
        gateway: Option<InferenceGateway>,
    ) -> Result<Self, RealtimeConfigurationError> {
        let options = RealtimeOptions {
            voice: Maybe::NotGiven,
            tool_choice,
            input_audio_transcription,
            input_audio_noise_reduction,
            turn_detection,
            speed,
            tracing,
            truncation,
            reasoning,
            max_response_output_tokens,
        };
        build(
            env,
            Config {
                model,
                voice,
                modalities,
                options,
                base_url,
                api_key,
                azure_deployment,
                azure_endpoint,
                entra_token,
                max_session_duration,
                conn_options: conn_options.unwrap_or_default(),
                dialect,
                label,
                gateway,
            },
        )
    }

    fn provider(&self) -> String {
        self.provider()
    }
    fn model(&self) -> String {
        self.model()
    }
    fn label(&self) -> String {
        self.label()
    }
    fn capabilities(&self) -> lk_agents_abi::llm::realtime::RealtimeCapabilities {
        self.capabilities()
    }
    fn update_options(
        &self,
        voice: Maybe<Voice>,
        turn_detection: Maybe<Option<TurnDetectionOption>>,
        tool_choice: Maybe<Option<ToolChoice>>,
        input_audio_transcription: Maybe<Option<AudioTranscription>>,
        input_audio_noise_reduction: Maybe<Option<NoiseReduction>>,
        max_response_output_tokens: Maybe<Option<u32>>,
        speed: Maybe<f64>,
        tracing: Maybe<Option<Tracing>>,
        truncation: Maybe<Option<Truncation>>,
        reasoning: Maybe<Option<Reasoning>>,
    ) {
        self.update_options(patch(RealtimeOptions {
            voice,
            tool_choice,
            input_audio_transcription,
            input_audio_noise_reduction,
            turn_detection,
            speed,
            tracing,
            truncation,
            reasoning,
            max_response_output_tokens,
        }));
    }
    fn bind_identification(&self, headers: crate::headers::Headers) {
        self.bind_identification(headers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::dialect::XAI;
    use lk_agents_abi::llm::realtime_options::{
        AzureDeepNoiseSuppression, AzureVoice, NoiseReductionType, RetentionRatio, ServerVad,
        TruncationStrategy,
    };

    fn config(api_key: Option<&str>, base_url: Option<&str>) -> Config {
        Config {
            api_key: api_key.map(str::to_owned),
            base_url: base_url.map(str::to_owned),
            ..Config::default()
        }
    }

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let pairs: Vec<(String, String)> =
            pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
        move |name| pairs.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    // Catches: the OpenAI key or URL not falling back to the environment, an empty key counted
    // as given, or a missing key accepted.
    #[test]
    fn openai_takes_its_key_and_url_from_the_environment_when_not_given() {
        let env = vars(&[("OPENAI_API_KEY", "env-key"), ("OPENAI_BASE_URL", "https://proxy/v1")]);
        assert_eq!(
            endpoint(&config(Some(""), None), None, &env).unwrap(),
            (Some("env-key".into()), "https://proxy/v1".into())
        );
        assert_eq!(
            endpoint(&config(Some("mine"), Some("https://x/v1")), None, &env).unwrap(),
            (Some("mine".into()), "https://x/v1".into())
        );
        assert_eq!(
            endpoint(&config(Some("k"), None), None, vars(&[])).unwrap().1,
            OPENAI.default_base_url
        );
        let missing = endpoint(&config(None, None), None, vars(&[])).unwrap_err();
        assert!(missing.contains("OPENAI_API_KEY"));
    }

    // Catches: a named dialect reading OpenAI's key or URL instead of its own.
    #[test]
    fn a_named_dialect_takes_its_own_key_and_url() {
        let env = vars(&[("OPENAI_API_KEY", "openai"), ("XAI_API_KEY", "xai")]);
        assert_eq!(
            endpoint(&config(None, None), Some(&XAI), &env).unwrap(),
            (Some("xai".into()), XAI.default_base_url.into())
        );
        let missing = endpoint(&config(None, None), Some(&XAI), vars(&[])).unwrap_err();
        assert!(missing.contains("XAI_API_KEY"));
    }

    // Catches: Azure reading OPENAI_API_KEY, its endpoint not joined under `/openai`, or a base
    // URL accepted together with an endpoint.
    #[test]
    fn azure_builds_its_url_from_the_endpoint_and_keeps_its_own_key() {
        let env = vars(&[("OPENAI_API_KEY", "openai"), ("AZURE_OPENAI_ENDPOINT", "https://az/")]);
        let mut azure = config(None, None);
        azure.azure_deployment = Some("d".into());
        assert!(endpoint(&azure, None, &env).unwrap_err().starts_with("Missing credentials"));
        azure.api_key = Some("k".into());
        assert_eq!(
            endpoint(&azure, None, &env).unwrap(),
            (Some("k".into()), "https://az/openai".into())
        );
        assert!(endpoint(&azure, None, vars(&[])).unwrap_err().contains("AZURE_OPENAI_ENDPOINT"));
        azure.api_key = None;
        let env =
            vars(&[("AZURE_OPENAI_API_KEY", "az-key"), ("AZURE_OPENAI_ENDPOINT", "https://az")]);
        assert_eq!(
            endpoint(&azure, None, &env).unwrap(),
            (Some("az-key".into()), "https://az/openai".into())
        );
        azure.azure_endpoint = Some("https://mine//".into());
        assert_eq!(endpoint(&azure, None, &env).unwrap().1, "https://mine/openai");
        azure.base_url = Some("https://b".into());
        assert!(endpoint(&azure, None, &env).unwrap_err().contains("mutually exclusive"));
    }

    // Catches: an omitted option and an explicit `None` collapsing into one another, which either
    // clears a setting the caller left alone or keeps one the caller cleared.
    #[test]
    fn an_omitted_option_is_left_out_of_the_update_and_an_explicit_none_clears_it() {
        let options = RealtimeOptions {
            tracing: Maybe::Given(None),
            turn_detection: Maybe::Given(None),
            ..RealtimeOptions::default()
        };
        let update = serde_json::to_value(super::super::encode::options_update(
            "e",
            patch(options),
            &crate::realtime::OPENAI,
        ))
        .unwrap();
        let session = &update["session"];
        assert_eq!(session["tracing"], Value::Null);
        assert!(session.get("truncation").is_none() && session.get("reasoning").is_none());
        assert_eq!(session["audio"]["input"]["turn_detection"], Value::Null);
        assert!(session["audio"]["input"].get("transcription").is_none());
    }

    // Catches: an option record sent in a shape the API does not take: a missing `type` tag, an
    // unset field sent as null, or an unlimited reply length sent as null.
    #[test]
    fn option_records_are_sent_as_the_session_fields_the_api_documents() {
        let patch = patch(RealtimeOptions {
            voice: Maybe::Given(Voice::Azure(AzureVoice {
                name: "en-US-AvaNeural".into(),
                r#type: lk_agents_abi::llm::realtime_options::AzureVoiceType::AzureStandard,
                locale: Some("en-US".into()),
                endpoint_id: None,
                temperature: None,
            })),
            turn_detection: Maybe::Given(Some(TurnDetectionOption::Server(ServerVad {
                threshold: Some(0.6),
                prefix_padding_ms: None,
                silence_duration_ms: None,
                idle_timeout_ms: None,
                create_response: Some(false),
                interrupt_response: None,
            }))),
            input_audio_noise_reduction: Maybe::Given(Some(NoiseReduction::Microphone(
                NoiseReductionType::FarField,
            ))),
            tracing: Maybe::Given(Some(Tracing {
                workflow_name: None,
                group_id: None,
                metadata: None,
            })),
            truncation: Maybe::Given(Some(Truncation::RetentionRatio(RetentionRatio {
                retention_ratio: 0.8,
                post_instructions_token_limit: Some(5000),
            }))),
            max_response_output_tokens: Maybe::Given(None),
            ..RealtimeOptions::default()
        });
        assert_eq!(
            patch.voice.given(),
            Some(json!({"name": "en-US-AvaNeural", "type": "azure-standard", "locale": "en-US"}))
        );
        assert_eq!(
            patch.turn_detection.given().flatten().map(|td| td.to_json()),
            Some(json!({"type": "server_vad", "threshold": 0.6, "create_response": false}))
        );
        assert_eq!(patch.noise_reduction.given(), Some(Some(json!({"type": "far_field"}))));
        assert_eq!(patch.tracing.given(), Some(Some(json!("auto"))));
        assert_eq!(
            patch.truncation.given(),
            Some(Some(json!({
                "type": "retention_ratio", "retention_ratio": 0.8,
                "token_limits": {"post_instructions": 5000},
            })))
        );
        assert_eq!(patch.max_output_tokens.given(), Some(json!("inf")));
        assert_eq!(wire(Truncation::Strategy(TruncationStrategy::Disabled)), json!("disabled"));
        assert_eq!(
            wire(NoiseReduction::Azure(AzureDeepNoiseSuppression {})),
            json!({"type": "azure_deep_noise_suppression"})
        );
    }
}
