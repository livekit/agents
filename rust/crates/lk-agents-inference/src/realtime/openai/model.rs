//! The ABI-declared OpenAI-protocol realtime facade. Provider defaults and option resolution stay here.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::Maybe;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::realtime::{
    OpenAIRealtimeConfig, OpenAIRealtimeModel, RealtimeConfigurationError, RealtimeOptions,
    RealtimeProvider,
};
use lk_agents_net::task::TokioSpawner;
use serde_json::{Value, json};

use super::session::{Auth, Connect, OpenAIRealtime};
use super::wire::Azure;
use crate::realtime::dialect::{Dialect, OPENAI, TurnDetection};
use crate::realtime::machine::{OptionsPatch, SessionOptions};
use lk_agents_net::env::ProcessEnv;

fn patch(options: RealtimeOptions) -> OptionsPatch {
    // A bare noise-reduction name is its type.
    let noise_reduction = match options.input_audio_noise_reduction {
        Maybe::Given(Some(Value::String(name))) => Maybe::Given(Some(json!({"type": name}))),
        other => other,
    };
    OptionsPatch {
        voice: options.voice,
        tool_choice: options.tool_choice,
        turn_detection: match options.turn_detection {
            Maybe::NotGiven => Maybe::NotGiven,
            Maybe::Given(value) => Maybe::Given(value.map(TurnDetection::Raw)),
        },
        max_output_tokens: match options.max_response_output_tokens {
            Maybe::NotGiven => Maybe::NotGiven,
            Maybe::Given(value) => Maybe::Given(value.unwrap_or(Value::Null)),
        },
        transcription: options.input_audio_transcription,
        noise_reduction,
        speed: options.speed,
        tracing: options.tracing,
        truncation: options.truncation,
        reasoning: options.reasoning,
    }
}

/// The caller's key and base URL, with the environment's defaults: `with_azure`'s rules first
/// (`azure_defaults`), then the constructor's. Errors are the messages a caller sees.
fn endpoint(
    config: &OpenAIRealtimeConfig,
    env: impl Fn(&str) -> Option<String>,
) -> Result<(Option<String>, String), &'static str> {
    let nonempty = |value: &Option<String>| value.clone().filter(|value| !value.is_empty());
    let azure_base = |endpoint: String| format!("{}/openai", endpoint.trim_end_matches('/'));
    let mut api_key = config.api_key.clone();
    let mut base_url = config.base_url.clone();
    if config.azure_defaults {
        api_key = nonempty(&api_key).or_else(|| env("AZURE_OPENAI_API_KEY"));
        if api_key.is_none() && config.entra_token.is_none() {
            return Err(
                "Missing credentials. Please pass one of `api_key`, `entra_token`, or the \
                 `AZURE_OPENAI_API_KEY` environment variable.",
            );
        }
        if base_url.is_none() {
            let endpoint =
                nonempty(&config.azure_endpoint).or_else(|| env("AZURE_OPENAI_ENDPOINT")).ok_or(
                    "Missing Azure endpoint. Please pass the `azure_endpoint` parameter or set the \
                     `AZURE_OPENAI_ENDPOINT` environment variable.",
                )?;
            base_url = Some(azure_base(endpoint));
        } else if config.azure_endpoint.is_some() {
            return Err("base_url and azure_endpoint are mutually exclusive");
        }
    }
    let is_azure = config.entra_token.is_some() || config.azure_deployment.is_some();
    if !is_azure {
        api_key = nonempty(&api_key).or_else(|| env(OPENAI.api_key_env));
        if api_key.is_none() {
            return Err("The api_key client option must be set either by passing api_key to the \
                 client or by setting the OPENAI_API_KEY environment variable");
        }
    }
    if is_azure && nonempty(&base_url).is_none() {
        let endpoint = env("AZURE_OPENAI_ENDPOINT").ok_or(
            "Missing Azure endpoint. Please pass base_url or set AZURE_OPENAI_ENDPOINT \
             environment variable.",
        )?;
        base_url = Some(azure_base(endpoint));
    }
    let base_url = nonempty(&base_url).unwrap_or_else(|| {
        env("OPENAI_BASE_URL").unwrap_or_else(|| OPENAI.default_base_url.to_owned())
    });
    Ok((api_key, base_url))
}

impl OpenAIRealtimeModel for OpenAIRealtime {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        config: OpenAIRealtimeConfig,
    ) -> Result<Self, RealtimeConfigurationError> {
        let named = config
            .dialect
            .as_deref()
            .map(str::parse::<Dialect>)
            .transpose()
            .map_err(|e| RealtimeConfigurationError::Invalid(e.to_string()))?;
        // the gateway's caller resolved its own credentials and URL
        let (api_key, base_url) = match &named {
            Some(dialect) if dialect.provider == RealtimeProvider::Inference => {
                (config.api_key.clone(), config.base_url.clone().unwrap_or_default())
            }
            _ => endpoint(&config, |name| std::env::var(name).ok())
                .map_err(|error| RealtimeConfigurationError::Invalid(error.to_owned()))?,
        };
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
            opts.transcription = Some(json!({"model": "grok-transcribe"}));
            opts.voice = json!("eve");
            opts.turn_detection = Some(TurnDetection::ServerVad {
                threshold: 0.5,
                prefix_padding_ms: 300,
                silence_duration_ms: 200,
                create_response: true,
                interrupt_response: true,
            });
        }
        if let Some(voice) = config.voice {
            opts.voice = voice;
        }
        if dialect.forced_modalities.is_none() {
            opts.audio_output = config
                .modalities
                .is_none_or(|modalities| modalities.iter().any(|value| value == "audio"));
        }
        if config.azure_defaults {
            opts.transcription = Some(json!({"model": "whisper-1"}));
            opts.turn_detection = Some(TurnDetection::Raw(json!({
                "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                "silence_duration_ms": 200, "create_response": true,
            })));
        }
        let turn_detection_given = config.options.turn_detection.is_given();
        let max_session_duration = config.options.max_session_duration;
        let options = patch(config.options);
        super::session::warn_on_half_disabled_turn_taking(&options.turn_detection);
        opts.apply(&options);
        if let Maybe::Given(seconds) = max_session_duration {
            opts.max_session_duration =
                seconds.map(Duration::try_from_secs_f64).transpose().map_err(|error| {
                    RealtimeConfigurationError::Invalid(format!(
                        "invalid max_session_duration: {error}"
                    ))
                })?;
        }
        let is_azure = config.azure_deployment.is_some() || config.entra_token.is_some();
        let auth = if is_inference {
            Auth::Inference {
                api_key: api_key.unwrap_or_default(),
                api_secret: config.api_secret.unwrap_or_default(),
                provider: config.inference_provider,
                inference_class: config.inference_class,
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
    fn update_options(&self, options: RealtimeOptions) {
        self.update_options(patch(options));
    }
    fn bind_identification(&self, headers: Vec<(String, String)>) {
        self.bind_identification(headers.into_iter().collect());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(api_key: Option<&str>, base_url: Option<&str>) -> OpenAIRealtimeConfig {
        OpenAIRealtimeConfig {
            api_key: api_key.map(str::to_owned),
            conn_options: Default::default(),
            options: RealtimeOptions::default(),
            dialect: None,
            model: None,
            voice: None,
            modalities: None,
            azure_deployment: None,
            entra_token: None,
            azure_defaults: false,
            base_url: base_url.map(str::to_owned),
            azure_endpoint: None,
            api_secret: None,
            inference_provider: None,
            inference_class: None,
            label: None,
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
            endpoint(&config(Some(""), None), &env).unwrap(),
            (Some("env-key".into()), "https://proxy/v1".into())
        );
        assert_eq!(
            endpoint(&config(Some("mine"), Some("https://x/v1")), &env).unwrap(),
            (Some("mine".into()), "https://x/v1".into())
        );
        assert_eq!(
            endpoint(&config(Some("k"), None), vars(&[])).unwrap().1,
            OPENAI.default_base_url
        );
        assert!(endpoint(&config(None, None), vars(&[])).unwrap_err().contains("OPENAI_API_KEY"));
    }

    // Catches: Azure reading OPENAI_API_KEY, its endpoint not joined under `/openai`, or
    // `with_azure` accepting both a base URL and an endpoint.
    #[test]
    fn azure_builds_its_url_from_the_endpoint_and_keeps_its_own_key() {
        let env = vars(&[("OPENAI_API_KEY", "openai"), ("AZURE_OPENAI_ENDPOINT", "https://az/")]);
        let mut azure = config(None, None);
        azure.azure_deployment = Some("d".into());
        assert_eq!(endpoint(&azure, &env).unwrap(), (None, "https://az/openai".into()));
        assert!(endpoint(&azure, vars(&[])).unwrap_err().contains("pass base_url"));

        azure.azure_defaults = true;
        assert!(endpoint(&azure, &env).unwrap_err().starts_with("Missing credentials"));
        let env =
            vars(&[("AZURE_OPENAI_API_KEY", "az-key"), ("AZURE_OPENAI_ENDPOINT", "https://az")]);
        assert_eq!(
            endpoint(&azure, &env).unwrap(),
            (Some("az-key".into()), "https://az/openai".into())
        );
        azure.azure_endpoint = Some("https://mine//".into());
        assert_eq!(endpoint(&azure, &env).unwrap().1, "https://mine/openai");
        azure.base_url = Some("https://b".into());
        assert!(endpoint(&azure, &env).unwrap_err().contains("mutually exclusive"));
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
}
