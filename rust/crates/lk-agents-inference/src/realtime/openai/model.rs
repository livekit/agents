//! The ABI-declared OpenAI-protocol realtime facade. Provider defaults and option resolution stay here.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::Maybe;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::realtime::{
    OpenAIRealtimeConfig, OpenAIRealtimeModel, RealtimeConfigurationError, RealtimeOptions,
};
use lk_agents_net::task::TokioSpawner;
use serde_json::{Value, json};

use super::session::{Auth, Connect, OpenAIRealtime};
use super::wire::Azure;
use crate::realtime::dialect::{Dialect, INFERENCE, TurnDetection};
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

impl OpenAIRealtimeModel for OpenAIRealtime {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        config: OpenAIRealtimeConfig,
    ) -> Result<Self, RealtimeConfigurationError> {
        let dialect = match config.dialect {
            Some(name) => name.parse().map_err(|e: crate::realtime::dialect::UnknownDialect| {
                RealtimeConfigurationError::Invalid(e.to_string())
            })?,
            None => Dialect::for_base_url(&config.base_url),
        };
        let is_inference = dialect.provider_label == INFERENCE.provider_label;
        let mut opts = SessionOptions::from_dialect(&dialect, false);
        if let Some(model) = config.model {
            opts.model = model;
        }
        if is_inference && opts.model.starts_with("xai/") {
            opts.transcription = Some(json!({"model": "grok-transcribe"}));
            opts.voice = "eve".to_owned();
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
                api_key: config.api_key.unwrap_or_default(),
                api_secret: config.api_secret.unwrap_or_default(),
                provider: config.inference_provider,
                inference_class: config.inference_class,
                identity: Default::default(),
            }
        } else if is_azure {
            Auth::Azure { api_key: config.api_key, entra_token: config.entra_token }
        } else {
            Auth::Bearer(config.api_key.unwrap_or_default())
        };
        let connect = Connect {
            base_url: config.base_url,
            azure: is_azure.then_some(Azure { deployment: config.azure_deployment }),
            auth,
            conn: config.conn_options,
        };
        let runtime =
            (env.runtime)().map_err(|error| ApiError::new(error).with_retryable(false))?;
        let mut native = OpenAIRealtime::new(
            dialect,
            "",
            Arc::new(lk_agents_net::websocket::TungsteniteTransport),
            Arc::new(TokioSpawner::new(runtime)),
            Arc::new(lk_agents_net::clock::RealClock::default()),
            env.images.clone(),
            env.media_pool,
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

    // Catches: an omitted option and an explicit `None` collapsing into one another, which either
    // clears a setting the caller left alone or keeps one the caller cleared.
    #[test]
    fn an_omitted_option_is_left_out_of_the_update_and_an_explicit_none_clears_it() {
        let options = RealtimeOptions {
            tracing: Maybe::Given(None),
            turn_detection: Maybe::Given(None),
            ..RealtimeOptions::default()
        };
        let update =
            super::super::encode::options_update("e", patch(options), &crate::realtime::OPENAI);
        let session = &update["session"];
        assert_eq!(session["tracing"], Value::Null);
        assert!(session.get("truncation").is_none() && session.get("reasoning").is_none());
        assert_eq!(session["audio"]["input"]["turn_detection"], Value::Null);
        assert!(session["audio"]["input"].get("transcription").is_none());
    }
}
