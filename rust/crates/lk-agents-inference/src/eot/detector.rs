use super::{
    CloudOptions, CloudTransport, DetectorStream, EotRequestId, EotSession, EotStream,
    EotTransport, LocalEotModel, LocalTransport, MetricsCallback, MountedTransport,
    TransportFactory,
};
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::{ApiError, HostError, ProviderError};
use lk_agents_abi::inference::EotSelection;
use lk_agents_abi::tokenize::language::{ThresholdOptions, ThresholdOverride, TurnDetectorModel};
use lk_agents_abi::turn::inference::{EotMetricsObserver, EotModel, EotModelFactory};
use lk_agents_abi::{ApiConnectOptions, AudioFrame};
use lk_agents_net::env::ProcessEnv;
use lk_agents_telemetry as telemetry;
use parking_lot::{Mutex, MutexGuard};
use std::sync::Arc;

/// `env` lends the runtime the streams run on.
pub struct AudioEot {
    thresholds: Arc<Mutex<ThresholdOptions>>,
    local_fallback: bool,
    base_url: String,
    api_key: String,
    api_secret: String,
    sample_rate: u32,
    local_model: Arc<dyn EotModelFactory>,
    on_metrics: Option<MetricsCallback>,
    env: ProcessEnv,
}

impl AudioEot {
    fn table(&self) -> MutexGuard<'_, ThresholdOptions> {
        self.thresholds.lock()
    }

    /// The gateway for `turn-detector-v1`, else the linked native mini model or the host's.
    fn transports(&self, cx: &telemetry::Cx, conn_options: ApiConnectOptions) -> TransportFactory {
        let cloud = CloudOptions {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            api_secret: self.api_secret.clone(),
            conn_options,
            sample_rate: self.sample_rate,
            cx: cx.clone(),
        };
        let thresholds = Arc::clone(&self.thresholds);
        let local_model = Arc::clone(&self.local_model);
        let clock = self.env.clock.clone();
        Arc::new(move |model| {
            if model == TurnDetectorModel::V1 {
                let (cloud, clock) = (cloud.clone(), clock.clone());
                let cx = cloud.cx.clone();
                return MountedTransport::mount(model.as_str(), thresholds.clone(), move |sink| {
                    let socket = Arc::new(lk_agents_net::websocket::TungsteniteTransport);
                    let transport = CloudTransport::new(cloud, sink, socket, clock);
                    transport.observe(cx);
                    Arc::new(transport) as Arc<dyn EotTransport>
                });
            }
            let local = match crate::local_models::native() {
                Some(native) => Ok((native.eot)()),
                None => local_model
                    .create()
                    .map(|model| Arc::new(HostModel(model)) as Arc<dyn LocalEotModel>),
            };
            MountedTransport::mount(model.as_str(), thresholds.clone(), move |sink| match local {
                Ok(model) => {
                    Arc::new(LocalTransport::new(model, sink, lk_agents_media::pool::media_pool()))
                        as Arc<dyn EotTransport>
                }
                Err(e) => Arc::new(FailedTransport(e.to_string())) as Arc<dyn EotTransport>,
            })
        })
    }
}

impl lk_agents_abi::turn::inference::InferenceEot for AudioEot {
    type Stream = DetectorStream;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        model: TurnDetectorModel,
        overrides: Option<ThresholdOverride>,
        backchannel_overrides: Option<ThresholdOverride>,
        local_fallback: bool,
        base_url: String,
        api_key: String,
        api_secret: String,
        sample_rate: u32,
        local_model: Arc<dyn EotModelFactory>,
        on_metrics: Option<Arc<dyn EotMetricsObserver>>,
    ) -> Self {
        let on_metrics = on_metrics.map(|observer| {
            Arc::new(move |metrics| {
                if let Err(error) = observer.collected(metrics) {
                    log::error!(target: lk_agents_abi::LOG_TARGET, "turn detector metrics observer failed: {error}");
                }
            }) as MetricsCallback
        });
        AudioEot {
            thresholds: Arc::new(Mutex::new(ThresholdOptions::new(
                model,
                overrides,
                backchannel_overrides,
            ))),
            local_fallback,
            base_url,
            api_key,
            api_secret,
            sample_rate,
            local_model,
            on_metrics,
            env: env.clone(),
        }
    }

    fn model(&self) -> TurnDetectorModel {
        self.table().model()
    }

    fn overrides(&self) -> Option<ThresholdOverride> {
        self.table().overrides().cloned()
    }

    fn backchannel_overrides(&self) -> Option<ThresholdOverride> {
        self.table().backchannel_overrides().cloned()
    }

    fn update_overrides(&self, overrides: Option<ThresholdOverride>) {
        self.table().update_overrides(overrides);
    }

    fn update_backchannel_overrides(&self, overrides: Option<ThresholdOverride>) {
        self.table().update_backchannel_overrides(overrides);
    }

    fn stream(
        &self,
        cx: telemetry::Cx,
        conn_options: ApiConnectOptions,
    ) -> Result<DetectorStream, ProviderError> {
        let spawner = (self.env.spawner)().map_err(|error| {
            ProviderError::Local(ApiError::new(format!("turn detector unavailable: {error}")))
        })?;
        let model = self.table().model();
        let session = EotSession::new(model, Arc::clone(&self.thresholds), self.local_fallback);
        let make = self.transports(&cx, conn_options);
        let on_metrics = self.on_metrics.clone();
        Ok(EotStream::open(
            session,
            self.sample_rate,
            make,
            lk_agents_media::pool::media_pool(),
            on_metrics,
            spawner.as_ref(),
        ))
    }
}

/// The host's mini model. A raise is scored 0.0 (the reference's), not 1.0, which would cut the
/// user off at every pause.
struct HostModel(Arc<dyn EotModel>);

impl LocalEotModel for HostModel {
    fn predict(&self, pcm: &[i16]) -> Result<f64, HostError> {
        self.0.predict(pcm.to_vec())
    }
}

/// A local model that could not be loaded: its `run()` fails, which the session handles as a
/// local failure (warned once, every prediction 1.0).
struct FailedTransport(String);

#[async_trait::async_trait]
impl EotTransport for FailedTransport {
    fn session_id(&self) -> Option<String> {
        None
    }
    fn run_inference(&self, _request_id: &EotRequestId) {}
    fn push_frame(&self, _frame: &AudioFrame) {}
    fn flush(&self) {}
    fn end_input(&self) {}
    fn detach(&self) {}
    async fn run(&self) -> Result<(), ApiError> {
        Err(ApiError::new(self.0.clone()))
    }
}

/// The turn detector a host constructs.
pub fn select(
    version: Option<&str>,
    hosted: bool,
    dev: bool,
    url: &str,
    key: &str,
    secret: &str,
    local_fallback: bool,
) -> Result<EotSelection, Invalid> {
    let mut cloud = version.map_or(hosted || dev, |v| v == "v1");
    let mut warnings = vec![];
    if cloud {
        let missing: Vec<_> = [
            (url, "LIVEKIT_INFERENCE_URL"),
            (key, "LIVEKIT_API_KEY"),
            (secret, "LIVEKIT_API_SECRET"),
        ]
        .into_iter()
        .filter(|(value, _)| value.is_empty())
        .map(|(_, name)| name)
        .collect();
        if !missing.is_empty() {
            let missing = missing.join(", ");
            if version.is_some() {
                return Err(Invalid(format!(
                    "TurnDetector(version='v1') requires {missing} (env or constructor argument)."
                )));
            }
            warnings.push(format!("LIVEKIT_INFERENCE_URL is set but {missing} missing; falling back to the turn-detector-v1-mini model"));
            cloud = false;
        }
    }
    let model = if cloud { TurnDetectorModel::V1 } else { TurnDetectorModel::V1Mini };
    if !local_fallback && !cloud {
        warnings.push(format!(
            "local_fallback=False has no effect on the {} model, which runs locally by design",
            model.as_str()
        ));
    }
    Ok(EotSelection { model, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_cloud_requires_credentials_but_auto_degrades() {
        assert!(select(Some("v1"), false, false, "url", "", "secret", true).is_err());
        let EotSelection { model, warnings } =
            select(None, true, false, "url", "", "secret", true).unwrap();
        assert_eq!(model, TurnDetectorModel::V1Mini);
        assert_eq!(warnings.len(), 1);
        let model = |version, hosted| select(version, hosted, true, "url", "key", "secret", true);
        assert_eq!(model(None, false).unwrap().model, TurnDetectorModel::V1);
        // Hosted on Cloud in production: the cloud model, as in development (1.x `is_hosted()`).
        let hosted = select(None, true, false, "url", "key", "secret", true).unwrap();
        assert_eq!(hosted.model, TurnDetectorModel::V1);
        assert_eq!(model(Some("v1-mini"), true).unwrap().model, TurnDetectorModel::V1Mini);
        assert_eq!(select(None, false, false, "", "", "", false).unwrap().warnings.len(), 1);
    }
}
