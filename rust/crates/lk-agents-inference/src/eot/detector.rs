//! `_core.InferenceEot` and `_core.EotStream`: `inference.TurnDetector`'s threshold table and the
//! streams it opens. One table is shared by the detector and all its streams, so a stream's
//! cloud-to-local fallback rescales what the detector answers.

use std::sync::{Arc, Mutex, MutexGuard};

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::{
    LanguageCode, ThresholdOptions, ThresholdOverride, TurnDetectorModel,
};
use lk_agents_abi::turn::inference::{EotMetricsObserver, EotModel, EotModelFactory};
use lk_agents_abi::{ApiConnectOptions, AudioFrame};

use super::{
    CloudOptions, CloudTransport, DetectorStream, EotRequestId, EotSession, EotStream,
    EotTransport, LocalEotModel, LocalTransport, MetricsCallback, TransportFactory, MountedTransport,
};
use lk_agents_net::env::ProcessEnv;

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
        self.thresholds.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The gateway for `turn-detector-v1`, else the linked native mini model or the host's.
    fn transports(
        &self,
        headers: Vec<(String, String)>,
        conn_options: ApiConnectOptions,
    ) -> TransportFactory {
        let cloud = CloudOptions {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            api_secret: self.api_secret.clone(),
            conn_options,
            sample_rate: self.sample_rate,
            headers: Arc::new(move || headers.clone()),
        };
        let thresholds = Arc::clone(&self.thresholds);
        let local_model = Arc::clone(&self.local_model);
        let clock = self.env.clock.clone();
        Arc::new(move |model| {
            if model == TurnDetectorModel::V1 {
                let (cloud, clock) = (cloud.clone(), clock.clone());
                return MountedTransport::mount(model.as_str(), thresholds.clone(), move |sink| {
                    let socket = Arc::new(lk_agents_net::websocket::TungsteniteTransport);
                    let transport = CloudTransport::new(cloud, sink, socket);
                    Arc::new(transport.with_clock(clock)) as Arc<dyn EotTransport>
                });
            }
            let local = match crate::local_models::native() {
                Some(native) => Ok((native.eot)()),
                None => local_model
                    .create()
                    .map(|model| Arc::new(HostModel(model)) as Arc<dyn LocalEotModel>),
            };
            MountedTransport::mount(model.as_str(), thresholds.clone(), move |sink| match local {
                Ok(model) => Arc::new(LocalTransport::new(model, sink)) as Arc<dyn EotTransport>,
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
                    log::error!(target: "livekit.agents", "turn detector metrics observer failed: {error}");
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

    fn lookup(&self, language: Option<LanguageCode>) -> Option<f64> {
        self.table().lookup(language.as_ref())
    }

    fn lookup_backchannel(&self, language: Option<LanguageCode>) -> Option<f64> {
        self.table().lookup_backchannel(language.as_ref())
    }

    fn supports(&self, language: Option<LanguageCode>) -> bool {
        self.table().supports(language.as_ref())
    }

    fn update_overrides(&self, overrides: Option<ThresholdOverride>) {
        self.table().update_overrides(overrides);
    }

    fn update_backchannel_overrides(&self, overrides: Option<ThresholdOverride>) {
        self.table().update_backchannel_overrides(overrides);
    }

    fn stream(
        &self,
        headers: Vec<(String, String)>,
        conn_options: ApiConnectOptions,
    ) -> Result<DetectorStream, ProviderError> {
        let spawner = (self.env.spawner)().map_err(|error| {
            ProviderError::Local(ApiError::new(format!("turn detector unavailable: {error}")))
        })?;
        let model = self.table().model();
        let session = EotSession::new(model, Arc::clone(&self.thresholds), self.local_fallback);
        let make = self.transports(headers, conn_options);
        let on_metrics = self.on_metrics.clone();
        Ok(EotStream::open(
            session,
            self.sample_rate,
            make,
            self.env.media_pool,
            on_metrics,
            spawner.as_ref(),
        ))
    }
}

/// The host's mini model. A raise is scored 0.0 (the reference's), not 1.0, which would cut the
/// user off at every pause.
struct HostModel(Arc<dyn EotModel>);

impl LocalEotModel for HostModel {
    fn predict(&self, pcm: &[i16]) -> Result<f64, String> {
        self.0.predict(pcm.to_vec()).map_err(|e| e.to_string())
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
