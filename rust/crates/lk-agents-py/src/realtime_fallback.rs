//! `_core.RealtimeFallbackModel`: the host's model objects resolved to the chain's members, a
//! native model by its registry entry and a plugin's through its host view.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::RealtimeCapabilities;
use lk_agents_abi::llm::realtime::RealtimeConfigurationError;
use lk_agents_abi::python::native;
use lk_agents_abi::{HostRef, Realtime, RealtimeSession};
use lk_agents_runtime::inference::fallback::realtime::{ChainModel, RealtimeFallbackModel};
use lk_agents_runtime::inference::fallback::{ChainObserver, IgnoreEvents};
use lk_agents_runtime::net::env::ProcessEnv;
use pyo3::prelude::*;

pub struct RealtimeFallback(RealtimeFallbackModel);

impl lk_agents_abi::llm::fallback::RealtimeFallbackModel for RealtimeFallback {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        models: Vec<HostRef>,
        cooldown: f64,
        attempt_timeout: Option<f64>,
        regenerate_on_swap: bool,
        on_availability: Option<HostRef>,
    ) -> Result<Self, RealtimeConfigurationError> {
        let (children, observer) = Python::attach(|py| -> PyResult<_> {
            let children = models
                .into_iter()
                .map(|model| {
                    let model = model.into_pyobject(py)?;
                    match native::<dyn Realtime>(&model) {
                        Some(native) => Ok(ChainModel::Native(native)),
                        None => crate::generated::llm::HostRealtimeProxy::from_host(&model)
                            .and_then(|host| {
                                Ok(ChainModel::Host(host, crate::host::host_spawner()?))
                            }),
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            let observer: Arc<dyn ChainObserver> = match on_availability {
                Some(observer) => crate::generated::llm::ChainObserverProxy::from_host(
                    &observer.into_pyobject(py)?,
                )?,
                None => Arc::new(IgnoreEvents),
            };
            Ok((children, observer))
        })
        .map_err(|error| Python::attach(|py| lk_agents_abi::python::capture_host(py, &error)))?;
        RealtimeFallbackModel::new(
            children,
            cooldown,
            attempt_timeout,
            regenerate_on_swap,
            observer,
            env.clock.clone(),
        )
        .map(RealtimeFallback)
        .map_err(RealtimeConfigurationError::Invalid)
    }

    fn active_index(&self) -> usize {
        self.0.active_index()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        Realtime::capabilities(&self.0)
    }

    async fn restart(&self, switch_model: bool) {
        self.0.restart(switch_model).await;
    }
}

impl Realtime for RealtimeFallback {
    fn label(&self) -> String {
        self.0.label()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        Realtime::capabilities(&self.0)
    }

    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move { self.0.session(turn_detection_disabled).await })
    }
}
