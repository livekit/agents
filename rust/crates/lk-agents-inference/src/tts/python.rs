//! `_core.InferenceTts`'s keyword constructor body and the two calls its declarations cannot say:
//! binding a session clock, and `prewarm()` (a background task over the shared client).

use std::sync::Arc;

use lk_agents_abi::python::{Ambient, check_pid};
use pyo3::prelude::*;

use super::{InferenceTts, NativeTts, TtsClientConfig, TtsOptions};
use crate::python::session_clock;

/// The keyword constructor's body (`TtsClientConfig`'s `#[host(client)]`).
pub(crate) fn build(_py: Python<'_>, cfg: TtsClientConfig) -> PyResult<InferenceTts> {
    let opts = TtsOptions {
        model: cfg.model,
        voice: cfg.voice,
        language: cfg.language,
        encoding: cfg.encoding,
        sample_rate: cfg.sample_rate,
        base_url: cfg.base_url,
        // The token is re-minted per connection, so a stream opened later is never stale.
        api_key: cfg.api_key,
        api_secret: cfg.api_secret,
        extra_kwargs: cfg.extra_kwargs.unwrap_or_default().0,
        fallback: cfg.fallback.map(|f| f.0).unwrap_or_default(),
        conn_options: cfg.conn_options,
    };
    let native = NativeTts::new(
        opts,
        Arc::new(crate::native::TungsteniteTransport),
        <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient(),
    )
    .with_clock(<Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient());
    Ok(InferenceTts { inner: Arc::new(native) })
}

#[pymethods]
impl InferenceTts {
    /// The client as a session serving the job `headers` (`get_inference_headers()`) identifies
    /// holds it: the same options and pooled connections, its syntheses carrying that job.
    /// `reporter` is the host `tts.TTS` whose `error` events the session's failures are.
    #[pyo3(signature = (headers, reporter = None))]
    fn identified(
        &self,
        headers: Vec<(String, String)>,
        reporter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<InferenceTts> {
        lk_agents_abi::python::check_pid()?;
        let reporter = reporter
            .map(|r| <dyn lk_agents_abi::Tts as lk_agents_abi::python::HostProxy>::from_host(&r, None))
            .transpose()?;
        let view = self.inner.identified(headers.into_iter().collect(), reporter);
        Ok(InferenceTts { inner: Arc::new(view) })
    }

    /// Bind the session's clock: every stream opened from here runs in session time.
    fn bind_clock(&self, clock: &Bound<'_, PyAny>) -> PyResult<()> {
        let clock = session_clock(clock, "clock")?;
        check_pid()?;
        self.inner.bind_clock(clock);
        Ok(())
    }

    /// `prewarm()`: warm DNS and the TLS session in the background. Failures are ignored.
    #[pyo3(signature = (*, headers = None))]
    fn prewarm(&self, headers: Option<Vec<(String, String)>>) -> PyResult<()> {
        check_pid()?;
        self.inner.prewarm_identified(headers.map(|pairs| pairs.into_iter().collect()));
        Ok(())
    }
}
