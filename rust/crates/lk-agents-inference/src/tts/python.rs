//! `_core.InferenceTts`'s keyword constructor body and the two calls its declarations cannot say:
//! binding a session clock, and `prewarm()` (a background task over the shared client).

use std::sync::Arc;

use lk_agents_abi::python::{Ambient, check_pid};
use lk_agents_abi::{SpawnerExt as _, StreamSink as _, StreamSource as _};
use pyo3::prelude::*;

use super::{InferenceTts, NativeTts, TtsClientConfig, TtsOptions};
use crate::python::{conn_options, session_clock};

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
        conn_options: conn_options(cfg.timeout),
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
        let inner = Arc::clone(&self.inner);
        <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient().spawn_unboxed(async move {
            // Opening and dropping a stream is the cheapest way to force the handshake.
            if let Ok((sink, source)) = inner.stream_identified(headers.map(|pairs| pairs.into_iter().collect())) {
                sink.end_input();
                Box::new(source).close(lk_agents_abi::python::NATIVE_CLOSE_DEADLINE).await;
            }
        });
        Ok(())
    }
}
