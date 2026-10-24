//! `_core.InferenceStt`'s keyword constructor and the gateway-specific stream messages: what the
//! declarations beside [`NativeStt`] cannot say.

use std::sync::Arc;

use lk_agents_abi::inference::fallback::SttFallbackModel;
use lk_agents_abi::python::{Ambient, HostProxy, check_pid};
use pyo3::prelude::*;

use serde_json::Value;

use super::options as stt_opts;
use super::{InferenceStt, InferenceSttStream, NativeStt, SttOptions};
use crate::extra::{ExtraKwargs, Fallbacks};
use crate::python::{conn_options, json_map, register_fn, session_clock};

#[pymethods]
impl InferenceStt {
    #[new]
    #[pyo3(signature = (
        *,
        model = None,
        language = None,
        encoding = "pcm_s16le".to_owned(),
        sample_rate = 24_000,
        base_url,
        api_key,
        api_secret,
        extra_kwargs = None,
        fallback = None,
        timeout = 10.0,
        vad = None,
        clock = None,
        headers = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Option<String>,
        language: Option<String>,
        encoding: String,
        sample_rate: u32,
        base_url: String,
        api_key: String,
        api_secret: String,
        extra_kwargs: Option<&Bound<'_, PyAny>>,
        fallback: Option<Fallbacks<SttFallbackModel>>,
        timeout: f64,
        vad: Option<&Bound<'_, PyAny>>,
        clock: Option<&Bound<'_, PyAny>>,
        headers: Option<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        let clock = clock.map(|c| session_clock(c, "clock")).transpose()?;
        check_pid()?;
        // A client is built per stream, so the session clock arrives here; absent, wall clock.
        let clock: Arc<dyn lk_agents_abi::Clock> =
            clock.unwrap_or_else(<Arc<dyn lk_agents_abi::Clock> as Ambient>::ambient);
        let extra_kwargs: ExtraKwargs = match extra_kwargs {
            Some(v) if !v.is_none() => v.extract()?,
            _ => ExtraKwargs::default(),
        };
        let opts = SttOptions {
            model,
            language,
            encoding,
            sample_rate,
            base_url,
            api_key,
            api_secret,
            extra_kwargs: extra_kwargs.0,
            fallback: fallback.map(|f| f.0).unwrap_or_default(),
            conn_options: conn_options(timeout),
        };
        // Not `parse_model_into`: the host already parsed `provider/model:language`; a second parse
        // would strip a legitimate `:` from the model name.
        let native = NativeStt::new(
            opts,
            Arc::new(crate::native::TungsteniteTransport),
            <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient(),
        )
        .with_clock(Arc::clone(&clock));
        // `get_inference_headers()`, read by the host where this stream opens.
        let native = match headers {
            Some(headers) => native.with_identification(headers.into_iter().collect()),
            None => native,
        };
        // Client-side VAD (Speechmatics): send `session.finalize` on end of speech.
        let native = match vad {
            // On the same clock as the socket: the VAD's `close` deadline is read against it.
            Some(obj) if !obj.is_none() => {
                native.with_vad(<dyn lk_agents_abi::Vad as HostProxy>::from_host(obj, Some(clock))?)
            }
            _ => native,
        };
        Ok(InferenceStt { inner: Arc::new(native) })
    }
}

#[pymethods]
impl InferenceSttStream {
    /// A mid-stream `session.update`, for the providers that take one (AssemblyAI, Deepgram Flux).
    #[pyo3(signature = (*, model = None, language = None, extra = None))]
    fn update_options(
        &self,
        model: Option<String>,
        language: Option<String>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        check_pid()?;
        let extra: Option<ExtraKwargs> = match extra {
            Some(v) if !v.is_none() => Some(v.extract()?),
            _ => None,
        };
        self.sink.update_options(model.as_deref(), language.as_deref(), extra.as_deref());
        Ok(())
    }

    /// Apply framework keyterms -- now if the user is silent, at end of speech if not.
    ///
    ///
    /// Deferred while the user speaks: keyterms mid-utterance make the gateway reconnect upstream
    /// and lose the tail of the speech.
    fn apply_keyterms(&self, extra: &Bound<'_, PyAny>) -> PyResult<()> {
        check_pid()?;
        let extra: ExtraKwargs =
            if extra.is_none() { ExtraKwargs::default() } else { extra.extract()? };
        self.sink.apply_keyterms(extra.0);
        Ok(())
    }
}

// `inference/stt.py`'s lookup tables. One table, no Python copy: a missing model sends a rejected
// parameter or claims alignment it lacks.

/// `_keyterms_extra_for_model`: the provider's keyterm `extra` entry, user terms merged with the
/// session's.
///
/// `None` means no keyterm prompting (the reference's capability check). Speechmatics'
/// `sounds_like` is preserved and Deepgram's bare-string `keyterm` is wrapped.
#[pyfunction]
#[pyo3(signature = (model = None, *, extra_kwargs = None, session_keyterms = None))]
pub fn stt_keyterms_extra_for_model<'py>(
    py: Python<'py>,
    model: Option<&str>,
    extra_kwargs: Option<&Bound<'py, PyAny>>,
    session_keyterms: Option<Vec<String>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    check_pid()?;
    let extra = json_map(extra_kwargs)?;
    let terms = session_keyterms.unwrap_or_default();
    let out = stt_opts::keyterms_extra_for_model(model, Some(&extra), &terms);
    match out {
        None => Ok(None),
        Some(m) => Ok(Some(lk_agents_abi::python::value_to_py(py, &Value::Object(m))?)),
    }
}

/// The whole `STTCapabilities` fold, as a dict the host constructs the dataclass from.
///
/// `aligned_transcript` is `all()` over the primary and fallback models: any turn may be served by
/// a fallback, and falsely claiming alignment disables the fast VAD interruption path.
#[pyfunction]
#[pyo3(signature = (model = None, *, fallback = None, extra_kwargs = None))]
pub fn stt_capabilities<'py>(
    py: Python<'py>,
    model: Option<&str>,
    fallback: Option<Vec<String>>,
    extra_kwargs: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    check_pid()?;
    let extra = json_map(extra_kwargs)?;
    // Only `model` is read off each entry.
    let models: Vec<stt_opts::FallbackModel> = fallback
        .unwrap_or_default()
        .into_iter()
        .map(|model| stt_opts::FallbackModel { model, extra_kwargs: None })
        .collect();
    let caps = stt_opts::capabilities(model, &models, Some(&extra));
    Ok(caps.into_pyobject(py)?.into_any())
}

register_fn!(stt_keyterms_extra_for_model, stt_capabilities);
