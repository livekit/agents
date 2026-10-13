//! One entry of the gateway's server-side fallback chain (`FallbackModel`). On the wire STT sends an
//! absent `extra_kwargs` as `null` and TTS as `{}`; that lives in the request builders.

use lk_agents_host_derive::Host;

/// Inference Fallback Adapter: configuration for a fallback STT model that runs server-side in LiveKit Inference, providing automatic fallback between providers.
///
/// Extra fields are passed through to the provider.
///
/// Example:
///     >>> FallbackModel(model="deepgram/nova-3", extra_kwargs={"keyterm": ["livekit"]})
//
// The `///` above is the reference's docstring (`inference/stt.py`), carried by the `TypedDict`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    typed_dict_from = "typing",
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "FallbackModel"
)]
#[node(interface)]
#[binding(emit)]
pub struct SttFallbackModel {
    /// Model name (e.g. "deepgram/nova-3", "assemblyai/universal-streaming", "cartesia/ink-whisper").
    #[python(ty = "Required[str]", import = "typing_extensions:Required")]
    #[node(ty = "string")]
    pub model: String,
    /// Extra configuration for the model.
    //
    // Omitted rather than nulled: `_normalize_fallback("x")` must equal `[{"model": "x"}]`. On the
    // read side absent and `None` are one request. `serde_json::` paths because the extract type is
    // copied verbatim into the generated module.
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    #[node(ty = "Record<string, unknown>")]
    #[binding(skip_if_none)]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Inference Fallback Adapter: configuration for a fallback TTS model that runs server-side in LiveKit Inference, providing automatic fallback between providers.
///
/// Extra fields are passed through to the provider.
///
/// Example:
///     >>> FallbackModel(model="cartesia/sonic", voice="")
//
// The `///` above is the reference's docstring (`inference/tts.py`, a `total=True` `TypedDict`).
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    typed_dict_from = "typing",
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "FallbackModel"
)]
#[node(interface)]
#[binding(emit)]
pub struct TtsFallbackModel {
    /// Model name (e.g. "cartesia/sonic", "elevenlabs/eleven_flash_v2", "rime/coda").
    pub model: String,
    /// Voice to use for the model.
    //
    // Required in the reference; `""` when a bare model string named no voice.
    pub voice: String,
    /// Extra configuration for the model.
    //
    // Omitted, not `None`: a `None` would reach the gateway as `"extra": null`, STT's protocol.
    #[python(
        ty = "NotRequired[dict[str, Any]]",
        import = "typing:Any",
        import = "typing_extensions:NotRequired"
    )]
    #[node(ty = "Record<string, unknown>")]
    #[binding(skip_if_none)]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Absent and `None` are one request; anything but a JSON object raises.
#[cfg(feature = "python")]
fn extra_kwargs(v: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Option<serde_json::Map<String, serde_json::Value>>> {
    use pyo3::types::PyAnyMethods as _;
    if v.is_none() {
        return Ok(None);
    }
    match crate::python::value_from_py(v, "inference")? {
        serde_json::Value::Object(map) => Ok(Some(map)),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!("expected a dict of extra kwargs, got {other}"))),
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for SttFallbackModel {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{item, item_or};
        const OWNER: &str = "SttFallbackModelFromPy";
        let obj = &*obj;
        Ok(SttFallbackModel { model: item(obj, OWNER, "model")?, extra_kwargs: item_or(obj, OWNER, "extra_kwargs", extra_kwargs, || None)? })
    }
}

/// A `None` voice is `""`.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for TtsFallbackModel {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{item, item_or};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "TtsFallbackModelFromPy";
        let obj = &*obj;
        Ok(TtsFallbackModel {
            model: item(obj, OWNER, "model")?,
            voice: item_or(obj, OWNER, "voice", |v| if v.is_none() { Ok(String::new()) } else { v.extract() }, String::new)?,
            extra_kwargs: item_or(obj, OWNER, "extra_kwargs", extra_kwargs, || None)?,
        })
    }
}
