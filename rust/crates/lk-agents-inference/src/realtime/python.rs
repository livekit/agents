//! `_core.NativeRealtimeModel`'s keyword constructor and `update_options`: what the declaration
//! beside [`NativeRealtime`] cannot say. The host passes the plugin's arguments as given (pydantic
//! objects, `NOT_GIVEN`, `None`); the defaults and conversions are here.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::python::{Ambient, MaybeNull, check_pid, value_from_py};
use lk_agents_abi::{ApiConnectOptions, Maybe, ToolChoice};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use serde_json::{Value, json};

use super::dialect::{Dialect, INFERENCE, TurnDetection};
use super::machine::{OptionsPatch, SessionOptions};
use super::session::{Auth, Connect, NativeRealtime, PyNativeRealtimeModel};
use super::wire::Azure;

/// `AZURE_DEFAULT_INPUT_AUDIO_TRANSCRIPTION`, `with_azure`'s default.
fn azure_default_transcription() -> Value {
    json!({"model": "whisper-1"})
}

/// `AZURE_DEFAULT_TURN_DETECTION`, `with_azure`'s default.
fn azure_default_turn_detection() -> TurnDetection {
    TurnDetection::Raw(json!({
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 200,
        "create_response": true,
    }))
}

/// LiveKit Inference serving an xAI model: xAI's own transcription, voice and server VAD.
fn inference_xai_defaults(opts: &mut SessionOptions) {
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

/// A plugin option as JSON: a pydantic object as `model_dump(by_alias=True, exclude_unset=True)`
/// (what the reference puts on the wire), anything else as its JSON value.
fn option(v: Option<&Bound<'_, PyAny>>, path: &str) -> PyResult<MaybeNull<Value>> {
    let Some(v) = v else { return Ok(MaybeNull::NotGiven) };
    let v = match v.extract::<MaybeNull<Bound<'_, PyAny>>>()? {
        MaybeNull::NotGiven => return Ok(MaybeNull::NotGiven),
        MaybeNull::Null => return Ok(MaybeNull::Null),
        MaybeNull::Given(v) => v,
    };
    let v = &v;
    if v.hasattr("model_dump")? {
        let kwargs = PyDict::new(v.py());
        kwargs.set_item("by_alias", true)?;
        kwargs.set_item("exclude_unset", true)?;
        let dumped = v.call_method("model_dump", (), Some(&kwargs))?;
        return Ok(MaybeNull::Given(value_from_py(&dumped, path)?));
    }
    Ok(MaybeNull::Given(value_from_py(v, path)?))
}

/// `to_noise_reduction`: a bare type name becomes `{"type": name}`.
fn noise_reduction(v: Option<&Bound<'_, PyAny>>) -> PyResult<MaybeNull<Value>> {
    Ok(match option(v, "input_audio_noise_reduction")? {
        MaybeNull::Given(Value::String(t)) => MaybeNull::Given(json!({"type": t})),
        other => other,
    })
}

fn maybe(v: MaybeNull<Value>) -> Maybe<Option<Value>> {
    v.into_core()
}

fn turn_detection(v: MaybeNull<Value>) -> Maybe<Option<TurnDetection>> {
    match v {
        MaybeNull::NotGiven => Maybe::NotGiven,
        MaybeNull::Null => Maybe::Given(None),
        MaybeNull::Given(td) => Maybe::Given(Some(TurnDetection::Raw(td))),
    }
}

fn tool_choice(v: Option<&Bound<'_, PyAny>>) -> PyResult<Maybe<Option<ToolChoice>>> {
    let Some(v) = v else { return Ok(Maybe::NotGiven) };
    Ok(match v.extract::<MaybeNull<Bound<'_, PyAny>>>()? {
        MaybeNull::NotGiven => Maybe::NotGiven,
        MaybeNull::Null => Maybe::Given(None),
        // `tool_choice or None`
        MaybeNull::Given(v) if v.is_instance_of::<PyString>() && v.extract::<String>()?.is_empty() => {
            Maybe::Given(None)
        }
        MaybeNull::Given(v) => Maybe::Given(Some(v.extract()?)),
    })
}

fn float(v: Option<&Bound<'_, PyAny>>) -> PyResult<Maybe<Option<f64>>> {
    let Some(v) = v else { return Ok(Maybe::NotGiven) };
    Ok(v.extract::<MaybeNull<f64>>()?.into_core())
}

#[pymethods]
impl PyNativeRealtimeModel {
    #[new]
    #[pyo3(signature = (
        *,
        api_key,
        base_url,
        dialect = None,
        model = None,
        voice = None,
        modalities = None,
        azure_deployment = None,
        entra_token = None,
        azure_defaults = false,
        api_secret = None,
        inference_provider = None,
        inference_class = None,
        timeout = 10.0,
        max_retry = 3,
        retry_interval = 2.0,
        label = None,
        **options,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        api_key: Option<String>,
        base_url: String,
        dialect: Option<String>,
        model: Option<String>,
        voice: Option<String>,
        modalities: Option<Vec<String>>,
        azure_deployment: Option<String>,
        entra_token: Option<String>,
        azure_defaults: bool,
        api_secret: Option<String>,
        inference_provider: Option<String>,
        inference_class: Option<String>,
        timeout: f64,
        max_retry: u32,
        retry_interval: f64,
        label: Option<String>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        check_pid()?;
        let o = Options::read(options, &["max_session_duration"])?;
        let (input_audio_transcription, input_audio_noise_reduction, turn_detection) =
            (o.get("input_audio_transcription"), o.get("input_audio_noise_reduction"), o.get("turn_detection"));
        let (tool_choice, speed, tracing, truncation, reasoning, max_session_duration) = (
            o.get("tool_choice"),
            o.get("speed"),
            o.get("tracing"),
            o.get("truncation"),
            o.get("reasoning"),
            o.get("max_session_duration"),
        );
        let dialect: Dialect = match dialect {
            Some(name) => name.parse().map_err(|e: super::dialect::UnknownDialect| PyValueError::new_err(e.to_string()))?,
            None => Dialect::for_base_url(&base_url),
        };
        let is_inference = dialect.provider_label == INFERENCE.provider_label;
        let mut opts = SessionOptions::from_dialect(&dialect, false);
        if let Some(model) = model {
            opts.model = model;
        }
        if is_inference && opts.model.starts_with("xai/") {
            inference_xai_defaults(&mut opts);
        }
        if let Some(voice) = voice {
            opts.voice = voice;
        }
        if dialect.forced_modalities.is_none() {
            // `["text", "audio"]` when not given
            opts.audio_output = modalities.is_none_or(|m| m.iter().any(|m| m == "audio"));
        }
        if azure_defaults {
            opts.transcription = Some(azure_default_transcription());
            opts.turn_detection = Some(azure_default_turn_detection());
        }
        if let Maybe::Given(v) = maybe(option(input_audio_transcription.as_ref(), "input_audio_transcription")?) {
            opts.transcription = v;
        }
        if let Maybe::Given(v) = maybe(noise_reduction(input_audio_noise_reduction.as_ref())?) {
            opts.noise_reduction = v;
        }
        let td = turn_detection_arg(turn_detection.as_ref())?;
        let turn_detection_given = td.is_given();
        super::session::warn_on_half_disabled_turn_taking(&td);
        if let Maybe::Given(v) = td {
            opts.turn_detection = v;
        }
        if let Maybe::Given(v) = self::tool_choice(tool_choice.as_ref())? {
            opts.tool_choice = v;
        }
        if let Maybe::Given(Some(v)) = float(speed.as_ref())? {
            opts.speed = v;
        }
        for (slot, arg, path) in [
            (&mut opts.tracing, tracing.as_ref(), "tracing"),
            (&mut opts.truncation, truncation.as_ref(), "truncation"),
            (&mut opts.reasoning, reasoning.as_ref(), "reasoning"),
        ] {
            if let Maybe::Given(v) = maybe(option(arg, path)?) {
                *slot = v;
            }
        }
        if let Maybe::Given(v) = float(max_session_duration.as_ref())? {
            opts.max_session_duration = v.map(Duration::from_secs_f64);
        }

        let is_azure = azure_deployment.is_some() || entra_token.is_some();
        let auth = if is_inference {
            Auth::Inference {
                api_key: api_key.unwrap_or_default(),
                api_secret: api_secret.unwrap_or_default(),
                provider: inference_provider,
                inference_class,
                identity: Default::default(),
            }
        } else if is_azure {
            Auth::Azure { api_key, entra_token }
        } else {
            Auth::Bearer(api_key.unwrap_or_default())
        };
        let connect = Connect {
            base_url,
            azure: is_azure.then_some(Azure { deployment: azure_deployment }),
            auth,
            conn: ApiConnectOptions { max_retry, retry_interval, timeout },
        };
        let mut native = NativeRealtime::new(
            dialect,
            "",
            Arc::new(crate::native::TungsteniteTransport),
            <Arc<dyn lk_agents_abi::Spawner> as Ambient>::ambient(),
            // The wall clock, not a session's `Timebase`: a model outlives any one session and
            // `max_session_duration` is measured against real time.
            Arc::new(lk_agents_abi::RealClock::default()),
            // Empty: images reaching a realtime session are already serialised by the host, and
            // resolving again would encode the same frame twice.
            Arc::new(lk_agents_abi::llm::provider_format::PreResolved::from_pairs(Vec::<(
                String,
                lk_agents_abi::llm::provider_format::SerializedImage,
            )>::new())),
        )
        .with_options(opts, turn_detection_given)
        .with_connect(connect);
        if let Some(label) = label {
            native = native.with_label(label);
        }
        Ok(PyNativeRealtimeModel { inner: Arc::new(native) })
    }

    /// `RealtimeModel.update_options`: the model's options, then every live session's. A keyword
    /// left out is not given; `None` is a value.
    #[pyo3(signature = (**options))]
    fn update_options(&self, options: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        check_pid()?;
        let o = Options::read(options, &["voice", "max_response_output_tokens", "temperature"])?;
        let patch = OptionsPatch {
            tool_choice: self::tool_choice(o.get("tool_choice").as_ref())?,
            voice: match option(o.get("voice").as_ref(), "voice")? {
                MaybeNull::Given(Value::String(v)) => Maybe::Given(v),
                _ => Maybe::NotGiven,
            },
            turn_detection: turn_detection_arg(o.get("turn_detection").as_ref())?,
            max_output_tokens: match option(o.get("max_response_output_tokens").as_ref(), "max_response_output_tokens")? {
                MaybeNull::NotGiven => Maybe::NotGiven,
                MaybeNull::Null => Maybe::Given(Value::Null),
                MaybeNull::Given(v) => Maybe::Given(v),
            },
            transcription: maybe(option(o.get("input_audio_transcription").as_ref(), "input_audio_transcription")?),
            noise_reduction: maybe(noise_reduction(o.get("input_audio_noise_reduction").as_ref())?),
            speed: match float(o.get("speed").as_ref())? {
                Maybe::Given(Some(v)) => Maybe::Given(v),
                _ => Maybe::NotGiven,
            },
            tracing: maybe(option(o.get("tracing").as_ref(), "tracing")?),
            truncation: maybe(option(o.get("truncation").as_ref(), "truncation")?),
            reasoning: maybe(option(o.get("reasoning").as_ref(), "reasoning")?),
        };
        self.inner.update_options(patch);
        Ok(())
    }

    /// `get_inference_headers()` of the job the model now serves.
    fn bind_identification(&self, headers: Vec<(String, String)>) -> PyResult<()> {
        check_pid()?;
        self.inner.bind_identification(headers.into_iter().collect());
        Ok(())
    }
}

fn turn_detection_arg(v: Option<&Bound<'_, PyAny>>) -> PyResult<Maybe<Option<TurnDetection>>> {
    Ok(turn_detection(option(v, "turn_detection")?))
}

/// The option keywords, where leaving one out (not given) and passing `None` (off) differ.
struct Options<'py>(Option<Bound<'py, PyDict>>);

impl<'py> Options<'py> {
    const KNOWN: &'static [&'static str] = &[
        "input_audio_transcription",
        "input_audio_noise_reduction",
        "turn_detection",
        "tool_choice",
        "speed",
        "tracing",
        "truncation",
        "reasoning",
    ];

    fn read(d: Option<&Bound<'py, PyDict>>, more: &[&str]) -> PyResult<Self> {
        if let Some(d) = d {
            for key in d.keys() {
                let key: String = key.extract()?;
                if !Self::KNOWN.contains(&key.as_str()) && !more.contains(&key.as_str()) {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "unexpected keyword argument '{key}'"
                    )));
                }
            }
        }
        Ok(Options(d.cloned()))
    }

    fn get(&self, key: &str) -> Option<Bound<'py, PyAny>> {
        self.0.as_ref().and_then(|d| d.get_item(key).ok().flatten())
    }
}
