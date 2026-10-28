//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `tts` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/tts/tts.rs.

/// [`::lk_agents_abi::Tts`] over a host `livekit.agents.tts.TTS`.
#[derive(Clone)]
pub struct TtsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::TtsCapabilities,
    id: ::lk_agents_abi::ProviderId,
    sample_rate: u32,
    num_channels: u16,
    markup_provider: ::std::string::String,
    lowers_markup: bool,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl TtsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Tts>(obj) {
            return Ok(native);
        }
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = obj.call_method0("_core_capabilities")?;
            v.extract::<::lk_agents_abi::TtsCapabilities>()?
        };
        let id = {
            let v = obj.call_method0("_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        let sample_rate = {
            let v = obj.call_method0("_core_sample_rate")?;
            v.extract::<u32>()?
        };
        let num_channels = {
            let v = obj.call_method0("_core_num_channels")?;
            v.extract::<u16>()?
        };
        let markup_provider = {
            let v = obj.call_method0("_core_markup_provider")?;
            v.extract::<::std::string::String>()?
        };
        let lowers_markup = {
            let v = obj.call_method0("_core_lowers_markup")?;
            v.extract::<bool>()?
        };
        Ok(::std::sync::Arc::new(TtsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, label, capabilities, id, sample_rate, num_channels, markup_provider, lowers_markup, conn: None }))
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::Tts>(|obj| TtsProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::Tts for TtsProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::TtsCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let runtime = match ::lk_agents_runtime::process::handle() {
            Ok(runtime) => runtime,
            Err(e) => {
                ::lk_agents_abi::python::log::warn!("Tts._core_prewarm failed: {e}");
                return;
            }
        };
        runtime.spawn(async move {
            let context = &context;
            if let Err(e) = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await {
                ::lk_agents_abi::python::log::warn!("Tts._core_prewarm failed: {e}");
            }
        });
    }

    fn sample_rate(&self) -> u32 {
        ::std::clone::Clone::clone(&self.sample_rate)
    }

    fn num_channels(&self) -> u16 {
        ::std::clone::Clone::clone(&self.num_channels)
    }

    fn markup_provider(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.markup_provider)
    }

    fn lowers_markup(&self) -> bool {
        ::std::clone::Clone::clone(&self.lowers_markup)
    }

    fn set_expressive(&self, enabled: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let enabled_arg = enabled;
        let context = ::std::clone::Clone::clone(&self.context);
        let runtime = match ::lk_agents_runtime::process::handle() {
            Ok(runtime) => runtime,
            Err(e) => {
                ::lk_agents_abi::python::log::warn!("Tts._core_set_expressive failed: {e}");
                return;
            }
        };
        runtime.spawn(async move {
            let context = &context;
            if let Err(e) = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(enabled_arg, py)?];
                Ok((host.bind(py).getattr("_core_set_expressive")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await {
                ::lk_agents_abi::python::log::warn!("Tts._core_set_expressive failed: {e}");
            }
        });
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Tts>> {
        Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let recoverable_arg = recoverable;
        let context = ::std::clone::Clone::clone(&self.context);
        let runtime = match ::lk_agents_runtime::process::handle() {
            Ok(runtime) => runtime,
            Err(e) => {
                ::lk_agents_abi::python::log::warn!("Tts._core_failed failed: {e}");
                return;
            }
        };
        runtime.spawn(async move {
            let context = &context;
            if let Err(e) = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))(error_arg).into_value(py).into_bound(py).into_any(), ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?];
                Ok((host.bind(py).getattr("_core_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await {
                ::lk_agents_abi::python::log::warn!("Tts._core_failed failed: {e}");
            }
        });
    }

    async fn synthesize(&self, text: &str) -> ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(py.import("livekit.agents.types")?.getattr("APIConnectOptions")?.call((), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_synthesize")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_runtime::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::SynthesizedAudio>()?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn stream(&self) -> ::std::result::Result<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::std::string::String>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(py.import("livekit.agents.types")?.getattr("APIConnectOptions")?.call((), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_stream")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_runtime::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(v, py)?), |o| Ok(o.extract::<::lk_agents_abi::SynthesizedAudio>()?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::std::string::String>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>))).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
