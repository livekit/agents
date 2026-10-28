//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `stt` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/stt/stt.rs.

/// [`::lk_agents_abi::Stt`] over a host `livekit.agents.stt.STT`.
#[derive(Clone)]
pub struct SttProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
    label: ::std::string::String,
    capabilities: ::lk_agents_abi::SttCapabilities,
    id: ::lk_agents_abi::ProviderId,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl SttProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Stt>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Stt>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Stt>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let capabilities = {
            let v = obj.call_method0("_core_capabilities")?;
            v.extract::<::lk_agents_abi::SttCapabilities>()?
        };
        let id = {
            let v = obj.call_method0("_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(SttProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?), label, capabilities, id, conn: None })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::Stt>(|obj| SttProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::Stt for SttProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn capabilities(&self) -> ::lk_agents_abi::SttCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "Stt._core_prewarm", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn update_keyterms(&self, keyterms: &[::std::string::String]) {
        let host = ::std::sync::Arc::clone(&self.host);
        let keyterms_arg = keyterms.to_vec();
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "Stt._core_update_keyterms", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(keyterms_arg, py)?];
            Ok((host.bind(py).getattr("_core_update_keyterms")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        let host = ::std::sync::Arc::clone(&self.host);
        let role_arg = role.to_owned();
        let text_arg = text.to_owned();
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "Stt._core_push_conversation_item", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(role_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
            Ok((host.bind(py).getattr("_core_push_conversation_item")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> { Ok((::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?).unbind()) }) {
            Ok(value) => value,
            Err(error) => {
                ::lk_agents_abi::python::log::warn!("Stt._core_failed failed: {error}");
                return;
            }
        };
        let recoverable_arg = recoverable;
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "Stt._core_failed", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [error_arg.into_bound(py), ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?];
            Ok((host.bind(py).getattr("_core_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Stt>> {
        Some(::lk_agents_runtime::stt::RetryingStt::bound(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }), conn, ::lk_agents_runtime::process::host_clock()))
    }

    async fn recognize(&self, buffer: &[::lk_agents_abi::AudioFrame], language: Option<&::lk_agents_abi::LanguageCode>) -> ::std::result::Result<::lk_agents_abi::SpeechEvent, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let buffer_arg = buffer.to_vec();
        let language_arg = language.map(::std::borrow::ToOwned::to_owned);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(buffer_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(language_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(py.import("livekit.agents.types")?.getattr("APIConnectOptions")?.call((), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_recognize")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::SpeechEvent>()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn stream(&self, language: Option<&::lk_agents_abi::LanguageCode>, start_time_offset: ::std::option::Option<f64>) -> ::std::result::Result<::std::option::Option<(::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SpeechEvent>>)>, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let language_arg = language.map(::std::borrow::ToOwned::to_owned);
        let start_time_offset_arg = start_time_offset;
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(language_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(start_time_offset_arg, py)?];
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
            let out = (if o.is_none() { ::pyo3::PyResult::Ok(None) } else { (::lk_agents_runtime::host::stream::HostStream::duplex(py, o, context, |py, v| Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(v, py)?), |o| Ok(o.extract::<::lk_agents_abi::SpeechEvent>()?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<::lk_agents_abi::AudioFrame>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SpeechEvent>>))).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
