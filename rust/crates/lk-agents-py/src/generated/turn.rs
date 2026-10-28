//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `turn` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/turn/detector.rs.

/// [`::lk_agents_abi::EotDetector`] over a host `livekit.agents._core_view.TurnDetector`.
#[derive(Clone)]
pub struct EotDetectorProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    id: ::lk_agents_abi::ProviderId,
}

impl EotDetectorProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::EotDetector>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::EotDetector>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::EotDetector>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(EotDetectorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default(), id })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::EotDetector>(|obj| EotDetectorProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::EotDetector for EotDetectorProxy {
    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn push_audio(&self, frame: ::lk_agents_abi::AudioFrame) {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        ::lk_agents_runtime::host::notify(&self.notifier, "EotDetector._core_push_audio", &self.context, &[], true, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn cancel(&self, timed_out: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let timed_out_arg = timed_out;
        ::lk_agents_runtime::host::notify(&self.notifier, "EotDetector._core_cancel", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(timed_out_arg, py)?];
                Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn flush(&self, reason: &str) {
        let host = ::std::sync::Arc::clone(&self.host);
        let reason_arg = reason.to_owned();
        ::lk_agents_runtime::host::notify(&self.notifier, "EotDetector._core_flush", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(reason_arg, py)?];
                Ok((host.bind(py).getattr("_core_flush")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn predict(&self, timeout: ::std::option::Option<::std::time::Duration>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::EotPrediction, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let timeout_arg = timeout;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "EotDetector._core_predict", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(timeout_arg.map(|d| d.as_secs_f64()), py)?];
                Ok((host.bind(py).getattr("_core_predict")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::turn_events::eot_prediction_from_py(&o)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn aclose(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "EotDetector._core_aclose", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("EotDetector._core_aclose failed: {e}");
        }
        })
    }
}
