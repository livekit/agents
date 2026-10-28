//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `voice` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.

/// [`::lk_agents_abi::voice::avatar::ClearBufferListener`] over a host `livekit.agents.voice.avatar.DataStreamAudioReceiver`.
#[derive(Clone)]
pub struct ClearBufferListenerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
}

impl ClearBufferListenerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ClearBufferListenerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(|obj| ClearBufferListenerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::avatar::ClearBufferListener for ClearBufferListenerProxy {
    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "ClearBufferListener._core_clear_buffer", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_clear_buffer")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.

/// [`::lk_agents_abi::voice::avatar::AvatarAudioReceiver`] over a host `livekit.agents.voice.avatar.AudioReceiver`.
#[derive(Clone)]
pub struct AvatarAudioReceiverProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AvatarAudioReceiverProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(AvatarAudioReceiverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(|obj| AvatarAudioReceiverProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::avatar::AvatarAudioReceiver for AvatarAudioReceiverProxy {
    async fn open_audio(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_open_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn next_audio(&self) -> ::std::result::Result<::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>>()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn notify_playback_started(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_notify_playback_started")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn notify_playback_finished(&self, playback_position: f64, interrupted: bool) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?];
                Ok((host.bind(py).getattr("_core_notify_playback_finished")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn aclose(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.

/// [`::lk_agents_abi::voice::avatar::AvatarVideoGenerator`] over a host `livekit.agents.voice.avatar.VideoGenerator`.
#[derive(Clone)]
pub struct AvatarVideoGeneratorProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AvatarVideoGeneratorProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(AvatarVideoGeneratorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(|obj| AvatarVideoGeneratorProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::avatar::AvatarVideoGenerator for AvatarVideoGeneratorProxy {
    async fn push_audio(&self, frame: ::std::option::Option<::lk_agents_abi::utils::video::MediaFrame>) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn clear_buffer(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_clear_buffer")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn open_frames(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_open_frames")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn next_frame(&self) -> ::std::result::Result<::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_frame")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>>()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.

/// [`::lk_agents_abi::voice::avatar::AvatarMedia`] over a host `livekit.agents.voice.avatar._AvatarMedia`.
#[derive(Clone)]
pub struct AvatarMediaProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AvatarMediaProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(AvatarMediaProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(|obj| AvatarMediaProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::avatar::AvatarMedia for AvatarMediaProxy {
    async fn publish(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_publish")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn push(&self, frame: ::lk_agents_abi::utils::video::MediaFrame) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn aclose(&self) -> ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.

/// [`::lk_agents_abi::voice::background_audio::FrameStream`] over a host `livekit.agents.voice.background_audio._FrameStream`.
#[derive(Clone)]
pub struct FrameStreamProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
}

impl FrameStreamProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(FrameStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(|obj| FrameStreamProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::background_audio::FrameStream for FrameStreamProxy {
    async fn next(&self) -> ::std::result::Result<::std::option::Option<::lk_agents_abi::AudioFrame>, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::option::Option<::lk_agents_abi::AudioFrame>>()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn cancel_pending(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "FrameStream._core_cancel_pending", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_cancel_pending")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    async fn close(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        if let Err(e) = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await {
            ::lk_agents_abi::python::log::warn!("FrameStream._core_close failed: {e}");
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.

/// [`::lk_agents_abi::voice::io::PlaybackListener`] over a host `livekit.agents.voice.io.AudioOutput`.
#[derive(Clone)]
pub struct PlaybackListenerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
}

impl PlaybackListenerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?.call1((obj,))?;
        let obj = &view;
        Ok(PlaybackListenerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(|obj| PlaybackListenerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::io::PlaybackListener for PlaybackListenerProxy {
    fn on_playback_started(&self, created_at: f64) {
        let host = ::std::sync::Arc::clone(&self.host);
        let created_at_arg = created_at;
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "PlaybackListener._core_on_playback_started", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(created_at_arg, py)?];
            Ok((host.bind(py).getattr("_core_on_playback_started")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        let host = ::std::sync::Arc::clone(&self.host);
        let started_at_arg = started_at;
        let offset_arg = offset;
        let duration_arg = duration;
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "PlaybackListener._core_on_playback_progressed", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(started_at_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(offset_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(duration_arg, py)?];
            Ok((host.bind(py).getattr("_core_on_playback_progressed")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_playback_finished(&self, playback_position: f64, interrupted: bool, synchronized_transcript: ::std::option::Option<::std::string::String>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let synchronized_transcript_arg = synchronized_transcript;
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "PlaybackListener._core_on_playback_finished", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(synchronized_transcript_arg, py)?];
            Ok((host.bind(py).getattr("_core_on_playback_finished")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.

/// [`::lk_agents_abi::voice::io::AudioInput`] over a host `livekit.agents.voice.io.AudioInput`.
#[derive(Clone)]
pub struct AudioInputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
    label: ::std::string::String,
}

impl AudioInputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = obj.py().import("livekit.agents._core_view")?.getattr("_AudioInput")?.call1((obj,))?;
        let obj = &view;
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(AudioInputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?), label })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::AudioInput>(|obj| AudioInputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::io::AudioInput for AudioInputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    async fn next_frame(&self) -> ::std::result::Result<::std::option::Option<::lk_agents_abi::AudioFrame>, ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_frame")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::option::Option<::lk_agents_abi::AudioFrame>>()).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioInput._core_on_attached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioInput._core_on_detached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.

/// [`::lk_agents_abi::voice::io::VideoInput`] over a host `livekit.agents.voice.io.VideoInput`.
#[derive(Clone)]
pub struct VideoInputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
    label: ::std::string::String,
}

impl VideoInputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = obj.py().import("livekit.agents._core_view")?.getattr("_VideoInput")?.call1((obj,))?;
        let obj = &view;
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(VideoInputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?), label })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::VideoInput>(|obj| VideoInputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::io::VideoInput for VideoInputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    async fn next_frame(&self) -> ::std::result::Result<::std::option::Option<::lk_agents_abi::voice::rtc::VideoFrameRef>, ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_frame")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::option::Option<::lk_agents_abi::voice::rtc::VideoFrameRef>>().map_err(::pyo3::PyErr::from)).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "VideoInput._core_on_attached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "VideoInput._core_on_detached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.

/// [`::lk_agents_abi::voice::io::AudioOutput`] over a host `livekit.agents.voice.io.AudioOutput`.
#[derive(Clone)]
pub struct AudioOutputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
    label: ::std::string::String,
    tracker: ::std::sync::Arc<::lk_agents_abi::voice::io::PlaybackTracker>,
    capabilities: ::lk_agents_abi::voice::io::AudioOutputCapabilities,
    sample_rate: ::std::option::Option<u32>,
}

impl AudioOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?.call1((obj,))?;
        let obj = &view;
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let tracker = {
            let v = obj.call_method0("_core_tracker")?;
            <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&v)?
        };
        let capabilities = {
            let v = obj.call_method0("_core_capabilities")?;
            v.extract::<::lk_agents_abi::voice::io::AudioOutputCapabilities>().map_err(::pyo3::PyErr::from)?
        };
        let sample_rate = {
            let v = obj.call_method0("_core_sample_rate")?;
            v.extract::<::std::option::Option<u32>>()?
        };
        Ok(AudioOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?), label, tracker, capabilities, sample_rate })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::AudioOutput>(|obj| AudioOutputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::io::AudioOutput for AudioOutputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn tracker(&self) -> &::std::sync::Arc<::lk_agents_abi::voice::io::PlaybackTracker> {
        &self.tracker
    }

    fn capabilities(&self) -> ::lk_agents_abi::voice::io::AudioOutputCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn sample_rate(&self) -> ::std::option::Option<u32> {
        ::std::clone::Clone::clone(&self.sample_rate)
    }

    async fn capture(&self, frame: &::lk_agents_abi::AudioFrame) -> ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn flush_segment(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioOutput._core_flush_segment", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_flush_segment")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioOutput._core_clear_buffer", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_clear_buffer")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioOutput._core_on_attached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioOutput._core_on_detached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn pause(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioOutput._core_pause", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_pause")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn resume(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "AudioOutput._core_resume", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_resume")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.

/// [`::lk_agents_abi::voice::io::TextOutput`] over a host `livekit.agents.voice.io.TextOutput`.
#[derive(Clone)]
pub struct TextOutputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
    label: ::std::string::String,
}

impl TextOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = obj.py().import("livekit.agents._core_view")?.getattr("_TextOutput")?.call1((obj,))?;
        let obj = &view;
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(TextOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?), label })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::TextOutput>(|obj| TextOutputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::voice::io::TextOutput for TextOutputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    async fn capture_timed(&self, text: &::lk_agents_abi::TimedString) -> ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = ::std::clone::Clone::clone(text);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture_timed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn capture_text(&self, text: &str) -> ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture_text")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn flush(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "TextOutput._core_flush", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_flush")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "TextOutput._core_on_attached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "TextOutput._core_on_detached", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}
