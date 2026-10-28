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
type DataStreamAudioOutputImpl = <crate::Selected as super::Implementations>::DataStreamAudioOutput;

fn _extract_data_stream_audio_output_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")).map(Some) }
}

fn _extract_data_stream_audio_output_new_wait_remote_track(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<i32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i32>(&obj, "wait_remote_track")).map(Some) }
}

fn _extract_data_stream_audio_output_new_clear_buffer_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "clear_buffer_timeout")).map(Some) }
}

/// The agent's native audio sink, sending one byte stream per avatar speech segment.
#[pyclass(frozen, name = "DataStreamAudioOutput", module = "livekit.agents._core")]
pub struct DataStreamAudioOutput(::std::sync::Arc<DataStreamAudioOutputImpl>);

#[pymethods]
impl DataStreamAudioOutput {
    #[new]
    #[pyo3(signature = (room, *, destination_identity, sample_rate = None, wait_remote_track = None, clear_buffer_timeout = Some(2.0), wait_playback_start = false))]
    fn new(room: ::lk_agents_abi::HostRef, destination_identity: ::std::string::String, #[pyo3(from_py_with = _extract_data_stream_audio_output_new_sample_rate)] sample_rate: ::std::option::Option<u32>, #[pyo3(from_py_with = _extract_data_stream_audio_output_new_wait_remote_track)] wait_remote_track: ::std::option::Option<i32>, #[pyo3(from_py_with = _extract_data_stream_audio_output_new_clear_buffer_timeout)] clear_buffer_timeout: ::std::option::Option<f64>, wait_playback_start: bool) -> PyResult<DataStreamAudioOutput> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::new(room, destination_identity, sample_rate, wait_remote_track, clear_buffer_timeout, wait_playback_start).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(DataStreamAudioOutput(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn label(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::label(&self.0);
        Ok(out)
    }

    #[getter]
    fn sample_rate(&self) -> PyResult<::std::option::Option<u32>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::sample_rate(&self.0);
        Ok(out)
    }

    /// Stop admitting audio, settle startup and pending writes, close streams and retire RPC routes.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::AudioOutput`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::AudioOutput>(),
        read: |obj| {
            let this = obj.cast::<DataStreamAudioOutput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/console.rs.
type ConsoleIoImpl = <crate::Selected as super::Implementations>::ConsoleIo;

/// `lk agent console`'s socket, opened when the session starts.
#[pyclass(frozen, name = "ConsoleIo", module = "livekit.agents._core")]
pub struct ConsoleIo(::std::sync::Arc<ConsoleIoImpl>);

#[pymethods]
impl ConsoleIo {
    #[new]
    #[pyo3(signature = (addr))]
    fn new(addr: ::std::string::String) -> PyResult<ConsoleIo> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <ConsoleIoImpl as ::lk_agents_abi::voice::console::ConsoleIo>::new(addr);
        Ok(ConsoleIo(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::console::ConsolePort`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::console::ConsolePort>(),
        read: |obj| {
            let this = obj.cast::<ConsoleIo>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::console::ConsolePort> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/console.rs.
type VoiceSessionExecutionImpl = <crate::Selected as super::Implementations>::VoiceSessionExecution;

#[pyfunction]
#[pyo3(signature = (session, capture_run))]
pub fn _core_voice_session_start<'py>(py: Python<'py>, session: ::lk_agents_abi::HostRef, capture_run: bool) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <VoiceSessionExecutionImpl as ::lk_agents_abi::voice::console::VoiceSessionExecution>::start(session, capture_run).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.

/// [`::lk_agents_abi::voice::avatar::ClearBufferListener`] over a host `livekit.agents.voice.avatar.DataStreamAudioReceiver`.
#[derive(Clone)]
pub struct ClearBufferListenerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
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
        Ok(ClearBufferListenerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(|obj| ClearBufferListenerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::avatar::ClearBufferListener for ClearBufferListenerProxy {
    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "ClearBufferListener._core_clear_buffer", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_clear_buffer")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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
        Ok(AvatarAudioReceiverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(|obj| AvatarAudioReceiverProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::avatar::AvatarAudioReceiver for AvatarAudioReceiverProxy {
    fn open_audio(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_open_audio", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_open_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn next_audio(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(u64, bool, ::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_next_audio", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 3 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 3 items", "AvatarAudioReceiver._core_next_audio"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; Ok(((::lk_agents_abi::python::number_from_py::<u64>(&item_0, "AvatarAudioReceiver._core_next_audio"))?, (item_1.extract::<bool>())?, (if item_2.is_none() { Ok(None) } else { (item_2.extract::<::lk_agents_abi::utils::video::AvatarFrame>()).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn notify_playback_started(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_notify_playback_started", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_notify_playback_started")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn notify_playback_finished(&self, playback_position: f64, interrupted: bool) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_notify_playback_finished", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?];
                Ok((host.bind(py).getattr("_core_notify_playback_finished")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn aclose(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_aclose", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
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
        Ok(AvatarVideoGeneratorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(|obj| AvatarVideoGeneratorProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::avatar::AvatarVideoGenerator for AvatarVideoGeneratorProxy {
    fn push_audio(&self, frame: ::std::option::Option<::lk_agents_abi::utils::video::MediaFrame>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarVideoGenerator._core_push_audio", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn clear_buffer(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarVideoGenerator._core_clear_buffer", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_clear_buffer")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn open_frames(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarVideoGenerator._core_open_frames", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_open_frames")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn next_frame(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarVideoGenerator._core_next_frame", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_frame")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (if o.is_none() { Ok(None) } else { (o.extract::<::lk_agents_abi::utils::video::AvatarFrame>()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
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
        Ok(AvatarMediaProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(|obj| AvatarMediaProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::avatar::AvatarMedia for AvatarMediaProxy {
    fn publish(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarMedia._core_publish", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_publish")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn push(&self, frame: ::lk_agents_abi::utils::video::MediaFrame) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarMedia._core_push", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_push")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn aclose(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarMedia._core_aclose", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
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
    notifier: crate::host::Notifier,
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
        Ok(FrameStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(|obj| FrameStreamProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::background_audio::FrameStream for FrameStreamProxy {
    fn next(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::AudioFrame>, ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "FrameStream._core_next", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (if o.is_none() { Ok(None) } else { (o.extract::<::lk_agents_abi::AudioFrame>()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn cancel_pending(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "FrameStream._core_cancel_pending", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_cancel_pending")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn close(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "FrameStream._core_close", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("FrameStream._core_close failed: {e}");
        }
        })
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
    notifier: crate::host::Notifier,
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
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?, (obj,), None)?;
        let obj = &view;
        Ok(PlaybackListenerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(|obj| PlaybackListenerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::io::PlaybackListener for PlaybackListenerProxy {
    fn on_playback_started(&self, created_at: f64) {
        let host = ::std::sync::Arc::clone(&self.host);
        let created_at_arg = created_at;
        ::lk_agents_runtime::host::notify(&self.notifier, "PlaybackListener._core_on_playback_started", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(created_at_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_playback_started")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        let host = ::std::sync::Arc::clone(&self.host);
        let started_at_arg = started_at;
        let offset_arg = offset;
        let duration_arg = duration;
        ::lk_agents_runtime::host::notify(&self.notifier, "PlaybackListener._core_on_playback_progressed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(started_at_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(offset_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(duration_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_playback_progressed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_playback_finished(&self, playback_position: f64, interrupted: bool, synchronized_transcript: ::std::option::Option<::std::string::String>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let synchronized_transcript_arg = synchronized_transcript;
        ::lk_agents_runtime::host::notify(&self.notifier, "PlaybackListener._core_on_playback_finished", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(synchronized_transcript_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_playback_finished")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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
    notifier: crate::host::Notifier,
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
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_AudioInput")?, (obj,), None)?;
        let obj = &view;
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(AudioInputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default(), label })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::AudioInput>(|obj| AudioInputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::io::AudioInput for AudioInputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn next_frame(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::AudioFrame>, ::lk_agents_abi::voice::io::IoError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AudioInput._core_next_frame", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_frame")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (o.extract::<::lk_agents_abi::AudioFrame>()).map(Some) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioInput._core_on_attached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioInput._core_on_detached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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
    notifier: crate::host::Notifier,
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
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_VideoInput")?, (obj,), None)?;
        let obj = &view;
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(VideoInputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default(), label })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::VideoInput>(|obj| VideoInputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::io::VideoInput for VideoInputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn next_frame(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::voice::rtc::VideoFrameRef>, ::lk_agents_abi::voice::io::IoError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "VideoInput._core_next_frame", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next_frame")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (o.extract::<::lk_agents_abi::voice::rtc::VideoFrameRef>().map_err(::pyo3::PyErr::from)).map(Some) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "VideoInput._core_on_attached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "VideoInput._core_on_detached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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
    notifier: crate::host::Notifier,
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
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?, (obj,), None)?;
        let obj = &view;
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let tracker = {
            let v = crate::host::read(obj, "_core_tracker")?;
            <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&v)?
        };
        let capabilities = {
            let v = crate::host::read(obj, "_core_capabilities")?;
            v.extract::<::lk_agents_abi::voice::io::AudioOutputCapabilities>().map_err(::pyo3::PyErr::from)?
        };
        let sample_rate = {
            let v = crate::host::read(obj, "_core_sample_rate")?;
            if v.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&v, "AudioOutput._core_sample_rate")).map(Some) }?
        };
        Ok(AudioOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default(), label, tracker, capabilities, sample_rate })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::AudioOutput>(|obj| AudioOutputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

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

    fn capture<'a>(&'a self, frame: &::lk_agents_abi::AudioFrame) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AudioOutput._core_capture", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn flush_segment(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioOutput._core_flush_segment", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_flush_segment")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioOutput._core_clear_buffer", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_clear_buffer")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioOutput._core_on_attached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioOutput._core_on_detached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn pause(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioOutput._core_pause", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_pause")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn resume(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "AudioOutput._core_resume", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_resume")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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
    notifier: crate::host::Notifier,
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
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_TextOutput")?, (obj,), None)?;
        let obj = &view;
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(TextOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default(), label })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::TextOutput>(|obj| TextOutputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::io::TextOutput for TextOutputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn capture_timed<'a>(&'a self, text: &::lk_agents_abi::TimedString) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = ::std::clone::Clone::clone(text);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TextOutput._core_capture_timed", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture_timed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn capture_text<'a>(&'a self, text: &str) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TextOutput._core_capture_text", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture_text")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
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
        })
    }

    fn flush(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "TextOutput._core_flush", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_flush")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "TextOutput._core_on_attached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_attached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "TextOutput._core_on_detached", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_on_detached")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.

/// [`::lk_agents_abi::voice::session_effects::SessionHooks`] over a host `livekit.agents.voice.agent_session._AgentHooks`.
#[derive(Clone)]
pub struct SessionHooksProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl SessionHooksProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(SessionHooksProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(|obj| SessionHooksProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::session_effects::SessionHooks for SessionHooksProxy {
    fn on_enter<'a>(&'a self, agent: &::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_on_enter", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(agent_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_enter")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn on_exit<'a>(&'a self, agent: &::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_on_exit", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(agent_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_exit")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn close_toolsets<'a>(&'a self, agent: Option<&::lk_agents_abi::HostRef>) -> ::lk_agents_abi::BoxFuture<'a, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = agent.map(::std::borrow::ToOwned::to_owned);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_close_toolsets", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(agent_arg, py)?];
                Ok((host.bind(py).getattr("_core_close_toolsets")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        if let Err(e) = started.await {
            ::lk_agents_abi::python::log::warn!("SessionHooks._core_close_toolsets failed: {e}");
        }
        })
    }

    fn setup_tools<'a>(&'a self, tools: &::lk_agents_abi::HostRef) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::vec::Vec<::lk_agents_abi::ToolSpec>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let tools_arg = ::std::clone::Clone::clone(tools);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_setup_tools", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(tools_arg, py)?];
                Ok((host.bind(py).getattr("_core_setup_tools")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ToolSpec>() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn closing(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_closing", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_closing")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn on_user_turn_completed<'a>(&'a self, agent: &::lk_agents_abi::HostRef, chat_ctx: ::std::vec::Vec<::lk_agents_abi::ChatItem>, new_transcript: &str, transcript_confidence: f64) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(::std::option::Option<::std::string::String>, ::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>, bool), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let chat_ctx_arg = chat_ctx;
        let new_transcript_arg = new_transcript.to_owned();
        let transcript_confidence_arg = transcript_confidence;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_on_user_turn_completed", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [::pyo3::IntoPyObjectExt::into_bound_py_any(agent_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(chat_ctx_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(new_transcript_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript_confidence_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_user_turn_completed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 3 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 3 items", "SessionHooks._core_on_user_turn_completed"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; Ok(((if item_0.is_none() { Ok(None) } else { (item_0.extract::<::std::string::String>()).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::ChatItem>().map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())).map(Some) })?, (item_2.extract::<bool>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn on_user_turn_exceeded<'a>(&'a self, agent: &::lk_agents_abi::HostRef, ev: &::lk_agents_abi::voice::events::UserTurnExceededEvent) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let ev_arg = ::std::clone::Clone::clone(ev);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_on_user_turn_exceeded", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(agent_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(ev_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_user_turn_exceeded")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn function_tools_executed(&self, function_calls: ::std::vec::Vec<::lk_agents_abi::ChatItem>, function_call_outputs: ::std::vec::Vec<::lk_agents_abi::ChatItem>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::vec::Vec<bool>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let function_calls_arg = function_calls;
        let function_call_outputs_arg = function_call_outputs;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_function_tools_executed", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(function_calls_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(function_call_outputs_arg, py)?];
                Ok((host.bind(py).getattr("_core_function_tools_executed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<bool>() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn filler_source<'a>(&'a self, source: &::lk_agents_abi::HostRef, step: u32) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::lk_agents_abi::voice::session_effects::FillerReply, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let source_arg = ::std::clone::Clone::clone(source);
        let step_arg = step;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "SessionHooks._core_filler_source", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(source_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(step_arg, py)?];
                Ok((host.bind(py).getattr("_core_filler_source")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::voice::session_effects::FillerReply>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn hook_failed(&self, hook: &str, error: ::std::option::Option<::lk_agents_abi::errors::CapturedError>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let hook_arg = hook.to_owned();
        let error_arg = error;
        ::lk_agents_runtime::host::notify(&self.notifier, "SessionHooks._core_hook_failed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(hook_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?];
                Ok((host.bind(py).getattr("_core_hook_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.

/// [`::lk_agents_abi::voice::session_effects::SessionEvents`] over a host `livekit.agents.voice.agent_session.VoiceSession`.
#[derive(Clone)]
pub struct SessionEventsProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl SessionEventsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(SessionEventsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>(|obj| SessionEventsProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::session_effects::SessionEvents for SessionEventsProxy {
    fn event(&self, wire: ::lk_agents_abi::voice::events::HostWire) {
        let host = ::std::sync::Arc::clone(&self.host);
        let wire_arg = wire;
        ::lk_agents_runtime::host::notify(&self.notifier, "SessionEvents._core_event", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(wire_arg, py)?];
                Ok((host.bind(py).getattr("_core_event")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn error_message(&self, error: &::lk_agents_abi::voice::events::SessionErrorWire) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = ::std::clone::Clone::clone(error);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?];
                Ok((host.bind(py).getattr("_core_error_message")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
