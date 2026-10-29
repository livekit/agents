//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `voice` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/voice/amd.rs.
type AmdFunctionsImpl = <crate::Selected as super::Implementations>::AmdFunctions;

/// These are the reference's, not tuning knobs: the default models are what the benchmark
/// ran on.
#[pyfunction]
pub fn amd_constants<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let out = <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_constants();
    Ok(crate::generated::records::amd::amd_constants_to_py(py, out)?)
}

/// `_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's,
/// else the default. `provided` is `None` only when the key was absent: an explicit default
/// still beats a disagreeing activity.
#[pyfunction]
#[pyo3(signature = (provided, activity))]
pub fn amd_resolve_max_endpointing_delay<'py>(py: Python<'py>, provided: Option<Bound<'_, PyAny>>, activity: Option<Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let provided = provided.map(|v| ::lk_agents_abi::python::number_from_py::<f64>(&v, "provided").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "provided"))))).transpose()?;
    let activity = activity.map(|v| ::lk_agents_abi::python::number_from_py::<f64>(&v, "activity").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "activity"))))).transpose()?;
    let out = <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_resolve_max_endpointing_delay(provided, activity);
    Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(out.as_secs_f64(), py)?)
}

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
        let env = crate::process::env();
        let out = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::new(&env, room, destination_identity, sample_rate, wait_remote_track, clear_buffer_timeout, wait_playback_start).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
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

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type DataStreamAudioReceiverImpl = <crate::Selected as super::Implementations>::DataStreamAudioReceiver;

fn _extract_data_stream_audio_receiver_new_frame_size_ms(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "frame_size_ms")
}

fn _extract_data_stream_audio_receiver_new_rpc_max_retries(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "rpc_max_retries")
}

fn _extract_data_stream_audio_receiver_notify_playback_finished_playback_position(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "playback_position")
}

/// The worker's receiver keeps stream identity with each delivered frame and playback report.
#[pyclass(frozen, name = "DataStreamAudioReceiver", module = "livekit.agents._core")]
pub struct DataStreamAudioReceiver(::std::sync::Arc<DataStreamAudioReceiverImpl>);

#[pymethods]
impl DataStreamAudioReceiver {
    #[new]
    #[pyo3(signature = (room, *, sender_identity = None, frame_size_ms = 100, rpc_max_retries = 3, on_clear_buffer = None))]
    fn new(room: Bound<'_, PyAny>, sender_identity: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_data_stream_audio_receiver_new_frame_size_ms)] frame_size_ms: u32, #[pyo3(from_py_with = _extract_data_stream_audio_receiver_new_rpc_max_retries)] rpc_max_retries: u32, on_clear_buffer: Option<Bound<'_, PyAny>>) -> PyResult<DataStreamAudioReceiver> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let room = crate::generated::voice::RtcRoomProxy::from_host(&room)?;
        let on_clear_buffer = on_clear_buffer.map(|v| crate::generated::voice::ClearBufferListenerProxy::from_host(&v)).transpose()?;
        let out = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::new(&env, room, sender_identity, frame_size_ms, rpc_max_retries, on_clear_buffer).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(DataStreamAudioReceiver(::std::sync::Arc::new(out)))
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn next_audio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::next_audio(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any({ let (x0, x1, x2, x3,) = out; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?, match x2 { None => py.None().into_bound(py), Some(x) => crate::generated::records::io_events::avatar_playback_segment_to_py(py, x)? }, match x3 { None => py.None().into_bound(py), Some(x) => crate::generated::records::avatar::audio_receiver_frame_to_py(py, x)? }])?.into_any() }, py))
        })
    }

    #[pyo3(signature = (segment))]
    fn notify_playback_started(&self, segment: Bound<'_, PyAny>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let segment = crate::generated::records::io_events::avatar_playback_segment_from_py(&segment)?;
        <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::notify_playback_started(&self.0, segment).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (segment, playback_position, interrupted))]
    fn notify_playback_finished(&self, segment: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_data_stream_audio_receiver_notify_playback_finished_playback_position)] playback_position: f64, interrupted: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let segment = crate::generated::records::io_events::avatar_playback_segment_from_py(&segment)?;
        <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::notify_playback_finished(&self.0, segment, playback_position, interrupted).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn aclose(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::aclose(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::avatar::AvatarAudioReceiver`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(),
        read: |obj| {
            let this = obj.cast::<DataStreamAudioReceiver>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type AvatarRunnerImpl = <crate::Selected as super::Implementations>::AvatarRunner;

fn _extract_avatar_runner_clear_buffer_epoch(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "epoch")
}

/// The avatar worker's native queue owner. Its decisions remain in RunnerCore.
#[pyclass(frozen, name = "AvatarRunner", module = "livekit.agents._core")]
pub struct AvatarRunner(::std::sync::Arc<AvatarRunnerImpl>);

#[pymethods]
impl AvatarRunner {
    #[new]
    #[pyo3(signature = (room, *, receiver, generator, media, lazy_publish))]
    fn new(room: ::lk_agents_abi::voice::room_owner::RoomSession, receiver: Bound<'_, PyAny>, generator: Bound<'_, PyAny>, media: Bound<'_, PyAny>, lazy_publish: bool) -> PyResult<AvatarRunner> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let receiver = crate::generated::voice::AvatarAudioReceiverProxy::from_host(&receiver)?;
        let generator = crate::generated::voice::AvatarVideoGeneratorProxy::from_host(&generator)?;
        let media = crate::generated::voice::AvatarMediaProxy::from_host(&media)?;
        let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::new(&env, room, receiver, generator, media, lazy_publish).map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AvatarRunner(::std::sync::Arc::new(out)))
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn wait_for_complete<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::wait_for_complete(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (epoch))]
    fn clear_buffer(&self, #[pyo3(from_py_with = _extract_avatar_runner_clear_buffer_epoch)] epoch: u64) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::clear_buffer(&self.0, epoch);
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
type PlayStateImpl = <crate::Selected as super::Implementations>::PlayState;

fn _extract_play_state_new_fade_out(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "fade_out")
}

/// A sound's monotone stop and completion handle.
#[pyclass(frozen, name = "PlayState", module = "livekit.agents._core")]
pub struct PlayState(::std::sync::Arc<PlayStateImpl>);

#[pymethods]
impl PlayState {
    #[new]
    #[pyo3(signature = (fade_out = 0.0))]
    fn new(#[pyo3(from_py_with = _extract_play_state_new_fade_out)] fade_out: f64) -> PyResult<PlayState> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::new(fade_out);
        Ok(PlayState(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn done(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::done(&self.0);
        Ok(out)
    }

    #[getter]
    fn stop_requested(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::stop_requested(&self.0);
        Ok(out)
    }

    fn stop(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::stop(&self.0);
        Ok(out)
    }

    #[pyo3(name = "mark_playout_done")]
    fn finish(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::finish(&self.0);
        Ok(out)
    }

    #[pyo3(name = "wait_for_playout")]
    fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::wait(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
type BackgroundAudioPlayerImpl = <crate::Selected as super::Implementations>::BackgroundAudioPlayer;

fn _extract_background_audio_player_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

fn _extract_background_audio_player_new_num_channels(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    ::lk_agents_abi::python::number_from_py::<u16>(&obj, "num_channels")
}

fn _extract_background_audio_player_new_blocksize(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "blocksize")
}

fn _extract_background_audio_player_new_stream_timeout_ms(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "stream_timeout_ms")
}

fn _extract_background_audio_player_start_draw(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "draw")
}

fn _extract_background_audio_player_play_draw(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "draw")
}

fn _extract_background_audio_player_on_agent_state_draw(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "draw")
}

/// Runtime-owned mixer and published background track.
#[pyclass(frozen, name = "BackgroundAudioPlayer", module = "livekit.agents._core")]
pub struct BackgroundAudioPlayer(::std::sync::Arc<BackgroundAudioPlayerImpl>);

#[pymethods]
impl BackgroundAudioPlayer {
    #[new]
    #[pyo3(signature = (sample_rate = 48000, num_channels = 1, *, blocksize = 4800, stream_timeout_ms = 200))]
    fn new(#[pyo3(from_py_with = _extract_background_audio_player_new_sample_rate)] sample_rate: u32, #[pyo3(from_py_with = _extract_background_audio_player_new_num_channels)] num_channels: u16, #[pyo3(from_py_with = _extract_background_audio_player_new_blocksize)] blocksize: usize, #[pyo3(from_py_with = _extract_background_audio_player_new_stream_timeout_ms)] stream_timeout_ms: u64) -> PyResult<BackgroundAudioPlayer> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::new(&env, sample_rate, num_channels, blocksize, stream_timeout_ms).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(BackgroundAudioPlayer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (room, draw, *, ambient, choose, source = None))]
    fn start<'py>(&self, py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, #[pyo3(from_py_with = _extract_background_audio_player_start_draw)] draw: f64, ambient: Bound<'_, PyAny>, choose: bool, source: ::std::option::Option<::lk_agents_abi::voice::room_io::TrackSource>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let ambient = ambient.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::background_audio_wire::sound_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::start(&this, room, draw, ambient, choose, source)
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (sounds, draw, *, looping = false, ambient = false, choose = true))]
    fn play(&self, sounds: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_background_audio_player_play_draw)] draw: f64, looping: bool, ambient: bool, choose: bool) -> PyResult<super::voice::PlayState> {
        ::lk_agents_abi::python::check_pid()?;
        let sounds = sounds.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::background_audio_wire::sound_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::play(&self.0, sounds, draw, looping, ambient, choose).map_err(|e: ::lk_agents_abi::voice::background_audio::PlaybackError| <::lk_agents_abi::voice::background_audio::PlaybackError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(super::voice::PlayState(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (sounds, draw, thinking, *, choose = true))]
    fn on_agent_state(&self, sounds: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_background_audio_player_on_agent_state_draw)] draw: f64, thinking: bool, choose: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let sounds = sounds.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::background_audio_wire::sound_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::on_agent_state(&self.0, sounds, draw, thinking, choose).map_err(|e: ::lk_agents_abi::voice::background_audio::PlaybackError| <::lk_agents_abi::voice::background_audio::PlaybackError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::closed(&self.0);
        Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
type PlaybackReportsImpl = <crate::Selected as super::Implementations>::PlaybackReports;

fn _extract_playback_reports_started_created_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "created_at")
}

fn _extract_playback_reports_progressed_started_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "started_at")
}

fn _extract_playback_reports_progressed_offset(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "offset")
}

fn _extract_playback_reports_progressed_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "duration")
}

fn _extract_playback_reports_finished_playback_position(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "playback_position")
}

#[pyfunction]
#[pyo3(signature = (tracker, segment))]
pub fn _core_playback_begin(tracker: Bound<'_, PyAny>, segment: Bound<'_, PyAny>) -> PyResult<bool> {
    ::lk_agents_abi::python::check_pid()?;
    let tracker = <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&tracker)?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::begin(tracker, segment);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker))]
pub fn _core_playback_current_segment<'py>(py: Python<'py>, tracker: Bound<'_, PyAny>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    ::lk_agents_abi::python::check_pid()?;
    let tracker = <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&tracker)?;
    let out = <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::current_segment(tracker);
    Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::io_events::playback_segment_to_py(py, value)?)).transpose()?)
}

#[pyfunction]
#[pyo3(signature = (tracker))]
pub fn _core_playback_note_capture<'py>(py: Python<'py>, tracker: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let tracker = <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&tracker)?;
    let out = <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::note_capture(tracker);
    Ok(crate::generated::records::io_events::playback_segment_to_py(py, out)?)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, created_at))]
pub fn _core_playback_started(tracker: Bound<'_, PyAny>, segment: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_playback_reports_started_created_at)] created_at: f64) -> PyResult<bool> {
    ::lk_agents_abi::python::check_pid()?;
    let tracker = <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&tracker)?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::started(tracker, segment, created_at);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, started_at, offset, duration))]
pub fn _core_playback_progressed(tracker: Bound<'_, PyAny>, segment: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_playback_reports_progressed_started_at)] started_at: f64, #[pyo3(from_py_with = _extract_playback_reports_progressed_offset)] offset: f64, #[pyo3(from_py_with = _extract_playback_reports_progressed_duration)] duration: f64) -> PyResult<bool> {
    ::lk_agents_abi::python::check_pid()?;
    let tracker = <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&tracker)?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::progressed(tracker, segment, started_at, offset, duration).map_err(|e: ::lk_agents_abi::voice::io::InvalidPlaybackTime| <::lk_agents_abi::voice::io::InvalidPlaybackTime as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, playback_position, interrupted, synchronized_transcript))]
pub fn _core_playback_finished(tracker: Bound<'_, PyAny>, segment: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_playback_reports_finished_playback_position)] playback_position: f64, interrupted: bool, synchronized_transcript: ::std::option::Option<::std::string::String>) -> PyResult<bool> {
    ::lk_agents_abi::python::check_pid()?;
    let tracker = <::lk_agents_abi::voice::io::PlaybackTracker as ::lk_agents_abi::python::ArcClass>::shared(&tracker)?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::finished(tracker, segment, playback_position, interrupted, synchronized_transcript).map_err(|e: ::lk_agents_abi::voice::io::InvalidPlaybackTime| <::lk_agents_abi::voice::io::InvalidPlaybackTime as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/events.rs.
type RtcRuntimeImpl = <crate::Selected as super::Implementations>::RtcRuntime;

#[pyfunction]
pub fn _core_create_room_event_bus() -> PyResult<::lk_agents_abi::voice::rtc::RoomEventBus> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::create_room_event_bus();
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_subscribed(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>, publication: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_subscribed(bus, participant, publication);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_unsubscribed(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>, publication: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_unsubscribed(bus, participant, publication);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_published(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>, publication: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_published(bus, participant, publication);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_unpublished(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>, publication: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_unpublished(bus, participant, publication);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, publication))]
pub fn _core_room_local_track_published(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, publication: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::local_track_published(bus, publication);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant))]
pub fn _core_room_participant_connected(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_connected(bus, participant);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant))]
pub fn _core_room_participant_active(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_active(bus, participant);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant))]
pub fn _core_room_participant_attributes_changed(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_attributes_changed(bus, participant);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, reason))]
pub fn _core_room_participant_disconnected(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, participant: Bound<'_, PyAny>, reason: ::lk_agents_abi::voice::room_io::DisconnectReason) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_disconnected(bus, participant, reason);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, state))]
pub fn _core_room_connection_state_changed(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, state: ::lk_agents_abi::voice::rtc::ConnectionState) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::connection_state_changed(bus, state);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, reason))]
pub fn _core_room_disconnected(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, reason: ::lk_agents_abi::voice::room_io::DisconnectReason) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::disconnected(bus, reason);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, digit, participant))]
pub fn _core_room_sip_dtmf_received(bus: ::lk_agents_abi::voice::rtc::RoomEventBus, digit: ::std::string::String, participant: ::std::option::Option<::lk_agents_abi::voice::room_io::ParticipantIdentity>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::sip_dtmf_received(bus, digit, participant);
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/playout.rs.
type PlayoutLedgerImpl = <crate::Selected as super::Implementations>::PlayoutLedger;

fn _extract_playout_ledger_add_pushed_samples_samples_per_channel(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "samples_per_channel")
}

fn _extract_playout_ledger_add_pushed_samples_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

#[pyclass(frozen, name = "PlayoutLedger", module = "livekit.agents._core")]
pub struct PlayoutLedger(::std::sync::Arc<PlayoutLedgerImpl>);

#[pymethods]
impl PlayoutLedger {
    #[new]
    fn new() -> PyResult<PlayoutLedger> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::new();
        Ok(PlayoutLedger(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (samples_per_channel, sample_rate))]
    fn add_pushed_samples(&self, #[pyo3(from_py_with = _extract_playout_ledger_add_pushed_samples_samples_per_channel)] samples_per_channel: u32, #[pyo3(from_py_with = _extract_playout_ledger_add_pushed_samples_sample_rate)] sample_rate: u32) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::add_pushed_samples(&self.0, samples_per_channel, sample_rate);
        Ok(())
    }

    #[pyo3(signature = (duration))]
    fn add_discarded(&self, duration: Bound<'_, PyAny>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let duration = ::lk_agents_abi::python::number_from_py::<f64>(&duration, "duration").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "duration"))))?;
        <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::add_discarded(&self.0, duration);
        Ok(())
    }

    #[pyo3(signature = (queued))]
    fn snapshot(&self, queued: Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::voice::rtc::PlaybackInfo> {
        ::lk_agents_abi::python::check_pid()?;
        let queued = ::lk_agents_abi::python::number_from_py::<f64>(&queued, "queued").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "queued"))))?;
        let out = <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::snapshot(&self.0, queued);
        Ok(out)
    }

    fn settle(&self) -> PyResult<::lk_agents_abi::voice::rtc::PlaybackInfo> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::settle(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
type SessionStartImpl = <crate::Selected as super::Implementations>::SessionStart;

/// The job's recording and whether this session is its primary (records) and hosts the
/// remote session protocol; `another_session_is_primary` is the job's claim already taken.
#[pyfunction]
#[pyo3(signature = (*, job_default, another_session_is_primary))]
pub fn plan_session_start<'py>(py: Python<'py>, job_default: bool, another_session_is_primary: bool) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <SessionStartImpl as ::lk_agents_abi::voice::session_effects::SessionStart>::plan_session_start(job_default, another_session_is_primary);
    Ok(crate::generated::records::session_wire::start_plan_to_py(py, out)?)
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows.rs.
type WorkflowChecksImpl = <crate::Selected as super::Implementations>::WorkflowChecks;

fn _extract_workflow_checks_card_expired_month(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "month")
}

fn _extract_workflow_checks_card_expired_year(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "year")
}

fn _extract_workflow_checks_normalize_date_of_birth_year(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "year")
}

fn _extract_workflow_checks_normalize_date_of_birth_month(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "month")
}

fn _extract_workflow_checks_normalize_date_of_birth_day(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "day")
}

/// The Luhn checksum over an all-digit string.
#[pyfunction]
#[pyo3(signature = (card_number))]
pub fn validate_card_number(card_number: ::std::string::String) -> PyResult<bool> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::validate_card_number(card_number);
    Ok(out)
}

/// Whether a two-digit expiry year and its month precede `today`'s month.
#[pyfunction]
#[pyo3(signature = (month, year, today))]
pub fn card_expired(#[pyo3(from_py_with = _extract_workflow_checks_card_expired_month)] month: i64, #[pyo3(from_py_with = _extract_workflow_checks_card_expired_year)] year: i64, today: ::lk_agents_abi::utils::dates::Date) -> PyResult<bool> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::card_expired(month, year, today);
    Ok(out)
}

/// A two-digit year is this century's when it is not past this year's two digits, else the
/// last century's; an invalid or future date is refused.
#[pyfunction]
#[pyo3(signature = (year, month, day, today))]
pub fn normalize_date_of_birth(#[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_year)] year: i64, #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_month)] month: i64, #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_day)] day: i64, today: ::lk_agents_abi::utils::dates::Date) -> PyResult<::lk_agents_abi::utils::dates::Date> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_date_of_birth(year, month, day, today).map_err(|e: ::lk_agents_abi::voice::workflows::DateOfBirthRefused| <::lk_agents_abi::voice::workflows::DateOfBirthRefused as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// Stripped, and refused unless it reads `local@label.….tld`.
#[pyfunction]
#[pyo3(signature = (email))]
pub fn normalize_email_address(email: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_email_address(email).map_err(|e: ::lk_agents_abi::voice::workflows::CaptureRefused| <::lk_agents_abi::voice::workflows::CaptureRefused as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// Whitespace, hyphens, parentheses and dots dropped, and refused unless it reads `+?[1-9]`
/// followed by six to fourteen digits.
#[pyfunction]
#[pyo3(signature = (phone_number))]
pub fn normalize_phone_number(phone_number: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_phone_number(phone_number).map_err(|e: ::lk_agents_abi::voice::workflows::CaptureRefused| <::lk_agents_abi::voice::workflows::CaptureRefused as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// Digits are themselves, `*` is 10, `#` is 11 and `A`-`D` are 12-15.
#[pyfunction]
#[pyo3(signature = (symbol))]
pub fn dtmf_code(symbol: ::std::string::String) -> PyResult<u8> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::dtmf_code(symbol).map_err(|e: ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol| <::lk_agents_abi::voice::workflows::UnknownDtmfSymbol as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// The space-joined digits that fill the DTMF read-back's `{digits}` hole.
#[pyfunction]
#[pyo3(signature = (symbols))]
pub fn format_dtmf(symbols: ::std::vec::Vec<::std::string::String>) -> PyResult<::std::string::String> {
    let out = <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::format_dtmf(symbols).map_err(|e: ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol| <::lk_agents_abi::voice::workflows::UnknownDtmfSymbol as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/dtmf.rs.
type GetDtmfResultImpl = <crate::Selected as super::Implementations>::DtmfResult;

/// A collected, space-separated keypad sequence.
#[pyclass(frozen, name = "GetDtmfResult", module = "livekit.agents._core")]
pub struct GetDtmfResult(::std::sync::Arc<GetDtmfResultImpl>);

#[pymethods]
impl GetDtmfResult {
    #[new]
    #[pyo3(signature = (*, user_input))]
    fn new(user_input: ::std::string::String) -> PyResult<GetDtmfResult> {
        let out = <GetDtmfResultImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfResult>::new(user_input);
        Ok(GetDtmfResult(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn user_input(&self) -> PyResult<::std::string::String> {
        let out = <GetDtmfResultImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfResult>::user_input(&self.0);
        Ok(out)
    }

    #[pyo3(name = "__repr__")]
    fn host_repr(&self) -> PyResult<::std::string::String> {
        let out = <GetDtmfResultImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfResult>::host_repr(&self.0);
        Ok(out)
    }

    #[pyo3(name = "__eq__")]
    fn host_eq(&self, other: PyRef<'_, GetDtmfResult>) -> PyResult<bool> {
        let out = <GetDtmfResultImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfResult>::host_eq(&self.0, &*other.0);
        Ok(out)
    }

    #[staticmethod]
    #[pyo3(signature = (events))]
    fn from_dtmf_inputs(events: ::std::vec::Vec<::std::string::String>) -> PyResult<GetDtmfResult> {
        let out = <GetDtmfResultImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfResult>::from_dtmf_inputs(events).map_err(|e: ::lk_agents_abi::voice::workflows::dtmf::DtmfInputError| <::lk_agents_abi::voice::workflows::dtmf::DtmfInputError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(GetDtmfResult(::std::sync::Arc::new(out)))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/dtmf.rs.
type DtmfIoImpl = <crate::Selected as super::Implementations>::DtmfIo;

fn _extract_dtmf_io_receive_dtmf_num_digits(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "num_digits")
}

fn _extract_dtmf_io_receive_dtmf_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "timeout")
}

#[pyfunction]
#[pyo3(signature = (room, events))]
pub fn send_dtmf_events<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, events: ::std::vec::Vec<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <DtmfIoImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfIo>::send_dtmf_events(room, events).await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, num_digits, timeout, stop_event))]
pub fn receive_dtmf<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, #[pyo3(from_py_with = _extract_dtmf_io_receive_dtmf_num_digits)] num_digits: usize, #[pyo3(from_py_with = _extract_dtmf_io_receive_dtmf_timeout)] timeout: f64, stop_event: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <DtmfIoImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfIo>::receive_dtmf(room, num_digits, timeout, stop_event).await.map_err(|e: ::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError| <::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::voice::GetDtmfResult(::std::sync::Arc::new(out)), py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/warm_transfer.rs.
type WarmTransferImpl = <crate::Selected as super::Implementations>::WarmTransfer;

fn _extract_warm_transfer_new_ringing_timeout(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "ringing_timeout")).map(Some) }
}

/// A transfer plan and the runtime ownership of its current dial.
#[pyclass(frozen, name = "WarmTransfer", module = "livekit.agents._core")]
pub struct WarmTransfer(::std::sync::Arc<WarmTransferImpl>);

#[pymethods]
impl WarmTransfer {
    #[new]
    #[pyo3(signature = (*, sip_call_to = None, sip_trunk_id = ::lk_agents_abi::Maybe::NotGiven, has_sip_connection = false, sip_number = ::lk_agents_abi::Maybe::NotGiven, sip_headers = None, dtmf = None, ringing_timeout = None, instructions = ::lk_agents_abi::Maybe::NotGiven, conversation = None, env_sip_trunk = None, env_sip_number = "".to_owned(), twilio = None))]
    fn new(sip_call_to: ::std::option::Option<::std::string::String>, sip_trunk_id: ::lk_agents_abi::Maybe<::std::option::Option<::std::string::String>>, has_sip_connection: bool, sip_number: ::lk_agents_abi::Maybe<::std::string::String>, sip_headers: ::std::option::Option<::std::collections::BTreeMap<::std::string::String, ::std::string::String>>, dtmf: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_warm_transfer_new_ringing_timeout)] ringing_timeout: ::std::option::Option<f64>, instructions: ::lk_agents_abi::Maybe<Bound<'_, PyAny>>, conversation: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>, env_sip_trunk: ::std::option::Option<::std::string::String>, env_sip_number: ::std::string::String, twilio: ::std::option::Option<::lk_agents_abi::voice::workflows::TwilioDial>) -> PyResult<WarmTransfer> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let instructions = match instructions { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((crate::generated::records::workflows::instructions_arg_from_py(&value))?) };
        let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::new(&env, sip_call_to, sip_trunk_id, has_sip_connection, sip_number, sip_headers, dtmf, ringing_timeout, instructions, conversation, env_sip_trunk, env_sip_number, twilio).map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused| <::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(WarmTransfer(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn instructions(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::instructions(&self.0);
        Ok(out)
    }

    fn tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::tools(&self.0);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok({ let d = ::pyo3::types::PyDict::new(py); for (k, x) in x { d.set_item(::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?, ::lk_agents_abi::python::value_to_py(py, &x)?)?; } d.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    #[pyo3(signature = (host, caller))]
    fn run<'py>(&self, py: Python<'py>, host: Bound<'_, PyAny>, caller: ::lk_agents_abi::voice::room_owner::RoomSession) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let host = crate::generated::voice::TransferHostProxy::from_host(&host)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::run(&this, host, caller)
                .await.map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed| <::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (name, arguments))]
    fn call_tool<'py>(&self, py: Python<'py>, name: ::std::string::String, arguments: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let arguments = ::lk_agents_abi::python::value_from_py(&arguments, "arguments")?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::call_tool(&this, name, arguments)
                .await.map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed| <::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
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
        let env = crate::process::env();
        let out = <ConsoleIoImpl as ::lk_agents_abi::voice::console::ConsoleIo>::new(&env, addr);
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
#[pyo3(signature = (session))]
pub fn _core_voice_session_start<'py>(py: Python<'py>, session: ::lk_agents_abi::HostRef) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <VoiceSessionExecutionImpl as ::lk_agents_abi::voice::console::VoiceSessionExecution>::start(session).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomRuntimeImpl = <crate::Selected as super::Implementations>::RoomRuntime;

fn _extract_room_runtime_wait_for_participant_kinds(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<::std::vec::Vec<i32>>> {
    if obj.is_none() { Ok(None) } else { (obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<i32>(&item, "kinds") }).collect::<PyResult<Vec<_>>>())).map(Some) }
}

fn _extract_room_runtime_wait_for_track_publication_kinds(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<::std::vec::Vec<i32>>> {
    if obj.is_none() { Ok(None) } else { (obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<i32>(&item, "kinds") }).collect::<PyResult<Vec<_>>>())).map(Some) }
}

#[pyfunction]
#[pyo3(signature = (room))]
pub fn _core_create_room_session(room: Option<Bound<'_, PyAny>>) -> PyResult<::lk_agents_abi::voice::room_owner::RoomSession> {
    ::lk_agents_abi::python::check_pid()?;
    let env = crate::process::env();
    let room = room.map(|v| crate::generated::voice::RtcRoomProxy::from_host(&v)).transpose()?;
    let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::create_room_session(&env, room);
    Ok(out)
}

/// Wait for the supplied room's connection, avatar participant and subscribed video track.
#[pyfunction]
#[pyo3(signature = (room, identity))]
pub fn wait_avatar_join<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, identity: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_avatar_join(room, identity).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, identity, kinds, include_local))]
pub fn wait_for_participant<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, identity: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_room_runtime_wait_for_participant_kinds)] kinds: ::std::option::Option<::std::vec::Vec<i32>>, include_local: bool) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_participant(room, identity, kinds, include_local).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, name))]
pub fn wait_for_agent<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, name: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_agent(room, name).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, identity, attribute, value))]
pub fn wait_for_participant_attribute<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, identity: ::std::string::String, attribute: ::std::string::String, value: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_participant_attribute(room, identity, attribute, value).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, identity, kinds, include_local, subscribed))]
pub fn wait_for_track_publication<'py>(py: Python<'py>, room: ::lk_agents_abi::voice::room_owner::RoomSession, identity: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_room_runtime_wait_for_track_publication_kinds)] kinds: ::std::option::Option<::std::vec::Vec<i32>>, include_local: bool, subscribed: bool) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_track_publication(room, identity, kinds, include_local, subscribed).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.

/// [`::lk_agents_abi::voice::avatar::ClearBufferListener`] over a host `livekit.agents.voice.avatar.AudioReceiver`.
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(ClearBufferListenerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AvatarAudioReceiverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
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

    fn next_audio(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(u64, bool, ::std::option::Option<::lk_agents_abi::voice::io::AvatarPlaybackSegment>, ::std::option::Option<::lk_agents_abi::utils::video::AudioReceiverFrame>), ::lk_agents_abi::errors::ProviderError>> {
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "AvatarAudioReceiver._core_next_audio"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((::lk_agents_abi::python::number_from_py::<u64>(&item_0, "AvatarAudioReceiver._core_next_audio"))?, (item_1.extract::<bool>())?, (if item_2.is_none() { Ok(None) } else { (crate::generated::records::io_events::avatar_playback_segment_from_py(&item_2)).map(Some) })?, (if item_3.is_none() { Ok(None) } else { (crate::generated::records::avatar::audio_receiver_frame_from_py(&item_3)).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn notify_playback_started(&self, segment: ::lk_agents_abi::voice::io::AvatarPlaybackSegment) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_notify_playback_started", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::io_events::avatar_playback_segment_to_py(py, segment_arg)?];
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

    fn notify_playback_finished(&self, segment: ::lk_agents_abi::voice::io::AvatarPlaybackSegment, playback_position: f64, interrupted: bool) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AvatarAudioReceiver._core_notify_playback_finished", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [crate::generated::records::io_events::avatar_playback_segment_to_py(py, segment_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?];
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AvatarVideoGeneratorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AvatarMediaProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(FrameStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?, (obj,), None)?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(PlaybackListenerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(|obj| PlaybackListenerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::io::PlaybackListener for PlaybackListenerProxy {
    fn on_playback_started(&self, segment: ::lk_agents_abi::voice::io::PlaybackSegment, created_at: f64) {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let created_at_arg = created_at;
        ::lk_agents_runtime::host::notify(&self.notifier, "PlaybackListener._core_on_playback_started", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_playback_started")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_playback_progressed(&self, segment: ::lk_agents_abi::voice::io::PlaybackSegment, started_at: f64, offset: f64, duration: f64) {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let started_at_arg = started_at;
        let offset_arg = offset;
        let duration_arg = duration;
        ::lk_agents_runtime::host::notify(&self.notifier, "PlaybackListener._core_on_playback_progressed", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(started_at_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(offset_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(duration_arg, py)?];
                Ok((host.bind(py).getattr("_core_on_playback_progressed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn on_playback_finished(&self, segment: ::lk_agents_abi::voice::io::PlaybackSegment, playback_position: f64, interrupted: bool, synchronized_transcript: ::std::option::Option<::std::string::String>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let synchronized_transcript_arg = synchronized_transcript;
        ::lk_agents_runtime::host::notify(&self.notifier, "PlaybackListener._core_on_playback_finished", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(synchronized_transcript_arg, py)?];
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_AudioInput")?, (obj,), None)?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(AudioInputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), label })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_VideoInput")?, (obj,), None)?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(VideoInputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), label })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?, (obj,), None)?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
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
        Ok(AudioOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), label, tracker, capabilities, sample_rate })
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

    fn begin_segment(&self, segment: ::lk_agents_abi::voice::io::PlaybackSegment) -> ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?];
                Ok((host.bind(py).getattr("_core_begin_segment")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let view = crate::host::construct(&obj.py().import("livekit.agents._core_view")?.getattr("_TextOutput")?, (obj,), None)?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(TextOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), label })
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

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::RtcRoom`] over a host `livekit.agents.voice.room_io.Room`.
#[derive(Clone)]
pub struct RtcRoomProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    local: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>,
    event_bus: ::lk_agents_abi::voice::rtc::RoomEventBus,
}

impl RtcRoomProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        let local = {
            let v = crate::host::read(obj, "_core_local")?;
            crate::generated::voice::LocalParticipantProxy::from_host(&v)?
        };
        let event_bus = {
            let v = crate::host::read(obj, "_core_event_bus")?;
            v.extract::<::lk_agents_abi::voice::rtc::RoomEventBus>().map_err(::pyo3::PyErr::from)?
        };
        Ok(RtcRoomProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), local, event_bus })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(|obj| RtcRoomProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::RtcRoom for RtcRoomProxy {
    fn name(&self) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_name")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn connection_state(&self) -> ::std::result::Result<::lk_agents_abi::voice::rtc::ConnectionState, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_connection_state")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::voice::rtc::ConnectionState>()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn local(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant> {
        &self.local
    }

    fn participant(&self, id: &::lk_agents_abi::voice::room_io::ParticipantIdentity) -> ::std::result::Result<::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let id_arg = ::std::clone::Clone::clone(id);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(id_arg, py)?];
                Ok((host.bind(py).getattr("_core_participant")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() { Ok(None) } else { (crate::generated::voice::RemoteParticipantProxy::from_host(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn remote_participants(&self) -> ::std::result::Result<::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_remote_participants")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::voice::RemoteParticipantProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn event_bus(&self) -> &::lk_agents_abi::voice::rtc::RoomEventBus {
        &self.event_bus
    }

    fn register_byte_stream_handler(&self, topic: &str, streams: ::lk_agents_abi::voice::rtc::ByteStreamSender) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let streams_arg = streams;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(streams_arg, py)?];
                Ok((host.bind(py).getattr("_core_register_byte_stream_handler")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn unregister_byte_stream_handler(&self, topic: &str) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?];
                Ok((host.bind(py).getattr("_core_unregister_byte_stream_handler")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn register_text_stream_handler(&self, topic: &str, streams: ::lk_agents_abi::voice::rtc::TextStreamSender) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let streams_arg = streams;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(streams_arg, py)?];
                Ok((host.bind(py).getattr("_core_register_text_stream_handler")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn unregister_text_stream_handler(&self, topic: &str) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?];
                Ok((host.bind(py).getattr("_core_unregister_text_stream_handler")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::LocalParticipant`] over a host `livekit.agents.voice.room_io.LocalParticipant`.
#[derive(Clone)]
pub struct LocalParticipantProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl LocalParticipantProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(LocalParticipantProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(|obj| LocalParticipantProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::LocalParticipant for LocalParticipantProxy {
    fn identity(&self) -> ::std::result::Result<::lk_agents_abi::voice::room_io::ParticipantIdentity, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_identity")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::voice::room_io::ParticipantIdentity>()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn kind(&self) -> ::std::result::Result<::lk_agents_abi::voice::room_io::ParticipantKind, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_kind")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::voice::room_io::ParticipantKind>()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn track_publications(&self) -> ::std::result::Result<::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_track_publications")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::voice::TrackPublicationProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn publish_audio_track<'a>(&'a self, opts: ::lk_agents_abi::voice::room_io::TrackPublishOptions) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>, ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_publish_audio_track", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(opts_arg, py)?];
                Ok((host.bind(py).getattr("_core_publish_audio_track")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "LocalParticipant._core_publish_audio_track"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::generated::voice::AudioSourceProxy::from_host(&item_0))?, (crate::generated::voice::TrackPublicationProxy::from_host(&item_1))?,)) })()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn unpublish_track<'a>(&'a self, sid: &::lk_agents_abi::voice::rtc::TrackSid) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let sid_arg = ::std::clone::Clone::clone(sid);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_unpublish_track", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(sid_arg, py)?];
                Ok((host.bind(py).getattr("_core_unpublish_track")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn set_attributes<'a>(&'a self, attrs: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let attrs_arg = attrs;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_set_attributes", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(attrs_arg, py)?];
                Ok((host.bind(py).getattr("_core_set_attributes")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn publish_transcription<'a>(&'a self, transcription: ::lk_agents_abi::voice::rtc::Transcription) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let transcription_arg = transcription;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_publish_transcription", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(transcription_arg, py)?];
                Ok((host.bind(py).getattr("_core_publish_transcription")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn stream_text<'a>(&'a self, opts: ::lk_agents_abi::voice::rtc::StreamTextOptions) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>, ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_stream_text", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(opts_arg, py)?];
                Ok((host.bind(py).getattr("_core_stream_text")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::voice::TextStreamWriterProxy::owned(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn stream_bytes<'a>(&'a self, opts: ::lk_agents_abi::voice::rtc::StreamBytesOptions) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>, ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_stream_bytes", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(opts_arg, py)?];
                Ok((host.bind(py).getattr("_core_stream_bytes")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::voice::ByteStreamWriterProxy::owned(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn perform_rpc<'a>(&'a self, destination: ::lk_agents_abi::voice::room_io::ParticipantIdentity, method: ::std::string::String, payload: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(u32, ::std::string::String), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let destination_arg = destination;
        let method_arg = method;
        let payload_arg = payload;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_perform_rpc", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(destination_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(payload_arg, py)?];
                Ok((host.bind(py).getattr("_core_perform_rpc")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "LocalParticipant._core_perform_rpc"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((::lk_agents_abi::python::number_from_py::<u32>(&item_0, "LocalParticipant._core_perform_rpc"))?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn register_rpc_method(&self, method: ::std::string::String, handler: ::lk_agents_abi::voice::rtc::RpcHandler) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let method_arg = method;
        let handler_arg = handler;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(handler_arg, py)?];
                Ok((host.bind(py).getattr("_core_register_rpc_method")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn unregister_rpc_method(&self, method: ::std::string::String) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let method_arg = method;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?];
                Ok((host.bind(py).getattr("_core_unregister_rpc_method")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn publish_dtmf<'a>(&'a self, code: u32, digit: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let code_arg = code;
        let digit_arg = digit;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "LocalParticipant._core_publish_dtmf", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(code_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(digit_arg, py)?];
                Ok((host.bind(py).getattr("_core_publish_dtmf")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::ByteStreamWriter`] over a host `livekit.agents.voice.room_io.ByteStreamWriter`.
#[derive(Clone)]
pub struct ByteStreamWriterProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    id: ::std::string::String,
}

impl ByteStreamWriterProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let id = {
            let v = crate::host::read(obj, "_core_id")?;
            v.extract::<::std::string::String>()?
        };
        Ok(ByteStreamWriterProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, id })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(|obj| ByteStreamWriterProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::ByteStreamWriter for ByteStreamWriterProxy {
    fn id(&self) -> &str {
        &self.id
    }

    fn write<'a>(&'a self, data: &[u8]) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let data_arg = data.to_vec();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ByteStreamWriter._core_write", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(data_arg, py)?];
                Ok((host.bind(py).getattr("_core_write")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn close(self: ::std::boxed::Box<Self>) -> ::lk_agents_abi::BoxFuture<'static, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "ByteStreamWriter._core_close", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::TextStreamWriter`] over a host `livekit.agents.voice.room_io.TextStreamWriter`.
#[derive(Clone)]
pub struct TextStreamWriterProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl TextStreamWriterProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(TextStreamWriterProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(|obj| TextStreamWriterProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::TextStreamWriter for TextStreamWriterProxy {
    fn write<'a>(&'a self, text: &str) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TextStreamWriter._core_write", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_write")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn close(self: ::std::boxed::Box<Self>, attributes: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>) -> ::lk_agents_abi::BoxFuture<'static, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let attributes_arg = attributes;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TextStreamWriter._core_close", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(attributes_arg, py)?];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::AudioSource`] over a host `livekit.agents.voice.room_io.AudioSource`.
#[derive(Clone)]
pub struct AudioSourceProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AudioSourceProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AudioSourceProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(|obj| AudioSourceProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::AudioSource for AudioSourceProxy {
    fn capture<'a>(&'a self, frame: &::lk_agents_abi::AudioFrame) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AudioSource._core_capture", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?];
                Ok((host.bind(py).getattr("_core_capture")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn clear_queue(&self) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_clear_queue")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn queued_duration(&self) -> ::std::result::Result<::std::time::Duration, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_queued_duration")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_abi::python::number_from_py::<f64>(&o, "AudioSource._core_queued_duration").and_then(|s| ::std::time::Duration::try_from_secs_f64(s).map_err(|e| ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "AudioSource._core_queued_duration"))))).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn wait_for_playout<'a>(&'a self) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::lk_agents_abi::voice::rtc::PlaybackInfo, ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AudioSource._core_wait_for_playout", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_wait_for_playout")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::voice::rtc::PlaybackInfo>().map_err(::pyo3::PyErr::from)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn playback_info(&self) -> ::std::result::Result<::lk_agents_abi::voice::rtc::PlaybackInfo, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_playback_info")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::voice::rtc::PlaybackInfo>().map_err(::pyo3::PyErr::from)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn close<'a>(&'a self) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AudioSource._core_close", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::RemoteParticipant`] over a host `livekit.agents.voice.room_io.RemoteParticipant`.
#[derive(Clone)]
pub struct RemoteParticipantProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    sid: ::std::string::String,
    identity: ::lk_agents_abi::voice::room_io::ParticipantIdentity,
    kind: ::lk_agents_abi::voice::room_io::ParticipantKind,
}

impl RemoteParticipantProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        let sid = {
            let v = crate::host::read(obj, "_core_sid")?;
            v.extract::<::std::string::String>()?
        };
        let identity = {
            let v = crate::host::read(obj, "_core_identity")?;
            v.extract::<::lk_agents_abi::voice::room_io::ParticipantIdentity>()?
        };
        let kind = {
            let v = crate::host::read(obj, "_core_kind")?;
            v.extract::<::lk_agents_abi::voice::room_io::ParticipantKind>()?
        };
        Ok(RemoteParticipantProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), sid, identity, kind })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(|obj| RemoteParticipantProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::RemoteParticipant for RemoteParticipantProxy {
    fn sid(&self) -> &str {
        &self.sid
    }

    fn identity(&self) -> &::lk_agents_abi::voice::room_io::ParticipantIdentity {
        &self.identity
    }

    fn kind(&self) -> ::lk_agents_abi::voice::room_io::ParticipantKind {
        ::std::clone::Clone::clone(&self.kind)
    }

    fn active(&self) -> ::std::result::Result<bool, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_active")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn attributes(&self) -> ::std::result::Result<::std::collections::BTreeMap<::std::string::String, ::std::string::String>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_attributes")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?))).collect::<PyResult<::std::collections::BTreeMap<_, _>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn track_publications(&self) -> ::std::result::Result<::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_track_publications")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::voice::TrackPublicationProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::TrackPublication`] over a host `livekit.agents.voice.room_io.TrackPublication`.
#[derive(Clone)]
pub struct TrackPublicationProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    sid: ::lk_agents_abi::voice::rtc::TrackSid,
    name: ::std::string::String,
    source: ::lk_agents_abi::voice::room_io::TrackSource,
    kind: i32,
    audio_features: ::std::vec::Vec<::lk_agents_abi::voice::rtc::AudioTrackFeature>,
}

impl TrackPublicationProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TrackPublication>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let sid = {
            let v = crate::host::read(obj, "_core_sid")?;
            v.extract::<::lk_agents_abi::voice::rtc::TrackSid>()?
        };
        let name = {
            let v = crate::host::read(obj, "_core_name")?;
            v.extract::<::std::string::String>()?
        };
        let source = {
            let v = crate::host::read(obj, "_core_source")?;
            v.extract::<::lk_agents_abi::voice::room_io::TrackSource>()?
        };
        let kind = {
            let v = crate::host::read(obj, "_core_kind")?;
            ::lk_agents_abi::python::number_from_py::<i32>(&v, "TrackPublication._core_kind")?
        };
        let audio_features = {
            let v = crate::host::read(obj, "_core_audio_features")?;
            v.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::voice::rtc::AudioTrackFeature>() }).collect::<PyResult<Vec<_>>>())?
        };
        Ok(TrackPublicationProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, sid, name, source, kind, audio_features })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(|obj| TrackPublicationProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::TrackPublication for TrackPublicationProxy {
    fn sid(&self) -> &::lk_agents_abi::voice::rtc::TrackSid {
        &self.sid
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn source(&self) -> ::lk_agents_abi::voice::room_io::TrackSource {
        ::std::clone::Clone::clone(&self.source)
    }

    fn kind(&self) -> i32 {
        ::std::clone::Clone::clone(&self.kind)
    }

    fn is_subscribed(&self) -> ::std::result::Result<bool, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_is_subscribed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn audio_features(&self) -> ::std::vec::Vec<::lk_agents_abi::voice::rtc::AudioTrackFeature> {
        ::std::clone::Clone::clone(&self.audio_features)
    }

    fn set_subscribed<'a>(&'a self, subscribed: bool) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let subscribed_arg = subscribed;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TrackPublication._core_set_subscribed", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(subscribed_arg, py)?];
                Ok((host.bind(py).getattr("_core_set_subscribed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn wait_for_subscription<'a>(&'a self) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TrackPublication._core_wait_for_subscription", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_wait_for_subscription")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn open_audio_stream(&self, opts: ::lk_agents_abi::voice::rtc::AudioStreamOptions) -> ::std::result::Result<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(opts_arg, py)?];
                Ok((host.bind(py).getattr("_core_open_audio_stream")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::voice::AudioInputStreamProxy::from_host(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn open_video_stream(&self) -> ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::voice::rtc::VideoFrameRef>>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_open_video_stream")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("TrackPublication._core_open_video_stream: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::voice::rtc::VideoFrameRef>().map_err(::pyo3::PyErr::from)?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::voice::rtc::VideoFrameRef>>)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::AudioInputStream`] over a host `livekit.agents.voice.room_io._AudioStream`.
#[derive(Clone)]
pub struct AudioInputStreamProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AudioInputStreamProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AudioInputStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(|obj| AudioInputStreamProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::rtc::AudioInputStream for AudioInputStreamProxy {
    fn frames(&self) -> ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_frames")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AudioInputStream._core_frames: the call captured no host context")));
                };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::AudioFrame>()?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn process_audio<'a>(&'a self, frames: ::std::vec::Vec<::lk_agents_abi::AudioFrame>) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<::std::vec::Vec<::lk_agents_abi::AudioFrame>, ::lk_agents_abi::voice::rtc::RtcError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frames_arg = frames;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "AudioInputStream._core_process_audio", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frames_arg, py)?];
                Ok((host.bind(py).getattr("_core_process_audio")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::std::boxed::Box::pin(async move {
        let out = started.await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::AudioFrame>() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(SessionHooksProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
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
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(agent_arg, py)?, crate::generated::records::events::user_turn_exceeded_event_to_py(py, ev_arg)?];
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
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(SessionEventsProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default() })
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
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::session_wire::host_wire_to_py(py, wire_arg)?];
                Ok((host.bind(py).getattr("_core_event")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn error_message(&self, error: &::lk_agents_abi::voice::events::SessionErrorWire) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = ::std::clone::Clone::clone(error);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::session_wire::session_error_wire_to_py(py, error_arg)?];
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

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/warm_transfer.rs.

/// [`::lk_agents_abi::voice::workflows::warm_transfer::TransferHost`] over a host `livekit.agents.beta.workflows._TransferHost`.
#[derive(Clone)]
pub struct TransferHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
    io_state: ::lk_agents_abi::voice::workflows::IoEnableState,
}

impl TransferHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let io_state = {
            let v = crate::host::read(obj, "_core_io_state")?;
            v.extract::<::lk_agents_abi::voice::workflows::IoEnableState>()?
        };
        Ok(TransferHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context, notifier: ::std::default::Default::default(), io_state })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>(|obj| TransferHostProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost for TransferHostProxy {
    fn start_hold(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::option::Option<::lk_agents_abi::HostRef>, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_start_hold", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_start_hold")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (if o.is_none() { Ok(None) } else { (o.extract::<::lk_agents_abi::HostRef>()).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn stop_hold(&self, hold: ::lk_agents_abi::HostRef) {
        let host = ::std::sync::Arc::clone(&self.host);
        let hold_arg = hold;
        ::lk_agents_runtime::host::notify(&self.notifier, "TransferHost._core_stop_hold", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(hold_arg, py)?];
                Ok((host.bind(py).getattr("_core_stop_hold")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn io_state(&self) -> ::lk_agents_abi::voice::workflows::IoEnableState {
        ::std::clone::Clone::clone(&self.io_state)
    }

    fn set_io(&self, state: ::lk_agents_abi::voice::workflows::IoEnableState) {
        let host = ::std::sync::Arc::clone(&self.host);
        let state_arg = state;
        ::lk_agents_runtime::host::notify(&self.notifier, "TransferHost._core_set_io", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(state_arg, py)?];
                Ok((host.bind(py).getattr("_core_set_io")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn connect_human(&self, room_name: ::std::string::String, identity: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::voice::room_owner::RoomSession, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let room_name_arg = room_name;
        let identity_arg = identity;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_connect_human", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(room_name_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?];
                Ok((host.bind(py).getattr("_core_connect_human")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<::lk_agents_abi::voice::room_owner::RoomSession>().map_err(::pyo3::PyErr::from)).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn start_human(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_start_human", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_start_human")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn close_human(&self, drain: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let drain_arg = drain;
        ::lk_agents_runtime::host::notify(&self.notifier, "TransferHost._core_close_human", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(drain_arg, py)?];
                Ok((host.bind(py).getattr("_core_close_human")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn disconnect_human(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_disconnect_human", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_disconnect_human")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn create_sip_participant(&self, request: ::lk_agents_abi::voice::workflows::SipParticipantRequest) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_create_sip_participant", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(request_arg, py)?];
                Ok((host.bind(py).getattr("_core_create_sip_participant")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn connect_twilio_call(&self, room_name: ::std::string::String, identity: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let room_name_arg = room_name;
        let identity_arg = identity;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_connect_twilio_call", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(room_name_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?];
                Ok((host.bind(py).getattr("_core_connect_twilio_call")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn http_post(&self, url: ::std::string::String, headers: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>, form: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(u16, ::std::string::String), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let url_arg = url;
        let headers_arg = headers;
        let form_arg = form;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_http_post", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(url_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(headers_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(form_arg, py)?];
                Ok((host.bind(py).getattr("_core_http_post")?, ::pyo3::types::PyTuple::new(py, args)?))
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
            let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "TransferHost._core_http_post"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((::lk_agents_abi::python::number_from_py::<u16>(&item_0, "TransferHost._core_http_post"))?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn move_participant(&self, room: ::std::string::String, identity: ::std::string::String, destination: ::std::string::String) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let room_arg = room;
        let identity_arg = identity;
        let destination_arg = destination;
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_move_participant", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [::pyo3::IntoPyObjectExt::into_bound_py_any(room_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(destination_arg, py)?];
                Ok((host.bind(py).getattr("_core_move_participant")?, ::pyo3::types::PyTuple::new(py, args)?))
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

    fn delete_caller_room(&self) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(&crate::host::Entry, "TransferHost._core_delete_caller_room", &self.context, &[], ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_delete_caller_room")?, ::pyo3::types::PyTuple::new(py, args)?))
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
}
