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
    let out = py.detach(|| {
        <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_constants()
    });
    Ok(crate::generated::records::amd::amd_constants_to_py(py, out)?)
}

/// `_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's,
/// else the default. `provided` is `None` only when the key was absent: an explicit default
/// still beats a disagreeing activity.
#[pyfunction]
#[pyo3(signature = (provided, activity))]
pub fn amd_resolve_max_endpointing_delay<'py>(
    py: Python<'py>,
    provided: Option<Bound<'_, PyAny>>,
    activity: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let provided = provided
        .map(|v| {
            crate::python::number_from_py::<f64>(&v, "provided").and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "provided"))
                })
            })
        })
        .transpose()?;
    let activity = activity
        .map(|v| {
            crate::python::number_from_py::<f64>(&v, "activity").and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "activity"))
                })
            })
        })
        .transpose()?;
    let out = py.detach(|| <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_resolve_max_endpointing_delay(provided, activity));
    Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(out.as_secs_f64(), py)?)
}

/// `AMDPredictionEvent.is_human`.
#[pyfunction]
#[pyo3(signature = (prediction))]
pub fn amd_prediction_is_human(py: Python<'_>, prediction: Bound<'_, PyAny>) -> PyResult<bool> {
    let prediction = crate::generated::records::amd::amd_prediction_event_from_py(&prediction)?;
    let out = py.detach(|| {
        <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_prediction_is_human(
            prediction,
        )
    });
    Ok(out)
}

/// `AMDPredictionEvent.is_machine`.
#[pyfunction]
#[pyo3(signature = (prediction))]
pub fn amd_prediction_is_machine(py: Python<'_>, prediction: Bound<'_, PyAny>) -> PyResult<bool> {
    let prediction = crate::generated::records::amd::amd_prediction_event_from_py(&prediction)?;
    let out = py.detach(|| {
        <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_prediction_is_machine(
            prediction,
        )
    });
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type DataStreamAudioOutputImpl = <crate::Selected as super::Implementations>::DataStreamAudioOutput;

fn _extract_data_stream_audio_output_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<u32>(&obj, "sample_rate")).map(Some)
    }
}

fn _extract_data_stream_audio_output_new_wait_remote_track(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<i32>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<i32>(&obj, "wait_remote_track")).map(Some)
    }
}

fn _extract_data_stream_audio_output_new_clear_buffer_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(&obj, "clear_buffer_timeout")).map(Some)
    }
}

/// The agent's native audio sink, sending one byte stream per avatar speech segment.
#[pyclass(frozen, name = "DataStreamAudioOutput", module = "livekit.agents._core")]
pub struct DataStreamAudioOutput(pub(crate) ::std::sync::Arc<DataStreamAudioOutputImpl>);

#[pymethods]
impl DataStreamAudioOutput {
    #[new]
    #[pyo3(signature = (room, *, destination_identity, sample_rate = None, wait_remote_track = None, clear_buffer_timeout = Some(2.0), wait_playback_start = false))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        destination_identity: ::std::string::String,
        #[pyo3(from_py_with = _extract_data_stream_audio_output_new_sample_rate)]
        sample_rate: ::std::option::Option<u32>,
        #[pyo3(from_py_with = _extract_data_stream_audio_output_new_wait_remote_track)]
        wait_remote_track: ::std::option::Option<i32>,
        #[pyo3(from_py_with = _extract_data_stream_audio_output_new_clear_buffer_timeout)]
        clear_buffer_timeout: ::std::option::Option<f64>,
        wait_playback_start: bool,
    ) -> PyResult<DataStreamAudioOutput> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let room = crate::conversions::host_ref_from_py(&room)?;
        let out = py.detach(|| <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::new(&env, room, destination_identity, sample_rate, wait_remote_track, clear_buffer_timeout, wait_playback_start)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
        Ok(DataStreamAudioOutput(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn label(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = py.detach(|| <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::label(&self.0));
        Ok(out)
    }

    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> PyResult<::std::option::Option<u32>> {
        crate::host::check_pid()?;
        let out = py.detach(|| <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::sample_rate(&self.0));
        Ok(out)
    }

    /// Stop admitting audio, settle startup and pending writes, close streams and retire RPC routes.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::AudioOutput`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::AudioOutput>(),
        read: |obj| {
            let this = obj.cast::<DataStreamAudioOutput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type DataStreamAudioReceiverImpl =
    <crate::Selected as super::Implementations>::DataStreamAudioReceiver;

fn _extract_data_stream_audio_receiver_new_frame_size_ms(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    crate::python::number_from_py::<u32>(&obj, "frame_size_ms")
}

fn _extract_data_stream_audio_receiver_new_rpc_max_retries(
    obj: &Bound<'_, PyAny>,
) -> PyResult<u32> {
    crate::python::number_from_py::<u32>(&obj, "rpc_max_retries")
}

fn _extract_data_stream_audio_receiver_notify_playback_finished_playback_position(
    obj: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "playback_position")
}

/// The worker's receiver keeps stream identity with each delivered frame and playback report.
#[pyclass(frozen, name = "DataStreamAudioReceiver", module = "livekit.agents._core")]
pub struct DataStreamAudioReceiver(pub(crate) ::std::sync::Arc<DataStreamAudioReceiverImpl>);

#[pymethods]
impl DataStreamAudioReceiver {
    #[new]
    #[pyo3(signature = (room, *, sender_identity = None, frame_size_ms = 100, rpc_max_retries = 3, on_clear_buffer = None))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        sender_identity: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_data_stream_audio_receiver_new_frame_size_ms)] frame_size_ms: u32,
        #[pyo3(from_py_with = _extract_data_stream_audio_receiver_new_rpc_max_retries)]
        rpc_max_retries: u32,
        on_clear_buffer: Option<Bound<'_, PyAny>>,
    ) -> PyResult<DataStreamAudioReceiver> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let room = crate::generated::voice::RtcRoomProxy::from_host(&room)?;
        let on_clear_buffer = on_clear_buffer
            .map(|v| crate::generated::voice::ClearBufferListenerProxy::from_host(&v))
            .transpose()?;
        let out = py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::new(&env, room, sender_identity, frame_size_ms, rpc_max_retries, on_clear_buffer)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
        Ok(DataStreamAudioReceiver(::std::sync::Arc::new(out)))
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn next_audio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::next_audio(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    {
                        let (x0, x1, x2, x3) = out;
                        ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?, match x2 { None => py.None().into_bound(py), Some(x) => crate::generated::records::io_events::avatar_playback_segment_to_py(py, x)? }, match x3 { None => py.None().into_bound(py), Some(x) => crate::generated::records::avatar::audio_receiver_frame_to_py(py, x)? }])?.into_any()
                    },
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (segment))]
    fn notify_playback_started(&self, py: Python<'_>, segment: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let segment =
            crate::generated::records::io_events::avatar_playback_segment_from_py(&segment)?;
        py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::notify_playback_started(&self.0, segment)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (segment, playback_position, interrupted))]
    fn notify_playback_finished(
        &self,
        py: Python<'_>,
        segment: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_data_stream_audio_receiver_notify_playback_finished_playback_position)]
        playback_position: f64,
        interrupted: bool,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let segment =
            crate::generated::records::io_events::avatar_playback_segment_from_py(&segment)?;
        py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::notify_playback_finished(&self.0, segment, playback_position, interrupted)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn aclose(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::aclose(&self.0)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::avatar::AvatarAudioReceiver`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
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
    crate::python::number_from_py::<u64>(&obj, "epoch")
}

/// The avatar worker's native queue owner. Its decisions remain in RunnerCore.
#[pyclass(frozen, name = "AvatarRunner", module = "livekit.agents._core")]
pub struct AvatarRunner(pub(crate) ::std::sync::Arc<AvatarRunnerImpl>);

#[pymethods]
impl AvatarRunner {
    #[new]
    #[pyo3(signature = (room, *, receiver, generator, media, lazy_publish))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        receiver: Bound<'_, PyAny>,
        generator: Bound<'_, PyAny>,
        media: Bound<'_, PyAny>,
        lazy_publish: bool,
    ) -> PyResult<AvatarRunner> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let room =
            crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "room",
                        ": expected a native room session"
                    ))
                })?;
        let receiver = crate::generated::voice::AvatarAudioReceiverProxy::from_host(&receiver)?;
        let generator = crate::generated::voice::AvatarVideoGeneratorProxy::from_host(&generator)?;
        let media = crate::generated::voice::AvatarMediaProxy::from_host(&media)?;
        let out = py
            .detach(|| {
                <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::new(
                    &env,
                    room,
                    receiver,
                    generator,
                    media,
                    lazy_publish,
                )
            })
            .map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(
                    &e,
                )
            })?;
        Ok(AvatarRunner(::std::sync::Arc::new(out)))
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn wait_for_complete<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::wait_for_complete(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (epoch))]
    fn clear_buffer(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_avatar_runner_clear_buffer_epoch)] epoch: u64,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::clear_buffer(
                &self.0, epoch,
            )
        });
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
type PlayStateImpl = <crate::Selected as super::Implementations>::PlayState;

fn _extract_play_state_new_fade_out(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "fade_out")
}

/// A sound's monotone stop and completion handle.
#[pyclass(frozen, name = "PlayState", module = "livekit.agents._core")]
pub struct PlayState(pub(crate) ::std::sync::Arc<PlayStateImpl>);

#[pymethods]
impl PlayState {
    #[new]
    #[pyo3(signature = (fade_out = 0.0))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_play_state_new_fade_out)] fade_out: f64,
    ) -> PyResult<PlayState> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::new(fade_out)
        });
        Ok(PlayState(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::done(&self.0)
        });
        Ok(out)
    }

    #[getter]
    fn stop_requested(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::stop_requested(
                &self.0,
            )
        });
        Ok(out)
    }

    fn stop(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::stop(&self.0)
        });
        Ok(out)
    }

    #[pyo3(name = "mark_playout_done")]
    fn finish(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::finish(&self.0)
        });
        Ok(out)
    }

    #[pyo3(name = "wait_for_playout")]
    fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out =
                <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::wait(&this)
                    .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
type BackgroundAudioPlayerImpl = <crate::Selected as super::Implementations>::BackgroundAudioPlayer;

fn _extract_background_audio_player_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    crate::python::number_from_py::<u32>(&obj, "sample_rate")
}

fn _extract_background_audio_player_new_num_channels(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    crate::python::number_from_py::<u16>(&obj, "num_channels")
}

fn _extract_background_audio_player_new_blocksize(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    crate::python::number_from_py::<usize>(&obj, "blocksize")
}

fn _extract_background_audio_player_new_stream_timeout_ms(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    crate::python::number_from_py::<u64>(&obj, "stream_timeout_ms")
}

fn _extract_background_audio_player_start_draw(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "draw")
}

fn _extract_background_audio_player_play_draw(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "draw")
}

fn _extract_background_audio_player_on_agent_state_draw(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "draw")
}

/// Runtime-owned mixer and published background track.
#[pyclass(frozen, name = "BackgroundAudioPlayer", module = "livekit.agents._core")]
pub struct BackgroundAudioPlayer(pub(crate) ::std::sync::Arc<BackgroundAudioPlayerImpl>);

#[pymethods]
impl BackgroundAudioPlayer {
    #[new]
    #[pyo3(signature = (sample_rate = 48000, num_channels = 1, *, blocksize = 4800, stream_timeout_ms = 200))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_background_audio_player_new_sample_rate)] sample_rate: u32,
        #[pyo3(from_py_with = _extract_background_audio_player_new_num_channels)] num_channels: u16,
        #[pyo3(from_py_with = _extract_background_audio_player_new_blocksize)] blocksize: usize,
        #[pyo3(from_py_with = _extract_background_audio_player_new_stream_timeout_ms)]
        stream_timeout_ms: u64,
    ) -> PyResult<BackgroundAudioPlayer> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::new(&env, sample_rate, num_channels, blocksize, stream_timeout_ms)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))?;
        Ok(BackgroundAudioPlayer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (room, draw, *, ambient, choose, source = None))]
    fn start<'py>(
        &self,
        py: Python<'py>,
        room: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_background_audio_player_start_draw)] draw: f64,
        ambient: Bound<'_, PyAny>,
        choose: bool,
        source: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let room =
            crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "room",
                        ": expected a native room session"
                    ))
                })?;
        let ambient = ambient.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::background_audio_wire::sound_from_py(&item)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let source = source.map(|v| crate::python::str_from_py(&v)).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::start(&this, room, draw, ambient, choose, source)
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(crate::python::str_to_py(py, &out), py)
            })
        })
    }

    #[pyo3(signature = (sounds, draw, *, looping = false, ambient = false, choose = true))]
    fn play(
        &self,
        py: Python<'_>,
        sounds: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_background_audio_player_play_draw)] draw: f64,
        looping: bool,
        ambient: bool,
        choose: bool,
    ) -> PyResult<super::voice::PlayState> {
        crate::host::check_pid()?;
        let sounds = sounds.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::background_audio_wire::sound_from_py(&item)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::play(&self.0, sounds, draw, looping, ambient, choose)).map_err(|e: ::lk_agents_abi::voice::background_audio::PlaybackError| <::lk_agents_abi::voice::background_audio::PlaybackError as crate::python::HostException>::to_py_err(&e))?;
        Ok(super::voice::PlayState(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (sounds, draw, thinking, *, choose = true))]
    fn on_agent_state(
        &self,
        py: Python<'_>,
        sounds: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_background_audio_player_on_agent_state_draw)] draw: f64,
        thinking: bool,
        choose: bool,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let sounds = sounds.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::background_audio_wire::sound_from_py(&item)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::on_agent_state(&self.0, sounds, draw, thinking, choose)).map_err(|e: ::lk_agents_abi::voice::background_audio::PlaybackError| <::lk_agents_abi::voice::background_audio::PlaybackError as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::closed(&self.0));
        Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
type PlaybackTrackerImpl = <crate::Selected as super::Implementations>::Playback;

/// `_core.PlaybackTracker`: the tracker a host sink reports to (`io.AudioOutput._core_tracker()`),
/// shared with the session that reads it.
#[pyclass(frozen, name = "PlaybackTracker", module = "livekit.agents._core")]
pub struct PlaybackTracker(pub(crate) ::std::sync::Arc<PlaybackTrackerImpl>);

#[pymethods]
impl PlaybackTracker {
    #[new]
    fn new(py: Python<'_>) -> PyResult<PlaybackTracker> {
        crate::host::check_pid()?;
        let out =
            py.detach(|| <PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::new());
        Ok(PlaybackTracker(::std::sync::Arc::new(out)))
    }

    fn note_flush(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::note_flush(&self.0)
        });
        Ok(())
    }

    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::reset(&self.0));
        Ok(())
    }

    /// `AudioOutput.wait_for_playout()`: the last event as `(playback_position, interrupted,
    /// synchronized_transcript)`.
    #[pyo3(name = "wait_for_playout")]
    fn wait_for_played<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out =
                <PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::wait_for_played(
                    &this,
                )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::PlaybackTracker`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(),
        read: |obj| {
            let this = obj.cast::<PlaybackTracker>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackTracker> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
type PlaybackReportsImpl = <crate::Selected as super::Implementations>::PlaybackReports;

fn _extract_playback_reports_started_created_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "created_at")
}

fn _extract_playback_reports_progressed_started_at(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "started_at")
}

fn _extract_playback_reports_progressed_offset(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "offset")
}

fn _extract_playback_reports_progressed_duration(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "duration")
}

fn _extract_playback_reports_finished_playback_position(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "playback_position")
}

#[pyfunction]
#[pyo3(signature = (tracker, segment))]
pub fn _core_playback_begin(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
) -> PyResult<bool> {
    crate::host::check_pid()?;
    let tracker =
        crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&tracker)
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "tracker",
                    ": expected a `_core.PlaybackTracker`"
                ))
            })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::begin(
            tracker, segment,
        )
    });
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker))]
pub fn _core_playback_current_segment<'py>(
    py: Python<'py>,
    tracker: Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    crate::host::check_pid()?;
    let tracker =
        crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&tracker)
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "tracker",
                    ": expected a `_core.PlaybackTracker`"
                ))
            })?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::current_segment(
            tracker,
        )
    });
    Ok(out
        .map(|value| {
            Ok::<_, ::pyo3::PyErr>(crate::generated::records::io_events::playback_segment_to_py(
                py, value,
            )?)
        })
        .transpose()?)
}

#[pyfunction]
#[pyo3(signature = (tracker))]
pub fn _core_playback_note_capture<'py>(
    py: Python<'py>,
    tracker: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let tracker =
        crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&tracker)
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "tracker",
                    ": expected a `_core.PlaybackTracker`"
                ))
            })?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::note_capture(tracker)
    });
    Ok(crate::generated::records::io_events::playback_segment_to_py(py, out)?)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, created_at))]
pub fn _core_playback_started(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_playback_reports_started_created_at)] created_at: f64,
) -> PyResult<bool> {
    crate::host::check_pid()?;
    let tracker =
        crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&tracker)
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "tracker",
                    ": expected a `_core.PlaybackTracker`"
                ))
            })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::started(
            tracker, segment, created_at,
        )
    });
    Ok(out)
}

/// Invalid durations are refused before updating playback or notifying observers.
#[pyfunction]
#[pyo3(signature = (tracker, segment, started_at, offset, duration))]
pub fn _core_playback_progressed(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_playback_reports_progressed_started_at)] started_at: f64,
    #[pyo3(from_py_with = _extract_playback_reports_progressed_offset)] offset: f64,
    #[pyo3(from_py_with = _extract_playback_reports_progressed_duration)] duration: f64,
) -> PyResult<bool> {
    crate::host::check_pid()?;
    let tracker =
        crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&tracker)
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "tracker",
                    ": expected a `_core.PlaybackTracker`"
                ))
            })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = py.detach(|| <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::progressed(tracker, segment, started_at, offset, duration)).map_err(|e: ::lk_agents_abi::voice::io::InvalidPlaybackTime| <::lk_agents_abi::voice::io::InvalidPlaybackTime as crate::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, playback_position, interrupted, synchronized_transcript))]
pub fn _core_playback_finished(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_playback_reports_finished_playback_position)] playback_position: f64,
    interrupted: bool,
    synchronized_transcript: ::std::option::Option<::std::string::String>,
) -> PyResult<bool> {
    crate::host::check_pid()?;
    let tracker =
        crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&tracker)
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "tracker",
                    ": expected a `_core.PlaybackTracker`"
                ))
            })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(&segment)?;
    let out = py.detach(|| <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::finished(tracker, segment, playback_position, interrupted, synchronized_transcript)).map_err(|e: ::lk_agents_abi::voice::io::InvalidPlaybackTime| <::lk_agents_abi::voice::io::InvalidPlaybackTime as crate::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type AgentNodesImpl = <crate::Selected as super::Implementations>::AgentNodes;

/// `_core.AgentNodes(host)`: the session's registered nodes over the host's `_SessionNodes`, taken
/// in the binding of the agent they run for.
#[pyclass(frozen, name = "AgentNodes", module = "livekit.agents._core")]
pub struct AgentNodes(pub(crate) ::std::sync::Arc<AgentNodesImpl>);

#[pymethods]
impl AgentNodes {
    #[new]
    #[pyo3(signature = (host))]
    fn new(py: Python<'_>, host: Bound<'_, PyAny>) -> PyResult<AgentNodes> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::voice::SessionNodesProxy::from_host(&host)?;
        let out = py.detach(|| {
            <AgentNodesImpl as ::lk_agents_abi::voice::nodes::AgentNodes>::new(&env, host)
        });
        Ok(AgentNodes(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::nodes::NodeFactory`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(),
        read: |obj| {
            let this = obj.cast::<AgentNodes>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::NodeFactory> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type CoreNodeCallsImpl = <crate::Selected as super::Implementations>::NodeCalls;

#[pyclass(frozen, name = "_core_NodeCalls", module = "livekit.agents._core")]
pub struct CoreNodeCalls(pub(crate) ::std::sync::Arc<CoreNodeCallsImpl>);

#[pymethods]
impl CoreNodeCalls {}

// Wherever the host passes a `::lk_agents_abi::voice::nodes::ProviderCalls`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>(),
        read: |obj| {
            let this = obj.cast::<CoreNodeCalls>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::ProviderCalls> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type TtsNodeFramesImpl = <crate::Selected as super::Implementations>::TtsNodeFrames;

/// The default `tts_node`'s frames.
#[pyclass(frozen, name = "TtsNodeFrames", module = "livekit.agents._core")]
pub struct TtsNodeFrames(pub(crate) ::std::sync::Arc<TtsNodeFramesImpl>);

#[pymethods]
impl TtsNodeFrames {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            match <TtsNodeFramesImpl as ::lk_agents_abi::voice::nodes::TtsNodeFrames>::next(&this).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::agent_nodes::node_frame_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type SttNodeEventsImpl = <crate::Selected as super::Implementations>::SttNodeEvents;

/// The default `stt_node`'s speech events.
#[pyclass(frozen, name = "SttNodeEvents", module = "livekit.agents._core")]
pub struct SttNodeEvents(pub(crate) ::std::sync::Arc<SttNodeEventsImpl>);

#[pymethods]
impl SttNodeEvents {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            match <SttNodeEventsImpl as ::lk_agents_abi::voice::nodes::SttNodeEvents>::next(&this).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::python::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::stt::speech_event_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type DefaultNodesImpl = <crate::Selected as super::Implementations>::DefaultNodes;

/// `conn_options`: the node's, else the provider's own policy (models.md), else the
/// defaults. `calls`: the running agent node's, when it runs this one.
#[pyfunction]
#[pyo3(signature = (tts, text, *, conn_options = None, calls = None))]
pub fn tts_node(
    py: Python<'_>,
    tts: Bound<'_, PyAny>,
    text: Bound<'_, PyAny>,
    conn_options: Option<Bound<'_, PyAny>>,
    calls: Option<Bound<'_, PyAny>>,
) -> PyResult<super::voice::TtsNodeFrames> {
    crate::host::check_pid()?;
    let env = crate::process::env();
    let tts = crate::generated::tts::TtsProxy::from_host(&tts)?;
    let text = crate::host::capture_context(text.py())
        .and_then(|context| {
            crate::host::stream::HostStream::open(
                text.py(),
                &text,
                &context,
                |o| Ok(o.extract::<::std::string::String>()?),
                ::lk_agents_runtime::process::host_clock(),
            )
        })
        .map(|s| {
            ::std::boxed::Box::new(s)
                as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::std::string::String>>
        })?;
    let conn_options =
        conn_options.map(|v| crate::conversions::api_connect_options_from_py(&v)).transpose()?;
    let calls = calls
        .map(|v| {
            crate::python::native::<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>(&v)
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "calls",
                        ": expected the calls of a registered tts_node"
                    ))
                })
        })
        .transpose()?;
    let out = py
        .detach(|| {
            <DefaultNodesImpl as ::lk_agents_abi::voice::nodes::DefaultNodes>::tts_node(
                &env,
                tts,
                text,
                conn_options,
                calls,
            )
        })
        .map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| {
            <::lk_agents_abi::ApiConnectOptionsError as crate::python::HostException>::to_py_err(&e)
        })?;
    Ok(super::voice::TtsNodeFrames(::std::sync::Arc::new(out)))
}

#[pyfunction]
#[pyo3(signature = (stt, audio, *, conn_options = None, vad = None))]
pub fn stt_node(
    py: Python<'_>,
    stt: Bound<'_, PyAny>,
    audio: Bound<'_, PyAny>,
    conn_options: Option<Bound<'_, PyAny>>,
    vad: Option<Bound<'_, PyAny>>,
) -> PyResult<super::voice::SttNodeEvents> {
    crate::host::check_pid()?;
    let env = crate::process::env();
    let stt = crate::generated::stt::SttProxy::from_host(&stt)?;
    let audio = crate::host::capture_context(audio.py())
        .and_then(|context| {
            crate::host::stream::HostStream::open(
                audio.py(),
                &audio,
                &context,
                |o| Ok(crate::conversions::audio_frame_from_py(&o)?),
                ::lk_agents_runtime::process::host_clock(),
            )
        })
        .map(|s| {
            ::std::boxed::Box::new(s)
                as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>
        })?;
    let conn_options =
        conn_options.map(|v| crate::conversions::api_connect_options_from_py(&v)).transpose()?;
    let vad = vad.map(|v| crate::generated::vad::VadProxy::from_host(&v)).transpose()?;
    let out = py
        .detach(|| {
            <DefaultNodesImpl as ::lk_agents_abi::voice::nodes::DefaultNodes>::stt_node(
                &env,
                stt,
                audio,
                conn_options,
                vad,
            )
        })
        .map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| {
            <::lk_agents_abi::ApiConnectOptionsError as crate::python::HostException>::to_py_err(&e)
        })?;
    Ok(super::voice::SttNodeEvents(::std::sync::Arc::new(out)))
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/events.rs.
type RoomEventBusImpl = <crate::Selected as super::Implementations>::RoomEventBus;

/// The fan-out a host room creates and hands its SDK callbacks.
#[pyclass(frozen, name = "RoomEventBus", module = "livekit.agents._core")]
pub struct RoomEventBus(pub(crate) ::std::sync::Arc<RoomEventBusImpl>);

#[pymethods]
impl RoomEventBus {
    #[new]
    fn new(py: Python<'_>) -> PyResult<RoomEventBus> {
        crate::host::check_pid()?;
        let out =
            py.detach(|| <RoomEventBusImpl as ::lk_agents_abi::voice::rtc::RoomEventBus>::new());
        Ok(RoomEventBus(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::RoomEventSink`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(),
        read: |obj| {
            let this = obj.cast::<RoomEventBus>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RoomEventSink> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/events.rs.
type RtcRuntimeImpl = <crate::Selected as super::Implementations>::RtcRuntime;

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_subscribed(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
    publication: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_subscribed(
            bus,
            participant,
            publication,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_unsubscribed(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
    publication: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_unsubscribed(
            bus,
            participant,
            publication,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_published(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
    publication: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_published(
            bus,
            participant,
            publication,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_unpublished(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
    publication: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::track_unpublished(
            bus,
            participant,
            publication,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, publication))]
pub fn _core_room_local_track_published(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    publication: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let publication = crate::generated::voice::TrackPublicationProxy::from_host(&publication)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::local_track_published(
            bus,
            publication,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant))]
pub fn _core_room_participant_connected(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_connected(
            bus,
            participant,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant))]
pub fn _core_room_participant_active(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_active(
            bus,
            participant,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant))]
pub fn _core_room_participant_attributes_changed(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    py.detach(|| <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_attributes_changed(bus, participant));
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, participant, reason))]
pub fn _core_room_participant_disconnected(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
    reason: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = crate::generated::voice::RemoteParticipantProxy::from_host(&participant)?;
    let reason = crate::python::str_from_py(&reason)?;
    py.detach(|| <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::participant_disconnected(bus, participant, reason));
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, state))]
pub fn _core_room_connection_state_changed(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    state: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let state = crate::python::str_from_py(&state)?;
    py.detach(|| <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::connection_state_changed(bus, state));
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, reason))]
pub fn _core_room_disconnected(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    reason: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let reason = crate::python::str_from_py(&reason)?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::disconnected(
            bus, reason,
        )
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (bus, digit, participant))]
pub fn _core_room_sip_dtmf_received(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    digit: ::std::string::String,
    participant: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    crate::host::check_pid()?;
    let bus = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&bus)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = participant.map(|v| crate::python::str_from_py(&v)).transpose()?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::sip_dtmf_received(
            bus,
            digit,
            participant,
        )
    });
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/playout.rs.
type PlayoutLedgerImpl = <crate::Selected as super::Implementations>::PlayoutLedger;

fn _extract_playout_ledger_add_pushed_samples_samples_per_channel(
    obj: &Bound<'_, PyAny>,
) -> PyResult<u32> {
    crate::python::number_from_py::<u32>(&obj, "samples_per_channel")
}

fn _extract_playout_ledger_add_pushed_samples_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    crate::python::number_from_py::<u32>(&obj, "sample_rate")
}

#[pyclass(frozen, name = "PlayoutLedger", module = "livekit.agents._core")]
pub struct PlayoutLedger(pub(crate) ::std::sync::Arc<PlayoutLedgerImpl>);

#[pymethods]
impl PlayoutLedger {
    #[new]
    fn new(py: Python<'_>) -> PyResult<PlayoutLedger> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::new()
        });
        Ok(PlayoutLedger(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (samples_per_channel, sample_rate))]
    fn add_pushed_samples(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_playout_ledger_add_pushed_samples_samples_per_channel)]
        samples_per_channel: u32,
        #[pyo3(from_py_with = _extract_playout_ledger_add_pushed_samples_sample_rate)] sample_rate: u32,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::add_pushed_samples(&self.0, samples_per_channel, sample_rate));
        Ok(())
    }

    #[pyo3(signature = (duration))]
    fn add_discarded(&self, py: Python<'_>, duration: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let duration =
            crate::python::number_from_py::<f64>(&duration, "duration").and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "duration"))
                })
            })?;
        py.detach(|| <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::add_discarded(&self.0, duration));
        Ok(())
    }

    #[pyo3(signature = (queued))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        queued: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let queued = crate::python::number_from_py::<f64>(&queued, "queued").and_then(|s| {
            ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "queued"))
            })
        })?;
        let out = py.detach(|| {
            <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::snapshot(
                &self.0, queued,
            )
        });
        Ok(crate::generated::records::rtc_wire::playback_info_to_py(py, out)?)
    }

    fn settle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::playout::PlayoutLedger>::settle(
                &self.0,
            )
        });
        Ok(crate::generated::records::rtc_wire::playback_info_to_py(py, out)?)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type ByteStreamSenderImpl = <crate::Selected as super::Implementations>::ByteStreamSender;

#[pyclass(frozen, name = "ByteStreamSender", module = "livekit.agents._core")]
pub struct ByteStreamSender(pub(crate) ::std::sync::Arc<ByteStreamSenderImpl>);

#[pymethods]
impl ByteStreamSender {
    #[pyo3(signature = (info, sender, chunks))]
    fn open(
        &self,
        py: Python<'_>,
        info: Bound<'_, PyAny>,
        sender: Bound<'_, PyAny>,
        chunks: Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        crate::host::check_pid()?;
        let info = crate::conversions::stream_info_from_py(&info)?;
        let sender = crate::python::str_from_py(&sender)?;
        let chunks = crate::host::capture_context(chunks.py())
            .and_then(|context| {
                crate::host::stream::HostStream::open(
                    chunks.py(),
                    &chunks,
                    &context,
                    |o| {
                        Ok({
                            if o.is_instance_of::<::pyo3::types::PyBytes>()
                                || o.is_instance_of::<::pyo3::types::PyByteArray>()
                                || o.is_instance_of::<::pyo3::types::PyMemoryView>()
                            {
                                crate::python::owned_bytes(&o, "chunks")
                            } else {
                                o.try_iter().and_then(|items| {
                                    items
                                        .map(|item| {
                                            let item = item?;
                                            crate::python::number_from_py::<u8>(&item, "chunks")
                                        })
                                        .collect::<PyResult<Vec<_>>>()
                                })
                            }
                        }?)
                    },
                    ::lk_agents_runtime::process::host_clock(),
                )
            })
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::std::vec::Vec<u8>>>
            })?;
        let out = py.detach(|| {
            <ByteStreamSenderImpl as ::lk_agents_abi::voice::rtc::ports::ByteStreamSender>::open(
                &self.0, info, sender, chunks,
            )
        });
        Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::ByteStreamIngress`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress>(),
        read: |obj| {
            let this = obj.cast::<ByteStreamSender>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type TextStreamSenderImpl = <crate::Selected as super::Implementations>::TextStreamSender;

#[pyclass(frozen, name = "TextStreamSender", module = "livekit.agents._core")]
pub struct TextStreamSender(pub(crate) ::std::sync::Arc<TextStreamSenderImpl>);

#[pymethods]
impl TextStreamSender {
    #[pyo3(signature = (info, sender, chunks))]
    fn open(
        &self,
        py: Python<'_>,
        info: Bound<'_, PyAny>,
        sender: Bound<'_, PyAny>,
        chunks: Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        crate::host::check_pid()?;
        let info = crate::conversions::stream_info_from_py(&info)?;
        let sender = crate::python::str_from_py(&sender)?;
        let chunks = crate::host::capture_context(chunks.py())
            .and_then(|context| {
                crate::host::stream::HostStream::open(
                    chunks.py(),
                    &chunks,
                    &context,
                    |o| Ok(o.extract::<::std::string::String>()?),
                    ::lk_agents_runtime::process::host_clock(),
                )
            })
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::std::string::String>>
            })?;
        let out = py.detach(|| {
            <TextStreamSenderImpl as ::lk_agents_abi::voice::rtc::ports::TextStreamSender>::open(
                &self.0, info, sender, chunks,
            )
        });
        Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::TextStreamIngress`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::TextStreamIngress>(),
        read: |obj| {
            let this = obj.cast::<TextStreamSender>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamIngress> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type RpcHandlerImpl = <crate::Selected as super::Implementations>::RpcHandler;

#[pyclass(frozen, name = "RpcHandler", module = "livekit.agents._core")]
pub struct RpcHandler(pub(crate) ::std::sync::Arc<RpcHandlerImpl>);

#[pymethods]
impl RpcHandler {
    #[pyo3(name = "call")]
    #[pyo3(signature = (caller_identity, payload))]
    fn respond(
        &self,
        py: Python<'_>,
        caller_identity: ::std::string::String,
        payload: ::std::string::String,
    ) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <RpcHandlerImpl as ::lk_agents_abi::voice::rtc::ports::RpcHandler>::respond(
                &self.0,
                caller_identity,
                payload,
            )
        });
        Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::RpcCallback`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::RpcCallback>(),
        read: |obj| {
            let this = obj.cast::<RpcHandler>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RpcCallback> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session.rs.
type SessionOptionsImpl = <crate::Selected as super::Implementations>::SessionOptions;

fn _extract_session_options_release_override_layer(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    crate::python::number_from_py::<u64>(&obj, "layer")
}

/// `AgentSession(**kwargs)`'s options before the session holds them: the raw kwargs, read once,
/// and the session layer they became (plus every `update_options` so far).
#[pyclass(frozen, name = "SessionOptions", module = "livekit.agents._core")]
pub struct SessionOptions(pub(crate) ::std::sync::Arc<SessionOptionsImpl>);

#[pymethods]
impl SessionOptions {
    #[new]
    #[pyo3(signature = (kwargs))]
    fn new(py: Python<'_>, kwargs: Bound<'_, PyAny>) -> PyResult<SessionOptions> {
        crate::host::check_pid()?;
        let kwargs = crate::generated::records::session_options::session_kwargs_from_py(&kwargs)?;
        let out = py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::new(kwargs)
        });
        Ok(SessionOptions(::std::sync::Arc::new(out)))
    }

    /// `update_options(..)` before the session runs, on the same `merge` the driver uses.
    #[pyo3(signature = (update))]
    fn update(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let update = crate::generated::records::session_options::update_kwargs_from_py(&update)?;
        py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::update(
                &self.0, update,
            )
        });
        Ok(())
    }

    /// `override_options(..)` before the session runs: a layer over the base, named by the id
    /// returned. The session started over these options holds it until its release.
    #[pyo3(signature = (update))]
    fn override_options(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<u64> {
        crate::host::check_pid()?;
        let update = crate::generated::records::session_options::update_kwargs_from_py(&update)?;
        let out = py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::override_options(&self.0, update));
        Ok(out)
    }

    /// The `override_options` block ended before the session ran.
    #[pyo3(signature = (layer))]
    fn release_override(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_session_options_release_override_layer)] layer: u64,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::release_override(&self.0, layer));
        Ok(())
    }

    /// The resolved configuration over the host-supplied legs, as the dict `session.options` reads
    /// before `start()`.
    #[pyo3(signature = (legs))]
    fn resolved<'py>(
        &self,
        py: Python<'py>,
        legs: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let legs = crate::generated::records::session_options::session_legs_from_py(&legs)?;
        let out = py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::resolved(
                &self.0, legs,
            )
        });
        Ok(crate::generated::records::session_options::resolved_session_options_to_py(py, out)?)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session.rs.
type VoiceSessionImpl = <crate::Selected as super::Implementations>::VoiceSession;

fn _extract_voice_session_collect_dtmf_num_digits(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    crate::python::number_from_py::<usize>(&obj, "num_digits")
}

fn _extract_voice_session_collect_dtmf_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "timeout")
}

fn _extract_voice_session_release_override_layer(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    crate::python::number_from_py::<u64>(&obj, "layer")
}

fn _extract_voice_session_ivr_start_max_silence(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "max_silence")
}

fn _extract_voice_session_filler_start_delay(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "delay")
}

fn _extract_voice_session_filler_start_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(&obj, "interval")).map(Some)
    }
}

fn _extract_voice_session_filler_start_max_steps(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<u32>(&obj, "max_steps")).map(Some)
    }
}

fn _extract_voice_session_filler_stop_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    crate::python::number_from_py::<u64>(&obj, "id")
}

fn _extract_voice_session_wait_for_idle_hold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<u64>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<u64>(&obj, "hold")).map(Some)
    }
}

fn _extract_voice_session_release_idle_hold_hold(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    crate::python::number_from_py::<u64>(&obj, "hold")
}

fn _extract_voice_session_commit_user_turn_transcript_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "transcript_timeout")
}

fn _extract_voice_session_commit_user_turn_stt_flush_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "stt_flush_duration")
}

/// `voice/agent_session.py`'s `AgentSession`: assembled at construction, started at `start()`,
/// closed at `aclose()`; every other method is one command to its loop.
#[pyclass(frozen, name = "VoiceSession", module = "livekit.agents._core")]
pub struct VoiceSession(pub(crate) ::std::sync::Arc<VoiceSessionImpl>);

#[pymethods]
impl VoiceSession {
    /// Assemble a session; the loop begins at `start()`. `room` supplies runtime room I/O, or no
    /// RTC attachment; `options` is the session layer `SessionOptions(kwargs)` built (none is an
    /// empty session layer).
    #[new]
    #[pyo3(signature = (room, args, options = None))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        args: Bound<'_, PyAny>,
        options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<VoiceSession> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let room =
            crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "room",
                        ": expected a native room session"
                    ))
                })?;
        let args = crate::generated::records::voice_session::voice_session_args_from_py(&args)?;
        let options = options
            .map(|v| {
                v.cast::<super::voice::SessionOptions>()
                    .map(|value| value.get().0.as_ref().clone())
                    .map_err(::pyo3::PyErr::from)
            })
            .transpose()?;
        let out = py
            .detach(|| {
                <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::new(
                    &env, room, args, options,
                )
            })
            .map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(
                    &e,
                )
            })?;
        Ok(VoiceSession(::std::sync::Arc::new(out)))
    }

    /// Open the providers and run the loop beside the host event drain. Resolves once the host
    /// has handled the opening events.
    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::start(&self.0, )).map_err(|e: ::lk_agents_abi::voice::session::OpenRefused| <::lk_agents_abi::voice::session::OpenRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::StartFailed| <::lk_agents_abi::voice::session::StartFailed as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `session.usage`: the latest snapshot of the session's usage, settled once it closed.
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::usage(&self.0)
        });
        Ok(crate::generated::records::usage::usage_to_py(py, out)?)
    }

    /// `session.current_agent`: the host's agent the loop installed last, the task's while a
    /// `VoiceTask` holds the floor.
    #[getter]
    fn current_agent<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::current_agent(
                &self.0,
            )
        });
        Ok(out
            .map(|value| Ok::<_, ::pyo3::PyErr>(crate::conversions::host_ref_to_py(py, value)?))
            .transpose()?)
    }

    /// The room's microphone leg, unless `RoomIoConfig(audio_input=False)`, else `None`. A fresh
    /// handle each call over the same leg; safe before or after `start()`.
    #[getter]
    fn audio_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::voice::RoomAudioInput>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::audio_input(
                &self.0,
            )
        });
        Ok(out
            .map(|value| {
                Ok::<_, ::pyo3::PyErr>(super::voice::RoomAudioInput(::std::sync::Arc::new(value)))
            })
            .transpose()?)
    }

    /// The room's video leg, when `RoomIoConfig(video_input=True)` asked for one, else `None`.
    #[getter]
    fn video_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::voice::RoomVideoInput>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::video_input(
                &self.0,
            )
        });
        Ok(out
            .map(|value| {
                Ok::<_, ::pyo3::PyErr>(super::voice::RoomVideoInput(::std::sync::Arc::new(value)))
            })
            .transpose()?)
    }

    /// Record the session into `recorder`; `None` detaches. Safe before or after `start()`; the
    /// file's `finish` stays the host's.
    #[pyo3(signature = (recorder))]
    fn attach_recorder(&self, py: Python<'_>, recorder: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::host::check_pid()?;
        let recorder = recorder
            .map(|v| {
                crate::python::native::<dyn ::lk_agents_abi::recorder::AudioRecorder>(&v)
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "recorder",
                            ": expected a native SessionRecorder"
                        ))
                    })
            })
            .transpose()?;
        py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::attach_recorder(
                &self.0, recorder,
            )
        });
        Ok(())
    }

    /// `Agent.update_options(stt=.., vad=.., llm=.., tts=..)` on a running agent: an in-place
    /// provider swap, not a handoff. Each leg is three-state; a given one goes under the agent's
    /// node when it overrides one.
    #[pyo3(signature = (*, llm = crate::conversions::Omittable::NOT_GIVEN, tts = crate::conversions::Omittable::NOT_GIVEN, stt = crate::conversions::Omittable::NOT_GIVEN, vad = crate::conversions::Omittable::NOT_GIVEN, llm_node = None, tts_node = None, stt_node = None))]
    fn update_models(
        &self,
        py: Python<'_>,
        llm: crate::conversions::Omittable<'_>,
        tts: crate::conversions::Omittable<'_>,
        stt: crate::conversions::Omittable<'_>,
        vad: crate::conversions::Omittable<'_>,
        llm_node: Option<Bound<'_, PyAny>>,
        tts_node: Option<Bound<'_, PyAny>>,
        stt_node: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let llm = llm.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::llm::LlmProxy::from_host(&value)).map(Some)
            }
        })?;
        let tts = tts.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::tts::TtsProxy::from_host(&value)).map(Some)
            }
        })?;
        let stt = stt.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::stt::SttProxy::from_host(&value)).map(Some)
            }
        })?;
        let vad = vad.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::vad::VadProxy::from_host(&value)).map(Some)
            }
        })?;
        let llm_node = llm_node
            .map(|v| {
                crate::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v)
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "llm_node",
                            ": expected the session's nodes (`_core.AgentNodes`)"
                        ))
                    })
            })
            .transpose()?;
        let tts_node = tts_node
            .map(|v| {
                crate::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v)
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "tts_node",
                            ": expected the session's nodes (`_core.AgentNodes`)"
                        ))
                    })
            })
            .transpose()?;
        let stt_node = stt_node
            .map(|v| {
                crate::python::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(&v)
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "stt_node",
                            ": expected the session's nodes (`_core.AgentNodes`)"
                        ))
                    })
            })
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_models(&self.0, llm, tts, stt, vad, llm_node, tts_node, stt_node)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The handoff: replace the agent, keep the session. Resolves `True` once the agent is
    /// installed (`on_enter` issued, not returned), `False` when it was refused (closing).
    #[pyo3(signature = (frame))]
    fn update_agent<'py>(
        &self,
        py: Python<'py>,
        frame: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let frame = crate::generated::records::voice_session::update_agent_args_from_py(&frame)?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_agent(&self.0, frame)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `VoiceTask.run()`: the task's agent, sent from inside a tool (`origin` is its
    /// `"tool:<call_id>"`). The agent holds the floor until one of its tools completes with a
    /// value the host validated, which this resolves to; the parent comes back without its hooks.
    /// A cancelled await ends the task, and every task it started, and puts the parent back.
    #[pyo3(signature = (frame, *, origin = None, opens_with_reply = false))]
    fn run_voice_task<'py>(
        &self,
        py: Python<'py>,
        frame: Bound<'_, PyAny>,
        origin: ::std::option::Option<::std::string::String>,
        opens_with_reply: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let frame = crate::generated::records::voice_session::update_agent_args_from_py(&frame)?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::run_voice_task(&self.0, frame, origin, opens_with_reply)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::VoiceTaskEnded| <::lk_agents_abi::voice::session::VoiceTaskEnded as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::conversions::host_ref_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// `GetDtmfTask`'s keypad half: the keys the linked participant presses, collected until
    /// `num_digits` arrived and a `timeout` (seconds) passed without a key, or `stop_event` was
    /// pressed. Resolves to the digits, space-separated; raises `VoiceTaskFailed` when the
    /// timeout fired short, and is cancelled when the session closes or another collection
    /// starts. With `ask_for_confirmation` the digits are read back instead and the task's
    /// `confirm_inputs` tool completes it. A cancelled await stops the collection.
    #[pyo3(signature = (num_digits, ask_for_confirmation, timeout, stop_event))]
    fn collect_dtmf<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_voice_session_collect_dtmf_num_digits)] num_digits: usize,
        ask_for_confirmation: bool,
        #[pyo3(from_py_with = _extract_voice_session_collect_dtmf_timeout)] timeout: f64,
        stop_event: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::collect_dtmf(&self.0, num_digits, ask_for_confirmation, timeout, stop_event)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::VoiceTaskEnded| <::lk_agents_abi::voice::session::VoiceTaskEnded as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// An `AvatarSession`'s own measurements (its join and playback latency), recorded with the
    /// session's metrics.
    #[pyo3(signature = (metrics))]
    fn report_avatar_metrics(&self, py: Python<'_>, metrics: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let metrics = crate::generated::records::metrics::avatar_metrics_from_py(&metrics)?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::report_avatar_metrics(&self.0, metrics)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.input.audio = ..` after `start()`; `None` detaches.
    #[pyo3(signature = (source))]
    fn set_audio_input(&self, py: Python<'_>, source: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::host::check_pid()?;
        let source =
            source.map(|v| crate::generated::voice::AudioInputProxy::from_host(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_audio_input(&self.0, source)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.input.video = ..` after `start()`: the video pump reads the new source, or stops
    /// on `None`.
    #[pyo3(signature = (source))]
    fn set_video_input(&self, py: Python<'_>, source: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::host::check_pid()?;
        let source =
            source.map(|v| crate::generated::voice::VideoInputProxy::from_host(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_video_input(&self.0, source)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// Select the destination for subsequent output segments. Current work retains its admitted
    /// binding, including when the same endpoint object is selected again.
    #[pyo3(signature = (sink))]
    fn set_audio_output(&self, py: Python<'_>, sink: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::host::check_pid()?;
        let sink =
            sink.map(|v| crate::generated::voice::AudioOutputProxy::from_host(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_audio_output(&self.0, sink)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.output.transcription = ..` after `start()`: the next reply forwards into the new
    /// leg; one in flight keeps the leg it started with.
    #[pyo3(signature = (sink))]
    fn set_transcription_output(
        &self,
        py: Python<'_>,
        sink: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let sink =
            sink.map(|v| crate::generated::voice::TextOutputProxy::from_host(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_transcription_output(&self.0, sink)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `Agent.update_tools()` on a running agent. All three parts go together: the agent hosts
    /// its tools.
    #[pyo3(signature = (tools, *, tools_host, voice_agent))]
    fn update_tools(
        &self,
        py: Python<'_>,
        tools: Option<Bound<'_, PyAny>>,
        tools_host: Option<Bound<'_, PyAny>>,
        voice_agent: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let tools = tools
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::conversions::tool_spec_from_py(&item)
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let tools_host =
            tools_host.map(|v| crate::conversions::host_ref_from_py(&v)).transpose()?;
        let voice_agent = voice_agent
            .map(|v| {
                crate::python::native::<dyn ::lk_agents_abi::agent::AgentFrame>(&v).ok_or_else(
                    || {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "voice_agent",
                            ": expected an agent's voice frame (`_core.VoiceAgent`)"
                        ))
                    },
                )
            })
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_tools(&self.0, tools, tools_host, voice_agent)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.update_options(..)` on a running session: merged into its session layer and
    /// re-resolved; `expressive` reaches the next reply, `keyterms` replace the user's list. A turn
    /// detector object is refused: swapping the detector of a running session is not ported.
    #[pyo3(signature = (update))]
    fn update_turn_options(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let update = crate::generated::records::session_options::update_kwargs_from_py(&update)?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_turn_options(&self.0, update)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `override_options(..)` on a running session: a layer over the base until
    /// `release_override` names the id returned. Refused as `update_turn_options` refuses.
    #[pyo3(signature = (update))]
    fn override_options(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<u64> {
        crate::host::check_pid()?;
        let update = crate::generated::records::session_options::update_kwargs_from_py(&update)?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::override_options(&self.0, update)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// The `override_options` block ended, however it ended. A closed session has nothing left
    /// to restore.
    #[pyo3(signature = (layer))]
    fn release_override(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_release_override_layer)] layer: u64,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::release_override(
                &self.0, layer,
            )
        });
        Ok(())
    }

    /// The core's accepted configuration; `None` for a session built without options.
    fn resolved_options<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::resolved_options(
                &self.0,
            )
        });
        Ok(out
            .map(|value| {
                Ok::<_, ::pyo3::PyErr>(
                    crate::generated::records::session_options::resolved_session_options_to_py(
                        py, value,
                    )?,
                )
            })
            .transpose()?)
    }

    /// `AgentActivity.push_video(frame)`: one sampled frame, on its way to the model. `False`
    /// when the loop did not take it (not started or closed); a non-image part raises.
    #[pyo3(signature = (content))]
    fn push_video(&self, py: Python<'_>, content: Bound<'_, PyAny>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let content = crate::generated::records::chat_items::chat_content_from_py(&content)?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::push_video(&self.0, content)).map_err(|e: ::lk_agents_abi::voice::session::NotAnImage| <::lk_agents_abi::voice::session::NotAnImage as crate::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// `AgentSession.generate_reply()`: reply now, with no end of turn behind it. The overrides
    /// apply to this reply only; `tools=`, `tools_host=` and `voice_agent=` go together.
    #[pyo3(signature = (*, user_input = None, instructions = None, allow_interruptions = None, tool_choice = None, chat_ctx = None, tools = None, tools_host = None, voice_agent = None, on_enter = false, origin = None))]
    fn generate_reply(
        &self,
        py: Python<'_>,
        user_input: ::std::option::Option<::std::string::String>,
        instructions: ::std::option::Option<::std::string::String>,
        allow_interruptions: ::std::option::Option<bool>,
        tool_choice: Option<Bound<'_, PyAny>>,
        chat_ctx: Option<Bound<'_, PyAny>>,
        tools: Option<Bound<'_, PyAny>>,
        tools_host: Option<Bound<'_, PyAny>>,
        voice_agent: Option<Bound<'_, PyAny>>,
        on_enter: bool,
        origin: ::std::option::Option<::std::string::String>,
    ) -> PyResult<crate::handles::SpeechHandle> {
        crate::host::check_pid()?;
        let tool_choice =
            tool_choice.map(|v| crate::conversions::tool_choice_from_py(&v)).transpose()?;
        let chat_ctx = chat_ctx
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::chat_items::chat_item_from_py(&item)
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let tools = tools
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::conversions::tool_spec_from_py(&item)
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let tools_host =
            tools_host.map(|v| crate::conversions::host_ref_from_py(&v)).transpose()?;
        let voice_agent = voice_agent
            .map(|v| {
                crate::python::native::<dyn ::lk_agents_abi::agent::AgentFrame>(&v).ok_or_else(
                    || {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "voice_agent",
                            ": expected an agent's voice frame (`_core.VoiceAgent`)"
                        ))
                    },
                )
            })
            .transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::generate_reply(&self.0, user_input, instructions, allow_interruptions, tool_choice, chat_ctx, tools, tools_host, voice_agent, on_enter, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(crate::handles::SpeechHandle::from(out))
    }

    /// `IVRActivity.start()`: `max_silence` is `max_silence_duration` in seconds; `tools` is
    /// `IVRActivity.tools`. A second start is journalled, not run.
    #[pyo3(signature = (max_silence = 5.0, tools = None, *, tools_host = None))]
    fn ivr_start(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_ivr_start_max_silence)] max_silence: f64,
        tools: Option<Bound<'_, PyAny>>,
        tools_host: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let tools = tools
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::conversions::tool_spec_from_py(&item)
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let tools_host =
            tools_host.map(|v| crate::conversions::host_ref_from_py(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::ivr_start(&self.0, max_silence, tools, tools_host)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AMD._run`: the answering-machine detector over its config, classifier LLM, tools and
    /// optional STT. It listens once the callee answers in the session's room; the verdict comes
    /// back as `amd_result`. A second attach is refused.
    #[pyo3(signature = (cfg, llm, tools, stt, participant_identity))]
    fn attach_amd(
        &self,
        py: Python<'_>,
        cfg: Bound<'_, PyAny>,
        llm: Bound<'_, PyAny>,
        tools: Option<Bound<'_, PyAny>>,
        stt: Option<Bound<'_, PyAny>>,
        participant_identity: ::std::option::Option<::std::string::String>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let cfg = crate::generated::records::amd::amd_config_from_py(&cfg)?;
        let llm = crate::generated::llm::LlmProxy::from_host(&llm)?;
        let tools = tools.map(|v| crate::conversions::host_ref_from_py(&v)).transpose()?;
        let stt = stt.map(|v| crate::generated::stt::SttProxy::from_host(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::attach_amd(&self.0, cfg, llm, tools, stt, participant_identity)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_aclose_impl`: the hooks' `closing`, the close itself, the turn-detector stream, the
    /// remote session host, the room io, then the session's toolsets. Idempotent: a second call
    /// waits for the first.
    #[pyo3(signature = (*, reason = "user_initiated".to_owned(), drain = false))]
    fn aclose<'py>(
        &self,
        py: Python<'py>,
        reason: ::std::string::String,
        drain: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::aclose(&self.0, reason, drain)).map_err(|e: ::lk_agents_abi::voice::session::UnknownCloseReason| <::lk_agents_abi::voice::session::UnknownCloseReason as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `AgentSession.shutdown(drain=)`: `aclose` without waiting for it.
    #[pyo3(signature = (*, drain = true))]
    fn shutdown(&self, py: Python<'_>, drain: bool) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::shutdown(
                &self.0, drain,
            )
        });
        Ok(())
    }

    /// `RoomIO.set_participant(identity)`: point the audio input and the user transcript at
    /// someone else. `None` unlinks.
    #[pyo3(signature = (identity))]
    fn set_participant<'py>(
        &self,
        py: Python<'py>,
        identity: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_participant(
                &self.0, identity,
            )
        });
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::io::IoError| {
                <::lk_agents_abi::voice::io::IoError as crate::python::HostException>::to_py_err(&e)
            })?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `RunContext.with_filler(..)` entered for tool `call_id` of `speech_id`: after `delay` s
    /// idle say `text` (or ask `source`), then repeat every `interval`. Returns the scope's number.
    #[pyo3(signature = (call_id, speech_id, *, delay, interval = None, max_steps = None, text = None, source = None))]
    fn filler_start(
        &self,
        py: Python<'_>,
        call_id: ::std::string::String,
        speech_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_voice_session_filler_start_delay)] delay: f64,
        #[pyo3(from_py_with = _extract_voice_session_filler_start_interval)]
        interval: ::std::option::Option<f64>,
        #[pyo3(from_py_with = _extract_voice_session_filler_start_max_steps)]
        max_steps: ::std::option::Option<u32>,
        text: ::std::option::Option<::std::string::String>,
        source: Option<Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        crate::host::check_pid()?;
        let source = source.map(|v| crate::conversions::host_ref_from_py(&v)).transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::filler_start(&self.0, call_id, speech_id, delay, interval, max_steps, text, source)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// `with_filler`'s exit.
    #[pyo3(signature = (id))]
    fn filler_stop(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_filler_stop_id)] id: u64,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::filler_stop(&self.0, id)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it.
    fn clear_user_turn(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::clear_user_turn(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `Agent.update_instructions()` on the agent that is already driving.
    #[pyo3(signature = (instructions))]
    fn update_instructions(
        &self,
        py: Python<'_>,
        instructions: ::std::string::String,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_instructions(&self.0, instructions)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `input.set_audio_enabled` / `output.set_audio_enabled` /
    /// `output.set_transcription_enabled`: `leg` is `"audio_input"`, `"audio_output"` or
    /// `"transcription"`.
    #[pyo3(signature = (leg, enabled))]
    fn set_io_enabled(
        &self,
        py: Python<'_>,
        leg: ::std::string::String,
        enabled: bool,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_io_enabled(&self.0, leg, enabled)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `RunContext.update()` after the first (an `update` outcome), or a detached tool's end.
    #[pyo3(signature = (call_id, report))]
    fn tool_update(
        &self,
        py: Python<'_>,
        call_id: ::std::string::String,
        report: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let report = crate::generated::records::tool_report::tool_report_from_py(&report)?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::tool_update(&self.0, call_id, report)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `Agent.update_chat_ctx()` on a running agent: replace the context.
    #[pyo3(signature = (items))]
    fn update_chat_ctx(&self, py: Python<'_>, items: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let items = items.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(&item)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_chat_ctx(&self.0, items)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.interrupt(force=)`: the current and queued speeches up to the first
    /// uninterruptible; answers when they are done, or with the refusal.
    #[pyo3(signature = (*, force = false))]
    fn interrupt<'py>(&self, py: Python<'py>, force: bool) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::interrupt(&self.0, force)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::NotInterruptible| <::lk_agents_abi::voice::session::NotInterruptible as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
    /// speech is left. Answers `None`, `"activity_changing"` or `"closing"`.
    fn drain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::drain(&self.0, )).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`).
    #[pyo3(signature = (*, hold = None))]
    fn wait_for_idle<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_voice_session_wait_for_idle_hold)]
        hold: ::std::option::Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::wait_for_idle(&self.0, hold)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::IdleRefused| <::lk_agents_abi::voice::session::IdleRefused as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `RunContext.foreground()`: wait for idle, then hold the floor until `release_idle_hold`.
    /// Answers the hold's id.
    #[pyo3(signature = (*, origin = None))]
    fn wait_for_idle_and_hold<'py>(
        &self,
        py: Python<'py>,
        origin: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::wait_for_idle_and_hold(&self.0, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::IdleRefused| <::lk_agents_abi::voice::session::IdleRefused as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `_wait_for_idle_and_hold`'s exit. A hold the loop no longer has is a no-op.
    #[pyo3(signature = (hold))]
    fn release_idle_hold(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_release_idle_hold_hold)] hold: u64,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::release_idle_hold(
                &self.0, hold,
            )
        });
        Ok(())
    }

    /// `AgentSession._claim_user_turn()`'s enter: pins the user state to `speaking` and holds
    /// `wait_for_idle` open. Reentrant.
    fn claim_user_turn(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::claim_user_turn(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_claim_user_turn()`'s exit: the last release re-derives the user state.
    fn release_user_turn(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::release_user_turn(
                &self.0,
            )
        });
        Ok(())
    }

    /// `VoiceSession._run(user_input=)`, answered on the stream: every run event carries `host`,
    /// the host's run recorder.
    #[pyo3(signature = (*, user_input = None, origin = None, host = None))]
    fn run(
        &self,
        py: Python<'_>,
        user_input: ::std::option::Option<::std::string::String>,
        origin: ::std::option::Option<::std::string::String>,
        host: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::host::check_pid()?;
        let host = host.map(|v| crate::conversions::host_ref_from_py(&v)).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::run(&self.0, user_input, origin, host)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.say(text)`: speak these words, no LLM. The handle is the promise the loop
    /// fulfils.
    #[pyo3(signature = (text, *, allow_interruptions = None, add_to_chat_ctx = true, origin = None))]
    fn say(
        &self,
        py: Python<'_>,
        text: ::std::string::String,
        allow_interruptions: ::std::option::Option<bool>,
        add_to_chat_ctx: bool,
        origin: ::std::option::Option<::std::string::String>,
    ) -> PyResult<crate::handles::SpeechHandle> {
        crate::host::check_pid()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::say(&self.0, text, allow_interruptions, add_to_chat_ctx, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(crate::handles::SpeechHandle::from(out))
    }

    /// `AgentSession.say(AsyncIterable[str])`: the speech reads `text` to its end.
    #[pyo3(signature = (text, *, allow_interruptions = None, add_to_chat_ctx = true, origin = None))]
    fn say_stream(
        &self,
        py: Python<'_>,
        text: Bound<'_, PyAny>,
        allow_interruptions: ::std::option::Option<bool>,
        add_to_chat_ctx: bool,
        origin: ::std::option::Option<::std::string::String>,
    ) -> PyResult<crate::handles::SpeechHandle> {
        crate::host::check_pid()?;
        let text = crate::host::capture_context(text.py())
            .and_then(|context| {
                crate::host::stream::HostStream::open(
                    text.py(),
                    &text,
                    &context,
                    |o| Ok(o.extract::<::std::string::String>()?),
                    ::lk_agents_runtime::process::host_clock(),
                )
            })
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::std::string::String>>
            })?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::say_stream(&self.0, text, allow_interruptions, add_to_chat_ctx, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(crate::handles::SpeechHandle::from(out))
    }

    /// `AgentSession.say(text, audio=..)`: pre-synthesised frames instead of TTS; `text` is still
    /// committed.
    #[pyo3(signature = (text, audio, *, allow_interruptions = None, add_to_chat_ctx = true, origin = None))]
    fn say_audio(
        &self,
        py: Python<'_>,
        text: ::std::string::String,
        audio: Bound<'_, PyAny>,
        allow_interruptions: ::std::option::Option<bool>,
        add_to_chat_ctx: bool,
        origin: ::std::option::Option<::std::string::String>,
    ) -> PyResult<crate::handles::SpeechHandle> {
        crate::host::check_pid()?;
        let audio = crate::host::capture_context(audio.py())
            .and_then(|context| {
                crate::host::stream::HostStream::open(
                    audio.py(),
                    &audio,
                    &context,
                    |o| Ok(crate::conversions::audio_frame_from_py(&o)?),
                    ::lk_agents_runtime::process::host_clock(),
                )
            })
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<
                        dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>,
                    >
            })?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::say_audio(&self.0, text, audio, allow_interruptions, add_to_chat_ctx, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(crate::handles::SpeechHandle::from(out))
    }

    /// `_start_ivr_detection(transcript)`: a final the navigator counts as if heard.
    #[pyo3(signature = (text))]
    fn ivr_final(&self, py: Python<'_>, text: ::std::string::String) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::ivr_final(&self.0, text)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AMD.started`: the detector listens (the callee answered, no verdict out yet).
    fn amd_listening(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::amd_listening(
                &self.0,
            )
        });
        Ok(out)
    }

    /// `AMD.execute()`'s interrupt and `_resume_authorization()`, after its IVR hand-off.
    fn amd_execute(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::amd_execute(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AMD.aclose()`. Idempotent.
    fn amd_close(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::amd_close(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.keyterms`: the list the recogniser was last handed; `None` before any push.
    fn keyterms(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<::std::vec::Vec<::std::string::String>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::keyterms(&self.0)
        });
        Ok(out)
    }

    /// `Agent.chat_ctx` of the driving agent; empty before `start()`.
    fn agent_chat_ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::agent_chat_ctx(
                &self.0,
            )
        });
        Ok(::pyo3::types::PyList::new(
            py,
            out.into_iter()
                .map(|x| {
                    PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any())
    }

    /// `AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
    /// transcript. Runs whether or not awaited; a newer commit supersedes it.
    #[pyo3(signature = (*, transcript_timeout = 2.0, stt_flush_duration = 2.0, skip_reply = false))]
    fn commit_user_turn<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_voice_session_commit_user_turn_transcript_timeout)]
        transcript_timeout: f64,
        #[pyo3(from_py_with = _extract_voice_session_commit_user_turn_stt_flush_duration)]
        stt_flush_duration: f64,
        skip_reply: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::commit_user_turn(&self.0, transcript_timeout, stt_flush_duration, skip_reply)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::python::HostException>::to_py_err(&e))?;
        crate::host::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::CommitSuperseded| <::lk_agents_abi::voice::session::CommitSuperseded as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
type SessionStartImpl = <crate::Selected as super::Implementations>::SessionStart;

/// The job's recording and whether this session is its primary (records) and hosts the
/// remote session protocol; `another_session_is_primary` is the job's claim already taken.
#[pyfunction]
#[pyo3(signature = (*, job_default, another_session_is_primary))]
pub fn plan_session_start<'py>(
    py: Python<'py>,
    job_default: bool,
    another_session_is_primary: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let out = py.detach(|| <SessionStartImpl as ::lk_agents_abi::voice::session_effects::SessionStart>::plan_session_start(job_default, another_session_is_primary));
    Ok(crate::generated::records::session_wire::start_plan_to_py(py, out)?)
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.
type InterruptionGateFunctionsImpl =
    <crate::Selected as super::Implementations>::InterruptionGateFunctions;

/// `AgentActivity._rt_turn_detection_enabled` from the host's reads. `turn_detection` is the
/// explicit mode string (agent over session), `None` when unsaid or a detector object;
/// `streaming_detector` says it was a streaming detector.
#[pyfunction]
#[pyo3(signature = (*, caps_turn_detection, can_disable_turn_detection, vad, turn_detection, streaming_detector, interruption_given))]
pub fn rt_server_turn_detection(
    py: Python<'_>,
    caps_turn_detection: bool,
    can_disable_turn_detection: bool,
    vad: bool,
    turn_detection: ::std::option::Option<::std::string::String>,
    streaming_detector: bool,
    interruption_given: bool,
) -> PyResult<bool> {
    crate::host::check_pid()?;
    let out = py.detach(|| <InterruptionGateFunctionsImpl as ::lk_agents_abi::voice::session_options::InterruptionGateFunctions>::rt_server_turn_detection(caps_turn_detection, can_disable_turn_detection, vad, turn_detection, streaming_detector, interruption_given));
    Ok(out)
}

/// `AgentActivity._resolve_interruption_detection`: whether to build the detector, and if not,
/// why. Constructing it stays the host's.
#[pyfunction]
#[pyo3(signature = (*, stt_aligned_transcript, stt_streaming, realtime, rt_server_turn_detection, has_vad, turn_detection, allow_interruptions, agent_mode, session_mode, hosted, dev_mode))]
pub fn resolve_interruption_detector<'py>(
    py: Python<'py>,
    stt_aligned_transcript: ::std::option::Option<bool>,
    stt_streaming: ::std::option::Option<bool>,
    realtime: bool,
    rt_server_turn_detection: bool,
    has_vad: bool,
    turn_detection: ::std::option::Option<::std::string::String>,
    allow_interruptions: bool,
    agent_mode: Option<Bound<'_, PyAny>>,
    session_mode: Option<Bound<'_, PyAny>>,
    hosted: bool,
    dev_mode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let agent_mode = agent_mode
        .map(|v| crate::generated::records::turn::interruption_mode_from_py(&v))
        .transpose()?;
    let session_mode = session_mode
        .map(|v| crate::generated::records::turn::interruption_mode_from_py(&v))
        .transpose()?;
    let out = py.detach(|| <InterruptionGateFunctionsImpl as ::lk_agents_abi::voice::session_options::InterruptionGateFunctions>::resolve_interruption_detector(stt_aligned_transcript, stt_streaming, realtime, rt_server_turn_detection, has_vad, turn_detection, allow_interruptions, agent_mode, session_mode, hosted, dev_mode));
    Ok(crate::generated::records::session_options::interruption_gate_verdict_to_py(py, out)?)
}

// Declared in rust/crates/lk-agents-abi/src/voice/speech_handle.rs.
type SpeechHandleImpl = <crate::Selected as super::Implementations>::SpeechHandle;

fn _extract_speech_handle_wait_for_generation_step(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<usize>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<usize>(&obj, "step")).map(Some)
    }
}

#[pyclass(frozen, name = "SpeechHandle", module = "livekit.agents._core")]
pub struct SpeechHandle(pub(crate) ::std::sync::Arc<SpeechHandleImpl>);

#[pymethods]
impl SpeechHandle {
    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    #[new]
    #[pyo3(signature = (*, speech_id = "".to_owned(), allow_interruptions = true))]
    fn new(
        py: Python<'_>,
        speech_id: ::std::string::String,
        allow_interruptions: bool,
    ) -> PyResult<SpeechHandle> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::new(
                speech_id,
                allow_interruptions,
            )
        });
        Ok(SpeechHandle(::std::sync::Arc::new(out)))
    }

    /// The host's object for this speech, as it set it; `None` until then.
    #[getter]
    #[pyo3(name = "host")]
    fn host_object<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::host_object(
                &self.0,
            )
        });
        Ok(out
            .map(|value| Ok::<_, ::pyo3::PyErr>(crate::conversions::host_ref_to_py(py, value)?))
            .transpose()?)
    }

    /// Set by the host when it creates its handle over this one: from then on every event, tool
    /// and hook naming the speech hands back this object.
    #[setter(host)]
    fn set_host_object(&self, py: Python<'_>, host: Bound<'_, PyAny>) -> PyResult<()> {
        crate::host::check_pid()?;
        let host = crate::conversions::host_ref_from_py(&host)?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::set_host_object(&self.0, host));
        Ok(())
    }

    /// `speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
    /// the id it was constructed with.
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::id(&self.0)
        });
        Ok(out)
    }

    #[getter]
    fn num_steps(&self, py: Python<'_>) -> PyResult<u32> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::num_steps(
                &self.0,
            )
        });
        Ok(out)
    }

    /// What this speech committed to the conversation; a copy.
    #[getter]
    fn chat_items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::chat_items(
                &self.0,
            )
        });
        Ok(::pyo3::types::PyList::new(
            py,
            out.into_iter()
                .map(|x| {
                    PyResult::Ok(crate::generated::records::chat_items::chat_item_to_py(py, x)?)
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any())
    }

    #[getter]
    fn scheduled(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::scheduled(
                &self.0,
            )
        });
        Ok(out)
    }

    /// `True` from the tick `interrupt()` was called until the speech is done, and once the loop
    /// has interrupted the speech for any reason (an interruption, a supersede, the close). A done
    /// speech reads the loop's verdict.
    #[getter]
    fn interrupted(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::interrupted(
                &self.0,
            )
        });
        Ok(out)
    }

    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::done(&self.0)
        });
        Ok(out)
    }

    /// The core's last accepted `allow_interruptions`.
    #[getter]
    fn allow_interruptions(&self, py: Python<'_>) -> PyResult<bool> {
        crate::host::check_pid()?;
        let out = py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::allow_interruptions(&self.0));
        Ok(out)
    }

    /// Enqueues the change; the accepted view changes when the loop applies it.
    #[setter(allow_interruptions)]
    fn set_allow_interruptions(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::set_allow_interruptions(&self.0, value)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The exception the speech failed with: the host's own object when a provider raised one.
    fn exception<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::exception(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(out
            .map(|value| {
                Ok::<_, ::pyo3::PyErr>(crate::python::ExceptionValue::to_exception(value, py)?)
            })
            .transpose()?)
    }

    /// `interrupt(force=)`, in the reference's order. If the speech does not exist yet, the
    /// command is sent when the loop creates it.
    #[pyo3(signature = (*, force = false))]
    fn interrupt(&self, py: Python<'_>, force: bool) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::interrupt(&self.0, force)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `EndCallTool`: close the session once this speech is done, or once a realtime model's own
    /// reply to the tool result is. Sent to this speech's own session.
    fn end_call(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::end_call(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_hold_interruptions`: disallow interruptions until every hold taken is released. Raises
    /// when the speech is already interrupted, without counting the attempt.
    fn hold_interruptions(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::hold_interruptions(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_release_interruptions`. Idempotent past zero, like the reference's.
    fn release_interruptions(&self, py: Python<'_>) -> PyResult<()> {
        crate::host::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::release_interruptions(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `wait_for_playout()`, and therefore `__await__`, from `origin` (`"tool:<call_id>"` inside a
    /// tool). Raises from inside a tool this speech waits on; otherwise never raises: a speech
    /// the session ended before creating, or a handle constructed directly, has nothing to play.
    #[pyo3(signature = (*, origin = None))]
    fn wait_for_playout<'py>(
        &self,
        py: Python<'py>,
        origin: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_for_playout(&this, origin)
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `_wait_for_generation(step_idx)`: one step of the turn, not the whole thing; `None` is the
    /// latest. No generation, no such step and a never-created speech are distinct errors.
    #[pyo3(signature = (step))]
    fn wait_for_generation<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_speech_handle_wait_for_generation_step)]
        step: ::std::option::Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_for_generation(&this, step)
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Resolves when the speech is interrupted, and only then: a speech that finishes normally
    /// never resolves this, so `wait_if_not_interrupted` can race it.
    fn wait_for_interrupted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_for_interrupted(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Resolves once the loop has created the speech and put it in the queue.
    fn wait_until_created<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_until_created(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::speech_handle::Speech`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::speech_handle::Speech>(),
        read: |obj| {
            let this = obj.cast::<SpeechHandle>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::speech_handle::Speech> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows.rs.
type WorkflowChecksImpl = <crate::Selected as super::Implementations>::WorkflowChecks;

fn _extract_workflow_checks_card_expired_month(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::python::number_from_py::<i64>(&obj, "month")
}

fn _extract_workflow_checks_card_expired_year(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::python::number_from_py::<i64>(&obj, "year")
}

fn _extract_workflow_checks_normalize_date_of_birth_year(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::python::number_from_py::<i64>(&obj, "year")
}

fn _extract_workflow_checks_normalize_date_of_birth_month(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::python::number_from_py::<i64>(&obj, "month")
}

fn _extract_workflow_checks_normalize_date_of_birth_day(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    crate::python::number_from_py::<i64>(&obj, "day")
}

/// The Luhn checksum over an all-digit string.
#[pyfunction]
#[pyo3(signature = (card_number))]
pub fn validate_card_number(py: Python<'_>, card_number: ::std::string::String) -> PyResult<bool> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::validate_card_number(card_number));
    Ok(out)
}

/// Whether a two-digit expiry year and its month precede `today`'s month.
#[pyfunction]
#[pyo3(signature = (month, year, today))]
pub fn card_expired(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_workflow_checks_card_expired_month)] month: i64,
    #[pyo3(from_py_with = _extract_workflow_checks_card_expired_year)] year: i64,
    today: Bound<'_, PyAny>,
) -> PyResult<bool> {
    let today = crate::conversions::date_from_py(&today)?;
    let out = py.detach(|| {
        <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::card_expired(
            month, year, today,
        )
    });
    Ok(out)
}

/// A two-digit year is this century's when it is not past this year's two digits, else the
/// last century's; an invalid or future date is refused.
#[pyfunction]
#[pyo3(signature = (year, month, day, today))]
pub fn normalize_date_of_birth<'py>(
    py: Python<'py>,
    #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_year)] year: i64,
    #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_month)] month: i64,
    #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_day)] day: i64,
    today: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let today = crate::conversions::date_from_py(&today)?;
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_date_of_birth(year, month, day, today)).map_err(|e: ::lk_agents_abi::voice::workflows::DateOfBirthRefused| <::lk_agents_abi::voice::workflows::DateOfBirthRefused as crate::python::HostException>::to_py_err(&e))?;
    Ok(crate::conversions::date_to_py(py, out)?)
}

/// Stripped, and refused unless it reads `local@label.….tld`.
#[pyfunction]
#[pyo3(signature = (email))]
pub fn normalize_email_address(
    py: Python<'_>,
    email: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_email_address(email)).map_err(|e: ::lk_agents_abi::voice::workflows::CaptureRefused| <::lk_agents_abi::voice::workflows::CaptureRefused as crate::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// Whitespace, hyphens, parentheses and dots dropped, and refused unless it reads `+?[1-9]`
/// followed by six to fourteen digits.
#[pyfunction]
#[pyo3(signature = (phone_number))]
pub fn normalize_phone_number(
    py: Python<'_>,
    phone_number: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_phone_number(phone_number)).map_err(|e: ::lk_agents_abi::voice::workflows::CaptureRefused| <::lk_agents_abi::voice::workflows::CaptureRefused as crate::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// Digits are themselves, `*` is 10, `#` is 11 and `A`-`D` are 12-15.
#[pyfunction]
#[pyo3(signature = (symbol))]
pub fn dtmf_code(py: Python<'_>, symbol: ::std::string::String) -> PyResult<u8> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::dtmf_code(symbol)).map_err(|e: ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol| <::lk_agents_abi::voice::workflows::UnknownDtmfSymbol as crate::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// The space-joined digits that fill the DTMF read-back's `{digits}` hole.
#[pyfunction]
#[pyo3(signature = (symbols))]
pub fn format_dtmf(
    py: Python<'_>,
    symbols: ::std::vec::Vec<::std::string::String>,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::format_dtmf(symbols)).map_err(|e: ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol| <::lk_agents_abi::voice::workflows::UnknownDtmfSymbol as crate::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/dtmf.rs.
type DtmfIoImpl = <crate::Selected as super::Implementations>::DtmfIo;

fn _extract_dtmf_io_receive_dtmf_num_digits(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    crate::python::number_from_py::<usize>(&obj, "num_digits")
}

fn _extract_dtmf_io_receive_dtmf_timeout(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    crate::python::number_from_py::<f64>(&obj, "timeout")
}

#[pyfunction]
#[pyo3(signature = (room, events))]
pub fn send_dtmf_events<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    events: ::std::vec::Vec<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let room = crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "room",
                ": expected a native room session"
            ))
        })?;
    crate::host::future_into_py(py, async move {
        let out =
            <DtmfIoImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfIo>::send_dtmf_events(
                room, events,
            )
            .await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, num_digits, timeout, stop_event))]
pub fn receive_dtmf<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_dtmf_io_receive_dtmf_num_digits)] num_digits: usize,
    #[pyo3(from_py_with = _extract_dtmf_io_receive_dtmf_timeout)] timeout: f64,
    stop_event: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let room = crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "room",
                ": expected a native room session"
            ))
        })?;
    crate::host::future_into_py(py, async move {
        let out = <DtmfIoImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfIo>::receive_dtmf(room, num_digits, timeout, stop_event).await.map_err(|e: ::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError| <::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError as crate::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/warm_transfer.rs.
type WarmTransferImpl = <crate::Selected as super::Implementations>::WarmTransfer;

fn _extract_warm_transfer_new_ringing_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::python::number_from_py::<f64>(&obj, "ringing_timeout")).map(Some)
    }
}

/// A transfer plan and the runtime ownership of its current dial.
#[pyclass(frozen, name = "WarmTransfer", module = "livekit.agents._core")]
pub struct WarmTransfer(pub(crate) ::std::sync::Arc<WarmTransferImpl>);

#[pymethods]
impl WarmTransfer {
    #[new]
    #[pyo3(signature = (*, sip_call_to = None, sip_trunk_id = crate::conversions::Omittable::NOT_GIVEN, has_sip_connection = false, sip_number = crate::conversions::Omittable::NOT_GIVEN, sip_headers = None, dtmf = None, ringing_timeout = None, instructions = crate::conversions::Omittable::NOT_GIVEN, conversation = None, env_sip_trunk = None, env_sip_number = "".to_owned(), twilio = None))]
    fn new(
        py: Python<'_>,
        sip_call_to: ::std::option::Option<::std::string::String>,
        sip_trunk_id: crate::conversions::Omittable<'_>,
        has_sip_connection: bool,
        sip_number: crate::conversions::Omittable<'_>,
        sip_headers: ::std::option::Option<
            ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
        >,
        dtmf: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_warm_transfer_new_ringing_timeout)]
        ringing_timeout: ::std::option::Option<f64>,
        instructions: crate::conversions::Omittable<'_>,
        conversation: ::std::option::Option<
            ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
        >,
        env_sip_trunk: ::std::option::Option<::std::string::String>,
        env_sip_number: ::std::string::String,
        twilio: Option<Bound<'_, PyAny>>,
    ) -> PyResult<WarmTransfer> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let sip_trunk_id = sip_trunk_id.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (value.extract::<::std::string::String>()).map(Some)
            }
        })?;
        let sip_number = sip_number.map(|value| value.extract::<::std::string::String>())?;
        let instructions = instructions
            .map(|value| crate::generated::records::workflows::instructions_arg_from_py(&value))?;
        let twilio = twilio
            .map(|v| crate::generated::records::workflows::twilio_dial_from_py(&v))
            .transpose()?;
        let out = py.detach(|| <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::new(&env, sip_call_to, sip_trunk_id, has_sip_connection, sip_number, sip_headers, dtmf, ringing_timeout, instructions, conversation, env_sip_trunk, env_sip_number, twilio)).map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused| <::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused as crate::python::HostException>::to_py_err(&e))?;
        Ok(WarmTransfer(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn instructions(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::host::check_pid()?;
        let out = py.detach(|| <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::instructions(&self.0));
        Ok(out)
    }

    fn tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let out = py.detach(|| <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::tools(&self.0));
        Ok(::pyo3::types::PyList::new(
            py,
            out.into_iter()
                .map(|x| {
                    PyResult::Ok({
                        let d = ::pyo3::types::PyDict::new(py);
                        for (k, x) in x {
                            d.set_item(
                                ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                                crate::python::value_to_py(py, &x)?,
                            )?;
                        }
                        d.into_any()
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any())
    }

    #[pyo3(signature = (host, caller))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        host: Bound<'_, PyAny>,
        caller: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let host = crate::generated::voice::TransferHostProxy::from_host(&host)?;
        let caller =
            crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&caller)
                .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "caller",
                    ": expected a native room session"
                ))
            })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::run(&this, host, caller)
                .await.map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed| <::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (name, arguments))]
    fn call_tool<'py>(
        &self,
        py: Python<'py>,
        name: ::std::string::String,
        arguments: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let arguments = crate::python::value_from_py(&arguments, "arguments")?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::call_tool(&this, name, arguments)
                .await.map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed| <::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/console.rs.
type ConsoleIoImpl = <crate::Selected as super::Implementations>::ConsoleIo;

/// `lk agent console`'s socket, opened when the session starts.
#[pyclass(frozen, name = "ConsoleIo", module = "livekit.agents._core")]
pub struct ConsoleIo(pub(crate) ::std::sync::Arc<ConsoleIoImpl>);

#[pymethods]
impl ConsoleIo {
    #[new]
    #[pyo3(signature = (addr))]
    fn new(py: Python<'_>, addr: ::std::string::String) -> PyResult<ConsoleIo> {
        crate::host::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| {
            <ConsoleIoImpl as ::lk_agents_abi::voice::console::ConsoleIo>::new(&env, addr)
        });
        Ok(ConsoleIo(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::console::ConsolePort`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::console::ConsolePort>(),
        read: |obj| {
            let this = obj.cast::<ConsoleIo>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::console::ConsolePort> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomAudioInputImpl = <crate::Selected as super::Implementations>::RoomAudioInput;

/// `_core.RoomAudioInput`: the room's microphone leg, taken as itself wherever the host passes
/// an audio input.
#[pyclass(frozen, name = "RoomAudioInput", module = "livekit.agents._core")]
pub struct RoomAudioInput(pub(crate) ::std::sync::Arc<RoomAudioInputImpl>);

#[pymethods]
impl RoomAudioInput {
    #[getter]
    fn reading<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <RoomAudioInputImpl as ::lk_agents_abi::voice::room_owner::RoomAudioInput>::reading(
                &self.0,
            )
        });
        Ok(out
            .map(|value| Ok::<_, ::pyo3::PyErr>(crate::python::str_to_py(py, &value)))
            .transpose()?)
    }

    #[pyo3(name = "next_frame")]
    fn read_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <RoomAudioInputImpl as ::lk_agents_abi::voice::room_owner::RoomAudioInput>::read_frame(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        Ok::<_, ::pyo3::PyErr>(crate::conversions::audio_frame_to_py(py, value)?)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <RoomAudioInputImpl as ::lk_agents_abi::voice::room_owner::RoomAudioInput>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::AudioInput`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::AudioInput>(),
        read: |obj| {
            let this = obj.cast::<RoomAudioInput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomVideoInputImpl = <crate::Selected as super::Implementations>::RoomVideoInput;

/// `_core.RoomVideoInput`: the room's video leg, taken as itself wherever the host passes a video
/// input.
#[pyclass(frozen, name = "RoomVideoInput", module = "livekit.agents._core")]
pub struct RoomVideoInput(pub(crate) ::std::sync::Arc<RoomVideoInputImpl>);

#[pymethods]
impl RoomVideoInput {
    #[getter]
    fn reading<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::host::check_pid()?;
        let out = py.detach(|| {
            <RoomVideoInputImpl as ::lk_agents_abi::voice::room_owner::RoomVideoInput>::reading(
                &self.0,
            )
        });
        Ok(out
            .map(|value| Ok::<_, ::pyo3::PyErr>(crate::python::str_to_py(py, &value)))
            .transpose()?)
    }

    #[pyo3(name = "next_frame")]
    fn read_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <RoomVideoInputImpl as ::lk_agents_abi::voice::room_owner::RoomVideoInput>::read_frame(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        Ok::<_, ::pyo3::PyErr>(crate::conversions::media_ref_to_py(py, value)?)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <RoomVideoInputImpl as ::lk_agents_abi::voice::room_owner::RoomVideoInput>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::VideoInput`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::VideoInput>(),
        read: |obj| {
            let this = obj.cast::<RoomVideoInput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type CoreRoomSessionImpl = <crate::Selected as super::Implementations>::RoomSession;

/// `CoreRoomSession`: what a session's room I/O is built from, handed to the host and taken
/// back as itself.
#[pyclass(frozen, name = "_core_RoomSession", module = "livekit.agents._core")]
pub struct CoreRoomSession(pub(crate) ::std::sync::Arc<CoreRoomSessionImpl>);

#[pymethods]
impl CoreRoomSession {}

// Wherever the host passes a `::lk_agents_abi::voice::room_owner::SessionRoom`, an instance of this class is itself.
inventory::submit! {
    crate::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(),
        read: |obj| {
            let this = obj.cast::<CoreRoomSession>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::room_owner::SessionRoom> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomRuntimeImpl = <crate::Selected as super::Implementations>::RoomRuntime;

fn _extract_room_runtime_wait_for_participant_kinds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<::std::vec::Vec<i32>>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::python::number_from_py::<i32>(&item, "kinds")
                })
                .collect::<PyResult<Vec<_>>>()
        }))
        .map(Some)
    }
}

fn _extract_room_runtime_wait_for_track_publication_kinds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<::std::vec::Vec<i32>>> {
    if obj.is_none() {
        Ok(None)
    } else {
        (obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::python::number_from_py::<i32>(&item, "kinds")
                })
                .collect::<PyResult<Vec<_>>>()
        }))
        .map(Some)
    }
}

#[pyfunction]
#[pyo3(signature = (room))]
pub fn _core_create_room_session(
    py: Python<'_>,
    room: Option<Bound<'_, PyAny>>,
) -> PyResult<super::voice::CoreRoomSession> {
    crate::host::check_pid()?;
    let env = crate::process::env();
    let room = room.map(|v| crate::generated::voice::RtcRoomProxy::from_host(&v)).transpose()?;
    let out = py.detach(|| {
        <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::create_room_session(
            &env, room,
        )
    });
    Ok(super::voice::CoreRoomSession(::std::sync::Arc::new(out)))
}

/// Wait for the supplied room's connection, avatar participant and subscribed video track.
#[pyfunction]
#[pyo3(signature = (room, identity))]
pub fn wait_avatar_join<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    identity: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let room = crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "room",
                ": expected a native room session"
            ))
        })?;
    crate::host::future_into_py(py, async move {
        let out =
            <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_avatar_join(
                room, identity,
            )
            .await
            .map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(
                    &e,
                )
            })?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

#[pyfunction]
#[pyo3(signature = (room, identity, kinds, include_local))]
pub fn wait_for_participant<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    identity: ::std::option::Option<::std::string::String>,
    #[pyo3(from_py_with = _extract_room_runtime_wait_for_participant_kinds)]
    kinds: ::std::option::Option<::std::vec::Vec<i32>>,
    include_local: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let room = crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "room",
                ": expected a native room session"
            ))
        })?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_participant(room, identity, kinds, include_local).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| {
            ::pyo3::IntoPyObjectExt::into_py_any(
                out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::python::str_to_py(py, &value)))
                    .transpose()?,
                py,
            )
        })
    })
}

#[pyfunction]
#[pyo3(signature = (room, name))]
pub fn wait_for_agent<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    name: ::std::option::Option<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let room = crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "room",
                ": expected a native room session"
            ))
        })?;
    crate::host::future_into_py(py, async move {
        let out =
            <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_agent(
                room, name,
            )
            .await
            .map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(
                    &e,
                )
            })?;
        Python::attach(|py| {
            ::pyo3::IntoPyObjectExt::into_py_any(crate::python::str_to_py(py, &out), py)
        })
    })
}

#[pyfunction]
#[pyo3(signature = (room, identity, kinds, include_local, subscribed))]
pub fn wait_for_track_publication<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    identity: ::std::option::Option<::std::string::String>,
    #[pyo3(from_py_with = _extract_room_runtime_wait_for_track_publication_kinds)]
    kinds: ::std::option::Option<::std::vec::Vec<i32>>,
    include_local: bool,
    subscribed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::host::check_pid()?;
    let room = crate::python::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(&room)
        .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "room",
                ": expected a native room session"
            ))
        })?;
    crate::host::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_track_publication(room, identity, kinds, include_local, subscribed).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::python::HostException>::to_py_err(&e))?;
        Python::attach(|py| {
            ::pyo3::IntoPyObjectExt::into_py_any(
                {
                    let (x0, x1) = out;
                    ::pyo3::types::PyTuple::new(
                        py,
                        [
                            match x0 {
                                None => py.None().into_bound(py),
                                Some(x) => crate::python::str_to_py(py, &x),
                            },
                            crate::python::str_to_py(py, &x1),
                        ],
                    )?
                    .into_any()
                },
                py,
            )
        })
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(ClearBufferListenerProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::voice::avatar::ClearBufferListener for ClearBufferListenerProxy {
    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "ClearBufferListener._core_clear_buffer",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_clear_buffer")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AvatarAudioReceiverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::avatar::AvatarAudioReceiver for AvatarAudioReceiverProxy {
    fn open_audio(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarAudioReceiver._core_open_audio",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_open_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn next_audio(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            (
                u64,
                bool,
                ::std::option::Option<::lk_agents_abi::voice::io::AvatarPlaybackSegment>,
                ::std::option::Option<::lk_agents_abi::utils::video::AudioReceiverFrame>,
            ),
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarAudioReceiver._core_next_audio",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "AvatarAudioReceiver._core_next_audio"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((crate::python::number_from_py::<u64>(&item_0, "AvatarAudioReceiver._core_next_audio"))?, (item_1.extract::<bool>())?, (if item_2.is_none() { Ok(None) } else { (crate::generated::records::io_events::avatar_playback_segment_from_py(&item_2)).map(Some) })?, (if item_3.is_none() { Ok(None) } else { (crate::generated::records::avatar::audio_receiver_frame_from_py(&item_3)).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn notify_playback_started(
        &self,
        segment: ::lk_agents_abi::voice::io::AvatarPlaybackSegment,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarAudioReceiver._core_notify_playback_started",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::io_events::avatar_playback_segment_to_py(
                        py,
                        segment_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_notify_playback_started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn notify_playback_finished(
        &self,
        segment: ::lk_agents_abi::voice::io::AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarAudioReceiver._core_notify_playback_finished",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    crate::generated::records::io_events::avatar_playback_segment_to_py(
                        py,
                        segment_arg,
                    )?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_notify_playback_finished")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn aclose(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarAudioReceiver._core_aclose",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AvatarVideoGeneratorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::avatar::AvatarVideoGenerator for AvatarVideoGeneratorProxy {
    fn push_audio(
        &self,
        frame: ::std::option::Option<::lk_agents_abi::utils::video::MediaFrame>,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarVideoGenerator._core_push_audio",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [match frame_arg {
                    None => py.None().into_bound(py),
                    Some(x) => crate::generated::records::avatar::media_frame_to_py(py, x)?,
                }];
                Ok((
                    host.bind(py).getattr("_core_push_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn clear_buffer(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarVideoGenerator._core_clear_buffer",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_clear_buffer")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn open_frames(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarVideoGenerator._core_open_frames",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_open_frames")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn next_frame(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarVideoGenerator._core_next_frame",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::avatar::avatar_frame_from_py(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AvatarMediaProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::avatar::AvatarMedia for AvatarMediaProxy {
    fn publish(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarMedia._core_publish",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_publish")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn push(
        &self,
        frame: ::lk_agents_abi::utils::video::MediaFrame,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarMedia._core_push",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::avatar::media_frame_to_py(py, frame_arg)?];
                Ok((host.bind(py).getattr("_core_push")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn aclose(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AvatarMedia._core_aclose",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_aclose")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::background_audio::FrameStream>>
    {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(FrameStreamProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::voice::background_audio::FrameStream for FrameStreamProxy {
    fn next(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::AudioFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "FrameStream._core_next",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_next")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::conversions::audio_frame_from_py(&o)).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn cancel_pending(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "FrameStream._core_cancel_pending",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_cancel_pending")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn close(&self) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "FrameStream._core_close",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            if let Err(e) = started.await {
                log::warn!("FrameStream._core_close failed: {e}");
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let view = crate::host::call(
            &obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?,
            (obj,),
            None,
        )?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(PlaybackListenerProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::voice::io::PlaybackListener for PlaybackListenerProxy {
    fn on_playback_started(
        &self,
        segment: ::lk_agents_abi::voice::io::PlaybackSegment,
        created_at: f64,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let created_at_arg = created_at;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "PlaybackListener._core_on_playback_started",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_playback_started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_playback_progressed(
        &self,
        segment: ::lk_agents_abi::voice::io::PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let started_at_arg = started_at;
        let offset_arg = offset;
        let duration_arg = duration;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "PlaybackListener._core_on_playback_progressed",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [
                    crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(started_at_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(offset_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(duration_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_playback_progressed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_playback_finished(
        &self,
        segment: ::lk_agents_abi::voice::io::PlaybackSegment,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: ::std::option::Option<::std::string::String>,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let playback_position_arg = playback_position;
        let interrupted_arg = interrupted;
        let synchronized_transcript_arg = synchronized_transcript;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "PlaybackListener._core_on_playback_finished",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [
                    crate::generated::records::io_events::playback_segment_to_py(py, segment_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(synchronized_transcript_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_playback_finished")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let view = crate::host::call(
            &obj.py().import("livekit.agents._core_view")?.getattr("_AudioInput")?,
            (obj,),
            None,
        )?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(AudioInputProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            label,
        })
    }
}

impl ::lk_agents_abi::voice::io::AudioInput for AudioInputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn next_frame(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::AudioFrame>,
            ::lk_agents_abi::voice::io::IoError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AudioInput._core_next_frame",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() {
                    Ok(None)
                } else {
                    (crate::conversions::audio_frame_from_py(&o)).map(Some)
                })
                .map_err(|e| {
                    <::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioInput._core_on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioInput._core_on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let view = crate::host::call(
            &obj.py().import("livekit.agents._core_view")?.getattr("_VideoInput")?,
            (obj,),
            None,
        )?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(VideoInputProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            label,
        })
    }
}

impl ::lk_agents_abi::voice::io::VideoInput for VideoInputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn next_frame(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::voice::rtc::VideoFrameRef>,
            ::lk_agents_abi::voice::io::IoError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "VideoInput._core_next_frame",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() {
                    Ok(None)
                } else {
                    (crate::generated::records::rtc_wire::video_frame_ref_from_py(&o)).map(Some)
                })
                .map_err(|e| {
                    <::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "VideoInput._core_on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "VideoInput._core_on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
    tracker: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackTracker>,
    capabilities: ::lk_agents_abi::voice::io::AudioOutputCapabilities,
    sample_rate: ::std::option::Option<u32>,
}

impl AudioOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::AudioOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let view = crate::host::call(
            &obj.py().import("livekit.agents._core_view")?.getattr("_AudioOutput")?,
            (obj,),
            None,
        )?;
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
            crate::python::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(&v)
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "AudioOutput._core_tracker",
                        ": expected a `_core.PlaybackTracker`"
                    ))
                })?
        };
        let capabilities = {
            let v = crate::host::read(obj, "_core_capabilities")?;
            crate::generated::records::rtc_wire::audio_output_capabilities_from_py(&v)?
        };
        let sample_rate = {
            let v = crate::host::read(obj, "_core_sample_rate")?;
            if v.is_none() {
                Ok(None)
            } else {
                (crate::python::number_from_py::<u32>(&v, "AudioOutput._core_sample_rate"))
                    .map(Some)
            }?
        };
        Ok(AudioOutputProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            label,
            tracker,
            capabilities,
            sample_rate,
        })
    }
}

impl ::lk_agents_abi::voice::io::AudioOutput for AudioOutputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn tracker(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackTracker> {
        &self.tracker
    }

    fn capabilities(&self) -> ::lk_agents_abi::voice::io::AudioOutputCapabilities {
        ::std::clone::Clone::clone(&self.capabilities)
    }

    fn begin_segment(
        &self,
        segment: ::lk_agents_abi::voice::io::PlaybackSegment,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::io_events::playback_segment_to_py(
                        py,
                        segment_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_begin_segment")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn sample_rate(&self) -> ::std::option::Option<u32> {
        ::std::clone::Clone::clone(&self.sample_rate)
    }

    fn capture<'a>(
        &'a self,
        frame: &::lk_agents_abi::AudioFrame,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AudioOutput._core_capture",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::audio_frame_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py).getattr("_core_capture")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn flush_segment(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput._core_flush_segment",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_flush_segment")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput._core_clear_buffer",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_clear_buffer")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput._core_on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput._core_on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn pause(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput._core_pause",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_pause")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
    }

    fn resume(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput._core_resume",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_resume")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let view = crate::host::call(
            &obj.py().import("livekit.agents._core_view")?.getattr("_TextOutput")?,
            (obj,),
            None,
        )?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let label = {
            let v = crate::host::read(obj, "_core_label")?;
            v.extract::<::std::string::String>()?
        };
        Ok(TextOutputProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            label,
        })
    }
}

impl ::lk_agents_abi::voice::io::TextOutput for TextOutputProxy {
    fn label(&self) -> &str {
        &self.label
    }

    fn capture_timed<'a>(
        &'a self,
        text: &::lk_agents_abi::TimedString,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = ::std::clone::Clone::clone(text);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TextOutput._core_capture_timed",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::timed_string_to_py(py, text_arg)?];
                Ok((
                    host.bind(py).getattr("_core_capture_timed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn capture_text<'a>(
        &'a self,
        text: &str,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TextOutput._core_capture_text",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_capture_text")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::voice::io::IoError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn flush(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TextOutput._core_flush",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_flush")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TextOutput._core_on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TextOutput._core_on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.

/// [`::lk_agents_abi::voice::nodes::SessionNodes`] over a host `livekit.agents.voice.agent_session._SessionNodes`.
#[derive(Clone)]
pub struct SessionNodesProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl SessionNodesProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::SessionNodes>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::nodes::SessionNodes>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::SessionNodes>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::nodes::SessionNodes>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::nodes::SessionNodes>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(SessionNodesProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::nodes::SessionNodes for SessionNodesProxy {
    fn llm<'a>(
        &'a self,
        req: &::lk_agents_abi::ChatRequest,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<
                dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::voice::nodes::NodeChunk>,
            >,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let req_arg = ::std::clone::Clone::clone(req);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionNodes.llm",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::Bound::new(
                    py,
                    crate::generated::llm::ChatRequest(::std::sync::Arc::new(req_arg)),
                )?
                .into_any()];
                Ok((host.bind(py).getattr("llm")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.llm: the call captured no host context")));
                };
                let o = out.bind(py);
                let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(crate::generated::records::agent_nodes::node_chunk_from_py(&o)?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::voice::nodes::NodeChunk>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn tts<'a>(
        &'a self,
        text: ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::std::string::String>>,
        calls: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text;
        let calls_arg = calls;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionNodes.tts",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::Bound::new(py, NodeText(::lk_agents_runtime::net::source::NativeSource::new(text_arg)))?.into_any(), <dyn ::lk_agents_abi::voice::nodes::ProviderCalls as super::NativeToHost>::to_host(py, calls_arg)?];
                Ok((host.bind(py).getattr("tts")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.tts: the call captured no host context")));
                };
                let o = out.bind(py);
                let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(crate::generated::records::tts::synthesized_audio_from_py(&o)?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SynthesizedAudio>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn stt<'a>(
        &'a self,
        audio: ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SpeechEvent>>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let audio_arg = audio;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionNodes.stt",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::Bound::new(
                    py,
                    NodeAudio(::lk_agents_runtime::net::source::NativeSource::new(audio_arg)),
                )?
                .into_any()];
                Ok((host.bind(py).getattr("stt")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.stt: the call captured no host context")));
                };
                let o = out.bind(py);
                let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(crate::generated::records::stt::speech_event_from_py(&o)?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::SpeechEvent>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn transcription<'a>(
        &'a self,
        text: ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::TimedString>>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::TimedString>>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionNodes.transcription",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::Bound::new(
                    py,
                    NodeTranscript(::lk_agents_runtime::net::source::NativeSource::new(text_arg)),
                )?
                .into_any()];
                Ok((
                    host.bind(py).getattr("transcription")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.transcription: the call captured no host context")));
                };
                let o = out.bind(py);
                let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(crate::conversions::timed_string_from_py(&o)?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::TimedString>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn realtime_audio<'a>(
        &'a self,
        audio: ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let audio_arg = audio;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionNodes.realtime_audio",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::Bound::new(
                    py,
                    NodeAudio(::lk_agents_runtime::net::source::NativeSource::new(audio_arg)),
                )?
                .into_any()];
                Ok((
                    host.bind(py).getattr("realtime_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.realtime_audio: the call captured no host context")));
                };
                let o = out.bind(py);
                let out = (crate::host::stream::HostStream::open(py, o, context, |o| Ok(crate::conversions::audio_frame_from_py(&o)?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::AudioFrame>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

/// One stream of `::std::string::String` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "NodeText", module = "livekit.agents._core")]
pub struct NodeText(::lk_agents_runtime::net::source::NativeSource<::std::string::String>);

#[pymethods]
impl NodeText {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| {
            ::pyo3::IntoPyObjectExt::into_py_any(item, py)
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

/// One stream of `::lk_agents_abi::AudioFrame` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "NodeAudio", module = "livekit.agents._core")]
pub struct NodeAudio(::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::AudioFrame>);

#[pymethods]
impl NodeAudio {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| {
            Ok(crate::conversions::audio_frame_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

/// One stream of `::lk_agents_abi::TimedString` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "NodeTranscript", module = "livekit.agents._core")]
pub struct NodeTranscript(
    ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::TimedString>,
);

#[pymethods]
impl NodeTranscript {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::recv(&self.0, py, |py, item| {
            Ok(crate::conversions::timed_string_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::host::check_pid()?;
        crate::host::source::aclose(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::RtcRoom`] over a host `livekit.agents.voice.room_io.Room`.
#[derive(Clone)]
pub struct RtcRoomProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    local: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>,
    event_bus: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>,
}

impl RtcRoomProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        if let Some(native) = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        if let Some(native) = crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        let local = {
            let v = crate::host::read(obj, "_core_local")?;
            crate::generated::voice::LocalParticipantProxy::from_host(&v)?
        };
        let event_bus = {
            let v = crate::host::read(obj, "_core_event_bus")?;
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(&v).ok_or_else(
                || {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "RtcRoom._core_event_bus",
                        ": expected a _core.RoomEventBus"
                    ))
                },
            )?
        };
        Ok(RtcRoomProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), local, event_bus })
    }
}

impl ::lk_agents_abi::voice::rtc::RtcRoom for RtcRoomProxy {
    fn name(
        &self,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_name")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn connection_state(
        &self,
    ) -> ::std::result::Result<
        ::lk_agents_abi::voice::rtc::ConnectionState,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_connection_state")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::python::str_from_py(&o)).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn local(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant> {
        &self.local
    }

    fn participant(
        &self,
        id: &::lk_agents_abi::voice::room_io::ParticipantIdentity,
    ) -> ::std::result::Result<
        ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let id_arg = ::std::clone::Clone::clone(id);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::python::str_to_py(py, &id_arg)];
                Ok((
                    host.bind(py).getattr("_core_participant")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (if o.is_none() {
                Ok(None)
            } else {
                (crate::generated::voice::RemoteParticipantProxy::from_host(&o)).map(Some)
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn remote_participants(
        &self,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_remote_participants")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::generated::voice::RemoteParticipantProxy::from_host(&item)
                    })
                    .collect::<PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn event_bus(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RoomEventSink> {
        &self.event_bus
    }

    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let streams_arg = streams;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?, <dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress as super::NativeToHost>::to_host(py, streams_arg)?];
                Ok((
                    host.bind(py).getattr("_core_register_byte_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn unregister_byte_stream_handler(
        &self,
        topic: &str,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_unregister_byte_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn register_text_stream_handler(
        &self,
        topic: &str,
        streams: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamIngress>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let streams_arg = streams;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?, <dyn ::lk_agents_abi::voice::rtc::TextStreamIngress as super::NativeToHost>::to_host(py, streams_arg)?];
                Ok((
                    host.bind(py).getattr("_core_register_text_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn unregister_text_stream_handler(
        &self,
        topic: &str,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_unregister_text_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(LocalParticipantProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::LocalParticipant for LocalParticipantProxy {
    fn identity(
        &self,
    ) -> ::std::result::Result<
        ::lk_agents_abi::voice::room_io::ParticipantIdentity,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_identity")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::python::str_from_py(&o)).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn kind(
        &self,
    ) -> ::std::result::Result<
        ::lk_agents_abi::voice::room_io::ParticipantKind,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_kind")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::python::str_from_py(&o)).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn track_publications(
        &self,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_track_publications")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::generated::voice::TrackPublicationProxy::from_host(&item)
                    })
                    .collect::<PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn publish_audio_track<'a>(
        &'a self,
        opts: ::lk_agents_abi::voice::room_io::TrackPublishOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            (
                ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>,
                ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>,
            ),
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_publish_audio_track",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::track_publish_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_publish_audio_track")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "LocalParticipant._core_publish_audio_track"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::generated::voice::AudioSourceProxy::from_host(&item_0))?, (crate::generated::voice::TrackPublicationProxy::from_host(&item_1))?,)) })()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn unpublish_track<'a>(
        &'a self,
        sid: &::lk_agents_abi::voice::rtc::TrackSid,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let sid_arg = ::std::clone::Clone::clone(sid);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_unpublish_track",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::python::str_to_py(py, &sid_arg)];
                Ok((
                    host.bind(py).getattr("_core_unpublish_track")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn set_attributes<'a>(
        &'a self,
        attrs: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let attrs_arg = attrs;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_set_attributes",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(attrs_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_set_attributes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn publish_transcription<'a>(
        &'a self,
        transcription: ::lk_agents_abi::voice::rtc::Transcription,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let transcription_arg = transcription;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_publish_transcription",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::transcription_to_py(
                        py,
                        transcription_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_publish_transcription")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn stream_text<'a>(
        &'a self,
        opts: ::lk_agents_abi::voice::rtc::StreamTextOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_stream_text",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::stream_text_options_to_py(py, opts_arg)?];
                Ok((
                    host.bind(py).getattr("_core_stream_text")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::voice::TextStreamWriterProxy::owned(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn stream_bytes<'a>(
        &'a self,
        opts: ::lk_agents_abi::voice::rtc::StreamBytesOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_stream_bytes",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::stream_bytes_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_stream_bytes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::voice::ByteStreamWriterProxy::owned(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn perform_rpc<'a>(
        &'a self,
        destination: ::lk_agents_abi::voice::room_io::ParticipantIdentity,
        method: ::std::string::String,
        payload: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(u32, ::std::string::String), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let destination_arg = destination;
        let method_arg = method;
        let payload_arg = payload;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_perform_rpc",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    crate::python::str_to_py(py, &destination_arg),
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(payload_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_perform_rpc")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "LocalParticipant._core_perform_rpc"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::python::number_from_py::<u32>(&item_0, "LocalParticipant._core_perform_rpc"))?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn register_rpc_method(
        &self,
        method: ::std::string::String,
        handler: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RpcCallback>,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let method_arg = method;
        let handler_arg = handler;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?,
                    <dyn ::lk_agents_abi::voice::rtc::RpcCallback as super::NativeToHost>::to_host(
                        py,
                        handler_arg,
                    )?,
                ];
                Ok((
                    host.bind(py).getattr("_core_register_rpc_method")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn unregister_rpc_method(
        &self,
        method: ::std::string::String,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let method_arg = method;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_unregister_rpc_method")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn publish_dtmf<'a>(
        &'a self,
        code: u32,
        digit: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let code_arg = code;
        let digit_arg = digit;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "LocalParticipant._core_publish_dtmf",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(code_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(digit_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_publish_dtmf")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
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

impl ::lk_agents_abi::voice::rtc::ByteStreamWriter for ByteStreamWriterProxy {
    fn id(&self) -> &str {
        &self.id
    }

    fn write<'a>(
        &'a self,
        data: &[u8],
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let data_arg = data.to_vec();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "ByteStreamWriter._core_write",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(data_arg, py)?];
                Ok((host.bind(py).getattr("_core_write")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn close(
        self: ::std::boxed::Box<Self>,
    ) -> ::lk_agents_abi::BoxFuture<
        'static,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "ByteStreamWriter._core_close",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(TextStreamWriterProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::TextStreamWriter for TextStreamWriterProxy {
    fn write<'a>(
        &'a self,
        text: &str,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TextStreamWriter._core_write",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("_core_write")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn close(
        self: ::std::boxed::Box<Self>,
        attributes: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    ) -> ::lk_agents_abi::BoxFuture<
        'static,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let attributes_arg = attributes;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TextStreamWriter._core_close",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(attributes_arg, py)?];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AudioSourceProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::AudioSource for AudioSourceProxy {
    fn capture<'a>(
        &'a self,
        frame: &::lk_agents_abi::AudioFrame,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AudioSource._core_capture",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::audio_frame_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py).getattr("_core_capture")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn clear_queue(&self) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_clear_queue")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn queued_duration(
        &self,
    ) -> ::std::result::Result<::std::time::Duration, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_queued_duration")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::python::number_from_py::<f64>(
                &o,
                "AudioSource._core_queued_duration",
            )
            .and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!(
                        "{}: {e}",
                        "AudioSource._core_queued_duration"
                    ))
                })
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn wait_for_playout<'a>(
        &'a self,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::voice::rtc::PlaybackInfo,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AudioSource._core_wait_for_playout",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_wait_for_playout")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::generated::records::rtc_wire::playback_info_from_py(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn playback_info(
        &self,
    ) -> ::std::result::Result<
        ::lk_agents_abi::voice::rtc::PlaybackInfo,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_playback_info")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::records::rtc_wire::playback_info_from_py(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn close<'a>(
        &'a self,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AudioSource._core_close",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        let sid = {
            let v = crate::host::read(obj, "_core_sid")?;
            v.extract::<::std::string::String>()?
        };
        let identity = {
            let v = crate::host::read(obj, "_core_identity")?;
            crate::python::str_from_py(&v)?
        };
        let kind = {
            let v = crate::host::read(obj, "_core_kind")?;
            crate::python::str_from_py(&v)?
        };
        Ok(RemoteParticipantProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            sid,
            identity,
            kind,
        })
    }
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
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_active")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn attributes(
        &self,
    ) -> ::std::result::Result<
        ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_attributes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(
                |dict| {
                    dict.iter()
                        .map(|(key, item)| {
                            Ok((
                                key.extract::<::std::string::String>()?,
                                item.extract::<::std::string::String>()?,
                            ))
                        })
                        .collect::<PyResult<::std::collections::BTreeMap<_, _>>>()
                },
            ))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn track_publications(
        &self,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_track_publications")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::generated::voice::TrackPublicationProxy::from_host(&item)
                    })
                    .collect::<PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TrackPublication>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let sid = {
            let v = crate::host::read(obj, "_core_sid")?;
            crate::python::str_from_py(&v)?
        };
        let name = {
            let v = crate::host::read(obj, "_core_name")?;
            v.extract::<::std::string::String>()?
        };
        let source = {
            let v = crate::host::read(obj, "_core_source")?;
            crate::python::str_from_py(&v)?
        };
        let kind = {
            let v = crate::host::read(obj, "_core_kind")?;
            crate::python::number_from_py::<i32>(&v, "TrackPublication._core_kind")?
        };
        let audio_features = {
            let v = crate::host::read(obj, "_core_audio_features")?;
            v.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::python::str_from_py(&item)
                    })
                    .collect::<PyResult<Vec<_>>>()
            })?
        };
        Ok(TrackPublicationProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            sid,
            name,
            source,
            kind,
            audio_features,
        })
    }
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
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_is_subscribed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn audio_features(&self) -> ::std::vec::Vec<::lk_agents_abi::voice::rtc::AudioTrackFeature> {
        ::std::clone::Clone::clone(&self.audio_features)
    }

    fn set_subscribed<'a>(
        &'a self,
        subscribed: bool,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let subscribed_arg = subscribed;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TrackPublication._core_set_subscribed",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(subscribed_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_set_subscribed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn wait_for_subscription<'a>(
        &'a self,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TrackPublication._core_wait_for_subscription",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_wait_for_subscription")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn open_audio_stream(
        &self,
        opts: ::lk_agents_abi::voice::rtc::AudioStreamOptions,
    ) -> ::std::result::Result<
        ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::audio_stream_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_open_audio_stream")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (crate::generated::voice::AudioInputStreamProxy::from_host(&o)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn open_video_stream(
        &self,
    ) -> ::std::result::Result<
        ::std::boxed::Box<
            dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::voice::rtc::VideoFrameRef>,
        >,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_open_video_stream")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("TrackPublication._core_open_video_stream: the call captured no host context")));
            };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::open(
                py,
                o,
                context,
                |o| Ok(crate::generated::records::rtc_wire::video_frame_ref_from_py(&o)?),
                ::lk_agents_runtime::process::host_clock(),
            )
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<
                        dyn ::lk_agents_abi::StreamSource<
                                ::lk_agents_abi::voice::rtc::VideoFrameRef,
                            >,
                    >
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(AudioInputStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::AudioInputStream for AudioInputStreamProxy {
    fn frames(
        &self,
    ) -> ::std::result::Result<
        ::std::boxed::Box<
            dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::voice::rtc::AudioStreamItem>,
        >,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("_core_frames")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AudioInputStream._core_frames: the call captured no host context")));
            };
            let o = out.bind(py);
            let out = (crate::host::stream::HostStream::open(
                py,
                o,
                context,
                |o| Ok(crate::generated::records::rtc_wire::audio_stream_item_from_py(&o)?),
                ::lk_agents_runtime::process::host_clock(),
            )
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<
                        dyn ::lk_agents_abi::StreamSource<
                                ::lk_agents_abi::voice::rtc::AudioStreamItem,
                            >,
                    >
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn process_audio<'a>(
        &'a self,
        frames: ::std::vec::Vec<::lk_agents_abi::AudioFrame>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::voice::rtc::AudioStreamItem>,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frames_arg = frames;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "AudioInputStream._core_process_audio",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::types::PyList::new(
                    py,
                    frames_arg
                        .into_iter()
                        .map(|x| PyResult::Ok(crate::conversions::audio_frame_to_py(py, x)?))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any()];
                Ok((
                    host.bind(py).getattr("_core_process_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::rtc_wire::audio_stream_item_from_py(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::python::FromHostErr>::from_host(py, &e))?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>>
    {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(SessionHooksProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::voice::session_effects::SessionHooks for SessionHooksProxy {
    fn on_enter<'a>(
        &'a self,
        agent: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_on_enter",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::host_ref_to_py(py, agent_arg)?];
                Ok((
                    host.bind(py).getattr("_core_on_enter")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn on_exit<'a>(
        &'a self,
        agent: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_on_exit",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::host_ref_to_py(py, agent_arg)?];
                Ok((
                    host.bind(py).getattr("_core_on_exit")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn close_toolsets<'a>(
        &'a self,
        agent: Option<&::lk_agents_abi::HostRef>,
    ) -> ::lk_agents_abi::BoxFuture<'a, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = agent.map(::std::borrow::ToOwned::to_owned);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_close_toolsets",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [match agent_arg {
                    None => py.None().into_bound(py),
                    Some(x) => crate::conversions::host_ref_to_py(py, x)?,
                }];
                Ok((
                    host.bind(py).getattr("_core_close_toolsets")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            if let Err(e) = started.await {
                log::warn!("SessionHooks._core_close_toolsets failed: {e}");
            }
        })
    }

    fn setup_tools<'a>(
        &'a self,
        tools: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::agent::FunctionToolInfo>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let tools_arg = ::std::clone::Clone::clone(tools);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_setup_tools",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::host_ref_to_py(py, tools_arg)?];
                Ok((
                    host.bind(py).getattr("_core_setup_tools")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::agent_run::function_tool_info_from_py(&item)
                        })
                        .collect::<PyResult<Vec<_>>>()
                }))
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn closing(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_closing",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_closing")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn on_user_turn_completed<'a>(
        &'a self,
        agent: &::lk_agents_abi::HostRef,
        chat_ctx: ::std::vec::Vec<::lk_agents_abi::ChatItem>,
        new_transcript: &str,
        transcript_confidence: f64,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            (
                ::std::option::Option<::std::string::String>,
                ::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>,
                bool,
            ),
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let chat_ctx_arg = chat_ctx;
        let new_transcript_arg = new_transcript.to_owned();
        let transcript_confidence_arg = transcript_confidence;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_on_user_turn_completed",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [
                    crate::conversions::host_ref_to_py(py, agent_arg)?,
                    ::pyo3::types::PyList::new(
                        py,
                        chat_ctx_arg
                            .into_iter()
                            .map(|x| {
                                PyResult::Ok(
                                    crate::generated::records::chat_items::chat_item_to_py(py, x)?,
                                )
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(new_transcript_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript_confidence_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_user_turn_completed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> {
                    let tuple = o.cast::<::pyo3::types::PyTuple>()?;
                    if tuple.len() != 3 {
                        return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                            "{}: expected a tuple of 3 items",
                            "SessionHooks._core_on_user_turn_completed"
                        )));
                    }
                    let item_0 = tuple.get_item(0)?;
                    let item_1 = tuple.get_item(1)?;
                    let item_2 = tuple.get_item(2)?;
                    Ok((
                        (if item_0.is_none() {
                            Ok(None)
                        } else {
                            (item_0.extract::<::std::string::String>()).map(Some)
                        })?,
                        (if item_1.is_none() {
                            Ok(None)
                        } else {
                            (item_1.try_iter().and_then(|items| {
                                items
                                    .map(|item| {
                                        let item = item?;
                                        crate::generated::records::chat_items::chat_item_from_py(
                                            &item,
                                        )
                                    })
                                    .collect::<PyResult<Vec<_>>>()
                            }))
                            .map(Some)
                        })?,
                        (item_2.extract::<bool>())?,
                    ))
                })())
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn on_user_turn_exceeded<'a>(
        &'a self,
        agent: &::lk_agents_abi::HostRef,
        ev: &::lk_agents_abi::voice::events::UserTurnExceededEvent,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let ev_arg = ::std::clone::Clone::clone(ev);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_on_user_turn_exceeded",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::conversions::host_ref_to_py(py, agent_arg)?,
                    crate::generated::records::events::user_turn_exceeded_event_to_py(py, ev_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_user_turn_exceeded")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn function_tools_executed(
        &self,
        function_calls: ::std::vec::Vec<::lk_agents_abi::ChatItem>,
        function_call_outputs: ::std::vec::Vec<::lk_agents_abi::ChatItem>,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<::std::vec::Vec<bool>, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let function_calls_arg = function_calls;
        let function_call_outputs_arg = function_call_outputs;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_function_tools_executed",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::types::PyList::new(
                        py,
                        function_calls_arg
                            .into_iter()
                            .map(|x| {
                                PyResult::Ok(
                                    crate::generated::records::chat_items::chat_item_to_py(py, x)?,
                                )
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    ::pyo3::types::PyList::new(
                        py,
                        function_call_outputs_arg
                            .into_iter()
                            .map(|x| {
                                PyResult::Ok(
                                    crate::generated::records::chat_items::chat_item_to_py(py, x)?,
                                )
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                ];
                Ok((
                    host.bind(py).getattr("_core_function_tools_executed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            item.extract::<bool>()
                        })
                        .collect::<PyResult<Vec<_>>>()
                }))
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn filler_source<'a>(
        &'a self,
        source: &::lk_agents_abi::HostRef,
        step: u32,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::voice::session_effects::FillerReply,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let source_arg = ::std::clone::Clone::clone(source);
        let step_arg = step;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "SessionHooks._core_filler_source",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::conversions::host_ref_to_py(py, source_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(step_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_filler_source")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::conversions::filler_reply_from_py(&o)).map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn hook_failed(
        &self,
        hook: &str,
        error: ::std::option::Option<::lk_agents_abi::errors::CapturedError>,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let hook_arg = hook.to_owned();
        let error_arg = error;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "SessionHooks._core_hook_failed",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(hook_arg, py)?,
                    match error_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::conversions::captured_error_to_py(py, x)?,
                    },
                ];
                Ok((
                    host.bind(py).getattr("_core_hook_failed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>>
    {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>>
    {
        if let Some(native) =
            crate::python::native::<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>>
    {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        Ok(SessionEventsProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::voice::session_effects::SessionEvents for SessionEventsProxy {
    fn event(&self, wire: ::lk_agents_abi::voice::events::HostWire) {
        let host = ::std::sync::Arc::clone(&self.host);
        let wire_arg = wire;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "SessionEvents._core_event",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::session_wire::host_wire_to_py(py, wire_arg)?];
                Ok((host.bind(py).getattr("_core_event")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
    }

    fn error_message(
        &self,
        error: &::lk_agents_abi::voice::events::SessionErrorWire,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = ::std::clone::Clone::clone(error);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::host::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::session_wire::session_error_wire_to_py(
                        py, error_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_error_message")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
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
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<
        ::std::sync::Arc<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>,
    > {
        if let Some(native) = crate::python::native::<
            dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<
        ::std::sync::Arc<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>,
    > {
        if let Some(native) = crate::python::native::<
            dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<
        ::std::boxed::Box<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>,
    > {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::host::capture_context(obj.py())?,
        };
        let io_state = {
            let v = crate::host::read(obj, "_core_io_state")?;
            crate::generated::records::workflows::io_enable_state_from_py(&v)?
        };
        Ok(TransferHostProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            io_state,
        })
    }
}

impl ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost for TransferHostProxy {
    fn start_hold(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::HostRef>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_start_hold",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_start_hold")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (if o.is_none() {
                    Ok(None)
                } else {
                    (crate::conversions::host_ref_from_py(&o)).map(Some)
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn stop_hold(&self, hold: ::lk_agents_abi::HostRef) {
        let host = ::std::sync::Arc::clone(&self.host);
        let hold_arg = hold;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TransferHost._core_stop_hold",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::conversions::host_ref_to_py(py, hold_arg)?];
                Ok((
                    host.bind(py).getattr("_core_stop_hold")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn io_state(&self) -> ::lk_agents_abi::voice::workflows::IoEnableState {
        ::std::clone::Clone::clone(&self.io_state)
    }

    fn set_io(&self, state: ::lk_agents_abi::voice::workflows::IoEnableState) {
        let host = ::std::sync::Arc::clone(&self.host);
        let state_arg = state;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TransferHost._core_set_io",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::workflows::io_enable_state_to_py(py, state_arg)?];
                Ok((host.bind(py).getattr("_core_set_io")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
    }

    fn connect_human(
        &self,
        room_name: ::std::string::String,
        identity: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let room_name_arg = room_name;
        let identity_arg = identity;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_connect_human",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(room_name_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_connect_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (crate::python::native::<
                    dyn ::lk_agents_abi::voice::room_owner::SessionRoom,
                >(&o)
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "TransferHost._core_connect_human",
                        ": expected a native room session"
                    ))
                }))
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn start_human(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_start_human",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_start_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn close_human(&self, drain: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let drain_arg = drain;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TransferHost._core_close_human",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(drain_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_close_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
    }

    fn disconnect_human(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_disconnect_human",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_disconnect_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn create_sip_participant(
        &self,
        request: ::lk_agents_abi::voice::workflows::SipParticipantRequest,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_create_sip_participant",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::workflows::sip_participant_request_to_py(
                        py,
                        request_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_create_sip_participant")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn connect_twilio_call(
        &self,
        room_name: ::std::string::String,
        identity: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let room_name_arg = room_name;
        let identity_arg = identity;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_connect_twilio_call",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(room_name_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_connect_twilio_call")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = (o.extract::<::std::string::String>()).map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn http_post(
        &self,
        url: ::std::string::String,
        headers: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
        form: ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(u16, ::std::string::String), ::lk_agents_abi::errors::HostError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let url_arg = url;
        let headers_arg = headers;
        let form_arg = form;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_http_post",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(url_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(headers_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(form_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_http_post")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> {
                    let tuple = o.cast::<::pyo3::types::PyTuple>()?;
                    if tuple.len() != 2 {
                        return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                            "{}: expected a tuple of 2 items",
                            "TransferHost._core_http_post"
                        )));
                    }
                    let item_0 = tuple.get_item(0)?;
                    let item_1 = tuple.get_item(1)?;
                    Ok((
                        (crate::python::number_from_py::<u16>(
                            &item_0,
                            "TransferHost._core_http_post",
                        ))?,
                        (item_1.extract::<::std::string::String>())?,
                    ))
                })())
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn move_participant(
        &self,
        room: ::std::string::String,
        identity: ::std::string::String,
        destination: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let room_arg = room;
        let identity_arg = identity;
        let destination_arg = destination;
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_move_participant",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(room_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(destination_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_move_participant")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }

    fn delete_caller_room(
        &self,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::host::Entry,
            "TransferHost._core_delete_caller_room",
            &self.context,
            &[],
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_delete_caller_room")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.bind(py);
                let out = ({
                    let _ = o;
                    ::pyo3::PyResult::Ok(())
                })
                .map_err(|e| {
                    <::lk_agents_abi::errors::HostError as crate::python::FromHostErr>::from_host(
                        py, &e,
                    )
                })?;
                Ok(out)
            })
        })
    }
}

/// Each `ALL` array's names, indexed by wire value, for the host adapter's reads of SDK ints:
/// `(connection states, disconnect reasons, participant kinds, track sources, audio features)`.
#[pyfunction]
pub fn rtc_wire_names(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::voice::rtc::rtc_wire_names(), py)
}

/// How many frames a `video_stream` queue holds before it evicts the oldest. Tiny on purpose: each
/// held frame pins a decoded host frame, and the newest frame is the one worth keeping. A host
/// video stream is opened with the same capacity, so both sides drop the oldest alike.
#[pyfunction]
pub fn video_queue_frames(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    crate::host::check_pid()?;
    ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES, py)
}

impl crate::python::HostException for ::lk_agents_abi::voice::avatar::RunnerError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closed => crate::python::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Failed(inner) => crate::python::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::avatar::ReceiverEnd {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closed => ::pyo3::exceptions::PyStopAsyncIteration::new_err(
                ::std::string::ToString::to_string(self),
            ),
            Self::BadStream(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::background_audio::PlaybackError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::LoopingStream(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::io::InvalidPlaybackTime {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::io::IoError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::rtc::RtcError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::python::HostException::to_py_err(inner),
            Self::Stream(inner) => crate::python::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::SessionRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closed => crate::python::host_exception(
                "livekit.agents._core",
                "SessionClosing",
                ::std::string::ToString::to_string(self),
            ),
            Self::LoopEnded => crate::python::host_exception(
                "livekit.agents._core",
                "SessionClosing",
                ::std::string::ToString::to_string(self),
            ),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::OpenRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::CommandRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Session(..) => ::pyo3::exceptions::PyRuntimeError::new_err(
                ::std::string::ToString::to_string(self),
            ),
            _ => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::NotInterruptible {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::StartFailed {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::python::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::UnknownCloseReason {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::NotAnImage {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::VoiceTaskEnded {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closing => crate::python::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Cancelled => crate::python::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Failed(..) => crate::python::host_exception(
                "livekit.agents.voice.agent_session",
                "VoiceTaskFailed",
                ::std::string::ToString::to_string(self),
            ),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

::pyo3::create_exception!(
    _core,
    SessionClosing,
    ::pyo3::exceptions::PyRuntimeError,
    "`wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing."
);
impl crate::python::HostException for ::lk_agents_abi::voice::session::IdleRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        SessionClosing::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::session::CommitSuperseded {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::python::host_exception(
            "asyncio",
            "CancelledError",
            ::std::string::ToString::to_string(self),
        )
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::speech_handle::SpeechRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::workflows::DateOfBirthRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::workflows::CaptureRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::python::host_exception(
            "livekit.agents.llm.tool_context",
            "ToolError",
            ::std::string::ToString::to_string(self),
        )
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::python::HostException for ::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::NoDigits => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::UnknownKey(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            _ => crate::python::host_exception(
                "livekit.agents.llm.tool_context",
                "ToolError",
                ::std::string::ToString::to_string(self),
            ),
        }
    }
}

impl crate::python::HostException
    for ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused
{
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::python::HostException::to_py_err(inner),
            _ => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
        }
    }
}

impl crate::python::HostException
    for ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed
{
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::NotRunning => ::pyo3::exceptions::PyRuntimeError::new_err(
                ::std::string::ToString::to_string(self),
            ),
            Self::UnknownTool(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::Host(inner) => crate::python::HostException::to_py_err(inner),
            _ => crate::python::host_exception(
                "livekit.agents.llm.tool_context",
                "ToolError",
                ::std::string::ToString::to_string(self),
            ),
        }
    }
}
