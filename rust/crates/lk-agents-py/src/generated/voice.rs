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
type AmdFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AmdFunctions;

/// These are the reference's, not tuning knobs: the default models are what the benchmark
/// ran on.
#[pyfunction]
pub fn amd_constants<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let out =
        py.detach(<AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_constants);
    crate::generated::records::amd::amd_constants_to_py(py, out)
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
            crate::py_values::number_from_py::<f64>(v.as_any(), "provided").and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "provided"))
                })
            })
        })
        .transpose()?;
    let activity = activity
        .map(|v| {
            crate::py_values::number_from_py::<f64>(v.as_any(), "activity").and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "activity"))
                })
            })
        })
        .transpose()?;
    let out = py.detach(|| <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_resolve_max_endpointing_delay(provided, activity));
    ::pyo3::IntoPyObjectExt::into_bound_py_any(out.as_secs_f64(), py)
}

/// `AMDPredictionEvent.is_human`.
#[pyfunction]
#[pyo3(signature = (prediction))]
pub fn amd_prediction_is_human(py: Python<'_>, prediction: Bound<'_, PyAny>) -> PyResult<bool> {
    let prediction =
        crate::generated::records::amd::amd_prediction_event_from_py(prediction.as_any())?;
    let out = py.detach(|| {
        <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_prediction_is_human(
            prediction,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// `AMDPredictionEvent.is_machine`.
#[pyfunction]
#[pyo3(signature = (prediction))]
pub fn amd_prediction_is_machine(py: Python<'_>, prediction: Bound<'_, PyAny>) -> PyResult<bool> {
    let prediction =
        crate::generated::records::amd::amd_prediction_event_from_py(prediction.as_any())?;
    let out = py.detach(|| {
        <AmdFunctionsImpl as ::lk_agents_abi::voice::amd::AmdFunctions>::amd_prediction_is_machine(
            prediction,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type DataStreamAudioOutputImpl =
    <crate::generated::implementations::Selected as super::Implementations>::DataStreamAudioOutput;

fn _extract_data_stream_audio_output_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DataStreamAudioOutputNewSampleRateInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")).map(Some)
    }
}
type DataStreamAudioOutputNewSampleRateInput = ::std::option::Option<u32>;

fn _extract_data_stream_audio_output_new_wait_remote_track(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DataStreamAudioOutputNewWaitRemoteTrackInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<i32>(obj.as_any(), "wait_remote_track")).map(Some)
    }
}
type DataStreamAudioOutputNewWaitRemoteTrackInput = ::std::option::Option<i32>;

fn _extract_data_stream_audio_output_new_clear_buffer_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DataStreamAudioOutputNewClearBufferTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "clear_buffer_timeout")).map(Some)
    }
}
type DataStreamAudioOutputNewClearBufferTimeoutInput = ::std::option::Option<f64>;
type DataStreamAudioOutputSampleRateOutput = ::std::option::Option<u32>;

/// The agent's native audio sink, sending one byte stream per avatar speech segment.
#[pyclass(frozen, subclass, name = "DataStreamAudioOutput", module = "livekit.agents._core")]
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
        sample_rate: DataStreamAudioOutputNewSampleRateInput,
        #[pyo3(from_py_with = _extract_data_stream_audio_output_new_wait_remote_track)]
        wait_remote_track: DataStreamAudioOutputNewWaitRemoteTrackInput,
        #[pyo3(from_py_with = _extract_data_stream_audio_output_new_clear_buffer_timeout)]
        clear_buffer_timeout: DataStreamAudioOutputNewClearBufferTimeoutInput,
        wait_playback_start: bool,
    ) -> PyResult<DataStreamAudioOutput> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let room = crate::generated::voice::RtcRoomProxy::from_host(room.as_any())?;
        let out = py.detach(|| <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::new(&env, room, destination_identity, sample_rate, wait_remote_track, clear_buffer_timeout, wait_playback_start)).map_err(|e: ::lk_agents_abi::voice::avatar::AvatarOutputError| <::lk_agents_abi::voice::avatar::AvatarOutputError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(DataStreamAudioOutput(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn label(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::label(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn sample_rate(&self, py: Python<'_>) -> PyResult<DataStreamAudioOutputSampleRateOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::sample_rate(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Stop admitting audio, settle startup and pending writes, close streams and retire RPC routes.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <DataStreamAudioOutputImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioOutput>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::AudioOutput`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::AudioOutput>(),
        read: |obj| {
            let this = obj.cast::<DataStreamAudioOutput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioOutput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type DataStreamAudioReceiverImpl = <crate::generated::implementations::Selected as super::Implementations>::DataStreamAudioReceiver;

fn _extract_data_stream_audio_receiver_new_frame_size_ms(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DataStreamAudioReceiverNewFrameSizeMsInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "frame_size_ms")
}
type DataStreamAudioReceiverNewFrameSizeMsInput = u32;

fn _extract_data_stream_audio_receiver_new_rpc_max_retries(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DataStreamAudioReceiverNewRpcMaxRetriesInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "rpc_max_retries")
}
type DataStreamAudioReceiverNewRpcMaxRetriesInput = u32;

fn _extract_data_stream_audio_receiver_notify_playback_finished_playback_position(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DataStreamAudioReceiverNotifyPlaybackFinishedPlaybackPositionInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "playback_position")
}
type DataStreamAudioReceiverNotifyPlaybackFinishedPlaybackPositionInput = f64;

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
        #[pyo3(from_py_with = _extract_data_stream_audio_receiver_new_frame_size_ms)] frame_size_ms: DataStreamAudioReceiverNewFrameSizeMsInput,
        #[pyo3(from_py_with = _extract_data_stream_audio_receiver_new_rpc_max_retries)]
        rpc_max_retries: DataStreamAudioReceiverNewRpcMaxRetriesInput,
        on_clear_buffer: Option<Bound<'_, PyAny>>,
    ) -> PyResult<DataStreamAudioReceiver> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let room = crate::generated::voice::RtcRoomProxy::from_host(room.as_any())?;
        let on_clear_buffer = on_clear_buffer
            .map(|v| crate::generated::voice::ClearBufferListenerProxy::from_host(v.as_any()))
            .transpose()?;
        let out = py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::new(&env, room, sender_identity, frame_size_ms, rpc_max_retries, on_clear_buffer)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(DataStreamAudioReceiver(::std::sync::Arc::new(out)))
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn next_audio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::next_audio(&this, )
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
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
        crate::py_calls::check_pid()?;
        let segment = crate::generated::records::io_events::avatar_playback_segment_from_py(
            segment.as_any(),
        )?;
        py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::notify_playback_started(&self.0, segment)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (segment, playback_position, interrupted))]
    fn notify_playback_finished(
        &self,
        py: Python<'_>,
        segment: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_data_stream_audio_receiver_notify_playback_finished_playback_position)] playback_position: DataStreamAudioReceiverNotifyPlaybackFinishedPlaybackPositionInput,
        interrupted: bool,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let segment = crate::generated::records::io_events::avatar_playback_segment_from_py(
            segment.as_any(),
        )?;
        py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::notify_playback_finished(&self.0, segment, playback_position, interrupted)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn aclose(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <DataStreamAudioReceiverImpl as ::lk_agents_abi::voice::avatar::DataStreamAudioReceiver>::aclose(&self.0)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::avatar::AvatarAudioReceiver`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(),
        read: |obj| {
            let this = obj.cast::<DataStreamAudioReceiver>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
type AvatarRunnerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AvatarRunner;

fn _extract_avatar_runner_new_queue_size_ms(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AvatarRunnerNewQueueSizeMsInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "queue_size_ms")
}
type AvatarRunnerNewQueueSizeMsInput = u32;

fn _extract_avatar_runner_clear_buffer_epoch(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AvatarRunnerClearBufferEpochInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "epoch")
}
type AvatarRunnerClearBufferEpochInput = u64;

/// The avatar worker's native queue owner. Its decisions remain in RunnerCore.
#[pyclass(frozen, name = "AvatarRunner", module = "livekit.agents._core")]
pub struct AvatarRunner(pub(crate) ::std::sync::Arc<AvatarRunnerImpl>);

#[pymethods]
impl AvatarRunner {
    #[new]
    #[pyo3(signature = (room, *, receiver, generator, options, queue_size_ms, lazy_publish))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        receiver: Bound<'_, PyAny>,
        generator: Bound<'_, PyAny>,
        options: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_avatar_runner_new_queue_size_ms)]
        queue_size_ms: AvatarRunnerNewQueueSizeMsInput,
        lazy_publish: bool,
    ) -> PyResult<AvatarRunner> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let room =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
                room.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "room",
                    ": expected a native room session"
                ))
            })?;
        let receiver =
            crate::generated::voice::AvatarAudioReceiverProxy::from_host(receiver.as_any())?;
        let generator =
            crate::generated::voice::AvatarVideoGeneratorProxy::from_host(generator.as_any())?;
        let options =
            crate::generated::records::avatar_options::avatar_options_from_py(options.as_any())?;
        let out = py.detach(|| <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::new(&env, room, receiver, generator, options, queue_size_ms, lazy_publish)).map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(AvatarRunner(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn av_sync(&self, py: Python<'_>) -> PyResult<super::voice::AVSynchronizer> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::av_sync(&self.0)
        });
        ::pyo3::PyResult::Ok(super::voice::AVSynchronizer(out))
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::start(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn wait_for_complete<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::wait_for_complete(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[pyo3(signature = (epoch))]
    fn clear_buffer(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_avatar_runner_clear_buffer_epoch)]
        epoch: AvatarRunnerClearBufferEpochInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::clear_buffer(
                &self.0, epoch,
            )
        });
        Ok(())
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AvatarRunnerImpl as ::lk_agents_abi::voice::avatar::AvatarRunner>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::avatar::RunnerError| <::lk_agents_abi::voice::avatar::RunnerError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar_queue.rs.
type CoreQueueAudioImpl =
    <crate::generated::implementations::Selected as super::Implementations>::QueueAudio;

fn _extract_queue_audio_capture_epoch(
    obj: &Bound<'_, PyAny>,
) -> PyResult<QueueAudioCaptureEpochInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "epoch")
}
type QueueAudioCaptureEpochInput = u64;

fn _extract_queue_audio_playback_finished_playback_position(
    obj: &Bound<'_, PyAny>,
) -> PyResult<QueueAudioPlaybackFinishedPlaybackPositionInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "playback_position")
}
type QueueAudioPlaybackFinishedPlaybackPositionInput = f64;

#[pyclass(frozen, name = "_core_QueueAudio", module = "livekit.agents._core")]
pub struct CoreQueueAudio(pub(crate) ::std::sync::Arc<CoreQueueAudioImpl>);

#[pymethods]
impl CoreQueueAudio {
    #[new]
    #[pyo3(signature = (wait_playback_start))]
    fn new(py: Python<'_>, wait_playback_start: bool) -> PyResult<CoreQueueAudio> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::new(
                wait_playback_start,
            )
        });
        ::pyo3::PyResult::Ok(CoreQueueAudio(::std::sync::Arc::new(out)))
    }

    fn check_open(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::check_open(&self.0)).map_err(|e: ::lk_agents_abi::voice::avatar_queue::QueueClosed| <::lk_agents_abi::voice::avatar_queue::QueueClosed as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (frame, segment, epoch))]
    fn capture<'py>(
        &self,
        py: Python<'py>,
        frame: Bound<'_, PyAny>,
        segment: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_queue_audio_capture_epoch)]
        epoch: QueueAudioCaptureEpochInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let frame = crate::generated::records::rtc_frames::audio_frame_from_py(frame.as_any())?;
        let segment =
            crate::generated::records::io_events::playback_segment_from_py(segment.as_any())?;
        let out = py.detach(|| <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::capture(&self.0, frame, segment, epoch)).map_err(|e: ::lk_agents_abi::voice::avatar_queue::QueueClosed| <::lk_agents_abi::voice::avatar_queue::QueueClosed as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::generated::records::avatar_queue::queue_effects_to_py(py, out)
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::flush(&self.0)).map_err(|e: ::lk_agents_abi::voice::avatar_queue::QueueClosed| <::lk_agents_abi::voice::avatar_queue::QueueClosed as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn clear<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::clear(&self.0)
        });
        crate::generated::records::avatar_queue::queue_effects_to_py(py, out)
    }

    #[pyo3(signature = (playback_position, interrupted, segment))]
    fn playback_finished<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_queue_audio_playback_finished_playback_position)]
        playback_position: QueueAudioPlaybackFinishedPlaybackPositionInput,
        interrupted: bool,
        segment: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let segment = segment
            .map(|v| {
                crate::generated::records::io_events::avatar_playback_segment_from_py(v.as_any())
            })
            .transpose()?;
        let out = py.detach(|| <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::playback_finished(&self.0, playback_position, interrupted, segment));
        crate::generated::records::avatar_queue::queue_effects_to_py(py, out)
    }

    #[pyo3(signature = (segment))]
    fn playback_started<'py>(
        &self,
        py: Python<'py>,
        segment: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let segment = segment
            .map(|v| {
                crate::generated::records::io_events::avatar_playback_segment_from_py(v.as_any())
            })
            .transpose()?;
        let out = py.detach(|| <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::playback_started(&self.0, segment));
        crate::generated::records::avatar_queue::queue_effects_to_py(py, out)
    }

    fn next_audio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::next_audio(&this, )
                .await;
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

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreQueueAudioImpl as ::lk_agents_abi::voice::avatar_queue::QueueAudio>::close(&self.0)
        });
        crate::generated::records::avatar_queue::queue_effects_to_py(py, out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
type PlayStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::PlayState;

fn _extract_play_state_new_fade_out(obj: &Bound<'_, PyAny>) -> PyResult<PlayStateNewFadeOutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "fade_out")
}
type PlayStateNewFadeOutInput = f64;

/// A sound's monotone stop and completion handle.
#[pyclass(frozen, name = "PlayState", module = "livekit.agents._core")]
pub struct PlayState(pub(crate) ::std::sync::Arc<PlayStateImpl>);

#[pymethods]
impl PlayState {
    #[new]
    #[pyo3(signature = (fade_out = 0.0))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_play_state_new_fade_out)] fade_out: PlayStateNewFadeOutInput,
    ) -> PyResult<PlayState> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::new(fade_out)
        });
        ::pyo3::PyResult::Ok(PlayState(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::done(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn stop_requested(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::stop_requested(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    fn stop(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::stop(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(name = "mark_playout_done")]
    fn finish(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::finish(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(name = "wait_for_playout")]
    fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () =
                <PlayStateImpl as ::lk_agents_abi::voice::background_audio::PlayState>::wait(&this)
                    .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
type BackgroundAudioPlayerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::BackgroundAudioPlayer;

fn _extract_background_audio_player_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerNewSampleRateInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type BackgroundAudioPlayerNewSampleRateInput = u32;

fn _extract_background_audio_player_new_num_channels(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerNewNumChannelsInput> {
    crate::py_values::number_from_py::<u16>(obj.as_any(), "num_channels")
}
type BackgroundAudioPlayerNewNumChannelsInput = u16;

fn _extract_background_audio_player_new_blocksize(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerNewBlocksizeInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "blocksize")
}
type BackgroundAudioPlayerNewBlocksizeInput = usize;

fn _extract_background_audio_player_new_stream_timeout_ms(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerNewStreamTimeoutMsInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "stream_timeout_ms")
}
type BackgroundAudioPlayerNewStreamTimeoutMsInput = u64;

fn _extract_background_audio_player_start_draw(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerStartDrawInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "draw")
}
type BackgroundAudioPlayerStartDrawInput = f64;

fn _extract_background_audio_player_play_draw(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerPlayDrawInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "draw")
}
type BackgroundAudioPlayerPlayDrawInput = f64;

fn _extract_background_audio_player_on_agent_state_draw(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BackgroundAudioPlayerOnAgentStateDrawInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "draw")
}
type BackgroundAudioPlayerOnAgentStateDrawInput = f64;

/// Runtime-owned mixer and published background track.
#[pyclass(frozen, name = "BackgroundAudioPlayer", module = "livekit.agents._core")]
pub struct BackgroundAudioPlayer(pub(crate) ::std::sync::Arc<BackgroundAudioPlayerImpl>);

#[pymethods]
impl BackgroundAudioPlayer {
    #[new]
    #[pyo3(signature = (sample_rate = 48000, num_channels = 1, *, blocksize = 4800, stream_timeout_ms = 200))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_background_audio_player_new_sample_rate)]
        sample_rate: BackgroundAudioPlayerNewSampleRateInput,
        #[pyo3(from_py_with = _extract_background_audio_player_new_num_channels)]
        num_channels: BackgroundAudioPlayerNewNumChannelsInput,
        #[pyo3(from_py_with = _extract_background_audio_player_new_blocksize)]
        blocksize: BackgroundAudioPlayerNewBlocksizeInput,
        #[pyo3(from_py_with = _extract_background_audio_player_new_stream_timeout_ms)]
        stream_timeout_ms: BackgroundAudioPlayerNewStreamTimeoutMsInput,
    ) -> PyResult<BackgroundAudioPlayer> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::new(&env, sample_rate, num_channels, blocksize, stream_timeout_ms)).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(BackgroundAudioPlayer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (room, draw, *, ambient, choose, source = None))]
    fn start<'py>(
        &self,
        py: Python<'py>,
        room: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_background_audio_player_start_draw)]
        draw: BackgroundAudioPlayerStartDrawInput,
        ambient: Bound<'_, PyAny>,
        choose: bool,
        source: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let room =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
                room.as_any(),
            )
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
                    crate::generated::records::background_audio_wire::sound_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let source = source.map(|v| crate::py_values::str_from_py(v.as_any())).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::start(&this, room, draw, ambient, choose, source)
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(crate::py_values::str_to_py(py, &out), py)
            })
        })
    }

    #[pyo3(signature = (sounds, draw, *, looping = false, ambient = false, choose = true))]
    fn play(
        &self,
        py: Python<'_>,
        sounds: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_background_audio_player_play_draw)]
        draw: BackgroundAudioPlayerPlayDrawInput,
        looping: bool,
        ambient: bool,
        choose: bool,
    ) -> PyResult<super::voice::PlayState> {
        crate::py_calls::check_pid()?;
        let sounds = sounds.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::background_audio_wire::sound_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::play(&self.0, sounds, draw, looping, ambient, choose)).map_err(|e: ::lk_agents_abi::voice::background_audio::PlaybackError| <::lk_agents_abi::voice::background_audio::PlaybackError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::voice::PlayState(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (sounds, draw, thinking, *, choose = true))]
    fn on_agent_state(
        &self,
        py: Python<'_>,
        sounds: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_background_audio_player_on_agent_state_draw)]
        draw: BackgroundAudioPlayerOnAgentStateDrawInput,
        thinking: bool,
        choose: bool,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let sounds = sounds.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::background_audio_wire::sound_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::on_agent_state(&self.0, sounds, draw, thinking, choose)).map_err(|e: ::lk_agents_abi::voice::background_audio::PlaybackError| <::lk_agents_abi::voice::background_audio::PlaybackError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::closed(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <BackgroundAudioPlayerImpl as ::lk_agents_abi::voice::background_audio::BackgroundAudioPlayer>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
type PlaybackTrackerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::Playback;

/// `_core.PlaybackTracker`: the tracker a host sink reports to (`io.AudioOutput._tracker`),
/// shared with the session that reads it.
#[pyclass(frozen, name = "PlaybackTracker", module = "livekit.agents._core")]
pub struct PlaybackTracker(pub(crate) ::std::sync::Arc<PlaybackTrackerImpl>);

#[pymethods]
impl PlaybackTracker {
    #[new]
    fn new(py: Python<'_>) -> PyResult<PlaybackTracker> {
        crate::py_calls::check_pid()?;
        let out = py.detach(<PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::new);
        ::pyo3::PyResult::Ok(PlaybackTracker(::std::sync::Arc::new(out)))
    }

    fn note_flush(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::note_flush(&self.0)
        });
        Ok(())
    }

    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <PlaybackTrackerImpl as ::lk_agents_abi::voice::io::Playback>::reset(&self.0));
        Ok(())
    }

    /// `AudioOutput.wait_for_playout()`: the last event as `(playback_position, interrupted,
    /// synchronized_transcript)`.
    #[pyo3(name = "wait_for_playout")]
    fn wait_for_played<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
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
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(),
        read: |obj| {
            let this = obj.cast::<PlaybackTracker>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackTracker> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
type PlaybackReportsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::PlaybackReports;

fn _extract_playback_reports_started_created_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlaybackReportsStartedCreatedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "created_at")
}
type PlaybackReportsStartedCreatedAtInput = f64;

fn _extract_playback_reports_progressed_started_at(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlaybackReportsProgressedStartedAtInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "started_at")
}
type PlaybackReportsProgressedStartedAtInput = f64;

fn _extract_playback_reports_progressed_offset(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlaybackReportsProgressedOffsetInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "offset")
}
type PlaybackReportsProgressedOffsetInput = f64;

fn _extract_playback_reports_progressed_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlaybackReportsProgressedDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "duration")
}
type PlaybackReportsProgressedDurationInput = f64;

fn _extract_playback_reports_finished_playback_position(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlaybackReportsFinishedPlaybackPositionInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "playback_position")
}
type PlaybackReportsFinishedPlaybackPositionInput = f64;

#[pyfunction]
#[pyo3(signature = (tracker, segment))]
pub fn _core_playback_begin(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
) -> PyResult<bool> {
    crate::py_calls::check_pid()?;
    let tracker = crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(
        tracker.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "tracker",
            ": expected a `_core.PlaybackTracker`"
        ))
    })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(segment.as_any())?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::begin(
            tracker, segment,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker))]
pub fn _core_playback_current_segment<'py>(
    py: Python<'py>,
    tracker: Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    crate::py_calls::check_pid()?;
    let tracker = crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(
        tracker.as_any(),
    )
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
    out.map(|value| crate::generated::records::io_events::playback_segment_to_py(py, value))
        .transpose()
}

#[pyfunction]
#[pyo3(signature = (tracker))]
pub fn _core_playback_note_capture<'py>(
    py: Python<'py>,
    tracker: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let tracker = crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(
        tracker.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "tracker",
            ": expected a `_core.PlaybackTracker`"
        ))
    })?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::note_capture(tracker)
    });
    crate::generated::records::io_events::playback_segment_to_py(py, out)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, created_at))]
pub fn _core_playback_started(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_playback_reports_started_created_at)]
    created_at: PlaybackReportsStartedCreatedAtInput,
) -> PyResult<bool> {
    crate::py_calls::check_pid()?;
    let tracker = crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(
        tracker.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "tracker",
            ": expected a `_core.PlaybackTracker`"
        ))
    })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(segment.as_any())?;
    let out = py.detach(|| {
        <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::started(
            tracker, segment, created_at,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// Invalid durations are refused before updating playback or notifying observers.
#[pyfunction]
#[pyo3(signature = (tracker, segment, started_at, offset, duration))]
pub fn _core_playback_progressed(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_playback_reports_progressed_started_at)]
    started_at: PlaybackReportsProgressedStartedAtInput,
    #[pyo3(from_py_with = _extract_playback_reports_progressed_offset)]
    offset: PlaybackReportsProgressedOffsetInput,
    #[pyo3(from_py_with = _extract_playback_reports_progressed_duration)]
    duration: PlaybackReportsProgressedDurationInput,
) -> PyResult<bool> {
    crate::py_calls::check_pid()?;
    let tracker = crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(
        tracker.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "tracker",
            ": expected a `_core.PlaybackTracker`"
        ))
    })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(segment.as_any())?;
    let out = py.detach(|| <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::progressed(tracker, segment, started_at, offset, duration)).map_err(|e: ::lk_agents_abi::voice::io::InvalidPlaybackTime| <::lk_agents_abi::voice::io::InvalidPlaybackTime as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tracker, segment, playback_position, interrupted, synchronized_transcript))]
pub fn _core_playback_finished(
    py: Python<'_>,
    tracker: Bound<'_, PyAny>,
    segment: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_playback_reports_finished_playback_position)] playback_position: PlaybackReportsFinishedPlaybackPositionInput,
    interrupted: bool,
    synchronized_transcript: ::std::option::Option<::std::string::String>,
) -> PyResult<bool> {
    crate::py_calls::check_pid()?;
    let tracker = crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(
        tracker.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "tracker",
            ": expected a `_core.PlaybackTracker`"
        ))
    })?;
    let segment = crate::generated::records::io_events::playback_segment_from_py(segment.as_any())?;
    let out = py.detach(|| <PlaybackReportsImpl as ::lk_agents_abi::voice::io::PlaybackReports>::finished(tracker, segment, playback_position, interrupted, synchronized_transcript)).map_err(|e: ::lk_agents_abi::voice::io::InvalidPlaybackTime| <::lk_agents_abi::voice::io::InvalidPlaybackTime as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type AgentNodesImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AgentNodes;

/// `_core.AgentNodes(host)`: the session's registered nodes over the host's `_SessionNodes`, taken
/// in the binding of the agent they run for.
#[pyclass(frozen, name = "AgentNodes", module = "livekit.agents._core")]
pub struct AgentNodes(pub(crate) ::std::sync::Arc<AgentNodesImpl>);

#[pymethods]
impl AgentNodes {
    #[new]
    #[pyo3(signature = (host))]
    fn new(py: Python<'_>, host: Bound<'_, PyAny>) -> PyResult<AgentNodes> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::voice::SessionNodesProxy::from_host(host.as_any())?;
        let out = py.detach(|| {
            <AgentNodesImpl as ::lk_agents_abi::voice::nodes::AgentNodes>::new(&env, host)
        });
        ::pyo3::PyResult::Ok(AgentNodes(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::nodes::NodeFactory`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(),
        read: |obj| {
            let this = obj.cast::<AgentNodes>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::NodeFactory> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type CoreNodeCallsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::NodeCalls;

#[pyclass(frozen, name = "_core_NodeCalls", module = "livekit.agents._core")]
pub struct CoreNodeCalls(pub(crate) ::std::sync::Arc<CoreNodeCallsImpl>);

#[pymethods]
impl CoreNodeCalls {}

// Wherever the host passes a `::lk_agents_abi::voice::nodes::ProviderCalls`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>(),
        read: |obj| {
            let this = obj.cast::<CoreNodeCalls>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::ProviderCalls> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type LlmNodeChunksImpl =
    <crate::generated::implementations::Selected as super::Implementations>::LlmNodeChunks;

/// The default `llm_node`'s chunks.
#[pyclass(frozen, name = "LlmNodeChunks", module = "livekit.agents._core")]
pub struct LlmNodeChunks(pub(crate) ::std::sync::Arc<LlmNodeChunksImpl>);

#[pymethods]
impl LlmNodeChunks {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            match <LlmNodeChunksImpl as ::lk_agents_abi::voice::nodes::LlmNodeChunks>::next(&this).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::llm::chat_chunk_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type TtsNodeFramesImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TtsNodeFrames;

/// The default `tts_node`'s frames.
#[pyclass(frozen, name = "TtsNodeFrames", module = "livekit.agents._core")]
pub struct TtsNodeFrames(pub(crate) ::std::sync::Arc<TtsNodeFramesImpl>);

#[pymethods]
impl TtsNodeFrames {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            match <TtsNodeFramesImpl as ::lk_agents_abi::voice::nodes::TtsNodeFrames>::next(&this).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::agent_nodes::node_frame_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type SttNodeEventsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SttNodeEvents;

/// The default `stt_node`'s speech events.
#[pyclass(frozen, name = "SttNodeEvents", module = "livekit.agents._core")]
pub struct SttNodeEvents(pub(crate) ::std::sync::Arc<SttNodeEventsImpl>);

#[pymethods]
impl SttNodeEvents {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            match <SttNodeEventsImpl as ::lk_agents_abi::voice::nodes::SttNodeEvents>::next(&this).await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))? {
                Some(out) => Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::stt::speech_event_to_py(py, out)?, py)),
                None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/nodes.rs.
type DefaultNodesImpl =
    <crate::generated::implementations::Selected as super::Implementations>::DefaultNodes;

/// A completion of `chat_ctx` (its items, and the host's `llm.ChatContext` a host LLM is
/// handed) offering `tools` (their specs, and the host's `list[Tool]`). `conn_options`: the
/// session's, else the defaults. `calls`: the running agent node's, when it runs this one.
#[pyfunction]
#[pyo3(signature = (llm, chat_ctx, chat_ctx_host, *, tools, tools_host = None, tool_choice = None, conn_options = None, calls = None))]
pub fn llm_node(
    py: Python<'_>,
    llm: Bound<'_, PyAny>,
    chat_ctx: Bound<'_, PyAny>,
    chat_ctx_host: Bound<'_, PyAny>,
    tools: Bound<'_, PyAny>,
    tools_host: Option<Bound<'_, PyAny>>,
    tool_choice: Option<Bound<'_, PyAny>>,
    conn_options: Option<Bound<'_, PyAny>>,
    calls: Option<Bound<'_, PyAny>>,
) -> PyResult<super::voice::LlmNodeChunks> {
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    let llm = crate::generated::llm::LlmProxy::from_host(llm.as_any())?;
    let chat_ctx = chat_ctx.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::generated::records::chat_items::chat_item_from_py(item.as_any())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let chat_ctx_host = crate::py_objects::object_ref_from_py(chat_ctx_host.as_any())?;
    let tools = tools.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::generated::records::tool_options::tool_spec_from_py(item.as_any())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let tools_host =
        tools_host.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
    let tool_choice = tool_choice
        .map(|v| {
            crate::generated::records::from_representation::<
                ::lk_agents_abi::ToolChoice,
                ::serde_json::Value,
            >(crate::py_values::value_from_py(v.as_any(), "tool_choice")?)
        })
        .transpose()?;
    let conn_options = conn_options
        .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
        .transpose()?;
    let calls = calls
        .map(|v| {
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>(
                v.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "calls",
                    ": expected the calls of a registered tts_node"
                ))
            })
        })
        .transpose()?;
    let cx = crate::py_calls::entry_cx();
    let out = py
        .detach(|| {
            <DefaultNodesImpl as ::lk_agents_abi::voice::nodes::DefaultNodes>::llm_node(
                &env,
                llm,
                chat_ctx,
                chat_ctx_host,
                tools,
                tools_host,
                tool_choice,
                conn_options,
                calls,
                cx,
            )
        })
        .map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| {
            <::lk_agents_abi::ApiConnectOptionsError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
    ::pyo3::PyResult::Ok(super::voice::LlmNodeChunks(::std::sync::Arc::new(out)))
}

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
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    let tts = crate::generated::tts::TtsProxy::from_host(tts.as_any())?;
    let text = crate::py_calls::capture_context(text.py())
        .and_then(|context| {
            crate::py_streams::HostStream::open(
                text.py(),
                text.as_any(),
                &context,
                |o| o.extract::<::std::string::String>(),
                ::lk_agents_runtime::process::host_clock(),
            )
        })
        .map(|s| {
            ::std::boxed::Box::new(s)
                as ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::std::string::String>>
        })?;
    let conn_options = conn_options
        .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
        .transpose()?;
    let calls = calls
        .map(|v| {
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>(
                v.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "calls",
                    ": expected the calls of a registered tts_node"
                ))
            })
        })
        .transpose()?;
    let cx = crate::py_calls::entry_cx();
    let out = py
        .detach(|| {
            <DefaultNodesImpl as ::lk_agents_abi::voice::nodes::DefaultNodes>::tts_node(
                &env,
                tts,
                text,
                conn_options,
                calls,
                cx,
            )
        })
        .map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| {
            <::lk_agents_abi::ApiConnectOptionsError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
    ::pyo3::PyResult::Ok(super::voice::TtsNodeFrames(::std::sync::Arc::new(out)))
}

#[pyfunction]
#[pyo3(signature = (stt, audio, *, conn_options = None, vad = None, calls = None))]
pub fn stt_node(
    py: Python<'_>,
    stt: Bound<'_, PyAny>,
    audio: Bound<'_, PyAny>,
    conn_options: Option<Bound<'_, PyAny>>,
    vad: Option<Bound<'_, PyAny>>,
    calls: Option<Bound<'_, PyAny>>,
) -> PyResult<super::voice::SttNodeEvents> {
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    let stt = crate::generated::stt::SttProxy::from_host(stt.as_any())?;
    let audio = crate::py_calls::capture_context(audio.py())
        .and_then(|context| {
            crate::py_streams::HostStream::open(
                audio.py(),
                audio.as_any(),
                &context,
                |o| crate::generated::records::rtc_frames::audio_frame_from_py(o.as_any()),
                ::lk_agents_runtime::process::host_clock(),
            )
        })
        .map(|s| {
            ::std::boxed::Box::new(s)
                as ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::AudioFrame>>
        })?;
    let conn_options = conn_options
        .map(|v| crate::generated::records::connect::api_connect_options_from_py(v.as_any()))
        .transpose()?;
    let vad = vad.map(|v| crate::generated::vad::VadProxy::from_host(v.as_any())).transpose()?;
    let calls = calls
        .map(|v| {
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>(
                v.as_any(),
            )
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
            <DefaultNodesImpl as ::lk_agents_abi::voice::nodes::DefaultNodes>::stt_node(
                &env,
                stt,
                audio,
                conn_options,
                vad,
                calls,
            )
        })
        .map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| {
            <::lk_agents_abi::ApiConnectOptionsError as crate::py_errors::HostException>::to_py_err(
                &e,
            )
        })?;
    ::pyo3::PyResult::Ok(super::voice::SttNodeEvents(::std::sync::Arc::new(out)))
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/events.rs.
type RoomEventBusImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RoomEventBus;

/// The fan-out a host room creates and hands its SDK callbacks.
#[pyclass(frozen, name = "RoomEventBus", module = "livekit.agents._core")]
pub struct RoomEventBus(pub(crate) ::std::sync::Arc<RoomEventBusImpl>);

#[pymethods]
impl RoomEventBus {
    #[new]
    fn new(py: Python<'_>) -> PyResult<RoomEventBus> {
        crate::py_calls::check_pid()?;
        let out = py.detach(<RoomEventBusImpl as ::lk_agents_abi::voice::rtc::RoomEventBus>::new);
        ::pyo3::PyResult::Ok(RoomEventBus(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::RoomEventSink`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(),
        read: |obj| {
            let this = obj.cast::<RoomEventBus>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RoomEventSink> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/events.rs.
type RtcRuntimeImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RtcRuntime;

#[pyfunction]
#[pyo3(signature = (bus, participant, publication))]
pub fn _core_room_track_subscribed(
    py: Python<'_>,
    bus: Bound<'_, PyAny>,
    participant: Bound<'_, PyAny>,
    publication: Bound<'_, PyAny>,
) -> PyResult<()> {
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
    let publication =
        crate::generated::voice::TrackPublicationProxy::from_host(publication.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
    let publication =
        crate::generated::voice::TrackPublicationProxy::from_host(publication.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
    let publication =
        crate::generated::voice::TrackPublicationProxy::from_host(publication.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
    let publication =
        crate::generated::voice::TrackPublicationProxy::from_host(publication.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let publication =
        crate::generated::voice::TrackPublicationProxy::from_host(publication.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant =
        crate::generated::voice::RemoteParticipantProxy::from_host(participant.as_any())?;
    let reason = crate::py_values::str_from_py(reason.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let state = crate::py_values::str_from_py(state.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let reason = crate::py_values::str_from_py(reason.as_any())?;
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
    crate::py_calls::check_pid()?;
    let bus =
        crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(bus.as_any())
            .ok_or_else(|| {
            ::pyo3::exceptions::PyTypeError::new_err(concat!(
                "bus",
                ": expected a _core.RoomEventBus"
            ))
        })?;
    let participant = participant.map(|v| crate::py_values::str_from_py(v.as_any())).transpose()?;
    py.detach(|| {
        <RtcRuntimeImpl as ::lk_agents_abi::voice::rtc::events::RtcRuntime>::sip_dtmf_received(
            bus,
            digit,
            participant,
        )
    });
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/media.rs.
type AVSynchronizerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AVSynchronizer;

fn _extract_a_v_synchronizer_push_timestamp(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AVSynchronizerPushTimestampInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "timestamp")).map(Some)
    }
}
type AVSynchronizerPushTimestampInput = ::std::option::Option<f64>;

/// The native synchronizer view returned by an avatar runner. Both media types use the shared
/// RTC sources; no language-specific avatar media manager participates in its lifetime.
#[pyclass(frozen, name = "AVSynchronizer", module = "livekit.agents._core")]
pub struct AVSynchronizer(pub(crate) ::std::sync::Arc<AVSynchronizerImpl>);

#[pymethods]
impl AVSynchronizer {
    #[pyo3(signature = (frame, timestamp = None))]
    fn push<'py>(
        &self,
        py: Python<'py>,
        frame: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_a_v_synchronizer_push_timestamp)]
        timestamp: AVSynchronizerPushTimestampInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let frame = crate::generated::records::avatar::media_frame_from_py(frame.as_any())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::push(&this, frame, timestamp)
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn clear_queue<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::clear_queue(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn wait_for_playout<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::wait_for_playout(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::reset(
                &self.0,
            )
        });
        Ok(())
    }

    #[getter]
    fn actual_fps(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::actual_fps(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn last_video_time(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::last_video_time(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn last_audio_time(&self, py: Python<'_>) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::last_audio_time(&self.0)).map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AVSynchronizerImpl as ::lk_agents_abi::voice::rtc::media::AVSynchronizer>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/media.rs.
type PlayoutLedgerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::PlayoutLedger;

fn _extract_playout_ledger_add_pushed_samples_samples_per_channel(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlayoutLedgerAddPushedSamplesSamplesPerChannelInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "samples_per_channel")
}
type PlayoutLedgerAddPushedSamplesSamplesPerChannelInput = u32;

fn _extract_playout_ledger_add_pushed_samples_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<PlayoutLedgerAddPushedSamplesSampleRateInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type PlayoutLedgerAddPushedSamplesSampleRateInput = u32;

#[pyclass(frozen, name = "PlayoutLedger", module = "livekit.agents._core")]
pub struct PlayoutLedger(pub(crate) ::std::sync::Arc<PlayoutLedgerImpl>);

#[pymethods]
impl PlayoutLedger {
    #[new]
    fn new(py: Python<'_>) -> PyResult<PlayoutLedger> {
        crate::py_calls::check_pid()?;
        let out = py
            .detach(<PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::media::PlayoutLedger>::new);
        ::pyo3::PyResult::Ok(PlayoutLedger(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (samples_per_channel, sample_rate))]
    fn add_pushed_samples(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_playout_ledger_add_pushed_samples_samples_per_channel)]
        samples_per_channel: PlayoutLedgerAddPushedSamplesSamplesPerChannelInput,
        #[pyo3(from_py_with = _extract_playout_ledger_add_pushed_samples_sample_rate)] sample_rate: PlayoutLedgerAddPushedSamplesSampleRateInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::media::PlayoutLedger>::add_pushed_samples(&self.0, samples_per_channel, sample_rate));
        Ok(())
    }

    #[pyo3(signature = (duration))]
    fn add_discarded(&self, py: Python<'_>, duration: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let duration = crate::py_values::number_from_py::<f64>(duration.as_any(), "duration")
            .and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "duration"))
                })
            })?;
        py.detach(|| {
            <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::media::PlayoutLedger>::add_discarded(
                &self.0, duration,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (queued))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        queued: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let queued =
            crate::py_values::number_from_py::<f64>(queued.as_any(), "queued").and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!("{}: {e}", "queued"))
                })
            })?;
        let out = py.detach(|| {
            <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::media::PlayoutLedger>::snapshot(
                &self.0, queued,
            )
        });
        crate::generated::records::rtc_wire::playback_info_to_py(py, out)
    }

    fn settle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <PlayoutLedgerImpl as ::lk_agents_abi::voice::rtc::media::PlayoutLedger>::settle(
                &self.0,
            )
        });
        crate::generated::records::rtc_wire::playback_info_to_py(py, out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type ByteStreamSenderImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ByteStreamSender;

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
        crate::py_calls::check_pid()?;
        let info = crate::generated::records::stream_info::stream_info_from_py(info.as_any())?;
        let sender = crate::py_values::str_from_py(sender.as_any())?;
        let chunks = crate::py_calls::capture_context(chunks.py())
            .and_then(|context| {
                crate::py_streams::HostStream::open(
                    chunks.py(),
                    chunks.as_any(),
                    &context,
                    |o| {
                        if o.is_instance_of::<::pyo3::types::PyBytes>()
                            || o.is_instance_of::<::pyo3::types::PyByteArray>()
                            || o.is_instance_of::<::pyo3::types::PyMemoryView>()
                        {
                            crate::py_values::owned_bytes(o.as_any(), "chunks")
                        } else {
                            o.try_iter().and_then(|items| {
                                items
                                    .map(|item| {
                                        let item = item?;
                                        crate::py_values::number_from_py::<u8>(
                                            item.as_any(),
                                            "chunks",
                                        )
                                    })
                                    .collect::<PyResult<Vec<_>>>()
                            })
                        }
                    },
                    ::lk_agents_runtime::process::host_clock(),
                )
            })
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::std::vec::Vec<u8>>>
            })?;
        let out = py.detach(|| {
            <ByteStreamSenderImpl as ::lk_agents_abi::voice::rtc::ports::ByteStreamSender>::open(
                &self.0, info, sender, chunks,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::ByteStreamIngress`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress>(),
        read: |obj| {
            let this = obj.cast::<ByteStreamSender>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type TextStreamSenderImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TextStreamSender;

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
        crate::py_calls::check_pid()?;
        let info = crate::generated::records::stream_info::stream_info_from_py(info.as_any())?;
        let sender = crate::py_values::str_from_py(sender.as_any())?;
        let chunks = crate::py_calls::capture_context(chunks.py())
            .and_then(|context| {
                crate::py_streams::HostStream::open(
                    chunks.py(),
                    chunks.as_any(),
                    &context,
                    |o| o.extract::<::std::string::String>(),
                    ::lk_agents_runtime::process::host_clock(),
                )
            })
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::std::string::String>>
            })?;
        let out = py.detach(|| {
            <TextStreamSenderImpl as ::lk_agents_abi::voice::rtc::ports::TextStreamSender>::open(
                &self.0, info, sender, chunks,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::TextStreamIngress`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::TextStreamIngress>(),
        read: |obj| {
            let this = obj.cast::<TextStreamSender>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TextStreamIngress> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type RpcHandlerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RpcHandler;

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
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <RpcHandlerImpl as ::lk_agents_abi::voice::rtc::ports::RpcHandler>::respond(
                &self.0,
                caller_identity,
                payload,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::RpcCallback`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::RpcCallback>(),
        read: |obj| {
            let this = obj.cast::<RpcHandler>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RpcCallback> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/ports.rs.
type CoreRoomRpcRouterImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RoomRpcRouter;

/// Private room-adapter service; unrelated rooms never share mutable RPC routes.
#[pyclass(frozen, name = "_core_RoomRpcRouter", module = "livekit.agents._core")]
pub struct CoreRoomRpcRouter(pub(crate) ::std::sync::Arc<CoreRoomRpcRouterImpl>);

#[pymethods]
impl CoreRoomRpcRouter {
    #[new]
    fn new(py: Python<'_>) -> PyResult<CoreRoomRpcRouter> {
        crate::py_calls::check_pid()?;
        let out = py.detach(
            <CoreRoomRpcRouterImpl as ::lk_agents_abi::voice::rtc::ports::RoomRpcRouter>::new,
        );
        ::pyo3::PyResult::Ok(CoreRoomRpcRouter(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::rtc::ports::RpcRoutes`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::rtc::ports::RpcRoutes>(),
        read: |obj| {
            let this = obj.cast::<CoreRoomRpcRouter>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ports::RpcRoutes> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/run_view.rs.
type CoreVoiceRunImpl =
    <crate::generated::implementations::Selected as super::Implementations>::VoiceRun;
type VoiceRunQuestionOutput = ::std::option::Option<::std::string::String>;
type VoiceRunVerbatimOutput = ::std::option::Option<::std::string::String>;

#[pyclass(name = "_core_VoiceRun", module = "livekit.agents._core")]
pub struct CoreVoiceRun(pub(crate) CoreVoiceRunImpl);

#[pymethods]
impl CoreVoiceRun {
    #[new]
    fn new(py: Python<'_>) -> PyResult<CoreVoiceRun> {
        crate::py_calls::check_pid()?;
        let out = py.detach(<CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::new);
        ::pyo3::PyResult::Ok(CoreVoiceRun(out))
    }

    #[pyo3(signature = (item))]
    fn record(&mut self, item: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let item = crate::generated::records::chat_items::chat_item_from_py(item.as_any())?;
        <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::record(&mut self.0, item);
        Ok(())
    }

    #[pyo3(signature = (status, output, question))]
    fn settle(
        &mut self,
        status: ::std::string::String,
        output: Option<Bound<'_, PyAny>>,
        question: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let output =
            output.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
        let question = question
            .map(|v| crate::generated::records::session_wire::run_question_from_py(v.as_any()))
            .transpose()?;
        <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::settle(
            &mut self.0,
            status,
            output,
            question,
        );
        Ok(())
    }

    #[getter]
    fn status(&self) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::status(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn output<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::output(&self.0);
        out.map(|value| crate::generated::records::session_wire::voice_run_output_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn question(&self) -> PyResult<VoiceRunQuestionOutput> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::question(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn verbatim(&self) -> PyResult<VoiceRunVerbatimOutput> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::verbatim(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = <CoreVoiceRunImpl as ::lk_agents_abi::voice::run_view::VoiceRun>::items(&self.0);
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session.rs.
type SessionOptionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SessionOptions;

fn _extract_session_options_release_override_layer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionOptionsReleaseOverrideLayerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "layer")
}
type SessionOptionsReleaseOverrideLayerInput = u64;

fn _extract_session_options_override_providers_layer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionOptionsOverrideProvidersLayerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "layer")
}
type SessionOptionsOverrideProvidersLayerInput = u64;

fn _extract_session_options_release_providers_layer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionOptionsReleaseProvidersLayerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "layer")
}
type SessionOptionsReleaseProvidersLayerInput = u64;

/// `AgentSession(**kwargs)`'s options before the session holds them: the raw kwargs, read once,
/// and the session layer they became (plus every `update_options` so far).
#[pyclass(frozen, name = "SessionOptions", module = "livekit.agents._core")]
pub struct SessionOptions(pub(crate) ::std::sync::Arc<SessionOptionsImpl>);

#[pymethods]
impl SessionOptions {
    #[new]
    #[pyo3(signature = (kwargs))]
    fn new(py: Python<'_>, kwargs: Bound<'_, PyAny>) -> PyResult<SessionOptions> {
        crate::py_calls::check_pid()?;
        let kwargs =
            crate::generated::records::session_options::session_kwargs_from_py(kwargs.as_any())?;
        let out = py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::new(kwargs)
        });
        ::pyo3::PyResult::Ok(SessionOptions(::std::sync::Arc::new(out)))
    }

    /// `update_options(..)` before the session runs, on the same `merge` the driver uses.
    #[pyo3(signature = (update))]
    fn update(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let update =
            crate::generated::records::session_options::update_kwargs_from_py(update.as_any())?;
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
        crate::py_calls::check_pid()?;
        let update =
            crate::generated::records::session_options::update_kwargs_from_py(update.as_any())?;
        let out = py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::override_options(&self.0, update));
        ::pyo3::PyResult::Ok(out)
    }

    /// The `override_options` block ended before the session ran.
    #[pyo3(signature = (layer))]
    fn release_override(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_session_options_release_override_layer)]
        layer: SessionOptionsReleaseOverrideLayerInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::release_override(&self.0, layer));
        Ok(())
    }

    /// Update base provider references; returns only providers whose effective value changed.
    /// The runtime retains the host objects; the voice core resolves their resource identities.
    #[pyo3(signature = (providers))]
    fn update_providers<'py>(
        &self,
        py: Python<'py>,
        providers: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let providers = providers
            .cast::<::pyo3::types::PyDict>()
            .map_err(::pyo3::PyErr::from)
            .and_then(|dict| {
                dict.iter()
                    .map(|(key, item)| {
                        Ok((
                            key.extract::<::std::string::String>()?,
                            if item.is_none() {
                                Ok(None)
                            } else {
                                (crate::py_objects::object_ref_from_py(item.as_any())).map(Some)
                            }?,
                        ))
                    })
                    .collect::<PyResult<::std::collections::BTreeMap<_, _>>>()
            })?;
        let out = py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::update_providers(&self.0, providers));
        ::pyo3::PyResult::Ok({
            let d = ::pyo3::types::PyDict::new(py);
            for (k, x) in out {
                d.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                    match x {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_objects::object_ref_to_py(py, x)?,
                    },
                )?;
            }
            d.into_any()
        })
    }

    /// Attach providers to the same override id used by turn settings, before or after start.
    #[pyo3(signature = (layer, providers))]
    fn override_providers<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_session_options_override_providers_layer)]
        layer: SessionOptionsOverrideProvidersLayerInput,
        providers: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let providers = providers
            .cast::<::pyo3::types::PyDict>()
            .map_err(::pyo3::PyErr::from)
            .and_then(|dict| {
                dict.iter()
                    .map(|(key, item)| {
                        Ok((
                            key.extract::<::std::string::String>()?,
                            if item.is_none() {
                                Ok(None)
                            } else {
                                (crate::py_objects::object_ref_from_py(item.as_any())).map(Some)
                            }?,
                        ))
                    })
                    .collect::<PyResult<::std::collections::BTreeMap<_, _>>>()
            })?;
        let out = py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::override_providers(&self.0, layer, providers));
        ::pyo3::PyResult::Ok({
            let d = ::pyo3::types::PyDict::new(py);
            for (k, x) in out {
                d.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                    match x {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_objects::object_ref_to_py(py, x)?,
                    },
                )?;
            }
            d.into_any()
        })
    }

    /// Release provider references and return the providers to restore. Idempotent.
    #[pyo3(signature = (layer))]
    fn release_providers<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_session_options_release_providers_layer)]
        layer: SessionOptionsReleaseProvidersLayerInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::release_providers(&self.0, layer));
        ::pyo3::PyResult::Ok({
            let d = ::pyo3::types::PyDict::new(py);
            for (k, x) in out {
                d.set_item(
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                    match x {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_objects::object_ref_to_py(py, x)?,
                    },
                )?;
            }
            d.into_any()
        })
    }

    /// `session.stt` / `tts` / `vad`: the host provider `slot` runs on now, the topmost held
    /// override's, else the base's; `None` when the slot has none or it was removed.
    #[pyo3(signature = (slot))]
    fn provider<'py>(
        &self,
        py: Python<'py>,
        slot: ::std::string::String,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::provider(
                &self.0, slot,
            )
        });
        out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()
    }

    /// The resolved configuration over the host-supplied legs, as the dict `session.options` reads
    /// before `start()`.
    #[pyo3(signature = (legs))]
    fn resolved<'py>(
        &self,
        py: Python<'py>,
        legs: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let legs = crate::generated::records::session_options::session_legs_from_py(legs.as_any())?;
        let out = py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::resolved(
                &self.0, legs,
            )
        });
        crate::generated::records::session_options::resolved_session_options_to_py(py, out)
    }

    /// The turn detection mode the session's own settings name (the constructor's and every
    /// `update_options` before `start()`); `None` when automatic or a detector object.
    #[getter]
    fn turn_detection<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::turn_detection(
                &self.0,
            )
        });
        out.map(|value| ::pyo3::IntoPyObjectExt::into_bound_py_any(value.as_str(), py)).transpose()
    }

    /// The interruption mode `VoiceSession(turn_handling=)` set; `None` when the session picks.
    #[getter]
    fn interruption_mode<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <SessionOptionsImpl as ::lk_agents_abi::voice::session::SessionOptions>::interruption_mode(&self.0));
        out.map(|value| crate::generated::records::turn::interruption_mode_to_py(py, value))
            .transpose()
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session.rs.
type VoiceSessionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::VoiceSession;

fn _extract_voice_session_collect_dtmf_num_digits(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionCollectDtmfNumDigitsInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "num_digits")
}
type VoiceSessionCollectDtmfNumDigitsInput = usize;

fn _extract_voice_session_collect_dtmf_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionCollectDtmfTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timeout")
}
type VoiceSessionCollectDtmfTimeoutInput = f64;

fn _extract_voice_session_release_override_layer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionReleaseOverrideLayerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "layer")
}
type VoiceSessionReleaseOverrideLayerInput = u64;

fn _extract_voice_session_ivr_start_max_silence(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionIvrStartMaxSilenceInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "max_silence")
}
type VoiceSessionIvrStartMaxSilenceInput = f64;

fn _extract_voice_session_filler_start_delay(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionFillerStartDelayInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "delay")
}
type VoiceSessionFillerStartDelayInput = f64;

fn _extract_voice_session_filler_start_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionFillerStartIntervalInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "interval")).map(Some)
    }
}
type VoiceSessionFillerStartIntervalInput = ::std::option::Option<f64>;

fn _extract_voice_session_filler_start_max_steps(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionFillerStartMaxStepsInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "max_steps")).map(Some)
    }
}
type VoiceSessionFillerStartMaxStepsInput = ::std::option::Option<u32>;

fn _extract_voice_session_filler_stop_id(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionFillerStopIdInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "id")
}
type VoiceSessionFillerStopIdInput = u64;

fn _extract_voice_session_tool_update_operation(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionToolUpdateOperationInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "operation")
}
type VoiceSessionToolUpdateOperationInput = u64;

fn _extract_voice_session_wait_for_idle_hold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionWaitForIdleHoldInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u64>(obj.as_any(), "hold")).map(Some)
    }
}
type VoiceSessionWaitForIdleHoldInput = ::std::option::Option<u64>;

fn _extract_voice_session_release_idle_hold_hold(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionReleaseIdleHoldHoldInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "hold")
}
type VoiceSessionReleaseIdleHoldHoldInput = u64;
type VoiceSessionKeytermsOutput = ::std::option::Option<::std::vec::Vec<::std::string::String>>;

fn _extract_voice_session_commit_user_turn_transcript_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionCommitUserTurnTranscriptTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "transcript_timeout")
}
type VoiceSessionCommitUserTurnTranscriptTimeoutInput = f64;

fn _extract_voice_session_commit_user_turn_stt_flush_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<VoiceSessionCommitUserTurnSttFlushDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "stt_flush_duration")
}
type VoiceSessionCommitUserTurnSttFlushDurationInput = f64;

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
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let room =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
                room.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "room",
                    ": expected a native room session"
                ))
            })?;
        let args =
            crate::generated::records::voice_session::voice_session_args_from_py(args.as_any())?;
        let options = options
            .map(|v| {
                v.cast::<super::voice::SessionOptions>()
                    .map(|value| value.get().0.as_ref().clone())
                    .map_err(::pyo3::PyErr::from)
            })
            .transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::new(&env, room, args, options)).map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(VoiceSession(::std::sync::Arc::new(out)))
    }

    /// Open the providers and run the loop beside the host event drain. Resolves once the host
    /// has handled the opening events.
    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::start(&self.0, )).map_err(|e: ::lk_agents_abi::voice::session::OpenRefused| <::lk_agents_abi::voice::session::OpenRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::voice::session::StartFailed| <::lk_agents_abi::voice::session::StartFailed as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `session.usage`: the latest snapshot of the session's usage, settled once it closed.
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::usage(&self.0)
        });
        crate::generated::records::usage::usage_to_py(py, out)
    }

    /// The session's part of its report (history and usage), read from the core now, or as its
    /// loop closed.
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::report(&self.0, )).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let out = call.await;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::telemetry::session_report_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// `session.current_agent`: the host's agent the loop installed last, the task's while a
    /// `VoiceTask` holds the floor.
    #[getter]
    fn current_agent<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::current_agent(
                &self.0,
            )
        });
        out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()
    }

    /// The room's microphone leg, unless `RoomIoConfig(audio_input=False)`, else `None`. A fresh
    /// handle each call over the same leg; safe before or after `start()`.
    #[getter]
    fn audio_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::voice::RoomAudioInput>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::audio_input(
                &self.0,
            )
        });
        out.map(|value| {
            ::pyo3::PyResult::Ok(super::voice::RoomAudioInput(::std::sync::Arc::new(value)))
        })
        .transpose()
    }

    /// The room's video leg, when `RoomIoConfig(video_input=True)` asked for one, else `None`.
    #[getter]
    fn video_input(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::voice::RoomVideoInput>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::video_input(
                &self.0,
            )
        });
        out.map(|value| {
            ::pyo3::PyResult::Ok(super::voice::RoomVideoInput(::std::sync::Arc::new(value)))
        })
        .transpose()
    }

    /// Record the session into `recorder`; `None` detaches. Safe before or after `start()`; the
    /// file's `finish` stays the host's.
    #[pyo3(signature = (recorder))]
    fn attach_recorder(&self, py: Python<'_>, recorder: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let recorder = recorder
            .map(|v| {
                crate::py_objects::native::<dyn ::lk_agents_abi::recorder::AudioRecorder>(
                    v.as_any(),
                )
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
    #[pyo3(signature = (*, llm = crate::py_values::Omittable::NOT_GIVEN, tts = crate::py_values::Omittable::NOT_GIVEN, stt = crate::py_values::Omittable::NOT_GIVEN, vad = crate::py_values::Omittable::NOT_GIVEN, llm_node = None, tts_node = None, stt_node = None))]
    fn update_models(
        &self,
        py: Python<'_>,
        llm: crate::py_values::Omittable<'_>,
        tts: crate::py_values::Omittable<'_>,
        stt: crate::py_values::Omittable<'_>,
        vad: crate::py_values::Omittable<'_>,
        llm_node: Option<Bound<'_, PyAny>>,
        tts_node: Option<Bound<'_, PyAny>>,
        stt_node: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let llm = llm.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::llm::LlmProxy::from_host(value.as_any())).map(Some)
            }
        })?;
        let tts = tts.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::tts::TtsProxy::from_host(value.as_any())).map(Some)
            }
        })?;
        let stt = stt.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::stt::SttProxy::from_host(value.as_any())).map(Some)
            }
        })?;
        let vad = vad.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::vad::VadProxy::from_host(value.as_any())).map(Some)
            }
        })?;
        let llm_node = llm_node
            .map(|v| {
                crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(
                    v.as_any(),
                )
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
                crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(
                    v.as_any(),
                )
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
                crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::NodeFactory>(
                    v.as_any(),
                )
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "stt_node",
                        ": expected the session's nodes (`_core.AgentNodes`)"
                    ))
                })
            })
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_models(&self.0, llm, tts, stt, vad, llm_node, tts_node, stt_node)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
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
        crate::py_calls::check_pid()?;
        let frame =
            crate::generated::records::voice_session::update_agent_args_from_py(frame.as_any())?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_agent(&self.0, frame)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let out = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `VoiceTask.run()`: the task's agent, sent from inside a tool (`origin` is its
    /// captured tool identity). The agent holds the floor until one of its tools completes with a
    /// value the host validated, which this resolves to; the parent comes back without its hooks.
    /// A cancelled await ends the task, and every task it started, and puts the parent back.
    #[pyo3(signature = (frame, *, origin = None, opens_with_reply = false, dtmf = None))]
    fn run_voice_task<'py>(
        &self,
        py: Python<'py>,
        frame: Bound<'_, PyAny>,
        origin: Option<Bound<'_, PyAny>>,
        opens_with_reply: bool,
        dtmf: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let frame =
            crate::generated::records::voice_session::update_agent_args_from_py(frame.as_any())?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let dtmf = dtmf
            .map(|v| {
                crate::generated::records::voice_session::dtmf_task_options_from_py(v.as_any())
            })
            .transpose()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::run_voice_task(&self.0, frame, origin, opens_with_reply, dtmf)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::VoiceTaskEnded| <::lk_agents_abi::voice::session::VoiceTaskEnded as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::voice_session::voice_task_value_to_py(py, out)?,
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
        #[pyo3(from_py_with = _extract_voice_session_collect_dtmf_num_digits)]
        num_digits: VoiceSessionCollectDtmfNumDigitsInput,
        ask_for_confirmation: bool,
        #[pyo3(from_py_with = _extract_voice_session_collect_dtmf_timeout)]
        timeout: VoiceSessionCollectDtmfTimeoutInput,
        stop_event: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::collect_dtmf(&self.0, num_digits, ask_for_confirmation, timeout, stop_event)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::VoiceTaskEnded| <::lk_agents_abi::voice::session::VoiceTaskEnded as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// An `AvatarSession`'s own measurements (its join and playback latency), recorded with the
    /// session's metrics.
    #[pyo3(signature = (metrics))]
    fn report_avatar_metrics(&self, py: Python<'_>, metrics: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let metrics = crate::generated::records::metrics::avatar_metrics_from_py(metrics.as_any())?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::report_avatar_metrics(&self.0, metrics)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.input.audio = ..` after `start()`; `None` detaches.
    #[pyo3(signature = (source))]
    fn set_audio_input(&self, py: Python<'_>, source: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let source = source
            .map(|v| crate::generated::voice::AudioInputProxy::from_host(v.as_any()))
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_audio_input(&self.0, source)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.input.video = ..` after `start()`: the video pump reads the new source, or stops
    /// on `None`.
    #[pyo3(signature = (source))]
    fn set_video_input(&self, py: Python<'_>, source: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let source = source
            .map(|v| crate::generated::voice::VideoInputProxy::from_host(v.as_any()))
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_video_input(&self.0, source)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// Select the destination for subsequent output segments. Current work retains its admitted
    /// binding, including when the same endpoint object is selected again.
    #[pyo3(signature = (sink))]
    fn set_audio_output(&self, py: Python<'_>, sink: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let sink = sink
            .map(|v| crate::generated::voice::AudioOutputProxy::from_host(v.as_any()))
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_audio_output(&self.0, sink)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
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
        crate::py_calls::check_pid()?;
        let sink = sink
            .map(|v| crate::generated::voice::TextOutputProxy::from_host(v.as_any()))
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_transcription_output(&self.0, sink)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
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
        crate::py_calls::check_pid()?;
        let tools = tools
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::tool_options::tool_spec_from_py(
                                item.as_any(),
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let tools_host =
            tools_host.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
        let voice_agent = voice_agent
            .map(|v| {
                crate::py_objects::native::<dyn ::lk_agents_abi::agent::AgentFrame>(v.as_any())
                    .ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "voice_agent",
                            ": expected an agent's voice frame (`_core.VoiceAgent`)"
                        ))
                    })
            })
            .transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_tools(&self.0, tools, tools_host, voice_agent)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.update_options(..)` on a running session: merged into its session layer and
    /// re-resolved; `expressive` reaches the next reply, `keyterms` replace the user's list. A turn
    /// detector object is refused: swapping the detector of a running session is not ported.
    #[pyo3(signature = (update))]
    fn update_turn_options(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let update =
            crate::generated::records::session_options::update_kwargs_from_py(update.as_any())?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_turn_options(&self.0, update)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `override_options(..)` on a running session: a layer over the base until
    /// `release_override` names the id returned. Refused as `update_turn_options` refuses.
    #[pyo3(signature = (update))]
    fn override_options(&self, py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<u64> {
        crate::py_calls::check_pid()?;
        let update =
            crate::generated::records::session_options::update_kwargs_from_py(update.as_any())?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::override_options(&self.0, update)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// The `override_options` block ended, however it ended. A closed session has nothing left
    /// to restore.
    #[pyo3(signature = (layer))]
    fn release_override(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_release_override_layer)]
        layer: VoiceSessionReleaseOverrideLayerInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
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
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::resolved_options(
                &self.0,
            )
        });
        out.map(|value| {
            crate::generated::records::session_options::resolved_session_options_to_py(py, value)
        })
        .transpose()
    }

    /// `AgentActivity.push_video(frame)`: one sampled frame, on its way to the model. `False`
    /// when the loop did not take it (not started or closed); a non-image part raises.
    #[pyo3(signature = (content))]
    fn push_video(&self, py: Python<'_>, content: Bound<'_, PyAny>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let content =
            crate::generated::records::chat_items::chat_content_from_py(content.as_any())?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::push_video(&self.0, content)).map_err(|e: ::lk_agents_abi::voice::session::NotAnImage| <::lk_agents_abi::voice::session::NotAnImage as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// `generate_reply()`: reply now, with no end of turn behind it. The overrides apply to this
    /// reply only; `tools` narrows the driving agent's tools to these names (its provider-owned
    /// tools stay), refused on a realtime model as `chat_ctx` is. Sent from the agent's
    /// `on_enter`, it is that hook's reply.
    #[pyo3(signature = (*, user_input = None, instructions = None, tool_choice = None, tools = None, allow_interruptions = None, chat_ctx = None, origin = None))]
    fn generate_reply(
        &self,
        py: Python<'_>,
        user_input: ::std::option::Option<::std::string::String>,
        instructions: ::std::option::Option<::std::string::String>,
        tool_choice: Option<Bound<'_, PyAny>>,
        tools: ::std::option::Option<::std::vec::Vec<::std::string::String>>,
        allow_interruptions: ::std::option::Option<bool>,
        chat_ctx: Option<Bound<'_, PyAny>>,
        origin: Option<Bound<'_, PyAny>>,
    ) -> PyResult<super::voice::SpeechHandle> {
        crate::py_calls::check_pid()?;
        let tool_choice = tool_choice
            .map(|v| {
                crate::generated::records::from_representation::<
                    ::lk_agents_abi::ToolChoice,
                    ::serde_json::Value,
                >(crate::py_values::value_from_py(v.as_any(), "tool_choice")?)
            })
            .transpose()?;
        let chat_ctx = chat_ctx
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::generate_reply(&self.0, user_input, instructions, tool_choice, tools, allow_interruptions, chat_ctx, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::voice::SpeechHandle(out))
    }

    /// `IVRActivity.start()`: `max_silence` is `max_silence_duration` in seconds; `tools` is
    /// `IVRActivity.tools`. A second start is journalled, not run.
    #[pyo3(signature = (max_silence = 5.0, tools = None, *, tools_host = None))]
    fn ivr_start(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_ivr_start_max_silence)]
        max_silence: VoiceSessionIvrStartMaxSilenceInput,
        tools: Option<Bound<'_, PyAny>>,
        tools_host: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let tools = tools
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::tool_options::tool_spec_from_py(
                                item.as_any(),
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let tools_host =
            tools_host.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::ivr_start(&self.0, max_silence, tools, tools_host)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AMD._run`: the answering-machine detector over its config, classifier LLM, tools and
    /// optional STT. It listens once the callee answers in the session's room; the verdict comes
    /// back as `amd_result`. Returns false if another detector is already attached; the host
    /// must not publish a rejected detector as the session's owner.
    #[pyo3(signature = (cfg, llm, tools, stt, participant_identity))]
    fn attach_amd(
        &self,
        py: Python<'_>,
        cfg: Bound<'_, PyAny>,
        llm: Bound<'_, PyAny>,
        tools: Option<Bound<'_, PyAny>>,
        stt: Option<Bound<'_, PyAny>>,
        participant_identity: ::std::option::Option<::std::string::String>,
    ) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let cfg = crate::generated::records::amd::amd_config_from_py(cfg.as_any())?;
        let llm = crate::generated::llm::LlmProxy::from_host(llm.as_any())?;
        let tools = tools.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
        let stt =
            stt.map(|v| crate::generated::stt::SttProxy::from_host(v.as_any())).transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::attach_amd(&self.0, cfg, llm, tools, stt, participant_identity)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
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
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::aclose(&self.0, reason, drain)).map_err(|e: ::lk_agents_abi::voice::session::UnknownCloseReason| <::lk_agents_abi::voice::session::UnknownCloseReason as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `AgentSession.shutdown(drain=)`: `aclose` without waiting for it.
    #[pyo3(signature = (*, drain = true))]
    fn shutdown(&self, py: Python<'_>, drain: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
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
        crate::py_calls::check_pid()?;
        let call = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_participant(
                &self.0, identity,
            )
        });
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::voice::io::IoError| {
                <::lk_agents_abi::voice::io::IoError as crate::py_errors::HostException>::to_py_err(
                    &e,
                )
            })?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `RunContext.with_filler(..)` entered for tool `call_id` of `speech_id`: after `delay` s
    /// idle say `text` (or ask `source`), then repeat every `interval`. Returns the scope's number.
    #[pyo3(signature = (call_id, speech_id, *, delay, interval = None, max_steps = None, text = None, source = None, origin = None))]
    fn filler_start(
        &self,
        py: Python<'_>,
        call_id: ::std::string::String,
        speech_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_voice_session_filler_start_delay)]
        delay: VoiceSessionFillerStartDelayInput,
        #[pyo3(from_py_with = _extract_voice_session_filler_start_interval)]
        interval: VoiceSessionFillerStartIntervalInput,
        #[pyo3(from_py_with = _extract_voice_session_filler_start_max_steps)]
        max_steps: VoiceSessionFillerStartMaxStepsInput,
        text: ::std::option::Option<::std::string::String>,
        source: Option<Bound<'_, PyAny>>,
        origin: Option<Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        crate::py_calls::check_pid()?;
        let source =
            source.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::filler_start(&self.0, call_id, speech_id, delay, interval, max_steps, text, source, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// `with_filler`'s exit.
    #[pyo3(signature = (id))]
    fn filler_stop(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_filler_stop_id)]
        id: VoiceSessionFillerStopIdInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::filler_stop(&self.0, id)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it.
    fn clear_user_turn(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::clear_user_turn(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `Agent.update_instructions()` on the agent that is already driving.
    #[pyo3(signature = (instructions))]
    fn update_instructions(
        &self,
        py: Python<'_>,
        instructions: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_instructions(&self.0, instructions)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
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
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::set_io_enabled(&self.0, leg, enabled)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `RunContext.update()` after the first (an `update` outcome), or a detached tool's end.
    #[pyo3(signature = (invocation_id, operation, call_id, report))]
    fn tool_update(
        &self,
        py: Python<'_>,
        invocation_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_voice_session_tool_update_operation)]
        operation: VoiceSessionToolUpdateOperationInput,
        call_id: ::std::string::String,
        report: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let report = crate::generated::records::tool_report::tool_report_from_py(report.as_any())?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::tool_update(&self.0, invocation_id, operation, call_id, report)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `Agent.update_chat_ctx()` on a running agent: replace the context.
    #[pyo3(signature = (items))]
    fn update_chat_ctx(&self, py: Python<'_>, items: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let items = items.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::update_chat_ctx(&self.0, items)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AgentSession.interrupt(force=)`: the current and queued speeches up to the first
    /// uninterruptible; answers when they are done, or with the refusal.
    #[pyo3(signature = (*, force = false))]
    fn interrupt<'py>(&self, py: Python<'py>, force: bool) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::interrupt(&self.0, force)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::voice::session::NotInterruptible| <::lk_agents_abi::voice::session::NotInterruptible as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
    /// speech is left. Answers `None`, `"activity_changing"` or `"closing"`.
    fn drain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::drain(&self.0, )).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
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
        hold: VoiceSessionWaitForIdleHoldInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::wait_for_idle(&self.0, hold)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::voice::session::IdleRefused| <::lk_agents_abi::voice::session::IdleRefused as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `RunContext.foreground()`: wait for idle, then hold the floor until `release_idle_hold`.
    /// Answers the hold's id.
    #[pyo3(signature = (*, origin = None))]
    fn wait_for_idle_and_hold<'py>(
        &self,
        py: Python<'py>,
        origin: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::wait_for_idle_and_hold(&self.0, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::IdleRefused| <::lk_agents_abi::voice::session::IdleRefused as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// `_wait_for_idle_and_hold`'s exit. A hold the loop no longer has is a no-op.
    #[pyo3(signature = (hold))]
    fn release_idle_hold(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_voice_session_release_idle_hold_hold)]
        hold: VoiceSessionReleaseIdleHoldHoldInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
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
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::claim_user_turn(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_claim_user_turn()`'s exit: the last release re-derives the user state.
    fn release_user_turn(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
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
        origin: Option<Bound<'_, PyAny>>,
        host: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let host = host.map(|v| crate::py_objects::object_ref_from_py(v.as_any())).transpose()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::run(&self.0, user_input, origin, host)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `say()`: speak `text` without the LLM, the words given or a stream of them read to its
    /// end. With `audio`, those frames play instead of the TTS and `text`, a `str`, is still
    /// committed. The handle is the promise the session fulfils.
    #[pyo3(signature = (text, *, audio = None, allow_interruptions = None, add_to_chat_ctx = true, origin = None))]
    fn say(
        &self,
        py: Python<'_>,
        text: Bound<'_, PyAny>,
        audio: Option<Bound<'_, PyAny>>,
        allow_interruptions: ::std::option::Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<Bound<'_, PyAny>>,
    ) -> PyResult<super::voice::SpeechHandle> {
        crate::py_calls::check_pid()?;
        let text = if text.is_instance_of::<::pyo3::types::PyString>() {
            text.extract::<::std::string::String>().map(::lk_agents_abi::OneOrStream::One)
        } else {
            crate::py_calls::capture_context(text.py())
                .and_then(|context| {
                    crate::py_streams::HostStream::open(
                        text.py(),
                        text.as_any(),
                        &context,
                        |o| o.extract::<::std::string::String>(),
                        ::lk_agents_runtime::process::host_clock(),
                    )
                })
                .map(|s| {
                    ::std::boxed::Box::new(s)
                        as ::std::boxed::Box<
                            dyn ::lk_agents_abi::StreamOutput<::std::string::String>,
                        >
                })
                .map(::lk_agents_abi::OneOrStream::Stream)
        }?;
        let audio = audio
            .map(|v| {
                crate::py_calls::capture_context(v.py())
                    .and_then(|context| {
                        crate::py_streams::HostStream::open(
                            v.py(),
                            v.as_any(),
                            &context,
                            |o| {
                                crate::generated::records::rtc_frames::audio_frame_from_py(
                                    o.as_any(),
                                )
                            },
                            ::lk_agents_runtime::process::host_clock(),
                        )
                    })
                    .map(|s| {
                        ::std::boxed::Box::new(s)
                            as ::std::boxed::Box<
                                dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::AudioFrame>,
                            >
                    })
            })
            .transpose()?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let out = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::say(&self.0, text, audio, allow_interruptions, add_to_chat_ctx, origin)).map_err(|e: ::lk_agents_abi::voice::session::CommandRefused| <::lk_agents_abi::voice::session::CommandRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(super::voice::SpeechHandle(out))
    }

    /// `_start_ivr_detection(transcript)`: a final the navigator counts as if heard.
    #[pyo3(signature = (text))]
    fn ivr_final(&self, py: Python<'_>, text: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::ivr_final(&self.0, text)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AMD.started`: the detector listens (the callee answered, no verdict out yet).
    fn amd_listening(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::amd_listening(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// `AMD.execute()`'s interrupt and `_resume_authorization()`, after its IVR hand-off.
    fn amd_execute(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::amd_execute(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `AMD.aclose()`. Idempotent.
    fn amd_close(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::amd_close(&self.0)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `session.keyterms`: the list the recogniser was last handed; `None` before any push.
    fn keyterms(&self, py: Python<'_>) -> PyResult<VoiceSessionKeytermsOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::keyterms(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// `Agent.chat_ctx` of the driving agent; empty before `start()`.
    fn agent_chat_ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::agent_chat_ctx(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// `AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
    /// transcript. Runs whether or not awaited; a newer commit supersedes it.
    #[pyo3(signature = (*, transcript_timeout = 2.0, stt_flush_duration = 2.0, skip_reply = false))]
    fn commit_user_turn<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_voice_session_commit_user_turn_transcript_timeout)]
        transcript_timeout: VoiceSessionCommitUserTurnTranscriptTimeoutInput,
        #[pyo3(from_py_with = _extract_voice_session_commit_user_turn_stt_flush_duration)]
        stt_flush_duration: VoiceSessionCommitUserTurnSttFlushDurationInput,
        skip_reply: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <VoiceSessionImpl as ::lk_agents_abi::voice::session::VoiceSession>::commit_user_turn(&self.0, transcript_timeout, stt_flush_duration, skip_reply)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let out = call.await.map_err(|e: ::lk_agents_abi::voice::session::CommitSuperseded| <::lk_agents_abi::voice::session::CommitSuperseded as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
type SessionStartImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SessionStart;

#[pyfunction]
#[pyo3(signature = (vad_supplied, detector_supplied, realtime))]
pub fn default_session_models<'py>(
    py: Python<'py>,
    vad_supplied: bool,
    detector_supplied: bool,
    realtime: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <SessionStartImpl as ::lk_agents_abi::voice::session_effects::SessionStart>::default_session_models(vad_supplied, detector_supplied, realtime));
    crate::generated::records::session_wire::default_model_plan_to_py(py, out)
}

#[pyfunction]
pub fn user_turn_exceeded_reply<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let out = py.detach(<SessionStartImpl as ::lk_agents_abi::voice::session_effects::SessionStart>::user_turn_exceeded_reply);
    crate::generated::records::session_wire::exceeded_turn_reply_to_py(py, out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
type CoreSessionStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SessionState;

/// `CoreSessionState`: one delivered state (the history and the agent's and user's states), as
/// the host reads it.
#[pyclass(frozen, name = "_core_SessionState", module = "livekit.agents._core")]
pub struct CoreSessionState(pub(crate) ::std::sync::Arc<CoreSessionStateImpl>);

#[pymethods]
impl CoreSessionState {
    /// `session.history`'s items, in history order: the host's own objects, the same ones each
    /// later state and every event naming the item hand back.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStateImpl as ::lk_agents_abi::voice::session_effects::SessionState>::history(&self.0));
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::py_objects::object_ref_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// `session.agent_state`: `"initializing"` until the first `agent_state_changed`.
    #[getter]
    fn agent_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStateImpl as ::lk_agents_abi::voice::session_effects::SessionState>::agent_state(&self.0));
        crate::generated::records::events::agent_state_to_py(py, out)
    }

    /// `session.user_state`: `"listening"` until the first `user_state_changed`.
    #[getter]
    fn user_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStateImpl as ::lk_agents_abi::voice::session_effects::SessionState>::user_state(&self.0));
        crate::generated::records::events::user_state_to_py(py, out)
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::session_effects::DeliveredState`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::session_effects::DeliveredState>(),
        read: |obj| {
            let this = obj.cast::<CoreSessionState>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::DeliveredState> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_options.rs.
type InterruptionGateFunctionsImpl = <crate::generated::implementations::Selected as super::Implementations>::InterruptionGateFunctions;

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
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <InterruptionGateFunctionsImpl as ::lk_agents_abi::voice::session_options::InterruptionGateFunctions>::rt_server_turn_detection(caps_turn_detection, can_disable_turn_detection, vad, turn_detection, streaming_detector, interruption_given));
    ::pyo3::PyResult::Ok(out)
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
    crate::py_calls::check_pid()?;
    let agent_mode = agent_mode
        .map(|v| crate::generated::records::turn::interruption_mode_from_py(v.as_any()))
        .transpose()?;
    let session_mode = session_mode
        .map(|v| crate::generated::records::turn::interruption_mode_from_py(v.as_any()))
        .transpose()?;
    let out = py.detach(|| <InterruptionGateFunctionsImpl as ::lk_agents_abi::voice::session_options::InterruptionGateFunctions>::resolve_interruption_detector(stt_aligned_transcript, stt_streaming, realtime, rt_server_turn_detection, has_vad, turn_detection, allow_interruptions, agent_mode, session_mode, hosted, dev_mode));
    crate::generated::records::session_options::interruption_gate_verdict_to_py(py, out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_owner.rs.
type CoreSessionBindingsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SessionPreparation;

#[pyclass(frozen, name = "_core_SessionBindings", module = "livekit.agents._core")]
pub struct CoreSessionBindings(pub(crate) ::std::sync::Arc<CoreSessionBindingsImpl>);

#[pymethods]
impl CoreSessionBindings {
    #[new]
    #[pyo3(signature = (room, args, *, record_audio = false))]
    fn new(
        py: Python<'_>,
        room: Bound<'_, PyAny>,
        args: Bound<'_, PyAny>,
        record_audio: bool,
    ) -> PyResult<CoreSessionBindings> {
        crate::py_calls::check_pid()?;
        let room =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
                room.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "room",
                    ": expected a native room session"
                ))
            })?;
        let args =
            crate::generated::records::voice_session::voice_session_args_from_py(args.as_any())?;
        let out = py.detach(|| <CoreSessionBindingsImpl as ::lk_agents_abi::voice::session_owner::SessionPreparation>::new(room, args, record_audio));
        ::pyo3::PyResult::Ok(CoreSessionBindings(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::session_owner::SessionBindingsPort`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::session_owner::SessionBindingsPort>(),
        read: |obj| {
            let this = obj.cast::<CoreSessionBindings>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::session_owner::SessionBindingsPort> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_owner.rs.
type CoreSessionOwnerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SessionOwner;
type SessionOwnerKeytermsOutput = ::std::vec::Vec<::std::string::String>;

fn _extract_session_owner_release_override_layer(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SessionOwnerReleaseOverrideLayerInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "layer")
}
type SessionOwnerReleaseOverrideLayerInput = u64;
type SessionOwnerStartedAtOutput = ::std::option::Option<f64>;
type SessionOwnerRecordingPathOutput = ::std::option::Option<::std::string::String>;
type SessionOwnerRecordingStartedAtOutput = ::std::option::Option<f64>;

#[pyclass(frozen, name = "_core_SessionOwner", module = "livekit.agents._core")]
pub struct CoreSessionOwner(pub(crate) ::std::sync::Arc<CoreSessionOwnerImpl>);

#[pymethods]
impl CoreSessionOwner {
    #[new]
    #[pyo3(signature = (options))]
    fn new(py: Python<'_>, options: Bound<'_, PyAny>) -> PyResult<CoreSessionOwner> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let options = options
            .cast::<super::voice::SessionOptions>()
            .map(|value| value.get().0.as_ref().clone())
            .map_err(::pyo3::PyErr::from)?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::new(
                &env, options,
            )
        });
        ::pyo3::PyResult::Ok(CoreSessionOwner(::std::sync::Arc::new(out)))
    }

    /// `job`: the job the session runs in, whose trace its spans join under the span current at
    /// the call, and which alone may read its report. Startup runs in order: the thread's storage
    /// opens; the host prepares the room and legs; a session of the job (on its room, or the
    /// console's) claims the job's primary session and sets the job's recording from its plan;
    /// the start time is set; the job's session opens its audio recording and the job's shutdown
    /// is made to close it; the recording is ready; the toolsets set up and the voice loop starts.
    #[pyo3(signature = (storage, host, *, job = None))]
    fn start<'py>(
        &self,
        py: Python<'py>,
        storage: Bound<'_, PyAny>,
        host: Bound<'_, PyAny>,
        job: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let storage = storage
            .cast::<super::voice::CoreSessionStorage>()
            .map(|value| ::std::sync::Arc::clone(&value.get().0))
            .map_err(::pyo3::PyErr::from)?;
        let host = crate::generated::voice::SessionHostProxy::from_host(host.as_any())?;
        let job = job
            .map(|v| crate::generated::records::session_owner::session_job_info_from_py(v.as_any()))
            .transpose()?;
        let cx = crate::py_calls::entry_cx();
        let call = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::start(&self.0, storage, host, job, cx)).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await.map_err(|e: ::lk_agents_abi::errors::HostError| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(
                    &e,
                )
            })?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn session(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::voice::VoiceSession>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::session(
                &self.0,
            )
        });
        out.map(|value| ::pyo3::PyResult::Ok(super::voice::VoiceSession(value))).transpose()
    }

    /// The session's `agent_session` span, from `start()`: the job's current span while the
    /// session runs.
    #[getter]
    fn span<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::span(
                &self.0,
            )
        });
        out.map(|value| crate::generated::records::telemetry::span_context_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn storage(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::voice::CoreSessionStorage>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::storage(
                &self.0,
            )
        });
        out.map(|value| ::pyo3::PyResult::Ok(super::voice::CoreSessionStorage(value))).transpose()
    }

    #[getter]
    fn current_agent<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::current_agent(&self.0));
        out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()
    }

    /// The latest state the host's events delivered; before `start()`, an empty history.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<super::voice::CoreSessionState> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::state(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(super::voice::CoreSessionState(out))
    }

    /// Empty before startup; the final snapshot remains readable after close.
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::usage(
                &self.0,
            )
        });
        crate::generated::records::usage::usage_to_py(py, out)
    }

    /// No keyterms are applied before a voice driver exists.
    #[getter]
    fn keyterms(&self, py: Python<'_>) -> PyResult<SessionOwnerKeytermsOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::keyterms(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// `update_agent(agent, chat_ctx=)`. Before `start()`, `agent` drives once the session
    /// starts, on `chat_ctx` unless `start()` names its own; after, `frame` is the handoff and
    /// `chat_ctx` the context `agent` works on instead of the conversation carried forward.
    #[pyo3(signature = (agent, frame, *, chat_ctx = None))]
    fn update_agent(
        &self,
        py: Python<'_>,
        agent: Bound<'_, PyAny>,
        frame: Option<Bound<'_, PyAny>>,
        chat_ctx: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let agent = crate::py_objects::object_ref_from_py(agent.as_any())?;
        let frame = frame
            .map(|v| {
                crate::generated::records::voice_session::update_agent_args_from_py(v.as_any())
            })
            .transpose()?;
        let chat_ctx = chat_ctx
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::update_agent(&self.0, agent, frame, chat_ctx)).map_err(|e: ::lk_agents_abi::voice::session::SessionRefused| <::lk_agents_abi::voice::session::SessionRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `update_options(..)`: change only the supplied settings. A replaced `stt`, `tts` or `vad`
    /// is installed in place on a running session, without restarting the call or interrupting
    /// speech; a setting an `override_options` layer holds takes the new value once the layer is
    /// released. `interruption` merges over the session's and takes no `mode`, which is set with
    /// `VoiceSession(turn_handling=)`.
    #[pyo3(signature = (*, stt = crate::py_values::Omittable::NOT_GIVEN, tts = crate::py_values::Omittable::NOT_GIVEN, vad = crate::py_values::Omittable::NOT_GIVEN, turn_detection = crate::py_values::Omittable::NOT_GIVEN, endpointing_opts = crate::py_values::Omittable::NOT_GIVEN, interruption = crate::py_values::Omittable::NOT_GIVEN, min_consecutive_speech_delay = crate::py_values::Omittable::NOT_GIVEN, tts_text_transforms = crate::py_values::Omittable::NOT_GIVEN, expressive = crate::py_values::Omittable::NOT_GIVEN, keyterms = crate::py_values::Omittable::NOT_GIVEN))]
    fn update_options(
        &self,
        py: Python<'_>,
        stt: crate::py_values::Omittable<'_>,
        tts: crate::py_values::Omittable<'_>,
        vad: crate::py_values::Omittable<'_>,
        turn_detection: crate::py_values::Omittable<'_>,
        endpointing_opts: crate::py_values::Omittable<'_>,
        interruption: crate::py_values::Omittable<'_>,
        min_consecutive_speech_delay: crate::py_values::Omittable<'_>,
        tts_text_transforms: crate::py_values::Omittable<'_>,
        expressive: crate::py_values::Omittable<'_>,
        keyterms: crate::py_values::Omittable<'_>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let stt = stt.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::py_objects::object_ref_from_py(value.as_any())).map(Some)
            }
        })?;
        let tts = tts.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::py_objects::object_ref_from_py(value.as_any())).map(Some)
            }
        })?;
        let vad = vad.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::py_objects::object_ref_from_py(value.as_any())).map(Some)
            }
        })?;
        let turn_detection = turn_detection
            .map(|value| crate::py_values::turn_detection_kwarg_from_py(value.as_any()))?;
        let endpointing_opts = endpointing_opts.map(|value| {
            crate::generated::records::turn::endpointing_options_from_py(value.as_any())
        })?;
        let interruption = interruption.map(|value| {
            crate::generated::records::turn::interruption_options_from_py(value.as_any())
        })?;
        let min_consecutive_speech_delay = min_consecutive_speech_delay.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "min_consecutive_speech_delay")
        })?;
        let tts_text_transforms = tts_text_transforms.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::records::from_representation::<
                    ::lk_agents_abi::voice::session_options::TextTransformSet,
                    ::std::vec::Vec<::std::string::String>,
                >(if value.as_any().is_instance_of::<::pyo3::types::PyList>()
                    || value.as_any().is_instance_of::<::pyo3::types::PyTuple>()
                {
                    value
                        .as_any()
                        .try_iter()?
                        .map(|item| {
                            let item = &item?;
                            item.extract::<::std::string::String>()
                        })
                        .collect::<PyResult<Vec<_>>>()
                } else {
                    Err(::pyo3::exceptions::PyTypeError::new_err(
                        "tts_text_transforms: expected a list or tuple",
                    ))
                }?))
                .map(Some)
            }
        })?;
        let expressive =
            expressive.map(|value| crate::py_values::expressive_from_py(value.as_any()))?;
        let keyterms = keyterms.map(|value| {
            value.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        item.extract::<::std::string::String>()
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
        })?;
        py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::update_options(&self.0, stt, tts, vad, turn_detection, endpointing_opts, interruption, min_consecutive_speech_delay, tts_text_transforms, expressive, keyterms)).map_err(|e: ::lk_agents_abi::voice::session_owner::OptionsRefused| <::lk_agents_abi::voice::session_owner::OptionsRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `override_options(..)` entered: a layer over the session's settings and providers, each
    /// taking its value from the topmost layer that sets it, named by the id returned, refused as
    /// `update_options` refuses.
    #[pyo3(signature = (*, stt = crate::py_values::Omittable::NOT_GIVEN, tts = crate::py_values::Omittable::NOT_GIVEN, vad = crate::py_values::Omittable::NOT_GIVEN, turn_detection = crate::py_values::Omittable::NOT_GIVEN, endpointing_opts = crate::py_values::Omittable::NOT_GIVEN, interruption = crate::py_values::Omittable::NOT_GIVEN, min_consecutive_speech_delay = crate::py_values::Omittable::NOT_GIVEN, tts_text_transforms = crate::py_values::Omittable::NOT_GIVEN))]
    fn override_options(
        &self,
        py: Python<'_>,
        stt: crate::py_values::Omittable<'_>,
        tts: crate::py_values::Omittable<'_>,
        vad: crate::py_values::Omittable<'_>,
        turn_detection: crate::py_values::Omittable<'_>,
        endpointing_opts: crate::py_values::Omittable<'_>,
        interruption: crate::py_values::Omittable<'_>,
        min_consecutive_speech_delay: crate::py_values::Omittable<'_>,
        tts_text_transforms: crate::py_values::Omittable<'_>,
    ) -> PyResult<u64> {
        crate::py_calls::check_pid()?;
        let stt = stt.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::py_objects::object_ref_from_py(value.as_any())).map(Some)
            }
        })?;
        let tts = tts.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::py_objects::object_ref_from_py(value.as_any())).map(Some)
            }
        })?;
        let vad = vad.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::py_objects::object_ref_from_py(value.as_any())).map(Some)
            }
        })?;
        let turn_detection = turn_detection
            .map(|value| crate::py_values::turn_detection_kwarg_from_py(value.as_any()))?;
        let endpointing_opts = endpointing_opts.map(|value| {
            crate::generated::records::turn::endpointing_options_from_py(value.as_any())
        })?;
        let interruption = interruption.map(|value| {
            crate::generated::records::turn::interruption_options_from_py(value.as_any())
        })?;
        let min_consecutive_speech_delay = min_consecutive_speech_delay.map(|value| {
            crate::py_values::number_from_py::<f64>(value.as_any(), "min_consecutive_speech_delay")
        })?;
        let tts_text_transforms = tts_text_transforms.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (crate::generated::records::from_representation::<
                    ::lk_agents_abi::voice::session_options::TextTransformSet,
                    ::std::vec::Vec<::std::string::String>,
                >(if value.as_any().is_instance_of::<::pyo3::types::PyList>()
                    || value.as_any().is_instance_of::<::pyo3::types::PyTuple>()
                {
                    value
                        .as_any()
                        .try_iter()?
                        .map(|item| {
                            let item = &item?;
                            item.extract::<::std::string::String>()
                        })
                        .collect::<PyResult<Vec<_>>>()
                } else {
                    Err(::pyo3::exceptions::PyTypeError::new_err(
                        "tts_text_transforms: expected a list or tuple",
                    ))
                }?))
                .map(Some)
            }
        })?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::override_options(&self.0, stt, tts, vad, turn_detection, endpointing_opts, interruption, min_consecutive_speech_delay, tts_text_transforms)).map_err(|e: ::lk_agents_abi::voice::session_owner::OptionsRefused| <::lk_agents_abi::voice::session_owner::OptionsRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }

    /// The `override_options` block ended, however it ended: its settings are released and the
    /// providers it changed restored. On a closed session the settings still release and the
    /// restore is refused (`SessionClosing`); there is nothing left to restore.
    #[pyo3(signature = (layer))]
    fn release_override(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_session_owner_release_override_layer)]
        layer: SessionOwnerReleaseOverrideLayerInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::release_override(&self.0, layer)).map_err(|e: ::lk_agents_abi::voice::session_owner::OptionsRefused| <::lk_agents_abi::voice::session_owner::OptionsRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (legs))]
    fn resolved_options<'py>(
        &self,
        py: Python<'py>,
        legs: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let legs = crate::generated::records::session_options::session_legs_from_py(legs.as_any())?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::resolved_options(&self.0, legs));
        crate::generated::records::session_options::resolved_session_options_to_py(py, out)
    }

    #[pyo3(signature = (reason, drain = false))]
    fn aclose<'py>(
        &self,
        py: Python<'py>,
        reason: ::std::string::String,
        drain: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let call = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::aclose(&self.0, reason, drain)).map_err(|e: ::lk_agents_abi::voice::session::UnknownCloseReason| <::lk_agents_abi::voice::session::UnknownCloseReason as crate::py_errors::HostException>::to_py_err(&e))?;
        crate::py_calls::future_into_py(py, async move {
            let () = call.await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[pyo3(signature = (reason, drain = false))]
    fn shutdown(&self, py: Python<'_>, reason: ::std::string::String, drain: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::shutdown(&self.0, reason, drain)).map_err(|e: ::lk_agents_abi::voice::session::UnknownCloseReason| <::lk_agents_abi::voice::session::UnknownCloseReason as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The recording's final result remains available to the job reporter.
    fn finish_recording<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::finish_recording(&this, )
                .await.map_err(|e: ::lk_agents_abi::recorder::RecordingFailure| <::lk_agents_abi::recorder::RecordingFailure as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// When the session started (wall clock seconds), once its startup prepared it.
    #[getter]
    fn started_at(&self, py: Python<'_>) -> PyResult<SessionOwnerStartedAtOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::started_at(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// What the session records: its job's plan for it, nothing outside a job or before it starts.
    #[getter]
    fn recording_options<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::recording_options(&self.0));
        crate::generated::records::telemetry::recording_options_to_py(py, out)
    }

    /// The session's audio recording file (`audio.ogg` in the job's session directory), when it
    /// records audio.
    #[getter]
    fn recording_path(&self, py: Python<'_>) -> PyResult<SessionOwnerRecordingPathOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::recording_path(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// When the audio recording's first frame was written.
    #[getter]
    fn recording_started_at(
        &self,
        py: Python<'_>,
    ) -> PyResult<SessionOwnerRecordingStartedAtOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::recording_started_at(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Whether the job `job_id` may read the session's report: the job the session first started
    /// in, or any job for a session started outside one.
    #[pyo3(signature = (job_id))]
    fn allows_report(&self, py: Python<'_>, job_id: ::std::string::String) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::allows_report(&self.0, job_id));
        ::pyo3::PyResult::Ok(out)
    }

    /// Whether the session's report keeps an event of `kind`: a known kind, its content only when
    /// the session's recording is not redacted.
    #[pyo3(signature = (kind))]
    fn accepts_event(&self, py: Python<'_>, kind: ::std::string::String) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::accepts_event(&self.0, kind));
        ::pyo3::PyResult::Ok(out)
    }

    /// An event of `kind` the report keeps is counted as retained; whether it was accepted.
    #[pyo3(signature = (kind))]
    fn admit_event(&self, py: Python<'_>, kind: ::std::string::String) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionOwnerImpl as ::lk_agents_abi::voice::session_owner::SessionOwner>::admit_event(&self.0, kind));
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_storage.rs.
type CoreSessionStorageImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SessionStorage;
type SessionStorageThreadIdOutput = ::std::option::Option<::std::string::String>;

#[pyclass(frozen, name = "_core_SessionStorage", module = "livekit.agents._core")]
pub struct CoreSessionStorage(pub(crate) ::std::sync::Arc<CoreSessionStorageImpl>);

#[pymethods]
impl CoreSessionStorage {
    #[new]
    #[pyo3(signature = (namespace, thread_id, lane, fallback_lane, initial_history))]
    fn new(
        py: Python<'_>,
        namespace: Option<Bound<'_, PyAny>>,
        thread_id: ::std::option::Option<::std::string::String>,
        lane: ::std::string::String,
        fallback_lane: ::std::string::String,
        initial_history: Option<Bound<'_, PyAny>>,
    ) -> PyResult<CoreSessionStorage> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let namespace = namespace
            .map(|v| {
                v.cast::<super::storage::Storage>()
                    .map(|value| ::std::sync::Arc::clone(&value.get().0))
                    .map_err(::pyo3::PyErr::from)
            })
            .transpose()?;
        let initial_history = initial_history
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::chat_items::chat_item_from_py(item.as_any())
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::new(&env, namespace, thread_id, lane, fallback_lane, initial_history));
        ::pyo3::PyResult::Ok(CoreSessionStorage(::std::sync::Arc::new(out)))
    }

    /// Load the supplied lane, or create the ephemeral context store selected by voice policy.
    fn open<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::open(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[getter]
    fn namespace(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::storage::Storage>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::namespace(&self.0));
        out.map(|value| ::pyo3::PyResult::Ok(super::storage::Storage(value))).transpose()
    }

    #[getter]
    fn thread_id(&self, py: Python<'_>) -> PyResult<SessionStorageThreadIdOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::thread_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn scope_id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::scope_id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn activation(&self, py: Python<'_>) -> PyResult<super::storage::CoreContextActivation> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::activation(&self.0));
        ::pyo3::PyResult::Ok(super::storage::CoreContextActivation(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn voice_store(
        &self,
        py: Python<'_>,
    ) -> PyResult<::std::option::Option<super::storage::CoreVoiceStore>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::voice_store(&self.0));
        out.map(|value| ::pyo3::PyResult::Ok(super::storage::CoreVoiceStore(value))).transpose()
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::history(&self.0));
        out.map(|value| {
            ::pyo3::PyResult::Ok(
                ::pyo3::types::PyList::new(
                    py,
                    value
                        .into_iter()
                        .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any(),
            )
        })
        .transpose()
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreSessionStorageImpl as ::lk_agents_abi::voice::session_storage::SessionStorage>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/speech_handle.rs.
type SpeechHandleImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SpeechHandle;

fn _extract_speech_handle_wait_for_generation_step(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SpeechHandleWaitForGenerationStepInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<usize>(obj.as_any(), "step")).map(Some)
    }
}
type SpeechHandleWaitForGenerationStepInput = ::std::option::Option<usize>;

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
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::new(
                speech_id,
                allow_interruptions,
            )
        });
        ::pyo3::PyResult::Ok(SpeechHandle(::std::sync::Arc::new(out)))
    }

    /// The host's object for this speech, as it set it; `None` until then.
    #[getter]
    #[pyo3(name = "host")]
    fn host_object<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::host_object(
                &self.0,
            )
        });
        out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()
    }

    /// Set by the host when it creates its handle over this one: from then on every event, tool
    /// and hook naming the speech hands back this object.
    #[setter(host)]
    fn set_host_object(&self, py: Python<'_>, host: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let host = crate::py_objects::object_ref_from_py(host.as_any())?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::set_host_object(&self.0, host));
        Ok(())
    }

    /// `speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
    /// the id it was constructed with.
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::id(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn num_steps(&self, py: Python<'_>) -> PyResult<u32> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::num_steps(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// What this speech committed to the conversation; a copy.
    #[getter]
    fn chat_items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::chat_items(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    #[getter]
    fn scheduled(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::scheduled(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// `True` from the tick `interrupt()` was called until the speech is done, and once the loop
    /// has interrupted the speech for any reason (an interruption, a supersede, the close). A done
    /// speech reads the loop's verdict.
    #[getter]
    fn interrupted(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::interrupted(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::done(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// The core's last accepted `allow_interruptions`.
    #[getter]
    fn allow_interruptions(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::allow_interruptions(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// Enqueues the change; the accepted view changes when the loop applies it.
    #[setter(allow_interruptions)]
    fn set_allow_interruptions(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::set_allow_interruptions(&self.0, value)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// The exception the speech failed with: the host's own object when a provider raised one.
    fn exception<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::exception(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        out.map(|value| crate::py_errors::ExceptionValue::to_exception(value, py)).transpose()
    }

    /// `interrupt(force=)`, in the reference's order. If the speech does not exist yet, the
    /// command is sent when the loop creates it.
    #[pyo3(signature = (*, force = false))]
    fn interrupt(&self, py: Python<'_>, force: bool) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::interrupt(&self.0, force)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `EndCallTool`: close the session once this speech is done, or once a realtime model's own
    /// reply to the tool result is. Sent to this speech's own session.
    fn end_call(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::end_call(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_hold_interruptions`: disallow interruptions until every hold taken is released. Raises
    /// when the speech is already interrupted, without counting the attempt.
    fn hold_interruptions(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::hold_interruptions(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `_release_interruptions`. Idempotent past zero, like the reference's.
    fn release_interruptions(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::release_interruptions(&self.0)).map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// `wait_for_playout()`, and therefore `__await__`, from `origin` (the captured tool invocation inside a tool). Raises from inside a tool this speech waits on; otherwise never raises: a speech
    /// the session ended before creating, or a handle constructed directly, has nothing to play.
    #[pyo3(signature = (*, origin = None))]
    fn wait_for_playout<'py>(
        &self,
        py: Python<'py>,
        origin: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let origin = origin
            .map(|v| {
                crate::generated::records::voice_session::voice_command_origin_from_py(v.as_any())
            })
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_for_playout(&this, origin)
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// `_wait_for_generation(step_idx)`: one step of the turn, not the whole thing; `None` is the
    /// latest. No generation, no such step and a never-created speech are distinct errors.
    #[pyo3(signature = (step))]
    fn wait_for_generation<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_speech_handle_wait_for_generation_step)]
        step: SpeechHandleWaitForGenerationStepInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_for_generation(&this, step)
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Resolves when the speech is interrupted, and only then: a speech that finishes normally
    /// never resolves this, so `wait_if_not_interrupted` can race it.
    fn wait_for_interrupted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_for_interrupted(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Resolves once the loop has created the speech and put it in the queue.
    fn wait_until_created<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_until_created(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::speech_handle::SpeechRefused| <::lk_agents_abi::voice::speech_handle::SpeechRefused as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Resolves once the session has handed the host this speech's `speech_done` event, so what
    /// waits on it runs after that event's callbacks; at once for a speech the session ended
    /// before creating and for a handle constructed directly, and when the session closes for one
    /// it never reported. Never raises.
    fn wait_reported<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <SpeechHandleImpl as ::lk_agents_abi::voice::speech_handle::SpeechHandle>::wait_reported(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::speech_handle::Speech`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::speech_handle::Speech>(),
        read: |obj| {
            let this = obj.cast::<SpeechHandle>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::speech_handle::Speech> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/mod.rs.
type WorkflowChecksImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WorkflowChecks;

fn _extract_workflow_checks_card_expired_month(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WorkflowChecksCardExpiredMonthInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "month")
}
type WorkflowChecksCardExpiredMonthInput = i64;

fn _extract_workflow_checks_card_expired_year(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WorkflowChecksCardExpiredYearInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "year")
}
type WorkflowChecksCardExpiredYearInput = i64;

fn _extract_workflow_checks_normalize_date_of_birth_year(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WorkflowChecksNormalizeDateOfBirthYearInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "year")
}
type WorkflowChecksNormalizeDateOfBirthYearInput = i64;

fn _extract_workflow_checks_normalize_date_of_birth_month(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WorkflowChecksNormalizeDateOfBirthMonthInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "month")
}
type WorkflowChecksNormalizeDateOfBirthMonthInput = i64;

fn _extract_workflow_checks_normalize_date_of_birth_day(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WorkflowChecksNormalizeDateOfBirthDayInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "day")
}
type WorkflowChecksNormalizeDateOfBirthDayInput = i64;

#[pyfunction]
#[pyo3(signature = (confirmation))]
pub fn dtmf_capture_plan<'py>(py: Python<'py>, confirmation: bool) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::dtmf_capture_plan(confirmation));
    crate::generated::records::workflows::capture_plan_to_py(py, out)
}

#[pyfunction]
pub fn email_capture_plan<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let out = py.detach(<WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::email_capture_plan);
    crate::generated::records::workflows::capture_plan_to_py(py, out)
}

#[pyfunction]
#[pyo3(signature = (email_address))]
pub fn email_read_back(
    py: Python<'_>,
    email_address: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::email_read_back(
            email_address,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// The Luhn checksum over an all-digit string.
#[pyfunction]
#[pyo3(signature = (card_number))]
pub fn validate_card_number(py: Python<'_>, card_number: ::std::string::String) -> PyResult<bool> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::validate_card_number(card_number));
    ::pyo3::PyResult::Ok(out)
}

/// Whether a two-digit expiry year and its month precede `today`'s month.
#[pyfunction]
#[pyo3(signature = (month, year, today))]
pub fn card_expired(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_workflow_checks_card_expired_month)]
    month: WorkflowChecksCardExpiredMonthInput,
    #[pyo3(from_py_with = _extract_workflow_checks_card_expired_year)]
    year: WorkflowChecksCardExpiredYearInput,
    today: Bound<'_, PyAny>,
) -> PyResult<bool> {
    let today = crate::generated::records::dates::date_from_py(today.as_any())?;
    let out = py.detach(|| {
        <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::card_expired(
            month, year, today,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// A two-digit year is this century's when it is not past this year's two digits, else the
/// last century's; an invalid or future date is refused.
#[pyfunction]
#[pyo3(signature = (year, month, day, today))]
pub fn normalize_date_of_birth<'py>(
    py: Python<'py>,
    #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_year)]
    year: WorkflowChecksNormalizeDateOfBirthYearInput,
    #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_month)]
    month: WorkflowChecksNormalizeDateOfBirthMonthInput,
    #[pyo3(from_py_with = _extract_workflow_checks_normalize_date_of_birth_day)]
    day: WorkflowChecksNormalizeDateOfBirthDayInput,
    today: Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let today = crate::generated::records::dates::date_from_py(today.as_any())?;
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_date_of_birth(year, month, day, today)).map_err(|e: ::lk_agents_abi::voice::workflows::DateOfBirthRefused| <::lk_agents_abi::voice::workflows::DateOfBirthRefused as crate::py_errors::HostException>::to_py_err(&e))?;
    crate::generated::records::dates::date_to_py(py, out)
}

/// Stripped, and refused unless it reads `local@label.….tld`.
#[pyfunction]
#[pyo3(signature = (email))]
pub fn normalize_email_address(
    py: Python<'_>,
    email: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_email_address(email)).map_err(|e: ::lk_agents_abi::voice::workflows::CaptureRefused| <::lk_agents_abi::voice::workflows::CaptureRefused as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

/// Whitespace, hyphens, parentheses and dots dropped, and refused unless it reads `+?[1-9]`
/// followed by six to fourteen digits.
#[pyfunction]
#[pyo3(signature = (phone_number))]
pub fn normalize_phone_number(
    py: Python<'_>,
    phone_number: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::normalize_phone_number(phone_number)).map_err(|e: ::lk_agents_abi::voice::workflows::CaptureRefused| <::lk_agents_abi::voice::workflows::CaptureRefused as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

/// Digits are themselves, `*` is 10, `#` is 11 and `A`-`D` are 12-15.
#[pyfunction]
#[pyo3(signature = (symbol))]
pub fn dtmf_code(py: Python<'_>, symbol: ::std::string::String) -> PyResult<u8> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::dtmf_code(symbol)).map_err(|e: ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol| <::lk_agents_abi::voice::workflows::UnknownDtmfSymbol as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

/// The space-joined digits that fill the DTMF read-back's `{digits}` hole.
#[pyfunction]
#[pyo3(signature = (symbols))]
pub fn format_dtmf(
    py: Python<'_>,
    symbols: ::std::vec::Vec<::std::string::String>,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <WorkflowChecksImpl as ::lk_agents_abi::voice::workflows::WorkflowChecks>::format_dtmf(symbols)).map_err(|e: ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol| <::lk_agents_abi::voice::workflows::UnknownDtmfSymbol as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/dtmf.rs.
type DtmfIoImpl = <crate::generated::implementations::Selected as super::Implementations>::DtmfIo;

fn _extract_dtmf_io_receive_dtmf_num_digits(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DtmfIoReceiveDtmfNumDigitsInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "num_digits")
}
type DtmfIoReceiveDtmfNumDigitsInput = usize;

fn _extract_dtmf_io_receive_dtmf_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<DtmfIoReceiveDtmfTimeoutInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timeout")
}
type DtmfIoReceiveDtmfTimeoutInput = f64;

#[pyfunction]
#[pyo3(signature = (room, events))]
pub fn send_dtmf_events<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    events: ::std::vec::Vec<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let room = crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
        room.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "room",
            ": expected a native room session"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
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
    #[pyo3(from_py_with = _extract_dtmf_io_receive_dtmf_num_digits)]
    num_digits: DtmfIoReceiveDtmfNumDigitsInput,
    #[pyo3(from_py_with = _extract_dtmf_io_receive_dtmf_timeout)]
    timeout: DtmfIoReceiveDtmfTimeoutInput,
    stop_event: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let room = crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
        room.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "room",
            ": expected a native room session"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
        let out = <DtmfIoImpl as ::lk_agents_abi::voice::workflows::dtmf::DtmfIo>::receive_dtmf(room, num_digits, timeout, stop_event).await.map_err(|e: ::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError| <::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/voice/workflows/warm_transfer.rs.
type WarmTransferImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WarmTransfer;

fn _extract_warm_transfer_new_ringing_timeout(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WarmTransferNewRingingTimeoutInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "ringing_timeout")).map(Some)
    }
}
type WarmTransferNewRingingTimeoutInput = ::std::option::Option<f64>;

/// A transfer plan and the runtime ownership of its current dial.
#[pyclass(frozen, name = "WarmTransfer", module = "livekit.agents._core")]
pub struct WarmTransfer(pub(crate) ::std::sync::Arc<WarmTransferImpl>);

#[pymethods]
impl WarmTransfer {
    #[new]
    #[pyo3(signature = (*, sip_call_to = None, sip_trunk_id = crate::py_values::Omittable::NOT_GIVEN, has_sip_connection = false, sip_number = crate::py_values::Omittable::NOT_GIVEN, sip_headers = None, dtmf = None, ringing_timeout = None, instructions = crate::py_values::Omittable::NOT_GIVEN, conversation = None, env_sip_trunk = None, env_sip_number = "".to_owned(), twilio = None))]
    fn new(
        py: Python<'_>,
        sip_call_to: ::std::option::Option<::std::string::String>,
        sip_trunk_id: crate::py_values::Omittable<'_>,
        has_sip_connection: bool,
        sip_number: crate::py_values::Omittable<'_>,
        sip_headers: ::std::option::Option<
            ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
        >,
        dtmf: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_warm_transfer_new_ringing_timeout)]
        ringing_timeout: WarmTransferNewRingingTimeoutInput,
        instructions: crate::py_values::Omittable<'_>,
        conversation: ::std::option::Option<
            ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
        >,
        env_sip_trunk: ::std::option::Option<::std::string::String>,
        env_sip_number: ::std::string::String,
        twilio: Option<Bound<'_, PyAny>>,
    ) -> PyResult<WarmTransfer> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let sip_trunk_id = sip_trunk_id.map(|value| {
            if value.is_none() {
                Ok(None)
            } else {
                (value.extract::<::std::string::String>()).map(Some)
            }
        })?;
        let sip_number = sip_number.map(|value| value.extract::<::std::string::String>())?;
        let instructions = instructions.map(|value| {
            crate::generated::records::workflows::instructions_arg_from_py(value.as_any())
        })?;
        let twilio = twilio
            .map(|v| crate::generated::records::workflows::twilio_dial_from_py(v.as_any()))
            .transpose()?;
        let out = py.detach(|| <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::new(&env, sip_call_to, sip_trunk_id, has_sip_connection, sip_number, sip_headers, dtmf, ringing_timeout, instructions, conversation, env_sip_trunk, env_sip_number, twilio)).map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused| <::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(WarmTransfer(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn instructions(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::instructions(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    fn tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::tools(&self.0));
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| {
                        ::pyo3::PyResult::Ok({
                            let d = ::pyo3::types::PyDict::new(py);
                            for (k, x) in x {
                                d.set_item(
                                    ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                                    crate::py_values::value_to_py(py, &x)?,
                                )?;
                            }
                            d.into_any()
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    #[pyo3(signature = (host, caller))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        host: Bound<'_, PyAny>,
        caller: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let host = crate::generated::voice::TransferHostProxy::from_host(host.as_any())?;
        let caller =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
                caller.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "caller",
                    ": expected a native room session"
                ))
            })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::run(&this, host, caller)
                .await.map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed| <::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed as crate::py_errors::HostException>::to_py_err(&e))?;
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
        crate::py_calls::check_pid()?;
        let arguments = crate::py_values::value_from_py(arguments.as_any(), "arguments")?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <WarmTransferImpl as ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer>::call_tool(&this, name, arguments)
                .await.map_err(|e: ::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed| <::lk_agents_abi::voice::workflows::warm_transfer::TransferFailed as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/console.rs.
type ConsoleIoImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ConsoleIo;

/// `lk agent console`'s socket, opened when the session starts.
#[pyclass(frozen, name = "ConsoleIo", module = "livekit.agents._core")]
pub struct ConsoleIo(pub(crate) ::std::sync::Arc<ConsoleIoImpl>);

#[pymethods]
impl ConsoleIo {
    #[new]
    #[pyo3(signature = (addr))]
    fn new(py: Python<'_>, addr: ::std::string::String) -> PyResult<ConsoleIo> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| {
            <ConsoleIoImpl as ::lk_agents_abi::voice::console::ConsoleIo>::new(&env, addr)
        });
        ::pyo3::PyResult::Ok(ConsoleIo(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::console::ConsolePort`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::console::ConsolePort>(),
        read: |obj| {
            let this = obj.cast::<ConsoleIo>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::console::ConsolePort> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomAudioInputImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RoomAudioInput;

/// `_core.RoomAudioInput`: the room's microphone leg, taken as itself wherever the host passes
/// an audio input.
#[pyclass(frozen, name = "RoomAudioInput", module = "livekit.agents._core")]
pub struct RoomAudioInput(pub(crate) ::std::sync::Arc<RoomAudioInputImpl>);

#[pymethods]
impl RoomAudioInput {
    #[getter]
    fn reading<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <RoomAudioInputImpl as ::lk_agents_abi::voice::room_owner::RoomAudioInput>::reading(
                &self.0,
            )
        });
        out.map(|value| ::pyo3::PyResult::Ok(crate::py_values::str_to_py(py, &value))).transpose()
    }

    #[pyo3(name = "next_frame")]
    fn read_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <RoomAudioInputImpl as ::lk_agents_abi::voice::room_owner::RoomAudioInput>::read_frame(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::rtc_frames::audio_frame_to_py(py, value)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <RoomAudioInputImpl as ::lk_agents_abi::voice::room_owner::RoomAudioInput>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::AudioInput`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::AudioInput>(),
        read: |obj| {
            let this = obj.cast::<RoomAudioInput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomVideoInputImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RoomVideoInput;

/// `_core.RoomVideoInput`: the room's video leg, taken as itself wherever the host passes a video
/// input.
#[pyclass(frozen, name = "RoomVideoInput", module = "livekit.agents._core")]
pub struct RoomVideoInput(pub(crate) ::std::sync::Arc<RoomVideoInputImpl>);

#[pymethods]
impl RoomVideoInput {
    #[getter]
    fn reading<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <RoomVideoInputImpl as ::lk_agents_abi::voice::room_owner::RoomVideoInput>::reading(
                &self.0,
            )
        });
        out.map(|value| ::pyo3::PyResult::Ok(crate::py_values::str_to_py(py, &value))).transpose()
    }

    #[pyo3(name = "next_frame")]
    fn read_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <RoomVideoInputImpl as ::lk_agents_abi::voice::room_owner::RoomVideoInput>::read_frame(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| crate::py_objects::object_ref_to_py(py, value)).transpose()?,
                    py,
                )
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <RoomVideoInputImpl as ::lk_agents_abi::voice::room_owner::RoomVideoInput>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::voice::io::IoError| <::lk_agents_abi::voice::io::IoError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::voice::io::VideoInput`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::io::VideoInput>(),
        read: |obj| {
            let this = obj.cast::<RoomVideoInput>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type CoreRoomSessionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RoomSession;

/// `CoreRoomSession`: what a session's room I/O is built from, handed to the host and taken
/// back as itself.
#[pyclass(frozen, name = "_core_RoomSession", module = "livekit.agents._core")]
pub struct CoreRoomSession(pub(crate) ::std::sync::Arc<CoreRoomSessionImpl>);

#[pymethods]
impl CoreRoomSession {}

// Wherever the host passes a `::lk_agents_abi::voice::room_owner::SessionRoom`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(),
        read: |obj| {
            let this = obj.cast::<CoreRoomSession>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::voice::room_owner::SessionRoom> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/room_owner.rs.
type RoomRuntimeImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RoomRuntime;

fn _extract_room_runtime_wait_for_participant_kinds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RoomRuntimeWaitForParticipantKindsInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::number_from_py::<i32>(item.as_any(), "kinds")
                })
                .collect::<PyResult<Vec<_>>>()
        }))
        .map(Some)
    }
}
type RoomRuntimeWaitForParticipantKindsInput = ::std::option::Option<::std::vec::Vec<i32>>;

fn _extract_room_runtime_wait_for_track_publication_kinds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<RoomRuntimeWaitForTrackPublicationKindsInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::number_from_py::<i32>(item.as_any(), "kinds")
                })
                .collect::<PyResult<Vec<_>>>()
        }))
        .map(Some)
    }
}
type RoomRuntimeWaitForTrackPublicationKindsInput = ::std::option::Option<::std::vec::Vec<i32>>;

#[pyfunction]
#[pyo3(signature = (room))]
pub fn _core_create_room_session(
    py: Python<'_>,
    room: Option<Bound<'_, PyAny>>,
) -> PyResult<super::voice::CoreRoomSession> {
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    let room =
        room.map(|v| crate::generated::voice::RtcRoomProxy::from_host(v.as_any())).transpose()?;
    let out = py.detach(|| {
        <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::create_room_session(
            &env, room,
        )
    });
    ::pyo3::PyResult::Ok(super::voice::CoreRoomSession(::std::sync::Arc::new(out)))
}

/// Wait for the supplied room's connection, avatar participant and subscribed video track.
#[pyfunction]
#[pyo3(signature = (room, identity))]
pub fn wait_avatar_join<'py>(
    py: Python<'py>,
    room: Bound<'_, PyAny>,
    identity: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let room = crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
        room.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "room",
            ": expected a native room session"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_avatar_join(room, identity).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
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
    kinds: RoomRuntimeWaitForParticipantKindsInput,
    include_local: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let room = crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
        room.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "room",
            ": expected a native room session"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_participant(room, identity, kinds, include_local).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| {
            ::pyo3::IntoPyObjectExt::into_py_any(
                out.map(|value| ::pyo3::PyResult::Ok(crate::py_values::str_to_py(py, &value)))
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
    crate::py_calls::check_pid()?;
    let room = crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
        room.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "room",
            ": expected a native room session"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_agent(room, name).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| {
            ::pyo3::IntoPyObjectExt::into_py_any(crate::py_values::str_to_py(py, &out), py)
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
    kinds: RoomRuntimeWaitForTrackPublicationKindsInput,
    include_local: bool,
    subscribed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let room = crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(
        room.as_any(),
    )
    .ok_or_else(|| {
        ::pyo3::exceptions::PyTypeError::new_err(concat!(
            "room",
            ": expected a native room session"
        ))
    })?;
    crate::py_calls::future_into_py(py, async move {
        let out = <RoomRuntimeImpl as ::lk_agents_abi::voice::room_owner::RoomRuntime>::wait_for_track_publication(room, identity, kinds, include_local, subscribed).await.map_err(|e: ::lk_agents_abi::voice::rtc::RtcError| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| {
            ::pyo3::IntoPyObjectExt::into_py_any(
                {
                    let (x0, x1) = out;
                    ::pyo3::types::PyTuple::new(
                        py,
                        [
                            match x0 {
                                None => py.None().into_bound(py),
                                Some(x) => crate::py_values::str_to_py(py, &x),
                            },
                            crate::py_values::str_to_py(py, &x1),
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
    notifier: crate::py_calls::Notifier,
}

impl ClearBufferListenerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::avatar::ClearBufferListener,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::avatar::ClearBufferListener,
        >(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
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
                    None,
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
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver,
        >(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(AvatarAudioReceiverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::avatar::AvatarAudioReceiver for AvatarAudioReceiverProxy {
    fn open_audio(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarAudioReceiver._core_open_audio",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_open_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn next_audio(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "AvatarAudioReceiver._core_next_audio",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 4 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 4 items", "AvatarAudioReceiver._core_next_audio"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; Ok(((crate::py_values::number_from_py::<u64>(item_0.as_any(), "AvatarAudioReceiver._core_next_audio"))?, (item_1.extract::<bool>())?, (if item_2.is_none() { Ok(None) } else { (crate::generated::records::io_events::avatar_playback_segment_from_py(item_2.as_any())).map(Some) })?, (if item_3.is_none() { Ok(None) } else { (crate::generated::records::avatar::audio_receiver_frame_from_py(item_3.as_any())).map(Some) })?,)) })()).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn notify_playback_started(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        segment: ::lk_agents_abi::voice::io::AvatarPlaybackSegment,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let segment_arg = segment;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarAudioReceiver._core_notify_playback_started",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::io_events::avatar_playback_segment_to_py(
                        py,
                        segment_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_notify_playback_started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn notify_playback_finished(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "AvatarAudioReceiver._core_notify_playback_finished",
            &self.context,
            &[],
            cx,
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
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn aclose(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarAudioReceiver._core_aclose",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_aclose")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
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
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator,
        >(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(AvatarVideoGeneratorProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::avatar::AvatarVideoGenerator for AvatarVideoGeneratorProxy {
    fn push_audio(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        frame: ::std::option::Option<::lk_agents_abi::utils::video::MediaFrame>,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarVideoGenerator._core_push_audio",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [match frame_arg {
                    None => py.None().into_bound(py),
                    Some(x) => crate::generated::records::avatar::media_frame_to_py(py, x)?,
                }];
                Ok((
                    host.bind(py).getattr("_core_push_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn clear_buffer(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarVideoGenerator._core_clear_buffer",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_clear_buffer")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn open_frames(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::errors::ProviderError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarVideoGenerator._core_open_frames",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_open_frames")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn next_frame(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::utils::video::AvatarFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AvatarVideoGenerator._core_next_frame",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::avatar::avatar_frame_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
    notifier: crate::py_calls::Notifier,
}

impl FrameStreamProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::background_audio::FrameStream,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::background_audio::FrameStream>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::background_audio::FrameStream,
        >(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::AudioFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "FrameStream._core_next",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_next")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::rtc_frames::audio_frame_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
                    None,
                ))
            }),
        );
    }

    fn close(&self, cx: &::lk_agents_abi::telemetry::Cx) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "FrameStream._core_close",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_close")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
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
    notifier: crate::py_calls::Notifier,
}

impl PlaybackListenerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::PlaybackListener>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
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
            "PlaybackListener.on_playback_started",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_playback_started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    Some({
                        let kwargs = ::pyo3::types::PyDict::new(py);
                        kwargs.set_item(
                            "segment",
                            crate::generated::records::io_events::playback_segment_to_py(
                                py,
                                segment_arg,
                            )?,
                        )?;
                        kwargs.set_item(
                            "created_at",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(created_at_arg, py)?,
                        )?;
                        kwargs
                    }),
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
            "PlaybackListener.on_playback_progressed",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_playback_progressed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    Some({
                        let kwargs = ::pyo3::types::PyDict::new(py);
                        kwargs.set_item(
                            "segment",
                            crate::generated::records::io_events::playback_segment_to_py(
                                py,
                                segment_arg,
                            )?,
                        )?;
                        kwargs.set_item(
                            "started_at",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(started_at_arg, py)?,
                        )?;
                        kwargs.set_item(
                            "offset",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(offset_arg, py)?,
                        )?;
                        kwargs.set_item(
                            "duration",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(duration_arg, py)?,
                        )?;
                        kwargs
                    }),
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
            "PlaybackListener.on_playback_finished",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_playback_finished")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    Some({
                        let kwargs = ::pyo3::types::PyDict::new(py);
                        kwargs.set_item(
                            "segment",
                            crate::generated::records::io_events::playback_segment_to_py(
                                py,
                                segment_arg,
                            )?,
                        )?;
                        kwargs.set_item(
                            "playback_position",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(playback_position_arg, py)?,
                        )?;
                        kwargs.set_item(
                            "interrupted",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(interrupted_arg, py)?,
                        )?;
                        kwargs.set_item(
                            "synchronized_transcript",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(
                                synchronized_transcript_arg,
                                py,
                            )?,
                        )?;
                        kwargs
                    }),
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
    notifier: crate::py_calls::Notifier,
    label: ::std::string::String,
}

impl AudioInputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::AudioInput>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let label = {
            let v = obj.getattr("label")?;
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::AudioFrame>,
            ::lk_agents_abi::voice::io::IoError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AudioInput.livekit.agents._bindings.next_audio_frame",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [host.bind(py).clone()];
                Ok((
                    py.import("livekit.agents._bindings")?.getattr("next_audio_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::rtc_frames::audio_frame_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioInput.on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioInput.on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
    notifier: crate::py_calls::Notifier,
    label: ::std::string::String,
}

impl VideoInputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::VideoInput>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let label = {
            let v = obj.getattr("label")?;
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::voice::rtc::VideoFrameRef>,
            ::lk_agents_abi::voice::io::IoError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "VideoInput.livekit.agents._bindings.next_video_frame",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [host.bind(py).clone()];
                Ok((
                    py.import("livekit.agents._bindings")?.getattr("next_video_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::generated::records::rtc_wire::video_frame_ref_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "VideoInput.on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "VideoInput.on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
    notifier: crate::py_calls::Notifier,
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let label = {
            let v = obj.getattr("label")?;
            v.extract::<::std::string::String>()?
        };
        let tracker = {
            let v = obj.getattr("_tracker")?;
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::PlaybackTracker>(v.as_any())
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "AudioOutput._tracker",
                        ": expected a `_core.PlaybackTracker`"
                    ))
                })?
        };
        let capabilities = {
            let v = crate::py_calls::call(
                &obj.py()
                    .import("livekit.agents._bindings")?
                    .getattr("audio_output_capabilities")?,
                (obj,),
                None,
            )?;
            crate::generated::records::rtc_wire::audio_output_capabilities_from_py(v.as_any())?
        };
        let sample_rate = {
            let v = obj.getattr("sample_rate")?;
            if v.is_none() {
                Ok(None)
            } else {
                (crate::py_values::number_from_py::<u32>(v.as_any(), "AudioOutput.sample_rate"))
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::io_events::playback_segment_to_py(
                        py,
                        segment_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("begin_segment")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn sample_rate(&self) -> ::std::option::Option<u32> {
        ::std::clone::Clone::clone(&self.sample_rate)
    }

    fn capture<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        frame: &::lk_agents_abi::AudioFrame,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AudioOutput.capture_frame",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_frames::audio_frame_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py).getattr("capture_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn flush_segment(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput.flush",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("flush")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
    }

    fn clear_buffer(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput.clear_buffer",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("clear_buffer")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput.on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput.on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn pause(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput.pause",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("pause")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
    }

    fn resume(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "AudioOutput.resume",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("resume")?, ::pyo3::types::PyTuple::new(py, args)?, None))
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
    notifier: crate::py_calls::Notifier,
    label: ::std::string::String,
}

impl TextOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::io::TextOutput>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let label = {
            let v = obj.getattr("label")?;
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
        cx: &::lk_agents_abi::telemetry::Cx,
        text: &::lk_agents_abi::TranscriptChunk,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = ::std::clone::Clone::clone(text);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TextOutput.capture_transcript",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::transcript::transcript_chunk_to_py(py, text_arg)?];
                Ok((
                    host.bind(py).getattr("capture_transcript")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn capture_text<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        text: &str,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::io::IoError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TextOutput.capture_text",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((
                    host.bind(py).getattr("capture_text")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::io::IoError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn flush(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TextOutput.flush",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("flush")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
    }

    fn on_attached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TextOutput.on_attached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_attached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn on_detached(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "TextOutput.on_detached",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("on_detached")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::SessionNodes>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::nodes::SessionNodes>(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(SessionNodesProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::nodes::SessionNodes for SessionNodesProxy {
    fn llm<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        req: &::lk_agents_abi::ChatRequest,
        calls: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<
                dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::voice::nodes::NodeChunk>,
            >,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let req_arg = ::std::clone::Clone::clone(req);
        let calls_arg = calls;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionNodes.llm",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::Bound::new(py, crate::generated::llm::ChatRequest(::std::sync::Arc::new(req_arg)))?.into_any(), <dyn ::lk_agents_abi::voice::nodes::ProviderCalls as super::NativeToHost>::to_host(py, calls_arg)?];
                Ok((host.bind(py).getattr("llm")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.llm: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::open(py, o, context, |o| crate::generated::records::agent_nodes::node_chunk_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::voice::nodes::NodeChunk>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn tts<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        text: ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::std::string::String>>,
        calls: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<
                dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SynthesizedAudio>,
            >,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text;
        let calls_arg = calls;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionNodes.tts",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::Bound::new(py, NodeText(::lk_agents_runtime::net::source::SharedStream::new(text_arg, ::lk_agents_runtime::process::host_clock())))?.into_any(), <dyn ::lk_agents_abi::voice::nodes::ProviderCalls as super::NativeToHost>::to_host(py, calls_arg)?];
                Ok((host.bind(py).getattr("tts")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.tts: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::open(py, o, context, |o| crate::generated::records::tts::synthesized_audio_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SynthesizedAudio>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn stt<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        audio: ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::AudioFrame>>,
        calls: ::std::sync::Arc<dyn ::lk_agents_abi::voice::nodes::ProviderCalls>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SpeechEvent>>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let audio_arg = audio;
        let calls_arg = calls;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionNodes.stt",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::Bound::new(py, NodeAudio(::lk_agents_runtime::net::source::SharedStream::new(audio_arg, ::lk_agents_runtime::process::host_clock())))?.into_any(), <dyn ::lk_agents_abi::voice::nodes::ProviderCalls as super::NativeToHost>::to_host(py, calls_arg)?];
                Ok((host.bind(py).getattr("stt")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.stt: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::open(py, o, context, |o| crate::generated::records::stt::speech_event_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::ProviderStream<::lk_agents_abi::SpeechEvent>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn transcription<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        text: ::std::boxed::Box<
            dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::TranscriptChunk>,
        >,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::TranscriptChunk>>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionNodes.transcription",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::Bound::new(
                    py,
                    NodeTranscript(::lk_agents_runtime::net::source::SharedStream::new(
                        text_arg,
                        ::lk_agents_runtime::process::host_clock(),
                    )),
                )?
                .into_any()];
                Ok((
                    host.bind(py).getattr("transcription")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionNodes.transcription: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (crate::py_streams::HostStream::open(py, o, context, |o| crate::generated::records::transcript::transcript_chunk_from_py(o.as_any()), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::TranscriptChunk>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

/// One stream of `::std::string::String` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "NodeText", module = "livekit.agents._core")]
pub struct NodeText(::lk_agents_runtime::net::source::SharedStream<::std::string::String>);

#[pymethods]
impl NodeText {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::recv_native(&self.0, py, |py, item| {
            ::pyo3::IntoPyObjectExt::into_py_any(item, py)
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::close_native(&self.0, py)
    }
}

/// One stream of `::lk_agents_abi::AudioFrame` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "NodeAudio", module = "livekit.agents._core")]
pub struct NodeAudio(::lk_agents_runtime::net::source::SharedStream<::lk_agents_abi::AudioFrame>);

#[pymethods]
impl NodeAudio {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::recv_native(&self.0, py, |py, item| {
            Ok(crate::generated::records::rtc_frames::audio_frame_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::close_native(&self.0, py)
    }
}

/// One stream of `::lk_agents_abi::TranscriptChunk` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "NodeTranscript", module = "livekit.agents._core")]
pub struct NodeTranscript(
    ::lk_agents_runtime::net::source::SharedStream<::lk_agents_abi::TranscriptChunk>,
);

#[pymethods]
impl NodeTranscript {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::recv_native(&self.0, py, |py, item| {
            Ok(crate::generated::records::transcript::transcript_chunk_to_py(py, item)?.unbind())
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        crate::py_streams::close_native(&self.0, py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::RtcRoom`] over a host `livekit.agents.voice.room_io.Room`.
#[derive(Clone)]
pub struct RtcRoomProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    local: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>,
    event_bus: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>,
    rpc_router: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ports::RpcRoutes>,
}

impl RtcRoomProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RtcRoom>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(obj)
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
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RtcRoom>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let local = {
            let v = obj.getattr("local")?;
            crate::generated::voice::LocalParticipantProxy::from_host(v.as_any())?
        };
        let event_bus = {
            let v = obj.getattr("bus")?;
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RoomEventSink>(v.as_any())
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "RtcRoom.bus",
                        ": expected a _core.RoomEventBus"
                    ))
                })?
        };
        let rpc_router = {
            let v = obj.getattr("rpc_router")?;
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::ports::RpcRoutes>(
                v.as_any(),
            )
            .ok_or_else(|| {
                ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "RtcRoom.rpc_router",
                    ": expected the room's native RPC router"
                ))
            })?
        };
        Ok(RtcRoomProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            local,
            event_bus,
            rpc_router,
        })
    }
}

impl ::lk_agents_abi::voice::rtc::RtcRoom for RtcRoomProxy {
    fn create_audio_source(
        &self,
        opts: ::lk_agents_abi::voice::room_io::TrackPublishOptions,
    ) -> ::std::result::Result<
        ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::track_publish_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_create_audio_source")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("RtcRoom._core_create_audio_source: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (super::voice::AudioSourceProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn create_video_source(
        &self,
        opts: ::lk_agents_abi::voice::rtc::media::VideoTrackOptions,
    ) -> ::std::result::Result<
        ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::VideoSource>,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let opts_arg = opts;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::video_track_options_to_py(py, opts_arg)?];
                Ok((
                    host.bind(py).getattr("_core_create_video_source")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("RtcRoom._core_create_video_source: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (super::voice::VideoSourceProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn name(
        &self,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "name").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "connection_state").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<i32>().map(|v| <::lk_agents_abi::voice::rtc::ConnectionState as ::std::convert::TryFrom<i32>>::try_from(v).unwrap_or_default())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_values::str_to_py(py, &id_arg)];
                Ok((
                    host.bind(py).getattr("_core_participant")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("RtcRoom._core_participant: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (if o.is_none() {
                ::pyo3::PyResult::Ok(None)
            } else {
                (super::voice::RemoteParticipantProxy::in_context(o.as_any(), context)).map(Some)
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_remote_participants")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("RtcRoom._core_remote_participants: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        super::voice::RemoteParticipantProxy::in_context(item.as_any(), context)
                    })
                    .collect::<::pyo3::PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn event_bus(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::RoomEventSink> {
        &self.event_bus
    }

    fn rpc_router(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::ports::RpcRoutes> {
        &self.rpc_router
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?, <dyn ::lk_agents_abi::voice::rtc::ByteStreamIngress as super::NativeToHost>::to_host(py, streams_arg)?];
                Ok((
                    host.bind(py).getattr("_core_register_byte_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn unregister_byte_stream_handler(
        &self,
        topic: &str,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("unregister_byte_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?, <dyn ::lk_agents_abi::voice::rtc::TextStreamIngress as super::NativeToHost>::to_host(py, streams_arg)?];
                Ok((
                    host.bind(py).getattr("_core_register_text_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn unregister_text_stream_handler(
        &self,
        topic: &str,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let topic_arg = topic.to_owned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(topic_arg, py)?];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("unregister_text_stream_handler")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::LocalParticipant>(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(LocalParticipantProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::LocalParticipant for LocalParticipantProxy {
    fn publish_audio_source(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        source: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>,
        opts: ::lk_agents_abi::voice::room_io::TrackPublishOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let source_arg = source;
        let opts_arg = opts;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant._core_publish_audio_source",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    super::voice::AudioSourceProxy::to_host(
                        py,
                        ::std::sync::Arc::clone(&source_arg),
                    )?,
                    crate::generated::records::rtc_wire::track_publish_options_to_py(py, opts_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_publish_audio_source")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("LocalParticipant._core_publish_audio_source: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (super::voice::TrackPublicationProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn publish_video_source(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        source: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::VideoSource>,
        opts: ::lk_agents_abi::voice::rtc::media::VideoTrackOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::TrackPublication>,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let source_arg = source;
        let opts_arg = opts;
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant._core_publish_video_source",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    super::voice::VideoSourceProxy::to_host(
                        py,
                        ::std::sync::Arc::clone(&source_arg),
                    )?,
                    crate::generated::records::rtc_wire::video_track_options_to_py(py, opts_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_publish_video_source")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("LocalParticipant._core_publish_video_source: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (super::voice::TrackPublicationProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn identity(
        &self,
    ) -> ::std::result::Result<
        ::lk_agents_abi::voice::room_io::ParticipantIdentity,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "identity").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::py_values::str_from_py(o.as_any())).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "kind").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<i32>().map(|v| <::lk_agents_abi::voice::room_io::ParticipantKind as ::std::convert::TryFrom<i32>>::try_from(v).unwrap_or_default())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn sid(
        &self,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "sid").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_track_publications")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("LocalParticipant._core_track_publications: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        super::voice::TrackPublicationProxy::in_context(item.as_any(), context)
                    })
                    .collect::<::pyo3::PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn publish_audio_track<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant._core_publish_audio_track",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::track_publish_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_publish_audio_track")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("LocalParticipant._core_publish_audio_track: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err("LocalParticipant._core_publish_audio_track: expected a tuple of 2 items")); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((super::voice::AudioSourceProxy::in_context(item_0.as_any(), context))?, (super::voice::TrackPublicationProxy::in_context(item_1.as_any(), context))?,)) })()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn unpublish_track<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        sid: &::lk_agents_abi::voice::rtc::TrackSid,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let sid_arg = ::std::clone::Clone::clone(sid);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant.sdk.unpublish_track",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_values::str_to_py(py, &sid_arg)];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("unpublish_track")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn set_attributes<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        attrs: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let attrs_arg = attrs;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant.sdk.set_attributes",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(attrs_arg, py)?];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("set_attributes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn publish_transcription<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        transcription: ::lk_agents_abi::voice::rtc::Transcription,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let transcription_arg = transcription;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant._core_publish_transcription",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::transcription_to_py(
                        py,
                        transcription_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_publish_transcription")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn stream_text<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant._core_stream_text",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::stream_text_options_to_py(py, opts_arg)?];
                Ok((
                    host.bind(py).getattr("_core_stream_text")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("LocalParticipant._core_stream_text: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (super::voice::TextStreamWriterProxy::owned_in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn stream_bytes<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "LocalParticipant._core_stream_bytes",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::stream_bytes_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_stream_bytes")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let Some(context) = context.as_ref() else {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("LocalParticipant._core_stream_bytes: the call captured no host context")));
                };
                let context = &out.returned_context(py, context).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                let o = out.value.bind(py);
                let out = (super::voice::ByteStreamWriterProxy::owned_in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn perform_rpc<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "LocalParticipant._core_perform_rpc",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    crate::py_values::str_to_py(py, &destination_arg),
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(payload_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_perform_rpc")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "LocalParticipant._core_perform_rpc"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::py_values::number_from_py::<u32>(item_0.as_any(), "LocalParticipant._core_perform_rpc"))?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
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
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn unregister_rpc_method(
        &self,
        method: ::std::string::String,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let method_arg = method;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(method_arg, py)?];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("unregister_rpc_method")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn publish_dtmf<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "LocalParticipant.sdk.publish_dtmf",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("publish_dtmf")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    Some({
                        let kwargs = ::pyo3::types::PyDict::new(py);
                        kwargs.set_item(
                            "code",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(code_arg, py)?,
                        )?;
                        kwargs.set_item(
                            "digit",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(digit_arg, py)?,
                        )?;
                        kwargs
                    }),
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::ByteStreamWriter`] over a host `livekit.rtc.ByteStreamWriter`.
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>(obj)
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

    /// A host object a call returned for the core to own, run in a copy of that call's context.
    pub fn owned_in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::ByteStreamWriter>> {
        Ok(::std::boxed::Box::new(Self::take(obj, Some(context))?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let id = {
            let v = obj.getattr("info")?.getattr("stream_id")?;
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
        cx: &::lk_agents_abi::telemetry::Cx,
        data: &[u8],
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let data_arg = data.to_vec();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ByteStreamWriter.write",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(data_arg, py)?];
                Ok((host.bind(py).getattr("write")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close(
        self: ::std::boxed::Box<Self>,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        'static,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "ByteStreamWriter.aclose",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("aclose")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::TextStreamWriter`] over a host `livekit.rtc.TextStreamWriter`.
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>(obj)
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

    /// A host object a call returned for the core to own, run in a copy of that call's context.
    pub fn owned_in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::TextStreamWriter>> {
        Ok(::std::boxed::Box::new(Self::take(obj, Some(context))?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(TextStreamWriterProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::TextStreamWriter for TextStreamWriterProxy {
    fn write<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        text: &str,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TextStreamWriter.write",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("write")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn close(
        self: ::std::boxed::Box<Self>,
        cx: &::lk_agents_abi::telemetry::Cx,
        attributes: ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    ) -> ::lk_agents_abi::BoxFuture<
        'static,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let attributes_arg = attributes;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TextStreamWriter.aclose",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("aclose")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    Some({
                        let kwargs = ::pyo3::types::PyDict::new(py);
                        kwargs.set_item(
                            "attributes",
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(attributes_arg, py)?,
                        )?;
                        kwargs
                    }),
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::AudioSource`] over a host `livekit.agents.voice.room_io.AudioSource`.
#[derive(Clone)]
pub struct AudioSourceProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host object it was taken from, handed back as itself.
    source: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl AudioSourceProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::AudioSource>(obj)
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
        let source = ::std::sync::Arc::new(obj.clone().unbind());
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(AudioSourceProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), source, context })
    }
}

impl AudioSourceProxy {
    /// The host object behind a proxy of `::lk_agents_abi::voice::rtc::AudioSource`; a native implementation has none.
    pub fn to_host<'py>(
        py: Python<'py>,
        value: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::AudioSource>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let any: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = value;
        match any.downcast::<Self>() {
            Ok(proxy) => Ok(proxy.source.bind(py).clone()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `AudioSource` has no host object to hand back",
            )),
        }
    }
}

impl ::lk_agents_abi::voice::rtc::AudioSource for AudioSourceProxy {
    fn capture<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        frame: &::lk_agents_abi::AudioFrame,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AudioSource._core_capture",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_frames::audio_frame_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py).getattr("_core_capture")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn clear_queue(&self) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_clear_queue")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn queued_duration(
        &self,
    ) -> ::std::result::Result<::std::time::Duration, ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "queued_duration").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::py_values::number_from_py::<f64>(
                o.as_any(),
                "AudioSource.sdk.queued_duration",
            )
            .and_then(|s| {
                ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                    ::pyo3::exceptions::PyValueError::new_err(format!(
                        "{}: {e}",
                        "AudioSource.sdk.queued_duration"
                    ))
                })
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn wait_for_playout<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<
            ::lk_agents_abi::voice::rtc::PlaybackInfo,
            ::lk_agents_abi::voice::rtc::RtcError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AudioSource._core_wait_for_playout",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_wait_for_playout")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (crate::generated::records::rtc_wire::playback_info_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_playback_info")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::generated::records::rtc_wire::playback_info_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn close<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "AudioSource.sdk.aclose",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("aclose")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

/// [`::lk_agents_abi::voice::rtc::RemoteParticipant`] over a host `livekit.agents.voice.room_io.RemoteParticipant`.
#[derive(Clone)]
pub struct RemoteParticipantProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::RemoteParticipant>(obj)
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
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let sid = {
            let v = obj.getattr("sdk")?.getattr("sid")?;
            v.extract::<::std::string::String>()?
        };
        let identity = {
            let v = obj.getattr("sdk")?.getattr("identity")?;
            crate::py_values::str_from_py(v.as_any())?
        };
        let kind = {
            let v = obj.getattr("sdk")?.getattr("kind")?;
            v.extract::<i32>().map(|v| <::lk_agents_abi::voice::room_io::ParticipantKind as ::std::convert::TryFrom<i32>>::try_from(v).unwrap_or_default())?
        };
        Ok(RemoteParticipantProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_active")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    host.bind(py).getattr("sdk")?,
                    ::pyo3::types::PyString::new(py, "attributes").into_any(),
                ];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
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
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_track_publications")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("RemoteParticipant._core_track_publications: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        super::voice::TrackPublicationProxy::in_context(item.as_any(), context)
                    })
                    .collect::<::pyo3::PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::TrackPublication>(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let sid = {
            let v = obj.getattr("sdk")?.getattr("sid")?;
            crate::py_values::str_from_py(v.as_any())?
        };
        let name = {
            let v = obj.getattr("sdk")?.getattr("name")?;
            v.extract::<::std::string::String>()?
        };
        let source = {
            let v = obj.getattr("sdk")?.getattr("source")?;
            v.extract::<i32>().map(|v| <::lk_agents_abi::voice::room_io::TrackSource as ::std::convert::TryFrom<i32>>::try_from(v).unwrap_or_default())?
        };
        let kind = {
            let v = obj.getattr("sdk")?.getattr("kind")?;
            crate::py_values::number_from_py::<i32>(v.as_any(), "TrackPublication.sdk.kind")?
        };
        let audio_features = {
            let v = obj.getattr("sdk")?.getattr("audio_features")?;
            v.try_iter().and_then(|items| items.map(|item| item?.extract::<i32>()).collect::<::pyo3::PyResult<Vec<i32>>>()).map(|v| v.into_iter().filter_map(|v| <::lk_agents_abi::voice::rtc::AudioTrackFeature as ::std::convert::TryFrom<i32>>::try_from(v).ok()).collect::<Vec<_>>())?
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
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_is_subscribed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
        cx: &::lk_agents_abi::telemetry::Cx,
        subscribed: bool,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let subscribed_arg = subscribed;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TrackPublication._core_set_subscribed",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(subscribed_arg, py)?];
                Ok((
                    host.bind(py).getattr("_core_set_subscribed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn wait_for_subscription<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        'a,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TrackPublication.sdk.wait_for_subscription",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("sdk")?.getattr("wait_for_subscription")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
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
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_wire::audio_stream_options_to_py(
                        py, opts_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_open_audio_stream")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("TrackPublication._core_open_audio_stream: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (super::voice::AudioInputStreamProxy::in_context(o.as_any(), context)).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    fn open_video_stream(
        &self,
    ) -> ::std::result::Result<
        ::std::boxed::Box<
            dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::voice::rtc::VideoFrameRef>,
        >,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_open_video_stream")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("TrackPublication._core_open_video_stream: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (crate::py_streams::HostStream::open(
                py,
                o,
                context,
                |o| crate::generated::records::rtc_wire::video_frame_ref_from_py(o.as_any()),
                ::lk_agents_runtime::process::host_clock(),
            )
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<
                        dyn ::lk_agents_abi::StreamOutput<
                                ::lk_agents_abi::voice::rtc::VideoFrameRef,
                            >,
                    >
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(obj)
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
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::AudioInputStream>(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(AudioInputStreamProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::rtc::AudioInputStream for AudioInputStreamProxy {
    fn frames(
        &self,
    ) -> ::std::result::Result<
        ::std::boxed::Box<
            dyn ::lk_agents_abi::StreamOutput<::lk_agents_abi::voice::rtc::AudioStreamItem>,
        >,
        ::lk_agents_abi::voice::rtc::RtcError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_frames")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("AudioInputStream._core_frames: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (crate::py_streams::HostStream::open(
                py,
                o,
                context,
                |o| crate::generated::records::rtc_wire::audio_stream_item_from_py(o.as_any()),
                ::lk_agents_runtime::process::host_clock(),
            )
            .map(|s| {
                ::std::boxed::Box::new(s)
                    as ::std::boxed::Box<
                        dyn ::lk_agents_abi::StreamOutput<
                                ::lk_agents_abi::voice::rtc::AudioStreamItem,
                            >,
                    >
            }))
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn process_audio<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "AudioInputStream._core_process_audio",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::types::PyList::new(
                    py,
                    frames_arg
                        .into_iter()
                        .map(|x| crate::generated::records::rtc_frames::audio_frame_to_py(py, x))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any()];
                Ok((
                    host.bind(py).getattr("_core_process_audio")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::rtc_wire::audio_stream_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/rtc/media.rs.

/// [`::lk_agents_abi::voice::rtc::VideoSource`] over a host `livekit.rtc.VideoSource`.
#[derive(Clone)]
pub struct VideoSourceProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host object it was taken from, handed back as itself.
    source: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl VideoSourceProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::VideoSource>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::VideoSource>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::VideoSource>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::rtc::VideoSource>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::rtc::VideoSource>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let source = ::std::sync::Arc::new(obj.clone().unbind());
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(VideoSourceProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), source, context })
    }
}

impl VideoSourceProxy {
    /// The host object behind a proxy of `::lk_agents_abi::voice::rtc::VideoSource`; a native implementation has none.
    pub fn to_host<'py>(
        py: Python<'py>,
        value: ::std::sync::Arc<dyn ::lk_agents_abi::voice::rtc::VideoSource>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let any: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = value;
        match any.downcast::<Self>() {
            Ok(proxy) => Ok(proxy.source.bind(py).clone()),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `VideoSource` has no host object to hand back",
            )),
        }
    }
}

impl ::lk_agents_abi::voice::rtc::VideoSource for VideoSourceProxy {
    fn capture(
        &self,
        frame: &::lk_agents_abi::utils::video::VideoFrame,
    ) -> ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::rtc_frames::video_frame_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py).getattr("capture_frame")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let () = ({
                let _ = o;
                ::pyo3::PyResult::Ok(())
            })
            .map_err(|e| {
                <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(())
        })
    }

    fn close(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<(), ::lk_agents_abi::voice::rtc::RtcError>,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "VideoSource.aclose",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("aclose")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::voice::rtc::RtcError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
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
    notifier: crate::py_calls::Notifier,
    toolset_scopes: ::std::sync::Arc<dyn ::lk_agents_abi::llm::toolsets::ScopesPort>,
}

impl SessionHooksProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::session_effects::SessionHooks,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::session_effects::SessionHooks,
        >(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let toolset_scopes = {
            let v = crate::py_calls::read(obj, "_core_toolset_scopes")?;
            crate::py_objects::native::<dyn ::lk_agents_abi::llm::toolsets::ScopesPort>(v.as_any())
                .ok_or_else(|| {
                    ::pyo3::exceptions::PyTypeError::new_err(concat!(
                        "SessionHooks._core_toolset_scopes",
                        ": expected a native toolset scope registry"
                    ))
                })?
        };
        Ok(SessionHooksProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
            toolset_scopes,
        })
    }
}

impl ::lk_agents_abi::voice::session_effects::SessionHooks for SessionHooksProxy {
    fn on_enter<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        agent: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHooks._core_on_enter",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, agent_arg)?];
                Ok((
                    host.bind(py).getattr("_core_on_enter")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn on_exit<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        agent: &::lk_agents_abi::HostRef,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHooks._core_on_exit",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, agent_arg)?];
                Ok((
                    host.bind(py).getattr("_core_on_exit")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn toolset_scopes(&self) -> &::std::sync::Arc<dyn ::lk_agents_abi::llm::toolsets::ScopesPort> {
        &self.toolset_scopes
    }

    fn toolset_resources(
        &self,
        tools: &::lk_agents_abi::HostRef,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::std::sync::Arc<dyn ::lk_agents_abi::llm::toolsets::ToolsetHooks>>,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let tools_arg = ::std::clone::Clone::clone(tools);
        let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, tools_arg)?];
                Ok((
                    host.bind(py).getattr("_core_toolset_resources")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let Some(context) = context.as_ref() else {
                return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("SessionHooks._core_toolset_resources: the call captured no host context")));
            };
            let context = &out.returned_context(py, context).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            let o = out.value.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        super::llm::ToolsetHooksProxy::in_context(item.as_any(), context)
                    })
                    .collect::<::pyo3::PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn tool_infos(
        &self,
        tools: &::lk_agents_abi::HostRef,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::lk_agents_abi::agent::FunctionToolInfo>,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let tools_arg = ::std::clone::Clone::clone(tools);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, tools_arg)?];
                Ok((
                    host.bind(py).getattr("_core_tool_infos")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::generated::records::agent_run::function_tool_info_from_py(
                            item.as_any(),
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn closing(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHooks._core_closing",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_closing")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn on_user_turn_completed<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "SessionHooks._core_on_user_turn_completed",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [
                    crate::py_objects::object_ref_to_py(py, agent_arg)?,
                    ::pyo3::types::PyList::new(
                        py,
                        chat_ctx_arg
                            .into_iter()
                            .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(new_transcript_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(transcript_confidence_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_user_turn_completed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 3 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 3 items", "SessionHooks._core_on_user_turn_completed"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; Ok(((if item_0.is_none() { Ok(None) } else { (item_0.extract::<::std::string::String>()).map(Some) })?, (if item_1.is_none() { Ok(None) } else { (item_1.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::chat_items::chat_item_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())).map(Some) })?, (item_2.extract::<bool>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn on_user_turn_exceeded<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
        agent: &::lk_agents_abi::HostRef,
        ev: &::lk_agents_abi::voice::events::UserTurnExceededEvent,
    ) -> ::lk_agents_abi::BoxFuture<'a, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let agent_arg = ::std::clone::Clone::clone(agent);
        let ev_arg = ::std::clone::Clone::clone(ev);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHooks._core_on_user_turn_exceeded",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::py_objects::object_ref_to_py(py, agent_arg)?,
                    crate::generated::records::events::user_turn_exceeded_event_to_py(py, ev_arg)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_on_user_turn_exceeded")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn function_tools_executed(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "SessionHooks._core_function_tools_executed",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::types::PyList::new(
                        py,
                        function_calls_arg
                            .into_iter()
                            .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    ::pyo3::types::PyList::new(
                        py,
                        function_call_outputs_arg
                            .into_iter()
                            .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                ];
                Ok((
                    host.bind(py).getattr("_core_function_tools_executed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<bool>() }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn filler_source<'a>(
        &'a self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "SessionHooks._core_filler_source",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::py_objects::object_ref_to_py(py, source_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(step_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_filler_source")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (crate::generated::records::session_effects::filler_reply_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
                        Some(x) => crate::py_objects::captured_error_to_py(py, x)?,
                    },
                ];
                Ok((
                    host.bind(py).getattr("_core_hook_failed")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
    notifier: crate::py_calls::Notifier,
}

impl SessionEventsProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::session_effects::SessionEvents,
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
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::voice::session_effects::SessionEvents,
        >(obj)
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(SessionEventsProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::voice::session_effects::SessionEvents for SessionEventsProxy {
    fn event(
        &self,
        wire: ::lk_agents_abi::voice::events::HostWire,
        state: ::std::sync::Arc<dyn ::lk_agents_abi::voice::session_effects::DeliveredState>,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let wire_arg = wire;
        let state_arg = state;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "SessionEvents._core_event",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::session_wire::host_wire_to_py(py, wire_arg)?, <dyn ::lk_agents_abi::voice::session_effects::DeliveredState as super::NativeToHost>::to_host(py, state_arg)?];
                Ok((
                    host.bind(py).getattr("_core_event")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn history_items(
        &self,
        items: ::std::vec::Vec<::lk_agents_abi::ChatItem>,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::lk_agents_abi::HostRef>,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let items_arg = items;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::types::PyList::new(
                    py,
                    items_arg
                        .into_iter()
                        .map(|x| crate::generated::records::chat_items::chat_item_to_py(py, x))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any()];
                Ok((
                    host.bind(py).getattr("_core_history_items")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::py_objects::object_ref_from_py(item.as_any())
                    })
                    .collect::<PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn error_message(
        &self,
        error: &::lk_agents_abi::voice::events::SessionErrorWire,
    ) -> ::std::result::Result<::std::string::String, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = ::std::clone::Clone::clone(error);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::session_wire::session_error_wire_to_py(
                        py, error_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_error_message")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<::std::string::String>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_owner.rs.

/// [`::lk_agents_abi::voice::session_owner::SessionHost`] over a host `livekit.agents.voice._session_bindings.SessionHost`.
#[derive(Clone)]
pub struct SessionHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl SessionHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_owner::SessionHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::session_owner::SessionHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::voice::session_owner::SessionHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::voice::session_owner::SessionHost>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::voice::session_owner::SessionHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(SessionHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context })
    }
}

impl ::lk_agents_abi::voice::session_owner::SessionHost for SessionHostProxy {
    fn session(
        &self,
    ) -> ::std::result::Result<::lk_agents_abi::HostRef, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] =
                    [host.bind(py).clone(), ::pyo3::types::PyString::new(py, "session").into_any()];
                Ok((
                    py.import("builtins")?.getattr("getattr")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::py_objects::object_ref_from_py(o.as_any())).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn prepare(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::sync::Arc<dyn ::lk_agents_abi::voice::session_owner::SessionBindingsPort>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHost.prepare",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("prepare")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (crate::py_objects::native::<dyn ::lk_agents_abi::voice::session_owner::SessionBindingsPort>(o.as_any()).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("SessionHost.prepare", ": expected native session bindings")))).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn started(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHost.started",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("started")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn closed(&self, cx: &::lk_agents_abi::telemetry::Cx) -> ::lk_agents_abi::BoxFuture<'_, ()> {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "SessionHost.closed",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("closed")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::std::boxed::Box::pin(async move {
            if let Err(e) = started.await {
                log::warn!("SessionHost.closed failed: {e}");
            }
        })
    }

    fn bind_providers(
        &self,
        providers: ::std::collections::BTreeMap<
            ::std::string::String,
            ::std::option::Option<::lk_agents_abi::HostRef>,
        >,
    ) -> ::std::result::Result<
        ::lk_agents_abi::voice::session_owner::ProviderBindings,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let providers_arg = providers;
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [{
                    let d = ::pyo3::types::PyDict::new(py);
                    for (k, x) in providers_arg {
                        d.set_item(
                            ::pyo3::IntoPyObjectExt::into_bound_py_any(k, py)?,
                            match x {
                                None => py.None().into_bound(py),
                                Some(x) => crate::py_objects::object_ref_to_py(py, x)?,
                            },
                        )?;
                    }
                    d.into_any()
                }];
                Ok((
                    host.bind(py).getattr("bind_providers")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::generated::records::session_owner::provider_bindings_from_py(
                o.as_any(),
            ))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
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
    notifier: crate::py_calls::Notifier,
    io_state: ::lk_agents_abi::voice::workflows::IoEnableState,
}

impl TransferHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<
        ::std::sync::Arc<dyn ::lk_agents_abi::voice::workflows::warm_transfer::TransferHost>,
    > {
        if let Some(native) = crate::py_objects::native::<
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
        if let Some(native) = crate::py_objects::native::<
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
            None => crate::py_calls::capture_context(obj.py())?,
        };
        let io_state = {
            let v = crate::py_calls::read(obj, "_core_io_state")?;
            crate::generated::records::workflows::io_enable_state_from_py(v.as_any())?
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
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::HostRef>,
            ::lk_agents_abi::errors::HostError,
        >,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TransferHost._core_start_hold",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_start_hold")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (if o.is_none() { Ok(None) } else { (crate::py_objects::object_ref_from_py(o.as_any())).map(Some) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
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
                    [crate::py_objects::object_ref_to_py(py, hold_arg)?];
                Ok((
                    host.bind(py).getattr("_core_stop_hold")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
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
                Ok((
                    host.bind(py).getattr("_core_set_io")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn connect_human(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "TransferHost._core_connect_human",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(room_name_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_connect_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (crate::py_objects::native::<dyn ::lk_agents_abi::voice::room_owner::SessionRoom>(o.as_any()).ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("TransferHost._core_connect_human", ": expected a native room session")))).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn start_human(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TransferHost._core_start_human",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_start_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
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
                    None,
                ))
            }),
        );
    }

    fn disconnect_human(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TransferHost._core_disconnect_human",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_disconnect_human")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn create_sip_participant(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        request: ::lk_agents_abi::voice::workflows::SipParticipantRequest,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let request_arg = request;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TransferHost._core_create_sip_participant",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::workflows::sip_participant_request_to_py(
                        py,
                        request_arg,
                    )?];
                Ok((
                    host.bind(py).getattr("_core_create_sip_participant")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn connect_twilio_call(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "TransferHost._core_connect_twilio_call",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(room_name_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_connect_twilio_call")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = (o.extract::<::std::string::String>()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn http_post(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "TransferHost._core_http_post",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(url_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(headers_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(form_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_http_post")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let out = ((|| -> ::pyo3::PyResult<_> { let tuple = o.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 2 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 2 items", "TransferHost._core_http_post"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; Ok(((crate::py_values::number_from_py::<u16>(item_0.as_any(), "TransferHost._core_http_post"))?, (item_1.extract::<::std::string::String>())?,)) })()).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn move_participant(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
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
            &crate::py_calls::Entry,
            "TransferHost._core_move_participant",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 3] = [
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(room_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(identity_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(destination_arg, py)?,
                ];
                Ok((
                    host.bind(py).getattr("_core_move_participant")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }

    fn delete_caller_room(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<(), ::lk_agents_abi::errors::HostError>>
    {
        let host = ::std::sync::Arc::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "TransferHost._core_delete_caller_room",
            &self.context,
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py).getattr("_core_delete_caller_room")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::std::boxed::Box::pin(async move {
            let out = started.await;
            ::pyo3::Python::attach(|py| {
                let out = match out {
                    Ok(out) => out,
                    Err(e) => {
                        return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                    }
                };
                let o = out.value.bind(py);
                let () = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(())
            })
        })
    }
}

/// Each `ALL` array's names, indexed by wire value, for the host's SDK events and track options:
/// `(connection states, disconnect reasons, track sources)`. Proxy reads take SDK numbers
/// through their `sdk` declaration instead.
#[pyfunction]
pub fn rtc_wire_names(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::voice::rtc::rtc_wire_names(), py)
}

/// How many frames a `video_stream` queue holds before it evicts the oldest. Tiny on purpose: each
/// held frame pins a decoded host frame, and the newest frame is the one worth keeping. A host
/// video stream is opened with the same capacity, so both sides drop the oldest alike.
#[pyfunction]
pub fn video_queue_frames(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    crate::py_calls::check_pid()?;
    ::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES, py)
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::avatar::AvatarOutputError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Provider(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::avatar::RunnerError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closed => crate::py_errors::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Failed(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::avatar::ReceiverEnd {
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

impl crate::py_errors::HostException for ::lk_agents_abi::voice::avatar_queue::QueueClosed {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::host_exception(
            "livekit.agents.utils.aio",
            "ChanClosed",
            ::std::string::ToString::to_string(self),
        )
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::background_audio::PlaybackError {
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

impl crate::py_errors::HostException for ::lk_agents_abi::voice::io::InvalidPlaybackTime {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::io::IoError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
impl crate::py_errors::FromHostErr for ::lk_agents_abi::voice::io::IoError {
    fn from_host(py: ::pyo3::Python<'_>, err: &::pyo3::PyErr) -> Self {
        Self::Host(
            <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                py, err,
            ),
        )
    }
}
impl crate::py_errors::HostException for ::lk_agents_abi::voice::room_io::RoomOptionsError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::rtc::RtcError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Stream(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}
impl crate::py_errors::FromHostErr for ::lk_agents_abi::voice::rtc::RtcError {
    fn from_host(py: ::pyo3::Python<'_>, err: &::pyo3::PyErr) -> Self {
        Self::Host(
            <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                py, err,
            ),
        )
    }
}
impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::SessionRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Closed => crate::py_errors::host_exception(
                "livekit.agents._core",
                "SessionClosing",
                ::std::string::ToString::to_string(self),
            ),
            Self::LoopEnded => crate::py_errors::host_exception(
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

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::OpenRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::CommandRefused {
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

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::NotInterruptible {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::StartFailed {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::UnknownCloseReason {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::NotAnImage {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::VoiceTaskEnded {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Setup(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Closing => crate::py_errors::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Cancelled => crate::py_errors::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Failed(..) => crate::py_errors::host_exception(
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
impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::IdleRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        SessionClosing::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session::CommitSuperseded {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::host_exception(
            "asyncio",
            "CancelledError",
            ::std::string::ToString::to_string(self),
        )
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::session_owner::OptionsRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Command(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Session(inner) => crate::py_errors::HostException::to_py_err(inner),
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::speech_handle::SpeechRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::workflows::DateOfBirthRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::workflows::CaptureRefused {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::host_exception(
            "livekit.agents.llm.tool_context",
            "ToolError",
            ::std::string::ToString::to_string(self),
        )
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::workflows::UnknownDtmfSymbol {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::voice::workflows::dtmf::ReceiveDtmfError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::NoDigits => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            Self::UnknownKey(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            _ => crate::py_errors::host_exception(
                "livekit.agents.llm.tool_context",
                "ToolError",
                ::std::string::ToString::to_string(self),
            ),
        }
    }
}

impl crate::py_errors::HostException
    for ::lk_agents_abi::voice::workflows::warm_transfer::WarmTransferRefused
{
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
        }
    }
}

impl crate::py_errors::HostException
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
            Self::Host(inner) => crate::py_errors::HostException::to_py_err(inner),
            _ => crate::py_errors::host_exception(
                "livekit.agents.llm.tool_context",
                "ToolError",
                ::std::string::ToString::to_string(self),
            ),
        }
    }
}
