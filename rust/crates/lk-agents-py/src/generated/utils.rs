//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `utils` classes.

#![allow(clippy::too_many_arguments, reason = "a host signature takes the parameters it declares")]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/utils/byte_stream.rs.
type AudioByteStreamImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AudioByteStream;

fn _extract_audio_byte_stream_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioByteStreamNewSampleRateInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type AudioByteStreamNewSampleRateInput = u32;

fn _extract_audio_byte_stream_new_num_channels(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioByteStreamNewNumChannelsInput> {
    crate::py_values::number_from_py::<u16>(obj.as_any(), "num_channels")
}
type AudioByteStreamNewNumChannelsInput = u16;

fn _extract_audio_byte_stream_new_samples_per_channel(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioByteStreamNewSamplesPerChannelInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "samples_per_channel")).map(Some)
    }
}
type AudioByteStreamNewSamplesPerChannelInput = ::std::option::Option<u32>;

fn _extract_audio_byte_stream_push_data(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioByteStreamPushDataInput> {
    crate::py_values::pcm_bytes(obj, "data")
}
type AudioByteStreamPushDataInput = ::std::vec::Vec<u8>;

/// Buffers bytes and emits whole frames: fixed-size, or progressive (20 ms first, doubling to
/// the target) to cut TTS time-to-first-audio.
#[pyclass(name = "AudioByteStream", module = "livekit.agents.utils.media")]
pub struct AudioByteStream(pub(crate) AudioByteStreamImpl);

#[pymethods]
impl AudioByteStream {
    /// ``samples_per_channel=None`` means 100 ms, as in the reference.
    #[new]
    #[pyo3(signature = (sample_rate, num_channels, samples_per_channel = None, progressive = false))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_audio_byte_stream_new_sample_rate)]
        sample_rate: AudioByteStreamNewSampleRateInput,
        #[pyo3(from_py_with = _extract_audio_byte_stream_new_num_channels)]
        num_channels: AudioByteStreamNewNumChannelsInput,
        #[pyo3(from_py_with = _extract_audio_byte_stream_new_samples_per_channel)]
        samples_per_channel: AudioByteStreamNewSamplesPerChannelInput,
        progressive: bool,
    ) -> PyResult<AudioByteStream> {
        let out = py.detach(|| {
            <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::new(
                sample_rate,
                num_channels,
                samples_per_channel,
                progressive,
            )
        });
        ::pyo3::PyResult::Ok(AudioByteStream(out))
    }

    /// Buffer ``data`` and emit every whole frame it completes.
    #[pyo3(signature = (data))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_audio_byte_stream_push_data)]
        data: AudioByteStreamPushDataInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out =
            <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::push(
                &mut self.0,
                data,
            );
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::rtc_frames::audio_frame_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// Emit the buffered whole samples, short of a frame or not; a trailing partial sample stays
    /// for the next push.
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out =
            <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::flush(
                &mut self.0,
            );
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::rtc_frames::audio_frame_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// Seconds of audio waiting for the next frame.
    #[getter]
    fn buffered_duration(&self) -> PyResult<f64> {
        let out = <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::buffered_duration(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    /// Restart progressive sizing, keeping the buffered audio: after a mid-stream flush, the
    /// next burst starts with small frames without losing a partial sample.
    fn reset_progressive(&mut self) -> PyResult<()> {
        <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::reset_progressive(&mut self.0);
        Ok(())
    }

    /// Drop everything buffered and restart progressive sizing (called on interruption, so the
    /// next word is not a full frame late).
    fn clear(&mut self) -> PyResult<()> {
        <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::clear(
            &mut self.0,
        );
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/connection_pool.rs.
type ConnectionPoolStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ConnectionPoolState;

fn _extract_connection_pool_state_new_maximum_age(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateNewMaximumAgeInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "maximum_age")).map(Some)
    }
}
type ConnectionPoolStateNewMaximumAgeInput = ::std::option::Option<f64>;

fn _extract_connection_pool_state_connected_id(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateConnectedIdInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "id")
}
type ConnectionPoolStateConnectedIdInput = u64;

fn _extract_connection_pool_state_connected_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateConnectedNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type ConnectionPoolStateConnectedNowInput = f64;

fn _extract_connection_pool_state_connected_generation(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateConnectedGenerationInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "generation")
}
type ConnectionPoolStateConnectedGenerationInput = u64;

fn _extract_connection_pool_state_connected_acquire_time(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateConnectedAcquireTimeInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "acquire_time")).map(Some)
    }
}
type ConnectionPoolStateConnectedAcquireTimeInput = ::std::option::Option<f64>;
type ConnectionPoolStateGetOutput = ::std::option::Option<u64>;

fn _extract_connection_pool_state_get_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateGetNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type ConnectionPoolStateGetNowInput = f64;

fn _extract_connection_pool_state_put_id(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStatePutIdInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "id")
}
type ConnectionPoolStatePutIdInput = u64;

fn _extract_connection_pool_state_remove_id(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateRemoveIdInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "id")
}
type ConnectionPoolStateRemoveIdInput = u64;
type ConnectionPoolStateTakeToCloseOutput = ::std::vec::Vec<u64>;

#[pyclass(name = "ConnectionPoolState", module = "livekit.agents._core")]
pub struct ConnectionPoolState(pub(crate) ConnectionPoolStateImpl);

#[pymethods]
impl ConnectionPoolState {
    #[new]
    #[pyo3(signature = (maximum_age, refresh_on_get))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_connection_pool_state_new_maximum_age)]
        maximum_age: ConnectionPoolStateNewMaximumAgeInput,
        refresh_on_get: bool,
    ) -> PyResult<ConnectionPoolState> {
        let out = py.detach(|| <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::new(maximum_age, refresh_on_get));
        ::pyo3::PyResult::Ok(ConnectionPoolState(out))
    }

    #[getter]
    fn generation(&self) -> PyResult<u64> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::generation(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn last_acquire_time(&self) -> PyResult<f64> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::last_acquire_time(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn last_connection_reused(&self) -> PyResult<bool> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::last_connection_reused(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (id, now, generation, acquire_time))]
    fn connected(
        &mut self,
        #[pyo3(from_py_with = _extract_connection_pool_state_connected_id)]
        id: ConnectionPoolStateConnectedIdInput,
        #[pyo3(from_py_with = _extract_connection_pool_state_connected_now)]
        now: ConnectionPoolStateConnectedNowInput,
        #[pyo3(from_py_with = _extract_connection_pool_state_connected_generation)]
        generation: ConnectionPoolStateConnectedGenerationInput,
        #[pyo3(from_py_with = _extract_connection_pool_state_connected_acquire_time)] acquire_time: ConnectionPoolStateConnectedAcquireTimeInput,
    ) -> PyResult<bool> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::connected(&mut self.0, id, now, generation, acquire_time);
        ::pyo3::PyResult::Ok(out)
    }

    fn empty(&self) -> PyResult<bool> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::empty(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (now))]
    fn get(
        &mut self,
        #[pyo3(from_py_with = _extract_connection_pool_state_get_now)]
        now: ConnectionPoolStateGetNowInput,
    ) -> PyResult<ConnectionPoolStateGetOutput> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::get(&mut self.0, now);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (id))]
    fn put(
        &mut self,
        #[pyo3(from_py_with = _extract_connection_pool_state_put_id)]
        id: ConnectionPoolStatePutIdInput,
    ) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::put(&mut self.0, id);
        Ok(())
    }

    #[pyo3(signature = (id))]
    fn remove(
        &mut self,
        #[pyo3(from_py_with = _extract_connection_pool_state_remove_id)]
        id: ConnectionPoolStateRemoveIdInput,
    ) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::remove(&mut self.0, id);
        Ok(())
    }

    fn invalidate(&mut self) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::invalidate(&mut self.0);
        Ok(())
    }

    fn close_all(&mut self) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::close_all(&mut self.0);
        Ok(())
    }

    /// Every connection queued for closing, taken off the queue: the caller now owns each close.
    fn take_to_close(&mut self) -> PyResult<ConnectionPoolStateTakeToCloseOutput> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::take_to_close(&mut self.0);
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/filters.rs.
type ExpFilterImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ExpFilter;

fn _extract_exp_filter_new_alpha(obj: &Bound<'_, PyAny>) -> PyResult<ExpFilterNewAlphaInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "alpha")
}
type ExpFilterNewAlphaInput = f64;

fn _extract_exp_filter_apply_exp(obj: &Bound<'_, PyAny>) -> PyResult<ExpFilterApplyExpInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "exp")
}
type ExpFilterApplyExpInput = f64;
type ExpFilterValueOutput = ::std::option::Option<f64>;

fn _extract_exp_filter_update_base_alpha(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ExpFilterUpdateBaseAlphaInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "alpha")
}
type ExpFilterUpdateBaseAlphaInput = f64;

#[pyclass(subclass, name = "ExpFilter", module = "livekit.agents.utils.exp_filter")]
pub struct ExpFilter(pub(crate) ExpFilterImpl);

#[pymethods]
impl ExpFilter {
    #[new]
    #[pyo3(signature = (alpha, max_val = crate::py_values::Omittable::NOT_GIVEN, min_val = crate::py_values::Omittable::NOT_GIVEN, initial = crate::py_values::Omittable::NOT_GIVEN))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_exp_filter_new_alpha)] alpha: ExpFilterNewAlphaInput,
        max_val: crate::py_values::Omittable<'_>,
        min_val: crate::py_values::Omittable<'_>,
        initial: crate::py_values::Omittable<'_>,
    ) -> PyResult<ExpFilter> {
        let max_val = max_val
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "max_val"))?;
        let min_val = min_val
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "min_val"))?;
        let initial = initial
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "initial"))?;
        let out = py
            .detach(|| {
                <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::new(
                    alpha, max_val, min_val, initial,
                )
            })
            .map_err(|e: ::lk_agents_abi::FilterError| {
                <::lk_agents_abi::FilterError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(ExpFilter(out))
    }

    #[pyo3(signature = (alpha = crate::py_values::Omittable::NOT_GIVEN, initial = crate::py_values::Omittable::NOT_GIVEN, min_val = crate::py_values::Omittable::NOT_GIVEN, max_val = crate::py_values::Omittable::NOT_GIVEN))]
    fn reset(
        &mut self,
        alpha: crate::py_values::Omittable<'_>,
        initial: crate::py_values::Omittable<'_>,
        min_val: crate::py_values::Omittable<'_>,
        max_val: crate::py_values::Omittable<'_>,
    ) -> PyResult<()> {
        let alpha =
            alpha.map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "alpha"))?;
        let initial = initial
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "initial"))?;
        let min_val = min_val
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "min_val"))?;
        let max_val = max_val
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "max_val"))?;
        <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::reset(&mut self.0, alpha, initial, min_val, max_val).map_err(|e: ::lk_agents_abi::utils::filters::FilterResetError| <::lk_agents_abi::utils::filters::FilterResetError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (exp, sample = crate::py_values::Omittable::NOT_GIVEN))]
    fn apply(
        &mut self,
        #[pyo3(from_py_with = _extract_exp_filter_apply_exp)] exp: ExpFilterApplyExpInput,
        sample: crate::py_values::Omittable<'_>,
    ) -> PyResult<f64> {
        let sample = sample
            .map(|value| crate::py_values::number_from_py::<f64>(value.as_any(), "sample"))?;
        let out = <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::apply(
            &mut self.0,
            exp,
            sample,
        )
        .map_err(|e: ::lk_agents_abi::FilterError| {
            <::lk_agents_abi::FilterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn value(&self) -> PyResult<ExpFilterValueOutput> {
        let out = <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::value(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (alpha))]
    fn update_base(
        &mut self,
        #[pyo3(from_py_with = _extract_exp_filter_update_base_alpha)]
        alpha: ExpFilterUpdateBaseAlphaInput,
    ) -> PyResult<()> {
        <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::update_base(
            &mut self.0,
            alpha,
        );
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/filters.rs.
type MovingAverageImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MovingAverage;

fn _extract_moving_average_new_window_size(
    obj: &Bound<'_, PyAny>,
) -> PyResult<MovingAverageNewWindowSizeInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "window_size")
}
type MovingAverageNewWindowSizeInput = usize;

fn _extract_moving_average_add_sample_sample(
    obj: &Bound<'_, PyAny>,
) -> PyResult<MovingAverageAddSampleSampleInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "sample")
}
type MovingAverageAddSampleSampleInput = f64;

#[pyclass(subclass, name = "MovingAverage", module = "livekit.agents.utils.moving_average")]
pub struct MovingAverage(pub(crate) MovingAverageImpl);

#[pymethods]
impl MovingAverage {
    #[new]
    #[pyo3(signature = (window_size))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_moving_average_new_window_size)]
        window_size: MovingAverageNewWindowSizeInput,
    ) -> PyResult<MovingAverage> {
        let out = py.detach(|| {
            <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::new(window_size)
        });
        ::pyo3::PyResult::Ok(MovingAverage(out))
    }

    #[pyo3(signature = (sample))]
    fn add_sample(
        &mut self,
        #[pyo3(from_py_with = _extract_moving_average_add_sample_sample)]
        sample: MovingAverageAddSampleSampleInput,
    ) -> PyResult<()> {
        <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::add_sample(
            &mut self.0,
            sample,
        );
        Ok(())
    }

    fn get_avg(&self) -> PyResult<f64> {
        let out =
            <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::get_avg(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    fn reset(&mut self) -> PyResult<()> {
        <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::reset(&mut self.0);
        Ok(())
    }

    fn size(&self) -> PyResult<usize> {
        let out =
            <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::size(&self.0);
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/resample.rs.
type AudioConverterImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AudioConverter;

fn _extract_audio_converter_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioConverterNewSampleRateInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type AudioConverterNewSampleRateInput = u32;

fn _extract_audio_converter_new_num_channels(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioConverterNewNumChannelsInput> {
    crate::py_values::number_from_py::<u16>(obj.as_any(), "num_channels")
}
type AudioConverterNewNumChannelsInput = u16;

/// `utils.media.AudioConverter`: frames of any rate and channel count, converted to one format
/// (resampled, and averaged to mono or copied across channels). A frame already in that format
/// passes through untouched; a change of input format ends the previous stream first.
#[pyclass(name = "AudioConverter", module = "livekit.agents.utils.media")]
pub struct AudioConverter(pub(crate) AudioConverterImpl);

#[pymethods]
impl AudioConverter {
    #[new]
    #[pyo3(signature = (sample_rate, num_channels))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_audio_converter_new_sample_rate)]
        sample_rate: AudioConverterNewSampleRateInput,
        #[pyo3(from_py_with = _extract_audio_converter_new_num_channels)]
        num_channels: AudioConverterNewNumChannelsInput,
    ) -> PyResult<AudioConverter> {
        let out = py.detach(|| {
            <AudioConverterImpl as ::lk_agents_abi::utils::resample::AudioConverter>::new(
                sample_rate,
                num_channels,
            )
        });
        ::pyo3::PyResult::Ok(AudioConverter(out))
    }

    /// The converted audio `frame` completes, after the tail of a stream it ends.
    #[pyo3(signature = (frame))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        frame: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let frame = crate::generated::records::rtc_frames::audio_frame_from_py(frame.as_any())?;
        let out = <AudioConverterImpl as ::lk_agents_abi::utils::resample::AudioConverter>::push(
            &mut self.0,
            frame,
        )
        .map_err(|e: ::lk_agents_abi::ResampleError| {
            <::lk_agents_abi::ResampleError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::rtc_frames::audio_frame_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// The audio still held by the resampler, at the end of a segment or stream.
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = <AudioConverterImpl as ::lk_agents_abi::utils::resample::AudioConverter>::flush(
            &mut self.0,
        )
        .map_err(|e: ::lk_agents_abi::ResampleError| {
            <::lk_agents_abi::ResampleError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::rtc_frames::audio_frame_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/websocket.rs.
type WebSocketSessionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WebSocketSession;

/// `_core.WebSocketSession`: one provider socket stream. `run()` connects through the protocol,
/// sends its input, hands it each received frame, keeps the connection alive and replaces it;
/// failures retry under `conn_options`.
#[pyclass(frozen, name = "WebSocketSession", module = "livekit.agents._core")]
pub struct WebSocketSession(pub(crate) ::std::sync::Arc<WebSocketSessionImpl>);

#[pymethods]
impl WebSocketSession {
    /// `label` names the provider in retries. With `mux`, the stream is the context
    /// `context_id` on one of its shared sockets. With `pacing`, its frames go no faster than
    /// the pacing allows.
    #[new]
    #[pyo3(signature = (host, conn_options, label, *, mux = None, context_id = None, pacing = None))]
    fn new(
        py: Python<'_>,
        host: Bound<'_, PyAny>,
        conn_options: Bound<'_, PyAny>,
        label: ::std::string::String,
        mux: Option<Bound<'_, PyAny>>,
        context_id: ::std::option::Option<::std::string::String>,
        pacing: Option<Bound<'_, PyAny>>,
    ) -> PyResult<WebSocketSession> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::utils::WebSocketHostProxy::from_host(host.as_any())?;
        let conn_options =
            crate::generated::records::connect::api_connect_options_from_py(conn_options.as_any())?;
        let mux = mux.map(|v| crate::py_objects::native::<dyn ::lk_agents_abi::utils::websocket::SharedWebSockets>(v.as_any()).and_then(|native| native.ok_or_else(|| ::pyo3::exceptions::PyTypeError::new_err(concat!("mux", ": expected a native WebSocketMux"))))).transpose()?;
        let pacing = pacing
            .map(|v| {
                crate::py_objects::native::<
                    dyn ::lk_agents_abi::utils::websocket::SharedWebSocketPacing,
                >(v.as_any())
                .and_then(|native| {
                    native.ok_or_else(|| {
                        ::pyo3::exceptions::PyTypeError::new_err(concat!(
                            "pacing",
                            ": expected a native WebSocketPacing"
                        ))
                    })
                })
            })
            .transpose()?;
        let out = py.detach(|| <WebSocketSessionImpl as ::lk_agents_abi::utils::websocket::WebSocketSession>::new(&env, host, conn_options, label, mux, context_id, pacing)).map_err(|e: ::lk_agents_abi::ApiConnectOptionsError| <::lk_agents_abi::ApiConnectOptionsError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(WebSocketSession(::std::sync::Arc::new(out)))
    }

    /// Replace the connection now, once the item being written went: close it, connect again.
    /// No retry is spent.
    #[pyo3(signature = (reason))]
    fn reconnect(&self, py: Python<'_>, reason: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <WebSocketSessionImpl as ::lk_agents_abi::utils::websocket::WebSocketSession>::reconnect(&self.0, reason));
        Ok(())
    }

    /// The provider's session is finished, as the stream read it: the connection closes before
    /// the next input item, which goes on a new connection.
    fn end_session(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <WebSocketSessionImpl as ::lk_agents_abi::utils::websocket::WebSocketSession>::end_session(&self.0));
        Ok(())
    }

    /// Run until the input ended and the provider finished, or the failure that ended it.
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let cx = crate::py_calls::entry_cx()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <WebSocketSessionImpl as ::lk_agents_abi::utils::websocket::WebSocketSession>::run(&this, cx)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Stop: the socket is closed, bounded, and `run()` returns.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <WebSocketSessionImpl as ::lk_agents_abi::utils::websocket::WebSocketSession>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/websocket.rs.
type WebSocketMuxImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WebSocketMux;

/// `_core.WebSocketMux`: the shared sockets of a multiplexing provider, opened on demand, each
/// carrying up to `max_contexts` streams and replaced after `max_session_duration`.
#[pyclass(frozen, name = "WebSocketMux", module = "livekit.agents._core")]
pub struct WebSocketMux(pub(crate) ::std::sync::Arc<WebSocketMuxImpl>);

#[pymethods]
impl WebSocketMux {
    #[new]
    #[pyo3(signature = (host, options))]
    fn new(
        py: Python<'_>,
        host: Bound<'_, PyAny>,
        options: Bound<'_, PyAny>,
    ) -> PyResult<WebSocketMux> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let host = crate::generated::utils::WebSocketMuxHostProxy::from_host(host.as_any())?;
        let options =
            crate::generated::records::websocket::web_socket_mux_options_from_py(options.as_any())?;
        let out = py.detach(|| {
            <WebSocketMuxImpl as ::lk_agents_abi::utils::websocket::WebSocketMux>::new(
                &env, host, options,
            )
        });
        ::pyo3::PyResult::Ok(WebSocketMux(::std::sync::Arc::new(out)))
    }

    /// Open a shared socket now, unless one has room: the first stream then finds it open.
    fn prewarm(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <WebSocketMuxImpl as ::lk_agents_abi::utils::websocket::WebSocketMux>::prewarm(&self.0)
        });
        Ok(())
    }

    /// Options changed: idle sockets close, busy ones close once their streams end.
    fn invalidate(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <WebSocketMuxImpl as ::lk_agents_abi::utils::websocket::WebSocketMux>::invalidate(
                &self.0,
            )
        });
        Ok(())
    }

    /// Close every socket, bounded.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <WebSocketMuxImpl as ::lk_agents_abi::utils::websocket::WebSocketMux>::aclose(
                &this,
            )
            .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::utils::websocket::SharedWebSockets`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::utils::websocket::SharedWebSockets>(),
        read: |obj| match obj.cast::<WebSocketMux>() {
            Ok(this) => {
                let native: ::std::sync::Arc<dyn ::lk_agents_abi::utils::websocket::SharedWebSockets> = this.get().0.clone();
                Ok(crate::py_objects::Taken::Native(::std::boxed::Box::new(native)))
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/websocket.rs.
type WebSocketPacingImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WebSocketPacing;

fn _extract_web_socket_pacing_new_min_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<WebSocketPacingNewMinIntervalInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "min_interval")).map(Some)
    }
}
type WebSocketPacingNewMinIntervalInput = ::std::option::Option<f64>;

/// `_core.WebSocketPacing`: how fast the streams that share it send. `min_interval`: seconds between
/// two frames. `real_time`: frames go no faster than the duration their stream gives them.
#[pyclass(frozen, name = "WebSocketPacing", module = "livekit.agents._core")]
pub struct WebSocketPacing(pub(crate) ::std::sync::Arc<WebSocketPacingImpl>);

#[pymethods]
impl WebSocketPacing {
    #[new]
    #[pyo3(signature = (*, min_interval = None, real_time = false))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_web_socket_pacing_new_min_interval)]
        min_interval: WebSocketPacingNewMinIntervalInput,
        real_time: bool,
    ) -> PyResult<WebSocketPacing> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py
            .detach(|| {
                <WebSocketPacingImpl as ::lk_agents_abi::utils::websocket::WebSocketPacing>::new(
                    &env,
                    min_interval,
                    real_time,
                )
            })
            .map_err(|e: ::lk_agents_abi::errors::Invalid| {
                <::lk_agents_abi::errors::Invalid as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(WebSocketPacing(::std::sync::Arc::new(out)))
    }
}

// Wherever the host passes a `::lk_agents_abi::utils::websocket::SharedWebSocketPacing`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::utils::websocket::SharedWebSocketPacing>(),
        read: |obj| match obj.cast::<WebSocketPacing>() {
            Ok(this) => {
                let native: ::std::sync::Arc<dyn ::lk_agents_abi::utils::websocket::SharedWebSocketPacing> = this.get().0.clone();
                Ok(crate::py_objects::Taken::Native(::std::boxed::Box::new(native)))
            }
            Err(_) => Ok(crate::py_objects::Taken::Other),
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/video.rs.

/// [`::lk_agents_abi::utils::video::RtcVideoFrame`] over a host `livekit.rtc.VideoFrame`.
#[derive(Clone)]
pub struct RtcVideoFrameProxy {
    host: crate::py_objects::HostObject,
    /// The host object it was taken from, handed back as itself.
    source: crate::py_objects::HostObject,
}

impl RtcVideoFrameProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None, true)?))
    }

    /// A host object a call returned, owned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context), true)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>(obj)?
            .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `RtcVideoFrame` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None, true)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
        owned: bool,
    ) -> PyResult<Self> {
        let source = match owned {
            true => crate::py_objects::HostObject::owned(obj),
            false => crate::py_objects::HostObject::weak(obj)?,
        };
        let host = match owned {
            true => crate::py_objects::HostObject::owned(obj),
            false => crate::py_objects::HostObject::weak(obj)?,
        };
        let _ = context;
        Ok(RtcVideoFrameProxy { host, source })
    }
}

impl RtcVideoFrameProxy {
    /// The host object behind a proxy of `::lk_agents_abi::utils::video::RtcVideoFrame`; a native implementation has none.
    pub fn to_host<'py>(
        py: Python<'py>,
        value: ::std::sync::Arc<dyn ::lk_agents_abi::utils::video::RtcVideoFrame>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let any: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = value;
        match any.downcast::<Self>() {
            Ok(proxy) => proxy.source.bind(py),
            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `RtcVideoFrame` has no host object to hand back",
            )),
        }
    }
}

impl ::lk_agents_abi::utils::video::RtcVideoFrame for RtcVideoFrameProxy {
    fn rgba(
        &self,
    ) -> ::std::result::Result<
        ::lk_agents_abi::utils::video::VideoFrame,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [host.bind(py)?];
                Ok((
                    py.import("livekit.agents._bindings")?.getattr("rgba_frame")?,
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
            let out = (crate::generated::records::rtc_frames::video_frame_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/websocket.rs.

/// [`::lk_agents_abi::utils::websocket::WebSocketHost`] over a host `livekit.agents._websocket.WebSocketHost`.
#[derive(Clone)]
pub struct WebSocketHostProxy {
    host: crate::py_objects::HostObject,
    /// The host's running loop and a copy of its context, captured when it was taken;
    /// none when taken with no running loop: each call then runs in its driving operation's.
    context: ::std::option::Option<::lk_agents_abi::utils::host_ref::HostRef>,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
}

impl WebSocketHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::utils::websocket::WebSocketHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::utils::websocket::WebSocketHost>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None, true)?))
    }

    /// A host object a call returned, owned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::utils::websocket::WebSocketHost>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::utils::websocket::WebSocketHost>(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context), true)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::utils::websocket::WebSocketHost>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::utils::websocket::WebSocketHost>(obj)?
            .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `WebSocketHost` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None, true)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
        owned: bool,
    ) -> PyResult<Self> {
        let host = match owned {
            true => crate::py_objects::HostObject::owned(obj),
            false => crate::py_objects::HostObject::weak(obj)?,
        };
        let context = match context {
            Some(context) => Some(::std::clone::Clone::clone(context)),
            None => crate::py_calls::proxy_context(obj.py()),
        };
        Ok(WebSocketHostProxy { host, context, notifier: ::std::default::Default::default() })
    }
}

impl ::lk_agents_abi::utils::websocket::WebSocketHost for WebSocketHostProxy {
    fn connect(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        conn_options: ::lk_agents_abi::ApiConnectOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::utils::websocket::WebSocketConnection,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let conn_options_arg = conn_options;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_connect",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::connect::api_connect_options_to_py(
                        py,
                        conn_options_arg,
                    )?];
                Ok((
                    host.bind(py)?.getattr("_core_connect")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || crate::generated::records::websocket::web_socket_connection_from_py(o.as_any())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn next_input(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::option::Option<::lk_agents_abi::utils::websocket::WebSocketInput>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_next_input",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_next_input")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || if o.is_none() { Ok(None) } else { (crate::generated::records::websocket::web_socket_input_from_py(o.as_any())).map(Some) }))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn next_control(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::utils::websocket::WsFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_next_control",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_next_control")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::websocket::ws_frame_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn on_message(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        frame: ::lk_agents_abi::utils::websocket::WsFrame,
        message: ::std::option::Option<::lk_agents_abi::HostRef>,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::utils::websocket::WebSocketRead,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let frame_arg = frame;
        let message_arg = message;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_on_message",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [
                    crate::generated::records::websocket::ws_frame_to_py(py, frame_arg)?,
                    match message_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::py_objects::object_ref_to_py(py, x)?,
                    },
                ];
                Ok((
                    host.bind(py)?.getattr("_core_on_message")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || crate::generated::records::websocket::web_socket_read_from_py(o.as_any())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn input_idle_after(
        &self,
    ) -> ::std::result::Result<::std::option::Option<f64>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_input_idle_after")?,
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
            let out = (if o.is_none() {
                Ok(None)
            } else {
                (crate::py_values::number_from_py::<f64>(
                    o.as_any(),
                    "WebSocketHost._core_input_idle_after",
                ))
                .map(Some)
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn input_idle(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::utils::websocket::WsFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_input_idle",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_input_idle")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::websocket::ws_frame_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn response_idle_after(
        &self,
    ) -> ::std::result::Result<::std::option::Option<f64>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_response_idle_after")?,
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
            let out = (if o.is_none() {
                Ok(None)
            } else {
                (crate::py_values::number_from_py::<f64>(
                    o.as_any(),
                    "WebSocketHost._core_response_idle_after",
                ))
                .map(Some)
            })
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn response_idle(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::utils::websocket::WsFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_response_idle",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_response_idle")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::websocket::ws_frame_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn close_context(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::utils::websocket::WebSocketContextsCancel,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketHost._core_close_context",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_close_context")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || crate::generated::records::websocket::web_socket_contexts_cancel_from_py(o.as_any())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn opened(&self, request_id: ::std::option::Option<::std::string::String>) {
        if self.host.gone() {
            return;
        }
        let host = ::std::clone::Clone::clone(&self.host);
        let request_id_arg = request_id;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "WebSocketHost._core_opened",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(request_id_arg, py)?];
                Ok((
                    host.bind(py)?.getattr("_core_opened")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn session_finished(&self) {
        if self.host.gone() {
            return;
        }
        let host = ::std::clone::Clone::clone(&self.host);
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "WebSocketHost._core_session_finished",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_session_finished")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn retrying(&self, error: ::lk_agents_abi::errors::ProviderError) {
        if self.host.gone() {
            return;
        }
        let host = ::std::clone::Clone::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(
            |py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> {
                Ok((crate::py_errors::ExceptionValue::to_exception(error_arg, py)?).unbind())
            },
        ) {
            Ok(value) => value,
            Err(error) => {
                log::warn!(target: ::lk_agents_abi::LOG_TARGET, "WebSocketHost._core_retrying failed: {error}");
                return;
            }
        };
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "WebSocketHost._core_retrying",
            &crate::py_calls::calling_context(&self.context),
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [error_arg.into_bound(py)];
                Ok((
                    host.bind(py)?.getattr("_core_retrying")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
    }

    fn owes_reply(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_owes_reply")?,
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
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn nothing_pending(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_nothing_pending")?,
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
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn expects_close(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_expects_close")?,
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
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }

    fn can_reconnect(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::clone::Clone::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_can_reconnect")?,
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
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/websocket.rs.

/// [`::lk_agents_abi::utils::websocket::WebSocketMuxHost`] over a host `livekit.agents._websocket.WebSocketMultiplexer`.
#[derive(Clone)]
pub struct WebSocketMuxHostProxy {
    host: crate::py_objects::HostObject,
    /// The host's running loop and a copy of its context, captured when it was taken;
    /// none when taken with no running loop: each call then runs in its driving operation's.
    context: ::std::option::Option<::lk_agents_abi::utils::host_ref::HostRef>,
}

impl WebSocketMuxHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::utils::websocket::WebSocketMuxHost>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::utils::websocket::WebSocketMuxHost,
        >(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None, true)?))
    }

    /// A host object a call returned, owned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::utils::websocket::WebSocketMuxHost>> {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::utils::websocket::WebSocketMuxHost,
        >(obj)?
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context), true)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::utils::websocket::WebSocketMuxHost>> {
        if crate::py_objects::native::<dyn ::lk_agents_abi::utils::websocket::WebSocketMuxHost>(
            obj,
        )?
        .is_some()
        {
            return Err(::pyo3::exceptions::PyTypeError::new_err(
                "a native `WebSocketMuxHost` is shared, never handed over to be owned",
            ));
        }
        Ok(::std::boxed::Box::new(Self::take(obj, None, true)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
        owned: bool,
    ) -> PyResult<Self> {
        let host = match owned {
            true => crate::py_objects::HostObject::owned(obj),
            false => crate::py_objects::HostObject::weak(obj)?,
        };
        let context = match context {
            Some(context) => Some(::std::clone::Clone::clone(context)),
            None => crate::py_calls::proxy_context(obj.py()),
        };
        Ok(WebSocketMuxHostProxy { host, context })
    }
}

impl ::lk_agents_abi::utils::websocket::WebSocketMuxHost for WebSocketMuxHostProxy {
    fn connect(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        conn_options: ::lk_agents_abi::ApiConnectOptions,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::utils::websocket::WebSocketConnection,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let conn_options_arg = conn_options;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketMuxHost._core_connect",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::connect::api_connect_options_to_py(
                        py,
                        conn_options_arg,
                    )?];
                Ok((
                    host.bind(py)?.getattr("_core_connect")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || crate::generated::records::websocket::web_socket_connection_from_py(o.as_any())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn route(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        frame: ::lk_agents_abi::utils::websocket::WsFrame,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::lk_agents_abi::utils::websocket::WebSocketMuxRoute,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let frame_arg = frame;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketMuxHost._core_route",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::generated::records::websocket::ws_frame_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py)?.getattr("_core_route")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || crate::generated::records::websocket::web_socket_mux_route_from_py(o.as_any())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn open_context(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
        context_id: ::std::string::String,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::utils::websocket::WsFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let context_id_arg = context_id;
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketMuxHost._core_open_context",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(context_id_arg, py)?];
                Ok((
                    host.bind(py)?.getattr("_core_open_context")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::websocket::ws_frame_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }

    fn goodbye(
        &self,
        cx: &::lk_agents_abi::telemetry::Cx,
    ) -> ::lk_agents_abi::BoxFuture<
        '_,
        ::std::result::Result<
            ::std::vec::Vec<::lk_agents_abi::utils::websocket::WsFrame>,
            ::lk_agents_abi::errors::ProviderError,
        >,
    > {
        let host = ::std::clone::Clone::clone(&self.host);
        let started = ::lk_agents_runtime::host::call(
            &crate::py_calls::Entry,
            "WebSocketMuxHost._core_goodbye",
            &crate::py_calls::calling_context(&self.context),
            &[],
            cx,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((
                    host.bind(py)?.getattr("_core_goodbye")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        let context = ::lk_agents_abi::agent::HostContext::host_context(
            &crate::py_calls::calling_context(&self.context),
        )
        .cloned();
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
                let out = (context.as_ref().map(|context| out.returned_context(py, context)).transpose().and_then(|context| crate::py_calls::returning(context, || o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::websocket::ws_frame_from_py(item.as_any()) }).collect::<PyResult<Vec<_>>>())))).map_err(|e| <::lk_agents_abi::errors::ProviderError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
                Ok(out)
            })
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::utils::dates::DateError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::FilterError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::utils::filters::FilterResetError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyAssertionError::new_err(::std::string::ToString::to_string(self))
    }
}

::pyo3::create_exception!(
    _core,
    StaleHandle,
    ::pyo3::exceptions::PyRuntimeError,
    "A handle whose originating host no longer owns a live entry."
);
impl crate::py_errors::HostException for ::lk_agents_abi::utils::host_ref::StaleHostRef {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        StaleHandle::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::ResampleError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}
