//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `utils` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

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
type ConnectionPoolStateNextClosingOutput = ::std::option::Option<u64>;

fn _extract_connection_pool_state_closed_id(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ConnectionPoolStateClosedIdInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "id")
}
type ConnectionPoolStateClosedIdInput = u64;

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

    fn next_closing(&self) -> PyResult<ConnectionPoolStateNextClosingOutput> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::next_closing(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (id))]
    fn closed(
        &mut self,
        #[pyo3(from_py_with = _extract_connection_pool_state_closed_id)]
        id: ConnectionPoolStateClosedIdInput,
    ) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::closed(&mut self.0, id);
        Ok(())
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
