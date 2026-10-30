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
type AudioByteStreamImpl = <crate::Selected as super::Implementations>::AudioByteStream;

fn _extract_audio_byte_stream_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

fn _extract_audio_byte_stream_new_num_channels(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    ::lk_agents_abi::python::number_from_py::<u16>(&obj, "num_channels")
}

fn _extract_audio_byte_stream_new_samples_per_channel(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&obj, "samples_per_channel")).map(Some) }
}

fn _extract_audio_byte_stream_push_data(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    ::lk_agents_abi::python::pcm_bytes(obj, "data")
}

fn _extract_audio_byte_stream_write_data(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    ::lk_agents_abi::python::pcm_bytes(obj, "data")
}

/// Buffers bytes and emits whole frames: fixed-size, or progressive (20 ms first, doubling to
/// the target) to cut TTS time-to-first-audio.
#[pyclass(name = "AudioByteStream", module = "livekit.agents.utils.audio")]
pub struct AudioByteStream(pub(crate) AudioByteStreamImpl);

#[pymethods]
impl AudioByteStream {
    /// ``samples_per_channel=None`` means 100 ms, as in the reference.
    #[new]
    #[pyo3(signature = (sample_rate, num_channels, samples_per_channel = None, progressive = false))]
    fn new(py: Python<'_>, #[pyo3(from_py_with = _extract_audio_byte_stream_new_sample_rate)] sample_rate: u32, #[pyo3(from_py_with = _extract_audio_byte_stream_new_num_channels)] num_channels: u16, #[pyo3(from_py_with = _extract_audio_byte_stream_new_samples_per_channel)] samples_per_channel: ::std::option::Option<u32>, progressive: bool) -> PyResult<AudioByteStream> {
        let out = py.detach(|| <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::new(sample_rate, num_channels, samples_per_channel, progressive));
        Ok(AudioByteStream(out))
    }

    /// Buffer ``data`` and emit every whole frame it completes.
    #[pyo3(signature = (data))]
    fn push<'py>(&mut self, py: Python<'py>, #[pyo3(from_py_with = _extract_audio_byte_stream_push_data)] data: ::std::vec::Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let out = <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::push(&mut self.0, data);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::conversions::audio_frame_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    /// Public alias of ``push`` in the reference.
    #[pyo3(signature = (data))]
    fn write<'py>(&mut self, py: Python<'py>, #[pyo3(from_py_with = _extract_audio_byte_stream_write_data)] data: ::std::vec::Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let out = <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::write(&mut self.0, data);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::conversions::audio_frame_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    /// Emit the buffered whole samples, short of a frame or not; a trailing partial sample stays
    /// for the next push.
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::flush(&mut self.0);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::conversions::audio_frame_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    /// Seconds of audio waiting for the next frame.
    #[getter]
    fn buffered_duration(&self) -> PyResult<f64> {
        let out = <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::buffered_duration(&self.0);
        Ok(out)
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
        <AudioByteStreamImpl as ::lk_agents_abi::utils::byte_stream::AudioByteStream>::clear(&mut self.0);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/connection_pool.rs.
type ConnectionPoolStateImpl = <crate::Selected as super::Implementations>::ConnectionPoolState;

fn _extract_connection_pool_state_new_maximum_age(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "maximum_age")).map(Some) }
}

fn _extract_connection_pool_state_connected_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "id")
}

fn _extract_connection_pool_state_connected_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_connection_pool_state_connected_generation(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "generation")
}

fn _extract_connection_pool_state_connected_acquire_time(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "acquire_time")).map(Some) }
}

fn _extract_connection_pool_state_get_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_connection_pool_state_put_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "id")
}

fn _extract_connection_pool_state_remove_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "id")
}

fn _extract_connection_pool_state_closed_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "id")
}

#[pyclass(name = "ConnectionPoolState", module = "livekit.agents._core")]
pub struct ConnectionPoolState(pub(crate) ConnectionPoolStateImpl);

#[pymethods]
impl ConnectionPoolState {
    #[new]
    #[pyo3(signature = (maximum_age, refresh_on_get))]
    fn new(py: Python<'_>, #[pyo3(from_py_with = _extract_connection_pool_state_new_maximum_age)] maximum_age: ::std::option::Option<f64>, refresh_on_get: bool) -> PyResult<ConnectionPoolState> {
        let out = py.detach(|| <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::new(maximum_age, refresh_on_get));
        Ok(ConnectionPoolState(out))
    }

    #[getter]
    fn generation(&self) -> PyResult<u64> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::generation(&self.0);
        Ok(out)
    }

    #[getter]
    fn last_acquire_time(&self) -> PyResult<f64> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::last_acquire_time(&self.0);
        Ok(out)
    }

    #[getter]
    fn last_connection_reused(&self) -> PyResult<bool> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::last_connection_reused(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (id, now, generation, acquire_time))]
    fn connected(&mut self, #[pyo3(from_py_with = _extract_connection_pool_state_connected_id)] id: u64, #[pyo3(from_py_with = _extract_connection_pool_state_connected_now)] now: f64, #[pyo3(from_py_with = _extract_connection_pool_state_connected_generation)] generation: u64, #[pyo3(from_py_with = _extract_connection_pool_state_connected_acquire_time)] acquire_time: ::std::option::Option<f64>) -> PyResult<bool> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::connected(&mut self.0, id, now, generation, acquire_time);
        Ok(out)
    }

    fn empty(&self) -> PyResult<bool> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::empty(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (now))]
    fn get(&mut self, #[pyo3(from_py_with = _extract_connection_pool_state_get_now)] now: f64) -> PyResult<::std::option::Option<u64>> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::get(&mut self.0, now);
        Ok(out)
    }

    #[pyo3(signature = (id))]
    fn put(&mut self, #[pyo3(from_py_with = _extract_connection_pool_state_put_id)] id: u64) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::put(&mut self.0, id);
        Ok(())
    }

    #[pyo3(signature = (id))]
    fn remove(&mut self, #[pyo3(from_py_with = _extract_connection_pool_state_remove_id)] id: u64) -> PyResult<()> {
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

    fn next_closing(&self) -> PyResult<::std::option::Option<u64>> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::next_closing(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (id))]
    fn closed(&mut self, #[pyo3(from_py_with = _extract_connection_pool_state_closed_id)] id: u64) -> PyResult<()> {
        <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::closed(&mut self.0, id);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/filters.rs.
type ExpFilterImpl = <crate::Selected as super::Implementations>::ExpFilter;

fn _extract_exp_filter_new_alpha(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "alpha")
}

fn _extract_exp_filter_apply_exp(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "exp")
}

fn _extract_exp_filter_update_base_alpha(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "alpha")
}

#[pyclass(subclass, name = "ExpFilter", module = "livekit.agents.utils.exp_filter")]
pub struct ExpFilter(pub(crate) ExpFilterImpl);

#[pymethods]
impl ExpFilter {
    #[new]
    #[pyo3(signature = (alpha, max_val = ::lk_agents_abi::python::Omittable::NOT_GIVEN, min_val = ::lk_agents_abi::python::Omittable::NOT_GIVEN, initial = ::lk_agents_abi::python::Omittable::NOT_GIVEN))]
    fn new(py: Python<'_>, #[pyo3(from_py_with = _extract_exp_filter_new_alpha)] alpha: f64, max_val: ::lk_agents_abi::python::Omittable<'_>, min_val: ::lk_agents_abi::python::Omittable<'_>, initial: ::lk_agents_abi::python::Omittable<'_>) -> PyResult<ExpFilter> {
        let max_val = match max_val.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "max_val"))?) };
        let min_val = match min_val.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "min_val"))?) };
        let initial = match initial.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "initial"))?) };
        let out = py.detach(|| <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::new(alpha, max_val, min_val, initial)).map_err(|e: ::lk_agents_abi::FilterError| <::lk_agents_abi::FilterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ExpFilter(out))
    }

    #[pyo3(signature = (alpha = ::lk_agents_abi::python::Omittable::NOT_GIVEN, initial = ::lk_agents_abi::python::Omittable::NOT_GIVEN, min_val = ::lk_agents_abi::python::Omittable::NOT_GIVEN, max_val = ::lk_agents_abi::python::Omittable::NOT_GIVEN))]
    fn reset(&mut self, alpha: ::lk_agents_abi::python::Omittable<'_>, initial: ::lk_agents_abi::python::Omittable<'_>, min_val: ::lk_agents_abi::python::Omittable<'_>, max_val: ::lk_agents_abi::python::Omittable<'_>) -> PyResult<()> {
        let alpha = match alpha.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "alpha"))?) };
        let initial = match initial.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "initial"))?) };
        let min_val = match min_val.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "min_val"))?) };
        let max_val = match max_val.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "max_val"))?) };
        <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::reset(&mut self.0, alpha, initial, min_val, max_val).map_err(|e: ::lk_agents_abi::utils::filters::FilterResetError| <::lk_agents_abi::utils::filters::FilterResetError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (exp, sample = ::lk_agents_abi::python::Omittable::NOT_GIVEN))]
    fn apply(&mut self, #[pyo3(from_py_with = _extract_exp_filter_apply_exp)] exp: f64, sample: ::lk_agents_abi::python::Omittable<'_>) -> PyResult<f64> {
        let sample = match sample.0 { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "sample"))?) };
        let out = <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::apply(&mut self.0, exp, sample).map_err(|e: ::lk_agents_abi::FilterError| <::lk_agents_abi::FilterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[getter]
    fn value(&self) -> PyResult<::std::option::Option<f64>> {
        let out = <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::value(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (alpha))]
    fn update_base(&mut self, #[pyo3(from_py_with = _extract_exp_filter_update_base_alpha)] alpha: f64) -> PyResult<()> {
        <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::update_base(&mut self.0, alpha);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/utils/filters.rs.
type MovingAverageImpl = <crate::Selected as super::Implementations>::MovingAverage;

fn _extract_moving_average_new_window_size(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "window_size")
}

fn _extract_moving_average_add_sample_sample(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "sample")
}

#[pyclass(subclass, name = "MovingAverage", module = "livekit.agents.utils.moving_average")]
pub struct MovingAverage(pub(crate) MovingAverageImpl);

#[pymethods]
impl MovingAverage {
    #[new]
    #[pyo3(signature = (window_size))]
    fn new(py: Python<'_>, #[pyo3(from_py_with = _extract_moving_average_new_window_size)] window_size: usize) -> PyResult<MovingAverage> {
        let out = py.detach(|| <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::new(window_size));
        Ok(MovingAverage(out))
    }

    #[pyo3(signature = (sample))]
    fn add_sample(&mut self, #[pyo3(from_py_with = _extract_moving_average_add_sample_sample)] sample: f64) -> PyResult<()> {
        <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::add_sample(&mut self.0, sample);
        Ok(())
    }

    fn get_avg(&self) -> PyResult<f64> {
        let out = <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::get_avg(&self.0);
        Ok(out)
    }

    fn reset(&mut self) -> PyResult<()> {
        <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::reset(&mut self.0);
        Ok(())
    }

    fn size(&self) -> PyResult<usize> {
        let out = <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::size(&self.0);
        Ok(out)
    }
}
