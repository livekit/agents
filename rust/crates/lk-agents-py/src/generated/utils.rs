//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `utils` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

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
pub struct ConnectionPoolState(ConnectionPoolStateImpl);

#[pymethods]
impl ConnectionPoolState {
    #[new]
    #[pyo3(signature = (maximum_age, refresh_on_get))]
    fn new(#[pyo3(from_py_with = _extract_connection_pool_state_new_maximum_age)] maximum_age: ::std::option::Option<f64>, refresh_on_get: bool) -> PyResult<ConnectionPoolState> {
        let out = <ConnectionPoolStateImpl as ::lk_agents_abi::utils::connection_pool::ConnectionPoolState>::new(maximum_age, refresh_on_get);
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

fn _extract_exp_filter_new_max_val(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "max_val"))?) })
}

fn _extract_exp_filter_new_min_val(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "min_val"))?) })
}

fn _extract_exp_filter_new_initial(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "initial"))?) })
}

fn _extract_exp_filter_reset_alpha(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "alpha"))?) })
}

fn _extract_exp_filter_reset_initial(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "initial"))?) })
}

fn _extract_exp_filter_reset_min_val(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "min_val"))?) })
}

fn _extract_exp_filter_reset_max_val(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "max_val"))?) })
}

fn _extract_exp_filter_apply_exp(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "exp")
}

fn _extract_exp_filter_apply_sample(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::Maybe<f64>> {
    Ok(match obj.extract::<::lk_agents_abi::Maybe<Bound<'_, PyAny>>>()? { ::lk_agents_abi::Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven, ::lk_agents_abi::Maybe::Given(value) => ::lk_agents_abi::Maybe::Given((::lk_agents_abi::python::number_from_py::<f64>(&value, "sample"))?) })
}

fn _extract_exp_filter_update_base_alpha(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "alpha")
}

#[pyclass(subclass, name = "ExpFilter", module = "livekit.agents.utils.exp_filter")]
pub struct ExpFilter(ExpFilterImpl);

#[pymethods]
impl ExpFilter {
    #[new]
    #[pyo3(signature = (alpha, max_val = ::lk_agents_abi::Maybe::NotGiven, min_val = ::lk_agents_abi::Maybe::NotGiven, initial = ::lk_agents_abi::Maybe::NotGiven))]
    fn new(#[pyo3(from_py_with = _extract_exp_filter_new_alpha)] alpha: f64, #[pyo3(from_py_with = _extract_exp_filter_new_max_val)] max_val: ::lk_agents_abi::Maybe<f64>, #[pyo3(from_py_with = _extract_exp_filter_new_min_val)] min_val: ::lk_agents_abi::Maybe<f64>, #[pyo3(from_py_with = _extract_exp_filter_new_initial)] initial: ::lk_agents_abi::Maybe<f64>) -> PyResult<ExpFilter> {
        let out = <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::new(alpha, max_val, min_val, initial).map_err(|e: ::lk_agents_abi::FilterError| <::lk_agents_abi::FilterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(ExpFilter(out))
    }

    #[pyo3(signature = (alpha = ::lk_agents_abi::Maybe::NotGiven, initial = ::lk_agents_abi::Maybe::NotGiven, min_val = ::lk_agents_abi::Maybe::NotGiven, max_val = ::lk_agents_abi::Maybe::NotGiven))]
    fn reset(&mut self, #[pyo3(from_py_with = _extract_exp_filter_reset_alpha)] alpha: ::lk_agents_abi::Maybe<f64>, #[pyo3(from_py_with = _extract_exp_filter_reset_initial)] initial: ::lk_agents_abi::Maybe<f64>, #[pyo3(from_py_with = _extract_exp_filter_reset_min_val)] min_val: ::lk_agents_abi::Maybe<f64>, #[pyo3(from_py_with = _extract_exp_filter_reset_max_val)] max_val: ::lk_agents_abi::Maybe<f64>) -> PyResult<()> {
        <ExpFilterImpl as ::lk_agents_abi::utils::filters::ExpFilter>::reset(&mut self.0, alpha, initial, min_val, max_val).map_err(|e: ::lk_agents_abi::utils::filters::FilterResetError| <::lk_agents_abi::utils::filters::FilterResetError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (exp, sample = ::lk_agents_abi::Maybe::NotGiven))]
    fn apply(&mut self, #[pyo3(from_py_with = _extract_exp_filter_apply_exp)] exp: f64, #[pyo3(from_py_with = _extract_exp_filter_apply_sample)] sample: ::lk_agents_abi::Maybe<f64>) -> PyResult<f64> {
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
pub struct MovingAverage(MovingAverageImpl);

#[pymethods]
impl MovingAverage {
    #[new]
    #[pyo3(signature = (window_size))]
    fn new(#[pyo3(from_py_with = _extract_moving_average_new_window_size)] window_size: usize) -> PyResult<MovingAverage> {
        let out = <MovingAverageImpl as ::lk_agents_abi::utils::filters::MovingAverage>::new(window_size);
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

// Declared in rust/crates/lk-agents-abi/src/utils/strings.rs.
type TextFunctionsImpl = <crate::Selected as super::Implementations>::TextFunctions;

#[pyfunction]
#[pyo3(signature = (name))]
pub fn camel_to_snake_case(name: ::std::string::String) -> PyResult<::std::string::String> {
    let out = <TextFunctionsImpl as ::lk_agents_abi::utils::strings::TextFunctions>::camel_to_snake_case(&name);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/utils/strings.rs.
type TransportFunctionsImpl = <crate::Selected as super::Implementations>::TransportFunctions;

#[pyfunction]
#[pyo3(signature = (url))]
pub fn is_cloud(url: ::std::string::String) -> PyResult<bool> {
    let out = <TransportFunctionsImpl as ::lk_agents_abi::utils::strings::TransportFunctions>::is_cloud_url(&url);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (livekit_url, env_override))]
pub fn observability_url(livekit_url: ::std::string::String, env_override: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <TransportFunctionsImpl as ::lk_agents_abi::utils::strings::TransportFunctions>::observability_url(&livekit_url, env_override);
    Ok(out)
}
