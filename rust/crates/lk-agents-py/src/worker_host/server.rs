//! Generated server classes over the bootstrap loop's Python adapters.

use std::any::Any;
use std::sync::Arc;

use lk_agents_abi::HostRef;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::server_host::{ServerConfig, ServerConfiguration, ServerRuntime};
use lk_agents_abi::worker::worker::ServerOptions;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::{PyServer, PyServerConfig};

fn with_python<T>(call: impl for<'py> FnOnce(Python<'py>) -> PyResult<T>) -> Result<T, HostError> {
    Python::attach(|py| call(py).map_err(|error| lk_agents_abi::python::capture_host(py, &error)))
}

impl ServerConfiguration for PyServerConfig {}

impl ServerConfig for PyServerConfig {
    fn new(options: ServerOptions) -> Result<Self, HostError> {
        with_python(|_| PyServerConfig::new(options))
    }

    fn update(&self, options: ServerOptions) -> Result<(), HostError> {
        with_python(|_| PyServerConfig::update(self, options))
    }

    fn ws_url(&self) -> String {
        PyServerConfig::ws_url(self)
    }
    fn api_key(&self) -> String {
        PyServerConfig::api_key(self)
    }
    fn api_secret(&self) -> String {
        PyServerConfig::api_secret(self)
    }
    fn agent_name(&self) -> String {
        PyServerConfig::agent_name(self)
    }
    fn deployment(&self) -> String {
        PyServerConfig::deployment(self)
    }
    fn cloud(&self) -> bool {
        PyServerConfig::cloud(self)
    }
    fn drain_timeout(&self) -> f64 {
        PyServerConfig::drain_timeout(self)
    }
    fn simulation(&self) -> bool {
        PyServerConfig::simulation(self)
    }
    fn set_simulation(&self, value: bool) {
        PyServerConfig::set_simulation(self, value);
    }
}

impl ServerRuntime for PyServer {
    fn new(config: HostRef, host: HostRef) -> Result<Self, HostError> {
        with_python(|py| {
            let config = lk_agents_abi::python::resolve_key(py, config.id())?;
            let config = lk_agents_abi::python::native::<dyn ServerConfiguration>(config.bind(py))
                .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected a ServerConfig"))?;
            let config: Arc<dyn Any + Send + Sync> = config;
            let config = config.downcast::<PyServerConfig>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("expected a native ServerConfig")
            })?;
            let host = lk_agents_abi::python::resolve_key(py, host.id())?;
            PyServer::new(py, &config, host)
        })
    }

    fn worker_id(&self) -> String {
        PyServer::worker_id(self)
    }
    fn draining(&self) -> bool {
        PyServer::draining(self)
    }
    fn http_port(&self) -> u16 {
        PyServer::http_port(self)
    }

    fn run(
        &self,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        publisher: bool,
        routes: HostRef,
        has_load_fnc: bool,
        has_on_request: bool,
    ) -> Result<HostRef, HostError> {
        with_python(|py| {
            let routes = lk_agents_abi::python::resolve_key(py, routes.id())?;
            let routes = routes.bind(py).cast::<PyList>()?;
            let awaitable = PyServer::run(
                self,
                py,
                devmode,
                unregistered,
                has_rtc,
                has_text_doors,
                publisher,
                routes,
                has_load_fnc,
                has_on_request,
            )?;
            Ok(lk_agents_abi::python::intern_host(awaitable.unbind()))
        })
    }

    fn drain(&self, timeout: Option<f64>) -> Result<HostRef, HostError> {
        with_python(|py| {
            Ok(lk_agents_abi::python::intern_host(PyServer::drain(self, py, timeout)?.unbind()))
        })
    }

    fn aclose(&self) -> Result<HostRef, HostError> {
        with_python(|py| {
            Ok(lk_agents_abi::python::intern_host(PyServer::aclose(self, py)?.unbind()))
        })
    }

    fn job_status(&self, job_id: String, status: String) {
        PyServer::job_status(self, job_id, &status);
    }

    fn simulate_job(
        &self,
        room: String,
        fake_job: bool,
        agent_identity: Option<String>,
        room_info: Option<Vec<u8>>,
        token: Option<String>,
    ) -> Result<HostRef, HostError> {
        with_python(|py| {
            let awaitable =
                PyServer::simulate_job(self, py, room, fake_job, agent_identity, room_info, token)?;
            Ok(lk_agents_abi::python::intern_host(awaitable.unbind()))
        })
    }
}
