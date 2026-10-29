//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `storage` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/storage/mod.rs.
type StorageImpl = <crate::Selected as super::Implementations>::Storage;

fn _extract_storage_new_lease_ttl(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "lease_ttl")).map(Some) }
}

fn _extract_storage_new_renew_interval(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<f64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<f64>(&obj, "renew_interval")).map(Some) }
}

fn _extract_storage_create_thread_idle_ttl_seconds(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "idle_ttl_seconds")
}

fn _extract_storage_list_threads_limit(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&obj, "limit")).map(Some) }
}

fn _extract_storage_update_thread_idle_ttl_seconds(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<i64>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<i64>(&obj, "idle_ttl_seconds")).map(Some) }
}

fn _extract_storage_get_thread_items_limit(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&obj, "limit")).map(Some) }
}

/// Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
/// directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
#[pyclass(frozen, name = "Storage", module = "livekit.agents._core")]
pub struct Storage(::std::sync::Arc<StorageImpl>);

#[pymethods]
impl Storage {
    #[pyo3(signature = (thread_id, lane, *, initial_history = None))]
    fn _core_voice_store<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, lane: ::std::string::String, initial_history: ::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::_core_voice_store(&this, thread_id, lane, initial_history)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::storage::_core_VoiceStore(::std::sync::Arc::new(out)), py))
        })
    }

    /// `directory`: threads in that local directory, resolved to an absolute path once.
    /// Otherwise agentstore through the public API derived from `url` (the project's
    /// `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
    #[new]
    #[pyo3(signature = (*, directory = None, url = None, api_key = None, api_secret = None, lease_ttl = None, renew_interval = None))]
    fn new(directory: ::std::option::Option<::std::string::String>, url: ::std::option::Option<::std::string::String>, api_key: ::std::option::Option<::std::string::String>, api_secret: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_storage_new_lease_ttl)] lease_ttl: ::std::option::Option<f64>, #[pyo3(from_py_with = _extract_storage_new_renew_interval)] renew_interval: ::std::option::Option<f64>) -> PyResult<Storage> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::new(directory, url, api_key, api_secret, lease_ttl, renew_interval).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(Storage(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (config, host))]
    fn _core_a2a_door<'py>(&self, py: Python<'py>, config: Bound<'_, PyAny>, host: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let config = crate::generated::records::a2a_door::door_config_from_py(&config)?;
        let host = crate::generated::a2a::TurnHostProxy::from_host(&host)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::_core_a2a_door(&this, config, host)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::a2a::_core_A2ADoor(::std::sync::Arc::new(out)), py))
        })
    }

    /// The absolute local directory, or the public API agentstore is reached through.
    #[getter]
    fn location(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::location(&self.0);
        Ok(out)
    }

    /// A new `AT_` thread, returned once its metadata and database exist. `attributes` maps
    /// strings to strings.
    #[pyo3(signature = (*, scope = None, attributes = None, idle_ttl_seconds = 0))]
    fn create_thread<'py>(&self, py: Python<'py>, scope: ::std::option::Option<::std::string::String>, attributes: Option<Bound<'_, PyAny>>, #[pyo3(from_py_with = _extract_storage_create_thread_idle_ttl_seconds)] idle_ttl_seconds: i64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let attributes = attributes.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "attributes")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("attributes", ": expected a dict"))) }).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::create_thread(&this, scope, attributes, idle_ttl_seconds)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_class::thread_info_to_py(py, out)?, py))
        })
    }

    /// Unknown, deleted or expired threads raise `ThreadNotFound`.
    #[pyo3(signature = (thread_id))]
    fn get_thread<'py>(&self, py: Python<'py>, thread_id: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread(&this, thread_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_class::thread_info_to_py(py, out)?, py))
        })
    }

    /// The threads of one scope, newest first; `before` is the previous page's cursor.
    #[pyo3(signature = (scope, *, before = None, limit = None))]
    fn list_threads<'py>(&self, py: Python<'py>, scope: ::std::string::String, before: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_storage_list_threads_limit)] limit: ::std::option::Option<u32>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::list_threads(&this, scope, before, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_class::thread_page_to_py(py, out)?, py))
        })
    }

    /// Merges `attributes` (an empty value deletes the key) and sets the idle TTL when given:
    /// the activity boundary that moves expiry.
    #[pyo3(signature = (thread_id, *, attributes = None, idle_ttl_seconds = None))]
    fn update_thread<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, attributes: Option<Bound<'_, PyAny>>, #[pyo3(from_py_with = _extract_storage_update_thread_idle_ttl_seconds)] idle_ttl_seconds: ::std::option::Option<i64>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let attributes = attributes.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "attributes")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("attributes", ": expected a dict"))) }).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::update_thread(&this, thread_id, attributes, idle_ttl_seconds)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_class::thread_info_to_py(py, out)?, py))
        })
    }

    /// Makes the thread unavailable and reclaims its content; repeat-safe.
    #[pyo3(signature = (thread_id))]
    fn delete_thread<'py>(&self, py: Python<'py>, thread_id: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::delete_thread(&this, thread_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// The thread's chat items, oldest first, of one agent's lane when `agent` is given.
    #[pyo3(signature = (thread_id, *, agent = None, after = None, limit = None))]
    fn get_thread_items<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, agent: ::std::option::Option<::std::string::String>, after: ::std::option::Option<::std::string::String>, #[pyo3(from_py_with = _extract_storage_get_thread_items_limit)] limit: ::std::option::Option<u32>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread_items(&this, thread_id, agent, after, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_class::thread_item_page_to_py(py, out)?, py))
        })
    }

    /// The thread's A2A tasks, of one `status` when given.
    #[pyo3(signature = (thread_id, *, status = None))]
    fn get_thread_tasks<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, status: Option<Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let status = status.map(|v| crate::generated::records::storage_class::task_state_from_py(&v)).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread_tasks(&this, thread_id, status)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_class::thread_task_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any(), py))
        })
    }

    /// The existing thread's handle, its writes under `holder`. Takes no lease.
    #[pyo3(signature = (thread_id, holder))]
    fn open<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, holder: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::open(&this, thread_id, holder)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::handles::Thread::from(out), py))
        })
    }

    /// A runtime-supplied field store whose explicit saves commit outside a managed turn.
    #[pyo3(signature = (thread_id, lane))]
    fn _core_context_store<'py>(&self, py: Python<'py>, thread_id: ::std::string::String, lane: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::_core_context_store(&this, thread_id, lane)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(super::storage::_core_ContextActivation(::std::sync::Arc::new(out)), py))
        })
    }

    /// The storage's response streams, shared by its threads.
    fn streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::streams(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::handles::Streams::from(out), py))
        })
    }

    /// Releases backend resources once open handles settled; repeat-safe.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
type _core_ContextActivationImpl = <crate::Selected as super::Implementations>::ContextActivation;

fn _extract_context_activation_save_revision(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "revision")
}

/// Activation state and revision/schema decisions. Application type validation stays in the host.
#[pyclass(frozen, name = "_core_ContextActivation", module = "livekit.agents._core")]
pub struct _core_ContextActivation(::std::sync::Arc<_core_ContextActivationImpl>);

#[pymethods]
impl _core_ContextActivation {
    #[new]
    #[pyo3(signature = (scopes, routing_state, read_only))]
    fn new(scopes: Bound<'_, PyAny>, routing_state: Option<Bound<'_, PyAny>>, read_only: bool) -> PyResult<_core_ContextActivation> {
        ::lk_agents_abi::python::check_pid()?;
        let scopes = scopes.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::storage_context::context_scope_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let routing_state = routing_state.map(|v| ::lk_agents_abi::python::value_from_py(&v, "routing_state")).transpose()?;
        let out = <_core_ContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::new(scopes, routing_state, read_only).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_ContextActivation(::std::sync::Arc::new(out)))
    }

    /// Check manifests before restoring values or accepting defaults for absent fields.
    #[pyo3(signature = (scope_id, identity, fields))]
    fn load<'py>(&self, py: Python<'py>, scope_id: ::std::string::String, identity: ::std::string::String, fields: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let fields = fields.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::storage_context::context_field_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_ContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::load(&this, scope_id, identity, fields)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_context::context_values_to_py(py, out)?, py))
        })
    }

    /// Stage the validated fields; a stale activation cannot overwrite a newer revision.
    #[pyo3(signature = (scope_id, identity, revision, fields))]
    fn save<'py>(&self, py: Python<'py>, scope_id: ::std::string::String, identity: ::std::string::String, #[pyo3(from_py_with = _extract_context_activation_save_revision)] revision: i64, fields: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let fields = fields.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::storage_context::context_field_from_py(&item) }).collect::<PyResult<Vec<_>>>())?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_ContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::save(&this, scope_id, identity, revision, fields)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(crate::generated::records::storage_context::context_values_to_py(py, out)?, py))
        })
    }

    #[getter]
    fn writes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_ContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::writes(&self.0);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::generated::records::storage_context::context_scope_write_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    fn acknowledge(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::acknowledge(&self.0);
        Ok(())
    }

    /// End the activation, refusing subsequent loads or saves.
    fn close(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_ContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::close(&self.0);
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/voice.rs.
type _core_VoiceStoreImpl = <crate::Selected as super::Implementations>::VoiceStore;

/// The stable context store and initial snapshot belonging to one live call's lane.
#[pyclass(frozen, name = "_core_VoiceStore", module = "livekit.agents._core")]
pub struct _core_VoiceStore(::std::sync::Arc<_core_VoiceStoreImpl>);

#[pymethods]
impl _core_VoiceStore {
    /// `None` preserves the application's explicit constructor history.
    #[getter]
    fn initial_history(&self) -> PyResult<::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_VoiceStoreImpl as ::lk_agents_abi::storage::voice::VoiceStore>::initial_history(&self.0);
        Ok(out)
    }

    #[getter]
    fn activation(&self) -> PyResult<super::storage::_core_ContextActivation> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_VoiceStoreImpl as ::lk_agents_abi::storage::voice::VoiceStore>::activation(&self.0);
        Ok(super::storage::_core_ContextActivation(::std::sync::Arc::new(out)))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_VoiceStoreImpl as ::lk_agents_abi::storage::voice::VoiceStore>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::storage::voice::VoiceStorage`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::storage::voice::VoiceStorage>(),
        read: |obj| {
            let this = obj.cast::<_core_VoiceStore>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::storage::voice::VoiceStorage> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// A `::lk_agents_abi::storage::voice::VoiceStorage` this class implements is handed to the host as an instance of it.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNativeClass {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::storage::voice::VoiceStorage>(),
        write: |py, native| {
            let native = native.downcast::<_core_VoiceStoreImpl>().ok()?;
            Some(::pyo3::Py::new(py, _core_VoiceStore(native)).map(::pyo3::Py::into_any))
        },
    }
}
