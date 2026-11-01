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
type StorageImpl = <crate::generated::implementations::Selected as super::Implementations>::Storage;

fn _extract_storage_new_lease_ttl(obj: &Bound<'_, PyAny>) -> PyResult<StorageNewLeaseTtlInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "lease_ttl")).map(Some)
    }
}
type StorageNewLeaseTtlInput = ::std::option::Option<f64>;

fn _extract_storage_new_renew_interval(
    obj: &Bound<'_, PyAny>,
) -> PyResult<StorageNewRenewIntervalInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<f64>(obj.as_any(), "renew_interval")).map(Some)
    }
}
type StorageNewRenewIntervalInput = ::std::option::Option<f64>;

fn _extract_storage_create_thread_idle_ttl_seconds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<StorageCreateThreadIdleTtlSecondsInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "idle_ttl_seconds")
}
type StorageCreateThreadIdleTtlSecondsInput = i64;

fn _extract_storage_list_threads_limit(
    obj: &Bound<'_, PyAny>,
) -> PyResult<StorageListThreadsLimitInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "limit")).map(Some)
    }
}
type StorageListThreadsLimitInput = ::std::option::Option<u32>;

fn _extract_storage_update_thread_idle_ttl_seconds(
    obj: &Bound<'_, PyAny>,
) -> PyResult<StorageUpdateThreadIdleTtlSecondsInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<i64>(obj.as_any(), "idle_ttl_seconds")).map(Some)
    }
}
type StorageUpdateThreadIdleTtlSecondsInput = ::std::option::Option<i64>;

fn _extract_storage_get_thread_items_limit(
    obj: &Bound<'_, PyAny>,
) -> PyResult<StorageGetThreadItemsLimitInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "limit")).map(Some)
    }
}
type StorageGetThreadItemsLimitInput = ::std::option::Option<u32>;

/// Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
/// directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
#[pyclass(frozen, name = "Storage", module = "livekit.agents._core")]
pub struct Storage(pub(crate) ::std::sync::Arc<StorageImpl>);

#[pymethods]
impl Storage {
    #[pyo3(signature = (thread_id, lane, *, initial_history = None))]
    fn _core_voice_store<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        lane: ::std::string::String,
        initial_history: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
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
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::_core_voice_store(&this, thread_id, lane, initial_history)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::storage::CoreVoiceStore(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    /// `directory`: threads in that local directory, resolved to an absolute path once.
    /// Otherwise agentstore through the public API derived from `url` (the project's
    /// `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
    #[new]
    #[pyo3(signature = (*, directory = None, url = None, api_key = None, api_secret = None, lease_ttl = None, renew_interval = None))]
    fn new(
        py: Python<'_>,
        directory: ::std::option::Option<::std::string::String>,
        url: ::std::option::Option<::std::string::String>,
        api_key: ::std::option::Option<::std::string::String>,
        api_secret: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_storage_new_lease_ttl)] lease_ttl: StorageNewLeaseTtlInput,
        #[pyo3(from_py_with = _extract_storage_new_renew_interval)]
        renew_interval: StorageNewRenewIntervalInput,
    ) -> PyResult<Storage> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <StorageImpl as ::lk_agents_abi::storage::Storage>::new(&env, directory, url, api_key, api_secret, lease_ttl, renew_interval)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(Storage(::std::sync::Arc::new(out)))
    }

    /// The absolute local directory, or the public API agentstore is reached through.
    #[getter]
    fn location(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <StorageImpl as ::lk_agents_abi::storage::Storage>::location(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// A new `AT_` thread, returned once its metadata and database exist.
    #[pyo3(signature = (*, scope = None, attributes = None, idle_ttl_seconds = 0))]
    fn create_thread<'py>(
        &self,
        py: Python<'py>,
        scope: ::std::option::Option<::std::string::String>,
        attributes: ::std::option::Option<
            ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
        >,
        #[pyo3(from_py_with = _extract_storage_create_thread_idle_ttl_seconds)]
        idle_ttl_seconds: StorageCreateThreadIdleTtlSecondsInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::create_thread(&this, scope, attributes, idle_ttl_seconds)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::thread_info_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// Unknown, deleted or expired threads raise `ThreadNotFound`.
    #[pyo3(signature = (thread_id))]
    fn get_thread<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread(&this, thread_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::thread_info_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// The threads of one scope, newest first; `before` is the previous page's cursor.
    #[pyo3(signature = (scope, *, before = None, limit = None))]
    fn list_threads<'py>(
        &self,
        py: Python<'py>,
        scope: ::std::string::String,
        before: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_storage_list_threads_limit)]
        limit: StorageListThreadsLimitInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::list_threads(&this, scope, before, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::thread_page_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// Merges `attributes` (an empty value deletes the key) and sets the idle TTL when given:
    /// the activity boundary that moves expiry.
    #[pyo3(signature = (thread_id, *, attributes = None, idle_ttl_seconds = None))]
    fn update_thread<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        attributes: ::std::option::Option<
            ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
        >,
        #[pyo3(from_py_with = _extract_storage_update_thread_idle_ttl_seconds)]
        idle_ttl_seconds: StorageUpdateThreadIdleTtlSecondsInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::update_thread(&this, thread_id, attributes, idle_ttl_seconds)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::thread_info_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// Makes the thread unavailable and reclaims its content; repeat-safe.
    #[pyo3(signature = (thread_id))]
    fn delete_thread<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <StorageImpl as ::lk_agents_abi::storage::Storage>::delete_thread(&this, thread_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// The thread's chat items, oldest first, of one agent's lane when `agent` is given.
    #[pyo3(signature = (thread_id, *, agent = None, after = None, limit = None))]
    fn get_thread_items<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        agent: ::std::option::Option<::std::string::String>,
        after: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_storage_get_thread_items_limit)]
        limit: StorageGetThreadItemsLimitInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread_items(&this, thread_id, agent, after, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::thread_item_page_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// The thread's A2A tasks, of one `status` when given.
    #[pyo3(signature = (thread_id, *, status = None))]
    fn get_thread_tasks<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        status: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let status = status
            .map(|v| crate::generated::records::storage_class::task_state_from_py(v.as_any()))
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::get_thread_tasks(&this, thread_id, status)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                crate::generated::records::storage_class::thread_task_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    /// The existing thread's handle, its writes under `holder`. Takes no lease.
    #[pyo3(signature = (thread_id, holder))]
    fn open<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        holder: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::open(&this, thread_id, holder)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::storage::Thread(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    /// A runtime-supplied field store whose explicit saves commit outside a managed turn.
    #[pyo3(signature = (thread_id, lane))]
    fn _core_context_store<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        lane: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::_core_context_store(&this, thread_id, lane)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::storage::CoreContextActivation(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    /// A managed turn's field store over the scopes and routing state its run loaded: saves
    /// and `transaction()` writes stage for the turn's commit.
    #[pyo3(signature = (thread_id, lane, scopes, *, routing_state = None, read_only = false))]
    fn _core_turn_store<'py>(
        &self,
        py: Python<'py>,
        thread_id: ::std::string::String,
        lane: ::std::string::String,
        scopes: Bound<'_, PyAny>,
        routing_state: Option<Bound<'_, PyAny>>,
        read_only: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let scopes = scopes.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::storage_context::context_scope_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let routing_state = routing_state
            .map(|v| crate::py_values::value_from_py(v.as_any(), "routing_state"))
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <StorageImpl as ::lk_agents_abi::storage::Storage>::_core_turn_store(&this, thread_id, lane, scopes, routing_state, read_only)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::storage::CoreContextActivation(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    /// Releases backend resources once open handles settled; repeat-safe.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <StorageImpl as ::lk_agents_abi::storage::Storage>::aclose(&this).await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::storage::StorageNamespace`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::storage::StorageNamespace>(),
        read: |obj| {
            let this = obj.cast::<Storage>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::storage::StorageNamespace> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
type CoreContextActivationImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ContextActivation;

fn _extract_context_activation_save_revision(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ContextActivationSaveRevisionInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "revision")
}
type ContextActivationSaveRevisionInput = i64;

/// Activation state and revision/schema decisions. Application type validation stays in the host.
#[pyclass(frozen, name = "_core_ContextActivation", module = "livekit.agents._core")]
pub struct CoreContextActivation(pub(crate) ::std::sync::Arc<CoreContextActivationImpl>);

#[pymethods]
impl CoreContextActivation {
    #[new]
    #[pyo3(signature = (scopes, routing_state, read_only))]
    fn new(
        py: Python<'_>,
        scopes: Bound<'_, PyAny>,
        routing_state: Option<Bound<'_, PyAny>>,
        read_only: bool,
    ) -> PyResult<CoreContextActivation> {
        crate::py_calls::check_pid()?;
        let scopes = scopes.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::storage_context::context_scope_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let routing_state = routing_state
            .map(|v| crate::py_values::value_from_py(v.as_any(), "routing_state"))
            .transpose()?;
        let out = py.detach(|| <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::new(scopes, routing_state, read_only)).map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(CoreContextActivation(::std::sync::Arc::new(out)))
    }

    /// Check manifests before restoring values or accepting defaults for absent fields.
    #[pyo3(signature = (scope_id, identity, fields))]
    fn load<'py>(
        &self,
        py: Python<'py>,
        scope_id: ::std::string::String,
        identity: ::std::string::String,
        fields: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let fields = fields.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::storage_context::context_field_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::load(&this, scope_id, identity, fields)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_context::context_values_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    /// Stage the validated fields; a stale activation cannot overwrite a newer revision.
    #[pyo3(signature = (scope_id, identity, revision, fields))]
    fn save<'py>(
        &self,
        py: Python<'py>,
        scope_id: ::std::string::String,
        identity: ::std::string::String,
        #[pyo3(from_py_with = _extract_context_activation_save_revision)]
        revision: ContextActivationSaveRevisionInput,
        fields: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let fields = fields.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::storage_context::context_field_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::save(&this, scope_id, identity, revision, fields)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_context::context_values_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[getter]
    fn writes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::writes(&self.0));
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| {
                        crate::generated::records::storage_context::context_scope_write_to_py(py, x)
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// The application statements managed transactions staged for the turn's commit.
    #[getter]
    fn statements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::statements(&self.0));
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::storage_context::app_statement_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// Application SQL (docs/context.md#application-records-and-transactions): `independent`
    /// commits when the transaction ends; otherwise it joins the managed turn's commit, or
    /// commits like an independent one outside a managed turn.
    #[pyo3(signature = (independent))]
    fn transaction<'py>(&self, py: Python<'py>, independent: bool) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::transaction(&this, independent)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::storage::CoreAppTransaction(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    fn acknowledge(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::acknowledge(&self.0));
        Ok(())
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::closed(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    /// End the activation, refusing subsequent loads or saves.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <CoreContextActivationImpl as ::lk_agents_abi::storage::context::ContextActivation>::close(&self.0));
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/context.rs.
type CoreAppTransactionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AppTransaction;

/// One application transaction. Reads observe committed records; writes are staged and
/// applied atomically when the transaction ends, so a read after a write in the same
/// transaction is refused.
#[pyclass(frozen, name = "_core_AppTransaction", module = "livekit.agents._core")]
pub struct CoreAppTransaction(pub(crate) ::std::sync::Arc<CoreAppTransactionImpl>);

#[pymethods]
impl CoreAppTransaction {
    /// Stages one statement; `params` are JSON scalars.
    #[pyo3(signature = (sql, params))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        sql: ::std::string::String,
        params: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let params = params.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::value_from_py(item.as_any(), "params")
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreAppTransactionImpl as ::lk_agents_abi::storage::context::AppTransaction>::execute(&this, sql, params)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// One read's rows, each a list of column values in select order.
    #[pyo3(signature = (sql, params))]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        sql: ::std::string::String,
        params: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let params = params.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::value_from_py(item.as_any(), "params")
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <CoreAppTransactionImpl as ::lk_agents_abi::storage::context::AppTransaction>::fetch(&this, sql, params)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                ::pyo3::PyResult::Ok(
                                    ::pyo3::types::PyList::new(
                                        py,
                                        x.into_iter()
                                            .map(|x| crate::py_values::value_to_py(py, &x))
                                            .collect::<PyResult<Vec<_>>>()?,
                                    )?
                                    .into_any(),
                                )
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    /// Ends the transaction: its writes commit (independent) or join the turn's commit.
    fn commit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreAppTransactionImpl as ::lk_agents_abi::storage::context::AppTransaction>::commit(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Ends the transaction, discarding its writes.
    fn rollback(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <CoreAppTransactionImpl as ::lk_agents_abi::storage::context::AppTransaction>::rollback(
                &self.0,
            )
        });
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/handles.rs.
type ThreadImpl = <crate::generated::implementations::Selected as super::Implementations>::Thread;

fn _extract_thread_journal_after_id(obj: &Bound<'_, PyAny>) -> PyResult<ThreadJournalAfterIdInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "after_id")
}
type ThreadJournalAfterIdInput = i64;

fn _extract_thread_journal_limit(obj: &Bound<'_, PyAny>) -> PyResult<ThreadJournalLimitInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "limit")
}
type ThreadJournalLimitInput = i64;

fn _extract_thread_tasks_limit(obj: &Bound<'_, PyAny>) -> PyResult<ThreadTasksLimitInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "limit")
}
type ThreadTasksLimitInput = i64;

fn _extract_thread_list_tasks_page_size(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ThreadListTasksPageSizeInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "page_size")
}
type ThreadListTasksPageSizeInput = i64;

fn _extract_thread_list_items_limit(obj: &Bound<'_, PyAny>) -> PyResult<ThreadListItemsLimitInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "limit")
}
type ThreadListItemsLimitInput = i64;

#[pyclass(frozen, name = "Thread", module = "livekit.agents._core")]
pub struct Thread(pub(crate) ::std::sync::Arc<ThreadImpl>);

#[pymethods]
impl Thread {
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::id(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn holder(&self, py: Python<'_>) -> PyResult<::std::string::String> {
        crate::py_calls::check_pid()?;
        let out = py
            .detach(|| <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::holder(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::lanes(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                crate::generated::records::storage_class::lane_row_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (after_id = 0, limit = 100))]
    fn journal<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_thread_journal_after_id)]
        after_id: ThreadJournalAfterIdInput,
        #[pyo3(from_py_with = _extract_thread_journal_limit)] limit: ThreadJournalLimitInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::journal(&this, after_id, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                crate::generated::records::storage_class::event_row_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane))]
    fn history<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::history(&this, lane)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                crate::generated::records::storage_class::item_row_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane, limit = 50))]
    fn tasks<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::string::String,
        #[pyo3(from_py_with = _extract_thread_tasks_limit)] limit: ThreadTasksLimitInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::tasks(&this, lane, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                crate::generated::records::storage_class::task_record_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (task_id))]
    fn task<'py>(
        &self,
        py: Python<'py>,
        task_id: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::task(&this, task_id)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::storage_class::task_record_to_py(py, value)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane = None, status = None, order = "updated".to_owned(), page_size = 50, page_token = None))]
    fn list_tasks<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::option::Option<::std::string::String>,
        status: Option<Bound<'_, PyAny>>,
        order: ::std::string::String,
        #[pyo3(from_py_with = _extract_thread_list_tasks_page_size)]
        page_size: ThreadListTasksPageSizeInput,
        page_token: ::std::option::Option<::std::string::String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let status = status
            .map(|v| crate::generated::records::storage_class::task_state_from_py(v.as_any()))
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::list_tasks(&this, lane, status, order, page_size, page_token)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::task_page_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane = None, after = None, limit = 100))]
    fn list_items<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::option::Option<::std::string::String>,
        after: ::std::option::Option<::std::string::String>,
        #[pyo3(from_py_with = _extract_thread_list_items_limit)] limit: ThreadListItemsLimitInput,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::list_items(&this, lane, after, limit)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    crate::generated::records::storage_class::thread_item_page_to_py(py, out)?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane))]
    fn lane_info<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::lane_info(&this, lane)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::storage_class::lane_row_to_py(py, value)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane, pending = false))]
    fn inbox<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::string::String,
        pending: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::inbox(&this, lane, pending)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    ::pyo3::types::PyList::new(
                        py,
                        out.into_iter()
                            .map(|x| {
                                crate::generated::records::storage_class::inbox_row_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane, task_id, metadata))]
    fn cancel_task<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::string::String,
        task_id: ::std::string::String,
        metadata: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let metadata = match crate::py_values::value_from_py(metadata.as_any(), "metadata")? {
            ::serde_json::Value::Object(map) => Ok(map),
            _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!(
                "metadata",
                ": expected a dict"
            ))),
        }?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::cancel_task(&this, lane, task_id, metadata)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::storage_class::task_record_to_py(py, value)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    #[pyo3(signature = (lane, run_id, *, channel = None, wait_timeout = None))]
    fn open_run<'py>(
        &self,
        py: Python<'py>,
        lane: ::std::string::String,
        run_id: ::std::string::String,
        channel: ::std::option::Option<::std::string::String>,
        wait_timeout: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let wait_timeout = wait_timeout
            .map(|v| {
                crate::py_values::number_from_py::<f64>(v.as_any(), "wait_timeout").and_then(|s| {
                    ::std::time::Duration::try_from_secs_f64(s).map_err(|e| {
                        ::pyo3::exceptions::PyValueError::new_err(format!(
                            "{}: {e}",
                            "wait_timeout"
                        ))
                    })
                })
            })
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::open_run(&this, lane, run_id, channel, wait_timeout)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    super::storage::Run(::std::sync::Arc::new(out)),
                    py,
                )
            })
        })
    }

    /// Schedule connection release without blocking the caller.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        py.detach(|| {
            <ThreadImpl as ::lk_agents_abi::storage::handles::Thread>::close(&self.0, &env)
        });
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/handles.rs.
type RunImpl = <crate::generated::implementations::Selected as super::Implementations>::Run;

#[pyclass(frozen, name = "Run", module = "livekit.agents._core")]
pub struct Run(pub(crate) ::std::sync::Arc<RunImpl>);

#[pymethods]
impl Run {
    #[getter]
    fn version(&self, py: Python<'_>) -> PyResult<i64> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <RunImpl as ::lk_agents_abi::storage::handles::Run>::version(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <RunImpl as ::lk_agents_abi::storage::handles::Run>::closed(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out =
            py.detach(|| <RunImpl as ::lk_agents_abi::storage::handles::Run>::history(&self.0));
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::storage_class::item_row_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| <RunImpl as ::lk_agents_abi::storage::handles::Run>::state(&self.0));
        out.map(|value| crate::py_values::value_to_py(py, &value)).transpose()
    }

    #[getter]
    fn context_scopes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <RunImpl as ::lk_agents_abi::storage::handles::Run>::context_scopes(&self.0)
        });
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::storage_context::context_scope_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    #[pyo3(signature = (history, state = None, *, scope_writes = None))]
    fn commit<'py>(
        &self,
        py: Python<'py>,
        history: Bound<'_, PyAny>,
        state: Option<Bound<'_, PyAny>>,
        scope_writes: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let history = history.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::generated::records::storage_class::item_row_from_py(item.as_any())
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let state =
            state.map(|v| crate::py_values::value_from_py(v.as_any(), "state")).transpose()?;
        let scope_writes = scope_writes
            .map(|v| {
                v.try_iter().and_then(|items| {
                    items
                        .map(|item| {
                            let item = item?;
                            crate::generated::records::storage_context::context_scope_write_from_py(
                                item.as_any(),
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()
                })
            })
            .transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <RunImpl as ::lk_agents_abi::storage::handles::Run>::commit(&this, history, state, scope_writes)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (outcome, reason))]
    fn abandon<'py>(
        &self,
        py: Python<'py>,
        outcome: ::std::string::String,
        reason: ::std::string::String,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <RunImpl as ::lk_agents_abi::storage::handles::Run>::abandon(&this, outcome, reason)
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/storage/voice.rs.
type CoreVoiceStoreImpl =
    <crate::generated::implementations::Selected as super::Implementations>::VoiceStore;

/// The stable context store and initial snapshot belonging to one live call's lane.
#[pyclass(frozen, name = "_core_VoiceStore", module = "livekit.agents._core")]
pub struct CoreVoiceStore(pub(crate) ::std::sync::Arc<CoreVoiceStoreImpl>);

#[pymethods]
impl CoreVoiceStore {
    /// `None` preserves the application's explicit constructor history.
    #[getter]
    fn initial_history<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreVoiceStoreImpl as ::lk_agents_abi::storage::voice::VoiceStore>::initial_history(
                &self.0,
            )
        });
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

    #[getter]
    fn activation(&self, py: Python<'_>) -> PyResult<super::storage::CoreContextActivation> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <CoreVoiceStoreImpl as ::lk_agents_abi::storage::voice::VoiceStore>::activation(&self.0)
        });
        ::pyo3::PyResult::Ok(super::storage::CoreContextActivation(::std::sync::Arc::new(out)))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <CoreVoiceStoreImpl as ::lk_agents_abi::storage::voice::VoiceStore>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::storage::StorageError| <::lk_agents_abi::storage::StorageError as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::storage::voice::VoiceStorage`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::storage::voice::VoiceStorage>(),
        read: |obj| {
            let this = obj.cast::<CoreVoiceStore>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::storage::voice::VoiceStorage> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::storage::StorageError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::Rejected { running_task_id } => crate::py_errors::host_exception_with(
                "livekit.agents.storage.errors",
                "RunRejected",
                ::std::string::ToString::to_string(self),
                |kwargs| {
                    let py = kwargs.py();
                    ::pyo3::types::PyDictMethods::set_item(
                        kwargs,
                        "running_task_id",
                        match ::std::clone::Clone::clone(running_task_id) {
                            None => py.None().into_bound(py),
                            Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?,
                        },
                    )?;
                    ::std::result::Result::Ok(())
                },
            ),
            Self::LeaseLost(..) => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "LeaseLost",
                ::std::string::ToString::to_string(self),
            ),
            Self::Unavailable(..) => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "StorageUnavailable",
                ::std::string::ToString::to_string(self),
            ),
            Self::NotFound(..) => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "ThreadNotFound",
                ::std::string::ToString::to_string(self),
            ),
            Self::VersionConflict { .. } => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "VersionConflict",
                ::std::string::ToString::to_string(self),
            ),
            Self::NotSerializable(..) => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "NotSerializable",
                ::std::string::ToString::to_string(self),
            ),
            Self::TooLarge { .. } => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "TooLarge",
                ::std::string::ToString::to_string(self),
            ),
            Self::Cancelled(..) => crate::py_errors::host_exception(
                "asyncio",
                "CancelledError",
                ::std::string::ToString::to_string(self),
            ),
            Self::Invalid(..) => {
                ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
            }
            _ => crate::py_errors::host_exception(
                "livekit.agents.storage.errors",
                "StorageError",
                ::std::string::ToString::to_string(self),
            ),
        }
    }
}
