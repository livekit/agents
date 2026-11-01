//! Captured exception lifetimes in an interpreter isolated from callback tests.

use _core::py_calls;

#[path = "py_test_support.rs"]
pub mod py_test_support;

mod side_tables {
    // The assertions count a process-wide exception table, so exercise each lifecycle in order.
    #[test]
    fn captured_exception_lifetimes() {
        crate::py_test_support::install_package(_core::register);
        retryable_and_status_code_are_readable_without_the_gil();
        a_captured_error_that_is_dropped_leaves_nothing_outstanding();
        re_raising_works_once_across_every_clone_and_the_drops_after_it_release_nothing();
        a_pyerr_is_captured_as_its_own_value_and_is_not_retryable();
        generated_wrappers_preserve_the_original_exception();
        foreign_exception_keys_cannot_peek_or_take_a_local_exception();
    }

    use _core::py_errors::{capture, capture_value, raise_original, resolve_captured};
    use _core::py_objects::exception_table;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    fn generated_wrappers_preserve_the_original_exception() {
        use _core::py_errors::{FromHostErr, HostException};
        Python::attach(|py| {
            let original = crate::py_test_support::call(py, "divide_by_zero", ()).unwrap_err();
            let rtc = lk_agents_abi::voice::rtc::RtcError::from_host(py, &original);
            assert!(rtc.to_py_err().value(py).is(original.value(py)));
            let io = lk_agents_abi::voice::io::IoError::from_host(py, &original);
            assert!(io.to_py_err().value(py).is(original.value(py)));
        });
    }

    fn foreign_exception_keys_cannot_peek_or_take_a_local_exception() {
        use _core::py_errors::release;
        use _core::py_objects::Table;
        use lk_agents_abi::errors::CapturedError;

        Python::attach(|py| {
            let before = outstanding();
            let value = crate::py_test_support::fixture(py, "plugin_error");
            let captured = capture_value(&value);
            let local = captured.host_ref().unwrap();
            let foreign_table = Table::new();
            let foreign = CapturedError::new(local.id(), foreign_table.registry());
            assert!(resolve_captured(py, foreign.clone()).is_none());
            assert!(release(foreign).is_none());
            assert!(resolve_captured(py, local.clone()).unwrap().bind(py).is(&value));
            assert!(release(local).unwrap().bind(py).is(&value));
            drop(captured);
            assert_eq!(outstanding(), before);
        });
    }

    // ------------------------------------------------------------------------ the side tables

    /// The table drained and counted; the drain is what decrefs.
    fn outstanding() -> usize {
        exception_table().drain();
        exception_table().outstanding()
    }

    // Catches: reading `retryable`/`status_code` off the object at decision time instead of at
    // capture. Decided and dropped outside `attach`: the no-GIL release must still work.
    fn retryable_and_status_code_are_readable_without_the_gil() {
        Python::initialize();
        let before = Python::attach(|_py| outstanding());
        let err = Python::attach(|py| {
            let exc = crate::py_test_support::fixture(py, "plugin_error").unbind();
            capture_value(exc.bind(py))
        });
        assert!(err.retryable());
        assert_eq!(err.status_code(), Some(429));
        assert_eq!(err.message(), "rate limited");
        drop(err);
        assert_eq!(Python::attach(|_py| outstanding()), before, "a no-GIL drop must still release");
    }

    // Catches: a dropped `ProviderError::Host` leaking its entry, or releasing per clone rather than
    // on the last drop.
    fn a_captured_error_that_is_dropped_leaves_nothing_outstanding() {
        Python::initialize();
        Python::attach(|py| {
            let before = outstanding();
            let err = capture_value(
                crate::py_test_support::fixture(py, "plugin_error").unbind().bind(py),
            );
            assert_eq!(outstanding(), before + 1);

            let key = err.host_ref().expect("a host error");
            let twin = err.clone();
            let mut slot = Some(err);
            assert!(slot.is_some());
            slot = None; // the chain's overwrite, in miniature
            assert_eq!(outstanding(), before + 1, "released while clones were still live");
            drop(twin);
            assert_eq!(outstanding(), before + 1, "released while the key clone was still live");
            drop(key);
            assert_eq!(outstanding(), before, "the last drop is the release");
            drop(slot);
        });
    }

    // Catches: `raise_original` handing the same object out twice through two clones.
    fn re_raising_works_once_across_every_clone_and_the_drops_after_it_release_nothing() {
        Python::initialize();
        Python::attach(|py| {
            let before = outstanding();
            let exc = crate::py_test_support::fixture(py, "plugin_error").unbind();
            let err = capture_value(exc.bind(py));
            let a = err.host_ref().unwrap();
            let b = a.clone();

            let raised = raise_original(py, a);
            assert!(raised.value(py).is(&exc), "the first re-raise is the original object");
            assert_eq!(outstanding(), before, "re-raising takes the entry out");

            let again = raise_original(py, b);
            assert!(again.is_instance_of::<PyRuntimeError>(py), "a second re-raise finds nothing");
            drop(err);
            assert_eq!(outstanding(), before);
        });
    }

    // Catches: `capture(PyErr)` capturing a different object, an exception outside the taxonomy
    // treated as retryable (`llm/llm.py:287` only retries `APIError`), or a raising `__str__`
    // taking the capture down.
    fn a_pyerr_is_captured_as_its_own_value_and_is_not_retryable() {
        Python::initialize();
        Python::attach(|py| {
            let err = crate::py_test_support::call(py, "divide_by_zero", ()).unwrap_err();
            let value = err.value(py).clone().into_any().unbind();

            let captured = capture(py, &err);
            assert_eq!(captured.api_kind(), None);
            assert!(!captured.retryable());
            let host_ref = captured.host_ref().unwrap();

            // Peeks, so the later re-raise still finds it.
            let peeked =
                resolve_captured(py, host_ref.clone()).expect("resolve must find the entry");
            assert!(peeked.bind(py).is(value.bind(py)));
            assert!(
                resolve_captured(py, host_ref.clone()).is_some(),
                "resolve must not consume the entry"
            );

            let raised = raise_original(py, host_ref);
            assert!(raised.value(py).is(value.bind(py)));

            // #6894: an exception whose `__str__` raises is still captured.
            let exc = crate::py_test_support::fixture(py, "hostile_error").unbind();
            let err = capture_value(exc.bind(py));
            assert_eq!(err.message(), "");
            assert!(!err.retryable());
            assert!(raise_original(py, err.host_ref().unwrap()).value(py).is(&exc));
        });
    }
}

mod session_bridge {
    //! The leaves and lists as their own impls cross them, and the call schemas the RTC seam sends.
    use _core::py_values::value_to_py;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    // Catches: a provider payload's tagged bytes reaching the host as a dict, or a user dict with
    // more than the one key mistaken for bytes.
    #[test]
    fn a_tagged_bytes_leaf_reads_back_as_bytes_and_nothing_else_does() {
        attach(|py| {
            let tag = lk_agents_abi::llm::provider_format::BYTES_TAG;
            let v = value_to_py(
                py,
                &serde_json::json!({"img": {tag: "AQID"}, "other": {tag: "AQID", "x": 1}}),
            )
            .unwrap();
            let img = v.get_item("img").unwrap();
            assert_eq!(img.cast::<pyo3::types::PyBytes>().unwrap().as_bytes(), &[1, 2, 3]);
            assert!(v.get_item("other").unwrap().is_instance_of::<PyDict>());
        });
    }
}
