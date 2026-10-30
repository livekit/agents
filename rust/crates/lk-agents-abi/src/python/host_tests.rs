//! The host bridge and the value crossings, against a live interpreter (moved from
//! `lk-agents-py`'s `runtime.rs` with the code they test).

mod side_tables {
    use crate::python::{capture, capture_value, errors, raise_original, resolve_captured};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    // ------------------------------------------------------------------------ the side tables

    /// A plugin error inside the SDK taxonomy when available, or the bare binding's fallback.
    fn make_exception(py: Python<'_>, code: &str) -> Py<PyAny> {
        let locals = pyo3::types::PyDict::new(py);
        py.run(std::ffi::CString::new(code).unwrap().as_c_str(), None, Some(&locals)).unwrap();
        locals.get_item("exc").unwrap().unwrap().unbind()
    }

    const PLUGIN_ERROR: &str = "\
try:
    from livekit.agents import APIStatusError
except ImportError:
    APIStatusError = Exception
class MyStatusError(APIStatusError):
    def __init__(self):
        super().__init__('rate limited')
        self.retryable = True
        self.status_code = 429
    def __str__(self):
        return 'rate limited'
exc = MyStatusError()
";

    /// #6894's shape: a third-party exception whose `__str__` raises.
    const HOSTILE_ERROR: &str = "\
class Hostile(Exception):
    def __str__(self):
        raise ValueError('nope')
exc = Hostile()
";

    /// The table drained and counted; the drain is what decrefs.
    fn outstanding() -> usize {
        errors().drain();
        errors().outstanding()
    }

    // Catches: reading `retryable`/`status_code` off the object at decision time instead of at
    // capture. Decided and dropped outside `attach`: the no-GIL release must still work.
    #[test]
    fn retryable_and_status_code_are_readable_without_the_gil() {
        Python::initialize();
        let before = Python::attach(|_py| outstanding());
        let err = Python::attach(|py| {
            let exc = make_exception(py, PLUGIN_ERROR);
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
    #[test]
    fn a_captured_error_that_is_dropped_leaves_nothing_outstanding() {
        Python::initialize();
        Python::attach(|py| {
            let before = outstanding();
            let err = capture_value(make_exception(py, PLUGIN_ERROR).bind(py));
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
    #[test]
    fn re_raising_works_once_across_every_clone_and_the_drops_after_it_release_nothing() {
        Python::initialize();
        Python::attach(|py| {
            let before = outstanding();
            let exc = make_exception(py, PLUGIN_ERROR);
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
    #[test]
    fn a_pyerr_is_captured_as_its_own_value_and_is_not_retryable() {
        Python::initialize();
        Python::attach(|py| {
            let err = py.eval(c"1 / 0", None, None).unwrap_err();
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
            let exc = make_exception(py, HOSTILE_ERROR);
            let err = capture_value(exc.bind(py));
            assert_eq!(err.message(), "");
            assert!(!err.retryable());
            assert!(raise_original(py, err.host_ref().unwrap()).value(py).is(&exc));
        });
    }
}

mod session_bridge {
    //! The leaves and lists as their own impls cross them, and the call schemas the RTC seam sends.
    use crate::python::value_to_py;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    /// `program`'s `out`, evaluated.
    fn out(py: Python<'_>, program: &str) -> Py<PyAny> {
        let globals = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!("{program}\n")).unwrap().as_c_str(),
            Some(&globals),
            None,
        )
        .unwrap();
        globals.get_item("out").unwrap().unwrap().unbind()
    }

    /// The keys of a built kwargs dict, sorted and compared whole, so an extra key fails too.
    fn keys(d: &Bound<'_, PyDict>) -> Vec<String> {
        let mut out: Vec<String> =
            d.keys().iter().map(|k| k.extract::<String>().expect("str keys")).collect();
        out.sort();
        out
    }

    /// Just enough of `livekit.rtc` for an `rtc.AudioFrame`.
    const RTC_STUBS: &str = r#"
import sys, types as _t
def _mod(name):
    m = _t.ModuleType(name); m.__path__ = []; sys.modules[name] = m; return m
lk = sys.modules.get('livekit') or _mod('livekit')
rtc = sys.modules.get('livekit.rtc') or _mod('livekit.rtc'); lk.rtc = rtc
# Always the stub, even when the real SDK is importable: its frame keeps what it was given.
class AudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self.data = data; self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel
        self.userdata = {}
rtc.AudioFrame = AudioFrame
"#;

    fn attach_rtc<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        attach(|py| {
            py.run(std::ffi::CString::new(RTC_STUBS).unwrap().as_c_str(), None, None).unwrap();
            f(py)
        })
    }

    // Catches: a provider payload's tagged bytes reaching the host as a dict, or a user dict with
    // more than the one key mistaken for bytes.
    #[test]
    fn a_tagged_bytes_leaf_reads_back_as_bytes_and_nothing_else_does() {
        attach(|py| {
            let tag = crate::llm::provider_format::BYTES_TAG;
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
