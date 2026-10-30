#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! Boundary regressions against a real interpreter, including the reverse conversion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use super::*;

fn attach(f: impl FnOnce(Python<'_>)) {
    Python::initialize();
    Python::attach(|py| {
        // Match host_tests: isolate ABI unit tests from the public package and SDK installation.
        // Pytest conformance covers actual SDK classes and generated public entrypoints.
        module(
            py,
            r#"
import sys, types
for name in ('livekit', 'livekit.agents', 'livekit.agents.types', 'livekit.rtc'):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
class NotGiven:
    def __bool__(self): return False
sys.modules['livekit.agents.types'].NotGiven = NotGiven
class AudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self._data = bytearray(data)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel
    @property
    def data(self): return memoryview(self._data).cast('h')
sys.modules['livekit.rtc'].AudioFrame = AudioFrame
"#,
        );
        f(py);
    });
}

fn module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = std::ffi::CString::new(code).unwrap();
    PyModule::from_code(py, &code, c"boundary.py", c"boundary").unwrap()
}

#[test]
fn omission_is_only_not_given_and_none_is_a_given_value() {
    attach(|py| {
        let read = |code: &str| {
            let value = module(py, &format!("from livekit.agents.types import NotGiven\nvalue = {code}\n"))
                .getattr("value")
                .unwrap();
            value.extract::<MaybeNull<f64>>().map(|v| v.into_core())
        };
        assert!(matches!(read("NotGiven()").unwrap(), crate::Maybe::NotGiven));
        assert!(matches!(read("None").unwrap(), crate::Maybe::Given(None)));
        assert!(matches!(read("0.0").unwrap(), crate::Maybe::Given(Some(0.0))));
        assert!(read("'x'").is_err());
    });
}

#[test]
fn defaults_preserve_lookup_exception_identity_and_cause() {
    attach(|py| {
        let m = module(
            py,
            r#"
cause = OSError('disk')
failure = RuntimeError('lookup failed')
failure.__cause__ = cause
class Broken:
    @property
    def field(self):
        raise failure
    def __getitem__(self, key):
        raise failure
broken = Broken()
key_failure = KeyError('backend lookup failed')
key_failure.__cause__ = cause
class BrokenDict(dict):
    def __getitem__(self, key):
        raise key_failure
broken_dict = BrokenDict(field=1, value='')
"#,
        );
        let obj = m.getattr("broken").unwrap();
        let failure = m.getattr("failure").unwrap();
        for err in [
            attr_or(&obj, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap_err(),
            item_or(&obj, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap_err(),
        ] {
            assert!(err.value(py).is(&failure));
            assert!(err.cause(py).unwrap().value(py).is(&m.getattr("cause").unwrap()));
        }
        let err = item_or(
            &m.getattr("broken_dict").unwrap(),
            "Record",
            "field",
            |v| v.extract::<u32>(),
            || 7,
        )
        .unwrap_err();
        assert!(err.value(py).is(&m.getattr("key_failure").unwrap()));
        assert!(err.cause(py).unwrap().value(py).is(&m.getattr("cause").unwrap()));
        let generated = m.getattr("broken_dict").unwrap().extract::<NullableUpdate>().unwrap_err();
        assert!(generated.value(py).is(&m.getattr("key_failure").unwrap()));
        let missing = PyDict::new(py);
        assert_eq!(item_or(&missing, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap(), 7);
        let obj = module(py, "class Empty: pass\nempty = Empty()\n").getattr("empty").unwrap();
        assert_eq!(attr_or(&obj, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap(), 7);
    });
}

#[test]
fn duration_refuses_invalid_seconds_before_constructing_duration() {
    attach(|py| {
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let obj = value.into_pyobject(py).unwrap().into_any();
            let err = secs_from_py(&obj).expect_err("must reject, never clamp or panic");
            assert!(err.is_instance_of::<PyValueError>(py));
        }
        let obj = 0.125_f64.into_pyobject(py).unwrap().into_any();
        assert_eq!(secs_from_py(&obj).unwrap(), std::time::Duration::from_millis(125));
    });
}

#[test]
fn large_video_snapshots_remain_isolated_from_mutable_host_aliases() {
    attach(|py| {
        let m = module(
            py,
            "owner = bytearray(b'x' * (1920 * 1080 * 4))\nalias = memoryview(owner).toreadonly()\n",
        );
        let alias = m.getattr("alias").unwrap();
        let snapshot = retained_bytes(&alias, "VideoFrame.data", None).unwrap();
        m.getattr("owner").unwrap().set_item(0, 0).unwrap();
        assert_eq!(snapshot[0], b'x');
        assert_eq!(snapshot.len(), 1920 * 1080 * 4);
        assert_eq!(snapshot[snapshot.len() - 1], b'x');
    });
}

#[test]
fn large_buffer_allocation_lets_other_python_threads_progress_before_the_snapshot() {
    attach(|py| {
        let m = module(
            py,
            r#"
import threading, time
owner = bytearray(b'x' * (32 * 1024 * 1024))
alias = memoryview(owner).toreadonly()
started = threading.Event()
armed = progressed = False
def run():
    global progressed
    started.set()
    while not armed:
        time.sleep(0.001)
    progressed = True
thread = threading.Thread(target=run)
thread.start()
started.wait()
"#,
        );
        let alias = m.getattr("alias").unwrap();
        m.setattr("armed", true).unwrap();
        for _ in 0..16 {
            let snapshot = retained_bytes(&alias, "VideoFrame.data", None).unwrap();
            assert_eq!(snapshot.len(), 32 * 1024 * 1024);
            assert_eq!(snapshot[0], b'x');
            assert_eq!(snapshot[snapshot.len() - 1], b'x');
        }
        let progressed = m.getattr("progressed").unwrap().extract::<bool>().unwrap();
        m.getattr("thread").unwrap().call_method0("join").unwrap();
        assert!(progressed, "large native allocation must release the GIL before copying");
    });
}

#[test]
fn bounded_buffers_refuse_oversized_host_buffers_before_copying() {
    attach(|py| {
        let m = module(
            py,
            r#"
immutable = b'x' * (1920 * 1080 * 4)
mutable = memoryview(bytearray(immutable)).toreadonly()
"#,
        );
        let clock = py.import("time").unwrap().getattr("thread_time").unwrap();
        for field in ["immutable", "mutable"] {
            let started = clock.call0().unwrap().extract::<f64>().unwrap();
            let result = retained_bytes(&m.getattr(field).unwrap(), "VideoFrame.data", Some(4));
            let elapsed = clock.call0().unwrap().extract::<f64>().unwrap() - started;
            let Err(error) = result else {
                panic!("oversized host buffer was accepted");
            };
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("limit is 4"));
            assert!(elapsed <= 0.002, "byte admission used {elapsed:.6}s CPU");
        }
    });
}

#[test]
fn retained_buffers_keep_immutable_owners_and_snapshot_writable_aliases() {
    attach(|py| {
        let m = module(
            py,
            r#"
from array import array
from ctypes import c_short
immutable = b'1234'
owner = bytearray(b'1234')
alias = memoryview(owner).toreadonly()
samples = memoryview(array('h', [1, -2]))
samples_le = memoryview((c_short.__ctype_le__ * 2)(1, -2))
wrong_format = memoryview(array('f', [1.0]))
wrong_layout = memoryview(owner)[::2]
"#,
        );
        let immutable = m.getattr("immutable").unwrap();
        let retained = retained_bytes(&immutable, "frame.data", None).unwrap();
        assert_eq!(
            retained.as_ptr(),
            immutable.cast::<pyo3::types::PyBytes>().unwrap().as_bytes().as_ptr()
        );
        let snapshot = retained_bytes(&m.getattr("alias").unwrap(), "frame.data", None).unwrap();
        m.getattr("owner").unwrap().set_item(0, 9).unwrap();
        m.delattr("immutable").unwrap();
        drop(immutable);
        m.delattr("alias").unwrap();
        m.delattr("owner").unwrap();
        let pcm = snapshot_buffer(
            &m.getattr("samples").unwrap(),
            BufferFormat::I16Le,
            "frame.data",
            None,
        )
        .unwrap();
        assert_eq!(&*pcm, &[1, 0, 254, 255]);
        let explicit_le = snapshot_buffer(
            &m.getattr("samples_le").unwrap(),
            BufferFormat::I16Le,
            "frame.data",
            None,
        )
        .unwrap();
        assert_eq!(&*explicit_le, &[1, 0, 254, 255]);
        for field in ["wrong_format", "wrong_layout"] {
            assert!(
                snapshot_buffer(
                    &m.getattr(field).unwrap(),
                    BufferFormat::I16Le,
                    "frame.data",
                    None
                )
                .is_err()
            );
        }
        m.delattr("wrong_layout").unwrap();
        assert_eq!(&retained[..], b"1234");
        assert_eq!(&snapshot[..], b"1234");
    });
}

#[test]
fn a_failing_property_is_not_an_absent_attribute() {
    attach(|py| {
        let obj = module(
            py,
            r#"
failure = AttributeError('descriptor failed')
class Broken:
    @property
    def field(self):
        raise failure
broken = Broken()
"#,
        );
        let err = attr_or(
            &obj.getattr("broken").unwrap(),
            "Record",
            "field",
            |v| v.extract::<u32>(),
            || 7,
        )
        .unwrap_err();
        assert!(err.value(py).is(&obj.getattr("failure").unwrap()));
    });
}

#[derive(Debug, lk_agents_host_derive::Host)]
#[binding(extract, emit, from_items)]
struct ScalarRecord {
    count: u32,
    values: std::collections::HashMap<u32, Vec<f64>>,
    #[binding(default)]
    labels: Vec<String>,
}

#[test]
fn generated_collections_validate_typed_keys_and_nested_values_in_both_directions() {
    attach(|py| {
        let m = module(py, "source = {'count': 7, 'values': {9: [0.125]}}\n");
        let source = m.getattr("source").unwrap();
        let mut record = source.extract::<ScalarRecord>().unwrap();
        let sibling = source.extract::<ScalarRecord>().unwrap();
        record.labels.push("first only".into());
        assert!(sibling.labels.is_empty(), "mutable defaults belong to each instance");
        source
            .get_item("values")
            .unwrap()
            .get_item(9)
            .unwrap()
            .call_method1("append", (2.0,))
            .unwrap();
        let output = (&record).into_pyobject(py).unwrap();
        assert_eq!(
            output
                .get_item("values")
                .unwrap()
                .unwrap()
                .get_item(9)
                .unwrap()
                .extract::<Vec<f64>>()
                .unwrap(),
            vec![0.125]
        );
        source.set_item("count", true).unwrap();
        assert!(source.extract::<ScalarRecord>().is_err(), "bool is not an integer field");
        source.set_item("count", 7).unwrap();
        source.get_item("values").unwrap().set_item("wrong", vec![0.5]).unwrap();
        assert!(source.extract::<ScalarRecord>().is_err(), "map keys must validate as u32");
    });
}

#[derive(Debug, Default, lk_agents_host_derive::Host)]
#[binding(extract, emit, from_items)]
struct NullableUpdate {
    #[binding(default, skip_if_none)]
    value: Option<String>,
}

#[test]
fn unicode_float_and_newtype_boundaries_keep_their_declared_semantics() {
    attach(|py| {
        let text = "e\u{301} / é / 日本語 / 🦀 / \0";
        let obj = text.into_pyobject(py).unwrap().into_any();
        let json = value_from_py(&obj, "payload.text").unwrap();
        assert_eq!(value_to_py(py, &json).unwrap().extract::<String>().unwrap(), text);
        for value in [f64::MIN_POSITIVE, f64::MAX, 0.12345678901234566] {
            let obj = value.into_pyobject(py).unwrap().into_any();
            assert_eq!(number_from_py::<f64>(&obj, "weight").unwrap().to_bits(), value.to_bits());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let obj = value.into_pyobject(py).unwrap().into_any();
            let roundtrip = number_from_py::<f64>(&obj, "weight").unwrap();
            assert!(roundtrip.is_nan() || roundtrip == value);
            assert!(
                value_from_py(&obj, "payload.weight").is_err(),
                "JSON rejects non-finite floats"
            );
        }
        let too_wide = f64::MAX.into_pyobject(py).unwrap().into_any();
        assert!(f32_from_py(&too_wide, "weight").is_err());
        let lone_surrogate = module(py, "value = '\\ud800'\n").getattr("value").unwrap();
        assert!(
            lone_surrogate.extract::<String>().is_err(),
            "Rust Unicode strings reject lone surrogates"
        );
    });
}

#[derive(Default)]
struct GuardedValue(u32);

// What `lk-agents-py` links in the extension: the fixture's entries pass the fork guard first.
inventory::submit! {
    super::bridge::HostRuntime {
        check_pid: || Ok(()),
        future_into_py: |_, _| unreachable!("no fixture awaits"),
    }
}

#[crate::host(object(cell = "mutex", rust_name = GuardedFixture))]
impl GuardedValue {
    #[host(method)]
    pub fn fail(&mut self) {
        self.0 = 99;
        panic!("interrupted state update");
    }

    #[host(getter)]
    pub fn value(&self) -> u32 {
        self.0
    }
}

#[test]
fn generated_mutex_object_does_not_report_success_after_a_panic() {
    attach(|py| {
        let object = Bound::new(py, GuardedFixture { inner: Default::default() }).unwrap();
        let trigger = module(
            py,
            "def trigger(obj):\n    try:\n        obj.fail()\n    except BaseException as failure:\n        return failure\n",
        );
        let failure = trigger.getattr("trigger").unwrap().call1((&object,)).unwrap();
        assert!(failure.is_instance_of::<pyo3::exceptions::PyBaseException>());
        let error = object.getattr("value").expect_err("poisoned state must never be reused");
        assert!(error.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
    });
}
