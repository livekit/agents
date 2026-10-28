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
fn presence_rejects_invalid_null_and_preserves_nullable_values() {
    attach(|py| {
        let none = py.None().into_bound(py);
        assert!(none.extract::<crate::Maybe<f64>>().is_err(), "null is not omission");
        assert!(matches!(
            none.extract::<crate::Maybe<Option<f64>>>().unwrap(),
            crate::Maybe::Given(None)
        ));
        let zero = 0.0_f64.into_pyobject(py).unwrap().into_any();
        assert!(matches!(zero.extract::<crate::Maybe<f64>>().unwrap(), crate::Maybe::Given(0.0)));
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
fn stale_host_reference_is_an_error_instead_of_null() {
    attach(|py| {
        let registry = media().registry();
        let stale = crate::HostRef::new(u64::MAX, registry);
        let error = (&stale).into_pyobject(py).unwrap_err();
        assert_eq!(error.get_type(py).name().unwrap().to_str().unwrap(), "StaleHandle");
    });
}

#[test]
fn foreign_host_reference_cannot_resolve_an_existing_local_id() {
    attach(|py| {
        let original = PyDict::new(py);
        let local = intern_host(original.clone().into_any().unbind());
        assert!((&local).into_pyobject(py).unwrap().is(&original));
        let foreign_table = Table::new();
        let foreign_id = foreign_table.insert(PyDict::new(py).into_any().unbind());
        assert_eq!(foreign_id, local.id());
        let foreign = crate::HostRef::new(foreign_id, foreign_table.registry());
        let error = (&foreign).into_pyobject(py).unwrap_err();
        assert_eq!(error.get_type(py).name().unwrap().to_str().unwrap(), "StaleHandle");
        assert_ne!(foreign, local);
    });
}

#[test]
fn media_collection_failure_keeps_the_invalid_element_path() {
    attach(|py| {
        let source = module(py, "frames = [object()]\n").getattr("frames").unwrap();
        let reference = intern_media(source.unbind());
        let error = provider_error_to_py(
            reference.audio_frames().expect_err("invalid frames must not become an empty list"),
        );
        let notes = error.value(py).getattr("__notes__").unwrap().extract::<Vec<String>>().unwrap();
        assert!(notes.iter().any(|note| note.contains("frames[0]")));
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
fn bounded_video_frames_refuse_oversized_host_buffers_before_copying() {
    attach(|py| {
        let m = module(
            py,
            r#"
class Frame:
    width = height = 1
    type = 0
    def __init__(self, data): self.data = data
immutable = b'x' * (1920 * 1080 * 4)
mutable = memoryview(bytearray(immutable)).toreadonly()
"#,
        );
        let clock = py.import("time").unwrap().getattr("thread_time").unwrap();
        for field in ["immutable", "mutable"] {
            let frame = m.getattr("Frame").unwrap().call1((m.getattr(field).unwrap(),)).unwrap();
            let started = clock.call0().unwrap().extract::<f64>().unwrap();
            let result = crate::utils::video::VideoFrame::from_host(&frame, Some(4));
            let elapsed = clock.call0().unwrap().extract::<f64>().unwrap() - started;
            let Err(error) = result else {
                panic!("oversized host buffer was accepted");
            };
            let cause = error.cause(py).expect("field extraction preserves the size error");
            assert!(cause.is_instance_of::<PyValueError>(py));
            assert!(cause.to_string().contains("limit is 4"));
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

#[test]
fn audio_frame_reads_the_buffer_protocol_without_calling_tobytes() {
    attach(|py| {
        let m = module(
            py,
            r#"
from types import SimpleNamespace
class Samples(bytearray):
    def tobytes(self):
        raise AssertionError('must use buffer protocol')
samples = Samples(b'\x01\x00\x02\x00')
frame = SimpleNamespace(data=samples, sample_rate=16000, num_channels=1, samples_per_channel=2)
"#,
        );
        let frame = m.getattr("frame").unwrap().extract::<crate::AudioFrame>().unwrap();
        m.getattr("samples").unwrap().set_item(0, 9).unwrap();
        assert_eq!(&*frame.data, &[1, 0, 2, 0]);
        let host = (&frame).into_pyobject(py).unwrap();
        let data = host.getattr("data").unwrap();
        if data.getattr("readonly").unwrap().extract::<bool>().unwrap() {
            assert!(data.set_item(0, 9).is_err());
        } else {
            data.set_item(0, 9).unwrap();
        }
        assert_eq!(&*frame.data, &[1, 0, 2, 0], "host output must not alias native storage");
        m.getattr("frame").unwrap().setattr("sample_rate", true).unwrap();
        assert!(
            m.getattr("frame").unwrap().extract::<crate::AudioFrame>().is_err(),
            "frame integer metadata rejects bool"
        );
    });
}

#[test]
fn video_frame_snapshots_a_sliced_mutable_buffer() {
    attach(|py| {
        let m = module(
            py,
            r#"
from types import SimpleNamespace
owner = bytearray(b'01234567')
frame = SimpleNamespace(data=memoryview(owner)[2:6], width=1, height=1, type=0)
"#,
        );
        let frame =
            m.getattr("frame").unwrap().extract::<crate::utils::video::VideoFrame>().unwrap();
        m.getattr("owner").unwrap().set_item(2, 99).unwrap();
        assert_eq!(&*frame.data, b"2345");
        m.getattr("frame").unwrap().setattr("width", true).unwrap();
        assert!(
            m.getattr("frame").unwrap().extract::<crate::utils::video::VideoFrame>().is_err(),
            "frame integer metadata rejects bool"
        );
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
    value: crate::Maybe<Option<String>>,
}

#[test]
fn generated_presence_roundtrips_omitted_null_and_empty_values() {
    attach(|py| {
        for literal in ["{}", "{'value': None}", "{'value': ''}"] {
            let source = module(py, &format!("source = {literal}\n")).getattr("source").unwrap();
            let record = source.extract::<NullableUpdate>().unwrap();
            let output = record.into_pyobject(py).unwrap();
            assert!(source.eq(output).unwrap());
        }
    });
}

#[test]
fn tool_defaults_apply_only_to_missing_fields() {
    attach(|py| {
        let value =
            module(py, "value = {'name': 'lookup', 'json_schema': {}}\n").getattr("value").unwrap();
        let tool = value.extract::<crate::llm::llm::ToolSpec>().unwrap();
        assert_eq!(tool.description, "");
        assert!(!tool.flags.ignore_on_enter);
        value.set_item("description", py.None()).unwrap();
        assert!(
            value.extract::<crate::llm::llm::ToolSpec>().is_err(),
            "null is not a missing description"
        );
    });
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
        let native = crate::utils::dates::Date::new(2024, 2, 29).unwrap();
        let host = native.into_pyobject(py).unwrap();
        assert_eq!(host.extract::<crate::utils::dates::Date>().unwrap(), native);
        let invalid = module(py, "from types import SimpleNamespace\nvalue = SimpleNamespace(year=2024, month=2, day=30)\n").getattr("value").unwrap();
        assert!(extract_at::<crate::utils::dates::Date>(&invalid, "birthday").is_err());
    });
}

#[derive(Default)]
struct GuardedValue(u32);

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
