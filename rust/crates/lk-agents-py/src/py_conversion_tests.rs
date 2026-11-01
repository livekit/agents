//! Boundary regressions against a real interpreter, including the reverse conversion.

use _core::py_calls;

#[path = "py_test_support.rs"]
pub mod py_test_support;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use _core::py_objects::*;
use _core::py_values::*;

fn attach(f: impl FnOnce(Python<'_>)) {
    Python::initialize();
    static FIXTURES: std::sync::Once = std::sync::Once::new();
    FIXTURES.call_once(|| {
        Python::attach(|py| {
            // Isolate this process from the public package and SDK installation.
            // Pytest conformance covers actual SDK classes and generated public entrypoints.
            module(py, "install_stub_package");
        })
    });
    Python::attach(f);
}

fn module<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyAny> {
    py_test_support::fixture(py, name)
}

#[test]
fn omission_is_only_not_given_and_none_is_a_given_value() {
    attach(|py| {
        let read = |case: &str| {
            let value = py_test_support::call(py, "omission_value", (case,)).unwrap();
            value.extract::<MaybeNull<f64>>().map(|v| v.into_core())
        };
        assert!(matches!(read("not_given").unwrap(), lk_agents_abi::Maybe::NotGiven));
        assert!(matches!(read("none").unwrap(), lk_agents_abi::Maybe::Given(None)));
        assert!(matches!(read("zero").unwrap(), lk_agents_abi::Maybe::Given(Some(0.0))));
        assert!(read("invalid").is_err());
    });
}

#[test]
fn defaults_preserve_lookup_exception_identity_and_cause() {
    attach(|py| {
        let m = module(py, "defaults_preserve_lookup_exception_identity_and_cause");
        let obj = m.getattr("broken").unwrap();
        let failure = m.getattr("failure").unwrap();
        for err in [
            attr_or(&obj, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap_err(),
            item_or(&obj, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap_err(),
        ] {
            assert!(err.value(py).is(&failure));
            assert!(err.cause(py).unwrap().value(py).is(m.getattr("cause").unwrap()));
        }
        let err = item_or(
            &m.getattr("broken_dict").unwrap(),
            "Record",
            "field",
            |v| v.extract::<u32>(),
            || 7,
        )
        .unwrap_err();
        assert!(err.value(py).is(m.getattr("key_failure").unwrap()));
        assert!(err.cause(py).unwrap().value(py).is(m.getattr("cause").unwrap()));
        let missing = PyDict::new(py);
        assert_eq!(item_or(&missing, "Record", "field", |v| v.extract::<u32>(), || 7).unwrap(), 7);
        let obj = module(py, "empty_object").getattr("empty").unwrap();
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
        let m = module(py, "large_video_snapshots_remain_isolated_from_mutable_host_aliases");
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
            "large_buffer_allocation_lets_other_python_threads_progress_before_the_snapshot",
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
        _core::py_calls::call(&m.getattr("thread").unwrap().getattr("join").unwrap(), (), None)
            .unwrap();
        assert!(progressed, "large native allocation must release the GIL before copying");
    });
}

#[test]
fn bounded_buffers_refuse_oversized_host_buffers_before_copying() {
    attach(|py| {
        let m = module(py, "bounded_buffers_refuse_oversized_host_buffers_before_copying");
        let clock = py.import("time").unwrap().getattr("thread_time").unwrap();
        for field in ["immutable", "mutable"] {
            let now = || _core::py_calls::call(&clock, (), None).unwrap().extract::<f64>().unwrap();
            let started = now();
            let result = retained_bytes(&m.getattr(field).unwrap(), "VideoFrame.data", Some(4));
            let elapsed = now() - started;
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
        let m = module(py, "retained_buffers_keep_immutable_owners_and_snapshot_writable_aliases");
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
        let obj = module(py, "a_failing_property_is_not_an_absent_attribute");
        let err = attr_or(
            &obj.getattr("broken").unwrap(),
            "Record",
            "field",
            |v| v.extract::<u32>(),
            || 7,
        )
        .unwrap_err();
        assert!(err.value(py).is(obj.getattr("failure").unwrap()));
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
        let lone_surrogate =
            module(py, "unicode_float_and_newtype_boundaries_keep_their_declared_semantics")
                .getattr("value")
                .unwrap();
        assert!(
            lone_surrogate.extract::<String>().is_err(),
            "Rust Unicode strings reject lone surrogates"
        );
    });
}

mod records {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// One tool spec with a declared omission or null case applied.
    fn parse_spec(py: Python<'_>, case: &str) -> PyResult<lk_agents_abi::ToolSpec> {
        let spec = crate::py_test_support::call(py, "tool_spec", (case,))?;
        _core::generated::records::tool_options::tool_spec_from_py(&spec)
    }

    // Catches: a key that reaches this reader and stops there. `ignore_on_enter` is asserted `true`
    // because `false` is both the default and what the N184 bug produced.
    #[test]
    fn every_key_of_a_tool_spec_lands_including_the_flag_that_used_to_be_dropped() {
        super::attach(|py| {
            let spec = parse_spec(py, "full").expect("the producers' dict reads");
            assert_eq!(
                (spec.name.as_str(), spec.description.as_str()),
                ("lookup", "look something up")
            );
            assert_eq!(spec.json_schema["type"], "object");
            assert!(spec.provider_tool.is_none());
            assert!(
                spec.flags.ignore_on_enter,
                "the flag was read as the default rather than the value"
            );
        });
    }

    // Defaults apply only to omission. Explicit JSON null remains a JSON value; fields whose
    // types are not nullable must refuse it.
    #[test]
    fn tool_defaults_apply_to_omission_and_nonnullable_fields_refuse_null() {
        super::attach(|py| {
            assert_eq!(parse_spec(py, "null_schema").unwrap().json_schema, serde_json::Value::Null);
            assert!(parse_spec(py, "null_description").is_err());
            assert!(parse_spec(py, "null_ignore").is_err());
            assert!(parse_spec(py, "missing_schema").is_err());
            assert!(!parse_spec(py, "missing_ignore").unwrap().flags.ignore_on_enter);
            assert_eq!(parse_spec(py, "missing_description").unwrap().description, "");
        });
    }

    // Catches: copying the audio of every VAD event off the host (`inference_done` fires ~30 a
    // second per stream) when only an end of speech's is recognized.
    #[test]
    fn only_an_end_of_speech_carries_its_audio_off_the_host() {
        super::attach(|py| {
            let read = |kind: &str, frames: usize| {
                let event = crate::py_test_support::call(py, "vad_event", (kind, frames)).unwrap();
                _core::generated::records::vad::vad_event_from_py(&event).unwrap()
            };
            assert!(read("inference_done", 1).frames.is_empty());
            let ev = read("end_of_speech", 2);
            assert_eq!(ev.frames.len(), 2);
        });
    }

    // Catches: a date saturated or wrapped into range instead of refused.
    #[test]
    fn a_date_crosses_both_ways_and_an_impossible_one_is_refused() {
        super::attach(|py| {
            let leap = lk_agents_abi::utils::dates::Date::new(2024, 2, 29).unwrap();
            assert_eq!(
                _core::generated::records::dates::date_from_py(
                    &_core::generated::records::dates::date_to_py(py, leap).unwrap()
                )
                .unwrap(),
                leap
            );
            let globals = crate::py_test_support::fixture(
                py,
                "a_date_crosses_both_ways_and_an_impossible_one_is_refused",
            );
            assert!(
                _core::generated::records::dates::date_from_py(&globals.getattr("out").unwrap())
                    .is_err()
            );
        });
    }

    // Catches: reading `data` through `tobytes()` or keeping a view the host can still mutate, a
    // bool taken as frame metadata, or the frame crossing back with its samples aliased.
    #[test]
    fn an_audio_frame_snapshots_its_buffer_both_ways() {
        super::attach(|py| {
            let globals = crate::py_test_support::fixture(
                py,
                "an_audio_frame_snapshots_its_buffer_both_ways",
            );
            let get = |name: &str| globals.getattr(name).unwrap();
            let frame =
                _core::generated::records::rtc_frames::audio_frame_from_py(&get("frame")).unwrap();
            get("samples").set_item(0, 9).unwrap();
            assert_eq!(&*frame.data, &[1, 0, 2, 0]);
            let host = _core::generated::records::rtc_frames::audio_frame_to_py(py, frame.clone())
                .unwrap();
            assert_eq!(host.getattr("samples_per_channel").unwrap().extract::<u32>().unwrap(), 2);
            let data = host.getattr("data").unwrap();
            if !data.getattr("readonly").unwrap().extract::<bool>().unwrap() {
                data.set_item(0, 9).unwrap();
            }
            assert_eq!(&*frame.data, &[1, 0, 2, 0], "host output must not alias native storage");
            get("frame").setattr("sample_rate", true).unwrap();
            assert!(
                _core::generated::records::rtc_frames::audio_frame_from_py(&get("frame")).is_err(),
                "metadata rejects bool"
            );
        });
    }

    // Catches: a frame keeping a view of a host buffer the host goes on mutating, or a bool taken
    // as frame metadata.
    #[test]
    fn a_video_frame_snapshots_a_sliced_mutable_buffer() {
        super::attach(|py| {
            let globals = crate::py_test_support::fixture(
                py,
                "a_video_frame_snapshots_a_sliced_mutable_buffer",
            );
            let get = |name: &str| globals.getattr(name).unwrap();
            let frame =
                _core::generated::records::rtc_frames::video_frame_from_py(&get("frame")).unwrap();
            get("owner").set_item(2, 99).unwrap();
            assert_eq!(&*frame.data, b"2345");
            get("frame").setattr("width", true).unwrap();
            assert!(
                _core::generated::records::rtc_frames::video_frame_from_py(&get("frame")).is_err(),
                "metadata rejects bool"
            );
        });
    }

    // Catches: a stale or foreign key resolving to null, or a foreign table's key resolving the
    // local entry that happens to share its id.
    #[test]
    fn object_refs_preserve_identity_and_reject_stale_or_foreign_registries() {
        use _core::py_objects::{Table, object_table, resolve, retain};
        use lk_agents_abi::utils::host_ref::{HostObject, MediaObject, ObjectRef};

        fn check<Kind>(py: Python<'_>) {
            let stale_name = |reference: ObjectRef<Kind>| {
                super::object_ref_to_py(py, reference)
                    .unwrap_err()
                    .get_type(py)
                    .name()
                    .unwrap()
                    .to_string()
            };
            assert_eq!(
                stale_name(ObjectRef::<Kind>::new(u64::MAX, object_table().registry())),
                "StaleHandle"
            );
            let original = PyDict::new(py);
            let local = retain::<Kind>(original.clone().into_any().unbind());
            assert!(resolve(py, &local).unwrap().bind(py).is(&original));
            assert!(super::object_ref_to_py(py, local.clone()).unwrap().is(&original));
            let foreign_table = Table::new();
            // An identical numeric ID must never resolve an object from another registry.
            let foreign = ObjectRef::<Kind>::new(local.id(), foreign_table.registry());
            assert!(resolve(py, &foreign).is_err());
            assert_eq!(stale_name(foreign), "StaleHandle");
            assert!(resolve(py, &local).unwrap().bind(py).is(&original));
        }

        super::attach(|py| {
            assert_ne!(
                std::any::TypeId::of::<lk_agents_abi::HostRef>(),
                std::any::TypeId::of::<lk_agents_abi::MediaRef>(),
                "retained reference roles remain distinct Rust types"
            );
            check::<HostObject>(py);
            check::<MediaObject>(py);
        });
    }

    mod predictions {
        //! The turn detector's prediction boundary: measurements retain zero versus missing values,
        //! and an unrelated host object cannot masquerade as a prediction.

        use _core::generated::records::turn_events::eot_prediction_from_py;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        #[test]
        fn detector_predictions_preserve_measurements_and_refuse_untyped_results() {
            super::super::attach(|py| {
                let prediction = crate::py_test_support::call(
                    py,
                    "detector_prediction",
                    (concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../../livekit-agents-next/livekit/agents/_generated/turn_events.py"
                    ),),
                )
                .unwrap();
                let result = eot_prediction_from_py(&prediction).unwrap();
                assert_eq!(result.end_of_turn_probability, 0.0);
                assert_eq!(result.last_speaking_time, 12.5);
                assert_eq!(result.detection_delay, Some(0.0));
                assert_eq!(result.inference_duration, None);
                assert_eq!(result.backchannel_probability, Some(0.0));

                let untyped = PyDict::new(py);
                untyped.set_item("end_of_turn_probability", 0.0).unwrap();
                let error = eot_prediction_from_py(untyped.as_any()).unwrap_err();
                assert!(error.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
                assert!(error.to_string().contains("expected EotPrediction"));
            });
        }
    }
}
