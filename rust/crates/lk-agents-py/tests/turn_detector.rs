//! The turn detector's prediction boundary: measurements retain zero versus missing values,
//! and an unrelated host object cannot masquerade as a prediction.

use _core::generated::records::turn_events::eot_prediction_from_py;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn detector_predictions_preserve_measurements_and_refuse_untyped_results() {
    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        globals
            .set_item(
                "MODELS",
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../../livekit-agents-next/livekit/agents/_generated/turn_events.py"
                ),
            )
            .unwrap();
        py.run(
            c"import importlib.util, sys, types
for name in ('livekit', 'livekit.agents', 'livekit.agents._generated'):
    sys.modules.setdefault(name, types.ModuleType(name))
spec = importlib.util.spec_from_file_location('livekit.agents._generated.turn_events', MODELS)
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)
prediction = models.EotPrediction(
    end_of_turn_probability=0.0, last_speaking_time=12.5,
    detection_delay=0.0, inference_duration=None, backchannel_probability=0.0,
)",
            Some(&globals),
            None,
        )
        .unwrap();
        let prediction = globals.get_item("prediction").unwrap().unwrap();
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
