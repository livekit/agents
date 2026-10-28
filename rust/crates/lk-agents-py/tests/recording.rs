//! The shipped generated recorder is the session's native tap, backed by the runtime sink.

#![cfg(feature = "ffmpeg")]

use std::sync::Arc;

use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};
use pyo3::prelude::*;

struct RecordingDirectory(std::path::PathBuf);

impl Drop for RecordingDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn generated_recorder_is_the_runtime_tap_and_finalizes_a_readable_ogg_file() {
    Python::initialize();
    let directory = RecordingDirectory(
        std::env::temp_dir().join(format!("generated-recorder-{}", std::process::id())),
    );
    let path = directory.0.join("audio.ogg");
    let (object, tap): (Py<PyAny>, Arc<dyn AudioRecorder>) = Python::attach(|py| {
        let module = PyModule::new(py, "recording_test").unwrap();
        _core::generated::register(&module).unwrap();
        let object =
            module.getattr("SessionRecorder").unwrap().call1((path.to_str().unwrap(),)).unwrap();
        let tap = lk_agents_abi::python::native::<dyn AudioRecorder>(&object)
            .expect("the generated class must expose the actual native tap");
        (object.unbind(), tap)
    });
    // The class's generated storage is the actual SessionRecorder. Feed it exactly as the live
    // room does, then finalize through its declared host method on a real asyncio loop.
    let samples =
        (0..4800).map(|sample| ((sample * 37) % 20_000) as i16 - 10_000).collect::<Vec<_>>();
    let frame =
        lk_agents_abi::AudioFrame::new(lk_agents_media::pcm_bytes(&samples), 48_000, 1, 4800);
    tap.on_input_frame(0.1, &frame);
    tap.flush(0.1).unwrap();
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("recorder", object.bind(py)).unwrap();
        py.run(
            c"import asyncio\nasync def finish():\n    await recorder.ready()\n    await recorder.finish()\nasyncio.run(finish())",
            Some(&globals),
            None,
        )
        .unwrap();
        assert_eq!(
            object.bind(py).getattr("bytes_written").unwrap().extract::<u64>().unwrap(),
            std::fs::metadata(&path).unwrap().len()
        );
    });
    assert!(matches!(tap.flush(0.2), Err(RecordingFailure::Closed)));
    drop(tap);
    drop(object);
    let mut reader = std::fs::File::open(&path).unwrap();
    let mut samples = 0_usize;
    lk_agents_media::ffmpeg::decode_stream(
        Some("ogg"),
        48_000,
        2,
        lk_agents_media::decoder::DecodeLimits {
            min_sample_rate: 8000,
            max_sample_rate: 192000,
            max_channels: 2,
            max_streams: 1,
            max_expansion_ratio: 128,
            max_samples_per_frame: 192000,
        },
        &mut reader,
        |format, pcm| {
            assert_eq!(format.num_channels, 2);
            samples += pcm.len() / 2;
            Ok(())
        },
    )
    .unwrap();
    assert!(samples.abs_diff(4800) <= 1920, "Opus boundary padding only: {samples}");
}

#[test]
fn generated_recorder_reports_filesystem_open_failures_as_oserror() {
    Python::initialize();
    let directory = RecordingDirectory(
        std::env::temp_dir().join(format!("generated-recorder-open-error-{}", std::process::id())),
    );
    std::fs::create_dir_all(&directory.0).unwrap();
    let blocker = directory.0.join("regular-file");
    std::fs::write(&blocker, b"not a directory").unwrap();
    let path = blocker.join("audio.ogg");
    Python::attach(|py| {
        let module = PyModule::new(py, "recording_open_error_test").unwrap();
        _core::generated::register(&module).unwrap();
        let recorder =
            module.getattr("SessionRecorder").unwrap().call1((path.to_str().unwrap(),)).unwrap();
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("recorder", recorder).unwrap();
        let error = py
            .run(
                c"import asyncio\nasync def open_recording():\n    await recorder.ready()\nasyncio.run(open_recording())",
                Some(&globals),
                None,
            )
            .unwrap_err();
        assert!(error.is_instance_of::<pyo3::exceptions::PyOSError>(py), "{error}");
    });
}
