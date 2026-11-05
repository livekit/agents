//! Generated adapters against the real public package: values, callbacks and native resources.

use _core::py_calls;

#[path = "py_test_support.rs"]
pub mod py_test_support;

mod models {
    use pyo3::prelude::*;

    #[test]
    fn room_options_are_read_from_resolved_objects_without_a_dictionary_wire() {
        use _core::generated::records::room_options::room_io_config_from_py;
        use lk_agents_abi::voice::room_io::RoomIoConfig;
        use lk_agents_abi::voice::rtc::TrackSource;
        use std::time::Duration;

        crate::py_test_support::install_package(_core::register);
        Python::attach(|py| {
            let read = |case: &str| {
                let value = crate::py_test_support::call(py, "room_options_for_native", (case,))?;
                room_io_config_from_py(&value)
            };
            assert_eq!(read("default").unwrap(), RoomIoConfig::default());
            let configured = read("configured").unwrap();
            assert_eq!(configured.participant_identity.unwrap().0, "caller");
            let input = configured.audio_input.unwrap();
            assert_eq!(input.sample_rate, 48_000);
            assert_eq!(input.pre_connect_audio_timeout, Duration::ZERO);
            assert!(!input.auto_gain_control);
            assert!(input.noise_cancellation.is_some());
            let output = configured.audio_output.unwrap().publish;
            assert_eq!(output.source, TrackSource::ScreenshareAudio);
            assert_eq!(output.name, "speaker");
            assert_eq!(configured.text_output.unwrap().sync_transcription, Some(false));
            let disabled = read("disabled").unwrap();
            assert!(disabled.audio_input.is_none() && disabled.audio_output.is_none());
            assert!(disabled.text_output.is_none());
            assert!(!disabled.text_input && !disabled.video_input);
            assert!(read("invalid_participant_kind").is_err());
        });
    }

    // Import the public package once before either test takes the GIL. Concurrent imports of
    // different submodules can otherwise form a cycle through Python's per-module import locks.

    mod audio_frames {
        //! The native receiver's declared output retains the SDK frame and public segment marker.

        use _core::generated::records::avatar::{
            audio_receiver_frame_from_py, audio_receiver_frame_to_py,
        };
        use lk_agents_abi::AudioFrame;
        use lk_agents_abi::utils::video::AudioReceiverFrame;
        use pyo3::prelude::*;

        #[test]
        fn receiver_frames_keep_the_sdk_audio_and_public_segment_end_types() {
            crate::py_test_support::install_package(_core::register);
            Python::attach(|py| {
                // The package itself, beside the crate; its dependencies come from the dev venv's
                // site-packages on `PYTHONPATH` (its `.pth` installs need a site directory).
                let frame = AudioFrame::new(vec![0; 640], 16000, 1, 320);
                let audio =
                    audio_receiver_frame_to_py(py, AudioReceiverFrame::Audio(frame)).unwrap();
                let audio_class = py.import("livekit.rtc").unwrap().getattr("AudioFrame").unwrap();
                assert!(audio.get_type().is(&audio_class));
                assert_eq!(
                    audio.getattr("samples_per_channel").unwrap().extract::<u32>().unwrap(),
                    320
                );

                assert_eq!(
                    audio_receiver_frame_from_py(&audio).unwrap(),
                    AudioReceiverFrame::Audio(AudioFrame::new(vec![0; 640], 16000, 1, 320))
                );
                assert!(audio_receiver_frame_from_py(py.None().bind(py)).is_err());
                let video = _core::generated::records::rtc_frames::video_frame_to_py(
                    py,
                    lk_agents_abi::utils::video::VideoFrame {
                        width: 1,
                        height: 1,
                        buffer_type: 0,
                        data: vec![0u8; 4].into(),
                    },
                )
                .unwrap();
                assert!(audio_receiver_frame_from_py(&video).is_err());

                let end = audio_receiver_frame_to_py(py, AudioReceiverFrame::SegmentEnd).unwrap();
                let marker = py
                    .import("livekit.agents.voice.avatar")
                    .unwrap()
                    .getattr("AudioSegmentEnd")
                    .unwrap();
                assert!(end.get_type().is(&marker));
                assert!(matches!(
                    audio_receiver_frame_from_py(&end).unwrap(),
                    AudioReceiverFrame::SegmentEnd
                ));

                let alias = py
                    .import("livekit.agents._generated.avatar")
                    .unwrap()
                    .getattr("AudioReceiverFrame")
                    .unwrap();
                let members =
                    py.import("typing").unwrap().call_method1("get_args", (alias,)).unwrap();
                assert!(members.get_item(0).unwrap().is(&audio_class));
                assert!(members.get_item(1).unwrap().is(&marker));
            });
        }
    }

    mod steering {
        //! `SpeechSteeringOptions` as the host writes it: absent and `None` keys alike, and
        //! `nonverbal_sounds` as `True`, `False` or a per-category mapping.

        use lk_agents_abi::tts::markup::{NonverbalSounds, Pace, SpeechSteering};
        use pyo3::exceptions::{PyTypeError, PyValueError};
        use pyo3::prelude::*;

        // Catches: `nonverbal_sounds` disagreeing with the declaration about absent-vs-`None`, or the
        // bool half being read through the dict reader.
        #[test]
        fn steering_reads_absent_and_none_alike_and_keeps_the_three_nonverbal_cases_apart() {
            crate::py_test_support::install_package(_core::register);
            Python::attach(|py| {
                // `NOT_GIVEN` is the package's: its dependencies' `.pth` installs need a site directory.
                let read = |case: &str| -> PyResult<SpeechSteering> {
                    _core::generated::records::expressive::speech_steering_options_from_py(
                        &crate::py_test_support::call(py, "steering_case", (case,))?,
                    )
                };
                assert_eq!(read("default").unwrap(), SpeechSteering::default());
                assert_eq!(
                    read("null_fields").unwrap(),
                    SpeechSteering::default(),
                    "an explicit None is absent, as the reference's `is not None` makes it"
                );
                let full = read("full").unwrap();
                assert_eq!(full.disfluencies, Some(false));
                assert_eq!(full.pace, Some(Pace::Slow));
                match full.nonverbal_sounds {
                    Some(NonverbalSounds::Per(opts)) => {
                        assert_eq!(opts.laughing, Some(false));
                        assert_eq!(opts.breathing, None, "an omitted category stays absent");
                    }
                    other => panic!("a dict is the per-category case, got {other:?}"),
                }
                assert_eq!(
                    read("enabled").unwrap().nonverbal_sounds,
                    Some(NonverbalSounds::Enabled(true))
                );
                assert_eq!(
                    read("disabled").unwrap().nonverbal_sounds,
                    Some(NonverbalSounds::Enabled(false))
                );
                let err = read("invalid_pace").unwrap_err();
                assert!(err.is_instance_of::<PyValueError>(py));
                assert_eq!(
                    err.value(py).to_string(),
                    "Pace: \"brisk\" is not one of 'slow', 'normal', 'fast'"
                );
                let err = read("invalid_sounds").unwrap_err();
                assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            });
        }
    }

    #[test]
    fn tool_flags_keep_the_public_enum_in_both_directions() {
        crate::py_test_support::install_package(_core::register);
        Python::attach(|py| {
            use _core::generated::records::mcp::{
                mcp_tool_options_from_py, resolved_tool_options_to_py,
            };
            use lk_agents_abi::llm::mcp::{McpToolOptions, ToolFlag};

            let flag = py
                .import("livekit.agents.llm.tool_context")
                .unwrap()
                .getattr("ToolFlag")
                .unwrap()
                .getattr("IGNORE_ON_ENTER")
                .unwrap();
            let options = pyo3::types::PyDict::new(py);
            options.set_item("flags", &flag).unwrap();
            let native = mcp_tool_options_from_py(options.as_any()).unwrap();
            assert_eq!(native.flags, Some(ToolFlag::IGNORE_ON_ENTER));
            let restored = resolved_tool_options_to_py(
                py,
                McpToolOptions { flags: native.flags.unwrap(), ..Default::default() },
            )
            .unwrap();
            assert!(restored.get_item("flags").unwrap().is(&flag));

            options.set_item("flags", 1).unwrap();
            assert!(
                mcp_tool_options_from_py(options.as_any())
                    .unwrap_err()
                    .is_instance_of::<pyo3::exceptions::PyTypeError>(py)
            );
        });
    }

    #[test]
    fn synthesis_metadata_is_a_complete_conversion_and_reads_the_frame_once() {
        crate::py_test_support::install_package(_core::register);
        Python::attach(|py| {
            let globals = crate::py_test_support::fixture(
                py,
                "synthesis_metadata_is_a_complete_conversion_and_reads_the_frame_once",
            );
            let chunk = globals.getattr("chunk").unwrap();
            let native = _core::generated::records::tts::synthesized_audio_from_py(&chunk).unwrap();
            assert_eq!(chunk.getattr("reads").unwrap().extract::<usize>().unwrap(), 1);
            assert_eq!(native.timed_transcript.len(), 1);
            assert_eq!(native.timed_transcript[0].text, "hello");
            assert_eq!(native.timed_transcript[0].start_time, Some(0.1));
            let wall = native.started_time.unwrap();
            let restored =
                _core::generated::records::tts::synthesized_audio_to_py(py, native).unwrap();
            let roundtrip =
                _core::generated::records::tts::synthesized_audio_from_py(&restored).unwrap();
            assert!((roundtrip.started_time.unwrap() - wall).abs() < 0.02);
            assert_eq!(roundtrip.timed_transcript[0].text, "hello");
            assert_eq!(roundtrip.frame.sample_rate, 16000);
        });
    }
}

mod proxies {
    use crate::py_test_support;
    use _core::generated::agent::AgentCallbacksProxy;
    use lk_agents_abi::agent::{Invocation, OutputCandidate, ValidatedOutput};
    use pyo3::prelude::*;

    /// A fresh host and loop for this native proxy test.
    fn host_globals(py: Python<'_>) -> Bound<'_, PyAny> {
        py_test_support::fixture(py, "agent_host")
    }

    fn invocation(context: lk_agents_abi::HostRef) -> Invocation {
        Invocation {
            operation: 7,
            invocation_id: "AI_1".into(),
            agent_name: "front".into(),
            execution_id: None,
            frame_id: None,
            context_scope: None,
            application_context: None,
            history: Vec::new(),
            function_call: None,
            approval: None,
            speech: None,
            host_context: Some(context),
        }
    }

    /// `invocation`'s copy for one tool call, as the HostInvoker derives it.
    fn tool_call(invocation: &Invocation) -> Invocation {
        let call = lk_agents_abi::FunctionCall {
            id: "item_1".into(),
            call_id: "call_1".into(),
            arguments: "{}".into(),
            name: "lookup".into(),
            created_at: std::time::SystemTime::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        Invocation { function_call: Some(call), ..invocation.clone() }
    }

    // Catches: a callback run outside the context its invocation captured, the captured context
    // handed to the host, a refusal read as a failure (or a failure as a refusal), a host shape read
    // by anything but its declaration, or a notification that never reaches the host.
    #[test]
    fn the_generated_agent_host_proxy_calls_the_host_in_the_captured_context() {
        crate::py_test_support::install_package(_core::register);
        let (globals, proxy, inv) = Python::attach(|py| {
            let globals = host_globals(py);
            let pair = globals.getattr("capture").unwrap().call1(("captured",)).unwrap();
            let context = _core::py_objects::retain(pair.unbind());
            let host = globals.getattr("host").unwrap();
            (globals.unbind(), AgentCallbacksProxy::from_host(&host).unwrap(), invocation(context))
        });
        let runtime = lk_agents_runtime::process::handle().unwrap();
        runtime.block_on(async {
            assert_eq!(proxy.render_instructions(&lk_agents_abi::telemetry::Cx::NONE, &inv).await.unwrap(), "be brief");
            let agent = inv.clone();
            let inv = tool_call(&agent);
            let args = |key: &str| {
                let mut m = serde_json::Map::new();
                m.insert(key.into(), true.into());
                m
            };
            let prepared = proxy.prepare_tool(&lk_agents_abi::telemetry::Cx::NONE, &inv, args("city")).await.unwrap().unwrap();
            assert_eq!(
                prepared.identity,
                lk_agents_abi::agent::ArgumentIdentity::Json(serde_json::json!({"city": true}))
            );
            let (report, extra) = proxy.run_tool(&lk_agents_abi::telemetry::Cx::NONE, &inv, &prepared.prepared).await.unwrap();
            assert!(matches!(
                report.outcome,
                lk_agents_abi::llm::tool_executor::ToolOutcome::Value { text, .. } if text == "answered"
            ));
            assert_eq!(extra["lk.delegation.task_id"], "original-task");
            for key in ["bad_arity", "bad_metadata"] {
                let prepared = proxy.prepare_tool(&lk_agents_abi::telemetry::Cx::NONE, &inv, args(key)).await.unwrap().unwrap();
                let error = proxy.run_tool(&lk_agents_abi::telemetry::Cx::NONE, &inv, &prepared.prepared).await.unwrap_err();
                assert_eq!(error.kind, "TypeError");
                let expected = if key == "bad_arity" { "tuple of 2 items" } else { "expected a dict" };
                assert!(error.message.contains(expected), "{error}");
            }
            let refused = proxy.prepare_tool(&lk_agents_abi::telemetry::Cx::NONE, &inv, args("bad")).await.unwrap().unwrap_err();
            assert_eq!(refused.message, "city is required");
            let failed = proxy.prepare_tool(&lk_agents_abi::telemetry::Cx::NONE, &inv, args("crash")).await.unwrap_err();
            assert_eq!(failed.kind, "KeyError");
            let verdict =
                proxy.validate_output(&lk_agents_abi::telemetry::Cx::NONE, &agent, OutputCandidate::Value(serde_json::json!(3))).await;
            assert!(matches!(verdict, Ok(ValidatedOutput::Retry(r)) if r == "not a number"));
        });
        proxy.cancel(&inv);
        let cancelled = || {
            Python::attach(|py| {
                let host = globals.bind(py).getattr("host").unwrap();
                host.getattr("cancelled")
                    .unwrap()
                    .call_method0("is_set")
                    .unwrap()
                    .is_truthy()
                    .unwrap()
            })
        };
        for _ in 0..500 {
            if cancelled() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(cancelled(), "the notification never reached the host");
        Python::attach(|py| {
            let host = globals.bind(py).getattr("host").unwrap();
            let seen = host.getattr("seen").unwrap();
            let (var, invocation): (String, Bound<'_, PyAny>) =
                seen.get_item("render").unwrap().extract().unwrap();
            assert_eq!(var, "captured");
            assert_eq!(
                invocation.getattr("agent_name").unwrap().extract::<String>().unwrap(),
                "front"
            );
            assert!(invocation.getattr("host_context").is_err(), "the context stays native");
            let candidate = seen.get_item("candidate").unwrap();
            let class = candidate.get_type().name().unwrap().to_string();
            assert_eq!(class, "OutputCandidateValue");
            assert_eq!(candidate.getattr("value").unwrap().extract::<i64>().unwrap(), 3);
            let (var, operation): (String, u64) =
                seen.get_item("cancel").unwrap().extract().unwrap();
            assert_eq!((var.as_str(), operation), ("captured", 7));
        });
    }

    // Catches: a tool callback dispatched without the tool call it belongs to (the host would run a
    // body with no call scope), or a refusal that enters the host anyway.
    #[test]
    fn a_tool_callback_without_a_tool_call_scope_is_refused_before_the_host() {
        crate::py_test_support::install_package(_core::register);
        let (globals, proxy, inv) = Python::attach(|py| {
            let globals = host_globals(py);
            crate::py_test_support::call(
                py,
                "a_tool_callback_without_a_tool_call_scope_is_refused_before_the_host",
                (globals.getattr("host").unwrap(),),
            )
            .unwrap();
            let pair = globals.getattr("capture").unwrap().call1(("captured",)).unwrap();
            let context = _core::py_objects::retain(pair.unbind());
            let host = globals.getattr("host").unwrap();
            (globals.unbind(), AgentCallbacksProxy::from_host(&host).unwrap(), invocation(context))
        });
        let runtime = lk_agents_runtime::process::handle().unwrap();
        let error = runtime
            .block_on(proxy.prepare_tool(
                &lk_agents_abi::telemetry::Cx::NONE,
                &inv,
                serde_json::Map::new(),
            ))
            .unwrap_err();
        assert_eq!(error.kind, "RuntimeError");
        assert!(
            error
                .message
                .contains("AgentCallbacks.prepare_tool: the call requires a tool call scope"),
            "{error}"
        );
        Python::attach(|py| {
            let seen = globals.bind(py).getattr("host").unwrap().getattr("seen").unwrap();
            assert!(seen.get_item("entered").is_err(), "the host was entered");
        });
    }

    // Catches: a record read from anything but its model (a kind-dict, a look-alike), an explicit
    // None accepted for a field that is not optional, or a default applied to anything but omission.
    #[test]
    fn records_cross_as_their_models_with_declared_presence() {
        use lk_agents_abi::agent::Retries;
        crate::py_test_support::install_package(_core::register);
        Python::attach(|py| {
            let case =
                |name: &str| py_test_support::call(py, "agent_record_case", (name,)).unwrap();

            let accepted = case("accepted");
            assert!(matches!(
                _core::generated::records::agent_run::validated_output_from_py(&accepted).unwrap(),
                ValidatedOutput::Accepted { text: None, value: None, serialized: None }
            ));
            let dict =
                _core::generated::records::agent_run::validated_output_from_py(&case("dict"))
                    .unwrap_err();
            assert!(dict.is_instance_of::<pyo3::exceptions::PyTypeError>(py), "{dict}");
            assert!(
                dict.to_string().contains("ValidatedOutputAccepted | ValidatedOutputRetry"),
                "{dict}"
            );

            let null = _core::generated::records::agent_run::read_back_answer_from_py(&case(
                "null_question",
            ))
            .unwrap_err();
            assert!(null.to_string().contains("ReadBackAnswerInputRequired.question"), "{null}");

            assert_eq!(
                _core::generated::records::agent_run::retries_from_py(&case("default_retries"))
                    .unwrap(),
                Retries { tools: 1, output: 1 }
            );
            assert_eq!(
                _core::generated::records::agent_run::retries_from_py(&case("zero_output"))
                    .unwrap(),
                Retries { tools: 1, output: 0 }
            );
            assert!(
                _core::generated::records::agent_run::retries_from_py(&case("null_tools")).is_err(),
                "None is not omission"
            );
        });
    }

    // Catches: a writer a host call returned taken off the host loop ("no running event loop"),
    // so a real room's session messages (the simulator's readiness check) never leave.
    #[test]
    fn a_returned_writer_runs_in_its_calls_context() {
        use _core::generated::voice::LocalParticipantProxy;
        use lk_agents_abi::voice::rtc::StreamBytesOptions;
        crate::py_test_support::install_package(_core::register);
        let (globals, participant) = Python::attach(|py| {
            let globals = crate::py_test_support::fixture(py, "byte_stream_host");
            let context = _core::py_objects::retain(globals.getattr("pair").unwrap().unbind());
            let host = globals.getattr("participant").unwrap();
            (globals.unbind(), LocalParticipantProxy::in_context(&host, &context).unwrap())
        });
        let runtime = lk_agents_runtime::process::handle().unwrap();
        runtime.block_on(async {
            let opts =
                StreamBytesOptions { topic: "lk.agent.session".into(), ..Default::default() };
            let writer =
                participant.stream_bytes(&lk_agents_abi::telemetry::Cx::NONE, opts).await.unwrap();
            assert_eq!(writer.id(), "BS_1");
            writer.write(&lk_agents_abi::telemetry::Cx::NONE, b"ready").await.unwrap();
            writer.close(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap();
        });
        Python::attach(|py| {
            let sent: Vec<(String, Vec<u8>)> =
                globals.bind(py).getattr("sent").unwrap().extract().unwrap();
            assert_eq!(sent, [("lk.agent.session".to_owned(), b"ready".to_vec())]);
        });
    }
}

#[cfg(feature = "ffmpeg")]
mod recording {
    use crate::py_test_support;
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
    fn generated_recorder_exposes_the_native_tap_and_finishes_its_output() {
        crate::py_test_support::install_package(_core::register);
        let directory = RecordingDirectory(
            std::env::temp_dir().join(format!("generated-recorder-{}", std::process::id())),
        );
        let path = directory.0.join("audio.ogg");
        let (object, tap): (Py<PyAny>, Arc<dyn AudioRecorder>) = Python::attach(|py| {
            let module = PyModule::new(py, "recording_test").unwrap();
            _core::generated::register(&module).unwrap();
            let object = module
                .getattr("SessionRecorder")
                .unwrap()
                .call1((path.to_str().unwrap(),))
                .unwrap();
            let tap = _core::py_objects::native::<dyn AudioRecorder>(&object)
                .expect("the generated class must expose the actual native tap");
            (object.unbind(), tap)
        });
        // The class's generated storage is the actual SessionRecorder. Feed it exactly as the live
        // room does, then finalize through its declared host method on a real asyncio loop.
        let samples =
            (0..4800).map(|sample| ((sample * 37) % 20_000) as i16 - 10_000).collect::<Vec<_>>();
        let frame = lk_agents_abi::AudioFrame::new(
            lk_agents_runtime::media::pcm_bytes(&samples),
            48_000,
            1,
            4800,
        );
        tap.on_input_frame(0.1, &frame);
        tap.flush(0.1).unwrap();
        Python::attach(|py| {
            py_test_support::call(
                py,
                "generated_recorder_exposes_the_native_tap_and_finishes_its_output",
                (object.bind(py),),
            )
            .unwrap();
            let bytes = object.bind(py).getattr("bytes_written").unwrap().extract::<u64>().unwrap();
            assert!(bytes > 0, "finish must settle the native tap's pending output");
            assert_eq!(bytes, std::fs::metadata(&path).unwrap().len());
        });
        assert!(matches!(tap.flush(0.2), Err(RecordingFailure::Closed)));
        drop(tap);
        drop(object);
    }

    #[test]
    fn generated_recorder_reports_filesystem_open_failures_as_oserror() {
        crate::py_test_support::install_package(_core::register);
        let directory = RecordingDirectory(
            std::env::temp_dir()
                .join(format!("generated-recorder-open-error-{}", std::process::id())),
        );
        std::fs::create_dir_all(&directory.0).unwrap();
        let blocker = directory.0.join("regular-file");
        std::fs::write(&blocker, b"not a directory").unwrap();
        let path = blocker.join("audio.ogg");
        Python::attach(|py| {
            let module = PyModule::new(py, "recording_open_error_test").unwrap();
            _core::generated::register(&module).unwrap();
            let recorder = module
                .getattr("SessionRecorder")
                .unwrap()
                .call1((path.to_str().unwrap(),))
                .unwrap();
            let error = py_test_support::call(
                py,
                "generated_recorder_reports_filesystem_open_failures_as_oserror",
                (recorder,),
            )
            .unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyOSError>(py), "{error}");
        });
    }
}

mod telemetry {
    use lk_agents_abi::invoke::CallContext;
    use lk_agents_abi::telemetry::record::{SpanContext};
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;

    // Catches: a host call made under a span not running with it current (an application span
    // started there would start a new trace), or the activation leaking into the captured context.
    #[test]
    fn a_host_call_runs_with_its_span_current() {
        crate::py_test_support::install_package(_core::register);
        let (globals, context) = Python::attach(|py| {
            let globals = crate::py_test_support::fixture(py, "telemetry_host");
            let pair = globals.getattr("pair").unwrap();
            (globals.unbind(), _core::py_objects::retain(pair.unbind()))
        });
        let span = SpanContext { trace_id: 0xabc, span_id: 0x42, sampled: true };
        let observe = |globals: &Py<PyAny>| -> _core::py_calls::HostCall {
            let globals = Python::attach(|py| globals.clone_ref(py));
            Box::new(move |py| Ok((globals.bind(py).getattr("observe")?, PyTuple::empty(py))))
        };
        let runtime = lk_agents_runtime::process::handle().unwrap();
        let (under, outside) = runtime.block_on(async {
            let scope = CallContext { host_context: Some(context.clone()), ..Default::default() };
            let entry = _core::py_calls::Entry;
            let under = lk_agents_abi::telemetry::Cx::NONE.continuing(Some(span));
            let a = lk_agents_runtime::host::call(
                &entry,
                "test",
                &scope,
                &[],
                &under,
                observe(&globals),
            );
            let none = &lk_agents_abi::telemetry::Cx::NONE;
            let b =
                lk_agents_runtime::host::call(&entry, "test", &scope, &[], none, observe(&globals));
            (a.await.unwrap(), b.await.unwrap())
        });
        Python::attach(|py| {
            let (current, trace): (Option<(u128, u64, bool)>, u128) =
                under.value.extract(py).unwrap();
            assert_eq!(current, Some((0xabc, 0x42, true)));
            assert_eq!(trace, 0xabc, "the application span continues the call's trace");
            let (current, trace): (Option<(u128, u64, bool)>, u128) =
                outside.value.extract(py).unwrap();
            assert_eq!(current, None, "the activation leaked into the captured context");
            assert_ne!(trace, 0xabc);
        });
    }

    // Catches: a proxy call under a span its proxy did not capture (the room's connection under
    // the session's `session_start`, a hook under its span) run with only the proxy's captured
    // context current, so the spans the host starts there begin their own trace.
    #[test]
    fn a_traced_proxy_call_runs_under_its_span_in_the_proxys_context() {
        use _core::generated::worker::SessionJobProxy;
        crate::py_test_support::install_package(_core::register);
        let (globals, job) = Python::attach(|py| {
            let globals = crate::py_test_support::fixture(py, "telemetry_host");
            let context = _core::py_objects::retain(globals.getattr("pair").unwrap().unbind());
            let job = SessionJobProxy::in_context(&globals.getattr("job").unwrap(), &context);
            (globals.unbind(), job.unwrap())
        });
        let span = SpanContext { trace_id: 0xabc, span_id: 0x43, sampled: true };
        let runtime = lk_agents_runtime::process::handle().unwrap();
        runtime.block_on(async {
            let cx = lk_agents_abi::telemetry::Cx::NONE.continuing(Some(span));
            job.connect(&cx).await.unwrap();
        });
        Python::attach(|py| {
            let parents = globals.bind(py).getattr("job").unwrap().getattr("parents").unwrap();
            let parents: Vec<Option<(u128, u64, bool)>> = parents.extract().unwrap();
            assert_eq!(parents, [Some((0xabc, 0x43, true))]);
        });
    }
}
