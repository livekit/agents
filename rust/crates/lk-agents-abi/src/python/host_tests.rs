//! The host bridge and the value crossings, against a live interpreter (moved from
//! `lk-agents-py`'s `runtime.rs` with the code they test). Stubs stand in for the package, which is
//! not importable here; nextest's process per test keeps each cached host class on them.

mod provider_bridge {
    //! The provider bridge's values as their own impls and the proxies cross them. The real
    //! package must not be imported here, so `livekit.agents.types` and `livekit.rtc` are stubs;
    //! nextest's process per test keeps each type's cached class on these.
    use crate::python::extract_generated;
    use crate::{SpeechEvent, VadEvent};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    const STUBS: &str = r#"
import sys, types as _t

def _mod(name):
    m = _t.ModuleType(name); m.__path__ = []; sys.modules[name] = m; return m

lk = _mod('livekit')
ag = _mod('livekit.agents'); lk.agents = ag
tp = _mod('livekit.agents.types'); ag.types = tp
rtc = _mod('livekit.rtc'); lk.rtc = rtc

class NotGiven:
    def __bool__(self): return False
NOT_GIVEN = NotGiven()

class TimedString(str):
    def __new__(cls, text, start_time=NOT_GIVEN, end_time=NOT_GIVEN, confidence=NOT_GIVEN,
                start_time_offset=NOT_GIVEN, speaker_id=None):
        o = super().__new__(cls, text)
        o.start_time = start_time; o.end_time = end_time; o.confidence = confidence
        o.start_time_offset = start_time_offset; o.speaker_id = speaker_id
        return o
tp.NotGiven = NotGiven; tp.NOT_GIVEN = NOT_GIVEN; tp.TimedString = TimedString

class AudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self._d = bytes(data); self.sample_rate = sample_rate
        self.num_channels = num_channels; self.samples_per_channel = samples_per_channel
    @property
    def data(self):
        return memoryview(self._d)
rtc.AudioFrame = AudioFrame

def frame(n=4):
    return AudioFrame(b'\x00' * (n * 2), 24000, 1, n)

class SpeechData:
    def __init__(self, **kw):
        self.language = kw.get('language', 'en-US'); self.text = kw.get('text', '')
        self.start_time = kw.get('start_time', 0.0); self.end_time = kw.get('end_time', 0.0)
        self.confidence = kw.get('confidence', 0.0)
        self.speaker_id = kw.get('speaker_id')
        self.words = kw.get('words'); self.source_languages = kw.get('source_languages')
        self.source_texts = kw.get('source_texts'); self.target_languages = kw.get('target_languages')
        self.target_texts = kw.get('target_texts'); self.metadata = kw.get('metadata')

class SpeechEvent:
    def __init__(self, type, request_id='r', alternatives=(), recognition_usage=None,
                 speech_start_time=None, created_at=None, speech_end_time=None):
        self.type = type; self.request_id = request_id; self.alternatives = list(alternatives)
        self.recognition_usage = recognition_usage; self.speech_start_time = speech_start_time
        self.created_at = created_at; self.speech_end_time = speech_end_time

class VADEvent:
    def __init__(self, type='inference_done', frames=None, **kw):
        self.type = type; self.frames = [] if frames is None else frames
        self.samples_index = kw.get('samples_index', 0); self.timestamp = kw.get('timestamp', 0.0)
        self.speech_duration = kw.get('speech_duration', 0.0)
        self.silence_duration = kw.get('silence_duration', 0.0)
        self.probability = kw.get('probability', 0.0)
        self.inference_duration = kw.get('inference_duration', 0.0)
        self.speaking = kw.get('speaking', False)
        self.raw_accumulated_silence = kw.get('raw_accumulated_silence', 0.0)
        self.raw_accumulated_speech = kw.get('raw_accumulated_speech', 0.0)
"#;

    /// Run `code` after the stubs, and return the object it bound to `out`.
    fn eval_with_stubs(py: Python<'_>, code: &str) -> Py<PyAny> {
        let globals = PyDict::new(py);
        let program = format!("{STUBS}\n{code}\n");
        py.run(std::ffi::CString::new(program).unwrap().as_c_str(), Some(&globals), None).unwrap();
        globals.get_item("out").unwrap().unwrap().unbind()
    }

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    fn speech_event(py: Python<'_>, code: &str) -> SpeechEvent {
        let obj = eval_with_stubs(py, code);
        extract_generated::<SpeechEvent>(obj.bind(py)).unwrap()
    }

    // Catches: a panic on an unknown `SpeechEventType` from a plugin, instead of an error that
    // names the value (§1.6 #6).
    #[test]
    fn an_unknown_speech_event_type_is_an_error_not_a_panic() {
        attach(|py| {
            let obj = eval_with_stubs(py, "out = SpeechEvent('teleport_transcript')");
            let err = extract_generated::<SpeechEvent>(obj.bind(py)).err().unwrap();
            assert!(err.to_string().contains("teleport_transcript"), "{err}");
        });
    }

    // Catches: collapsing `None` and `[]` on the translation lists, losing `language`
    // normalisation (`stt/stt.py:87-102`), or the `is_none()` guards on optional sub-objects
    // (`recognition_usage`, `words`, `metadata`) inverting: real data dropped, or `AttributeError`.
    #[test]
    fn speech_data_reads_optional_fields_without_collapsing_or_inventing_them() {
        attach(|py| {
            let ev = speech_event(
                py,
                r#"
class Usage:
    audio_duration = 1.5; input_tokens = 7; output_tokens = 9
out = SpeechEvent('recognition_usage', recognition_usage=Usage(), alternatives=[
    SpeechData(language='en_us', source_languages=[], target_texts=None,
               words=[TimedString('w', start_time=0.25)], metadata={'voice': 'a'})])
"#,
            );
            let usage = ev.recognition_usage.as_ref().expect("a usage event must carry its usage");
            assert_eq!((usage.audio_duration, usage.input_tokens), (1.5, 7));
            let alt = &ev.alternatives[0];
            assert_eq!(alt.language.as_str(), "en-US", "the underscore form must normalise");
            assert_eq!(alt.source_languages.as_deref(), Some(&[][..]), "[] is not None");
            assert_eq!(alt.target_texts, None, "None is not []");
            let words = alt.words.as_ref().expect("word alignment must survive");
            assert_eq!(
                (words[0].text.as_str(), words[0].start_time, words[0].end_time),
                ("w", Some(0.25), None)
            );
            assert_eq!(alt.metadata.as_ref().expect("plugin metadata must survive")["voice"], "a");

            let bare = speech_event(
                py,
                "out = SpeechEvent('final_transcript', alternatives=[SpeechData()])",
            );
            assert!(bare.recognition_usage.is_none());
            assert!(bare.alternatives[0].words.is_none());
            assert!(bare.alternatives[0].metadata.is_none());
        });
    }

    // Catches: copying the audio of every VAD event off the host (`inference_done` fires ~30 a
    // second per stream) when only an end of speech's is recognized.
    #[test]
    fn only_an_end_of_speech_carries_its_audio_off_the_host() {
        attach(|py| {
            let obj = eval_with_stubs(
                py,
                "out = VADEvent('inference_done', probability=0.9, frames=[frame()])",
            );
            let ev: VadEvent = extract_generated::<VadEvent>(obj.bind(py)).unwrap();
            assert!(ev.frames.is_empty());

            let obj =
                eval_with_stubs(py, "out = VADEvent('end_of_speech', frames=[frame(), frame()])");
            let ev: VadEvent = extract_generated::<VadEvent>(obj.bind(py)).unwrap();
            assert_eq!(ev.frames.len(), 2);
        });
    }

    // Catches: the emitted dict keyed differently from the dataclass it feeds -- the Rust `kind`
    // instead of the host's `type`, the variant name instead of the literal, a translating STT's
    // `target_*`/`source_*` lists lost crossing back behind an adapter, or words handed back as
    // anything but the host's `TimedString`s.
    #[test]
    fn the_emitted_speech_event_has_the_dataclass_shape() {
        attach(|py| {
            let ev = speech_event(
                py,
                "out = SpeechEvent('interim_transcript', alternatives=[SpeechData(\
                 language='fr', text='bonjour', source_languages=['en'], source_texts=['hello'], \
                 target_languages=['fr'], target_texts=['bonjour'], words=[TimedString('bonjour', start_time=0.5)])])",
            );
            let dict = (&ev).into_pyobject(py).unwrap();
            assert!(dict.get_item("kind").unwrap().is_none(), "`kind` is the Rust spelling");
            let tag = dict.get_item("type").unwrap().expect("the host reads d[\"type\"]");
            assert_eq!(tag.extract::<String>().unwrap(), "interim_transcript");

            let alts = dict.get_item("alternatives").unwrap().unwrap();
            let alt = alts.get_item(0).unwrap().cast_into::<PyDict>().unwrap();
            let mut keys: Vec<String> = alt.keys().extract().unwrap();
            keys.sort();
            assert_eq!(
                keys,
                [
                    "confidence",
                    "end_time",
                    "language",
                    "metadata",
                    "source_languages",
                    "source_texts",
                    "speaker_id",
                    "start_time",
                    "target_languages",
                    "target_texts",
                    "text",
                    "words",
                ]
            );
            let got = |k: &str| alt.get_item(k).unwrap().unwrap().extract::<Vec<String>>().unwrap();
            assert_eq!(got("target_texts"), ["bonjour"]);
            assert_eq!(got("target_languages"), ["fr"]);
            assert_eq!(got("source_texts"), ["hello"]);
            assert_eq!(got("source_languages"), ["en"]);
            let words = alt.get_item("words").unwrap().unwrap();
            let word = words.get_item(0).unwrap();
            assert_eq!(word.get_type().name().unwrap().to_string(), "TimedString");
            assert_eq!(word.getattr("start_time").unwrap().extract::<f64>().unwrap(), 0.5);
        });
    }
}

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
    use crate::python::{extract_generated, value_to_py};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use serde_json::Value;

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

    /// One tool spec as `agent_activity.py` builds it, with `patch` applied after.
    fn parse_specs(py: Python<'_>, patch: &str) -> PyResult<Vec<crate::ToolSpec>> {
        let list = out(
            py,
            &format!(
                "out = [{{'name': 'lookup', 'description': 'look something up', \
                 'json_schema': {{'type': 'object', 'properties': {{}}}}, 'ignore_on_enter': True}}]\n{patch}"
            ),
        );
        list.bind(py).extract::<Vec<crate::ToolSpec>>()
    }

    // Catches: a key that reaches this reader and stops there. `ignore_on_enter` is asserted `true`
    // because `false` is both the default and what the N184 bug produced.
    #[test]
    fn every_key_of_a_tool_spec_lands_including_the_flag_that_used_to_be_dropped() {
        attach(|py| {
            let specs = parse_specs(py, "").expect("the producers' dict reads");
            let [crate::ToolSpec { name, description, json_schema, flags, provider_tool }] =
                <[_; 1]>::try_from(specs).unwrap();
            assert_eq!((name.as_str(), description.as_str()), ("lookup", "look something up"));
            assert_eq!(json_schema["type"], "object");
            assert!(provider_tool.is_none());
            assert!(
                flags.ignore_on_enter,
                "the flag was read as the default rather than the value"
            );
        });
    }

    // Defaults apply only to omission. Explicit JSON null remains a JSON value; fields whose
    // types are not nullable must refuse it.
    #[test]
    fn tool_defaults_apply_to_omission_and_nonnullable_fields_refuse_null() {
        attach(|py| {
            let none = parse_specs(py, "out[0]['json_schema'] = None").unwrap();
            assert_eq!(none[0].json_schema, Value::Null);
            assert!(parse_specs(py, "out[0]['description'] = None").is_err());
            assert!(parse_specs(py, "out[0]['ignore_on_enter'] = None").is_err());
            assert!(parse_specs(py, "del out[0]['json_schema']").is_err());
            assert!(
                !parse_specs(py, "del out[0]['ignore_on_enter']").unwrap()[0].flags.ignore_on_enter
            );
            assert_eq!(parse_specs(py, "del out[0]['description']").unwrap()[0].description, "");
        });
    }

    /// The keys of a built kwargs dict, sorted and compared whole, so an extra key fails too.
    fn keys(d: &Bound<'_, PyDict>) -> Vec<String> {
        let mut out: Vec<String> =
            d.keys().iter().map(|k| k.extract::<String>().expect("str keys")).collect();
        out.sort();
        out
    }

    // Catches: a renamed or dropped keyword on `rtc.TranscriptionSegment`, especially `final`.
    #[test]
    fn a_transcription_segment_keys_all_six_and_spells_final_plainly() {
        attach(|py| {
            let seg = crate::voice::rtc::TranscriptionSegment {
                id: "i".into(),
                text: "t".into(),
                start_time: 0,
                end_time: 0,
                language: "en".into(),
                is_final: true,
            };
            let d = (&seg).into_pyobject(py).unwrap();
            assert_eq!(keys(&d), ["end_time", "final", "id", "language", "start_time", "text"]);
        });
    }

    /// Just enough of `livekit.rtc` for an `rtc.AudioFrame`.
    const RTC_STUBS: &str = r#"
import sys, types as _t
def _mod(name):
    m = _t.ModuleType(name); m.__path__ = []; sys.modules[name] = m; return m
lk = sys.modules.get('livekit') or _mod('livekit')
rtc = sys.modules.get('livekit.rtc') or _mod('livekit.rtc'); lk.rtc = rtc
if not hasattr(rtc, 'AudioFrame'):
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

    // Catches: `rtc.AudioFrame` keys drifting, or `data` crossing as `list[int]` instead of bytes.
    #[test]
    fn an_audio_frame_keys_four_names_and_carries_its_samples_as_bytes() {
        attach_rtc(|py| {
            let frame = crate::AudioFrame::new(vec![1u8, 2, 3, 4], 24_000, 1, 2);
            let host = (&frame).into_pyobject(py).unwrap();
            assert_eq!(host.getattr("sample_rate").unwrap().extract::<u32>().unwrap(), 24_000);
            assert_eq!(host.getattr("num_channels").unwrap().extract::<u16>().unwrap(), 1);
            assert_eq!(host.getattr("samples_per_channel").unwrap().extract::<u32>().unwrap(), 2);
            assert!(
                host.getattr("data").unwrap().is_instance_of::<pyo3::types::PyBytes>(),
                "a list[int] here is read as a buffer of the wrong width"
            );
        });
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
