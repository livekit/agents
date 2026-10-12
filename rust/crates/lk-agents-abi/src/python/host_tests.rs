//! The host bridge and the value crossings, against a live interpreter (moved from
//! `lk-agents-py`'s `runtime.rs` with the code they test). Stubs stand in for the package, which is
//! not importable here; nextest's process per test keeps each cached host class on them.

mod provider_bridge {
    //! The provider bridge's values as their own impls and the proxies cross them. The real
    //! package must not be imported here, so `livekit.agents.types` and `livekit.rtc` are stubs;
    //! nextest's process per test keeps each type's cached class on these.
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use crate::python::{HostBridge as _, HostHandle, extract_generated, pyo3_async_runtimes};
    use crate::{AudioFrame, LlmChunk, Role, SpeechEvent, VadEvent};

    fn host_ref_table_len() -> usize {
        crate::python::media().len_after_drain()
    }

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
        self.speaker_id = kw.get('speaker_id'); self.is_primary_speaker = kw.get('is_primary_speaker')
        self.words = kw.get('words'); self.source_languages = kw.get('source_languages')
        self.source_texts = kw.get('source_texts'); self.target_languages = kw.get('target_languages')
        self.target_texts = kw.get('target_texts'); self.metadata = kw.get('metadata')

class SpeechEvent:
    def __init__(self, type, request_id='r', alternatives=(), recognition_usage=None,
                 speech_start_time=None):
        self.type = type; self.request_id = request_id; self.alternatives = list(alternatives)
        self.recognition_usage = recognition_usage; self.speech_start_time = speech_start_time

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

class FunctionToolCall:
    def __init__(self, name, arguments, call_id, extra=None):
        self.name = name; self.arguments = arguments; self.call_id = call_id; self.extra = extra

class ChoiceDelta:
    def __init__(self, role=None, content=None, tool_calls=(), extra=None):
        self.role = role; self.content = content
        self.tool_calls = list(tool_calls); self.extra = extra

class CompletionUsage:
    def __init__(self, **kw):
        self.completion_tokens = kw.get('completion_tokens', 0)
        self.prompt_tokens = kw.get('prompt_tokens', 0)
        self.prompt_cached_tokens = kw.get('prompt_cached_tokens', 0)
        self.cache_creation_tokens = kw.get('cache_creation_tokens', 0)
        self.cache_read_tokens = kw.get('cache_read_tokens', 0)
        self.total_tokens = kw.get('total_tokens', 0)
        self.service_tier = kw.get('service_tier')

class ChatChunk:
    def __init__(self, id='c', delta=None, usage=None):
        self.id = id; self.delta = delta; self.usage = usage
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

    // Catches: the generated EOT proxy calling its host off the loop thread (a silent stall), out
    // of order, or under any name but the `_core_*` surface's.
    #[test]
    fn the_eot_proxy_calls_its_host_on_the_loop_thread_in_order() {
        use crate::EotDetector as _;
        attach(|py| {
            let out = eval_with_stubs(
                py,
                r#"
import asyncio, threading
loop = asyncio.new_event_loop()
ready = threading.Event()
def run():
    asyncio.set_event_loop(loop); loop.call_soon(ready.set); loop.run_forever()
thread = threading.Thread(target=run, daemon=True); thread.start(); ready.wait()
class Detector:
    def __init__(self): self.calls = []; self.done = threading.Event()
    def _core_id(self): return {"label": "t", "model": "m", "provider": "p"}
    def _core_push_audio(self, f): self.calls.append(("push", f.sample_rate, threading.get_ident()))
    def _core_flush(self, reason): self.calls.append(("flush", reason, threading.get_ident()))
    def _core_cancel(self, timed_out):
        self.calls.append(("cancel", timed_out, threading.get_ident())); self.done.set()
out = (loop, thread.ident, Detector())
"#,
            );
            let (event_loop, loop_thread, detector): (Py<PyAny>, u64, Py<PyAny>) = out.extract(py).unwrap();
            let handle = HostHandle::from_parts(
                detector.clone_ref(py),
                pyo3_async_runtimes::TaskLocals::new(event_loop.bind(py).clone()),
            );
            let proxy = crate::turn::detector::PyEotDetector::from_bridge(handle).unwrap();
            assert_eq!(proxy.id().model, "m", "the identity is read once, off `_core_id`");
            proxy.push_audio(AudioFrame::new(vec![0u8; 4], 16_000, 1, 2));
            proxy.flush("end_of_speech");
            proxy.cancel(true);
            // `Event.wait` releases the GIL, so the loop thread can run the three callbacks.
            let detector = detector.bind(py);
            let done = detector.getattr("done").unwrap().call_method1("wait", (5.0,)).unwrap();
            assert!(done.extract::<bool>().unwrap(), "the calls never reached the loop");
            let calls: Vec<Bound<'_, PyAny>> = detector.getattr("calls").unwrap().extract().unwrap();
            let got: Vec<String> = calls.iter().map(|c| c.repr().unwrap().to_string()).collect();
            let want = [
                format!("('push', 16000, {loop_thread})"),
                format!("('flush', 'end_of_speech', {loop_thread})"),
                format!("('cancel', True, {loop_thread})"),
            ];
            assert_eq!(got, want);
            let stop = event_loop.bind(py).getattr("stop").unwrap();
            event_loop.bind(py).call_method1("call_soon_threadsafe", (stop,)).unwrap();
        });
    }

    // Catches: a duck-typed plugin (the reference API's shape, no base class) refused by a bare
    // proxy instead of seen through the host class's own `_core_*` functions.
    #[test]
    fn a_duck_typed_host_object_is_seen_through_the_host_class() {
        attach(|py| {
            let out = eval_with_stubs(
                py,
                r#"
import asyncio, functools
view = _mod('livekit.agents._core_view')
class _Duck:
    def __init__(self, obj, base): self._obj = obj; self._base = base
    def __getattr__(self, name):
        if name.startswith('_core_'): return functools.partial(getattr(self._base, name), self._obj)
        return getattr(self._obj, name)
view.duck = _Duck
host = _mod('host')
class Base:
    def _core_id(self): return {"label": self.label, "model": "unknown", "provider": "unknown"}
host.Base = Base
class Plain:
    label = "plain"
out = (asyncio.new_event_loop(), Plain())
"#,
            );
            let (event_loop, plain): (Py<PyAny>, Py<PyAny>) = out.extract(py).unwrap();
            let handle =
                HostHandle::from_parts(plain, pyo3_async_runtimes::TaskLocals::new(event_loop.bind(py).clone()));
            let seen = handle.core_surface("host.Base", "_core_id").unwrap();
            let id = seen.object().bind(py).call_method0("_core_id").unwrap();
            assert_eq!(id.get_item("label").unwrap().extract::<String>().unwrap(), "plain");
        });
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
            assert_eq!((words[0].text.as_str(), words[0].start_time, words[0].end_time), ("w", Some(0.25), None));
            assert_eq!(alt.metadata.as_ref().expect("plugin metadata must survive")["voice"], "a");

            let bare = speech_event(py, "out = SpeechEvent('final_transcript', alternatives=[SpeechData()])");
            assert!(bare.recognition_usage.is_none());
            assert!(bare.alternatives[0].words.is_none());
            assert!(bare.alternatives[0].metadata.is_none());
        });
    }

    // Catches: interning an empty `VADEvent.frames` list: ~30 events a second per stream, each a
    // side-table entry, on the hottest path.
    #[test]
    fn an_empty_vad_frame_list_never_reaches_the_side_table() {
        attach(|py| {
            let before = host_ref_table_len();
            let obj = eval_with_stubs(py, "out = VADEvent('inference_done', probability=0.9)");
            let ev: VadEvent = extract_generated::<VadEvent>(obj.bind(py)).unwrap();
            assert!(ev.frames.is_none());
            assert_eq!(host_ref_table_len(), before, "no entry for []");

            let obj = eval_with_stubs(py, "out = VADEvent('end_of_speech', frames=[frame(), frame()])");
            let ev: VadEvent = extract_generated::<VadEvent>(obj.bind(py)).unwrap();
            assert!(ev.frames.is_some(), "real audio must be interned");
            assert_eq!(host_ref_table_len(), before + 1, "one entry for the whole list, not one per frame");
        });
    }

    // Catches: failing on `ChoiceDelta.role = None` (every chunk after the first) or on a role the
    // reference reads as absent, losing `tool_calls[].extra` (Google's thought signatures), or
    // inventing a delta for the final usage-only chunk of an OpenAI stream.
    #[test]
    fn an_llm_chunk_carries_its_tool_calls_role_and_usage() {
        attach(|py| {
            let chunk = |code: &str| -> LlmChunk {
                let obj = eval_with_stubs(py, code);
                extract_generated::<LlmChunk>(obj.bind(py)).unwrap()
            };
            let full = chunk(
                "out = ChatChunk(id='x', \
                 delta=ChoiceDelta(role='assistant', content='', tool_calls=[\
                     FunctionToolCall('f', '{\"a\":1}', 'call-1', extra={'sig': 'abc'})]), \
                 usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8))",
            );
            let delta = full.delta.as_ref().unwrap();
            assert_eq!(delta.role, Some(Role::Assistant));
            assert_eq!(delta.tool_calls[0].call_id, "call-1");
            assert_eq!(delta.tool_calls[0].extra.as_ref().unwrap()["sig"], "abc");
            assert_eq!(full.usage.as_ref().unwrap().total_tokens, 8);
            assert!(full.has_response(), "a tool call is a response even with empty content");

            let role_less = chunk("out = ChatChunk(delta=ChoiceDelta(content=''))");
            assert_eq!(role_less.delta.as_ref().unwrap().role, None, "role is absent after the first chunk");
            let odd = chunk("out = ChatChunk(delta=ChoiceDelta(role='narrator', content='hi'))");
            assert_eq!(odd.delta.as_ref().unwrap().role, None, "an unknown role reads as absent");

            let usage_only = chunk("out = ChatChunk(delta=None, usage=CompletionUsage(total_tokens=11))");
            assert!(usage_only.delta.is_none(), "there is no delta to invent");
            assert_eq!(usage_only.usage.unwrap().total_tokens, 11);
        });
    }

    // Catches: an audio frame losing its shape across the audio-in/TTS-out legs, or handed to the
    // host as the dict rather than its `rtc.AudioFrame`.
    #[test]
    fn an_audio_frame_round_trips_through_the_host_class() {
        attach(|py| {
            let _ = eval_with_stubs(py, "out = None");
            let original = AudioFrame::new(vec![1u8, 0, 2, 0, 3, 0], 48_000, 1, 3);
            let host = (&original).into_pyobject(py).unwrap();
            assert_eq!(host.get_type().name().unwrap().to_string(), "AudioFrame");
            let back: AudioFrame = extract_generated::<AudioFrame>(&host).unwrap();
            assert_eq!(back, original);
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
                    "confidence", "end_time", "is_primary_speaker", "language", "metadata",
                    "source_languages", "source_texts", "speaker_id", "start_time",
                    "target_languages", "target_texts", "text", "words",
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
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use crate::python::{capture, capture_value, errors, raise_original, resolve_captured};

    // ------------------------------------------------------------------------ the side tables


    /// A stand-in for a plugin's `APIStatusError` subclass. The package is not importable here, so
    /// these pin the side table; the isinstance half is pinned in `tests/test_exceptions.py`.
    fn make_exception(py: Python<'_>, code: &str) -> Py<PyAny> {
        let locals = pyo3::types::PyDict::new(py);
        py.run(std::ffi::CString::new(code).unwrap().as_c_str(), None, Some(&locals)).unwrap();
        locals.get_item("exc").unwrap().unwrap().unbind()
    }

    const PLUGIN_ERROR: &str = "\
class MyStatusError(Exception):
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
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use serde_json::{Map, Value};
    use crate::python::{extract_generated, value_to_py};

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    /// `program`'s `out`, evaluated.
    fn out(py: Python<'_>, program: &str) -> Py<PyAny> {
        let globals = PyDict::new(py);
        py.run(std::ffi::CString::new(format!("{program}\n")).unwrap().as_c_str(), Some(&globals), None).unwrap();
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
            let [crate::ToolSpec { name, description, json_schema, flags }] = <[_; 1]>::try_from(specs).unwrap();
            assert_eq!((name.as_str(), description.as_str()), ("lookup", "look something up"));
            assert_eq!(json_schema["type"], "object");
            assert!(flags.ignore_on_enter, "the flag was read as the default rather than the value");
        });
    }

    // Catches: a `None` json_schema answered with `Null` (providers reject it), a `None`
    // description refused, or an omitted key read as a default rather than refused.
    #[test]
    fn none_reads_as_empty_and_an_omitted_key_raises() {
        attach(|py| {
            let none = parse_specs(py, "out[0]['json_schema'] = None").unwrap();
            assert_eq!(none[0].json_schema, Value::Object(Map::new()));
            assert_eq!(parse_specs(py, "out[0]['description'] = None").unwrap()[0].description, "");
            assert!(parse_specs(py, "del out[0]['json_schema']").is_err());
            assert!(parse_specs(py, "del out[0]['ignore_on_enter']").is_err());
        });
    }

    fn answer(py: Python<'_>, program: &str) -> PyResult<crate::llm::tool_executor::RawToolAnswer> {
        let v = out(py, program);
        extract_generated::<crate::llm::tool_executor::RawToolAnswer>(v.bind(py))
    }

    // Catches: `value` or `agents` read past, `None` refused, or a non-JSON value accepted.
    #[test]
    fn a_tool_answer_carries_its_json_value_and_interns_its_agents() {
        attach(|py| {
            let a = answer(py, "out = {'kind': 'returned', 'value': {'temp': 12, 'ok': True}, 'agents': [object(), object()]}").unwrap();
            assert_eq!(a.value, Some(serde_json::json!({"temp": 12, "ok": true})));
            assert_eq!(a.agents.len(), 2, "both agents cross as refs; two is `decide`'s to refuse");
            let a = answer(py, "out = {'kind': 'returned', 'value': None, 'agents': None}").unwrap();
            assert_eq!((a.value, a.agents.len()), (None, 0), "`None` means the same as absent");
            let a = answer(py, "out = {'kind': 'returned'}").unwrap();
            assert_eq!((a.value, a.agents.len()), (None, 0), "absent optional keys are absent, not malformed");
            assert!(answer(py, "out = {'kind': 'returned', 'value': object()}").is_err(), "a value that is not JSON");
            assert!(answer(py, "out = 'sunny'").is_err(), "not a dict at all");
        });
    }

    /// The keys of a built kwargs dict, sorted and compared whole, so an extra key fails too.
    fn keys(d: &Bound<'_, PyDict>) -> Vec<String> {
        let mut out: Vec<String> = d.keys().iter().map(|k| k.extract::<String>().expect("str keys")).collect();
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

    // --- the realtime seam: commands out, flat events in (`RealtimeCommand`, `Event`) --------

    use crate::llm::realtime::{self as rt, Event, RealtimeCommand};

    /// Just enough of `livekit.rtc` for a pushed frame's `rtc.AudioFrame`.
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

    fn command<'py>(py: Python<'py>, cmd: RealtimeCommand) -> Bound<'py, PyDict> {
        (&cmd).into_pyobject(py).unwrap().into_any().cast_into().unwrap()
    }

    fn chat_item() -> crate::ChatItem {
        crate::ChatItem::Message(crate::ChatMessage::text(
            "i1",
            crate::Role::User,
            "hi",
            std::time::SystemTime::UNIX_EPOCH,
        ))
    }

    // Catches: a command dict that is a key short, a key long, or carrying the wrong `kind`.
    #[test]
    fn every_realtime_command_carries_its_kind_and_exactly_its_own_payload() {
        attach_rtc(|py| {
            let every = vec![
                RealtimeCommand::UpdateInstructions(rt::UpdateInstructions { instructions: "be brief".into() }),
                RealtimeCommand::UpdateChatCtx(rt::UpdateChatCtx {
                    req_id: "r1".into(),
                    chat_ctx: crate::ChatContext::new(vec![chat_item()]),
                }),
                RealtimeCommand::UpdateTools(rt::UpdateTools { tools: Vec::new() }),
                RealtimeCommand::PushAudio(rt::PushAudio { frame: crate::AudioFrame::new(vec![0u8; 4], 24_000, 1, 2) }),
                RealtimeCommand::PushItem(rt::PushItem { item: Box::new(chat_item()), reply_required: true }),
                RealtimeCommand::GenerateReply(rt::GenerateReply { req_id: "r2".into(), instructions: None, tool_choice: None }),
                RealtimeCommand::Say(rt::Say { req_id: "r3".into(), text: "hello".into(), add_to_chat_ctx: true }),
                RealtimeCommand::CancelReply(rt::CancelReply { req_id: "r4".into() }),
                RealtimeCommand::CommitAudio(rt::CommitAudio {}),
                RealtimeCommand::ClearAudio(rt::ClearAudio {}),
                RealtimeCommand::Interrupt(rt::Interrupt { response_id: Some("resp_1".into()) }),
                RealtimeCommand::Truncate(rt::Truncate { message_id: "m1".into(), has_audio: true, audio_end_ms: 120, audio_transcript: None }),
            ];
            let shapes: Vec<(String, Vec<String>)> = every
                .into_iter()
                .map(|c| {
                    let d = command(py, c);
                    (d.get_item("kind").unwrap().unwrap().extract().unwrap(), keys(&d))
                })
                .collect();
            let k = |s: &str| s.to_owned();
            assert_eq!(
                shapes,
                vec![
                    (k("update_instructions"), vec![k("instructions"), k("kind")]),
                    (k("update_chat_ctx"), vec![k("items"), k("kind"), k("req_id")]),
                    (k("update_tools"), vec![k("kind"), k("names")]),
                    (k("push_audio"), vec![k("frame"), k("kind")]),
                    (k("push_item"), vec![k("item"), k("kind"), k("reply_required")]),
                    (k("generate_reply"), vec![k("instructions"), k("kind"), k("req_id"), k("tool_choice")]),
                    (k("say"), vec![k("add_to_chat_ctx"), k("kind"), k("req_id"), k("text")]),
                    (k("cancel_reply"), vec![k("kind"), k("req_id")]),
                    (k("commit_audio"), vec![k("kind")]),
                    (k("clear_audio"), vec![k("kind")]),
                    (k("interrupt"), vec![k("kind"), k("response_id")]),
                    (k("truncate"), vec![k("audio_end_ms"), k("audio_transcript"), k("has_audio"), k("kind"), k("message_id")]),
                ]
            );
        });
    }

    // Catches: `tool_choice` in a shape `generate_reply(tool_choice=)` does not take: `Forbidden`
    // must be the string `"none"` (Python `None` means absent), `Named` the two-level dict.
    #[test]
    fn generate_reply_carries_its_tool_choice_in_the_hosts_shape() {
        attach_rtc(|py| {
            let choice = |tool_choice| {
                let reply = rt::GenerateReply { req_id: "r".into(), instructions: None, tool_choice };
                let d = command(py, RealtimeCommand::GenerateReply(reply));
                d.get_item("tool_choice").unwrap().unwrap().repr().unwrap().to_string()
            };
            use crate::ToolChoice;
            assert_eq!(choice(None), "None", "absent is `None`, present as a key");
            assert_eq!(choice(Some(ToolChoice::Auto)), "'auto'");
            assert_eq!(choice(Some(ToolChoice::Required)), "'required'");
            assert_eq!(choice(Some(ToolChoice::Forbidden)), "'none'", "not Python None");
            assert_eq!(choice(Some(ToolChoice::Named("lookup".into()))), "{'type': 'function', 'function': {'name': 'lookup'}}");
        });
    }

    /// A Python dict literal, evaluated.
    fn dict<'py>(py: Python<'py>, literal: &str) -> Bound<'py, PyDict> {
        let code = std::ffi::CString::new(literal).unwrap();
        py.eval(code.as_c_str(), None, None).unwrap().cast_into().unwrap()
    }

    fn host_event(d: &Bound<'_, PyDict>) -> PyResult<Event> {
        extract_generated::<Event>(d.as_any())
    }

    fn ev(d: &Bound<'_, PyDict>) -> Event {
        host_event(d).unwrap().anchored(None)
    }

    fn refused(d: &Bound<'_, PyDict>) -> String {
        host_event(d).expect_err("must raise").to_string()
    }

    // Catches: `error` being dropped, or its optional correlation keys becoming required. With no
    // anchor the timestamp lands on `Ts::ZERO`; the stamp itself is required.
    #[test]
    fn an_error_is_forwarded_with_its_optional_correlation() {
        attach(|py| {
            let d = dict(py, "{'kind': 'error', 'timestamp': 1.0, 'label': 'fake', 'message': 'boom', 'recoverable': True}");
            let Event::Error(e) = ev(&d) else { panic!() };
            assert_eq!((e.label.as_str(), e.message.as_str(), e.recoverable), ("fake", "boom", true));
            assert_eq!((e.req_id, e.code, e.timestamp), (None, None, crate::Ts::ZERO));
            d.set_item("req_id", "req_7").unwrap();
            d.set_item("code", "busy").unwrap();
            let Event::Error(e) = ev(&d) else { panic!() };
            assert_eq!((e.req_id.as_deref(), e.code.as_deref()), (Some("req_7"), Some("busy")));
            d.del_item("timestamp").unwrap();
            assert!(refused(&d).contains("timestamp"));
        });
    }

    // Catches: the server-created item losing its position; absent and `None` `previous_item_id`
    // must both mean "appended"; a non-`ChatItem` refused.
    #[test]
    fn a_remote_item_is_read_with_its_position_or_appended_without_one() {
        attach(|py| {
            let d = dict(py, "{'kind': 'remote_item_added', 'previous_item_id': 'i0'}");
            d.set_item("item", chat_item()).unwrap();
            let previous = |d: &Bound<'_, PyDict>| match ev(d) {
                Event::RemoteItemAdded(rt::RemoteItemAdded { previous_item_id, item: got }) => {
                    assert_eq!(*got, chat_item());
                    previous_item_id
                }
                other => panic!("{other:?}"),
            };
            assert_eq!(previous(&d).as_deref(), Some("i0"));
            d.set_item("previous_item_id", py.None()).unwrap();
            assert_eq!(previous(&d), None, "`None` is appended");
            d.del_item("previous_item_id").unwrap();
            assert_eq!(previous(&d), None, "absent is appended too");
            d.set_item("item", "a string").unwrap();
            assert!(refused(&d).contains("item"));
        });
    }

    // Catches: a `default` key becoming required, or a required one optional or defaulted (to
    // `false`, which turns user transcription off) instead of raising; an unknown kind read as
    // something; and the flattened error read from a sub-dict instead of keys beside `req_id`.
    #[test]
    fn an_optional_key_may_be_absent_and_a_required_one_may_not() {
        attach(|py| {
            let Event::GenerationCreated(v) = ev(&dict(py, "{'kind': 'generation_created', 'response_id': 'r'}")) else { panic!() };
            assert_eq!(v.req_id, None, "an absent `req_id` is a server-initiated turn");
            assert!(refused(&dict(py, "{'kind': 'generation_created'}")).contains("response_id"));
            let msg = refused(&dict(py, "{'kind': 'input_speech_stopped'}"));
            assert!(msg.contains("user_transcription_enabled"), "the error must name the key: {msg}");
            assert!(refused(&dict(py, "{'kind': 'not_an_event'}")).contains("not_an_event"));
            assert_eq!(ev(&dict(py, "{'kind': 'input_speech_started'}")), Event::InputSpeechStarted(rt::InputSpeechStarted {}));
            assert_eq!(ev(&dict(py, "{'kind': 'session_reconnected'}")), Event::SessionReconnected(rt::SessionReconnected {}));

            let d = dict(py, "{'kind': 'generate_reply_failed', 'req_id': 'r9', 'message': 'busy', 'code': 'active'}");
            let Event::GenerateReplyFailed(v) = ev(&d) else { panic!() };
            assert_eq!((v.req_id.as_str(), v.error.message.as_str(), v.error.code.as_deref()), ("r9", "busy", Some("active")));
            d.del_item("code").unwrap();
            let Event::GenerateReplyFailed(v) = ev(&d) else { panic!() };
            assert_eq!(v.error.code, None, "most non-OpenAI errors carry no code");
        });
    }

    // Catches: modalities read as anything but "is audio present" (it decides whether the driver
    // waits for playout), or an unknown status word failing the turn instead of completing it.
    #[test]
    fn modalities_and_status_words_are_read_as_the_reference_does() {
        use crate::llm::realtime::{GenerationStatus, Modalities};
        attach(|py| {
            for (word, want) in [("text", Modalities::TextOnly), ("audio", Modalities::AudioAndText)] {
                let d = dict(py, &format!("{{'kind': 'message_modalities', 'message_id': 'm1', 'modalities': '{word}'}}"));
                let Event::MessageModalities(v) = ev(&d) else { panic!() };
                assert_eq!(v.modalities, want, "{word}");
            }
            for (word, want) in [("something_new", GenerationStatus::Completed), ("cancelled", GenerationStatus::Cancelled)] {
                let d = dict(py, &format!("{{'kind': 'generation_done', 'response_id': 'r', 'status': '{word}'}}"));
                let Event::GenerationDone(v) = ev(&d) else { panic!() };
                assert_eq!(v.status, want, "{word}");
            }
            let Event::GenerationDone(v) = ev(&dict(py, "{'kind': 'generation_done', 'response_id': 'r'}")) else { panic!() };
            assert_eq!(v.status, GenerationStatus::Completed, "an absent status completes");
        });
    }

    // Catches: `turn_started_at` converted, clamped or dropped on the way in: it is the host's wall
    // seconds, used as given (a provider may report a turn that began before the session, which
    // the message is placed at); absent or `None` is no turn start; a missing required `is_final`
    // defaulting instead of raising.
    #[test]
    fn an_input_transcription_keeps_its_turn_start_as_the_hosts_wall_seconds() {
        attach(|py| {
            let d = dict(py, "{'kind': 'input_transcription', 'item_id': 'item_9', 'transcript': 'hello there', 'is_final': True}");
            d.set_item("turn_started_at", 1_000_000.5).unwrap();
            let read = || match host_event(&d).unwrap().anchored(None) {
                Event::HostInputTranscription(v) => v,
                other => panic!("{other:?}"),
            };
            let v = read();
            let t = &v.transcription;
            assert_eq!((t.item_id.as_str(), t.transcript.as_str(), t.is_final), ("item_9", "hello there", true));
            assert_eq!(t.confidence, None, "an absent confidence is absent");
            assert_eq!(v.turn_started_at, Some(1_000_000.5));
            d.set_item("turn_started_at", py.None()).unwrap();
            assert_eq!(read().turn_started_at, None);
            d.del_item("turn_started_at").unwrap();
            assert_eq!(read().turn_started_at, None);
            d.del_item("is_final").unwrap();
            assert!(refused(&d).contains("is_final"));
        });
    }

    // Catches: a provider payload's tagged bytes reaching the host as a dict, or a user dict with
    // more than the one key mistaken for bytes.
    #[test]
    fn a_tagged_bytes_leaf_reads_back_as_bytes_and_nothing_else_does() {
        attach(|py| {
            let tag = crate::llm::provider_format::BYTES_TAG;
            let v = value_to_py(py, &serde_json::json!({"img": {tag: "AQID"}, "other": {tag: "AQID", "x": 1}})).unwrap();
            let img = v.get_item("img").unwrap();
            assert_eq!(img.cast::<pyo3::types::PyBytes>().unwrap().as_bytes(), &[1, 2, 3]);
            assert!(v.get_item("other").unwrap().is_instance_of::<PyDict>());
        });
    }
}
