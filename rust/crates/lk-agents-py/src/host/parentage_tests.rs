//! Standard OTel child instrumentation through the real HostInvoker context boundary.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::call_in;

const SETUP: &str = r#"
import asyncio, contextvars, threading
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

previous_provider = trace.get_tracer_provider()
provider = TracerProvider(shutdown_on_exit=False)
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracer = trace.get_tracer('host-callback', tracer_provider=provider)
job = contextvars.ContextVar('job', default='outside')
loop = asyncio.new_event_loop()
ready = threading.Event()
cancelled = threading.Event()

def run():
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    loop.run_forever()
thread = threading.Thread(target=run, daemon=True)
thread.start()
ready.wait(timeout=2)

def capture(identity, trace_id, span_id, sampled):
    # The logical parent exists even when span-start observation was dropped or never recorded.
    context = contextvars.copy_context()
    def enter():
        job.set(identity)
        parent = trace.NonRecordingSpan(trace.SpanContext(trace_id, span_id, False,
            trace.TraceFlags(1 if sampled else 0)))
        from opentelemetry import context as otel_context
        otel_context.attach(trace.set_span_in_context(parent))
    context.run(enter)
    return (loop, context)

async def callback(mode):
    with tracer.start_as_current_span('tool-or-node') as span:
        span.set_attribute('job', job.get())
        await asyncio.sleep(0)
        if mode == 'failure':
            raise ValueError('callback failure')
        if mode == 'cancel':
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        job.set('changed in callback')
        observed = (span.get_span_context().trace_id, span.is_recording())
    return observed

def verify():
    spans = exporter.get_finished_spans()
    assert len(spans) == 4, [s.name for s in spans]
    for span in spans:
        expected = {'one': (101, 11), 'two': (202, 22)}[span.attributes['job']]
        assert (span.context.trace_id, span.parent.span_id) == expected
    assert job.get() == 'outside'
    assert not trace.get_current_span().get_span_context().is_valid
    assert trace.get_tracer_provider() is previous_provider

def close():
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    assert not thread.is_alive()
    loop.close()
    provider.shutdown()
"#;

struct HostLoop(Py<PyDict>);

impl Drop for HostLoop {
    fn drop(&mut self) {
        Python::attach(|py| {
            if let Ok(Some(close)) = self.0.bind(py).get_item("close") {
                let _ = close.call0();
            }
        });
    }
}

fn callback(
    globals: Py<PyDict>,
    mode: &'static str,
) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)>
+ Send
+ 'static {
    move |py| {
        Ok((globals.bind(py).get_item("callback")?.expect("defined"), PyTuple::new(py, [mode])?))
    }
}

#[tokio::test]
async fn invocation_trace_parentage_is_isolated_and_restored_after_failure_and_cancellation() {
    Python::initialize();
    let (host, first, second) = Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&std::ffi::CString::new(SETUP).unwrap(), Some(&globals), None).unwrap();
        let capture = globals.get_item("capture").unwrap().unwrap();
        let first = capture.call1(("one", 101u64, 11u64, true)).unwrap();
        let second = capture.call1(("two", 202u64, 22u64, true)).unwrap();
        (
            HostLoop(globals.unbind()),
            crate::python::intern_host(first.unbind()),
            crate::python::intern_host(second.unbind()),
        )
    });
    fn owned(host: &HostLoop) -> Py<PyDict> {
        Python::attach(|py| host.0.clone_ref(py))
    }
    let (one, two) = tokio::join!(
        call_in(&first, callback(owned(&host), "success")),
        call_in(&second, callback(owned(&host), "success")),
    );
    Python::attach(|py| {
        assert_eq!(one.unwrap().extract::<(u64, bool)>(py).unwrap(), (101, true));
        assert_eq!(two.unwrap().extract::<(u64, bool)>(py).unwrap(), (202, true));
    });
    let failure = call_in(&first, callback(owned(&host), "failure")).await.unwrap_err();
    Python::attach(|py| assert!(failure.is_instance_of::<pyo3::exceptions::PyValueError>(py)));
    let cancel = call_in(&second, callback(owned(&host), "cancel"));
    assert!(tokio::time::timeout(std::time::Duration::from_millis(100), cancel).await.is_err());
    let settled = Python::attach(|py| {
        host.0
            .bind(py)
            .get_item("cancelled")
            .unwrap()
            .unwrap()
            .call_method1("wait", (2,))
            .unwrap()
            .extract::<bool>()
            .unwrap()
    });
    assert!(settled, "HostInvoker must settle the cancelled callback");
    let unrecorded = Python::attach(|py| {
        let pair = host
            .0
            .bind(py)
            .get_item("capture")
            .unwrap()
            .unwrap()
            .call1(("unrecorded", 303u64, 33u64, false))
            .unwrap();
        crate::python::intern_host(pair.unbind())
    });
    let result = call_in(&unrecorded, callback(owned(&host), "success")).await.unwrap();
    Python::attach(|py| assert_eq!(result.extract::<(u64, bool)>(py).unwrap(), (303, false)));
    Python::attach(|py| {
        host.0.bind(py).get_item("verify").unwrap().unwrap().call0().unwrap();
        for captured in [&first, &second] {
            let pair = crate::python::resolve_key(py, captured.id()).unwrap();
            let context = pair.bind(py).get_item(1).unwrap();
            let var = host.0.bind(py).get_item("job").unwrap().unwrap();
            let value: String = context.call_method1("get", (var,)).unwrap().extract().unwrap();
            assert!(value == "one" || value == "two", "callback changed the captured context");
        }
    });
}
