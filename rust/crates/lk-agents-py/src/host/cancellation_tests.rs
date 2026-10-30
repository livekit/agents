//! Cancellation before the host loop starts a call and while it publishes the host task.

use std::sync::Arc;
use lk_agents_runtime::net::lock::Mutex;
use std::task::{Context, Poll, Waker};

use lk_agents_abi::invoke::HostEntry;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

struct ManualLoop(Py<PyDict>);

impl ManualLoop {
    fn new() -> Self {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                c"import asyncio, contextvars\nloop = asyncio.new_event_loop()\ncontext = contextvars.copy_context()\nentered = []\ndef callback():\n    entered.append('entered')\n",
                Some(&globals),
                None,
            )
            .unwrap();
            Self(globals.unbind())
        })
    }

    fn tick(&self) {
        Python::attach(|py| {
            let globals = self.0.bind(py);
            py.run(c"loop.call_soon(loop.stop)\nloop.run_forever()", Some(globals), None).unwrap();
        });
    }

    fn entered(&self) -> bool {
        Python::attach(|py| {
            self.0.bind(py).get_item("entered").unwrap().unwrap().is_truthy().unwrap()
        })
    }
}

impl Drop for ManualLoop {
    fn drop(&mut self) {
        Python::attach(|py| {
            if let Ok(Some(loop_)) = self.0.bind(py).get_item("loop") {
                let _ = loop_.call_method0("close");
            }
        });
    }
}

#[test]
fn cancelled_queued_call_never_enters_the_callback_when_the_loop_resumes() {
    let host = ManualLoop::new();
    let (context, globals) = Python::attach(|py| {
        let globals = host.0.bind(py);
        let pair = PyTuple::new(
            py,
            [
                globals.get_item("loop").unwrap().unwrap(),
                globals.get_item("context").unwrap().unwrap(),
            ],
        )
        .unwrap();
        (lk_agents_abi::python::intern_host(pair.into_any().unbind()), host.0.clone_ref(py))
    });
    let mut call = super::Entry.enter(
        &context,
        Box::new(move |py| {
            Ok((globals.bind(py).get_item("callback")?.unwrap(), PyTuple::empty(py)))
        }),
    );
    assert!(matches!(call.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
    drop(call);
    host.tick();
    assert!(!host.entered(), "a cancelled queued call entered application code");
}

#[test]
fn cancellation_during_task_creation_is_applied_when_the_task_is_published() {
    let host = ManualLoop::new();
    let started = Arc::new(Mutex::new(None::<super::Started>));
    Python::attach(|py| {
        let globals = host.0.bind(py);
        let cancel = started.clone();
        let hook = PyCFunction::new_closure(py, None, None, move |_, _| -> PyResult<()> {
            cancel.lock().as_ref().unwrap().cancel();
            Ok(())
        })
        .unwrap();
        globals.set_item("cancel_during_creation", hook).unwrap();
        py.run(
            c"original_create_task = loop.create_task\ndef create_task(*args, **kwargs):\n    task = original_create_task(*args, **kwargs)\n    cancel_during_creation()\n    return task\nloop.create_task = create_task\nasync def callback():\n    entered.append('entered')\n",
            Some(globals),
            None,
        )
        .unwrap();
        let callback = globals.get_item("callback").unwrap().unwrap().unbind();
        let call = super::start_in(
            py,
            &globals.get_item("loop").unwrap().unwrap(),
            &globals.get_item("context").unwrap().unwrap(),
            move |py| callback.bind(py).call0(),
        )
        .unwrap();
        *started.lock() = Some(call);
    });
    host.tick();
    host.tick();
    host.tick();
    assert!(!host.entered(), "cancellation during task publication was lost");
}
