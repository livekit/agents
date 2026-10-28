//! The Python half of `HostInvoker`: a callback runs in a copy of the context its operation
//! captured at entry, on that context's event loop. The call itself runs inside the copy (a sync
//! callable included), an awaitable it returns becomes a task of the same copy, and the caller's
//! context is never entered or changed. Dropping the call cancels the host task; it still settles.
//!
//! The captured context crosses as a `HostRef` to the host's `(loop, contextvars.Context)` pair,
//! taken at the operation's entry (`asyncio.get_running_loop(), contextvars.copy_context()`).

use std::sync::{Arc, Mutex};

use lk_agents_abi::python::resolve_key;
use lk_agents_abi::utils::host_ref::HostRef;
use pyo3::prelude::*;
use pyo3::types::PyCFunction;
use tokio::sync::oneshot;

/// The host task of a call in flight, cancelled when the call is dropped before it settles.
struct InFlight {
    event_loop: Py<PyAny>,
    task: Arc<Mutex<Option<Py<PyAny>>>>,
    settled: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for InFlight {
    fn drop(&mut self) {
        if self.settled.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let task = self.task.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(task) = task {
            Python::attach(|py| {
                let cancel = task.bind(py).getattr("cancel");
                if let Ok(cancel) = cancel {
                    let _ =
                        self.event_loop.bind(py).call_method1("call_soon_threadsafe", (cancel,));
                }
            });
        }
    }
}

/// Runs `call` (building the callable and its arguments) in a copy of `context`. Resolves to
/// the call's value, or the value its awaitable resolved to.
pub async fn call_in<F>(context: &HostRef, call: F) -> PyResult<Py<PyAny>>
where
    F: for<'py> FnOnce(
            Python<'py>,
        ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)>
        + Send
        + 'static,
{
    let (sender, receiver) = oneshot::channel::<PyResult<Py<PyAny>>>();
    let task = Arc::new(Mutex::new(None));
    let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let event_loop = Python::attach(|py| -> PyResult<Py<PyAny>> {
        let captured = resolve_key(py, context.id())?;
        let (event_loop, ctx): (Bound<'_, PyAny>, Bound<'_, PyAny>) =
            captured.bind(py).extract()?;
        let ctx = ctx.call_method0("copy")?;
        let sender = Arc::new(Mutex::new(Some(sender)));
        let deliver = {
            let sender = sender.clone();
            let settled = settled.clone();
            move |result: PyResult<Py<PyAny>>| {
                settled.store(true, std::sync::atomic::Ordering::SeqCst);
                if let Some(sender) = sender.lock().unwrap_or_else(|p| p.into_inner()).take() {
                    let _ = sender.send(result);
                }
            }
        };
        let task_slot = task.clone();
        let call = Mutex::new(Some(call));
        // Runs on the loop thread, inside `ctx`: the call, then its awaitable as a task of `ctx`.
        let start = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
            let py = args.py();
            let Some(call) = call.lock().unwrap_or_else(|p| p.into_inner()).take() else {
                return Ok(());
            };
            let returned = call(py).and_then(|(callable, arguments)| callable.call1(arguments));
            let value = match returned {
                Ok(value) => value,
                Err(e) => {
                    deliver(Err(e));
                    return Ok(());
                }
            };
            let inspect = py.import("inspect")?;
            if !inspect.call_method1("isawaitable", (&value,))?.is_truthy()? {
                deliver(Ok(value.unbind()));
                return Ok(());
            }
            // Inside `ctx.run`: the task copies `ctx`, mutations of the sync prefix included.
            let future = py.import("asyncio")?.call_method1("ensure_future", (&value,))?;
            *task_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(future.clone().unbind());
            let deliver = Mutex::new(Some(deliver.clone()));
            let done = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
                let finished = args.get_item(0)?;
                let result = finished.call_method0("result").map(Bound::unbind);
                if let Some(deliver) = deliver.lock().unwrap_or_else(|p| p.into_inner()).take() {
                    deliver(result);
                }
                Ok(())
            })?;
            future.call_method1("add_done_callback", (done,))?;
            Ok(())
        })?;
        event_loop.call_method1("call_soon_threadsafe", (ctx.getattr("run")?, start))?;
        Ok(event_loop.unbind())
    })?;
    let in_flight = InFlight { event_loop, task, settled };
    let result = receiver.await.unwrap_or_else(|_| {
        Err(pyo3::exceptions::PyRuntimeError::new_err("the host event loop dropped the call"))
    });
    drop(in_flight);
    result
}

#[cfg(test)]
mod tests {
    //! Against a live interpreter and a running loop: the context a call sees, what it leaves
    //! behind, failures and cancellation.
    use super::*;
    use pyo3::types::{PyDict, PyTuple};

    const SETUP: &str = r#"
import asyncio, contextvars, threading
var = contextvars.ContextVar('var', default='unset')
seen = {}
loop = asyncio.new_event_loop()
ready = threading.Event()
def run():
    asyncio.set_event_loop(loop); loop.call_soon(ready.set); loop.run_forever()
threading.Thread(target=run, daemon=True).start()
ready.wait()

def capture(value):
    var.set(value)
    return (loop, contextvars.copy_context())

def sync_call(tag):
    seen[tag] = (var.get(), threading.current_thread() is not threading.main_thread())
    var.set('changed by ' + tag)
    return tag

async def async_call(tag):
    var.set('before await')
    await asyncio.sleep(0)
    seen[tag] = var.get()
    return tag

async def failing():
    raise ValueError('boom')

blocked = {}
async def blocking():
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        blocked['cancelled'] = True
        raise
"#;

    fn globals(py: Python<'_>) -> Bound<'_, PyDict> {
        let globals = PyDict::new(py);
        py.run(&std::ffi::CString::new(SETUP).unwrap(), Some(&globals), None).unwrap();
        globals
    }

    fn captured(globals: &Bound<'_, PyDict>, value: &str) -> HostRef {
        let pair = globals.get_item("capture").unwrap().unwrap().call1((value,)).unwrap();
        lk_agents_abi::python::intern_host(pair.unbind())
    }

    fn callable(
        globals: Py<PyDict>,
        name: &'static str,
        args: Vec<&'static str>,
    ) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)>
    + Send
    + 'static {
        move |py| {
            let f = globals.bind(py).get_item(name)?.expect("defined");
            Ok((f, PyTuple::new(py, args)?))
        }
    }

    // Catches: a sync callback run outside the captured context, off the loop thread, or one whose
    // context change leaks into the captured context or the caller's.
    #[tokio::test]
    async fn a_sync_call_runs_in_a_copy_of_the_captured_context_on_the_loop() {
        Python::initialize();
        let (globals, context) = Python::attach(|py| {
            let g = globals(py);
            let c = captured(&g, "captured");
            (g.unbind(), c)
        });
        let g = Python::attach(|py| globals.clone_ref(py));
        let value = call_in(&context, callable(g, "sync_call", vec!["sync"])).await.unwrap();
        let again = Python::attach(|py| globals.clone_ref(py));
        call_in(&context, callable(again, "sync_call", vec!["second"])).await.unwrap();
        Python::attach(|py| {
            assert_eq!(value.extract::<String>(py).unwrap(), "sync");
            let seen = globals.bind(py).get_item("seen").unwrap().unwrap();
            let first: (String, bool) = seen.get_item("sync").unwrap().extract().unwrap();
            let second: (String, bool) = seen.get_item("second").unwrap().extract().unwrap();
            assert_eq!(first, ("captured".into(), true));
            assert_eq!(second, ("captured".into(), true), "the first call's change leaked");
            // The caller's own context is as the capture left it.
            let var = globals.bind(py).get_item("var").unwrap().unwrap();
            assert_eq!(var.call_method0("get").unwrap().extract::<String>().unwrap(), "captured");
        });
    }

    // Catches: an awaitable run in a fresh context, so a change made before its first await is
    // lost after it (one context per resumption).
    #[tokio::test]
    async fn an_async_call_keeps_its_context_across_awaits() {
        Python::initialize();
        let (globals, context) = Python::attach(|py| {
            let g = globals(py);
            let c = captured(&g, "captured");
            (g.unbind(), c)
        });
        let g = Python::attach(|py| globals.clone_ref(py));
        call_in(&context, callable(g, "async_call", vec!["async"])).await.unwrap();
        Python::attach(|py| {
            let seen = globals.bind(py).get_item("seen").unwrap().unwrap();
            assert_eq!(
                seen.get_item("async").unwrap().extract::<String>().unwrap(),
                "before await"
            );
        });
    }

    // Catches: a host exception lost or replaced, or a dropped call leaving its host task running.
    #[tokio::test]
    async fn failures_propagate_and_a_dropped_call_cancels_its_task() {
        Python::initialize();
        let (globals, context) = Python::attach(|py| {
            let g = globals(py);
            let c = captured(&g, "captured");
            (g.unbind(), c)
        });
        let g = Python::attach(|py| globals.clone_ref(py));
        let error = call_in(&context, callable(g, "failing", vec![])).await.unwrap_err();
        Python::attach(|py| assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py)));

        let g = Python::attach(|py| globals.clone_ref(py));
        let call = call_in(&context, callable(g, "blocking", vec![]));
        assert!(tokio::time::timeout(std::time::Duration::from_millis(50), call).await.is_err());
        for _ in 0..100 {
            let cancelled = Python::attach(|py| {
                let blocked = globals.bind(py).get_item("blocked").unwrap().unwrap();
                blocked.get_item("cancelled").is_ok()
            });
            if cancelled {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the dropped call's host task was not cancelled");
    }
}
