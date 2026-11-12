use super::*;

mod invocation {
    //! Against a live interpreter and a running loop: the context a call sees, what it leaves
    //! behind, failures and cancellation.
    use lk_agents_abi::{HostRef, invoke::HostNotices};

    use super::*;
    use pyo3::types::PyTuple;

    fn globals(py: Python<'_>) -> Bound<'_, PyAny> {
        crate::py_test_support::fixture(py, "invocation")
    }

    fn captured(globals: &Bound<'_, PyAny>, value: &str) -> HostRef {
        let pair = globals.getattr("capture").unwrap().call1((value,)).unwrap();
        globals.setattr("pair", &pair).unwrap();
        crate::py_objects::retain(pair.unbind())
    }

    fn callable(
        globals: Py<PyAny>,
        name: &'static str,
        args: Vec<&'static str>,
    ) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<crate::py_calls::Invocation<'py>> + Send + 'static
    {
        move |py| {
            let f = globals.bind(py).getattr(name)?;
            Ok((f, PyTuple::new(py, args)?, None))
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
            let seen = globals.bind(py).getattr("seen").unwrap();
            let first: (String, bool) = seen.get_item("sync").unwrap().extract().unwrap();
            let second: (String, bool) = seen.get_item("second").unwrap().extract().unwrap();
            assert_eq!(first, ("captured".into(), true));
            assert_eq!(second, ("captured".into(), true), "the first call's change leaked");
            // The captured context is as the capture left it (the calls ran in copies).
            let var = globals.bind(py).getattr("var").unwrap();
            let ctx = globals.bind(py).getattr("pair").unwrap().get_item(1).unwrap();
            assert_eq!(ctx.get_item(var).unwrap().extract::<String>().unwrap(), "captured");
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
            let seen = globals.bind(py).getattr("seen").unwrap();
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
                let blocked = globals.bind(py).getattr("blocked").unwrap();
                blocked.get_item("cancelled").is_ok()
            });
            if cancelled {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the dropped call's host task was not cancelled");
    }

    // Catches: notifications reordered on their way to the host (a task per call), or one run
    // off the loop thread or outside a copy of its context.
    #[tokio::test]
    async fn notifications_reach_the_host_in_call_order_in_their_context() {
        Python::initialize();
        let (globals, context) = Python::attach(|py| {
            let g = globals(py);
            crate::py_test_support::call(
                py,
                "notifications_reach_the_host_in_call_order_in_their_context",
                (&g,),
            )
            .unwrap();
            let c = captured(&g, "captured");
            (g.unbind(), c)
        });
        let notifier = Notifier::default();
        for i in 0..200 {
            let g = Python::attach(|py| globals.clone_ref(py));
            notifier.notify(
                context.clone(),
                "note",
                false,
                Box::new(move |py| {
                    let f = g.bind(py).getattr("note")?;
                    Ok((f, PyTuple::new(py, [i])?, None))
                }),
            );
        }
        for _ in 0..200 {
            let order: Vec<(i64, String, bool)> =
                Python::attach(|py| globals.bind(py).getattr("order").unwrap().extract().unwrap());
            if order.len() == 200 {
                let expected: Vec<(i64, String, bool)> =
                    (0..200).map(|i| (i, "captured".to_owned(), true)).collect();
                assert_eq!(order, expected);
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the notifications did not all reach the host");
    }

    // Catches: a control notification (a flush, a clear, a session event) dropped because the
    // host loop fell behind; only a lossy one (a stream's frame) may be.
    #[tokio::test]
    async fn a_notification_behind_a_stalled_loop_is_delivered_and_a_lossy_one_dropped() {
        Python::initialize();
        let (globals, context) = Python::attach(|py| {
            let g = globals(py);
            crate::py_test_support::call(
                py,
                "a_notification_behind_a_stalled_loop_is_delivered_and_a_lossy_one_dropped",
                (&g,),
            )
            .unwrap();
            let c = captured(&g, "captured");
            (g.unbind(), c)
        });
        let notifier = Notifier::default();
        fn call(
            g: Py<PyAny>,
            name: &'static str,
            i: i64,
        ) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<crate::py_calls::Invocation<'py>>
        + Send
        + 'static {
            move |py| Ok((g.bind(py).getattr(name)?, PyTuple::new(py, [i])?, None))
        }
        let send = |name: &'static str, i: i64, lossy: bool| {
            let g = Python::attach(|py| globals.clone_ref(py));
            notifier.notify(context.clone(), "note", lossy, Box::new(call(g, name, i)))
        };
        send("stall", 0, false);
        let total = NOTIFY_QUEUE as i64 * 2;
        for i in 0..total {
            send("note", i, false);
        }
        for i in 0..total {
            send("note", total + i, true);
        }
        // The lossy ones arrived while every control one still waited: all dropped.
        let seen = || -> Vec<i64> {
            Python::attach(|py| globals.bind(py).getattr("seen").unwrap().extract().unwrap())
        };
        for _ in 0..300 {
            if seen().len() >= total as usize {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                assert_eq!(
                    seen(),
                    (0..total).collect::<Vec<_>>(),
                    "every control notification, in order"
                );
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the control notifications did not all reach the host");
    }
}

mod cancellation {
    //! Cancellation before the host loop starts a call and while it publishes the host task.

    use parking_lot::Mutex;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    use lk_agents_abi::invoke::HostEntry;
    use pyo3::prelude::*;
    use pyo3::types::{PyCFunction, PyTuple};

    pub(super) struct ManualLoop(pub(super) Py<PyAny>);

    impl ManualLoop {
        pub(super) fn new() -> Self {
            Python::initialize();
            Python::attach(|py| {
                let globals = crate::py_test_support::fixture(py, "manual_loop");
                Self(globals.unbind())
            })
        }

        pub(super) fn tick(&self) {
            Python::attach(|py| {
                let globals = self.0.bind(py);
                crate::py_test_support::call(py, "tick_loop", (globals,)).unwrap();
            });
        }

        fn entered(&self) -> bool {
            Python::attach(|py| self.0.bind(py).getattr("entered").unwrap().is_truthy().unwrap())
        }
    }

    impl Drop for ManualLoop {
        fn drop(&mut self) {
            Python::attach(|py| {
                if let Ok(loop_) = self.0.bind(py).getattr("loop") {
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
                [globals.getattr("loop").unwrap(), globals.getattr("context").unwrap()],
            )
            .unwrap();
            (crate::py_objects::retain(pair.into_any().unbind()), host.0.clone_ref(py))
        });
        let mut call = super::Entry.enter(
            &context,
            None,
            Box::new(move |py| {
                Ok((globals.bind(py).getattr("callback")?, PyTuple::empty(py), None))
            }),
        );
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
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
            globals.setattr("cancel_during_creation", hook).unwrap();
            crate::py_test_support::call(
                py,
                "cancellation_during_task_creation_is_applied_when_the_task_is_published",
                (globals,),
            )
            .unwrap();
            let callback = globals.getattr("callback").unwrap().unbind();
            let call = super::start_in(
                py,
                &globals.getattr("loop").unwrap(),
                &globals.getattr("context").unwrap(),
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
}

mod teardown {
    //! A closing loop, and a drop holding a lock, never leave a waiter or a queue asleep.

    use parking_lot::Mutex;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use lk_agents_abi::invoke::HostEntry as _;

    use super::cancellation::ManualLoop;
    use lk_agents_abi::invoke::HostNotices as _;

    fn pair(host: &ManualLoop) -> lk_agents_abi::HostRef {
        Python::attach(|py| {
            let globals = host.0.bind(py);
            let pair = PyTuple::new(
                py,
                [globals.getattr("loop").unwrap(), globals.getattr("context").unwrap()],
            )
            .unwrap();
            crate::py_objects::retain(pair.into_any().unbind())
        })
    }

    fn start_forever(host: &ManualLoop) -> super::Started {
        Python::attach(|py| {
            let globals = host.0.bind(py);
            let forever = crate::py_test_support::fixture(py, "pending_forever").unbind();
            super::start_in(
                py,
                &globals.getattr("loop").unwrap(),
                &globals.getattr("context").unwrap(),
                move |py| forever.bind(py).call0(),
            )
            .unwrap()
        })
    }

    fn close(host: &ManualLoop) {
        Python::attach(|py| {
            host.0.bind(py).getattr("loop").unwrap().call_method0("close").unwrap();
        });
    }

    // Catches: a drop that attaches to the interpreter on the dropping thread, which deadlocks
    // when that thread holds a lock a thread holding the GIL is waiting on.
    #[test]
    fn dropping_a_pending_call_under_a_lock_never_waits_for_the_gil() {
        let host = ManualLoop::new();
        let started = start_forever(&host);
        let lock = Arc::new(Mutex::new(()));
        let (locked, is_locked) = std::sync::mpsc::channel();
        let (go, gone) = std::sync::mpsc::channel::<()>();
        let (dropped, was_dropped) = std::sync::mpsc::channel();
        let dropper = std::thread::spawn({
            let lock = lock.clone();
            move || {
                let _held = lock.lock();
                locked.send(()).unwrap();
                gone.recv().unwrap();
                drop(started);
                dropped.send(()).unwrap();
            }
        });
        is_locked.recv().unwrap();
        let (attached, is_attached) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            Python::attach(|_| {
                attached.send(()).unwrap();
                // Bounded, so a regression fails the test instead of hanging the process.
                drop(lock.try_lock_for(Duration::from_secs(3)));
            })
        });
        is_attached.recv().unwrap();
        go.send(()).unwrap();
        let in_time = was_dropped.recv_timeout(Duration::from_secs(2)).is_ok();
        holder.join().unwrap();
        dropper.join().unwrap();
        assert!(in_time, "dropping a pending call waited for the GIL under a lock");
    }

    // Catches: an async host call whose coroutine is built before its loop runs it, so a loop that
    // stopped and closes first leaves "coroutine ... was never awaited" (the worker's periodic
    // host calls scheduled as its loop closed at shutdown).
    #[test]
    fn an_async_call_queued_as_its_loop_closes_leaves_no_coroutine_unawaited() {
        let host = ManualLoop::new();
        let context = pair(&host);
        let mut call = super::Entry.enter(
            &context,
            None,
            Box::new(|py| {
                let forever = crate::py_test_support::fixture(py, "pending_forever");
                Ok((forever, PyTuple::empty(py), None))
            }),
        );
        let mut poll = || call.as_mut().poll(&mut Context::from_waker(Waker::noop()));
        assert!(poll().is_pending());

        let unawaited: Vec<String> = Python::attach(|py| {
            let globals = host.0.bind(py);
            crate::py_test_support::call(py, "close_loop_collecting_unawaited", (globals,))
                .unwrap()
                .extract()
                .unwrap()
        });

        assert_eq!(unawaited, Vec::<String>::new());
        assert!(matches!(poll(), Poll::Ready(Err(_))), "the call outlived its closed loop");
    }

    // Catches: a drain the loop dropped unrun (it closed) leaving the queue marked scheduled, so
    // every later notification waits silently for a drain that never comes.
    #[test]
    fn a_notification_queue_whose_loop_closed_is_released() {
        let host = ManualLoop::new();
        let context = pair(&host);
        let notifier = super::Notifier::default();
        let note = || -> super::HostCall {
            Box::new(|py| Ok((py.import("builtins")?.getattr("print")?, PyTuple::empty(py), None)))
        };
        notifier.notify(context.clone(), "note", false, note());
        assert!(notifier.0.queue.lock().scheduled);
        close(&host);
        let released = |notifier: &super::Notifier| {
            let queue = notifier.0.queue.lock();
            !queue.scheduled && queue.entries.is_empty()
        };
        assert!(released(&notifier), "the closed loop's drain left the queue scheduled");
        notifier.notify(context, "note", false, note());
        assert!(released(&notifier), "a notification after the loop closed was queued");
    }
}

mod parentage {
    //! Standard OTel child instrumentation through the real HostInvoker context boundary.

    use pyo3::prelude::*;
    use pyo3::types::PyTuple;

    use super::call_in;

    struct HostLoop(Py<PyAny>);

    impl Drop for HostLoop {
        fn drop(&mut self) {
            Python::attach(|py| {
                if let Ok(close) = self.0.bind(py).getattr("close") {
                    let _ = close.call0();
                }
            });
        }
    }

    fn callback(
        globals: Py<PyAny>,
        mode: &'static str,
    ) -> impl for<'py> FnOnce(Python<'py>) -> PyResult<crate::py_calls::Invocation<'py>> + Send + 'static
    {
        move |py| Ok((globals.bind(py).getattr("callback")?, PyTuple::new(py, [mode])?, None))
    }

    #[tokio::test]
    async fn invocation_trace_parentage_is_isolated_and_restored_after_failure_and_cancellation() {
        Python::initialize();
        let (host, first, second) = Python::attach(|py| {
            let globals = crate::py_test_support::fixture(py, "parentage");
            let capture = globals.getattr("capture").unwrap();
            let first = capture.call1(("one", 101u64, 11u64, true)).unwrap();
            let second = capture.call1(("two", 202u64, 22u64, true)).unwrap();
            (
                HostLoop(globals.unbind()),
                crate::py_objects::retain(first.unbind()),
                crate::py_objects::retain(second.unbind()),
            )
        });
        fn owned(host: &HostLoop) -> Py<PyAny> {
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
                .getattr("cancelled")
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
                .getattr("capture")
                .unwrap()
                .call1(("unrecorded", 303u64, 33u64, false))
                .unwrap();
            crate::py_objects::retain(pair.unbind())
        });
        let result = call_in(&unrecorded, callback(owned(&host), "success")).await.unwrap();
        Python::attach(|py| assert_eq!(result.extract::<(u64, bool)>(py).unwrap(), (303, false)));
        Python::attach(|py| {
            host.0.bind(py).getattr("verify").unwrap().call0().unwrap();
            for captured in [&first, &second] {
                let pair = crate::py_objects::resolve(py, captured).unwrap();
                let context = pair.bind(py).get_item(1).unwrap();
                let var = host.0.bind(py).getattr("job").unwrap();
                let value: String = context.call_method1("get", (var,)).unwrap().extract().unwrap();
                assert!(value == "one" || value == "two", "callback changed the captured context");
            }
        });
    }
}

mod futures {
    //! The shared Rust-to-asyncio bridge: results, task context, cancellation, and runtime cost.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use pyo3::prelude::*;
    use pyo3::types::{PyCFunction, PyTuple};
    use pyo3_async_runtimes::generic::ContextExt;

    fn globals() -> Py<PyAny> {
        Python::initialize();
        Python::attach(|py| crate::py_test_support::fixture(py, "bridge_context").unbind())
    }

    struct FailingConversion;

    impl<'py> IntoPyObject<'py> for FailingConversion {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;

        fn into_pyobject(self, _: Python<'py>) -> PyResult<Self::Output> {
            Err(pyo3::exceptions::PyValueError::new_err("conversion failure"))
        }
    }

    #[test]
    fn async_bridge_preserves_results_errors_and_task_context() {
        let globals = globals();
        Python::attach(|py| {
            let globals = globals.bind(py);
            let job = globals.getattr("job").unwrap().unbind();
            let context =
                PyCFunction::new_closure(py, None, None, move |args: &Bound<'_, PyTuple>, _| {
                    let job = job.clone_ref(args.py());
                    super::future_into_py(args.py(), async move {
                        tokio::task::yield_now().await;
                        let locals = super::ProcessRuntime::get_task_locals().unwrap();
                        Python::attach(|py| {
                            locals
                                .context(py)
                                .call_method1("get", (job.bind(py),))?
                                .extract::<String>()
                        })
                    })
                    .map(Bound::unbind)
                })
                .unwrap();
            globals.setattr("read_context", context).unwrap();
            let value = PyCFunction::new_closure(py, None, None, |args: &Bound<'_, PyTuple>, _| {
                let present = args.get_item(0)?.extract::<bool>()?;
                super::future_into_py(args.py(), async move { Ok(present.then_some(42_u64)) })
                    .map(Bound::unbind)
            })
            .unwrap();
            globals.setattr("value", value).unwrap();
            let failure =
                PyCFunction::new_closure(py, None, None, |args: &Bound<'_, PyTuple>, _| {
                    super::future_into_py(args.py(), async {
                        Err::<u64, _>(pyo3::exceptions::PyValueError::new_err("bridge failure"))
                    })
                    .map(Bound::unbind)
                })
                .unwrap();
            globals.setattr("failure", failure).unwrap();
            let conversion =
                PyCFunction::new_closure(py, None, None, |args: &Bound<'_, PyTuple>, _| {
                    super::future_into_py(args.py(), async { Ok(FailingConversion) })
                        .map(Bound::unbind)
                })
                .unwrap();
            globals.setattr("conversion_failure", conversion).unwrap();
            crate::py_test_support::call(py, "verify_async_bridge", (globals,)).unwrap();
        });
    }

    struct Dropped(Arc<AtomicBool>);

    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn cancelling_the_python_future_drops_the_pending_rust_future() {
        let globals = globals();
        let entered = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        Python::attach(|py| {
            let globals = globals.bind(py);
            let started = entered.clone();
            let finished = dropped.clone();
            let pending =
                PyCFunction::new_closure(py, None, None, move |args: &Bound<'_, PyTuple>, _| {
                    let entered = entered.clone();
                    let dropped = dropped.clone();
                    super::future_into_py(args.py(), async move {
                        let _dropped = Dropped(dropped);
                        entered.store(true, Ordering::SeqCst);
                        std::future::pending::<PyResult<u64>>().await
                    })
                    .map(Bound::unbind)
                })
                .unwrap();
            globals.setattr("pending", pending).unwrap();
            for (name, flag) in [("started", started), ("dropped", finished)] {
                let read =
                    PyCFunction::new_closure(py, None, None, move |_, _| -> PyResult<bool> {
                        Ok(flag.load(Ordering::SeqCst))
                    })
                    .unwrap();
                globals.setattr(name, read).unwrap();
            }
            crate::py_test_support::call(
                py,
                "cancelling_the_python_future_drops_the_pending_rust_future",
                (globals,),
            )
            .unwrap();
        });
    }
}
