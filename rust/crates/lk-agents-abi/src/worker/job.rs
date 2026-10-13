//! The end of a job (`job.py`). Deliberately better than the reference's unbounded `gather`: the
//! shutdown callbacks run in order, each failure isolated, each bounded.

use std::future::Future;
use std::time::Duration;

use crate::{Clock, is_cloud_url, timeout};

/// `_observability_url(livekit_url)` (`job.py:58`). The env override wins unvalidated; otherwise
/// only LiveKit Cloud URLs get one, at `https://` + the ws url's hostname.
#[crate::host(export)]
pub fn observability_url(livekit_url: &str, env_override: Option<&str>) -> Option<String> {
    if let Some(u) = env_override.filter(|u| !u.is_empty()) {
        return Some(u.to_string());
    }
    let host = crate::utils::strings::host_of(livekit_url)?;
    is_cloud_url(livekit_url).then(|| format!("https://{host}"))
}

// ---------------------------------------------------------------------------------------------
// the shutdown callbacks
// ---------------------------------------------------------------------------------------------

/// The reference has none: above any legitimate cleanup, below the supervisor's kill timeout.
pub const SHUTDOWN_CALLBACK_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallbackOutcome {
    Ok,
    /// The rest still run.
    Raised(String),
    /// The rest still run, and it is handed back to the host.
    TimedOut,
}

/// Returns the callbacks that overran, still running: Rust cannot cancel a host coroutine, so the
/// host cancels them once the rest are done.
pub async fn run_shutdown_callbacks<C, F>(
    clock: &dyn Clock,
    callbacks: Vec<C>,
    per_callback: Duration,
    mut report: impl FnMut(usize, &CallbackOutcome),
) -> Vec<F>
where
    C: FnOnce() -> F,
    F: Future<Output = Result<(), String>> + Unpin,
{
    let mut abandoned = Vec::new();
    for (i, start) in callbacks.into_iter().enumerate() {
        let mut running = start();
        let outcome = match timeout(clock, per_callback, &mut running).await {
            Ok(Ok(())) => CallbackOutcome::Ok,
            Ok(Err(e)) => CallbackOutcome::Raised(e),
            Err(_) => {
                abandoned.push(running);
                CallbackOutcome::TimedOut
            }
        };
        report(i, &outcome);
    }
    abandoned
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use crate::python::{CancelPythonTask, check_pid, pyo3_async_runtimes, schedule_coroutine};
    use crate::CallbackOutcome;
    use crate::python::Ambient;

    /// The logger the job's own lines go to, as `job_proc_lazy_main.py`'s did.
    const LOG: &str = "livekit.agents";

    /// `JobContext`'s shutdown callbacks, run in registration order on the host loop, each bounded; a
    /// failure or overrun does not stop the rest, and overrunning tasks are cancelled and awaited at
    /// the end.
    #[pyfunction]
    #[pyo3(signature = (callbacks, reason, per_callback = 10.0))]
    pub fn run_shutdown_callbacks<'py>(
        py: Python<'py>,
        callbacks: Vec<Py<PyAny>>,
        reason: String,
        per_callback: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let per = match per_callback {
            s if s.is_finite() && s >= 0.0 => std::time::Duration::from_secs_f64(s),
            _ => super::SHUTDOWN_CALLBACK_TIMEOUT,
        };
        // The loop's clock when it is on virtual time, so a bound is in the test's seconds.
        let clock = <std::sync::Arc<dyn crate::Clock> as Ambient>::ambient();
        crate::python::future_into_py(py, async move {
            let starts: Vec<_> = callbacks
                .into_iter()
                .map(|cb| {
                    let (locals, reason) = (locals.clone(), reason.clone());
                    move || ShutdownCallback::start(&cb, &reason, &locals)
                })
                .collect();
            let mut failed = false;
            let abandoned = super::run_shutdown_callbacks(&*clock, starts, per, |i, o| {
                let index = i as u64;
                match o {
                    CallbackOutcome::Ok => return,
                    CallbackOutcome::Raised(error) => log::error!(
                        target: LOG, callback_index = index, error = error.as_str();
                        "a job shutdown callback raised; continuing with the rest"
                    ),
                    CallbackOutcome::TimedOut => log::error!(
                        target: LOG, callback_index = index;
                        "a job shutdown callback overran its deadline; abandoning it"
                    ),
                }
                failed = true;
            })
            .await;
            for callback in abandoned {
                callback.cancel_and_wait().await;
            }
            if failed {
                log::warn!(target: LOG, "one or more job shutdown callbacks failed or overran");
            }
            Ok(())
        })
    }

    crate::register_host!(fn self::run_shutdown_callbacks);

    /// Dropping the guard cancels the task.
    struct ShutdownCallback {
        guard: Option<CancelPythonTask>,
        done: Option<tokio::sync::oneshot::Receiver<PyResult<Py<PyAny>>>>,
    /// It raised when called, or is not a coroutine.
        failed: Option<String>,
    }

    impl ShutdownCallback {
        fn start(cb: &Py<PyAny>, reason: &str, locals: &pyo3_async_runtimes::TaskLocals) -> Self {
            let scheduled = Python::attach(|py| {
                let coro = cb.bind(py).call1((reason,))?;
                schedule_coroutine(py, locals, coro)
            });
            match scheduled {
                Ok((guard, done)) => ShutdownCallback { guard: Some(guard), done: Some(done), failed: None },
                Err(e) => ShutdownCallback { guard: None, done: None, failed: Some(e.to_string()) },
            }
        }

        async fn cancel_and_wait(mut self) {
            // Dropping the guard is the cancel.
            drop(self.guard.take());
            if let Some(done) = self.done.take() {
                let _ = done.await;
            }
        }
    }

    impl std::future::Future for ShutdownCallback {
        type Output = Result<(), String>;

        fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
            use std::task::Poll;
            if let Some(e) = self.failed.take() {
                return Poll::Ready(Err(e));
            }
            let Some(done) = self.done.as_mut() else { return Poll::Ready(Ok(())) };
            match std::pin::Pin::new(done).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    self.done = None;
                    Poll::Ready(match result {
                        Ok(Ok(_)) => Ok(()),
                        Ok(Err(e)) => Err(e.to_string()),
                        Err(_) => Err("the Python task disappeared".to_owned()),
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::future::{Ready, pending, ready};
    use std::pin::Pin;
    use std::rc::Rc;

    use super::*;
    use crate::SimClock;

    type Cb = Pin<Box<dyn Future<Output = Result<(), String>>>>;

    #[test]
    fn the_observability_url_is_cloud_only_unless_overridden() {
        assert_eq!(
            observability_url("wss://project.livekit.cloud", None).as_deref(),
            Some("https://project.livekit.cloud")
        );
        // Port and path are dropped; the host is lowercased.
        assert_eq!(
            observability_url("wss://Project.LiveKit.Cloud:443/rtc", None).as_deref(),
            Some("https://project.livekit.cloud")
        );
        assert_eq!(observability_url("ws://localhost:7880", None), None);
        assert_eq!(observability_url("wss://sfu.example.com", None), None);
        assert_eq!(
            observability_url("ws://localhost:7880", Some("http://collector:4318")).as_deref(),
            Some("http://collector:4318")
        );
        assert_eq!(
            observability_url("wss://project.livekit.cloud", Some("")).as_deref(),
            Some("https://project.livekit.cloud")
        );
    }

    // Catches: one raising callback aborting the rest.
    #[tokio::test(start_paused = true)]
    async fn a_raising_shutdown_callback_does_not_abort_the_rest() {
        let clock = SimClock::new();
        let results: Vec<Result<(), String>> = vec![Err("boom".into()), Ok(()), Ok(())];
        let callbacks: Vec<_> = results.into_iter().map(|r| move || ready(r)).collect();
        let mut seen = vec![];
        let abandoned: Vec<Ready<_>> = run_shutdown_callbacks(
            &clock,
            callbacks,
            SHUTDOWN_CALLBACK_TIMEOUT,
            |i, o| seen.push((i, o.clone())),
        )
        .await;
        assert!(abandoned.is_empty());
        assert_eq!(
            seen,
            [
                (0, CallbackOutcome::Raised("boom".into())),
                (1, CallbackOutcome::Ok),
                (2, CallbackOutcome::Ok),
            ]
        );
    }

    // Catches: a hanging callback wedging shutdown, or one dropped rather than handed back.
    #[tokio::test(start_paused = true)]
    async fn a_hanging_callback_hits_its_deadline_is_handed_back_and_the_rest_still_run() {
        let clock = SimClock::new();
        let callbacks: Vec<Box<dyn FnOnce() -> Cb>> = vec![
            Box::new(|| Box::pin(pending())),
            Box::new(|| Box::pin(ready(Ok(())))),
        ];
        let start = clock.now();
        let mut seen = vec![];
        let abandoned = run_shutdown_callbacks(
            &clock,
            callbacks,
            Duration::from_secs(10),
            |i, o| seen.push((i, o.clone())),
        )
        .await;
        assert_eq!(seen, [(0, CallbackOutcome::TimedOut), (1, CallbackOutcome::Ok)]);
        assert_eq!(abandoned.len(), 1);
        assert_eq!(clock.now(), start.saturating_add(Duration::from_secs(10)));
    }

    // Catches: starting every callback up front instead of in turn.
    #[tokio::test(start_paused = true)]
    async fn callbacks_start_one_at_a_time_in_registration_order() {
        let clock = SimClock::new();
        let log = Rc::new(RefCell::new(Vec::new()));
        let callbacks: Vec<_> = [7, 3, 5]
            .into_iter()
            .map(|n| {
                let log = Rc::clone(&log);
                move || {
                    log.borrow_mut().push(format!("start {n}"));
                    ready(Ok(()))
                }
            })
            .collect();
        let report_log = Rc::clone(&log);
        let _: Vec<Ready<_>> =
            run_shutdown_callbacks(&clock, callbacks, SHUTDOWN_CALLBACK_TIMEOUT, |i, _| {
                report_log.borrow_mut().push(format!("done {i}"))
            })
            .await;
        assert_eq!(*log.borrow(), ["start 7", "done 0", "start 3", "done 1", "start 5", "done 2"]);
    }
}
