//! The generated `_core.LoopMonitor`'s Python host adapter: `telemetry/loop_monitor.py`'s watchdog. A heartbeat on the loop
//! (`loop.call_later`) stamps each tick; a watchdog thread notices a tick that is late, samples
//! the loop thread's stack while the block lasts, and the late tick reports the block, measured.
//! The report carries its span's attributes and its log line, rate limited; the host emits them
//! and records the metric. The thresholds come from the environment, the tick from the warn
//! threshold.

use std::sync::{Arc, Mutex};

use lk_agents_runtime::telemetry::loop_monitor::{
    LoopMonitorClock, LoopMonitorOptions, LoopMonitorPolicy, LoopWatchdog,
};
use lk_agents_abi::python::{capture_host, resolve_key};
use lk_agents_abi::telemetry::loop_monitor::{
    BlockedReport, LoopBlock, LoopMonitorEmission, LoopMonitorLog, LoopMonitorSpan,
};
use lk_agents_abi::{HostRef, errors::HostError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict};
use serde_json::{Map, Value};

const MAX_STACK_FRAMES: usize = 20;
const LOG: &str = "livekit.agents";

/// One frame of a sampled stack, outermost first.
#[derive(Clone)]
struct Frame {
    filename: String,
    lineno: Option<i64>,
    name: String,
}

struct StackSample {
    lag: f64,
    task_name: Option<String>,
    frames: Vec<Frame>,
    /// The span current in the blocked task, as the host's helper read it.
    span_context: Option<Py<PyAny>>,
    /// The module a lazy import was loading when sampled.
    importing: Option<String>,
}

/// The watchdog's view of one block, keyed by the heartbeat sequence it interrupted.
struct Incident {
    tick_seq: u64,
    samples: Vec<StackSample>,
}

struct State {
    closed: bool,
    started: bool,
    heartbeat: Option<Arc<Py<PyAny>>>,
    policy: LoopMonitorPolicy,
    incident: Option<Incident>,
    loop_thread: Option<std::thread::ThreadId>,
    loop_thread_ident: Option<u64>,
    timer: Option<Py<PyAny>>,
    gc_callback: Option<Py<PyAny>>,
    watchdog: Option<LoopWatchdog>,
}

struct Inner {
    event_loop: Py<PyAny>,
    report: Py<PyAny>,
    /// `span_of(task_context)`: the span current in a task's context, or `None`.
    span_of: Py<PyAny>,
    options: LoopMonitorOptions,
    name: String,
    asyncio_dir: String,
    clock: LoopMonitorClock,
    state: Mutex<State>,
}

impl Inner {
    fn now(&self) -> f64 {
        self.clock.now()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

pub struct LoopMonitor {
    inner: Arc<Inner>,
}

fn gc_callback(inner: &Inner, phase: &str) {
    let now = inner.now();
    let mut st = inner.lock();
    if st.loop_thread != Some(std::thread::current().id()) {
        return;
    }
    match phase {
        "start" => st.policy.gc_started(now),
        "stop" => st.policy.gc_ended(now),
        _ => {}
    }
}

fn heartbeat(py: Python<'_>, inner: &Arc<Inner>) -> PyResult<()> {
    let thread_cpu: f64 = py.import("time")?.getattr("thread_time")?.call0()?.extract()?;
    let now = inner.now();
    let (block, samples, heartbeat) = {
        let mut st = inner.lock();
        if st.closed {
            return Ok(());
        }
        let block = st.policy.tick(now, inner.clock.epoch_seconds(), thread_cpu);
        let incident = st.incident.take();
        let samples = block
            .as_ref()
            .and_then(|block| incident.filter(|i| i.tick_seq == block.tick_sequence))
            .map(|i| i.samples)
            .unwrap_or_default();
        (block, samples, st.heartbeat.clone())
    };
    if let Some(heartbeat) = heartbeat {
        let timer = inner
            .event_loop
            .bind(py)
            .call_method1("call_later", (inner.options.tick, heartbeat.bind(py)))?;
        let previous = inner.lock().timer.replace(timer.unbind());
        drop(previous);
    }
    let Some(block) = block else { return Ok(()) };
    let report = build_report(py, inner, block, samples)?;
    inner.report.bind(py).call1((report,))?;
    Ok(())
}

/// `_build_report`, with the rate limiters' verdicts: `emit_span`, `emit_log`, `suppressed`.
fn build_report<'py>(
    py: Python<'py>,
    inner: &Inner,
    report: LoopBlock,
    samples: Vec<StackSample>,
) -> PyResult<Bound<'py, PyAny>> {
    let reported = report.reported;
    let lag = report.duration;
    let gc_time = report.gc_time;
    let cpu_time = report.cpu_time;
    let watchdog_gap = report.watchdog_gap;
    let process_descheduled = report.process_descheduled;
    let watchdog_starved = watchdog_gap >= lag * 0.5;
    let mut stacks: Vec<String> = samples
        .iter()
        .filter(|s| !s.frames.is_empty())
        .map(|s| {
            format!(
                "# loop thread sampled {:.0}ms into the stall\n{}",
                s.lag * 1000.0,
                format_frames(&s.frames, s.importing.as_deref(), &inner.asyncio_dir)
            )
        })
        .collect();
    if stacks.is_empty() && !process_descheduled {
        stacks.push(if watchdog_starved {
            "# no sample: the loop thread held the GIL for the whole stall, so the sampler could not run (a native call that does not release the GIL)".to_owned()
        } else {
            "# no sample: the block ended before the watchdog looked".to_owned()
        });
    }
    let severity = report.severity.as_str();
    let emit_span = report.emit_span;
    let emit_log = report.emit_log;
    let suppressed = report.suppressed;
    let block = Block {
        lag,
        warn: report.warn_threshold,
        severity,
        gc_time: gc_time.min(lag),
        cpu_time,
        task_name: samples.iter().find_map(|s| s.task_name.clone()),
        importing: samples.iter().find_map(|s| s.importing.clone()),
        location: stacks.last().and_then(|s| innermost_location(s)),
        stacks,
    };
    let span = emit_span.then(|| block.span(suppressed));
    let log = emit_log.then(|| block.log());
    let parent_span_context = samples.iter().find_map(|sample| {
        sample
            .span_context
            .as_ref()
            .map(|context| lk_agents_abi::python::intern_host(context.clone_ref(py)))
    });
    let report = BlockedReport {
        duration: lag,
        started_at: report.started_at,
        warn_threshold: report.warn_threshold,
        severity: report.severity,
        gc_time: block.gc_time,
        cpu_time,
        watchdog_gap,
        process_descheduled,
        task_name: block.task_name,
        importing: block.importing,
        location: block.location,
        stacks: block.stacks,
        parent_span_context,
    };
    LoopMonitorEmission { report, span, log, reported }.into_pyobject(py)
}

/// One reported block, as its span and its log line describe it.
struct Block {
    lag: f64,
    warn: f64,
    severity: &'static str,
    gc_time: f64,
    cpu_time: f64,
    task_name: Option<String>,
    importing: Option<String>,
    location: Option<String>,
    stacks: Vec<String>,
}

impl Block {
    fn span(&self, suppressed: u64) -> LoopMonitorSpan {
        let mut attributes = Map::new();
        attributes.insert("lk.blocking.duration".into(), self.lag.into());
        attributes.insert("lk.blocking.threshold".into(), self.warn.into());
        attributes.insert("lk.blocking.severity".into(), self.severity.into());
        attributes.insert("lk.blocking.gc_time".into(), self.gc_time.into());
        attributes.insert("lk.blocking.cpu_time".into(), self.cpu_time.into());
        if let Some(task) = &self.task_name {
            attributes.insert("lk.blocking.task".into(), task.clone().into());
        }
        if !self.stacks.is_empty() {
            attributes.insert("lk.blocking.stack".into(), self.stacks.join("\n---\n").into());
        }
        if let Some(module) = &self.importing {
            attributes.insert("lk.blocking.import".into(), module.clone().into());
        }
        if suppressed > 0 {
            attributes.insert("lk.blocking.suppressed".into(), suppressed.into());
        }
        let error = (self.severity == "error")
            .then(|| format!("event loop blocked for {:.0}ms", self.lag * 1000.0));
        LoopMonitorSpan { attributes, error }
    }

    fn log(&self) -> LoopMonitorLog {
        let round4 = |v: f64| Value::from((v * 1e4).round() / 1e4);
        let mut extra = Map::new();
        extra.insert("duration".into(), round4(self.lag));
        extra.insert("threshold".into(), self.warn.into());
        extra.insert("gc_time".into(), round4(self.gc_time));
        extra.insert("cpu_time".into(), round4(self.cpu_time));
        if let Some(task) = &self.task_name {
            extra.insert("task".into(), task.clone().into());
        }
        if let Some(stack) = self.stacks.last() {
            extra.insert("stack".into(), stack.clone().into());
        }
        if let Some(module) = &self.importing {
            extra.insert("import".into(), module.clone().into());
        }
        let at = self.location.as_ref().map(|l| format!(" at {l}")).unwrap_or_default();
        let ms = self.lag * 1000.0;
        let message = match &self.importing {
            Some(module) => format!(
                "event loop blocked for {ms:.0}ms importing {module}{at}; import it at process warm-up (prewarm) instead of on first use"
            ),
            None => format!(
                "event loop blocked for {ms:.0}ms{at}; synchronous work on the agent loop delays audio and turn handling, move it to a thread or an async client"
            ),
        };
        LoopMonitorLog { message, extra }
    }
}

/// `_format_frames`: the event loop machinery dropped (never the innermost frame), cut through the
/// innermost job-runner frame, runs of import machinery collapsed, the last 20 entries kept.
fn format_frames(frames: &[Frame], importing: Option<&str>, asyncio_dir: &str) -> String {
    let last = frames.len().saturating_sub(1);
    let mut trimmed: Vec<&Frame> = frames
        .iter()
        .enumerate()
        .filter(|(i, f)| *i == last || !f.filename.starts_with(asyncio_dir))
        .map(|(_, f)| f)
        .collect();
    let runner = trimmed.iter().rposition(|f| f.filename.contains(&ipc_fragment()));
    if let Some(r) = runner
        && r + 1 < trimmed.len()
    {
        trimmed.drain(..=r);
    }
    let import_line = |run: usize| {
        let what = importing.map(|m| format!(", importing {m}")).unwrap_or_default();
        format!("  [import system: {run} frames{what}]\n")
    };
    let mut entries = Vec::new();
    let mut run = 0;
    let n = trimmed.len();
    for (i, f) in trimmed.iter().enumerate() {
        if f.filename.starts_with("<frozen importlib") && i + 1 != n {
            run += 1;
            continue;
        }
        if run > 0 {
            entries.push(import_line(run));
            run = 0;
        }
        let lineno = f.lineno.map_or_else(|| "None".to_owned(), |l| l.to_string());
        entries.push(format!("  File \"{}\", line {}, in {}\n", f.filename, lineno, f.name));
    }
    if run > 0 {
        entries.push(import_line(run));
    }
    let start = entries.len().saturating_sub(MAX_STACK_FRAMES);
    entries[start..].concat().trim_end().to_owned()
}

fn ipc_fragment() -> String {
    let sep = std::path::MAIN_SEPARATOR;
    format!("{sep}livekit{sep}agents{sep}ipc{sep}")
}

/// `_innermost_location`: the innermost frame that is not the import machinery.
fn innermost_location(stack: &str) -> Option<String> {
    stack
        .lines()
        .filter(|l| l.trim_start().starts_with("File ") && !l.contains("<frozen importlib"))
        .last()
        .map(|l| l.trim().trim_start_matches("File ").to_owned())
}

impl Inner {
    fn watchdog_check(&self) -> PyResult<()> {
        let request = {
            let mut st = self.lock();
            if st.closed {
                return Ok(());
            }
            let request = st.policy.sample_due(self.now());
            if let Some(request) = request
                && st
                    .incident
                    .as_ref()
                    .is_none_or(|incident| incident.tick_seq != request.tick_sequence)
            {
                st.incident =
                    Some(Incident { tick_seq: request.tick_sequence, samples: Vec::new() });
            }
            request
        };
        let Some(request) = request else { return Ok(()) };
        let Some(sample) = Python::try_attach(|py| self.sample_loop_thread(py, request.lag)) else {
            return Ok(());
        };
        let sample = sample?;
        let mut st = self.lock();
        if let Some(incident) = st.incident.as_mut()
            && incident.tick_seq == request.tick_sequence
        {
            incident.samples.push(sample);
        }
        Ok(())
    }

    /// `_sample_loop_thread`: the blocked task, the span it was in, and the loop thread's stack,
    /// read without source lookups (the loop thread is waiting for the GIL this holds).
    fn sample_loop_thread(&self, py: Python<'_>, lag: f64) -> PyResult<StackSample> {
        let asyncio = py.import("asyncio")?;
        let mut task_name = None;
        let mut span_context = None;
        let task = asyncio
            .call_method1("current_task", (self.event_loop.bind(py),))
            .ok()
            .filter(|t| !t.is_none());
        if let Some(t) = &task {
            task_name = t.call_method0("get_name").and_then(|n| n.extract::<String>()).ok();
            // `Task.get_context()` is 3.12+; older interpreters fall back to the session root
            if let Ok(ctx) = t.call_method0("get_context")
                && let Ok(span) = self.span_of.bind(py).call1((ctx,))
                && !span.is_none()
            {
                span_context = Some(span.unbind());
            }
        }

        let sys = py.import("sys")?;
        let current = sys.call_method0("_current_frames")?.cast_into::<PyDict>()?;
        let own: u64 = py.import("threading")?.call_method0("get_ident")?.extract()?;
        // this thread's entry holds the dict in a local: the cycle would keep every thread's
        // frames alive until a gc pass
        current.del_item(own).ok();
        let ident = self.lock().loop_thread_ident;
        let mut frame = ident.and_then(|i| current.get_item(i).ok().flatten());
        if frame.is_none()
            && let Some(t) = &task
        {
            // the loop moved to another thread: find the stack holding the task's coroutine
            let coro_frame = t.call_method0("get_coro").and_then(|c| c.getattr("cr_frame")).ok();
            if let Some(coro_frame) = coro_frame.filter(|f| !f.is_none()) {
                for (id, candidate) in current.iter() {
                    let mut f = Some(candidate.clone());
                    while let Some(cur) = f {
                        if cur.is(&coro_frame) {
                            frame = Some(candidate.clone());
                            self.lock().loop_thread_ident = id.extract().ok();
                            break;
                        }
                        f = cur.getattr("f_back").ok().filter(|b| !b.is_none());
                    }
                    if frame.is_some() {
                        break;
                    }
                }
            }
        }
        let mut frames = Vec::new();
        let mut importing = None;
        let mut f = frame;
        while let Some(cur) = f {
            if frames.len() == MAX_STACK_FRAMES * 2 {
                break;
            }
            let code = cur.getattr("f_code")?;
            let filename: String = code.getattr("co_filename")?.extract()?;
            let name: String = code.getattr("co_name")?.extract()?;
            if importing.is_none()
                && name == "_find_and_load"
                && filename.starts_with("<frozen importlib")
            {
                importing = cur
                    .getattr("f_locals")
                    .and_then(|l| l.get_item("name"))
                    .and_then(|n| n.extract::<String>())
                    .ok();
            }
            let lineno = cur.getattr("f_lineno").ok().and_then(|l| l.extract::<i64>().ok());
            frames.push(Frame { filename, lineno, name });
            f = cur.getattr("f_back").ok().filter(|b| !b.is_none());
        }
        frames.reverse(); // outermost first, like extract_stack
        Ok(StackSample { lag, task_name, frames, span_context, importing })
    }
}

impl lk_agents_abi::telemetry::loop_monitor::LoopMonitor for LoopMonitor {
    fn new(
        event_loop: HostRef,
        warn_threshold: f64,
        error_threshold: f64,
        tick_interval: Option<f64>,
        name: String,
        emit_spans: bool,
        report: HostRef,
        span_of: HostRef,
    ) -> Result<Self, HostError> {
        Python::attach(|py| {
            let build = || -> PyResult<Self> {
                let options = LoopMonitorOptions::new(
                    warn_threshold,
                    error_threshold,
                    tick_interval,
                    emit_spans,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let asyncio_file: String = py.import("asyncio")?.getattr("__file__")?.extract()?;
                let asyncio_dir = std::path::Path::new(&asyncio_file)
                    .parent()
                    .map(|path| format!("{}{}", path.display(), std::path::MAIN_SEPARATOR))
                    .unwrap_or_default();
                Ok(Self {
                    inner: Arc::new(Inner {
                        event_loop: resolve_key(py, event_loop.id())?,
                        report: resolve_key(py, report.id())?,
                        span_of: resolve_key(py, span_of.id())?,
                        options,
                        name,
                        asyncio_dir,
                        clock: LoopMonitorClock::default(),
                        state: Mutex::new(State {
                            closed: false,
                            started: false,
                            heartbeat: None,
                            policy: LoopMonitorPolicy::new(options),
                            incident: None,
                            loop_thread: None,
                            loop_thread_ident: None,
                            timer: None,
                            gc_callback: None,
                            watchdog: None,
                        }),
                    }),
                })
            };
            build().map_err(|error| capture_host(py, &error))
        })
    }

    fn warn_threshold(&self) -> f64 {
        self.inner.options.warn
    }
    fn error_threshold(&self) -> f64 {
        self.inner.options.error
    }

    fn start(&self) -> Result<(), HostError> {
        Python::attach(|py| {
            let inner = self.inner.clone();
            let schedule = || -> PyResult<()> {
                let callback = PyCFunction::new_closure(py, None, None, move |args, _| {
                    arm(args.py(), &inner)
                })?;
                self.inner.event_loop.bind(py).call_method1("call_soon_threadsafe", (callback,))?;
                Ok(())
            };
            schedule().map_err(|error| capture_host(py, &error))
        })
    }

    fn stop(&self) -> Result<(), HostError> {
        let (timer, gc, watchdog, heartbeat) = {
            let mut state = self.inner.lock();
            if state.closed {
                return Ok(());
            }
            state.closed = true;
            (
                state.timer.take(),
                state.gc_callback.take(),
                state.watchdog.take(),
                state.heartbeat.take(),
            )
        };
        if let Some(watchdog) = watchdog {
            watchdog.stop();
        }
        Python::attach(|py| {
            let cleanup = || -> PyResult<()> {
                let timer_error =
                    timer.and_then(|timer| timer.bind(py).call_method0("cancel").err());
                let gc_error = gc.and_then(|gc| {
                    py.import("gc")
                        .and_then(|module| module.getattr("callbacks"))
                        .and_then(|callbacks| callbacks.call_method1("remove", (gc,)))
                        .err()
                });
                drop(heartbeat);
                match timer_error.or(gc_error) {
                    Some(error) => Err(error),
                    None => Ok(()),
                }
            };
            cleanup().map_err(|error| capture_host(py, &error))
        })
    }
}

impl Drop for LoopMonitor {
    fn drop(&mut self) {
        let watchdog = self.inner.lock().watchdog.take();
        if let Some(watchdog) = watchdog {
            watchdog.stop();
        }
        let _ = Python::try_attach(|_| {
            <Self as lk_agents_abi::telemetry::loop_monitor::LoopMonitor>::stop(self)
        });
    }
}

fn arm(py: Python<'_>, inner: &Arc<Inner>) -> PyResult<()> {
    let ident: u64 = py.import("threading")?.call_method0("get_ident")?.extract()?;
    let thread_cpu: f64 = py.import("time")?.getattr("thread_time")?.call0()?.extract()?;
    {
        let mut state = inner.lock();
        if state.closed || state.started {
            return Ok(());
        }
        state.started = true;
        state.loop_thread = Some(std::thread::current().id());
        state.loop_thread_ident = Some(ident);
        state.policy.arm(inner.now(), thread_cpu);
    }
    let result = arm_callbacks(py, inner);
    if result.is_err() {
        inner.lock().started = false;
    }
    result
}

fn arm_callbacks(py: Python<'_>, inner: &Arc<Inner>) -> PyResult<()> {
    let gc_inner = inner.clone();
    let gc = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<()> {
        let phase: String = args.get_item(0)?.extract()?;
        gc_callback(&gc_inner, &phase);
        Ok(())
    })?
    .into_any()
    .unbind();
    let heartbeat_inner = inner.clone();
    let beat = PyCFunction::new_closure(py, None, None, move |args, _| {
        heartbeat(args.py(), &heartbeat_inner)
    })?
    .into_any()
    .unbind();
    let watchdog = inner.clone();
    let mut previous = inner.now();
    let runtime = lk_agents_runtime::process::handle()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    let monitor = LoopWatchdog::start(
        &runtime,
        format!("livekit-loop-monitor-{}", inner.name),
        inner.options.tick,
        move || {
            let now = watchdog.now();
            watchdog.lock().policy.watchdog_woke(previous, now);
            if let Err(error) = watchdog.watchdog_check() {
                log::error!(target: LOG, "event loop watchdog failed: {error}");
            }
            previous = watchdog.now();
        },
        lk_agents_runtime::telemetry::work::record_incomplete_cleanup,
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    let callbacks = py.import("gc")?.getattr("callbacks")?;
    callbacks.call_method1("append", (gc.bind(py),))?;
    let timer = match inner
        .event_loop
        .bind(py)
        .call_method1("call_later", (inner.options.tick, beat.bind(py)))
    {
        Ok(timer) => timer.unbind(),
        Err(error) => {
            let _ = callbacks.call_method1("remove", (gc.bind(py),));
            return Err(error);
        }
    };
    let mut state = inner.lock();
    if state.closed {
        drop(state);
        let _ = timer.bind(py).call_method0("cancel");
        let _ = callbacks.call_method1("remove", (gc.bind(py),));
        return Ok(());
    }
    state.gc_callback = Some(gc);
    state.heartbeat = Some(Arc::new(beat));
    state.timer = Some(timer);
    state.watchdog = Some(monitor);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(filename: &str, name: &str) -> Frame {
        Frame { filename: filename.into(), lineno: Some(1), name: name.into() }
    }

    // Catches: the import machinery's dozen frames per module level pushing the caller out of
    // the frame budget, or the location naming the machinery instead of the caller.
    #[test]
    fn import_machinery_collapses_and_the_location_is_the_caller() {
        let mut frames = vec![frame("/app/agent.py", "entrypoint")];
        frames.extend((0..30).map(|_| frame("<frozen importlib._bootstrap>", "_find_and_load")));
        frames.push(frame("/site/slowmod.py", "<module>"));
        let out = format_frames(&frames, Some("slowmod"), "/py/asyncio/");
        assert!(out.contains("[import system: 30 frames, importing slowmod]"), "{out}");
        assert_eq!(
            innermost_location(&out).as_deref(),
            Some("\"/site/slowmod.py\", line 1, in <module>")
        );
    }

    // Catches: the runner's frames (the same in every sample) kept above the agent's own code.
    #[test]
    fn the_stack_is_cut_through_the_job_runner() {
        let ipc = format!("/x{}job_proc_lazy_main.py", ipc_fragment());
        let frames = vec![
            frame("/main.py", "main"),
            frame(&ipc, "_run_job"),
            frame("/app/agent.py", "tool"),
        ];
        let out = format_frames(&frames, None, "/py/asyncio/");
        assert!(!out.contains("main.py") && out.contains("agent.py"), "{out}");
        let inside = vec![frame("/main.py", "main"), frame(&ipc, "_run_job")];
        assert!(format_frames(&inside, None, "/py/asyncio/").contains("job_proc_lazy_main"));
    }
}
