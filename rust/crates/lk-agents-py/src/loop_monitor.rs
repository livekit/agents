//! `_core.LoopMonitor`: `telemetry/loop_monitor.py`'s watchdog. A heartbeat on the loop
//! (`loop.call_later`) stamps each tick; a watchdog thread notices a tick that is late, samples
//! the loop thread's stack while the block lasts, and the late tick reports the block, measured.
//! The report carries its span's attributes and its log line, rate limited; the host emits them
//! and records the metric. The thresholds come from the environment, the tick from the warn
//! threshold.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use lk_agents_abi::python::log;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// A second stack sample once a block has lasted this many warn thresholds.
const LATE_SAMPLE_FACTOR: f64 = 10.0;
const MAX_STACK_FRAMES: usize = 20;
const MAX_SPANS_PER_MINUTE: usize = 30;
const MAX_LOGS_PER_MINUTE: usize = 5;
const LOG: &str = "livekit.agents";

const DEFAULT_WARN_THRESHOLD: f64 = 0.1;
const DEFAULT_ERROR_THRESHOLD: f64 = 0.5;
const ENV_WARN_THRESHOLD_MS: &str = "LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS";
const ENV_ERROR_THRESHOLD_MS: &str = "LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS";
/// A block is measured to within one tick, so fewer ticks per warn threshold miss blocks just
/// over it; more cost wake-ups (two per tick, heartbeat and watchdog) for no gain.
const TICKS_PER_WARN_THRESHOLD: f64 = 5.0;
/// The monitor never ticks faster than this: a warn threshold below it is raised to it.
const MIN_TICK_INTERVAL: f64 = 0.02;
const MAX_TICK_INTERVAL: f64 = 0.05;

/// A threshold in seconds from its millisecond variable; an invalid value falls back to the
/// default with a warning rather than disabling the monitor silently.
fn env_seconds(name: &str, default: f64) -> f64 {
    let raw = match std::env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => raw,
        _ => return default,
    };
    match raw.trim().parse::<f64>() {
        Ok(ms) if ms.is_finite() && ms >= 0.0 => ms / 1000.0,
        Ok(_) => {
            log::warn!(target: LOG, "invalid {name}={raw:?}, must be finite and >= 0; using {:.0}ms", default * 1000.0);
            default
        }
        Err(_) => {
            log::warn!(target: LOG, "invalid {name}={raw:?}, expected milliseconds; using {:.0}ms", default * 1000.0);
            default
        }
    }
}

/// `(warn, error)` from the environment; `None` when `LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS=0`
/// disables the monitor.
fn thresholds_from_env() -> Option<(f64, f64)> {
    let warn = env_seconds(ENV_WARN_THRESHOLD_MS, DEFAULT_WARN_THRESHOLD);
    let error = env_seconds(ENV_ERROR_THRESHOLD_MS, DEFAULT_ERROR_THRESHOLD);
    if warn <= 0.0 {
        return None;
    }
    if error < warn {
        log::warn!(
            target: LOG,
            "{ENV_ERROR_THRESHOLD_MS} ({:.0}ms) is below {ENV_WARN_THRESHOLD_MS} ({:.0}ms); using the warn threshold for both",
            error * 1000.0,
            warn * 1000.0
        );
        return Some((warn, warn));
    }
    Some((warn, error))
}

/// A fifth of the warn threshold, within bounds.
fn tick_interval_for(warn: f64) -> f64 {
    (warn / TICKS_PER_WARN_THRESHOLD).clamp(MIN_TICK_INTERVAL, MAX_TICK_INTERVAL)
}

/// At most `limit` events per rolling 60 s window; what was refused is counted.
struct RateLimiter {
    limit: usize,
    events: VecDeque<f64>,
    suppressed: u64,
}

impl RateLimiter {
    fn new(limit: usize) -> Self {
        RateLimiter { limit, events: VecDeque::new(), suppressed: 0 }
    }

    fn allow(&mut self, now: f64) -> bool {
        while self.events.front().is_some_and(|t| *t < now - 60.0) {
            self.events.pop_front();
        }
        if self.events.len() >= self.limit {
            self.suppressed += 1;
            return false;
        }
        self.events.push_back(now);
        true
    }

    fn take_suppressed(&mut self) -> u64 {
        std::mem::take(&mut self.suppressed)
    }
}

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
    late_sampled: bool,
}

struct State {
    closed: bool,
    last_tick_at: f64,
    tick_seq: u64,
    last_thread_cpu: f64,
    gc_started_at: Option<f64>,
    gc_time: f64,
    /// The watchdog's worst late wake-up since the last tick: `(woke_at, gap)`.
    watchdog_late: Option<(f64, f64)>,
    watchdog_last_wake: f64,
    incident: Option<Incident>,
    loop_thread: Option<std::thread::ThreadId>,
    /// `threading.get_ident()` of the loop thread, as `sys._current_frames()` keys it.
    loop_thread_ident: Option<u64>,
    span_limiter: RateLimiter,
    log_limiter: RateLimiter,
    timer: Option<Py<PyAny>>,
    gc_callback: Option<Py<PyAny>>,
}

struct Inner {
    event_loop: Py<PyAny>,
    report: Py<PyAny>,
    /// `span_of(task_context)`: the span current in a task's context, or `None`.
    span_of: Py<PyAny>,
    warn: f64,
    error: f64,
    tick: f64,
    first_sample_lag: f64,
    name: String,
    emit_spans: bool,
    asyncio_dir: String,
    epoch: Instant,
    state: Mutex<State>,
    stop: (Mutex<bool>, Condvar),
    /// Set by the watchdog when it exits, so `stop()` can bound its join.
    exited: (Mutex<bool>, Condvar),
}

impl Inner {
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// How long the watchdog was kept from running during this tick's window: its recorded late
    /// wake-up if it fell inside the window, and the time since its last run if it has not run
    /// since before the window opened.
    fn consume_watchdog_gap(st: &mut State, now: f64, window_start: f64, tick: f64) -> f64 {
        let mut gap = 0.0;
        if let Some((woke_at, late)) = st.watchdog_late.take()
            && woke_at >= window_start
        {
            gap = late;
        }
        if st.watchdog_last_wake < window_start {
            gap = f64::max(gap, now - st.watchdog_last_wake - tick);
        }
        gap.max(0.0)
    }
}

#[pyclass(frozen, name = "LoopMonitor", module = "livekit.agents._core")]
pub struct LoopMonitor {
    inner: Arc<Inner>,
}

/// The heartbeat: `loop.call_later(tick, heartbeat)`, rescheduling itself.
#[pyclass(frozen)]
struct Heartbeat(Arc<Inner>);

/// `gc.callbacks`: GC time spent on the loop thread, attributed to the block it fell in.
#[pyclass(frozen)]
struct GcCallback(Arc<Inner>);

#[pymethods]
impl GcCallback {
    fn __call__(&self, phase: &str, _info: &Bound<'_, PyAny>) {
        let inner = &self.0;
        let now = inner.now();
        let mut st = inner.lock();
        if st.loop_thread != Some(std::thread::current().id()) {
            return;
        }
        match phase {
            "start" => st.gc_started_at = Some(now),
            "stop" => {
                if let Some(started) = st.gc_started_at.take() {
                    st.gc_time += now - started;
                }
            }
            _ => {}
        }
    }
}

#[pymethods]
impl Heartbeat {
    fn __call__(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let inner = Arc::clone(&slf.get().0);
        let thread_cpu: f64 = py.import("time")?.getattr("thread_time")?.call0()?.extract()?;
        let now = inner.now();
        let (lag, gc_time, cpu_time, watchdog_gap, samples) = {
            let mut st = inner.lock();
            if st.closed {
                return Ok(());
            }
            let expected_at = st.last_tick_at + inner.tick;
            let lag = now - expected_at;
            let cpu_time = thread_cpu - st.last_thread_cpu;
            let gc_time = std::mem::take(&mut st.gc_time);
            let blocked_seq = st.tick_seq;
            st.tick_seq += 1;
            st.last_tick_at = now;
            st.last_thread_cpu = thread_cpu;
            let watchdog_gap = Inner::consume_watchdog_gap(&mut st, now, expected_at - inner.tick, inner.tick);
            let incident = st.incident.take();
            if lag < inner.warn {
                (lag, gc_time, cpu_time, watchdog_gap, None)
            } else {
                let samples = incident.filter(|i| i.tick_seq == blocked_seq).map(|i| i.samples).unwrap_or_default();
                (lag, gc_time, cpu_time, watchdog_gap, Some(samples))
            }
        };
        let timer = inner.event_loop.bind(py).call_method1("call_later", (inner.tick, slf))?;
        inner.lock().timer = Some(timer.unbind());
        let Some(samples) = samples else { return Ok(()) };
        let report = build_report(py, &inner, lag, gc_time, cpu_time, watchdog_gap, samples)?;
        inner.report.bind(py).call1((report,))?;
        Ok(())
    }
}

/// `_build_report`, with the rate limiters' verdicts: `emit_span`, `emit_log`, `suppressed`.
fn build_report<'py>(
    py: Python<'py>,
    inner: &Inner,
    lag: f64,
    gc_time: f64,
    cpu_time: f64,
    watchdog_gap: f64,
    samples: Vec<StackSample>,
) -> PyResult<Bound<'py, PyDict>> {
    // the watchdog is an independent thread: if it too woke late by most of the stall, either
    // the process was not being scheduled or the loop thread held the GIL throughout. CPU time
    // tells them apart: a descheduled process burns none, and that is never more than a warning.
    let watchdog_starved = watchdog_gap >= lag * 0.5;
    let process_descheduled = watchdog_starved && cpu_time < lag * 0.5;
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
    let severity = if lag >= inner.error && !process_descheduled { "error" } else { "warning" };
    let now = inner.now();
    let (emit_span, emit_log, suppressed) = {
        let mut st = inner.lock();
        let emit_span = st.span_limiter.allow(now);
        // host descheduling is nothing the developer can fix: no log, and no log quota spent
        let emit_log = !process_descheduled && st.log_limiter.allow(now);
        let suppressed = if emit_span && inner.emit_spans { st.span_limiter.take_suppressed() } else { 0 };
        (emit_span, emit_log, suppressed)
    };
    let block = Block {
        lag,
        warn: inner.warn,
        severity,
        gc_time: gc_time.min(lag),
        cpu_time,
        task_name: samples.iter().find_map(|s| s.task_name.clone()),
        importing: samples.iter().find_map(|s| s.importing.clone()),
        location: stacks.last().and_then(|s| innermost_location(s)),
        stacks,
    };
    let d = PyDict::new(py);
    d.set_item("duration", lag)?;
    // the block started no earlier than the last on-time tick
    d.set_item("started_at", lk_agents_abi::unix_now() - lag)?;
    d.set_item("warn_threshold", inner.warn)?;
    d.set_item("severity", severity)?;
    d.set_item("gc_time", block.gc_time)?;
    d.set_item("cpu_time", cpu_time)?;
    d.set_item("watchdog_gap", watchdog_gap)?;
    d.set_item("process_descheduled", process_descheduled)?;
    d.set_item("task_name", &block.task_name)?;
    d.set_item("importing", &block.importing)?;
    d.set_item("location", &block.location)?;
    d.set_item("stacks", PyList::new(py, &block.stacks)?)?;
    d.set_item("parent_span_context", samples.iter().find_map(|s| s.span_context.as_ref().map(|c| c.clone_ref(py))))?;
    // what the rate limiters let through: the span's `(attributes, error status)` (none on a
    // loop without spans) and the log's `(message, extra)`
    let span = if emit_span && inner.emit_spans { Some(block.span(py, suppressed)?) } else { None };
    d.set_item("span", span)?;
    d.set_item("log", if emit_log { Some(block.log(py)?) } else { None })?;
    d.set_item("reported", emit_span || emit_log)?;
    Ok(d)
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
    /// The `event_loop_blocked` span's attributes, and its error status when it is one.
    fn span<'py>(&self, py: Python<'py>, suppressed: u64) -> PyResult<(Bound<'py, PyDict>, Option<String>)> {
        let a = PyDict::new(py);
        a.set_item("lk.blocking.duration", self.lag)?;
        a.set_item("lk.blocking.threshold", self.warn)?;
        a.set_item("lk.blocking.severity", self.severity)?;
        a.set_item("lk.blocking.gc_time", self.gc_time)?;
        a.set_item("lk.blocking.cpu_time", self.cpu_time)?;
        if let Some(task) = &self.task_name {
            a.set_item("lk.blocking.task", task)?;
        }
        if !self.stacks.is_empty() {
            a.set_item("lk.blocking.stack", self.stacks.join("\n---\n"))?;
        }
        if let Some(module) = &self.importing {
            a.set_item("lk.blocking.import", module)?;
        }
        if suppressed > 0 {
            a.set_item("lk.blocking.suppressed", suppressed)?;
        }
        let status = (self.severity == "error").then(|| format!("event loop blocked for {:.0}ms", self.lag * 1000.0));
        Ok((a, status))
    }

    /// The warning's message and `extra`: a lazy import names the module, to preload it.
    fn log<'py>(&self, py: Python<'py>) -> PyResult<(String, Bound<'py, PyDict>)> {
        let round4 = |v: f64| (v * 1e4).round() / 1e4;
        let extra = PyDict::new(py);
        extra.set_item("duration", round4(self.lag))?;
        extra.set_item("threshold", self.warn)?;
        extra.set_item("gc_time", round4(self.gc_time))?;
        extra.set_item("cpu_time", round4(self.cpu_time))?;
        if let Some(task) = &self.task_name {
            extra.set_item("task", task)?;
        }
        if let Some(stack) = self.stacks.last() {
            extra.set_item("stack", stack)?;
        }
        if let Some(module) = &self.importing {
            extra.set_item("import", module)?;
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
        Ok((message, extra))
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
    fn watchdog_main(self: Arc<Self>) {
        loop {
            let before = self.now();
            {
                let (lock, cv) = &self.stop;
                let stopped = lock.lock().unwrap_or_else(|p| p.into_inner());
                let (stopped, _) = cv
                    .wait_timeout_while(stopped, Duration::from_secs_f64(self.tick), |s| !*s)
                    .unwrap_or_else(|p| p.into_inner());
                if *stopped {
                    break;
                }
            }
            let woke_at = self.now();
            let gap = woke_at - before - self.tick;
            {
                let mut st = self.lock();
                if st.watchdog_late.is_none_or(|(_, late)| gap > late) {
                    st.watchdog_late = Some((woke_at, gap));
                }
                st.watchdog_last_wake = woke_at;
            }
            if let Err(e) = self.watchdog_check() {
                log::error!(target: LOG, "event loop watchdog failed: {e}");
            }
        }
        let (lock, cv) = &self.exited;
        *lock.lock().unwrap_or_else(|p| p.into_inner()) = true;
        cv.notify_all();
    }

    fn watchdog_check(&self) -> PyResult<()> {
        let (seq, lag) = {
            let st = self.lock();
            (st.tick_seq, self.now() - (st.last_tick_at + self.tick))
        };
        if lag < self.first_sample_lag {
            return Ok(());
        }
        let (want_first, want_late) = {
            let mut st = self.lock();
            if st.incident.as_ref().is_none_or(|i| i.tick_seq != seq) {
                st.incident = Some(Incident { tick_seq: seq, samples: Vec::new(), late_sampled: false });
            }
            let incident = st.incident.as_ref().expect("just set");
            (incident.samples.is_empty(), !incident.late_sampled && lag >= self.warn * LATE_SAMPLE_FACTOR)
        };
        if !(want_first || want_late) {
            return Ok(());
        }
        // `None` once the interpreter is finalizing: nothing left to report to.
        let Some(sample) = Python::try_attach(|py| self.sample_loop_thread(py, lag)) else { return Ok(()) };
        let sample = sample?;
        let mut st = self.lock();
        if let Some(incident) = st.incident.as_mut()
            && incident.tick_seq == seq
        {
            incident.samples.push(sample);
            if want_late {
                incident.late_sampled = true;
            }
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
            let code = cur.getattr("f_code")?;
            let filename: String = code.getattr("co_filename")?.extract()?;
            let name: String = code.getattr("co_name")?.extract()?;
            if importing.is_none() && name == "_find_and_load" && filename.starts_with("<frozen importlib") {
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

#[pymethods]
impl LoopMonitor {
    /// `report(dict)` runs on the loop thread for each block past `warn_threshold`; `span_of`
    /// reads the span current in a task's context. Thresholds in seconds. Without
    /// `tick_interval`, the tick is a fifth of the warn threshold within bounds, and a warn
    /// threshold below the smallest tick is raised to it.
    #[new]
    #[pyo3(signature = (event_loop, *, warn_threshold, error_threshold, tick_interval, name, emit_spans, report, span_of))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        event_loop: Py<PyAny>,
        mut warn_threshold: f64,
        mut error_threshold: f64,
        tick_interval: Option<f64>,
        name: String,
        emit_spans: bool,
        report: Py<PyAny>,
        span_of: Py<PyAny>,
    ) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;
        let tick_interval = match tick_interval {
            Some(tick) => tick,
            None => {
                if warn_threshold < MIN_TICK_INTERVAL {
                    log::warn!(
                        target: LOG,
                        "loop monitor warn threshold {:.0}ms raised to {:.0}ms, the smallest it measures",
                        warn_threshold * 1000.0,
                        MIN_TICK_INTERVAL * 1000.0
                    );
                    warn_threshold = MIN_TICK_INTERVAL;
                    error_threshold = error_threshold.max(MIN_TICK_INTERVAL);
                }
                tick_interval_for(warn_threshold)
            }
        };
        if warn_threshold <= 0.0 {
            return Err(PyValueError::new_err("warn_threshold must be > 0"));
        }
        if error_threshold < warn_threshold {
            return Err(PyValueError::new_err("error_threshold must be >= warn_threshold"));
        }
        if tick_interval <= 0.0 || tick_interval > warn_threshold {
            return Err(PyValueError::new_err("tick_interval must be > 0 and <= warn_threshold"));
        }
        let asyncio_file: String = py.import("asyncio")?.getattr("__file__")?.extract()?;
        let asyncio_dir = std::path::Path::new(&asyncio_file)
            .parent()
            .map(|p| format!("{}{}", p.display(), std::path::MAIN_SEPARATOR))
            .unwrap_or_default();
        Ok(LoopMonitor {
            inner: Arc::new(Inner {
                event_loop,
                report,
                span_of,
                warn: warn_threshold,
                error: error_threshold,
                tick: tick_interval,
                // sampled one interval before the threshold, so a block that only just crosses
                // it has a sample
                first_sample_lag: f64::max(warn_threshold - tick_interval, tick_interval),
                name,
                emit_spans,
                asyncio_dir,
                epoch: Instant::now(),
                state: Mutex::new(State {
                    closed: false,
                    last_tick_at: 0.0,
                    tick_seq: 0,
                    last_thread_cpu: 0.0,
                    gc_started_at: None,
                    gc_time: 0.0,
                    watchdog_late: None,
                    watchdog_last_wake: 0.0,
                    incident: None,
                    loop_thread: None,
                    loop_thread_ident: None,
                    span_limiter: RateLimiter::new(MAX_SPANS_PER_MINUTE),
                    log_limiter: RateLimiter::new(MAX_LOGS_PER_MINUTE),
                    timer: None,
                    gc_callback: None,
                }),
                stop: (Mutex::new(false), Condvar::new()),
                exited: (Mutex::new(false), Condvar::new()),
            }),
        })
    }

    /// `(warn, error)` from `LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS` and `_ERROR_MS`, `None` when
    /// the monitor is disabled.
    #[staticmethod]
    fn thresholds_from_env() -> Option<(f64, f64)> {
        thresholds_from_env()
    }

    #[getter]
    fn warn_threshold(&self) -> f64 {
        self.inner.warn
    }

    #[getter]
    fn error_threshold(&self) -> f64 {
        self.inner.error
    }

    /// Arm the heartbeat on the loop thread. Safe to call from any thread.
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        let arm = Py::new(py, Arm(Arc::clone(&self.inner)))?;
        self.inner.event_loop.bind(py).call_method1("call_soon_threadsafe", (arm,))?;
        Ok(())
    }

    /// Stop the heartbeat and the watchdog. Idempotent.
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        let (timer, gc_cb) = {
            let mut st = inner.lock();
            if st.closed {
                return Ok(());
            }
            st.closed = true;
            (st.timer.take(), st.gc_callback.take())
        };
        {
            let (lock, cv) = &inner.stop;
            *lock.lock().unwrap_or_else(|p| p.into_inner()) = true;
            cv.notify_all();
        }
        if let Some(t) = timer {
            t.bind(py).call_method0("cancel")?;
        }
        if let Some(cb) = gc_cb {
            let _ = py.import("gc")?.getattr("callbacks")?.call_method1("remove", (cb,));
        }
        // the watchdog may be waiting for the GIL to sample: wait for it without holding it
        let inner = Arc::clone(inner);
        py.detach(move || {
            let (lock, cv) = &inner.exited;
            let exited = lock.lock().unwrap_or_else(|p| p.into_inner());
            let _ = cv.wait_timeout_while(exited, Duration::from_secs(1), |e| !*e);
        });
        Ok(())
    }
}

/// `_arm`, on the loop thread: the first tick, the GC hook and the watchdog.
#[pyclass(frozen)]
struct Arm(Arc<Inner>);

#[pymethods]
impl Arm {
    fn __call__(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.0;
        let ident: u64 = py.import("threading")?.call_method0("get_ident")?.extract()?;
        let thread_cpu: f64 = py.import("time")?.getattr("thread_time")?.call0()?.extract()?;
        let gc_cb = Py::new(py, GcCallback(Arc::clone(inner)))?.into_any();
        let heartbeat = Py::new(py, Heartbeat(Arc::clone(inner)))?;
        {
            let mut st = inner.lock();
            if st.closed {
                return Ok(());
            }
            st.loop_thread = Some(std::thread::current().id());
            st.loop_thread_ident = Some(ident);
            st.last_tick_at = inner.now();
            st.watchdog_last_wake = inner.now();
            st.last_thread_cpu = thread_cpu;
            st.gc_callback = Some(gc_cb.clone_ref(py));
        }
        py.import("gc")?.getattr("callbacks")?.call_method1("append", (gc_cb,))?;
        let timer = inner.event_loop.bind(py).call_method1("call_later", (inner.tick, heartbeat))?;
        inner.lock().timer = Some(timer.unbind());
        let watchdog = Arc::clone(inner);
        std::thread::Builder::new()
            .name(format!("livekit-loop-monitor-{}", inner.name))
            .spawn(move || watchdog.watchdog_main())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

lk_agents_abi::register_host!(class LoopMonitor);

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
        assert_eq!(innermost_location(&out).as_deref(), Some("\"/site/slowmod.py\", line 1, in <module>"));
    }

    // Catches: the runner's frames (the same in every sample) kept above the agent's own code.
    #[test]
    fn the_stack_is_cut_through_the_job_runner() {
        let ipc = format!("/x{}job_proc_lazy_main.py", ipc_fragment());
        let frames = vec![frame("/main.py", "main"), frame(&ipc, "_run_job"), frame("/app/agent.py", "tool")];
        let out = format_frames(&frames, None, "/py/asyncio/");
        assert!(!out.contains("main.py") && out.contains("agent.py"), "{out}");
        let inside = vec![frame("/main.py", "main"), frame(&ipc, "_run_job")];
        assert!(format_frames(&inside, None, "/py/asyncio/").contains("job_proc_lazy_main"));
    }

    // Catches: a tick too coarse to measure blocks near the threshold, or one waking too often.
    #[test]
    fn the_tick_is_a_fifth_of_the_warn_threshold_within_bounds() {
        assert!((tick_interval_for(0.1) - 0.02).abs() < 1e-12);
        assert!((tick_interval_for(0.25) - 0.05).abs() < 1e-12);
        assert!((tick_interval_for(1.0) - 0.05).abs() < 1e-12);
        assert!((tick_interval_for(0.01) - 0.02).abs() < 1e-12);
    }

    // Catches: the limiter forgetting what it refused, or refusing after the window rolled.
    #[test]
    fn the_rate_limiter_counts_what_it_refused() {
        let mut r = RateLimiter::new(2);
        assert!(r.allow(0.0) && r.allow(1.0));
        assert!(!r.allow(2.0));
        assert_eq!(r.take_suppressed(), 1);
        assert!(r.allow(61.5), "the first event left the window");
    }
}
