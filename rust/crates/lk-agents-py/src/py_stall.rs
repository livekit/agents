use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, LazyLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant, SystemTime};

use lk_agents_runtime::telemetry;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

// Stacks are read with py-spy over the process's own memory, never under the GIL. Beats and
// checks take no lock; a job's profile is locked once per sample.

/// How often the loop beats, and the detector checks.
const TICK: Duration = Duration::from_millis(20);
/// Stack samples of one stall: at detection, then this often, at most `SAMPLES`.
const SAMPLE_EVERY: Duration = Duration::from_millis(100);
const SAMPLES: usize = 5;
/// The job profiles' sampling period (10 Hz).
const PROFILE_EVERY: Duration = Duration::from_millis(100);
/// A stall still going after this long is recorded while it lasts (a deadlocked loop).
const CAP: Duration = Duration::from_secs(5);
/// How long after a still-stuck report a log line must still be on its way for the report to go
/// to stderr too.
const FORWARDING_GRACE: Duration = Duration::from_millis(200);
/// The recorded stack: its deepest frames, at most this long.
const FRAMES: usize = 20;

/// Real time, never a session's clock: the detector measures how long a loop's thread really
/// stopped, from its own sampler thread.
#[allow(
    clippy::disallowed_methods,
    reason = "the stall detector's real-time boundary: a blocked loop is measured in real time"
)]
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn now() -> u64 {
    u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

fn unix_nanos() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| u64::try_from(since.as_nanos()).unwrap_or(u64::MAX))
}

/// The declared `_core_LoopWatch`: one asyncio loop, beaten by its own timer.
pub struct LoopWatch(Arc<Watched>);

struct Watched {
    /// The loop's OS thread (`threading.get_native_id()`).
    os_thread: u64,
    parent: Option<telemetry::record::SpanContext>,
    job: Option<String>,
    /// The process runs this job alone: its other running threads join the job's profile.
    whole_process: bool,
    last_beat: AtomicU64,
    closed: CancellationToken,
    /// The job's profile, which the recording upload takes.
    profile: Arc<Mutex<telemetry::profile::Profile>>,
}

impl telemetry::host::LoopWatch for LoopWatch {
    fn new(
        os_thread: u64,
        parent: Option<telemetry::record::SpanContext>,
        job: Option<String>,
        whole_process: bool,
    ) -> Self {
        let watched = Arc::new(Watched {
            os_thread,
            parent: parent.and_then(telemetry::record::SpanContext::valid),
            job,
            whole_process,
            last_beat: AtomicU64::new(now()),
            closed: CancellationToken::new(),
            profile: Arc::default(),
        });
        if let Some(job) = &watched.job {
            telemetry::profile::register(
                telemetry::record::JobKey(job.clone()),
                watched.profile.clone(),
            );
        }
        detector().watch(watched.clone());
        LoopWatch(watched)
    }

    fn beat(&self) {
        self.0.last_beat.store(now(), Ordering::Relaxed);
    }

    fn close(&self) {
        self.0.close();
    }
}

impl Drop for LoopWatch {
    fn drop(&mut self) {
        self.0.close();
    }
}

impl Watched {
    fn close(&self) {
        self.closed.cancel();
        if let Some(job) = &self.job {
            telemetry::profile::forget(&telemetry::record::JobKey(job.clone()), &self.profile);
        }
    }
}

/// The process's detector thread and how new watches reach it. A forked child starts its own:
/// threads do not survive a fork.
struct Detector {
    pid: u32,
    watches: Sender<Arc<Watched>>,
    thread: Thread,
}

impl Detector {
    fn watch(&self, watched: Arc<Watched>) {
        if self.watches.send(watched).is_ok() {
            self.thread.unpark();
        }
    }
}

static DETECTOR: Mutex<Option<Arc<Detector>>> = Mutex::new(None);

/// Taken when a watch opens, never on a beat or a check.
fn detector() -> Arc<Detector> {
    let mut slot = DETECTOR.lock();
    let pid = std::process::id();
    if let Some(running) = slot.as_ref().filter(|d| d.pid == pid) {
        return running.clone();
    }
    let (watches, received) = channel();
    let thread = match thread::Builder::new()
        .name("lk-loop-stalls".into())
        .spawn(move || run(&received))
    {
        Ok(handle) => handle.thread().clone(),
        Err(error) => {
            // Watches still beat; nothing checks them in this process.
            log::warn!(target: lk_agents_abi::LOG_TARGET, "event-loop stall detection unavailable: {error}");
            thread::current()
        }
    };
    let started = Arc::new(Detector { pid, watches, thread });
    *slot = Some(started.clone());
    started
}

/// The detector's state between ticks.
struct Detection {
    sampler: Sampler,
    watches: Vec<Entry>,
    last_wake: u64,
    next_profile: u64,
    last_profile: Option<u64>,
}

fn run(received: &Receiver<Arc<Watched>>) {
    let t = now();
    let mut detection = Detection {
        sampler: Sampler::new(),
        watches: Vec::new(),
        last_wake: t,
        next_profile: t,
        last_profile: None,
    };
    let mut panicked = false;
    loop {
        // Nothing to watch: sleep until a watch arrives (`watch` unparks).
        if detection.watches.is_empty() {
            thread::park();
        } else {
            thread::park_timeout(TICK);
        }
        detection.watches.extend(received.try_iter().map(Entry::new));
        // A panic in one tick (a sample, a record) must not end detection for the process.
        if catch_unwind(AssertUnwindSafe(|| detection.tick())).is_err()
            && !std::mem::replace(&mut panicked, true)
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "event-loop stall detection panicked; it goes on");
        }
    }
}

impl Detection {
    fn tick(&mut self) {
        let t = now();
        // Late itself: the whole process was paused (SIGSTOP, throttling, a suspended machine),
        // not one loop. Every watch starts over from now.
        let paused =
            t.saturating_sub(self.last_wake) >= nanos(telemetry::policy::BLOCKED_WARNING + TICK);
        self.last_wake = t;
        let profile = !self.watches.is_empty() && t >= self.next_profile;
        let stalled = self.watches.iter().any(|entry| entry.wants_sample(t));
        // One process sample per tick, shared by the profiles and every stalled loop.
        let threads = if profile || stalled { self.sampler.sample_all() } else { None };
        if profile {
            let wall =
                self.last_profile.map_or(nanos(PROFILE_EVERY), |last| t.saturating_sub(last));
            self.last_profile = Some(t);
            self.next_profile = t.saturating_add(nanos(PROFILE_EVERY));
            if let Some(threads) = &threads {
                add_samples(&self.watches, threads, unix_nanos(), i64::try_from(wall).unwrap_or(0));
            }
        }
        self.watches.retain_mut(|entry| {
            let checked = entry.check(t, paused, threads.as_deref());
            if let Some(stall) = checked.record {
                telemetry::loop_stalled(stall);
            }
            if let Some(message) = checked.stderr {
                use std::io::Write as _;
                let _ = std::io::stderr().write_all(format!("{message}\n").as_bytes());
            }
            checked.keep
        });
    }
}

/// One profile sample of every watched job: its loop thread always (wall clock: its waiting is
/// what a slow turn is made of); when the process runs the job alone, its other Python threads
/// too while they run (an idle pool thread would only add waiting). With several jobs in one
/// process, other threads cannot be told apart by job.
fn add_samples(watches: &[Entry], threads: &[ThreadStack], at: u64, wall: i64) {
    for entry in watches {
        let mut profile = entry.watched.profile.lock();
        for thread in threads {
            let loop_thread = thread.os_thread == Some(entry.watched.os_thread);
            if loop_thread || (entry.watched.whole_process && thread.active) {
                let frames = thread.frames.iter().map(|frame| telemetry::profile::Frame {
                    function: &frame.function,
                    file: &frame.file,
                    line: frame.line,
                });
                profile.add(at, wall, &thread.name, thread.active, frames);
            }
        }
    }
}

/// The deepest frames of the thread `os_thread`.
fn frames_of(
    threads: &[ThreadStack],
    os_thread: u64,
) -> Option<Vec<telemetry::record::StackFrame>> {
    let thread = threads.iter().find(|thread| thread.os_thread == Some(os_thread))?;
    Some(thread.frames.iter().take(FRAMES).cloned().collect())
}

struct Entry {
    watched: Arc<Watched>,
    /// No beat before this counts as stale: the watch's start, or the end of a process pause.
    fresh_since: u64,
    stall: Option<Stall>,
}

/// What one check found.
struct Checked {
    /// No once the watch is closed and done with.
    keep: bool,
    record: Option<telemetry::LoopStall>,
    /// The still-stuck report, for stderr: the core's log lines are not reaching Python's
    /// `logging` (one has been on its way since the report: waiting for the GIL, or in a handler).
    stderr: Option<String>,
}

struct Stall {
    since: u64,
    /// The still-stuck report and when to see whether its log line was forwarded.
    unforwarded: Option<(u64, String)>,
    next_sample: u64,
    samples: Vec<Vec<telemetry::record::StackFrame>>,
    recorded: bool,
}

impl Entry {
    fn new(watched: Arc<Watched>) -> Self {
        Entry { watched, fresh_since: now(), stall: None }
    }

    /// A stall in progress whose next stack sample is due.
    fn wants_sample(&self, t: u64) -> bool {
        self.stall.as_ref().is_some_and(|s| s.samples.len() < SAMPLES && t >= s.next_sample)
    }

    /// One check at `t`, with this tick's process sample, if one was taken.
    fn check(&mut self, t: u64, paused: bool, threads: Option<&[ThreadStack]>) -> Checked {
        let closed = self.watched.closed.is_cancelled();
        let last = self.watched.last_beat.load(Ordering::Relaxed).max(self.fresh_since);
        let mut checked = Checked { keep: !closed, record: None, stderr: None };
        if paused {
            // The process was paused: whatever the loop looked like is the pause, not a stall.
            self.fresh_since = t;
            self.stall = None;
            return checked;
        }
        let Some(stall) = self.stall.as_mut() else {
            if !closed && t.saturating_sub(last) >= nanos(telemetry::policy::BLOCKED_WARNING) {
                self.stall = Some(Stall {
                    since: last,
                    unforwarded: None,
                    next_sample: t,
                    samples: Vec::new(),
                    recorded: false,
                });
            }
            return checked;
        };
        let ended = last > stall.since;
        if ended || closed {
            let end = if ended { last.saturating_sub(nanos(TICK)) } else { t };
            checked.record = stall.record(&self.watched, end, telemetry::record::StallState::Ended);
            self.stall = None;
            return checked;
        }
        if stall.samples.len() < SAMPLES && t >= stall.next_sample {
            if let Some(frames) =
                threads.and_then(|threads| frames_of(threads, self.watched.os_thread))
            {
                stall.samples.push(frames);
            }
            stall.next_sample = t.saturating_add(nanos(SAMPLE_EVERY));
        }
        if t.saturating_sub(stall.since) >= nanos(CAP) {
            checked.record = stall.record(&self.watched, t, telemetry::record::StallState::Ongoing);
            if let Some(report) = &checked.record {
                let message = telemetry::stall_message(
                    "livekit.agents: event loop still blocked after",
                    report.duration,
                    report.stack.as_deref(),
                    report.import.as_deref(),
                );
                stall.unforwarded = Some((t.saturating_add(nanos(FORWARDING_GRACE)), message));
            }
        }
        if let Some((_, message)) = stall.unforwarded.take_if(|(due, _)| t >= *due) {
            let blocked = crate::py_logging::forwarding_blocked(FORWARDING_GRACE);
            checked.stderr = blocked.then_some(message);
        }
        checked
    }
}

impl Stall {
    /// The stall ending at `end`, the first time only.
    fn record(
        &mut self,
        watched: &Watched,
        end: u64,
        state: telemetry::record::StallState,
    ) -> Option<telemetry::LoopStall> {
        if std::mem::replace(&mut self.recorded, true) {
            return None;
        }
        let (stack, import) = match most_frequent(&self.samples) {
            Some((frames, import)) => (Some(frames.to_vec()), import),
            None => (None, None),
        };
        Some(telemetry::LoopStall {
            duration: Duration::from_nanos(end.saturating_sub(self.since)),
            parent: watched.parent,
            job: watched.job.clone(),
            stack,
            import,
            state,
        })
    }
}

/// The sample seen most often (the latest of equals), with the file being imported, if any.
fn most_frequent(
    samples: &[Vec<telemetry::record::StackFrame>],
) -> Option<(&[telemetry::record::StackFrame], Option<String>)> {
    let best = samples
        .iter()
        .rev()
        .max_by_key(|sample| samples.iter().filter(|other| other == sample).count())?;
    Some((best, importing(best)))
}

/// The file a stack is importing: its innermost module body under the import machinery.
fn importing(frames: &[telemetry::record::StackFrame]) -> Option<String> {
    if !frames.iter().any(|frame| {
        frame.function.contains("importlib._bootstrap")
            || frame.file.contains("importlib._bootstrap")
    }) {
        return None;
    }
    let module = frames.iter().find(|frame| frame.function == "<module>")?;
    Some(module.file.clone())
}

pub(crate) use sampler::Sampler;

/// One Python thread's stack at a sample.
pub(crate) struct ThreadStack {
    pub(crate) os_thread: Option<u64>,
    pub(crate) name: String,
    /// Running, not waiting (py-spy's judgement from the thread's OS state).
    pub(crate) active: bool,
    /// Innermost first.
    pub(crate) frames: Vec<telemetry::record::StackFrame>,
}

/// A thread waiting in Python's own idle calls: the loop's selector, a `threading` wait, a queue
/// `get`, a thread pool's idle worker. py-spy's own idle test misses them in-process.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn idle_frame(frame: &py_spy::Frame) -> bool {
    let function = frame.name.rsplit('.').next().unwrap_or(&frame.name);
    let file = frame.filename.as_str();
    (file.ends_with("selectors.py") && matches!(function, "select" | "poll"))
        || (file.ends_with("threading.py") && matches!(function, "wait" | "_wait_for_tstate_lock"))
        || (file.ends_with("queue.py") && function == "get")
        // a pool worker between work items: its queue `get` is C, so `_worker` is innermost
        || (file.ends_with("futures/thread.py") && function == "_worker")
}

/// py-spy over the process's own memory, built at the first sample. Any failure turns stacks
/// off for the process with one warning; stalls are still recorded, profiles stay empty.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
mod sampler {
    use super::{AssertUnwindSafe, ThreadStack, catch_unwind, telemetry};

    pub(crate) struct Sampler(State);

    enum State {
        Unstarted,
        Ready(Box<py_spy::PythonSpy>),
        Off,
    }

    impl Sampler {
        pub(crate) fn new() -> Self {
            Sampler(State::Unstarted)
        }

        /// Every Python thread's stack.
        pub(crate) fn sample_all(&mut self) -> Option<Vec<ThreadStack>> {
            if let State::Unstarted = self.0 {
                self.0 = start();
            }
            let State::Ready(spy) = &mut self.0 else { return None };
            catch_unwind(AssertUnwindSafe(|| threads(spy))).ok().flatten()
        }
    }

    fn start() -> State {
        let mut config = py_spy::Config::default();
        config.blocking = py_spy::config::LockingStrategy::NonBlocking;
        // a loop thread is wanted whatever py-spy thinks of it: blocked in a sleep or a read is
        // exactly a stall, and a wall-clock profile counts waiting too
        config.include_idle = true;
        let pid = std::process::id() as py_spy::Pid;
        let why = match catch_unwind(AssertUnwindSafe(|| py_spy::PythonSpy::new(pid, &config))) {
            Ok(Ok(spy)) => return State::Ready(Box::new(spy)),
            Ok(Err(error)) => format!("{error:#}"),
            Err(_) => "the sampler panicked".to_owned(),
        };
        log::warn!(target: lk_agents_abi::LOG_TARGET, "event-loop stall stacks unavailable: {why}");
        State::Off
    }

    fn threads(spy: &mut py_spy::PythonSpy) -> Option<Vec<ThreadStack>> {
        let traces = spy.get_stack_traces().ok()?;
        let thread = |trace: py_spy::StackTrace| ThreadStack {
            os_thread: trace.os_thread_id,
            active: trace.active && !trace.frames.first().is_some_and(super::idle_frame),
            name: trace.thread_name.unwrap_or_else(|| format!("thread {}", trace.thread_id)),
            frames: trace
                .frames
                .into_iter()
                .map(|frame| telemetry::record::StackFrame {
                    // The full path, as Python's tracebacks print it: a short one cannot tell
                    // an installed package (`pydantic/main.py`) from the application's code.
                    file: frame.filename,
                    function: frame.name,
                    line: i64::from(frame.line),
                })
                .collect(),
        };
        Some(traces.into_iter().map(thread).collect())
    }
}

/// Where py-spy does not run, stalls are recorded without a stack and profiles stay empty.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod sampler {
    use super::ThreadStack;

    pub(crate) struct Sampler;

    impl Sampler {
        pub(crate) fn new() -> Self {
            Sampler
        }

        pub(crate) fn sample_all(&mut self) -> Option<Vec<ThreadStack>> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn entry(job: &str) -> Entry {
        entry_on(job, 1, true)
    }

    fn entry_on(job: &str, os_thread: u64, whole_process: bool) -> Entry {
        let watched = Arc::new(Watched {
            os_thread,
            parent: None,
            job: Some(job.into()),
            whole_process,
            last_beat: AtomicU64::new(0),
            closed: CancellationToken::new(),
            profile: Arc::default(),
        });
        Entry { watched, fresh_since: 0, stall: None }
    }

    fn beat(entry: &Entry, at: u64) {
        entry.watched.last_beat.store(at, Ordering::Relaxed);
    }

    fn thread(os_thread: u64, name: &str, active: bool) -> ThreadStack {
        let frame = telemetry::record::StackFrame {
            function: name.to_owned(),
            file: "app.py".into(),
            line: 1,
        };
        ThreadStack { os_thread: Some(os_thread), name: name.into(), active, frames: vec![frame] }
    }

    // Catches: a stall missed or recorded twice, or its duration off by more than a tick.
    #[test]
    fn a_stall_is_recorded_once_when_the_loop_beats_again() {
        let mut e = entry("AJ_1");
        assert!(e.check(80 * MS, false, None).record.is_none());
        let checked = e.check(120 * MS, false, None);
        assert!(checked.keep && checked.record.is_none(), "detected, not yet recorded");
        beat(&e, 420 * MS);
        let stalled = e.check(440 * MS, false, None).record.expect("recorded on the next beat");
        assert_eq!(stalled.duration, Duration::from_millis(400));
        assert_eq!(stalled.job.as_deref(), Some("AJ_1"));
        assert!(
            stalled.state == telemetry::record::StallState::Ended && stalled.stack.is_none(),
            "no sample, no stack"
        );
        assert!(e.check(460 * MS, false, None).record.is_none());
    }

    // Catches: the stack not taken from the tick's shared sample, or from another thread.
    #[test]
    fn a_stall_s_stack_is_its_loop_thread_s_from_the_shared_sample() {
        let mut e = entry("AJ_1");
        e.check(120 * MS, false, None);
        assert!(e.wants_sample(140 * MS));
        let threads = [thread(9, "other", true), thread(1, "busy", true)];
        e.check(140 * MS, false, Some(&threads));
        assert!(!e.wants_sample(160 * MS), "the next sample is 100 ms later");
        beat(&e, 300 * MS);
        let stalled = e.check(320 * MS, false, None).record.unwrap();
        assert_eq!(stalled.stack.unwrap(), threads[1].frames);
    }

    // Catches: a whole-process pause (the detector late too) reported as a loop stall, before
    // or during a stall.
    #[test]
    fn a_process_pause_is_not_a_stall() {
        let mut e = entry("AJ_1");
        assert!(e.check(5_000 * MS, true, None).record.is_none());
        assert!(e.stall.is_none(), "watching starts over after the pause");
        assert!(e.check(5_020 * MS, false, None).record.is_none());

        let mut e = entry("AJ_1");
        e.check(150 * MS, false, None);
        assert!(e.stall.is_some());
        assert!(e.check(9_000 * MS, true, None).record.is_none(), "the pause is not the stall's");
        assert!(e.stall.is_none());
        assert!(e.check(9_020 * MS, false, None).record.is_none());
    }

    // Catches: a stuck loop never reported, reported again when it finally ends, not marked as
    // ongoing (and so rate-limited away), or its report sent to stderr while forwarding works.
    #[test]
    fn a_loop_stuck_past_the_cap_is_reported_while_stuck_and_once() {
        let mut e = entry("AJ_1");
        assert!(e.check(120 * MS, false, None).record.is_none());
        let capped = e.check(5_020 * MS, false, None).record.expect("reported at the cap");
        assert_eq!(capped.duration, Duration::from_millis(5_020));
        assert_eq!(capped.state, telemetry::record::StallState::Ongoing);
        let later = e.check(5_240 * MS, false, None);
        assert!(later.stderr.is_none(), "this test's log forwarding is not waiting for the GIL");
        beat(&e, 9_000 * MS);
        assert!(e.check(9_020 * MS, false, None).record.is_none(), "once");
        assert!(e.stall.is_none());
    }

    // Catches: a stall in progress lost when its watch closes, or a closed watch kept.
    #[test]
    fn closing_records_the_stall_in_progress_and_stops_watching() {
        let mut e = entry("AJ_1");
        assert!(e.check(150 * MS, false, None).record.is_none());
        e.watched.closed.cancel();
        let checked = e.check(300 * MS, false, None);
        assert!(!checked.keep);
        assert_eq!(checked.record.map(|x| x.duration), Some(Duration::from_millis(300)));
    }

    // Catches: the stack not the most frequent sample, or an import missed.
    #[test]
    fn the_recorded_stack_is_the_most_frequent_sample() {
        let import = vec![
            telemetry::record::StackFrame {
                function: "_build".into(),
                file: "slowmod.py".into(),
                line: 6,
            },
            telemetry::record::StackFrame {
                function: "<module>".into(),
                file: "slowmod.py".into(),
                line: 10,
            },
            telemetry::record::StackFrame {
                function: "_call_with_frames_removed".into(),
                file: "<frozen importlib._bootstrap>".into(),
                line: 488,
            },
        ];
        let other = vec![telemetry::record::StackFrame {
            function: "busy".into(),
            file: "app.py".into(),
            line: 7,
        }];
        assert!(importing(&other).is_none());
        let samples = [import.clone(), other, import.clone()];
        let (frames, file) = most_frequent(&samples).expect("a sample");
        assert_eq!(frames, import.as_slice());
        assert_eq!(file.as_deref(), Some("slowmod.py"));
    }

    // Catches: a job's profile missing its loop thread while it waits, filled with idle pool
    // threads, or (a job sharing its process) taking the worker's or another job's threads.
    #[test]
    fn a_profile_has_its_loop_thread_and_running_threads_only() {
        let threads =
            [thread(1, "loop", false), thread(2, "worker", true), thread(3, "idle-pool", false)];
        let alone = [entry_on("AJ_1", 1, true)];
        add_samples(&alone, &threads, 1_000, 100);
        assert_eq!(
            alone[0].watched.profile.lock().samples(),
            2,
            "the waiting loop, the running worker"
        );
        let shared = [entry_on("AJ_1", 1, false), entry_on("AJ_2", 2, false)];
        add_samples(&shared, &threads, 1_000, 100);
        assert_eq!(shared[0].watched.profile.lock().samples(), 1, "its loop thread only");
        assert_eq!(shared[1].watched.profile.lock().samples(), 1);
        let one_of_a_pool = [entry_on("AJ_3", 1, false)];
        add_samples(&one_of_a_pool, &threads, 1_000, 100);
        assert_eq!(
            one_of_a_pool[0].watched.profile.lock().samples(),
            1,
            "not the worker's threads"
        );
    }
}
