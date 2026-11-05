//! Event-loop stalls (docs/internals/telemetry.md "Event-loop stalls"). Each watched asyncio
//! loop's timer beats its [`LoopWatch`]; one detector thread per process notices a beat 100 ms
//! old, reads the loop thread's Python stack with py-spy over the process's own memory (never
//! the GIL) and records the stall once: when the loop beats again, when its watch closes, or at
//! 30 s while it is still stuck. Beats and the detector's checks take no lock; the sampler is the
//! detector thread's own.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, LazyLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use lk_agents_abi::telemetry::record::SpanContext;
use lk_agents_runtime::telemetry::LoopStall;
use parking_lot::Mutex;

/// How often the loop beats, and the detector checks.
const TICK: Duration = Duration::from_millis(20);
/// A beat this old is a stall.
const THRESHOLD: Duration = Duration::from_millis(100);
/// Stack samples of one stall: at detection, then this often, at most `SAMPLES`.
const SAMPLE_EVERY: Duration = Duration::from_millis(100);
const SAMPLES: usize = 5;
/// A stall still going after this long is recorded while it lasts (a deadlocked loop).
const CAP: Duration = Duration::from_secs(30);
/// The recorded stack: its deepest frames, at most this long.
const FRAMES: usize = 20;
const STACK_BYTES: usize = 4096;

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn now() -> u64 {
    u64::try_from(EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// The declared `_core_LoopWatch`: one asyncio loop, beaten by its own timer.
pub struct LoopWatch(Arc<Watched>);

struct Watched {
    /// The loop's OS thread (`threading.get_native_id()`).
    os_thread: u64,
    parent: Option<SpanContext>,
    job: Option<String>,
    last_beat: AtomicU64,
    closed: AtomicBool,
}

impl lk_agents_abi::telemetry::host::LoopWatch for LoopWatch {
    fn new(os_thread: u64, parent: Option<SpanContext>, job: Option<String>) -> Self {
        let watched = Arc::new(Watched {
            os_thread,
            parent: parent.and_then(SpanContext::valid),
            job,
            last_beat: AtomicU64::new(now()),
            closed: AtomicBool::new(false),
        });
        detector().watch(watched.clone());
        LoopWatch(watched)
    }

    fn beat(&self) {
        self.0.last_beat.store(now(), Ordering::Relaxed);
    }

    fn close(&self) {
        self.0.closed.store(true, Ordering::Relaxed);
    }
}

impl Drop for LoopWatch {
    fn drop(&mut self) {
        self.0.closed.store(true, Ordering::Relaxed);
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
    let thread = thread::Builder::new()
        .name("lk-loop-stalls".into())
        .spawn(move || run(&received))
        .map_or_else(|_| thread::current(), |handle| handle.thread().clone());
    let started = Arc::new(Detector { pid, watches, thread });
    *slot = Some(started.clone());
    started
}

fn run(received: &Receiver<Arc<Watched>>) {
    let mut sampler = Sampler::new();
    let mut watches: Vec<Entry> = Vec::new();
    let mut last_wake = now();
    loop {
        thread::park_timeout(TICK);
        watches.extend(received.try_iter().map(Entry::new));
        let t = now();
        // Late itself: the whole process was paused (SIGSTOP, throttling, a suspended machine),
        // not one loop. Every watch starts over from now.
        let paused = t.saturating_sub(last_wake) >= nanos(THRESHOLD + TICK);
        last_wake = t;
        watches.retain_mut(|entry| {
            let (keep, stalled) = entry.check(t, paused, &mut sampler);
            if let Some(stall) = stalled {
                lk_agents_runtime::telemetry::loop_stalled(stall);
            }
            keep
        });
    }
}

struct Entry {
    watched: Arc<Watched>,
    /// No beat before this counts as stale: the watch's start, or the end of a process pause.
    fresh_since: u64,
    stall: Option<Stall>,
}

struct Stall {
    since: u64,
    next_sample: u64,
    samples: Vec<Vec<String>>,
    recorded: bool,
}

impl Entry {
    fn new(watched: Arc<Watched>) -> Self {
        Entry { watched, fresh_since: now(), stall: None }
    }

    /// One check at `t`: whether to keep watching (no once closed), and the stall to record.
    fn check(&mut self, t: u64, paused: bool, sampler: &mut Sampler) -> (bool, Option<LoopStall>) {
        let closed = self.watched.closed.load(Ordering::Relaxed);
        let last = self.watched.last_beat.load(Ordering::Relaxed).max(self.fresh_since);
        let Some(stall) = self.stall.as_mut() else {
            if paused {
                self.fresh_since = t;
            } else if !closed && t.saturating_sub(last) >= nanos(THRESHOLD) {
                self.stall = Some(Stall {
                    since: last,
                    next_sample: t,
                    samples: Vec::new(),
                    recorded: false,
                });
            }
            return (!closed, None);
        };
        let ended = last > stall.since;
        if ended || closed {
            let end = if ended { last.saturating_sub(nanos(TICK)) } else { t };
            let stalled = stall.record(&self.watched, end);
            self.stall = None;
            return (!closed, stalled);
        }
        if stall.samples.len() < SAMPLES && t >= stall.next_sample {
            if let Some(frames) = sampler.sample(self.watched.os_thread) {
                stall.samples.push(frames);
            }
            stall.next_sample = t.saturating_add(nanos(SAMPLE_EVERY));
        }
        let capped = t.saturating_sub(stall.since) >= nanos(CAP);
        (true, if capped { stall.record(&self.watched, t) } else { None })
    }
}

impl Stall {
    /// The stall ending at `end`, the first time only.
    fn record(&mut self, watched: &Watched, end: u64) -> Option<LoopStall> {
        if std::mem::replace(&mut self.recorded, true) {
            return None;
        }
        let (stack, import) = match most_frequent(&self.samples) {
            Some((frames, import)) => (Some(stack_text(frames)), import),
            None => (None, None),
        };
        Some(LoopStall {
            duration: Duration::from_nanos(end.saturating_sub(self.since)),
            parent: watched.parent,
            job: watched.job.clone(),
            stack,
            import,
        })
    }
}

/// The sample seen most often (the latest of equals), with the file being imported, if any.
fn most_frequent(samples: &[Vec<String>]) -> Option<(&[String], Option<String>)> {
    let best = samples
        .iter()
        .rev()
        .max_by_key(|sample| samples.iter().filter(|other| other == sample).count())?;
    Some((best, importing(best)))
}

/// The file a stack is importing: its innermost module body under the import machinery.
fn importing(frames: &[String]) -> Option<String> {
    if !frames.iter().any(|frame| frame.contains("importlib._bootstrap")) {
        return None;
    }
    let module = frames.iter().find(|frame| frame.starts_with("<module> ("))?;
    let file = module.trim_start_matches("<module> (").trim_end_matches(')');
    Some(file.rsplit_once(':').map_or(file, |(file, _)| file).to_owned())
}

fn stack_text(frames: &[String]) -> String {
    let mut text = String::new();
    for frame in frames {
        if text.len() + frame.len() + 1 > STACK_BYTES {
            break;
        }
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(frame);
    }
    text
}

pub(crate) use sampler::Sampler;

/// py-spy over the process's own memory, built at the first stall. Any failure turns stacks
/// off for the process with one warning; stalls are still recorded.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
mod sampler {
    use super::{AssertUnwindSafe, FRAMES, catch_unwind};

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

        #[cfg(test)]
        pub(crate) fn off() -> Self {
            Sampler(State::Off)
        }

        /// The deepest frames of the thread `os_thread`, `function (file:line)`.
        pub(crate) fn sample(&mut self, os_thread: u64) -> Option<Vec<String>> {
            if let State::Unstarted = self.0 {
                self.0 = start();
            }
            let State::Ready(spy) = &mut self.0 else { return None };
            catch_unwind(AssertUnwindSafe(|| frames(spy, os_thread))).ok().flatten()
        }
    }

    fn start() -> State {
        let mut config = py_spy::Config::default();
        config.blocking = py_spy::config::LockingStrategy::NonBlocking;
        // the loop thread is wanted whatever py-spy thinks of it: blocked in a sleep or a read
        // is exactly a stall
        config.include_idle = true;
        let pid = std::process::id() as py_spy::Pid;
        let why = match catch_unwind(AssertUnwindSafe(|| py_spy::PythonSpy::new(pid, &config))) {
            Ok(Ok(spy)) => return State::Ready(Box::new(spy)),
            Ok(Err(error)) => format!("{error:#}"),
            Err(_) => "the sampler panicked".to_owned(),
        };
        log::warn!(target: "livekit.agents", "event-loop stall stacks unavailable: {why}");
        State::Off
    }

    fn frames(spy: &mut py_spy::PythonSpy, os_thread: u64) -> Option<Vec<String>> {
        let traces = spy.get_stack_traces().ok()?;
        let trace = traces.into_iter().find(|trace| trace.os_thread_id == Some(os_thread))?;
        let frame = |frame: &py_spy::Frame| {
            let file = frame.short_filename.as_deref().unwrap_or(&frame.filename);
            format!("{} ({file}:{})", frame.name, frame.line)
        };
        Some(trace.frames.iter().take(FRAMES).map(frame).collect())
    }
}

/// Where py-spy does not run, stalls are recorded without a stack.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod sampler {
    pub(crate) struct Sampler;

    impl Sampler {
        pub(crate) fn new() -> Self {
            Sampler
        }

        #[cfg(test)]
        pub(crate) fn off() -> Self {
            Sampler
        }

        pub(crate) fn sample(&mut self, _os_thread: u64) -> Option<Vec<String>> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn entry(job: &str) -> Entry {
        let watched = Arc::new(Watched {
            os_thread: 1,
            parent: None,
            job: Some(job.into()),
            last_beat: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        });
        Entry { watched, fresh_since: 0, stall: None }
    }

    fn beat(entry: &Entry, at: u64) {
        entry.watched.last_beat.store(at, Ordering::Relaxed);
    }

    // Catches: a stall missed or recorded twice, or its duration off by more than a tick.
    #[test]
    fn a_stall_is_recorded_once_when_the_loop_beats_again() {
        let (mut e, mut s) = (entry("AJ_1"), Sampler::off());
        assert_eq!(e.check(80 * MS, false, &mut s).1.map(|x| x.duration), None);
        let (keep, stalled) = e.check(120 * MS, false, &mut s);
        assert!(keep && stalled.is_none(), "detected, not yet recorded");
        beat(&e, 420 * MS);
        let stalled = e.check(440 * MS, false, &mut s).1.expect("recorded on the next beat");
        assert_eq!(stalled.duration, Duration::from_millis(400));
        assert_eq!(stalled.job.as_deref(), Some("AJ_1"));
        assert!(stalled.stack.is_none(), "no sampler, no stack");
        assert!(e.check(460 * MS, false, &mut s).1.is_none());
    }

    // Catches: a whole-process pause (the detector late too) reported as a loop stall.
    #[test]
    fn a_process_pause_is_not_a_stall() {
        let (mut e, mut s) = (entry("AJ_1"), Sampler::off());
        assert!(e.check(5_000 * MS, true, &mut s).1.is_none());
        assert!(e.stall.is_none(), "watching starts over after the pause");
        assert!(e.check(5_020 * MS, false, &mut s).1.is_none());
    }

    // Catches: a deadlocked loop never reported, or reported again when it finally ends.
    #[test]
    fn a_loop_stuck_past_the_cap_is_reported_while_stuck_and_once() {
        let (mut e, mut s) = (entry("AJ_1"), Sampler::off());
        assert!(e.check(120 * MS, false, &mut s).1.is_none());
        let capped = e.check(30_020 * MS, false, &mut s).1.expect("reported at the cap");
        assert_eq!(capped.duration, Duration::from_millis(30_020));
        beat(&e, 40_000 * MS);
        assert!(e.check(40_020 * MS, false, &mut s).1.is_none(), "once");
        assert!(e.stall.is_none());
    }

    // Catches: a stall in progress lost when its watch closes, or a closed watch kept.
    #[test]
    fn closing_records_the_stall_in_progress_and_stops_watching() {
        let (mut e, mut s) = (entry("AJ_1"), Sampler::off());
        assert!(e.check(150 * MS, false, &mut s).1.is_none());
        e.watched.closed.store(true, Ordering::Relaxed);
        let (keep, stalled) = e.check(300 * MS, false, &mut s);
        assert!(!keep);
        assert_eq!(stalled.map(|x| x.duration), Some(Duration::from_millis(300)));
    }

    // Catches: the stack not the most frequent sample, an import missed, or the 4 KB cap broken.
    #[test]
    fn the_recorded_stack_is_the_most_frequent_sample() {
        let import = vec![
            "_build (slowmod.py:6)".to_owned(),
            "<module> (slowmod.py:10)".to_owned(),
            "_call_with_frames_removed (<frozen importlib._bootstrap>:488)".to_owned(),
        ];
        let other = vec!["busy (app.py:7)".to_owned()];
        let samples = [import.clone(), other, import.clone()];
        let (frames, file) = most_frequent(&samples).expect("a sample");
        assert_eq!(frames, import.as_slice());
        assert_eq!(file.as_deref(), Some("slowmod.py"));
        assert_eq!(stack_text(frames).lines().next(), Some("_build (slowmod.py:6)"));
        assert!(importing(&["busy (app.py:7)".to_owned()]).is_none());
        let long = vec!["f (x.py:1)".repeat(100); 20];
        assert!(stack_text(&long).len() <= STACK_BYTES);
    }
}
