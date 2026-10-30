//! The core's `log` lines into Python's `logging`, without the GIL at the call site. A line is
//! queued (bounded, `try_send`) and one drain thread per process forwards the queue in order; a
//! full queue drops lines and the drain logs how many. So a line logged under a lock, by any crate,
//! cannot deadlock against a Python thread waiting for that lock.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use lk_agents_runtime::net::lock::Mutex;
use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const QUEUE: usize = 4096;
const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

struct Line {
    level: Level,
    target: String,
    message: String,
    file: Option<String>,
    line: Option<u32>,
    key_values: Vec<(String, String)>,
    at: SystemTime,
}

enum Entry {
    Line(Line),
    Flush(SyncSender<()>),
}

/// The queue's sender, for the process that started its drain (a fork starts its own).
struct Queue {
    pid: u32,
    sender: SyncSender<Entry>,
}

struct Forwarder {
    queue: Mutex<Option<Queue>>,
    dropped: AtomicU64,
    /// Each logger's effective level, as the drain last read it: `enabled` reads it without the
    /// GIL. A target not read yet is enabled.
    levels: Mutex<BTreeMap<String, LevelFilter>>,
}

static FORWARDER: OnceLock<Forwarder> = OnceLock::new();

fn forwarder() -> &'static Forwarder {
    FORWARDER.get_or_init(|| Forwarder {
        queue: Mutex::new(None),
        dropped: AtomicU64::new(0),
        levels: Mutex::new(BTreeMap::new()),
    })
}

/// Routes the core's `log` lines to `_core`'s Python process, and flushes them at its exit.
/// Starts nothing: the drain starts on the first line.
pub fn install(module: &Bound<'_, PyModule>) -> PyResult<()> {
    install_logger();
    let flush_logs = module.getattr("flush_logs")?;
    module.py().import("atexit")?.call_method1("register", (flush_logs,))?;
    Ok(())
}

/// Installs the forwarder as the `log` logger.
pub fn install_logger() {
    struct Installed;
    impl Log for Installed {
        fn enabled(&self, metadata: &Metadata) -> bool {
            forwarder().enabled(metadata)
        }
        fn log(&self, record: &Record) {
            forwarder().queue(record);
        }
        fn flush(&self) {
            flush();
        }
    }
    if log::set_logger(&Installed).is_ok() {
        log::set_max_level(LevelFilter::Debug);
    }
}

/// The declared `_core.flush_logs`.
pub struct Logging;

impl lk_agents_abi::invoke::Logging for Logging {
    fn flush_logs() {
        flush();
    }
}

/// Waits (up to [`FLUSH_TIMEOUT`], without the GIL) until every line queued before the call has
/// reached Python's `logging`, then reads the loggers' levels again. A test boundary and the
/// interpreter's exit call it.
pub fn flush() {
    // A logger's level may have changed since the drain read it (a test's `caplog`).
    forwarder().levels.lock().clear();
    let pid = std::process::id();
    let queued = forwarder()
        .queue
        .lock()
        .as_ref()
        .filter(|queue| queue.pid == pid)
        .map(|queue| queue.sender.clone());
    let Some(sender) = queued else { return };
    let (done, flushed) = sync_channel(1);
    let wait = move || {
        if sender.send(Entry::Flush(done)).is_ok() {
            let _ = flushed.recv_timeout(FLUSH_TIMEOUT);
        }
    };
    if crate::host::holds_gil() {
        Python::attach(|py| py.detach(wait));
    } else {
        wait();
    }
}

impl Forwarder {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let levels = self.levels.lock();
        levels.get(metadata.target()).is_none_or(|filter| metadata.level() <= *filter)
    }

    fn queue(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let mut key_values = Vec::new();
        let _ = record.key_values().visit(&mut KeyValues(&mut key_values));
        let line = Line {
            level: record.level(),
            target: record.target().to_owned(),
            message: record.args().to_string(),
            file: record.file().map(str::to_owned),
            line: record.line(),
            key_values,
            at: SystemTime::now(),
        };
        let Some(sender) = self.sender() else { return };
        if let Err(TrySendError::Full(_)) = sender.try_send(Entry::Line(line)) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// This process's sender, its drain started on first use.
    fn sender(&self) -> Option<SyncSender<Entry>> {
        let pid = std::process::id();
        let mut queue = self.queue.lock();
        if let Some(queue) = queue.as_ref().filter(|queue| queue.pid == pid) {
            return Some(queue.sender.clone());
        }
        let (sender, entries) = sync_channel(QUEUE);
        std::thread::Builder::new()
            .name("lk-log-drain".into())
            .spawn(move || drain(entries))
            .ok()?;
        *queue = Some(Queue { pid, sender: sender.clone() });
        Some(sender)
    }
}

struct KeyValues<'a>(&'a mut Vec<(String, String)>);

impl<'kvs> log::kv::VisitSource<'kvs> for KeyValues<'_> {
    fn visit_pair(
        &mut self,
        key: log::kv::Key<'kvs>,
        value: log::kv::Value<'kvs>,
    ) -> Result<(), log::kv::Error> {
        self.0.push((key.as_str().to_owned(), value.to_string()));
        Ok(())
    }
}

fn drain(entries: Receiver<Entry>) {
    for entry in entries {
        match entry {
            Entry::Line(line) => Python::attach(|py| {
                let dropped = forwarder().dropped.swap(0, Ordering::Relaxed);
                if dropped > 0 {
                    let note = format!("{dropped} log lines were dropped: the queue was full");
                    let _ =
                        emit(py, &Line { level: Level::Warn, message: note, ..line.clone_meta() });
                }
                if let Err(error) = emit(py, &line) {
                    error.print(py);
                }
            }),
            Entry::Flush(done) => {
                let _ = done.send(());
            }
        }
    }
}

impl Line {
    fn clone_meta(&self) -> Line {
        Line {
            level: self.level,
            target: self.target.clone(),
            message: String::new(),
            file: self.file.clone(),
            line: self.line,
            key_values: Vec::new(),
            at: self.at,
        }
    }
}

/// `logging.getLogger(target)` handles the line, stamped when it was logged.
fn emit(py: Python<'_>, line: &Line) -> PyResult<()> {
    let name = line.target.replace("::", ".");
    let logger = py.import("logging")?.call_method1("getLogger", (&name,))?;
    let level = match line.level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    };
    let effective: u32 = logger.call_method0("getEffectiveLevel")?.extract()?;
    forwarder().levels.lock().insert(line.target.clone(), filter(effective));
    if level < effective {
        return Ok(());
    }
    let extra = PyDict::new(py);
    for (key, value) in &line.key_values {
        extra.set_item(key, value)?;
    }
    let record = logger.call_method1(
        "makeRecord",
        (
            &name,
            level,
            line.file.as_deref().unwrap_or(""),
            line.line.unwrap_or(0),
            &line.message,
            pyo3::types::PyTuple::empty(py),
            py.None(),
            py.None(),
            extra,
        ),
    )?;
    let created = line.at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
    record.setattr("created", created)?;
    record.setattr("msecs", (created.fract() * 1000.0).floor())?;
    logger.call_method1("handle", (record,))?;
    Ok(())
}

/// The most verbose `log` level a Python logger at `effective` passes.
fn filter(effective: u32) -> LevelFilter {
    match effective {
        0..=5 => LevelFilter::Trace,
        6..=10 => LevelFilter::Debug,
        11..=20 => LevelFilter::Info,
        21..=30 => LevelFilter::Warn,
        31..=40 => LevelFilter::Error,
        _ => LevelFilter::Off,
    }
}
