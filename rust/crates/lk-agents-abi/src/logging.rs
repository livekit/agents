//! A nonblocking logging service supplied by the hosting runtime.

/// A record delivered on the logging thread, with the number of records dropped since the
/// preceding delivery. Host callbacks must capture their invocation before submission.
pub type LogCallback = Box<dyn FnOnce(u64) + Send>;

/// The process's logging service. Holding a handle starts nothing; the runtime owns the writer.
#[derive(Clone, Copy)]
pub struct LogWriter {
    initialize: fn() -> Result<(), String>,
    submit: fn(LogCallback),
}

impl LogWriter {
    pub const fn new(initialize: fn() -> Result<(), String>, submit: fn(LogCallback)) -> Self {
        Self { initialize, submit }
    }

    /// A sink for callers that do not supply a logging service, including hermetic drivers.
    pub const fn disabled() -> Self {
        Self::new(|| Ok(()), |_| {})
    }

    pub fn initialize(self) -> Result<(), String> {
        (self.initialize)()
    }

    /// Submission never waits for the writer or calls the host inline.
    pub fn submit_host(self, record: impl FnOnce(u64) + Send + 'static) {
        (self.submit)(Box::new(record));
    }

    pub fn emit(self, level: log::Level, message: String) {
        self.submit_host(move |dropped| {
            if dropped > 0 {
                log::warn!(target: "livekit.agents", "dropped {dropped} log records while the writer was stalled");
            }
            log::log!(target: "livekit.agents", level, "{message}");
        });
    }
}
