//! Bounded process-wide logging, independent of operation deadlines.
//!
//! Python logging and stalled output may occupy the writer. Producers never invoke a logger,
//! wait for capacity, or create a task per record. The next delivered record reports drops.
//! The process runtime owns the service and replaces it after a fork.

use std::sync::mpsc::SyncSender;

use lk_agents_abi::logging::{LogCallback, LogWriter};

pub(crate) struct LogService {
    sender: SyncSender<LogCallback>,
    dropped: tokio::sync::watch::Sender<u64>,
}

impl LogService {
    pub(crate) fn start(capacity: usize) -> Result<Self, String> {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<LogCallback>(capacity);
        let (dropped, counts) = tokio::sync::watch::channel(0_u64);
        std::thread::Builder::new()
            .name("lk-log-writer".into())
            .spawn(move || {
                let mut reported = 0;
                for record in receiver {
                    let dropped = *counts.borrow();
                    record(dropped - reported);
                    reported = dropped;
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self { sender, dropped })
    }

    fn submit(&self, record: LogCallback) {
        if self.sender.try_send(record).is_err() {
            self.dropped.send_modify(|count| *count = count.saturating_add(1));
        }
    }
}

fn initialize() -> Result<(), String> {
    crate::process::logging().map(|_| ()).map_err(|error| error.to_string())
}

fn submit_host(record: LogCallback) {
    if let Some(service) = crate::process::started_logging() {
        service.submit(record);
    }
}

/// The process logging service, read without starting its writer.
pub const fn writer() -> LogWriter {
    LogWriter::new(initialize, submit_host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn a_stalled_writer_does_not_block_submission_and_reports_drops_once() {
        let service = LogService::start(1).unwrap();
        let (started, running) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let (delivered, records) = mpsc::channel();
        service.submit(Box::new(move |dropped| {
            started.send(dropped).unwrap();
            resume.recv_timeout(Duration::from_secs(5)).unwrap();
        }));
        assert_eq!(running.recv_timeout(Duration::from_secs(5)).unwrap(), 0);

        let second = delivered.clone();
        service.submit(Box::new(move |dropped| second.send(dropped).unwrap()));
        // The writer is stalled and its one queued slot is occupied. This returns immediately.
        service.submit(Box::new(|_| panic!("the full queue must drop this record")));
        release.send(()).unwrap();
        assert_eq!(records.recv_timeout(Duration::from_secs(5)).unwrap(), 1);

        service.submit(Box::new(move |dropped| delivered.send(dropped).unwrap()));
        assert_eq!(records.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
    }
}
