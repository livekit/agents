//! Bounded process-wide diagnostics execution, independent of worker and process deadlines.
//!
//! Python logging and stalled output may occupy the writer. Producers never invoke a logger,
//! wait for capacity, or create a task per record. The next delivered record reports drops.

use std::sync::{OnceLock, mpsc::SyncSender};

type HostDiagnostic = Box<dyn FnOnce(u64) + Send>;

struct Diagnostics {
    sender: SyncSender<HostDiagnostic>,
    dropped: tokio::sync::watch::Sender<u64>,
}

static DIAGNOSTICS: OnceLock<Result<Diagnostics, String>> = OnceLock::new();

fn service() -> Result<&'static Diagnostics, &'static String> {
    DIAGNOSTICS
        .get_or_init(|| {
            let (sender, receiver) = std::sync::mpsc::sync_channel::<HostDiagnostic>(256);
            let (dropped, counts) = tokio::sync::watch::channel(0_u64);
            std::thread::Builder::new()
                .name("worker-diagnostics".into())
                .spawn(move || {
                    let mut reported = 0;
                    for diagnostic in receiver {
                        let dropped = *counts.borrow();
                        diagnostic(dropped - reported);
                        reported = dropped;
                    }
                })
                .map_err(|error| error.to_string())?;
            Ok(Diagnostics { sender, dropped })
        })
        .as_ref()
}

pub fn initialize() -> Result<(), String> {
    service().map(|_| ()).map_err(Clone::clone)
}

pub fn submit_host(diagnostic: impl FnOnce(u64) + Send + 'static) {
    let Some(Ok(service)) = DIAGNOSTICS.get() else {
        return;
    };
    if service.sender.try_send(Box::new(diagnostic)).is_err() {
        service.dropped.send_modify(|count| *count = count.saturating_add(1));
    }
}

pub fn emit(level: log::Level, message: String) {
    submit_host(move |dropped| {
        if dropped > 0 {
            log::warn!(target: "livekit.agents", "dropped {dropped} worker diagnostics while the writer was stalled");
        }
        log::log!(target: "livekit.agents", level, "{message}");
    });
}
