//! Process-wide signal registration with one active launcher owner.
//!
//! `ctrlc` installs a dedicated native listener once for the lifetime of the process.
//! The CLI owns these signals; it must not be mixed with another host signal handler.

use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;

use lk_agents_abi::cli::CliError;

use crate::shutdown::{Watchdog, terminate};

static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
static OWNER: Mutex<Weak<Watchdog>> = Mutex::new(Weak::new());

pub(crate) fn listen(watchdog: &Arc<Watchdog>) -> Result<(), CliError> {
    let mut owner = OWNER.lock();
    if watchdog.is_closed() {
        return Err(CliError::Invalid("cannot listen after shutdown has closed".into()));
    }
    if let Some(current) = owner.upgrade() {
        return if Arc::ptr_eq(&current, watchdog) {
            Ok(())
        } else {
            Err(CliError::Invalid("another launcher already owns process signals".into()))
        };
    }
    INSTALLED
        .get_or_init(|| ctrlc::set_handler(on_signal).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| {
            CliError::Invalid(format!("cannot install process signal handler: {error}"))
        })?;
    *owner = Arc::downgrade(watchdog);
    Ok(())
}

pub(crate) fn release(watchdog: &Arc<Watchdog>) {
    let mut owner = OWNER.lock();
    if owner.ptr_eq(&Arc::downgrade(watchdog)) {
        *owner = Weak::new();
    }
}

fn on_signal() {
    let owner = OWNER.lock().upgrade();
    if let Some(owner) = owner {
        owner.begin();
    } else {
        // The permanent OS handler must never swallow signals between launcher runs.
        terminate(b"livekit.agents: termination requested without an active launcher\n");
    }
}
