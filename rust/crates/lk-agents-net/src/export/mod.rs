//! Telemetry's export: what the process's spans, metrics and report records become
//! (docs/internals/telemetry.md "Destinations"). Spans are declared in `lk-agents-abi` and recorded
//! through `lk-agents-telemetry`; the process tracer records to this module's exporter.
//!
//! - [`otlp`]: the OTLP exporter to LiveKit Cloud and the application's endpoint.
//! - [`session_report`]: the LiveKit Cloud session report's records and recording upload.
//! - [`host`]: `_core.TelemetrySpan`, a span the host's OpenTelemetry API starts.
//! - [`stall`]: event-loop stalls, measured and spanned within a rate limit.
//! - [`ExporterSlot`], [`exporter`]: the process's exporter once the host installs telemetry.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod host;
pub mod otlp;
pub mod profile;
pub mod session_report;
pub mod stall;

use std::sync::{Arc, LazyLock};

use lk_agents_abi::telemetry::record::{Record, RecordSink};
use parking_lot::RwLock;

use otlp::Exporter;

/// The process's exporter once the host installed telemetry: the process tracer's sink. Records
/// before the installation are discarded.
#[derive(Default)]
pub struct ExporterSlot(RwLock<Option<Arc<Exporter>>>);

impl ExporterSlot {
    /// The process's slot: the process tracer records to it and job runs read its exporter.
    pub fn process() -> Arc<ExporterSlot> {
        static SLOT: LazyLock<Arc<ExporterSlot>> = LazyLock::new(Arc::default);
        SLOT.clone()
    }

    pub fn get(&self) -> Option<Arc<Exporter>> {
        self.0.read().clone()
    }

    /// Installs the exporter `start` builds, unless one is installed.
    pub fn install(&self, start: impl FnOnce() -> Exporter) -> Arc<Exporter> {
        self.0.write().get_or_insert_with(|| Arc::new(start())).clone()
    }

    /// Removes the installed exporter, so a later install starts a new one.
    pub fn take(&self) -> Option<Arc<Exporter>> {
        self.0.write().take()
    }

    /// Abandons a fork-inherited exporter without dropping it: its task belongs to the parent's
    /// runtime.
    pub fn forget(&self) {
        std::mem::forget(self.0.write().take());
    }
}

impl RecordSink for ExporterSlot {
    fn record(&self, record: Record) {
        if let Some(exporter) = self.0.read().as_ref() {
            exporter.record(record);
        }
    }
}

/// The process exporter, for its job registry and configuration; `None` before installation.
pub fn exporter() -> Option<Arc<Exporter>> {
    ExporterSlot::process().get()
}
