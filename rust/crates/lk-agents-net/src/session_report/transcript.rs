//! The session transcript, prepared incrementally as bounded JSON, and the console report.

use std::sync::{Arc, Mutex};

use lk_agents_abi::telemetry::upload::{PreparedTranscript, UploadError};
use tokio::sync::OwnedSemaphorePermit;

use super::{TRANSCRIPT_BYTES, upload_error};
use crate::env::ProcessEnv;

pub(super) const TRANSCRIPT_ITEMS: usize = 10_000;
pub(super) const TRANSCRIPT_ITEM_BYTES: usize = 64 * 1024;

/// Retains only bounded JSON, rather than a second copy of every native chat item.
/// Prepared under the process's preparation admission; the console report serializes on the
/// caller's blocking pool.
pub struct TranscriptBuilder {
    state: Mutex<TranscriptState>,
    permit: Arc<OwnedSemaphorePermit>,
    env: ProcessEnv,
}

struct TranscriptState {
    bytes: Vec<u8>,
    items: usize,
    input_items: usize,
    events: Vec<serde_json::Value>,
    event_bytes: usize,
    failure: Option<String>,
    finished: bool,
}

impl PreparedTranscript for TranscriptBuilder {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv) -> Result<Self, UploadError> {
        let permit = (env.report_admission)()
            .map_err(UploadError)?
            .preparation()
            .ok_or_else(|| UploadError("session report preparation capacity exhausted".into()))?;
        let mut bytes = Vec::with_capacity(TRANSCRIPT_BYTES);
        bytes.extend_from_slice(b"{\"items\":[");
        Ok(Self {
            state: Mutex::new(TranscriptState {
                bytes,
                items: 0,
                input_items: 0,
                events: Vec::new(),
                event_bytes: 0,
                failure: None,
                finished: false,
            }),
            permit: Arc::new(permit),
            env: env.clone(),
        })
    }

    fn input_batch_size(&self) -> usize {
        1
    }

    fn content_policy(&self) -> (bool, bool) {
        let options = transcript_options();
        (!options.exclude_image, !options.exclude_audio)
    }

    fn check_input(&self, items: usize, item_bytes: usize) -> Result<(), UploadError> {
        let failure = if items > TRANSCRIPT_ITEMS {
            Some("session transcript exceeds the 10000 item limit")
        } else if item_bytes > TRANSCRIPT_ITEM_BYTES {
            Some("session transcript item exceeds the 64 KiB input limit")
        } else {
            None
        };
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.finished {
            return Err(UploadError("session transcript is already finished".into()));
        }
        if let Some(failure) = failure {
            state.failure = Some(failure.into());
            state.bytes.clear();
        }
        if let Some(failure) = &state.failure { Err(UploadError(failure.clone())) } else { Ok(()) }
    }

    fn append(&self, item: lk_agents_abi::ChatItem) -> Result<(), UploadError> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(failure) = &state.failure {
            return Err(UploadError(failure.clone()));
        }
        if state.finished {
            return Err(UploadError("session transcript is already finished".into()));
        }
        let result = append_item(&mut state, &item);
        if let Err(error) = &result {
            state.failure = Some(error.0.clone());
            state.bytes.clear();
        }
        result
    }

    fn finish(&self) -> Result<String, UploadError> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(failure) = &state.failure {
            return Err(UploadError(failure.clone()));
        }
        if !state.finished {
            state.bytes.extend_from_slice(b"]}");
            state.finished = true;
        }
        String::from_utf8(state.bytes.clone()).map_err(upload_error)
    }

    fn append_event(&self, event: serde_json::Value) -> Result<bool, UploadError> {
        use lk_agents_abi::telemetry::policy;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(failure) = &state.failure {
            return Err(UploadError(failure.clone()));
        }
        if state.finished {
            return Err(UploadError("session transcript is already finished".into()));
        }
        let mut encoded =
            BoundedWriter { bytes: Vec::new(), limit: policy::REPORT_EVENT_RECORD_BYTES };
        if serde_json::to_writer(&mut encoded, &event).is_err() {
            return Ok(false);
        }
        if state.events.len() >= policy::REPORT_EVENT_ITEMS
            || encoded.bytes.len() > policy::REPORT_EVENT_BYTES.saturating_sub(state.event_bytes)
        {
            return Ok(false);
        }
        state.event_bytes += encoded.bytes.len();
        state.events.push(event);
        Ok(true)
    }

    async fn finish_report(
        &self,
        mut report: lk_agents_abi::telemetry::report::projection::SessionReportSnapshot,
        path: String,
    ) -> Result<(), UploadError> {
        let (bytes, events) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(failure) = &state.failure {
                return Err(UploadError(failure.clone()));
            }
            if !state.finished {
                state.bytes.extend_from_slice(b"]}");
                state.finished = true;
            }
            (std::mem::take(&mut state.bytes), std::mem::take(&mut state.events))
        };
        let permit = self.permit.clone();
        self.env
            .caller_runtime()
            .map_err(UploadError)?
            .spawn_blocking(move || {
                let _retained = permit;
                report.events = events;
                // Keep the existing bounded transcript bytes; a second parsed history would retain
                // one allocation per JSON node during serialization.
                let history = serde_json::value::RawValue::from_string(
                    String::from_utf8(bytes).map_err(upload_error)?,
                )
                .map_err(upload_error)?;
                report.chat_history = None;
                #[derive(serde::Serialize)]
                struct ConsoleReport<'a> {
                    #[serde(flatten)]
                    report: lk_agents_abi::telemetry::report::projection::SessionReportSnapshot,
                    chat_history: &'a serde_json::value::RawValue,
                }
                let mut encoded = BoundedWriter {
                    bytes: Vec::new(),
                    limit: lk_agents_abi::telemetry::policy::REPORT_RETAINED_BYTES,
                };
                serde_json::to_writer(
                    &mut encoded,
                    &ConsoleReport { report, chat_history: &history },
                )
                .map_err(upload_error)?;
                let path = std::path::Path::new(&path);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(upload_error)?;
                }
                std::fs::write(path, &encoded.bytes).map_err(upload_error)
            })
            .await
            .map_err(upload_error)?
    }
}

fn transcript_options() -> lk_agents_abi::ToDictOptions {
    lk_agents_abi::ToDictOptions { exclude_timestamp: false, ..Default::default() }
}

fn append_item(
    state: &mut TranscriptState,
    item: &lk_agents_abi::ChatItem,
) -> Result<(), UploadError> {
    if state.input_items >= TRANSCRIPT_ITEMS {
        return Err(UploadError("session transcript exceeds the 10000 item limit".into()));
    }
    state.input_items += 1;
    let options = transcript_options();
    let Some(projected) =
        lk_agents_core::llm::chat_dict::item_to_dict(item, &options, &Default::default())
            .map_err(upload_error)?
    else {
        return Ok(());
    };
    let mut item = BoundedWriter { bytes: Vec::new(), limit: TRANSCRIPT_ITEM_BYTES };
    serde_json::to_writer(&mut item, &projected).map_err(upload_error)?;
    let separator = usize::from(state.items > 0);
    if item.bytes.len().saturating_add(separator + 2)
        > TRANSCRIPT_BYTES.saturating_sub(state.bytes.len())
    {
        return Err(UploadError("session transcript exceeds the 2 MiB upload limit".into()));
    }
    if separator > 0 {
        state.bytes.push(b',');
    }
    state.bytes.extend_from_slice(&item.bytes);
    state.items += 1;
    Ok(())
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl std::io::Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::FileTooLarge,
                format!("session recording exceeds its {} KiB serialized limit", self.limit / 1024),
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
