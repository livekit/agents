//! A keyterm pass's provider operations, reported to the voice extraction state.

use crate::telemetry;
use crate::voice::session::SessionShared;
use lk_agents_abi::{ChatRequest, Llm};
use lk_agents_voice::keyterm_detection::{DetectionPass, DetectionPassEvent};
use std::sync::Arc;

/// One pass, its request an `llm_request` under the pass's `keyterm_detection` span.
pub async fn detection_pass(
    shared: Arc<SessionShared>,
    llm: Arc<dyn Llm>,
    mut request: ChatRequest,
    req: u64,
    span: Option<telemetry::SpanRef>,
) {
    let provider = llm.id();
    let started_at = shared.time.wall_secs();
    let mut pass = DetectionPass::new(req, provider, started_at);
    let cx = match span {
        Some(span) => shared.trace.clone().under::<telemetry::KeytermDetection>(span),
        None => shared.trace.clone(),
    };
    let mut trace = telemetry::LlmTrace::start(&cx, llm.as_ref(), &mut request);
    let attempt = trace.attempt();
    let mut stream = match llm.chat(&request).await {
        Ok(stream) => stream,
        Err(error) => {
            trace.end_attempt(attempt, Some(&error));
            trace.fail(&error);
            if let Some(answer) = pass.handle(DetectionPassEvent::OpenedFailed, started_at) {
                shared.report(answer);
            }
            return;
        }
    };
    let mut attempt = Some(attempt);
    loop {
        let event = match stream.next().await {
            Ok(Some(chunk)) => {
                trace.observe(&chunk);
                DetectionPassEvent::Chunk(chunk)
            }
            Ok(None) => DetectionPassEvent::Ended,
            Err(error) => {
                trace.end_attempt(attempt.take().flatten(), Some(&error));
                trace.fail(&error);
                DetectionPassEvent::StreamFailed
            }
        };
        if let Some(answer) = pass.handle(event, shared.time.wall_secs()) {
            if let Some(attempt) = attempt.take() {
                trace.end_attempt(attempt, None);
                trace.end(false, None);
            }
            shared.report(answer);
            return;
        }
    }
}
