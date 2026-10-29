//! A keyterm pass's provider operations, reported to the voice extraction state.

use lk_agents_abi::{ChatRequest, Llm};
use lk_agents_voice::agent_session::SessionShared;
use lk_agents_voice::keyterm_detection::{DetectionPass, DetectionPassEvent};
use std::sync::Arc;

pub async fn detection_pass(
    shared: Arc<SessionShared>,
    llm: Arc<dyn Llm>,
    request: ChatRequest,
    req: u64,
) {
    let provider = llm.id();
    let started_at = shared.time.wall_secs();
    let mut pass = DetectionPass::new(req, provider, started_at);
    let mut stream = match llm.chat(&request).await {
        Ok(stream) => stream,
        Err(_) => {
            if let Some(answer) = pass.handle(DetectionPassEvent::OpenedFailed, started_at) {
                shared.report(answer);
            }
            return;
        }
    };
    loop {
        let event = match stream.next().await {
            Ok(Some(chunk)) => DetectionPassEvent::Chunk(chunk),
            Ok(None) => DetectionPassEvent::Ended,
            Err(_) => DetectionPassEvent::StreamFailed,
        };
        if let Some(answer) = pass.handle(event, shared.time.wall_secs()) {
            shared.report(answer);
            return;
        }
    }
}
