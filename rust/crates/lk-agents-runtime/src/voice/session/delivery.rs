use super::*;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_voice::agent_session::ToolUpdate;

impl VoiceSession {
    /// Queue before returning its reader. The session's existing command and speech owners
    /// retain both delivery and cleanup when that reader is canceled or never polled.
    pub(super) fn enqueue_tool_say(
        &self,
        invocation_id: String,
        operation: u64,
        call_id: String,
        text: String,
    ) -> Result<BoxFuture<'static, Result<(), ProviderError>>, CommandRefused> {
        self.api()?;
        let (speech, call, reply) = self.speech_call(None)?;
        let origin = VoiceCommandOrigin::Tool {
            invocation_id: invocation_id.clone(),
            operation,
            call_id: call_id.clone(),
        };
        self.send_awaited(
            origin,
            SpeechCommand::ToolUpdate {
                invocation_id: invocation_id.clone(),
                operation,
                call_id: call_id.clone(),
                update: ToolUpdate::Exact { text: text.clone(), call: Some(call) },
            },
            reply,
        )?;
        // A first exact output releases the originating speech's blocking step. The body
        // remains registered; exact output alone does not ask the model to acknowledge it.
        let progress = ToolOutcome::Update {
            message: "The tool is providing an exact progress update.".into(),
            raw: String::new(),
            reply_required: false,
        };
        // Once the step was released there is no second synthetic progress report to send.
        drop(self.tool_bodies.first_tool_progress(
            &invocation_id,
            operation,
            &call_id,
            progress.into(),
        ));
        Ok(Box::pin(async move { speech.exact_delivery(&text).await }))
    }
}

#[cfg(test)]
#[path = "delivery_tests.rs"]
mod tests;
