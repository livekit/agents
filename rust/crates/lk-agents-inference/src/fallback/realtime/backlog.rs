//! The requests a session holds while a replacement model waits for its conversation frame.

/// Requests may wait for a replacement's context, never for unbounded application backlog.
pub const FALLBACK_PENDING_REPLIES: usize = 32;
pub const FALLBACK_PENDING_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BacklogOverload {
    #[error("realtime fallback pending reply limit exceeded")]
    Requests,
    #[error("realtime fallback pending reply byte limit exceeded")]
    Bytes,
}

/// Ordered requests admitted while a new model waits for its conversation frame.
#[derive(Default)]
pub struct FallbackBacklog {
    held: Vec<lk_agents_abi::RealtimeCommand>,
    bytes: usize,
}

impl FallbackBacklog {
    pub fn hold(
        &mut self,
        command: lk_agents_abi::RealtimeCommand,
    ) -> Result<(), (BacklogOverload, lk_agents_abi::RealtimeCommand)> {
        if self.held.len() == FALLBACK_PENDING_REPLIES {
            return Err((BacklogOverload::Requests, command));
        }
        let bytes = reply_bytes(&command);
        if bytes > FALLBACK_PENDING_BYTES.saturating_sub(self.bytes) {
            return Err((BacklogOverload::Bytes, command));
        }
        self.bytes += bytes;
        self.held.push(command);
        Ok(())
    }

    pub fn cancel(&mut self, request_id: &str) {
        self.held.retain(|command| command.req_id() != Some(request_id));
        self.bytes = self.held.iter().map(reply_bytes).sum();
    }

    pub fn take(&mut self) -> Vec<lk_agents_abi::RealtimeCommand> {
        self.bytes = 0;
        std::mem::take(&mut self.held)
    }

    pub fn clear(&mut self) {
        self.held.clear();
        self.bytes = 0;
    }
}

fn reply_bytes(command: &lk_agents_abi::RealtimeCommand) -> usize {
    match command {
        lk_agents_abi::RealtimeCommand::GenerateReply(reply) => reply
            .req_id
            .capacity()
            .saturating_add(reply.instructions.as_ref().map_or(0, String::capacity))
            .saturating_add(match &reply.tool_choice {
                Some(lk_agents_abi::ToolChoice::Named(name)) => name.capacity(),
                _ => 0,
            }),
        lk_agents_abi::RealtimeCommand::Say(reply) => {
            reply.req_id.capacity().saturating_add(reply.text.capacity())
        }
        _ => 0,
    }
}
