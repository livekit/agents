//! The declared text-turn surface over the A2A executor.

use super::Turn;
use lk_agents_abi::a2a::turn::Turn as TurnContract;
use lk_agents_abi::a2a::{
    options::TurnOptions,
    turn::{TurnHost, TurnResult},
};
use lk_agents_abi::storage::StorageError;
use lk_agents_net::env::ProcessEnv;
use lk_agents_storage::Thread;
use std::sync::Arc;

impl TurnContract for Turn {
    type Env = ProcessEnv;
    type Thread = Thread;

    fn new(
        env: &ProcessEnv,
        thread: Arc<Thread>,
        lane: String,
        host: Arc<dyn TurnHost>,
        options: Option<TurnOptions>,
    ) -> Self {
        Self::new(
            thread.as_ref().clone(),
            &lane,
            host,
            env.clock.clone(),
            options.unwrap_or_default(),
        )
    }
    fn lane(&self) -> String {
        self.lane().to_owned()
    }
    fn policy_for(&self, channel: String) -> String {
        self.policy_for(&channel).as_str().to_owned()
    }
    async fn run(
        &self,
        message: serde_json::Value,
        task_id: Option<String>,
        context: lk_agents_abi::invoke::CallContext,
    ) -> Result<TurnResult, StorageError> {
        let mut turn = self.clone();
        if let Some(context) = context.host_context {
            turn = turn.with_context(context);
        }
        Ok(turn.run(message, task_id).await?.into())
    }
}
