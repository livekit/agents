//! Native runner sources adapted in Rust, without crossing the host boundary.

use lk_agents_abi::agent::run::{InvalidRun, RunUpdate};
use lk_agents_abi::agent::stream::StreamOutcome;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::text_session::{TextSessionDelivery, TextSessionUpdate};
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{BoxFuture, Closed, StreamOutput, Ts};
use lk_agents_core::run_stream::projection::Envelope;
use std::sync::Arc;
use tokio::sync::watch;

pub enum Message {
    Event(Envelope),
    Release(String),
    Terminal(StreamOutcome),
    Incomplete(Vec<String>),
}

pub trait Input: Send {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<Message>, ProviderError>>;
    fn release(&self, delivery: TextSessionDelivery) -> Result<(), InvalidRun>;
    fn usage(&self) -> watch::Receiver<Usage>;
    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed>;
}

pub struct Direct {
    pub source: Box<dyn StreamOutput<RunUpdate>>,
    pub usage: watch::Receiver<Usage>,
    pub agent: String,
    pub invocation: String,
}
impl Input for Direct {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<Message>, ProviderError>> {
        Box::pin(async move {
            Ok(self.source.next().await?.map(|update| match update {
                RunUpdate::Stream(event) => Message::Event(Envelope {
                    sequence: None,
                    frame_id: None,
                    agent_name: self.agent.clone(),
                    invocation_id: self.invocation.clone(),
                    event,
                }),
                RunUpdate::IncompleteCleanup(pending) => Message::Incomplete(pending),
                terminal => Message::Terminal(StreamOutcome::Run(terminal)),
            }))
        })
    }
    fn release(&self, _: TextSessionDelivery) -> Result<(), InvalidRun> {
        Ok(())
    }
    fn usage(&self) -> watch::Receiver<Usage> {
        self.usage.clone()
    }
    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.source.close(deadline)
    }
}

pub struct Session {
    pub source: Box<dyn StreamOutput<TextSessionUpdate>>,
    pub usage: watch::Receiver<Usage>,
    pub release: Arc<dyn Fn(TextSessionDelivery) -> Result<(), InvalidRun> + Send + Sync>,
}
impl Input for Session {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<Message>, ProviderError>> {
        Box::pin(async move {
            Ok(self.source.next().await?.map(|update| match update {
                TextSessionUpdate::Stream {
                    sequence,
                    frame_id,
                    agent_name,
                    invocation_id,
                    event,
                } => Message::Event(Envelope {
                    sequence: Some(sequence),
                    frame_id: Some(frame_id.value),
                    agent_name,
                    invocation_id,
                    event,
                }),
                TextSessionUpdate::InvocationSettled(id) | TextSessionUpdate::Parked(id) => {
                    Message::Release(id)
                }
                TextSessionUpdate::IncompleteCleanup(pending) => Message::Incomplete(pending),
                terminal => Message::Terminal(StreamOutcome::Session(terminal)),
            }))
        })
    }
    fn release(&self, delivery: TextSessionDelivery) -> Result<(), InvalidRun> {
        (self.release)(delivery)
    }
    fn usage(&self) -> watch::Receiver<Usage> {
        self.usage.clone()
    }
    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.source.close(deadline)
    }
}
