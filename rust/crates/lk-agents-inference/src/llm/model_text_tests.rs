use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::StreamOutput;
use lk_agents_abi::{BoxFuture, ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, Ts};

use super::ModelText;

/// A provider that streams `deltas`, under `model`.
struct Scripted {
    model: &'static str,
    deltas: Vec<LlmChunk>,
}

struct Chunks(std::vec::IntoIter<LlmChunk>);

impl StreamOutput<LlmChunk> for Chunks {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { Ok(self.0.next()) })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Chunks {}

impl Llm for Scripted {
    fn label(&self) -> String {
        "scripted".into()
    }
    fn model(&self) -> String {
        self.model.into()
    }
    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _req: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        let deltas = self.deltas.clone();
        Box::pin(async move { Ok(Box::new(Chunks(deltas.into_iter())) as LlmStream) })
    }
}

fn text(t: &str) -> LlmChunk {
    let delta = ChoiceDelta { content: Some(t.into()), ..Default::default() };
    LlmChunk { id: "c".into(), delta: Some(delta), ..Default::default() }
}

/// What the reader of `model`'s stream gets: every chunk.
fn read(model: &'static str, deltas: Vec<LlmChunk>) -> Vec<LlmChunk> {
    let llm = ModelText::wrap(Arc::new(Scripted { model, deltas }));
    let rt = tokio::runtime::Builder::new_current_thread().build().expect("runtime");
    rt.block_on(async {
        let mut stream = llm
            .chat(&lk_agents_abi::telemetry::Cx::NONE, &ChatRequest::default())
            .await
            .expect("opens");
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await.expect("reads") {
            out.push(chunk);
        }
        out
    })
}

fn said(chunks: &[LlmChunk]) -> String {
    chunks.iter().filter_map(|c| c.delta.as_ref()?.content.clone()).collect()
}

#[test]
fn reasoning_never_reaches_the_reader_nor_counts_as_output() {
    let out =
        read("openai/gpt-4o", vec![text("<think>plan"), text(" more</thi"), text("nk>Hello")]);
    assert_eq!(said(&out), "Hello");
    // the chunks that carried only reasoning are gone: no time to first token, nothing to retry
    assert_eq!(out.iter().filter(|c| c.has_response()).count(), 1);
}

#[test]
fn what_the_lexer_held_comes_out_when_the_provider_finishes() {
    // a trailing `<` might have been a delimiter until the stream ended; a cut one is dropped
    assert_eq!(said(&read("m", vec![text("3 <")])), "3 ");
    assert_eq!(said(&read("m", vec![text("a"), text("<thi")])), "a");
    assert_eq!(said(&read("m", vec![text("5 <"), text(" 6")])), "5 < 6");
}

#[test]
fn chunks_carrying_more_than_text_are_kept() {
    let mut usage = text("<think>x</think>");
    usage.usage = Some(Box::default());
    let out = read("m", vec![usage]);
    assert_eq!(out.len(), 1, "usage must reach the reader");
    assert_eq!(out[0].delta.as_ref().and_then(|d| d.content.clone()), None);
    // an empty delta stays an empty delta
    let out = read("m", vec![text("")]);
    assert_eq!(out[0].delta.as_ref().and_then(|d| d.content.clone()), Some(String::new()));
}

#[test]
fn the_models_own_syntax_is_used() {
    // gpt-oss: only the final channel is said
    let reply = "<|channel|>analysis<|message|>secret<|end|><|start|>assistant<|channel|>final<|message|>Hi<|return|>";
    assert_eq!(said(&read("openai/gpt-oss-120b", vec![text(reply)])), "Hi");
    // Gemma: its channel delimiters, not `<think>`
    let gemma = "<|channel>thought plan<channel|>Hi <think>shown</think>";
    assert_eq!(said(&read("google/gemma-4-31b-it", vec![text(gemma)])), "Hi <think>shown</think>");
}

#[test]
fn a_bound_view_keeps_the_filter() {
    let llm = ModelText::wrap(Arc::new(Scripted { model: "m", deltas: vec![] }));
    // a provider that binds no options keeps the wrapper as it is
    assert!(llm.with_conn_options(&Default::default()).is_none());
    assert_eq!(llm.model(), "m");
}

// Catches: reasoning a model writes after its tool calls in the same stream reaching the reader.
#[test]
fn reasoning_after_a_tool_call_never_reaches_the_reader() {
    let call = lk_agents_abi::FunctionToolCall {
        name: "transfer".into(),
        arguments: "{}".into(),
        call_id: "call-1".into(),
        ..Default::default()
    };
    let tool = LlmChunk {
        id: "c".into(),
        delta: Some(ChoiceDelta { tool_calls: vec![call], ..Default::default() }),
        ..Default::default()
    };
    let chunks = read("fake", vec![tool, text("<think>planning the reply</think>Be")]);
    assert_eq!(said(&chunks), "Be");
}
