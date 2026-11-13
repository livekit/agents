//! What a model writes that is not for the person (reasoning, role turns it invents, template
//! tokens, tool calls it wrote as text), taken out of every provider's stream once, by the model's
//! [`ModelSyntax`]. Applied to the provider a session or a run calls, so every provider is covered
//! and an `llm_node` override's own output is its own.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{ProviderStream, StreamOutput};
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, Ts,
};
use lk_agents_core::lexer::{Hidden, Kind, Lexer, ModelSyntax, Projector, Rules, View};

use crate::telemetry;

/// `llm` whose streams carry only what the model says to the person.
pub struct ModelText {
    inner: Arc<dyn Llm>,
    syntax: ModelSyntax,
}

impl ModelText {
    pub fn wrap(llm: Arc<dyn Llm>) -> Arc<dyn Llm> {
        let syntax = crate::llm::models::model_syntax(&llm.model());
        Arc::new(ModelText { inner: llm, syntax })
    }
}

impl Llm for ModelText {
    fn instance_id(&self) -> Option<usize> {
        self.inner.instance_id()
    }

    fn label(&self) -> String {
        self.inner.label()
    }

    fn model(&self) -> String {
        self.inner.model()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.inner.id()
    }

    fn prewarm(&self, cx: &lk_agents_abi::telemetry::Cx) {
        self.inner.prewarm(cx);
    }

    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Llm>> {
        let inner = self.inner.with_conn_options(conn)?;
        Some(Arc::new(ModelText { inner, syntax: self.syntax }))
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.inner.failed(error, recoverable);
    }

    fn request_span(&self) -> telemetry::RequestSpan {
        self.inner.request_span()
    }

    fn chat<'a>(
        &'a self,
        cx: &telemetry::Cx,
        req: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        let opened = self.inner.chat(cx, req);
        let syntax = self.syntax;
        let label = self.inner.label();
        Box::pin(async move {
            let inner = opened.await?;
            Ok(Box::new(TextStream::new(inner, syntax, label)) as LlmStream)
        })
    }
}

/// One stream's text as the model says it to the person.
pub struct SaidText {
    lexer: Lexer,
    view: Projector,
    /// Who to name in the warning for a tool call written as text; `None` stays silent.
    warn_as: Option<String>,
}

impl SaidText {
    pub fn new(syntax: ModelSyntax, warn_as: Option<String>) -> SaidText {
        SaidText {
            lexer: Lexer::new(Rules::model(syntax)),
            view: Projector::new(View::Model),
            warn_as,
        }
    }

    /// A delta in; what of it is said, now that it is settled.
    pub fn push(&mut self, text: &str) -> String {
        let tokens = self.lexer.push(text);
        self.said(&tokens)
    }

    /// The stream ended: what was held settles, a cut span dropped.
    pub fn finish(&mut self) -> String {
        let tokens = self.lexer.finish();
        self.said(&tokens)
    }

    fn said(&mut self, tokens: &[lk_agents_core::lexer::Token]) -> String {
        if tokens.iter().any(|t| t.kind == Kind::Hidden(Hidden::ToolCall))
            && let Some(llm) = self.warn_as.take()
        {
            log::warn!(
                target: lk_agents_abi::LOG_TARGET,
                llm = llm.as_str();
                "the model wrote a tool call as text; it was hidden and not run"
            );
        }
        self.view.text(tokens)
    }
}

/// One attempt's stream, its text read through the model's syntax.
struct TextStream {
    inner: LlmStream,
    said: SaidText,
    last_id: String,
    finished: bool,
}

impl TextStream {
    fn new(inner: LlmStream, syntax: ModelSyntax, label: String) -> TextStream {
        TextStream {
            inner,
            said: SaidText::new(syntax, Some(label)),
            last_id: String::new(),
            finished: false,
        }
    }

    /// The chunk with its text as said; `None` when hidden text was all it carried.
    fn read(&mut self, mut chunk: LlmChunk) -> Option<LlmChunk> {
        self.last_id.clone_from(&chunk.id);
        let Some(delta) = chunk.delta.as_mut() else { return Some(chunk) };
        let Some(content) = delta.content.take() else { return Some(chunk) };
        let said = self.said.push(&content);
        // an empty delta stays an empty delta; hidden text is no delta
        delta.content = (!said.is_empty() || content.is_empty()).then_some(said);
        let carries = delta.content.is_some()
            || delta.role.is_some()
            || !delta.tool_calls.is_empty()
            || delta.extra.is_some()
            || chunk.usage.is_some()
            || chunk.flush;
        carries.then_some(chunk)
    }

    /// The provider finished: what the lexer held settles, a cut span dropped.
    fn tail(&mut self) -> Option<LlmChunk> {
        let said = self.said.finish();
        (!said.is_empty()).then(|| LlmChunk {
            id: self.last_id.clone(),
            delta: Some(ChoiceDelta { content: Some(said), ..ChoiceDelta::default() }),
            ..LlmChunk::default()
        })
    }
}

impl StreamOutput<LlmChunk> for TextStream {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            loop {
                if self.finished {
                    return Ok(None);
                }
                match self.inner.next().await? {
                    Some(chunk) => {
                        if let Some(chunk) = self.read(chunk) {
                            return Ok(Some(chunk));
                        }
                    }
                    None => {
                        self.finished = true;
                        return Ok(self.tail());
                    }
                }
            }
        })
    }

    /// Each chunk is read whole once it arrives, so a dropped pull loses no more than the
    /// provider's.
    fn cancel_safe(&self) -> bool {
        self.inner.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.inner.close(deadline)
    }
}

impl ProviderStream<LlmChunk> for TextStream {
    fn started_time(&self) -> Option<f64> {
        self.inner.started_time()
    }

    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        self.inner.take_call_metrics(in_flight)
    }

    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        self.inner.call_ledger()
    }

    fn report_retries(&mut self, report: lk_agents_abi::utils::stream::RetryReport) {
        self.inner.report_retries(report);
    }
}

#[cfg(test)]
#[path = "model_text_tests.rs"]
mod tests;
