use std::sync::Arc;
use std::time::UNIX_EPOCH;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::session::InputRoute;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatMessage, ChatRequest, Llm, Role, ToolChoice,
};
use lk_agents_core::session::{InputRoutingRequest, input_routing};
use lk_agents_core::usage::{Ledger, Measurement, ModelKind};
use lk_agents_net::owned::{Background, Owned};

use crate::telemetry;

/// A detector has its own configured model. An invalid or failed verdict is reported to the
/// core for clarification; no answer is guessed and no frontend model is substituted.
pub async fn classify(
    llm: Arc<dyn Llm>,
    background: &Background,
    cx: &telemetry::Cx,
    input: InputRoutingRequest,
    options: ApiConnectOptions,
) -> (Result<InputRoute, ProviderError>, Usage) {
    let request = ChatRequest {
        chat_ctx: ChatContext::new(vec![
            ChatMessage::text(
                "lk.routing.system",
                Role::System,
                input_routing::INSTRUCTIONS,
                UNIX_EPOCH,
            )
            .into(),
            ChatMessage::text(
                "lk.routing.input",
                Role::User,
                input_routing::prompt(&input.message, &input.history, &input.candidates),
                UNIX_EPOCH,
            )
            .into(),
        ]),
        tools: vec![input_routing::tool()],
        tool_choice: Some(ToolChoice::Named(input_routing::TOOL.into())),
        parallel_tool_calls: Some(false),
        conn_options: options,
        extra: Some(crate::evals::prompts::request_extra(&llm.model())),
        ..Default::default()
    };
    let mut ledger = Ledger::default();
    ledger.start(0, ModelKind::Llm, &llm.label(), &llm.model());
    let mut trace = telemetry::LlmTrace::start(cx, llm.as_ref(), &request);
    let attempt = trace.attempt();
    let result = match crate::llm::chat(llm, trace.cx(), &request, background.clock.clone(), None)
        .await
    {
        Ok(stream) => {
            let mut stream = Owned::stream(background, stream);
            let mut arguments = None;
            let read = async {
                while let Some(chunk) = stream.next().await? {
                    trace.observe(&chunk);
                    if let Some(usage) = &chunk.usage {
                        ledger.report(0, Measurement::from(usage.as_ref()));
                    }
                    for call in chunk.delta.into_iter().flat_map(|delta| delta.tool_calls) {
                        if call.name != input_routing::TOOL || arguments.is_some() {
                            return Err(ApiError::protocol(
                                "routing requires exactly one routing verdict",
                                call.name,
                            )
                            .into());
                        }
                        arguments = Some(call.arguments);
                    }
                }
                Ok::<_, ProviderError>(())
            }
            .await;
            ledger.finish(0, read.is_ok());
            if stream.close().await == lk_agents_abi::Closed::TimedOut {
                log::error!(target: lk_agents_abi::LOG_TARGET, "routing model stream cleanup incomplete");
            }
            read.and_then(|()| {
                let arguments = arguments
                    .ok_or_else(|| ApiError::protocol("routing returned no verdict", ""))?;
                input_routing::parse(&arguments, &input.candidates)
                    .map_err(|error| ApiError::protocol(error.to_string(), arguments).into())
            })
        }
        Err(error) => {
            ledger.finish(0, false);
            Err(error)
        }
    };
    trace.end_attempt(attempt, result.as_ref().err());
    if let Err(error) = &result {
        trace.fail(error);
    }
    trace.end(Default::default());
    (result, ledger.snapshot())
}
