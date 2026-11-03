//! The host's side of a session's event stream: each event serialised to the host, each hook
//! call answered under its span.

use super::*;

/// The host's side of the event stream: each event serialised to the host, each hook call answered.
pub(super) struct HostDrain {
    pub(super) events: Option<Arc<dyn SessionEvents>>,
    pub(super) hooks: Option<Arc<dyn SessionHooks>>,
    pub(super) remote: Option<Arc<dyn SessionHosting>>,
    /// The session's view, whose usage a remote client is told of as it moves.
    pub(super) view: tokio::sync::watch::Receiver<Arc<lk_agents_voice::agent_session::SessionView>>,
    pub(super) anchor: lk_agents_abi::WallAnchor,
    pub(super) agents: HostAgents,
    pub(super) spawner: Arc<dyn Spawner>,
    pub(super) tasks: Arc<TaskSet>,
}

impl HostDrain {
    pub(super) async fn run(self, mut runner: SessionRunner) {
        let mut answering = Vec::new();
        let mut forwarded = 0;
        while let Some(ev) = runner.next_host_event().await {
            match ev {
                VoiceHostEvent::Hook(call) => {
                    if let (Some(remote), HookCall::FunctionToolsExecuted { event, .. }) =
                        (&self.remote, &call)
                    {
                        remote.forward_tools_executed(event);
                    }
                    let Some(hooks) = &self.hooks else { continue };
                    let answer = answer_hook(
                        Arc::clone(hooks),
                        call,
                        self.agents.clone(),
                        runner.api().clone(),
                        Arc::clone(&self.spawner),
                        Arc::clone(&self.tasks),
                    );
                    // The host call is made by this first poll, in event order; the rest of the
                    // answer runs on its own task.
                    let mut answer = Box::pin(answer);
                    let first =
                        std::future::poll_fn(|cx| std::task::Poll::Ready(answer.as_mut().poll(cx)))
                            .await;
                    if first.is_pending() {
                        answering.retain(|task| self.tasks.contains(*task));
                        answering.push(self.tasks.spawn(
                            self.spawner.as_ref(),
                            "hook answer",
                            answer,
                        ));
                    }
                }
                ev => {
                    let speeches = Arc::clone(&runner.speeches);
                    let Some(wire) = ev.wire(self.anchor, &|id| speeches.object(id)) else {
                        continue;
                    };
                    // The core accounts before the host sees an event; only a remote client is
                    // told the usage moved (`session_usage_updated`).
                    if let Some(remote) = &self.remote {
                        let view = Arc::clone(&self.view.borrow());
                        if view.usage_revision != forwarded {
                            forwarded = view.usage_revision;
                            remote.forward_usage(&view.usage);
                        }
                    }
                    self.deliver(wire);
                }
            }
        }
        answering.into_iter().for_each(|task| self.tasks.cancel(task));
        // Dropping the runner tells `aclose()` the host has seen everything.
        drop(runner);
        // However the session closed, its remote session host closes with it.
        if let Some(remote) = self.remote {
            remote.aclose().await;
        }
    }

    fn deliver(&self, wire: HostWire) {
        match (&self.remote, &self.events, &wire) {
            (Some(remote), Some(events), HostWire::Error(error)) => {
                // The host's record of the error, else the error's own text.
                let message = events.error_message(error).unwrap_or_else(|e| {
                    log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "error record unreadable");
                    error.error.to_string()
                });
                remote.forward_error(message);
            }
            (Some(remote), ..) => remote.forward(&wire),
            (None, ..) => {}
        }
        if let Some(events) = &self.events {
            events.event(wire);
        }
    }
}

/// Runs the hook `call` asks for through `hooks` and answers it. Calls the host first, before any
/// other await, so a caller polling it once has handed the call to the host loop in event order.
pub(super) async fn answer_hook(
    hooks: Arc<dyn SessionHooks>,
    call: HookCall,
    agents: HostAgents,
    api: Tx,
    spawner: Arc<dyn Spawner>,
    tasks: Arc<TaskSet>,
) {
    use lk_agents_voice::agent::TurnEdit;
    use lk_agents_voice::agent_session::{HookId, HookOutcome};
    let id = call.id();
    let kind = call.kind();
    // A hook runs with its span current: the application's spans inside it nest under it.
    let under =
        |span| lk_agents_abi::telemetry::record::TraceContext { span, ..Default::default() };
    let outcome: Result<HookOutcome, HostError> = match call {
        HookCall::OnEnter { agent, scope, span, .. } => match agents.get(&agent, scope) {
            Some(agent) => {
                hooks.on_enter(&agent, under(span)).await.map(|()| HookOutcome::Returned)
            }
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnExit { agent, scope, span, .. } => match agents.get(&agent, scope) {
            Some(agent) => {
                let outcome =
                    hooks.on_exit(&agent, under(span)).await.map(|()| HookOutcome::Returned);
                // The toolsets close after the answer, and outlive the hook's cancellation.
                let scopes = hooks.toolset_scopes().clone();
                let close = async move {
                    if let Err(error) = scopes.release(Some(scope)).await {
                        log::warn!("the agent toolsets could not close: {error}");
                    }
                };
                tasks.spawn(spawner.as_ref(), "agent toolsets close", close);
                outcome
            }
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnUserTurnCompleted {
            agent,
            scope,
            chat_ctx,
            new_transcript,
            transcript_confidence,
            span,
            ..
        } => match agents.get(&agent, scope) {
            Some(agent) => hooks
                .on_user_turn_completed(
                    &agent,
                    chat_ctx,
                    &new_transcript,
                    transcript_confidence,
                    under(span),
                )
                .await
                .map(|(user_text, items, stop)| {
                    let mut edits = Vec::new();
                    if stop {
                        edits.push(TurnEdit::StopResponse);
                    }
                    if let Some(user_text) = user_text.filter(|t| *t != new_transcript) {
                        edits.push(TurnEdit::UpdateUserMessage(user_text));
                    }
                    if let Some(items) = items {
                        edits.push(TurnEdit::ReplaceItems(items));
                    }
                    HookOutcome::Turn(edits)
                }),
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnUserTurnExceeded { agent, scope, event, .. } => match agents.get(&agent, scope)
        {
            Some(agent) => {
                hooks.on_user_turn_exceeded(&agent, &event).await.map(|()| HookOutcome::Returned)
            }
            None => Ok(HookOutcome::Failed),
        },
        HookCall::FunctionToolsExecuted { event, .. } => {
            use lk_agents_abi::ChatItem;
            let calls = event.function_calls.into_iter().map(ChatItem::FunctionCall).collect();
            let outputs =
                event.function_call_outputs.into_iter().map(ChatItem::FunctionCallOutput).collect();
            hooks
                .function_tools_executed(calls, outputs)
                .await
                .map(|reply_required| HookOutcome::FunctionToolsExecuted { reply_required })
        }
        // Unanswered: the reference's templates.
        HookCall::FillerSource { source: None, .. } => {
            Ok(HookOutcome::Filler { text: None, spoke: false })
        }
        HookCall::FillerSource { source: Some(source), step, .. } => {
            hooks.filler_source(&source, step).await.map(|reply| match reply {
                FillerReply::Text(text) => HookOutcome::Filler { text: Some(text), spoke: false },
                FillerReply::Spoke => HookOutcome::Filler { text: None, spoke: true },
                FillerReply::Declined => HookOutcome::Filler { text: None, spoke: false },
            })
        }
    };
    // A raised hook carries its exception, which its span records.
    let outcome = outcome.unwrap_or_else(|e| {
        hooks.hook_failed(kind.as_str(), e.host_ref.clone());
        HookOutcome::Raised(Box::new(e))
    });
    let _ =
        api.send_from(VoiceCommandOrigin::Host, Command::HookDone { id: HookId(id.0), outcome });
}
