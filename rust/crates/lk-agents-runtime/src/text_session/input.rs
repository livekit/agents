//! The model call is owned work; cancellation still reaches the session while it runs.

use super::*;
use lk_agents_net::task::ScopedTask;

pub(super) struct RoutingWork {
    request_id: String,
    _task: ScopedTask,
    result: signal::Cell<Option<SessionEvent>>,
}

impl Driver {
    pub(super) fn route(&mut self, input: lk_agents_core::session::InputRoutingRequest) {
        let request_id = input.request_id.clone();
        let lost_request_id = request_id.clone();
        let delegation = self.delegation.clone();
        let background = self.routing_background.clone();
        let cx = self.cx.clone();
        let options = self.request.conn_options;
        let (done, result) = signal::once();
        let task = self.routing_background.tasks.scoped(
            &*self.spawner,
            "input response routing",
            async move {
                let event =
                    if let Some(model) = delegation.and_then(|options| options.routing_model) {
                        let (result, usage) = lk_agents_inference::delegation::input::classify(
                            model,
                            &background,
                            &cx,
                            input,
                            options,
                        )
                        .await;
                        SessionEvent::InputRouted { request_id, result: Some(result), usage }
                    } else {
                        SessionEvent::InputRouted {
                            request_id,
                            result: Some(Ok(lk_agents_abi::session::InputRoute::NoMatch)),
                            usage: Default::default(),
                        }
                    };
                done.resolve(event);
            },
        );
        self.routing = Some(RoutingWork { request_id: lost_request_id, _task: task, result });
    }
}

pub(super) async fn routed(work: &Option<RoutingWork>) -> SessionEvent {
    let Some(work) = work else { return std::future::pending().await };
    work.result.wait().await.unwrap_or_else(|_| SessionEvent::InputRouted {
        request_id: work.request_id.clone(),
        result: None,
        usage: Default::default(),
    })
}
