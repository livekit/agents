//! A task driver serializes exchange replacement, native continuations and reconciliation.

use super::*;

impl Delegation {
    pub(super) async fn send(
        &mut self,
        message: Message,
        origin: Option<lk_agents_abi::session::ToolCallRef>,
        input_request: Option<lk_agents_abi::session::InputRequestRef>,
    ) -> Result<Option<Owned<Reply>>, AgentClientError> {
        let message = MessageInput::Message(Box::new(message));
        let message = lk_agents_net::a2a_client::host::message_to_wire(message)?;
        let observer = self.observer_for_input(
            lk_agents_net::strings::shortuuid("request_"),
            origin,
            input_request,
        );
        let reply = self
            .client
            .send_observed(message, None, None, None, &self.request.cx, observer)
            .await
            .map_err(AgentClientError::from)?;
        self.claim_reply(reply).await
    }

    /// `None` means the peer admitted work onto an already-owned task. Its duplicate local
    /// stream has closed; the existing driver retains progress and result ownership.
    pub(super) async fn claim_reply(
        &mut self,
        reply: Reply,
    ) -> Result<Option<Owned<Reply>>, AgentClientError> {
        let reply = self.own_reply(reply);
        if !reply.task_id.is_empty() {
            let task = lk_agents_abi::delegation::remote::RemoteTaskRef {
                task_id: reply.task_id.clone(),
                context_id: reply.thread_id.clone(),
            };
            if let Some(driver) = self.driver_elsewhere(&task).await? {
                close_observation(reply).await?;
                driver.reconcile().await;
                return Ok(None);
            }
        }
        Ok(Some(reply))
    }

    pub(super) async fn driver_elsewhere(
        &mut self,
        task: &lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> Result<Option<Arc<dyn RemoteTaskDriver>>, AgentClientError> {
        if let Some(owner) = &self.request.owner {
            let endpoint = self.interaction.0.lock().policy.snapshot().id.endpoint;
            if let Some(existing) =
                owner.remote_driver(&endpoint, &endpoint, task).await.map_err(input_error)?
            {
                let ours: Arc<dyn RemoteTaskDriver> = self.driver.clone();
                if !Arc::ptr_eq(&existing, &ours) {
                    return Ok(Some(existing));
                }
            }
        }
        Ok(None)
    }

    pub(super) async fn message_command(
        &mut self,
        command: driver::Command,
        reply: &mut Option<Owned<Reply>>,
    ) -> Result<Option<RemoteEvent>, AgentClientError> {
        let response_claim = command.response_claim;
        let message = match &command.message {
            driver::Message::Input(input) => match self.interaction.task_message(input) {
                Ok(message) => Some(message),
                Err(error) => {
                    drop(response_claim);
                    command.done.resolve(Err(error));
                    return Ok(None);
                }
            },
            driver::Message::Resume { .. } | driver::Message::Reconcile => None,
        };
        // The old source is fully closed before another exchange starts, so an artifact delta
        // can never be observed by two local response streams for this task.
        if let Some(previous) = reply.take() {
            close_observation(previous).await?;
        }
        let sent = match command.message {
            driver::Message::Input(input) => {
                self.send(
                    message
                        .expect("an ordinary task message was built before closing the old reply"),
                    command.origin,
                    input.input_request,
                )
                .await
            }
            driver::Message::Resume { request_id, request, streaming, input_request } => {
                let observer =
                    self.observer_for_input(request_id, command.origin, input_request)
                        .ok_or_else(|| protocol("a recovered send needs its native owner"))?;
                match self.client.resume_send(*request, streaming, &self.request.cx, observer).await
                {
                    Ok(reply) => self.claim_reply(reply).await,
                    Err(error) => Err(error.into()),
                }
            }
            driver::Message::Reconcile => {
                drop(response_claim);
                let event = self.reconcile_task().await;
                // The read belongs to this original call. The duplicate operation that
                // requested observation was already admitted and must not become a retry.
                command.done.resolve(Ok(()));
                return event.map(Some);
            }
        };
        drop(response_claim);
        match sent {
            Ok(Some(sent)) => {
                *reply = Some(sent);
                command.done.resolve(Ok(()));
                Ok(None)
            }
            Ok(None) => {
                command.done.resolve(Ok(()));
                Ok(Some(forwarded()))
            }
            Err(error) => {
                command.done.resolve(Err(error));
                if !self.commands.is_empty() {
                    // A failed input does not discard a different, already-admitted message.
                    return Ok(None);
                }
                // A send may have committed before losing its acknowledgement, or the task
                // may have ended while the old stream closed. Reconcile; never resend it.
                self.reconcile_task().await.map(Some)
            }
        }
    }

    /// Retrying a local wait never creates a second observation of an already-owned task.
    /// An undispatched operation keeps its exact saved message and journal identity.
    pub(super) async fn recover_live(
        &mut self,
        recovery: &RemoteRecovery,
    ) -> Result<bool, AgentClientError> {
        let task = match recovery {
            RemoteRecovery::GetTask(task) => task.clone(),
            RemoteRecovery::Dispatch { request, .. } => match request.as_ref() {
                A2ARequest::Send(send) => match (&send.message.task_id, &send.message.context_id) {
                    (Some(task_id), Some(context_id)) => {
                        lk_agents_abi::delegation::remote::RemoteTaskRef {
                            task_id: task_id.clone(),
                            context_id: context_id.clone(),
                        }
                    }
                    _ => return Ok(false),
                },
                _ => return Ok(false),
            },
            RemoteRecovery::Observed(_) | RemoteRecovery::Uncertain => return Ok(false),
        };
        let Some(owner) = &self.request.owner else { return Ok(false) };
        let endpoint = self.interaction.0.lock().policy.snapshot().id.endpoint;
        let Some(driver) =
            owner.remote_driver(&endpoint, &endpoint, &task).await.map_err(input_error)?
        else {
            return Ok(false);
        };
        match recovery {
            RemoteRecovery::Dispatch { request_id, request, streaming, input_request } => {
                let A2ARequest::Send(send) = request.as_ref() else {
                    return Err(protocol("a recovered task message needs its saved send"));
                };
                driver
                    .resume(
                        request_id.clone(),
                        send.clone(),
                        *streaming,
                        input_request.clone(),
                        self.request.origin.clone(),
                    )
                    .await?;
            }
            RemoteRecovery::GetTask(_) => driver.reconcile().await,
            RemoteRecovery::Observed(_) | RemoteRecovery::Uncertain => {}
        }
        Ok(true)
    }

    async fn reconcile_task(&mut self) -> Result<RemoteEvent, AgentClientError> {
        let state = self.interaction.0.lock().policy.snapshot();
        let task = lk_agents_core::a2a::rpc::TaskRef {
            id: state.pending_task_id.ok_or_else(|| protocol("the live task lost its id"))?,
            context_id: state
                .context_id
                .ok_or_else(|| protocol("the live task lost its context"))?,
        };
        let observer = self.observer();
        let task = self.client.get_task(task, &self.request.cx, observer).await?;
        self.interaction.reconciled(a2a::StreamResponse::Task(task))
    }

    pub(super) fn observer(
        &mut self,
    ) -> Option<Arc<dyn lk_agents_net::a2a_client::observer::RequestObserver>> {
        self.observer_for(lk_agents_net::strings::shortuuid("request_"))
    }

    pub(super) fn observer_for(
        &mut self,
        request_id: String,
    ) -> Option<Arc<dyn lk_agents_net::a2a_client::observer::RequestObserver>> {
        let input = self.interaction.0.lock().policy.snapshot().input_request;
        self.observer_for_input(request_id, None, input)
    }

    pub(super) fn observer_for_input(
        &mut self,
        request_id: String,
        origin: Option<lk_agents_abi::session::ToolCallRef>,
        input_request: Option<lk_agents_abi::session::InputRequestRef>,
    ) -> Option<Arc<dyn lk_agents_net::a2a_client::observer::RequestObserver>> {
        let owner = self.request.owner.clone()?;
        let state = self.interaction.0.lock().policy.snapshot();
        let endpoint = state.id.endpoint;
        let driver: Arc<dyn RemoteTaskDriver> = self.driver.clone();
        owner.register_remote_driver(&endpoint, &endpoint, &request_id, Arc::downgrade(&driver));
        let journal = Arc::new(journal::Journal::new(
            owner,
            endpoint.clone(),
            endpoint,
            request_id,
            origin.or_else(|| self.request.origin.clone()),
            input_request,
        ));
        self.journal = Some(journal.clone());
        Some(journal)
    }

    pub(super) fn own_reply(&self, reply: Reply) -> Owned<Reply> {
        Owned::new(
            &self.background,
            reply,
            Box::new(|reply, deadline| Box::pin(async move { reply.aclose(deadline).await })),
        )
    }
}
