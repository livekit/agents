use super::*;
use std::future::Future;
use std::pin::Pin;

enum LiveRead {
    Checkpoint(super::checkpoint::Write),
    Handler(String, Option<Read>),
    Poll,
}

type HandlerRead<'a> = Pin<Box<dyn Future<Output = (String, Option<Read>)> + Send + 'a>>;

pub(super) async fn next_handler(
    handlers: &mut std::collections::BTreeMap<String, Handler>,
) -> (String, Option<Read>) {
    let readers: Vec<HandlerRead<'_>> = handlers
        .iter_mut()
        .filter(|(_, handler)| !handler.returned())
        .map(|(id, handler)| -> HandlerRead<'_> {
            let id = id.clone();
            Box::pin(async move { (id, handler.next().await) })
        })
        .collect();
    if readers.is_empty() {
        return std::future::pending().await;
    }
    let (read, _, _) = futures_util::future::select_all(readers).await;
    read
}

impl Turn {
    /// Answer only the selected source's checkpoint, then keep every retained report source
    /// serviceable. A source's session write can unblock the current foreground handler.
    pub(super) async fn next_all(
        &self,
        call: &mut Call,
        should_yield: Option<bool>,
        pause: std::time::Duration,
    ) -> Result<Completion, StorageError> {
        if let Some(should_yield) = should_yield {
            let handler =
                call.handler_id.as_ref().and_then(|id| call.handlers.get_mut(id)).ok_or_else(
                    || StorageError::Invalid("the checkpoint lost its handler".into()),
                )?;
            handler.answer(should_yield);
        }
        self.read_live(call, Some(pause)).await
    }

    pub(super) async fn wait_live(
        &self,
        call: &mut Call,
        pause: std::time::Duration,
    ) -> Result<Completion, StorageError> {
        self.read_live(call, Some(pause)).await
    }

    /// One leased executor reads every retained handler. Sources keep their own task identity;
    /// selecting the speaking/model turn never changes ownership of a checkpoint or result.
    async fn read_live(
        &self,
        call: &mut Call,
        pause: Option<std::time::Duration>,
    ) -> Result<Completion, StorageError> {
        let lease =
            call.run.as_ref().and_then(Run::lease).cloned().ok_or_else(|| {
                StorageError::Invalid("live scheduling lost its fenced run".into())
            })?;
        lease.check()?;
        if let Some(completion) = call.deferred.pop_front() {
            return Ok(completion);
        }
        let poll_at = pause.map(|pause| self.clock.now().saturating_add(pause));
        loop {
            // An opaque return has no remaining host work. A native report's producer stays
            // with its handler until that exact report's fenced commit is acknowledged.
            let retired: Vec<_> = call
                .handlers
                .iter()
                .filter(|(_, handler)| handler.returned() && !handler.awaiting_live_commit())
                .map(|(id, _)| id.clone())
                .collect();
            for id in retired {
                let handler = call.handlers.remove(&id).expect("selected retired handler exists");
                if !handler
                    .close(&*self.clock, self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT))
                    .await
                {
                    return Err(StorageError::Invalid(
                        "a returned live handler did not finish its owned output work".into(),
                    ));
                }
            }
            lease.check()?;
            let poll = async {
                match poll_at {
                    Some(at) => self.clock.wait_until(at).await,
                    None => std::future::pending().await,
                }
            };
            let read = tokio::select! {
                biased;
                Some(write) = call.checkpoints.recv() => LiveRead::Checkpoint(write),
                (id, read) = next_handler(&mut call.handlers) => LiveRead::Handler(id, read),
                error = lease.lost() => return Err(error),
                () = poll => LiveRead::Poll,
            };
            let (id, read) = match read {
                LiveRead::Checkpoint(write) => {
                    self.framework_checkpoint(call, write).await?;
                    continue;
                }
                LiveRead::Handler(id, read) => (id, read),
                LiveRead::Poll => {
                    return Ok(Completion::Inbox(call.partition_key.routed_inbox().await?));
                }
            };
            call.handler_id = Some(id.clone());
            lease.check()?;
            if let Some(completion) = self.live_report(call, id, read).await? {
                return Ok(completion);
            }
        }
    }

    pub(super) async fn live_report(
        &self,
        call: &mut Call,
        id: String,
        read: Option<Read>,
    ) -> Result<Option<Completion>, StorageError> {
        match read {
            Some(Read::Session(write)) => {
                let history = write
                    .history
                    .iter()
                    .map(lk_agents_abi::llm::chat_context::protocol::encode_chat_item)
                    .collect();
                let result = call.run()?.checkpoint_session(write.state, history).await;
                let _ = write.reply.send(result.clone());
                result?;
            }
            Some(Read::Exact(output)) => {
                let result = match &self.publisher {
                    Some(publisher) => publisher.exact(&output.task_id, &output.message).await,
                    None => Err(lk_agents_abi::errors::HostError::new(
                        "exact output requires a live A2A observer",
                    )),
                };
                output.delivered.resolve(result);
            }
            Some(Read::Event(event)) => {
                if let Some(publisher) = &self.publisher {
                    publisher.event(&event).await;
                }
            }
            Some(Read::Checkpoint) => {
                return Ok(Some(Completion::Event(HandlerEvent::LiveCheckpoint {
                    request_id: id,
                })));
            }
            Some(Read::Live(report)) => {
                return Ok(Some(Completion::Event(HandlerEvent::Managed(Box::new(report)))));
            }
            Some(Read::Output(output)) => {
                let event = match *output {
                    Ok(output) => self.handler_output(output)?,
                    Err(error) => HandlerEvent::Output(Box::new(
                        lk_agents_core::a2a::live_handler_error(error)?,
                    )),
                };
                let HandlerEvent::Output(output) = event else {
                    unreachable!("handler_output returns its classified output")
                };
                return Ok(Some(Completion::Event(HandlerEvent::LiveOutput {
                    request_id: id,
                    output,
                })));
            }
            None => {
                return Err(StorageError::Invalid(
                    "an owned live handler ended without its result".into(),
                ));
            }
        }
        Ok(None)
    }
}
