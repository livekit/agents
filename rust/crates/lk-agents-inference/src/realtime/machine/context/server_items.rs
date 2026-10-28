//! Place unsolicited server items beside local insertions without scanning a history in one step.

use super::{ContextWork, RealtimeMachine, Step};
use crate::realtime::RemoteChatContext;
use lk_agents_abi::{ChatItem, Ts};
use std::collections::VecDeque;

const SERVER_ITEMS: usize = 32;

#[derive(Debug)]
struct ServerItem {
    item_id: String,
    next: Option<String>,
    generation: u64,
}

#[derive(Debug)]
pub(in crate::realtime::machine) struct ServerItemsWork {
    items: VecDeque<ServerItem>,
    source: Option<(u64, RemoteChatContext)>,
    pub(super) prior: Option<Box<ContextWork>>,
    pub(super) replay: bool,
    pub(super) failed: bool,
}

impl ServerItemsWork {
    pub(super) fn advance(&mut self, machine: &mut RealtimeMachine) -> bool {
        let Some(pending) = self.items.front() else { return true };
        let source = self.item_source(pending.generation, &machine.remote);
        if source.get(pending.item_id.as_str()).is_none() {
            self.items.pop_front();
            return false;
        }
        let previous = match &pending.next {
            Some(next) if machine.desired.get(next).is_some() => {
                machine.desired.previous_id(next).map(str::to_owned)
            }
            Some(next) => {
                let next = if source.get(next).is_some() {
                    source.next_id(next)
                } else {
                    source.next_id(pending.item_id.as_str())
                }
                .map(str::to_owned);
                self.items.front_mut().unwrap().next = next;
                return false;
            }
            None => machine.desired.tail_id().map(str::to_owned),
        };
        let item = source.get(pending.item_id.as_str()).unwrap().clone();
        self.items.pop_front();
        let _ = machine.desired.insert(previous.as_deref(), item);
        self.items.is_empty()
    }

    fn item_source<'a>(
        &'a self,
        generation: u64,
        current: &'a RemoteChatContext,
    ) -> &'a RemoteChatContext {
        self.source
            .as_ref()
            .filter(|(epoch, _)| *epoch == generation)
            .map_or(current, |(_, source)| source)
    }

    pub(super) fn contains_call(&self, call_id: &str, current: &RemoteChatContext) -> bool {
        self.items.iter().any(|item| matches!(self.item_source(item.generation, current).get(&item.item_id), Some(ChatItem::FunctionCall(call)) if call.call_id == call_id))
    }

    pub(super) fn reconnect(
        &mut self,
        generation: u64,
        source: RemoteChatContext,
    ) -> Option<RemoteChatContext> {
        self.replay = true;
        if self.source.is_none() {
            self.source = Some((generation, source));
            None
        } else {
            // New framework context edits wait behind this work. A second live source would
            // require another ownership slot; keep it for cleanup and refuse that overload.
            self.failed |= self.items.iter().any(|item| item.generation == generation);
            Some(source)
        }
    }

    pub(super) fn retained_units(&self) -> usize {
        self.items.capacity()
            + self.source.as_ref().map_or(0, |(_, source)| source.retained_units())
            + self.prior.as_ref().map_or(0, |prior| prior.retained_units())
    }
}

impl RealtimeMachine {
    pub(in crate::realtime::machine) fn place_server_item(
        &mut self,
        now: Ts,
        item_id: &str,
    ) -> Step {
        let next = self.remote.next_id(item_id).map(str::to_owned);
        if next.as_ref().is_none_or(|next| self.desired.get(next).is_some()) {
            let previous = match &next {
                Some(next) => self.desired.previous_id(next).map(str::to_owned),
                None => self.desired.tail_id().map(str::to_owned),
            };
            if let Some(item) = self.remote.get(item_id).cloned() {
                let _ = self.desired.insert(previous.as_deref(), item);
            }
            return Step::default();
        }
        let mut update = self.ctx_update.take().unwrap_or_default();
        let mut work = match update.work.take() {
            Some(ContextWork::ServerItems(work)) => work,
            prior => Box::new(ServerItemsWork {
                items: VecDeque::new(),
                source: None,
                prior: prior.map(Box::new),
                replay: false,
                failed: false,
            }),
        };
        work.items.push_back(ServerItem {
            item_id: item_id.to_owned(),
            next,
            generation: self.connection_generation,
        });
        let overloaded = work.items.len() > SERVER_ITEMS;
        update.work = Some(ContextWork::ServerItems(work));
        self.ctx_update = Some(update);
        if overloaded {
            let mut step = self.close(now);
            step.fatal = Some(self.error(now, "realtime server item capacity exceeded", false));
            return step;
        }
        self.continue_context(now)
    }
}
