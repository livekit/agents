//! Runtime-owned disposal of retired context state.

use super::{ContextWork, CtxUpdate, DeferredCommand};
use crate::realtime::RemoteChatContext;
use lk_agents_abi::ChatContext;
use std::collections::VecDeque;

/// An owned cleanup effect. Admission precedes taking it from the machine.
/// The runtime drops it on an admitted worker, even if its receipt is canceled.
#[derive(Debug)]
pub struct ContextRetirement {
    states: Vec<Retired>,
}

#[derive(Debug, Default)]
pub(in crate::realtime::machine) struct RetirementQueue {
    batches: VecDeque<(usize, ContextRetirement)>,
}

impl RetirementQueue {
    pub(in crate::realtime::machine) fn len(&self) -> usize {
        self.batches.len()
    }
    pub(in crate::realtime::machine) fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub(in crate::realtime::machine) fn push_back(&mut self, state: Retired) {
        let units = state.retained_units().max(1);
        if units <= super::CONTEXT_PAGE
            && let Some((size, batch)) = self.batches.back_mut()
            && size.saturating_add(units) <= super::CONTEXT_PAGE
        {
            *size += units;
            batch.states.push(state);
            return;
        }
        self.batches.push_back((units, ContextRetirement { states: vec![state] }));
    }

    fn pop_front(&mut self) -> Option<ContextRetirement> {
        self.batches.pop_front().map(|(_, batch)| batch)
    }
}

impl super::RealtimeMachine {
    pub fn has_context_retirement(&self) -> bool {
        !self.retired.is_empty()
            || self.discarded_context_items.len() >= super::CONTEXT_PAGE
            || (self.closing && !self.discarded_context_items.is_empty())
    }

    /// The caller must hold cleanup worker admission before taking this ownership.
    pub fn take_context_retirement(&mut self) -> Option<ContextRetirement> {
        self.retired.pop_front().or_else(|| {
            (self.discarded_context_items.len() >= super::CONTEXT_PAGE
                || (self.closing && !self.discarded_context_items.is_empty()))
            .then(|| ContextRetirement {
                states: vec![Retired::Context(ChatContext::new(std::mem::take(
                    &mut self.discarded_context_items,
                )))],
            })
        })
    }
}

#[derive(Debug)]
pub(in crate::realtime::machine) enum Retired {
    Context(ChatContext),
    Update(CtxUpdate),
    Work(ContextWork),
    Remote(RemoteChatContext),
    Deferred(VecDeque<DeferredCommand>),
}

impl Retired {
    fn retained_units(&self) -> usize {
        match self {
            Self::Deferred(commands) => commands.capacity(),
            Self::Context(context) => context.items.capacity(),
            Self::Remote(remote) => remote.retained_units(),
            Self::Update(update) => {
                update.target.items.capacity()
                    + update.awaiting_create.len()
                    + update.awaiting_delete.len()
                    + update.by_event_id.len()
                    + update.events_by_item.len()
                    + update.touched.len()
                    + update.tail_creates.len()
                    + update.work.as_ref().map_or(0, ContextWork::retained_units)
            }
            Self::Work(work) => work.retained_units(),
        }
    }
}

impl ContextWork {
    pub(super) fn retained_units(&self) -> usize {
        match self {
            Self::ServerItems(work) => work.retained_units(),
            Self::Prepare(work) => work.retained_units(),
            Self::Replay(work) => work.calls.len() + work.seen.len() + work.empty_allowed.len(),
            Self::Overlay(work) => {
                work.remote.retained_units()
                    + work.target_storage
                    + work.target.items.capacity()
                    + work.nodes.capacity()
                    + work.collected.capacity()
                    + work.retired_items.capacity()
                    + work.retired_sources.capacity()
                    + work.touched.len()
                    + work.target_ids.len()
                    + work.tail_creates.len()
                    + work.positions.len()
                    + work.empty_allowed.len()
                    + work.extra_storage
                    + work.retired.as_ref().map_or(0, |events| {
                        events.creates.len()
                            + events.deletes.len()
                            + events.events.len()
                            + events.touched.len()
                            + events.tails.len()
                            + events.inverse.len()
                    })
            }
        }
    }
}
