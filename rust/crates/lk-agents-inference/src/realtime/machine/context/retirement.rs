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
    // The count charges only coalesced small owners. A large owner has its own slot
    // and may retain at most CONTEXT_PAGE additional small-owner units.
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
            && let Some((small_units, batch)) = self.batches.back_mut()
            && small_units.saturating_add(units) <= super::CONTEXT_PAGE
        {
            *small_units += units;
            batch.states.push(state);
            return;
        }
        let small_units = if units <= super::CONTEXT_PAGE { units } else { 0 };
        self.batches.push_back((small_units, ContextRetirement { states: vec![state] }));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_owners_have_a_separate_bounded_allowance_beside_large_owners() {
        let mut queue = RetirementQueue::default();
        for _ in 0..4 {
            queue.push_back(Retired::Context(ChatContext::new(Vec::with_capacity(10_000))));
        }
        assert_eq!(queue.len(), 4, "large owners must retain separate slots");
        for _ in 0..super::super::CONTEXT_PAGE {
            queue.push_back(Retired::Context(ChatContext::new(Vec::with_capacity(1))));
        }
        assert_eq!(queue.len(), 4, "a large owner must allow a bounded small-owner tail");
        queue.push_back(Retired::Context(ChatContext::new(Vec::with_capacity(1))));
        assert_eq!(queue.len(), 5, "small-owner coalescing must remain charged and bounded");
    }
}
