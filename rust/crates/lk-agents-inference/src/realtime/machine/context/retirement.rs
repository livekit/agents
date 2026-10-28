//! Runtime-owned incremental disposal of retired context state.

use super::{ContextWork, CtxUpdate, DeferredCommand, OverlayWork, ReplayWork};
use crate::realtime::RemoteChatContext;
use lk_agents_abi::ChatContext;
use std::collections::VecDeque;

/// An owned cleanup effect. Admission precedes taking it from the machine.
/// The runtime advances and finally drops it on an admitted worker, even if its receipt is canceled.
#[derive(Debug)]
pub struct ContextRetirement {
    states: Vec<Retired>,
}

impl ContextRetirement {
    pub fn advance(&mut self, budget: usize) -> bool {
        for _ in 0..budget {
            let Some(state) = self.states.last_mut() else { return true };
            if !state.step() {
                self.states.pop();
            }
        }
        false
    }
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

    pub(super) fn step(&mut self) -> bool {
        match self {
            Self::Deferred(commands) => commands.pop_front().is_some(),
            Self::Context(context) => context.items.pop().is_some(),
            Self::Update(update) => update.retire_step(),
            Self::Work(work) => work.retire_step(),
            Self::Remote(remote) => remote.pop_front().is_some(),
        }
    }
}

impl CtxUpdate {
    fn retire_step(&mut self) -> bool {
        self.target.items.pop().is_some()
            || self.awaiting_create.pop_first().is_some()
            || self.awaiting_delete.pop_first().is_some()
            || self.by_event_id.pop_first().is_some()
            || pop_inverse(&mut self.events_by_item)
            || self.touched.pop_first().is_some()
            || self.tail_creates.pop_first().is_some()
            || self.work.as_mut().is_some_and(ContextWork::retire_step)
    }
}

pub(super) fn pop_inverse(
    inverse: &mut std::collections::BTreeMap<
        (bool, String),
        std::collections::BTreeMap<u64, String>,
    >,
) -> bool {
    let Some(mut entry) = inverse.first_entry() else { return false };
    if entry.get_mut().pop_first().is_none() {
        entry.remove();
    }
    true
}

impl ContextWork {
    pub(super) fn retained_units(&self) -> usize {
        match self {
            Self::ServerItems(work) => work.retained_units(),
            Self::Prepare(work) => work.retained_units(),
            Self::Replay(work) => {
                work.selected.capacity()
                    + work.calls.len()
                    + work.seen.len()
                    + work.empty_allowed.len()
                    + work.retired_items.capacity()
                    + work.retired_update.as_ref().map_or(0, |events| {
                        events.creates.len()
                            + events.deletes.len()
                            + events.events.len()
                            + events.touched.len()
                            + events.tails.len()
                            + events.inverse.len()
                    })
            }
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

    pub(super) fn retire_step(&mut self) -> bool {
        match self {
            Self::ServerItems(work) => work.retire_step(),
            Self::Replay(replay) => replay.retire_step(),
            Self::Overlay(overlay) => overlay.retire_step(),
            Self::Prepare(prepare) => prepare.retire_step(),
        }
    }
}

impl ReplayWork {
    fn retire_step(&mut self) -> bool {
        self.calls.pop_first().is_some()
            || self.seen.pop_first().is_some()
            || self.empty_allowed.pop_first().is_some()
            || self.retired_update.as_mut().is_some_and(super::RetiredContextEvents::step)
            || self.selected.pop().is_some()
            || self.retired_items.pop().is_some()
    }
}

impl OverlayWork {
    pub(super) fn retire_step(&mut self) -> bool {
        if self.remote.retire_step()
            || self.target.items.pop().is_some()
            || self.target_items.as_mut().is_some_and(|items| items.next().is_some())
            || self.touched.pop_first().is_some()
            || self.target_ids.pop_first().is_some()
            || self.tail_creates.pop_first().is_some()
            || self.nodes.pop().is_some()
            || self.positions.pop_first().is_some()
            || self.collected.pop().is_some()
            || self.retired_items.pop().is_some()
            || self.empty_allowed.pop_first().is_some()
            || self.retired.as_mut().is_some_and(super::RetiredContextEvents::step)
        {
            return true;
        }
        if let Some(source) = self.retired_sources.last_mut() {
            if !source.retire_step() {
                self.retired_sources.pop();
            }
            return true;
        }
        if let Some(items) = self.extra.front_mut() {
            if !items.retire_step() {
                self.extra.pop_front();
            }
            return true;
        }
        false
    }
}
