//! Incremental snapshot, validation and edit preparation for a requested conversation.

use super::{CtxUpdate, RealtimeMachine, Retired, Step};
use lk_agents_core::llm::ctx_diff::{ChatContextDiff, DiffOps};
use lk_agents_abi::{ChatContext, ChatItem};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy)]
enum Phase {
    Mirror,
    Edit,
    OldIndex,
    NewIndex,
    Held,
    Filter,
    FilteredIndex,
    Diff,
    Delete,
    Create,
    UpdateDelete,
    UpdateCreate,
    Cleanup,
}

#[derive(Debug)]
pub(in crate::realtime::machine) struct PrepareWork {
    phase: Phase,
    input: ChatContext,
    filtered_out: Vec<ChatItem>,
    retired_frames: Vec<super::Outgoing>,
    edit: Option<lk_agents_abi::llm::realtime::Truncate>,
    input_items: Option<std::vec::IntoIter<ChatItem>>,
    input_storage: usize,
    filtered: ChatContext,
    mirror: ChatContext,
    revision: (u64, u64),
    cursor: usize,
    mirror_cursor: Option<String>,
    old: ChatContext,
    old_by_id: BTreeMap<String, usize>,
    input_by_id: BTreeMap<String, usize>,
    new_by_id: BTreeMap<String, usize>,
    calls: BTreeSet<String>,
    held_id: Option<String>,
    held_old: Option<usize>,
    held_at: Option<usize>,
    held_emitted: bool,
    diff: ChatContextDiff,
    removes: std::vec::IntoIter<String>,
    creates: std::vec::IntoIter<(Option<String>, String)>,
    updates: std::vec::IntoIter<(Option<String>, String)>,
    update_item: Option<(Option<String>, String)>,
    pub(super) installed: bool,
    pub(super) replay: bool,
}

impl PrepareWork {
    pub(super) fn new(machine: &RealtimeMachine, input: ChatContext) -> Self {
        let input_storage = input.items.capacity();
        Self {
            phase: Phase::Mirror,
            input,
            filtered_out: Vec::new(),
            retired_frames: Vec::new(),
            edit: None,
            input_items: None,
            input_storage,
            filtered: ChatContext::empty(),
            mirror: ChatContext::empty(),
            revision: (machine.desired_epoch, machine.desired.revision()),
            cursor: 0,
            mirror_cursor: machine.desired.first_id().map(str::to_owned),
            old: ChatContext::empty(),
            old_by_id: BTreeMap::new(),
            input_by_id: BTreeMap::new(),
            new_by_id: BTreeMap::new(),
            calls: BTreeSet::new(),
            held_id: machine.held.as_ref().map(|held| held.item_id.clone()),
            held_old: None,
            held_at: None,
            held_emitted: false,
            diff: ChatContextDiff::default(),
            removes: Vec::new().into_iter(),
            creates: Vec::new().into_iter(),
            updates: Vec::new().into_iter(),
            update_item: None,
            installed: false,
            replay: false,
        }
    }

    pub(super) fn truncate(
        machine: &RealtimeMachine,
        edit: lk_agents_abi::llm::realtime::Truncate,
    ) -> Self {
        let mut work = Self::new(machine, ChatContext::empty());
        work.edit = Some(edit);
        work
    }

    /// One unit of preparation or emission; callers share the page budget with retirement.
    pub(super) fn advance(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) -> bool {
        match self.phase {
            Phase::Mirror => {
                let remote = &machine.desired;
                if (machine.desired_epoch, remote.revision()) != self.revision {
                    if !self.mirror.is_empty() {
                        machine
                            .retired
                            .push_back(Retired::Context(std::mem::take(&mut self.mirror)));
                    }
                    self.revision = (machine.desired_epoch, remote.revision());
                    self.mirror_cursor = remote.first_id().map(str::to_owned);
                }
                if let Some(id) = self.mirror_cursor.take() {
                    let Some(item) = remote.get(&id) else {
                        if !self.mirror.is_empty() {
                            machine
                                .retired
                                .push_back(Retired::Context(std::mem::take(&mut self.mirror)));
                        }
                        self.mirror_cursor = remote.first_id().map(str::to_owned);
                        return false;
                    };
                    self.mirror.items.push(item.clone());
                    self.mirror_cursor = remote.next_id(&id).map(str::to_owned);
                } else {
                    self.old = std::mem::take(&mut self.mirror);
                    self.cursor = 0;
                    self.phase = if self.edit.is_some() { Phase::Edit } else { Phase::OldIndex };
                }
            }
            Phase::Edit => {
                if let Some(item) = self.old.items.get(self.cursor) {
                    let edit = self.edit.as_ref().unwrap();
                    if item.id() != edit.message_id {
                        self.input.items.push(item.clone());
                    } else if !edit.has_audio {
                        let mut item = item.clone();
                        if let Some(message) = item.as_message_mut() {
                            message.content = vec![lk_agents_abi::ChatContent::Text(
                                edit.audio_transcript.clone().unwrap(),
                            )];
                        }
                        self.input.items.push(item);
                    }
                    self.cursor += 1;
                } else {
                    self.cursor = 0;
                    self.phase = Phase::OldIndex;
                }
            }
            Phase::OldIndex => {
                if let Some(item) = self.old.items.get(self.cursor) {
                    self.old_by_id.insert(item.id().to_owned(), self.cursor);
                    if let ChatItem::FunctionCall(call) = item {
                        self.calls.insert(call.call_id.clone());
                    }
                    self.cursor += 1;
                } else {
                    self.cursor = 0;
                    self.phase = Phase::NewIndex;
                }
            }
            Phase::NewIndex => {
                if let Some(item) = self.input.items.get(self.cursor) {
                    self.input_by_id.insert(item.id().to_owned(), self.cursor);
                    if let ChatItem::FunctionCall(call) = item {
                        self.calls.insert(call.call_id.clone());
                    }
                    self.cursor += 1;
                } else {
                    self.held_old = self.held_id.as_ref().and_then(|id| {
                        (!self.input_by_id.contains_key(id))
                            .then(|| self.old_by_id.get(id).copied())
                            .flatten()
                    });
                    self.cursor = self.held_old.unwrap_or(0);
                    self.phase = Phase::Held;
                }
            }
            Phase::Held => {
                if self.held_old.is_some() && self.cursor > 0 {
                    self.cursor -= 1;
                    if let Some(index) = self.input_by_id.get(self.old.items[self.cursor].id()) {
                        self.held_at = Some(index + 1);
                        self.cursor = 0;
                    }
                } else {
                    if self.held_old.is_some() && self.held_at.is_none() {
                        self.held_at = Some(0);
                    }
                    self.input_items = Some(std::mem::take(&mut self.input.items).into_iter());
                    self.cursor = 0;
                    self.phase = Phase::Filter;
                }
            }
            Phase::Filter => {
                if !self.held_emitted && self.held_at == Some(self.cursor) {
                    self.filtered.items.push(self.old.items[self.held_old.unwrap()].clone());
                    self.held_emitted = true;
                } else if let Some(item) = self.input_items.as_mut().unwrap().next() {
                    let empty_local = matches!(&item, ChatItem::Message(message) if message.content.is_empty())
                        && !self.old_by_id.contains_key(item.id());
                    let orphan_output = matches!(&item, ChatItem::FunctionCallOutput(output) if !self.calls.contains(&output.call_id));
                    if !empty_local && !orphan_output {
                        self.filtered.items.push(item);
                    } else {
                        self.filtered_out.push(item);
                    }
                    self.cursor += 1;
                } else {
                    self.cursor = 0;
                    self.phase = Phase::FilteredIndex;
                }
            }
            Phase::FilteredIndex => {
                if let Some(item) = self.filtered.items.get(self.cursor) {
                    self.new_by_id.insert(item.id().to_owned(), self.cursor);
                    self.cursor += 1;
                } else {
                    self.phase = Phase::Diff;
                }
            }
            Phase::Diff => {
                if self.diff.advance(&self.old, &self.filtered, 1) {
                    let DiffOps { to_remove, to_create, to_update } = self.diff.take_operations();
                    self.removes = to_remove.into_iter();
                    self.creates = to_create.into_iter();
                    self.updates = to_update.into_iter();
                    let old =
                        std::mem::replace(&mut update.target, std::mem::take(&mut self.filtered));
                    if !old.is_empty() {
                        machine.retired.push_back(Retired::Context(old));
                    }
                    self.installed = true;
                    self.phase = Phase::Delete;
                }
            }
            Phase::Delete => {
                if let Some(id) = self.removes.next() {
                    self.delete(machine, update, step, id);
                } else {
                    self.phase = Phase::Create;
                }
            }
            Phase::Create => {
                if let Some((previous, id)) = self.creates.next() {
                    self.create(machine, update, step, previous, id);
                } else {
                    self.phase = Phase::UpdateDelete;
                }
            }
            Phase::UpdateDelete => {
                if let Some((previous, id)) = self.updates.next() {
                    if self.old_by_id.get(&id).is_some_and(|&at| {
                        matches!(&self.old.items[at], ChatItem::Message(message)
                            if message.content.is_empty())
                    }) {
                        return false;
                    }
                    self.delete(machine, update, step, id.clone());
                    self.update_item = Some((previous, id));
                    self.phase = Phase::UpdateCreate;
                } else {
                    self.phase = Phase::Cleanup;
                }
            }
            Phase::UpdateCreate => {
                let (previous, id) = self.update_item.take().unwrap();
                self.create(machine, update, step, previous, id);
                self.phase = Phase::UpdateDelete;
            }
            Phase::Cleanup => return true,
        }
        false
    }

    fn delete(
        &self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
        id: String,
    ) {
        if let Some(&at) = self.old_by_id.get(&id)
            && let ChatItem::FunctionCall(call) = &self.old.items[at]
        {
            if machine.deleted_calls.len() == super::super::DELETED_CALLS_REMEMBERED {
                machine.deleted_calls.pop_front();
            }
            machine.deleted_calls.push_back(call.call_id.clone());
        }
        machine.delete_ctx_item(&mut step.frames, update, &id);
    }

    fn create(
        &self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
        previous: Option<String>,
        id: String,
    ) {
        if let Some(&at) = self.new_by_id.get(&id) {
            let item = update.target.items[at].clone();
            if let ChatItem::FunctionCall(call) = &item {
                machine.deleted_calls.retain(|id| *id != call.call_id);
            }
            machine.create_ctx_item(
                &mut step.frames,
                update,
                Some(previous.unwrap_or_else(|| "root".to_owned())),
                item,
            );
        }
    }

    pub(super) fn retire_frame(&mut self, frame: super::Outgoing) {
        self.retired_frames.push(frame);
    }

    pub(super) fn retained_units(&self) -> usize {
        self.input_storage
            + self.input.items.capacity()
            + self.filtered_out.capacity()
            + self.retired_frames.capacity()
            + self.filtered.items.capacity()
            + self.mirror.items.capacity()
            + self.old.items.capacity()
            + self.old_by_id.len()
            + self.input_by_id.len()
            + self.new_by_id.len()
            + self.calls.len()
            + self.diff.retained_units()
    }

    pub(super) fn completed(&self) -> bool {
        !self.replay && matches!(self.phase, Phase::Cleanup)
    }

    pub(super) fn replay_ready(&self) -> bool {
        self.replay && matches!(self.phase, Phase::Cleanup)
    }
}
