//! Incremental snapshot, validation and edit preparation for a requested conversation.

use super::{CtxUpdate, RealtimeMachine, Retired, Step};
use lk_agents_abi::{ChatContext, ChatItem};
use lk_agents_core::llm::ctx_diff::{ChatContextDiff, DiffOps};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

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

    /// One unit of preparation or emission; callers share the page budget with retirement. Breaks
    /// once the preparation is done.
    pub(super) fn advance(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) -> ControlFlow<()> {
        match self.phase {
            Phase::Mirror => self.mirror_next(machine),
            Phase::Edit => self.apply_edit_next(),
            Phase::OldIndex => self.index_old_next(),
            Phase::NewIndex => self.index_new_next(),
            Phase::Held => self.place_held(),
            Phase::Filter => self.filter_next(),
            Phase::FilteredIndex => self.index_filtered_next(),
            Phase::Diff => self.install_diff(machine, update),
            Phase::Delete => self.delete_next(machine, update, step),
            Phase::Create => self.create_next(machine, update, step),
            Phase::UpdateDelete => self.update_delete_next(machine, update, step),
            Phase::UpdateCreate => self.update_create(machine, update, step),
            Phase::Cleanup => return ControlFlow::Break(()),
        }
        ControlFlow::Continue(())
    }

    /// Copies one item of the desired context; a changed revision or a vanished item restarts
    /// the copy from the first item.
    fn mirror_next(&mut self, machine: &mut RealtimeMachine) {
        let remote = &machine.desired;
        if (machine.desired_epoch, remote.revision()) != self.revision {
            if !self.mirror.is_empty() {
                machine.retired.push_back(Retired::Context(std::mem::take(&mut self.mirror)));
            }
            self.revision = (machine.desired_epoch, remote.revision());
            self.mirror_cursor = remote.first_id().map(str::to_owned);
        }
        let Some(id) = self.mirror_cursor.take() else {
            self.old = std::mem::take(&mut self.mirror);
            self.cursor = 0;
            self.phase = if self.edit.is_some() { Phase::Edit } else { Phase::OldIndex };
            return;
        };
        let Some(item) = remote.get(&id) else {
            if !self.mirror.is_empty() {
                machine.retired.push_back(Retired::Context(std::mem::take(&mut self.mirror)));
            }
            self.mirror_cursor = remote.first_id().map(str::to_owned);
            return;
        };
        self.mirror.items.push(item.clone());
        self.mirror_cursor = remote.next_id(&id).map(str::to_owned);
    }

    /// Copies one old item into the input, truncated to its transcript when it is the edited one.
    fn apply_edit_next(&mut self) {
        let Some(item) = self.old.items.get(self.cursor) else {
            self.cursor = 0;
            self.phase = Phase::OldIndex;
            return;
        };
        let edit = self.edit.as_ref().expect("the edit phase runs only for a truncation");
        if item.id() != edit.message_id {
            self.input.items.push(item.clone());
        } else if !edit.has_audio {
            let mut item = item.clone();
            if let Some(message) = item.as_message_mut() {
                let transcript = edit
                    .audio_transcript
                    .clone()
                    .expect("a truncation without audio has a transcript");
                message.content = vec![lk_agents_abi::ChatContent::Text(transcript)];
            }
            self.input.items.push(item);
        }
        self.cursor += 1;
    }

    fn index_old_next(&mut self) {
        let Some(item) = self.old.items.get(self.cursor) else {
            self.cursor = 0;
            self.phase = Phase::NewIndex;
            return;
        };
        self.old_by_id.insert(item.id().to_owned(), self.cursor);
        if let ChatItem::FunctionCall(call) = item {
            self.calls.insert(call.call_id.clone());
        }
        self.cursor += 1;
    }

    /// Indexes one input item; at the end, finds the held item the input dropped.
    fn index_new_next(&mut self) {
        let Some(item) = self.input.items.get(self.cursor) else {
            self.held_old = self.held_id.as_ref().and_then(|id| {
                (!self.input_by_id.contains_key(id))
                    .then(|| self.old_by_id.get(id).copied())
                    .flatten()
            });
            self.cursor = self.held_old.unwrap_or(0);
            self.phase = Phase::Held;
            return;
        };
        self.input_by_id.insert(item.id().to_owned(), self.cursor);
        if let ChatItem::FunctionCall(call) = item {
            self.calls.insert(call.call_id.clone());
        }
        self.cursor += 1;
    }

    /// Walks back from the held item to the nearest one the input keeps: the held item goes after
    /// it, or first.
    fn place_held(&mut self) {
        if self.held_old.is_some() && self.cursor > 0 {
            self.cursor -= 1;
            if let Some(index) = self.input_by_id.get(self.old.items[self.cursor].id()) {
                self.held_at = Some(index + 1);
                self.cursor = 0;
            }
            return;
        }
        if self.held_old.is_some() && self.held_at.is_none() {
            self.held_at = Some(0);
        }
        self.input_items = Some(std::mem::take(&mut self.input.items).into_iter());
        self.cursor = 0;
        self.phase = Phase::Filter;
    }

    /// Emits the held item at its place, or filters one input item: a new empty message and an
    /// output without its call are left out.
    fn filter_next(&mut self) {
        if !self.held_emitted && self.held_at == Some(self.cursor) {
            let held = self.held_old.expect("a held position belongs to a held item");
            self.filtered.items.push(self.old.items[held].clone());
            self.held_emitted = true;
            return;
        }
        let items = self.input_items.as_mut().expect("filtering follows the held placement");
        let Some(item) = items.next() else {
            self.cursor = 0;
            self.phase = Phase::FilteredIndex;
            return;
        };
        let empty_local = matches!(&item, ChatItem::Message(message) if message.content.is_empty())
            && !self.old_by_id.contains_key(item.id());
        let orphan_output = matches!(&item, ChatItem::FunctionCallOutput(output)
            if !self.calls.contains(&output.call_id));
        if !empty_local && !orphan_output {
            self.filtered.items.push(item);
        } else {
            self.filtered_out.push(item);
        }
        self.cursor += 1;
    }

    fn index_filtered_next(&mut self) {
        let Some(item) = self.filtered.items.get(self.cursor) else {
            self.phase = Phase::Diff;
            return;
        };
        self.new_by_id.insert(item.id().to_owned(), self.cursor);
        self.cursor += 1;
    }

    /// Advances the diff; once done, its result becomes the update's target and its operations
    /// are queued.
    fn install_diff(&mut self, machine: &mut RealtimeMachine, update: &mut CtxUpdate) {
        if !self.diff.advance(&self.old, &self.filtered, 1) {
            return;
        }
        let DiffOps { to_remove, to_create, to_update } = self.diff.take_operations();
        self.removes = to_remove.into_iter();
        self.creates = to_create.into_iter();
        self.updates = to_update.into_iter();
        let old = std::mem::replace(&mut update.target, std::mem::take(&mut self.filtered));
        if !old.is_empty() {
            machine.retired.push_back(Retired::Context(old));
        }
        self.installed = true;
        self.phase = Phase::Delete;
    }

    fn delete_next(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) {
        match self.removes.next() {
            Some(id) => self.delete(machine, update, step, id),
            None => self.phase = Phase::Create,
        }
    }

    fn create_next(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) {
        match self.creates.next() {
            Some((previous, id)) => self.create(machine, update, step, previous, id),
            None => self.phase = Phase::UpdateDelete,
        }
    }

    /// An updated item is deleted, then created again; an old empty message is skipped.
    fn update_delete_next(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) {
        let Some((previous, id)) = self.updates.next() else {
            self.phase = Phase::Cleanup;
            return;
        };
        let old_empty = self.old_by_id.get(&id).is_some_and(|&at| {
            matches!(&self.old.items[at], ChatItem::Message(message) if message.content.is_empty())
        });
        if old_empty {
            return;
        }
        self.delete(machine, update, step, id.clone());
        self.update_item = Some((previous, id));
        self.phase = Phase::UpdateCreate;
    }

    fn update_create(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) {
        let (previous, id) =
            self.update_item.take().expect("an update's create follows its delete");
        self.create(machine, update, step, previous, id);
        self.phase = Phase::UpdateDelete;
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
                Some(previous.unwrap_or_else(|| crate::realtime::protocol::ROOT_ITEM.to_owned())),
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
