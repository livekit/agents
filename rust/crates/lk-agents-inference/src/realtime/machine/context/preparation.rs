use super::{CtxUpdate, RealtimeMachine, Retired, Step};
use lk_agents_abi::{ChatContext, ChatItem};
use lk_agents_core::llm::ctx_diff::{ChatContextDiff, DiffOps};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

#[derive(Debug)]
enum Phase {
    Mirror(Option<Edit>),
    Edit(Edit),
    OldIndex,
    NewIndex,
    Held { old_index: usize, cursor: usize },
    Placed(Option<HeldPlacement>),
    Filter { items: std::vec::IntoIter<ChatItem>, held: Option<HeldPlacement>, position: usize },
    FilteredIndex,
    Diff,
    Delete,
    Create,
    UpdateDelete,
    UpdateCreate { previous: Option<String>, id: String },
    Cleanup,
}

#[derive(Debug)]
struct Edit {
    message_id: String,
    replacement: Replacement,
}

#[derive(Debug)]
enum Replacement {
    Remove,
    Transcript(String),
}

#[derive(Debug)]
struct HeldPlacement {
    old_index: usize,
    at: usize,
}

#[derive(Debug)]
pub(in crate::realtime::machine) struct PrepareWork {
    phase: Phase,
    input: ChatContext,
    filtered_out: Vec<ChatItem>,
    retired_frames: Vec<super::Outgoing>,
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
    diff: ChatContextDiff,
    removes: std::vec::IntoIter<String>,
    creates: std::vec::IntoIter<(Option<String>, String)>,
    updates: std::vec::IntoIter<(Option<String>, String)>,
    pub(super) installed: bool,
    pub(super) replay: bool,
}

impl PrepareWork {
    pub(super) fn new(machine: &RealtimeMachine, input: ChatContext) -> Self {
        let input_storage = input.items.capacity();
        Self {
            phase: Phase::Mirror(None),
            input,
            filtered_out: Vec::new(),
            retired_frames: Vec::new(),
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
            diff: ChatContextDiff::default(),
            removes: Vec::new().into_iter(),
            creates: Vec::new().into_iter(),
            updates: Vec::new().into_iter(),
            installed: false,
            replay: false,
        }
    }

    pub(super) fn truncate(
        machine: &RealtimeMachine,
        edit: lk_agents_abi::llm::realtime::Truncate,
    ) -> Self {
        let mut work = Self::new(machine, ChatContext::empty());
        let replacement = if edit.has_audio {
            Some(Replacement::Remove)
        } else {
            edit.audio_transcript.map(Replacement::Transcript)
        };
        work.phase = Phase::Mirror(
            replacement.map(|replacement| Edit { message_id: edit.message_id, replacement }),
        );
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
        let phase = std::mem::replace(&mut self.phase, Phase::Cleanup);
        self.phase = match phase {
            Phase::Mirror(edit) => self.mirror_next(machine, edit),
            Phase::Edit(edit) => self.apply_edit_next(edit),
            Phase::OldIndex => self.index_old_next(),
            Phase::NewIndex => self.index_new_next(),
            Phase::Held { old_index, cursor } => self.place_held(old_index, cursor),
            Phase::Placed(held) => self.begin_filter(held),
            Phase::Filter { items, held, position } => self.filter_next(items, held, position),
            Phase::FilteredIndex => self.index_filtered_next(),
            Phase::Diff => self.install_diff(machine, update),
            Phase::Delete => self.delete_next(machine, update, step),
            Phase::Create => self.create_next(machine, update, step),
            Phase::UpdateDelete => self.update_delete_next(machine, update, step),
            Phase::UpdateCreate { previous, id } => {
                self.create(machine, update, step, previous, id);
                Phase::UpdateDelete
            }
            Phase::Cleanup => return ControlFlow::Break(()),
        };
        ControlFlow::Continue(())
    }

    /// Copies one item of the desired context; a changed revision or a vanished item restarts
    /// the copy from the first item.
    fn mirror_next(&mut self, machine: &mut RealtimeMachine, edit: Option<Edit>) -> Phase {
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
            return edit.map_or(Phase::OldIndex, Phase::Edit);
        };
        let Some(item) = remote.get(&id) else {
            if !self.mirror.is_empty() {
                machine.retired.push_back(Retired::Context(std::mem::take(&mut self.mirror)));
            }
            self.mirror_cursor = remote.first_id().map(str::to_owned);
            return Phase::Mirror(edit);
        };
        self.mirror.items.push(item.clone());
        self.mirror_cursor = remote.next_id(&id).map(str::to_owned);
        Phase::Mirror(edit)
    }

    /// Copies one old item into the input, truncated to its transcript when it is the edited one.
    fn apply_edit_next(&mut self, edit: Edit) -> Phase {
        let Some(item) = self.old.items.get(self.cursor) else {
            self.cursor = 0;
            return Phase::OldIndex;
        };
        if item.id() != edit.message_id {
            self.input.items.push(item.clone());
        } else if let Replacement::Transcript(transcript) = &edit.replacement {
            let mut item = item.clone();
            if let Some(message) = item.as_message_mut() {
                message.content = vec![lk_agents_abi::ChatContent::Text(transcript.clone())];
            }
            self.input.items.push(item);
        }
        self.cursor += 1;
        Phase::Edit(edit)
    }

    fn index_old_next(&mut self) -> Phase {
        let Some(item) = self.old.items.get(self.cursor) else {
            self.cursor = 0;
            return Phase::NewIndex;
        };
        self.old_by_id.insert(item.id().to_owned(), self.cursor);
        if let ChatItem::FunctionCall(call) = item {
            self.calls.insert(call.call_id.clone());
        }
        self.cursor += 1;
        Phase::OldIndex
    }

    /// Indexes one input item; at the end, finds the held item the input dropped.
    fn index_new_next(&mut self) -> Phase {
        let Some(item) = self.input.items.get(self.cursor) else {
            let held_old = self.held_id.as_ref().and_then(|id| {
                (!self.input_by_id.contains_key(id))
                    .then(|| self.old_by_id.get(id).copied())
                    .flatten()
            });
            self.cursor = 0;
            return match held_old {
                Some(old_index) => Phase::Held { old_index, cursor: old_index },
                None => Phase::Placed(None),
            };
        };
        self.input_by_id.insert(item.id().to_owned(), self.cursor);
        if let ChatItem::FunctionCall(call) = item {
            self.calls.insert(call.call_id.clone());
        }
        self.cursor += 1;
        Phase::NewIndex
    }

    /// Walk back to the nearest item retained by the input, one item per step.
    fn place_held(&mut self, old_index: usize, cursor: usize) -> Phase {
        if cursor == 0 {
            return self.begin_filter(Some(HeldPlacement { old_index, at: 0 }));
        }
        let cursor = cursor - 1;
        match self.input_by_id.get(self.old.items[cursor].id()) {
            Some(index) => Phase::Placed(Some(HeldPlacement { old_index, at: index + 1 })),
            None => Phase::Held { old_index, cursor },
        }
    }

    fn begin_filter(&mut self, held: Option<HeldPlacement>) -> Phase {
        Phase::Filter {
            items: std::mem::take(&mut self.input.items).into_iter(),
            held,
            position: 0,
        }
    }

    /// Emit the held item at its place, or filter one new empty message or orphan output.
    fn filter_next(
        &mut self,
        mut items: std::vec::IntoIter<ChatItem>,
        held: Option<HeldPlacement>,
        position: usize,
    ) -> Phase {
        let held = match held {
            Some(held) if held.at == position => {
                self.filtered.items.push(self.old.items[held.old_index].clone());
                return Phase::Filter { items, held: None, position };
            }
            other => other,
        };
        let Some(item) = items.next() else {
            self.cursor = 0;
            return Phase::FilteredIndex;
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
        Phase::Filter { items, held, position: position + 1 }
    }

    fn index_filtered_next(&mut self) -> Phase {
        let Some(item) = self.filtered.items.get(self.cursor) else {
            return Phase::Diff;
        };
        self.new_by_id.insert(item.id().to_owned(), self.cursor);
        self.cursor += 1;
        Phase::FilteredIndex
    }

    /// Advances the diff; once done, its result becomes the update's target and its operations
    /// are queued.
    fn install_diff(&mut self, machine: &mut RealtimeMachine, update: &mut CtxUpdate) -> Phase {
        if !self.diff.advance(&self.old, &self.filtered, 1) {
            return Phase::Diff;
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
        Phase::Delete
    }

    fn delete_next(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) -> Phase {
        match self.removes.next() {
            Some(id) => {
                self.delete(machine, update, step, id);
                Phase::Delete
            }
            None => Phase::Create,
        }
    }

    fn create_next(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) -> Phase {
        match self.creates.next() {
            Some((previous, id)) => {
                self.create(machine, update, step, previous, id);
                Phase::Create
            }
            None => Phase::UpdateDelete,
        }
    }

    /// An updated item is deleted, then created again; an old empty message is skipped.
    fn update_delete_next(
        &mut self,
        machine: &mut RealtimeMachine,
        update: &mut CtxUpdate,
        step: &mut Step,
    ) -> Phase {
        let Some((previous, id)) = self.updates.next() else {
            return Phase::Cleanup;
        };
        let old_empty = self.old_by_id.get(&id).is_some_and(|&at| {
            matches!(&self.old.items[at], ChatItem::Message(message) if message.content.is_empty())
        });
        if old_empty {
            return Phase::UpdateDelete;
        }
        self.delete(machine, update, step, id.clone());
        Phase::UpdateCreate { previous, id }
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
