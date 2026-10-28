//! Bounded preparation of the conversation edit script.

use super::{DiffOps, message_changed};
use crate::ChatContext;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy)]
enum Phase {
    Prefix,
    OldIndex,
    NewIndex,
    Subsequence,
    Trace,
    Repeated,
    Remove,
    Create,
    Done,
}

#[derive(Debug)]
pub struct ChatContextDiff {
    phase: Phase,
    cursor: usize,
    prefix: usize,
    old_by_id: BTreeMap<String, usize>,
    positions: BTreeMap<String, usize>,
    duplicate_new: bool,
    tails: Vec<usize>,
    tail_items: Vec<usize>,
    previous: Vec<Option<usize>>,
    trace: Option<usize>,
    kept: BTreeSet<String>,
    repeated: Vec<RepeatedRange>,
    row: Option<RepeatedRows>,
    spare_rows: Option<RowsStorage>,
    previous_id: Option<String>,
    operations: DiffOps,
}

impl Default for ChatContextDiff {
    fn default() -> Self {
        Self {
            phase: Phase::Prefix,
            cursor: 0,
            prefix: 0,
            old_by_id: BTreeMap::new(),
            positions: BTreeMap::new(),
            duplicate_new: false,
            tails: Vec::new(),
            tail_items: Vec::new(),
            previous: Vec::new(),
            trace: None,
            kept: BTreeSet::new(),
            repeated: Vec::new(),
            row: None,
            spare_rows: None,
            previous_id: None,
            operations: DiffOps::default(),
        }
    }
}

impl ChatContextDiff {
    /// Each unit inspects at most one item or one repeated-ID dynamic-programming cell.
    /// Supply the same unchanged contexts until this returns `true`.
    pub fn advance(&mut self, old: &ChatContext, new: &ChatContext, budget: usize) -> bool {
        for _ in 0..budget {
            match self.phase {
                Phase::Prefix => match (old.items.get(self.cursor), new.items.get(self.cursor)) {
                    (Some(a), Some(b)) if a.id() == b.id() => {
                        self.kept.insert(a.id().to_owned());
                        self.cursor += 1;
                    }
                    _ => {
                        self.prefix = self.cursor;
                        self.cursor = 0;
                        self.phase = Phase::OldIndex;
                    }
                },
                Phase::OldIndex => {
                    if let Some(item) = old.items.get(self.cursor) {
                        self.old_by_id.insert(item.id().to_owned(), self.cursor);
                        self.cursor += 1;
                    } else {
                        self.cursor = self.prefix;
                        self.phase = Phase::NewIndex;
                    }
                }
                Phase::NewIndex => {
                    if let Some(item) = new.items.get(self.cursor) {
                        self.duplicate_new |=
                            self.positions.insert(item.id().to_owned(), self.cursor).is_some();
                        self.cursor += 1;
                    } else if self.duplicate_new {
                        self.repeated.push(RepeatedRange {
                            old_start: self.prefix,
                            old_end: old.len(),
                            new_start: self.prefix,
                            new_end: new.len(),
                        });
                        self.phase = Phase::Repeated;
                    } else {
                        self.cursor = self.prefix;
                        self.phase = Phase::Subsequence;
                    }
                }
                Phase::Subsequence => {
                    if let Some(item) = old.items.get(self.cursor) {
                        let mut predecessor = None;
                        if let Some(&position) = self.positions.get(item.id()) {
                            let length = self.tails.partition_point(|&tail| tail < position);
                            predecessor = length.checked_sub(1).map(|i| self.tail_items[i]);
                            if length == self.tails.len() {
                                self.tails.push(position);
                                self.tail_items.push(self.cursor);
                            } else {
                                self.tails[length] = position;
                                self.tail_items[length] = self.cursor;
                            }
                        }
                        self.previous.push(predecessor);
                        self.cursor += 1;
                    } else {
                        self.trace = self.tail_items.last().copied();
                        self.phase = Phase::Trace;
                    }
                }
                Phase::Trace => {
                    if let Some(index) = self.trace {
                        self.kept.insert(old.items[index].id().to_owned());
                        self.trace = self.previous[index - self.prefix];
                    } else {
                        self.cursor = 0;
                        self.phase = Phase::Remove;
                    }
                }
                Phase::Repeated => {
                    if let Some(rows) = &mut self.row {
                        if rows.advance(old, new) {
                            let rows = self.row.take().unwrap();
                            let range = rows.range;
                            let middle = range.old_start + (range.old_end - range.old_start) / 2;
                            let split = range.new_start + rows.split;
                            self.spare_rows = Some(rows.storage);
                            self.repeated.push(RepeatedRange {
                                old_start: middle,
                                old_end: range.old_end,
                                new_start: split,
                                new_end: range.new_end,
                            });
                            self.repeated.push(RepeatedRange {
                                old_start: range.old_start,
                                old_end: middle,
                                new_start: range.new_start,
                                new_end: split,
                            });
                        }
                    } else if let Some(range) = self.repeated.pop() {
                        if range.old_start == range.old_end || range.new_start == range.new_end {
                            continue;
                        }
                        if range.old_end - range.old_start == 1 {
                            let id = old.items[range.old_start].id();
                            if new.items[range.new_start].id() == id {
                                self.kept.insert(id.to_owned());
                            } else if range.new_start + 1 < range.new_end {
                                self.repeated.push(RepeatedRange {
                                    new_start: range.new_start + 1,
                                    ..range
                                });
                            }
                        } else {
                            let storage = self.spare_rows.take().unwrap_or_else(|| {
                                RowsStorage::new(range.new_end - range.new_start + 1)
                            });
                            self.row = Some(RepeatedRows::new(range, storage));
                        }
                    } else {
                        self.cursor = 0;
                        self.phase = Phase::Remove;
                    }
                }
                Phase::Remove => {
                    if let Some(item) = old.items.get(self.cursor) {
                        if !self.kept.contains(item.id()) {
                            self.operations.to_remove.push(item.id().to_owned());
                        }
                        self.cursor += 1;
                    } else {
                        self.cursor = 0;
                        self.phase = Phase::Create;
                    }
                }
                Phase::Create => {
                    if let Some(item) = new.items.get(self.cursor) {
                        if !self.kept.contains(item.id()) {
                            self.operations
                                .to_create
                                .push((self.previous_id.clone(), item.id().to_owned()));
                        } else if self
                            .old_by_id
                            .get(item.id())
                            .is_some_and(|&at| message_changed(&old.items[at], item))
                        {
                            self.operations
                                .to_update
                                .push((self.previous_id.clone(), item.id().to_owned()));
                        }
                        self.previous_id = Some(item.id().to_owned());
                        self.cursor += 1;
                    } else {
                        self.phase = Phase::Done;
                    }
                }
                Phase::Done => return true,
            }
        }
        matches!(self.phase, Phase::Done)
    }

    pub fn take_operations(&mut self) -> DiffOps {
        assert!(matches!(self.phase, Phase::Done), "context edit preparation is incomplete");
        std::mem::take(&mut self.operations)
    }

    /// Constant-time storage estimate for deciding whether cleanup can share an admitted batch.
    pub fn retained_units(&self) -> usize {
        self.old_by_id.len()
            + self.positions.len()
            + self.kept.len()
            + self.tails.capacity()
            + self.tail_items.capacity()
            + self.previous.capacity()
            + self.repeated.capacity()
            + self.operations.to_remove.capacity()
            + self.operations.to_create.capacity()
            + self.operations.to_update.capacity()
            + self.row.as_ref().map_or(0, |row| row.storage.capacity())
            + self.spare_rows.as_ref().map_or(0, RowsStorage::capacity)
    }

    /// Release owned identity keys below the same continuation budget as preparation.
    pub fn retire_step(&mut self) -> bool {
        self.operations.to_remove.pop().is_some()
            || self.operations.to_create.pop().is_some()
            || self.operations.to_update.pop().is_some()
            || self.previous_id.take().is_some()
            || self.old_by_id.pop_first().is_some()
            || self.positions.pop_first().is_some()
            || self.kept.pop_first().is_some()
    }
}

#[derive(Debug)]
struct RepeatedRange {
    old_start: usize,
    old_end: usize,
    new_start: usize,
    new_end: usize,
}

#[derive(Debug)]
struct RepeatedRows {
    range: RepeatedRange,
    storage: RowsStorage,
    reverse: bool,
    row: usize,
    column: usize,
    diagonal: usize,
    split_at: usize,
    split: usize,
    longest: usize,
}

#[derive(Debug)]
struct RowsStorage {
    left: Vec<usize>,
    right: Vec<usize>,
}

impl RowsStorage {
    fn new(width: usize) -> Self {
        Self { left: Vec::with_capacity(width), right: Vec::with_capacity(width) }
    }

    fn capacity(&self) -> usize {
        self.left.capacity() + self.right.capacity()
    }
}

impl RepeatedRows {
    fn new(range: RepeatedRange, mut storage: RowsStorage) -> Self {
        // Every child range fits the original rows. Clearing scalar slots retains the two
        // allocations; zero initialization is budgeted below instead of freeing in a step.
        storage.left.clear();
        storage.right.clear();
        Self {
            range,
            storage,
            reverse: false,
            row: 0,
            column: 0,
            diagonal: 0,
            split_at: 0,
            split: 0,
            longest: 0,
        }
    }

    fn advance(&mut self, old: &ChatContext, new: &ChatContext) -> bool {
        let width = self.range.new_end - self.range.new_start;
        if self.storage.left.len() <= width {
            self.storage.left.push(0);
            return false;
        }
        if self.storage.right.len() <= width {
            self.storage.right.push(0);
            return false;
        }
        let middle = self.range.old_start + (self.range.old_end - self.range.old_start) / 2;
        let rows =
            if self.reverse { self.range.old_end - middle } else { middle - self.range.old_start };
        if self.row < rows {
            let old_at = if self.reverse {
                self.range.old_end - 1 - self.row
            } else {
                self.range.old_start + self.row
            };
            let new_at = if self.reverse {
                self.range.new_end - 1 - self.column
            } else {
                self.range.new_start + self.column
            };
            let lengths =
                if self.reverse { &mut self.storage.right } else { &mut self.storage.left };
            let previous = lengths[self.column + 1];
            lengths[self.column + 1] = if old.items[old_at].id() == new.items[new_at].id() {
                self.diagonal + 1
            } else {
                lengths[self.column].max(previous)
            };
            self.diagonal = previous;
            self.column += 1;
            if self.column == width {
                self.row += 1;
                self.column = 0;
                self.diagonal = 0;
            }
            return false;
        }
        if !self.reverse {
            self.reverse = true;
            self.row = 0;
            return false;
        }
        if self.split_at <= width {
            let length =
                self.storage.left[self.split_at] + self.storage.right[width - self.split_at];
            if length > self.longest {
                self.longest = length;
                self.split = self.split_at;
            }
            self.split_at += 1;
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatMessage, Role};

    #[test]
    fn repeated_id_edits_yield_between_cells_and_preserve_anchors() {
        let context = |ids: [&str; 2]| {
            ChatContext::new(
                (0..128)
                    .flat_map(|_| ids)
                    .map(|id| ChatMessage::text(id, Role::User, id, std::time::UNIX_EPOCH).into())
                    .collect(),
            )
        };
        let old = context(["a", "c"]);
        let new = context(["b", "a"]);
        let mut diff = ChatContextDiff::default();
        for _ in 0..1000 {
            assert!(!diff.advance(&old, &new, 1), "repeated-ID work ignored its dispatch budget");
        }
        let mut steps = 1000;
        while !diff.advance(&old, &new, 1) {
            steps += 1;
            assert!(steps < 1_000_000, "the repeated-ID edit did not finish");
        }
        let operations = diff.take_operations();
        assert_eq!(operations.to_remove, vec!["c"; 128]);
        let mut creates = vec![(None, "b".to_owned())];
        creates.extend((1..128).map(|_| (Some("a".to_owned()), "b".to_owned())));
        assert_eq!(operations.to_create, creates);
        assert!(operations.to_update.is_empty());
        while diff.retire_step() {}
    }
}
