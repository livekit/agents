//! Desired and acknowledged realtime contexts, operation preparation and replay.

use super::{Outgoing, RealtimeMachine, Step};
use crate::realtime::protocol::ROOT_ITEM;
use lk_agents_abi::{ChatContext, ChatItem, Ts};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

mod preparation;
use preparation::PrepareWork;
mod server_items;
use server_items::ServerItemsWork;
mod retirement;
pub use retirement::ContextRetirement;
pub(super) use retirement::{Retired, RetirementQueue};

impl RealtimeMachine {
    pub(super) fn discard_context_item(&mut self, item: ChatItem) {
        if self.discarded_context_items.len() == CONTEXT_PAGE {
            self.retired.push_back(Retired::Context(ChatContext::new(std::mem::take(
                &mut self.discarded_context_items,
            ))));
        }
        self.discarded_context_items.push(item);
    }

    pub(super) fn enforce_retirement_capacity(&mut self, now: Ts, step: &mut Step) -> bool {
        if !self.closing && self.retired.len() > RETIRED_CONTEXTS {
            step.merge(self.close(now));
            step.fatal =
                Some(self.error(now, "realtime context retirement capacity exceeded", false));
            return true;
        }
        false
    }

    /// A function output must name a call already submitted in the intended conversation.
    pub(super) fn holds_call(&self, call_id: &str) -> bool {
        !self.deleted_calls.iter().any(|id| id == call_id)
            && (self.desired.contains_call(call_id)
                || self.ctx_update.as_ref().and_then(|update| update.work.as_ref()).is_some_and(
                    |work| {
                        matches!(work, ContextWork::ServerItems(server)
                        if server.contains_call(call_id, &self.remote))
                    },
                ))
    }

    pub(super) fn update_ctx(&mut self, now: Ts, chat_ctx: ChatContext) -> Step {
        self.update_ctx_budget(now, chat_ctx, CONTEXT_PAGE)
    }

    fn update_ctx_budget(&mut self, now: Ts, chat_ctx: ChatContext, budget: usize) -> Step {
        self.deferred_commands.retain(|command| !matches!(command, DeferredCommand::Item(_)));
        if self.ctx_update.as_ref().is_some_and(|update| {
            matches!(update.work, Some(ContextWork::Overlay(_) | ContextWork::ServerItems(_)))
        }) {
            if let Some(previous) = self.pending_context.replace(chat_ctx) {
                self.retired.push_back(Retired::Context(previous));
            }
            return self.continue_context_budget(now, budget);
        }
        let mut update = self.ctx_update.take().unwrap_or_default();
        if let Some(work) = update.work.take() {
            self.retired.push_back(Retired::Work(work));
        }
        if update.target.is_empty()
            && update.awaiting_create.is_empty()
            && self.desired.is_empty()
            && self.held.is_none()
        {
            update.target = chat_ctx;
            update.work = Some(ContextWork::Replay(Box::new(ReplayWork::new(false))));
        } else {
            update.work = Some(ContextWork::Prepare(Box::new(PrepareWork::new(self, chat_ctx))));
        }
        self.ctx_update = Some(update);
        self.continue_context_budget(now, budget)
    }

    pub(super) fn truncate_ctx(
        &mut self,
        now: Ts,
        edit: lk_agents_abi::llm::realtime::Truncate,
        budget: usize,
    ) -> Step {
        let mut update = self.ctx_update.take().unwrap_or_default();
        update.work = Some(ContextWork::Prepare(Box::new(PrepareWork::truncate(self, edit))));
        self.ctx_update = Some(update);
        self.continue_context_budget(now, budget)
    }

    pub(super) fn delete_ctx_item(
        &mut self,
        frames: &mut Vec<Outgoing>,
        update: &mut CtxUpdate,
        id: &str,
    ) {
        if let Ok(item) = self.desired.delete(id) {
            self.discard_context_item(item);
        }
        let event_id = self.ids.short(DELETE_EVENT_PREFIX);
        frames.push(Outgoing::ItemDelete { event_id: event_id.clone(), item_id: id.to_owned() });
        update.touched.insert(id.to_owned());
        update.awaiting_delete.insert(id.to_owned());
        update.register_event(event_id, CtxOp::Delete, id.to_owned(), self.ids.next.get());
    }

    pub(super) fn create_ctx_item(
        &mut self,
        frames: &mut Vec<Outgoing>,
        update: &mut CtxUpdate,
        previous_item_id: Option<String>,
        item: ChatItem,
    ) {
        let id = item.id().to_owned();
        let anchor = match previous_item_id.as_deref() {
            Some(ROOT_ITEM) => None,
            Some(id) => Some(id.to_owned()),
            None => self.desired.tail_id().map(str::to_owned),
        };
        if let Ok(item) = self.desired.delete(&id) {
            self.discard_context_item(item);
        }
        let _ = self.desired.insert(anchor.as_deref(), item.clone());
        let event_id = self.ids.short(CREATE_EVENT_PREFIX);
        frames.push(Outgoing::ItemCreate { event_id: event_id.clone(), item, previous_item_id });
        update.touched.insert(id.clone());
        update.awaiting_create.insert(id.clone());
        update.register_event(event_id, CtxOp::Create, id, self.ids.next.get());
    }

    pub(super) fn ctx_event_is_latest(&self, id: &str, sequence: u64) -> bool {
        !self.ctx_update.as_ref().is_some_and(|update| {
            [CtxOp::Create, CtxOp::Delete].into_iter().any(|kind| {
                update
                    .events_by_item
                    .get(&(kind, id.to_owned()))
                    .and_then(|events| events.last_key_value())
                    .is_some_and(|(&latest, _)| latest > sequence)
            })
        })
    }

    pub(super) fn reject_ctx_item(&mut self, now: Ts, op: CtxOp, id: &str, sequence: u64) -> Step {
        if !self.ctx_event_is_latest(id, sequence) {
            return Step::default();
        }
        if op == CtxOp::Create
            && let Ok(item) = self.desired.delete(id)
        {
            self.discard_context_item(item);
        }
        let delete_pending =
            self.ctx_update.as_ref().is_some_and(|update| update.awaiting_delete.contains(id));
        if self.desired.get(id).is_none()
            && (op == CtxOp::Delete || !delete_pending)
            && self.remote.get(id).is_some()
        {
            return self.place_server_item(now, id);
        }
        Step::default()
    }

    pub(super) fn settle_ctx_item(&mut self, op: CtxOp, item_id: &str) -> Step {
        self.settle_ctx_event(op, item_id, None)
    }

    pub(super) fn settle_ctx_event(
        &mut self,
        op: CtxOp,
        item_id: &str,
        sequence: Option<u64>,
    ) -> Step {
        let Some(update) = &mut self.ctx_update else { return Step::default() };
        let key = (op, item_id.to_owned());
        let Some(events) = update.events_by_item.get_mut(&key) else { return Step::default() };
        let event_id = match sequence {
            Some(sequence) => events.remove(&sequence),
            None => events.pop_first().map(|(_, id)| id),
        };
        if let Some(event_id) = event_id {
            update.by_event_id.remove(&event_id);
        }
        if events.is_empty() {
            update.events_by_item.remove(&key);
            let set = match op {
                CtxOp::Delete => &mut update.awaiting_delete,
                CtxOp::Create => &mut update.awaiting_create,
            };
            set.remove(item_id);
            if !update.awaiting_create.contains(item_id)
                && !update.awaiting_delete.contains(item_id)
            {
                update.touched.remove(item_id);
            }
        }
        if !update.pending() {
            let update = self.ctx_update.take().expect("the update being settled is parked");
            self.retired.push_back(Retired::Update(update));
        }
        Step::default()
    }
}

/// Which half of a context edit an item event is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CtxOp {
    Create,
    Delete,
}

/// The client `event_id` prefixes of item creates and deletes.
const CREATE_EVENT_PREFIX: &str = "chat_ctx_create_";
const DELETE_EVENT_PREFIX: &str = "chat_ctx_delete_";

#[derive(Debug, Default)]
pub(super) struct CtxUpdate {
    /// `_item_create_future` / `_item_delete_future`, keyed by item id.
    pub(super) awaiting_create: BTreeSet<String>,
    pub(super) awaiting_delete: BTreeSet<String>,
    /// `_chat_ctx_event_futures`, keyed by client `event_id`: an update sends a delete and a
    /// create under the same item id, so only the event id says which one a rejection answers.
    pub(super) by_event_id: BTreeMap<String, (CtxOp, String, u64)>,
    events_by_item: BTreeMap<(CtxOp, String), BTreeMap<u64, String>>,
    /// The context it brings the server to, and the items it deletes or creates: what a
    /// reconnection replays when the lost connection had not confirmed it.
    pub(super) target: ChatContext,
    pub(super) touched: BTreeSet<String>,
    tail_creates: BTreeSet<String>,
    work: Option<ContextWork>,
}

impl CtxUpdate {
    fn register_event(&mut self, event_id: String, op: CtxOp, id: String, sequence: u64) {
        self.events_by_item.entry((op, id.clone())).or_default().insert(sequence, event_id.clone());
        self.by_event_id.insert(event_id, (op, id, sequence));
    }

    pub(super) fn pending(&self) -> bool {
        self.work.is_some() || !self.awaiting_create.is_empty() || !self.awaiting_delete.is_empty()
    }
}

/// Context preparation and submission share a bounded continuation budget.
pub(super) const CONTEXT_PAGE: usize = 64;
// Continuations pause before consuming their completion allowance. External updates already
// transfer ownership through synchronous send; replacing large parked inputs can instead exhaust
// this cap. That overload closes the session, settles pending replies, and retains every owner
// for admitted cleanup. Retaining unlimited replacements would defeat the ownership bound.
const RETIRED_CONTEXTS: usize = 4;
const QUEUED_CONTEXT_ITEMS: usize = 32;

#[derive(Debug)]
pub(super) enum ContextWork {
    Replay(Box<ReplayWork>),
    Overlay(Box<OverlayWork>),
    ServerItems(Box<ServerItemsWork>),
    Prepare(Box<PrepareWork>),
}

impl ContextWork {
    /// Finished work, retired at the end of a continuation: an emitted replay or a completed
    /// preparation.
    fn is_complete(&self, target_len: usize) -> bool {
        match self {
            Self::Replay(replay) => {
                matches!(replay.phase, ReplayPhase::Emit) && replay.cursor == target_len
            }
            Self::Prepare(prepare) => prepare.completed(),
            Self::Overlay(_) | Self::ServerItems(_) => false,
        }
    }
}

/// Why a continuation stopped advancing the update before its budget ran out.
enum UpdateHalt {
    ServerItemsFailed,
    ReplayAfterServerItems,
}

#[derive(Debug, Default)]
struct RetiredContextEvents {
    creates: BTreeSet<String>,
    deletes: BTreeSet<String>,
    events: BTreeMap<String, (CtxOp, String, u64)>,
    touched: BTreeSet<String>,
    tails: BTreeSet<String>,
    inverse: BTreeMap<(CtxOp, String), BTreeMap<u64, String>>,
}

impl RetiredContextEvents {
    fn take(update: &mut CtxUpdate) -> Self {
        Self {
            creates: std::mem::take(&mut update.awaiting_create),
            deletes: std::mem::take(&mut update.awaiting_delete),
            events: std::mem::take(&mut update.by_event_id),
            touched: std::mem::take(&mut update.touched),
            tails: std::mem::take(&mut update.tail_creates),
            inverse: std::mem::take(&mut update.events_by_item),
        }
    }
}

#[derive(Debug)]
pub(super) struct ReplayWork {
    phase: ReplayPhase,
    cursor: usize,
    emitted: usize,
    calls: BTreeSet<String>,
    seen: BTreeSet<String>,
    preserve_empty: bool,
    empty_allowed: BTreeSet<String>,
    previous: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum ReplayPhase {
    Calls,
    Emit,
}

impl ReplayWork {
    fn new(preserve_empty: bool) -> Self {
        Self {
            phase: ReplayPhase::Calls,
            cursor: 0,
            emitted: 0,
            calls: BTreeSet::new(),
            seen: BTreeSet::new(),
            preserve_empty,
            empty_allowed: BTreeSet::new(),
            previous: None,
        }
    }

    /// Notes the next target item's call, or moves to emitting.
    fn index_call(&mut self, target: &ChatContext) -> Progress {
        let Some(item) = target.items.get(self.cursor) else {
            self.phase = ReplayPhase::Emit;
            self.cursor = 0;
            return Progress::Moved;
        };
        if let ChatItem::FunctionCall(call) = item {
            self.calls.insert(call.call_id.clone());
        }
        self.cursor += 1;
        Progress::Unit
    }

    /// Whether `item` is replayed: not an empty message (unless kept), not an output without its
    /// call, and not seen before.
    fn selects(&mut self, item: &ChatItem) -> bool {
        let empty = matches!(item, ChatItem::Message(message) if message.content.is_empty());
        let orphan = matches!(item, ChatItem::FunctionCallOutput(output)
            if !self.calls.contains(&output.call_id));
        let valid =
            (!empty || self.preserve_empty || self.empty_allowed.contains(item.id())) && !orphan;
        valid && self.seen.insert(item.id().to_owned())
    }
}

#[derive(Debug)]
struct ContextNode {
    item: Option<ChatItem>,
    next: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum OverlayPhase {
    Index,
    Remote,
    Target,
    Extra,
    Collect,
    Cleanup,
}

/// An owned overlay: each item moves at most twice, and no history is cloned in one step.
#[derive(Debug)]
enum ProjectionItems {
    Context { items: std::vec::IntoIter<ChatItem>, capacity: usize },
    Mirror { remote: crate::realtime::RemoteChatContext, cursor: Option<String>, remaining: usize },
}

impl From<ChatContext> for ProjectionItems {
    fn from(context: ChatContext) -> Self {
        let capacity = context.items.capacity();
        Self::Context { items: context.items.into_iter(), capacity }
    }
}

impl From<crate::realtime::RemoteChatContext> for ProjectionItems {
    fn from(remote: crate::realtime::RemoteChatContext) -> Self {
        let cursor = remote.first_id().map(str::to_owned);
        let remaining = remote.len();
        Self::Mirror { remote, cursor, remaining }
    }
}

impl ProjectionItems {
    fn len(&self) -> usize {
        match self {
            Self::Context { items, .. } => items.len(),
            Self::Mirror { remaining, .. } => *remaining,
        }
    }

    fn retained_units(&self) -> usize {
        match self {
            Self::Context { capacity, .. } => *capacity,
            Self::Mirror { remote, .. } => remote.retained_units(),
        }
    }
}

impl Iterator for ProjectionItems {
    type Item = ChatItem;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Context { items, .. } => items.next(),
            Self::Mirror { remote, cursor, remaining } => {
                let id = cursor.take()?;
                let item = remote.get(&id).unwrap().clone();
                *cursor = remote.next_id(&id).map(str::to_owned);
                *remaining -= 1;
                Some(item)
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct OverlayWork {
    phase: OverlayPhase,
    remote: ProjectionItems,
    target: ChatContext,
    target_items: Option<std::vec::IntoIter<ChatItem>>,
    target_storage: usize,
    cursor: usize,
    touched: BTreeSet<String>,
    unsent_from: Option<usize>,
    target_ids: BTreeMap<String, bool>,
    tail_creates: BTreeSet<String>,
    nodes: Vec<ContextNode>,
    positions: BTreeMap<String, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    previous: Option<usize>,
    collect_at: Option<usize>,
    collected: Vec<ChatItem>,
    extra: std::collections::VecDeque<ProjectionItems>,
    extra_storage: usize,
    preserve_empty: bool,
    empty_allowed: BTreeSet<String>,
    retired: Option<RetiredContextEvents>,
    retired_items: Vec<ChatItem>,
    retired_sources: Vec<ProjectionItems>,
}

/// What an overlay lays over the remote context: the unconfirmed update's target and edits.
#[derive(Debug)]
struct OverlaySeed {
    target: ChatContext,
    touched: BTreeSet<String>,
    unsent_from: Option<usize>,
    tail_creates: BTreeSet<String>,
    preserve_empty: bool,
    empty_allowed: BTreeSet<String>,
}

impl OverlaySeed {
    /// No update to lay over: the remote context is replayed as it is.
    fn replay_only() -> Self {
        Self {
            target: ChatContext::empty(),
            touched: BTreeSet::new(),
            unsent_from: None,
            tail_creates: BTreeSet::new(),
            preserve_empty: true,
            empty_allowed: BTreeSet::new(),
        }
    }
}

impl OverlayWork {
    fn new(remote: impl Into<ProjectionItems>, seed: OverlaySeed) -> Self {
        let OverlaySeed {
            target,
            touched,
            unsent_from,
            tail_creates,
            preserve_empty,
            empty_allowed,
        } = seed;
        let remote = remote.into();
        let capacity = remote.len() + target.len();
        let target_storage = target.items.capacity();
        Self {
            phase: OverlayPhase::Index,
            remote,
            target,
            target_items: None,
            target_storage,
            cursor: 0,
            touched,
            unsent_from,
            target_ids: BTreeMap::new(),
            tail_creates,
            nodes: Vec::with_capacity(capacity),
            positions: BTreeMap::new(),
            head: None,
            tail: None,
            previous: None,
            collect_at: None,
            collected: Vec::with_capacity(capacity),
            extra: std::collections::VecDeque::new(),
            extra_storage: 0,
            preserve_empty,
            empty_allowed,
            retired: None,
            retired_items: Vec::new(),
            retired_sources: Vec::new(),
        }
    }

    fn add_node(&mut self, item: ChatItem, previous: Option<usize>) {
        let index = self.nodes.len();
        let next = previous.map_or(self.head, |at| self.nodes[at].next);
        self.positions.insert(item.id().to_owned(), index);
        self.nodes.push(ContextNode { item: Some(item), next });
        if let Some(previous) = previous {
            self.nodes[previous].next = Some(index);
        } else {
            self.head = Some(index);
        }
        if next.is_none() {
            self.tail = Some(index);
        }
        self.previous = Some(index);
    }

    fn add_remote(&mut self, remote: crate::realtime::RemoteChatContext) {
        if remote.revision() != 0 {
            self.extra_storage += remote.retained_units();
            self.extra.push_back(remote.into());
            if matches!(self.phase, OverlayPhase::Cleanup) {
                self.phase = OverlayPhase::Extra;
            }
        }
    }

    /// One unit of the overlay, or the overlaid context once it is done. Each phase method handles
    /// at most one item; with nothing left it moves to the next phase.
    fn step(&mut self) -> Option<ChatContext> {
        loop {
            let progress = match self.phase {
                OverlayPhase::Index => self.index_next_target(),
                OverlayPhase::Remote => self.place_next_remote(),
                OverlayPhase::Target => self.place_next_target(),
                OverlayPhase::Extra => self.place_next_extra(),
                OverlayPhase::Collect => self.collect_next(),
                OverlayPhase::Cleanup => {
                    return Some(ChatContext::new(std::mem::take(&mut self.collected)));
                }
            };
            if progress == Progress::Unit {
                return None;
            }
        }
    }

    /// Notes whether a target item is the update's own edit.
    fn index_next_target(&mut self) -> Progress {
        let Some(item) = self.target.items.get(self.cursor) else {
            self.target_items = Some(std::mem::take(&mut self.target.items).into_iter());
            self.phase = OverlayPhase::Remote;
            return Progress::Moved;
        };
        self.target_ids.insert(
            item.id().to_owned(),
            self.touched.contains(item.id())
                || self.unsent_from.is_some_and(|from| self.cursor >= from),
        );
        self.cursor += 1;
        Progress::Unit
    }

    /// Keeps a remote item the update does not replace.
    fn place_next_remote(&mut self) -> Progress {
        let Some(item) = self.remote.next() else {
            self.previous = None;
            self.phase = OverlayPhase::Target;
            return Progress::Moved;
        };
        if matches!(&item, ChatItem::Message(message) if message.content.is_empty()) {
            self.empty_allowed.insert(item.id().to_owned());
        }
        if !self.touched.contains(item.id())
            && !self.target_ids.get(item.id()).copied().unwrap_or(false)
        {
            self.add_node(item, self.tail);
        } else {
            self.retired_items.push(item);
        }
        Progress::Unit
    }

    /// Places an edited target item after its predecessor, or at the tail for a tail create.
    fn place_next_target(&mut self) -> Progress {
        let Some(item) = self.target_items.as_mut().and_then(Iterator::next) else {
            self.phase = OverlayPhase::Extra;
            return Progress::Moved;
        };
        if let Some(&index) = self.positions.get(item.id()) {
            self.previous = Some(index);
            self.retired_items.push(item);
        } else if self.target_ids.get(item.id()).copied().unwrap_or(false) {
            let previous =
                if self.tail_creates.contains(item.id()) { self.tail } else { self.previous };
            self.add_node(item, previous);
        } else {
            self.retired_items.push(item);
        }
        Progress::Unit
    }

    /// Appends an item of a remote context that arrived during the overlay. An exhausted source
    /// is retired without spending the unit.
    fn place_next_extra(&mut self) -> Progress {
        let Some(items) = self.extra.front_mut() else {
            if self.collected.is_empty() {
                self.collect_at = self.head;
            }
            self.phase = OverlayPhase::Collect;
            return Progress::Moved;
        };
        let Some(item) = items.next() else {
            if let Some(source) = self.extra.pop_front() {
                self.retired_sources.push(source);
            }
            return Progress::Moved;
        };
        if matches!(&item, ChatItem::Message(message) if message.content.is_empty()) {
            self.empty_allowed.insert(item.id().to_owned());
        }
        if self.positions.contains_key(item.id()) {
            self.retired_items.push(item);
            return Progress::Unit;
        }
        let index = self.nodes.len();
        self.add_node(item, self.tail);
        // Collection already started: it resumes at the first item appended after it.
        if self.collect_at.is_none() && !self.collected.is_empty() {
            self.collect_at = Some(index);
        }
        Progress::Unit
    }

    /// Moves the next placed item into the result; extras that arrived meanwhile go first.
    fn collect_next(&mut self) -> Progress {
        if !self.extra.is_empty() {
            self.phase = OverlayPhase::Extra;
            return Progress::Moved;
        }
        let Some(index) = self.collect_at else {
            self.phase = OverlayPhase::Cleanup;
            return Progress::Moved;
        };
        let node = &mut self.nodes[index];
        self.collect_at = node.next;
        self.collected.push(node.item.take().expect("context node collected once"));
        Progress::Unit
    }
}

/// What one call of a context work's phase method did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Progress {
    /// Handled one item: the unit the page budget counts.
    Unit,
    /// Handled nothing: the work moved on to its next phase or source.
    Moved,
}

impl RealtimeMachine {
    /// The runtime schedules this work below ready controls and only with write-queue capacity.
    pub fn context_pages_pending(&self) -> bool {
        self.ctx_update.as_ref().is_some_and(|update| update.work.is_some())
            || !self.deferred_commands.is_empty()
            || self.pending_context.is_some()
    }

    /// Semantic work can pause while admitted cleanup makes ownership capacity available.
    pub fn context_continuation_ready(&self) -> bool {
        self.context_pages_pending()
            && (self.retired.len() <= RETIRED_CONTEXTS - 3
                || (self.pending_context.is_none()
                    && self.ctx_update.as_ref().is_none_or(|update| update.work.is_none())))
    }

    pub(super) fn take_desired(&mut self) -> crate::realtime::RemoteChatContext {
        self.desired_epoch = self.desired_epoch.checked_add(1).expect("context epoch exhausted");
        std::mem::take(&mut self.desired)
    }

    pub fn context_work_pending(&self) -> bool {
        self.context_pages_pending() || self.has_context_retirement()
    }

    /// Continue at most one context page; transport admission precedes the next continuation.
    pub fn continue_context(&mut self, now: Ts) -> Step {
        self.continue_context_budget(now, CONTEXT_PAGE)
    }

    fn continue_context_budget(&mut self, now: Ts, mut budget: usize) -> Step {
        let mut step = Step::default();
        if self.closing {
            return step;
        }
        let mut halt = None;
        if let Some(mut update) = self.ctx_update.take() {
            halt = self.advance_update(&mut update, &mut budget, &mut step);
            self.settle_update(update);
        }
        match halt {
            Some(UpdateHalt::ServerItemsFailed) => {
                step.merge(self.close(now));
                step.fatal =
                    Some(self.error(now, "realtime server item source capacity exceeded", false));
                return step;
            }
            Some(UpdateHalt::ReplayAfterServerItems) => {
                self.start_context_replay();
                return step;
            }
            None => {}
        }
        if self.enforce_retirement_capacity(now, &mut step) {
            return step;
        }
        if self.pending_context_admitted(budget)
            && let Some(target) = self.pending_context.take()
        {
            step.merge(self.update_ctx_budget(now, target, budget));
            // The nested preparation owns the remaining allowance, including deferred commands.
            return step;
        }
        if !self.ctx_update.as_ref().is_some_and(|update| update.work.is_some()) {
            self.release_deferred_commands(now, budget, &mut step);
        }
        step
    }

    /// Advances the update's work unit by unit until the budget, the ownership capacity or the
    /// work runs out, or server items stop it.
    fn advance_update(
        &mut self,
        update: &mut CtxUpdate,
        budget: &mut usize,
        step: &mut Step,
    ) -> Option<UpdateHalt> {
        while *budget > 0 {
            // A unit may finish work, its target and the now-settled update. Reserve all
            // three ownership slots before advancing. Incoming growth has one terminal slot.
            if self.retired.len() > RETIRED_CONTEXTS - 3 {
                break;
            }
            let Some(work) = update.work.take() else { break };
            match work {
                ContextWork::ServerItems(server) => {
                    if let Some(halt) = self.advance_server_items(update, server, budget) {
                        return Some(halt);
                    }
                }
                ContextWork::Prepare(prepare) => {
                    self.advance_prepare(update, step, prepare, budget)
                }
                ContextWork::Overlay(overlay) => self.advance_overlay(update, overlay, budget),
                ContextWork::Replay(replay) => self.advance_replay(update, step, replay, budget),
            }
        }
        None
    }

    fn advance_server_items(
        &mut self,
        update: &mut CtxUpdate,
        mut server: Box<ServerItemsWork>,
        budget: &mut usize,
    ) -> Option<UpdateHalt> {
        *budget -= 1;
        if server.failed {
            update.work = Some(ContextWork::ServerItems(server));
            return Some(UpdateHalt::ServerItemsFailed);
        }
        if !server.advance(self) {
            update.work = Some(ContextWork::ServerItems(server));
            return None;
        }
        let replay = server.replay;
        update.work = server.prior.take().map(|prior| *prior);
        self.retired.push_back(Retired::Work(ContextWork::ServerItems(server)));
        replay.then_some(UpdateHalt::ReplayAfterServerItems)
    }

    /// One preparation unit. A preparation interrupted by a reconnect drops the frames it made and
    /// hands over to a replay of the desired context.
    fn advance_prepare(
        &mut self,
        update: &mut CtxUpdate,
        step: &mut Step,
        mut prepare: Box<PrepareWork>,
        budget: &mut usize,
    ) {
        *budget -= 1;
        let previous_frames = step.frames.len();
        let done = prepare.advance(self, update, step).is_break();
        if prepare.replay {
            for frame in step.frames.drain(previous_frames..).rev() {
                prepare.retire_frame(frame);
            }
        }
        if prepare.replay_ready() {
            let source = self.take_desired();
            self.retired.push_back(Retired::Context(std::mem::take(&mut update.target)));
            let retired = RetiredContextEvents::take(update);
            let mut overlay = OverlayWork::new(source, OverlaySeed::replay_only());
            overlay.retired = Some(retired);
            self.retired.push_back(Retired::Work(ContextWork::Prepare(prepare)));
            update.work = Some(ContextWork::Overlay(Box::new(overlay)));
        } else if !done {
            update.work = Some(ContextWork::Prepare(prepare));
        } else {
            self.retired.push_back(Retired::Work(ContextWork::Prepare(prepare)));
        }
    }

    /// One overlay unit; a finished overlay becomes the update's target and its replay starts.
    fn advance_overlay(
        &mut self,
        update: &mut CtxUpdate,
        mut overlay: Box<OverlayWork>,
        budget: &mut usize,
    ) {
        *budget -= 1;
        let Some(target) = overlay.step() else {
            update.work = Some(ContextWork::Overlay(overlay));
            return;
        };
        update.target = target;
        let mut replay = Box::new(ReplayWork::new(overlay.preserve_empty));
        replay.empty_allowed = std::mem::take(&mut overlay.empty_allowed);
        self.retired.push_back(Retired::Work(ContextWork::Overlay(overlay)));
        update.work = Some(ContextWork::Replay(replay));
    }

    /// One replay unit. Indexing spends budget only on an item; emitting spends it on every item.
    fn advance_replay(
        &mut self,
        update: &mut CtxUpdate,
        step: &mut Step,
        mut replay: Box<ReplayWork>,
        budget: &mut usize,
    ) {
        match replay.phase {
            ReplayPhase::Calls => {
                if replay.index_call(&update.target) == Progress::Unit {
                    *budget -= 1;
                }
                update.work = Some(ContextWork::Replay(replay));
            }
            ReplayPhase::Emit => self.emit_replay_item(update, step, replay, budget),
        }
    }

    fn emit_replay_item(
        &mut self,
        update: &mut CtxUpdate,
        step: &mut Step,
        mut replay: Box<ReplayWork>,
        budget: &mut usize,
    ) {
        let Some(item) = update.target.items.get(replay.cursor) else {
            self.retired.push_back(Retired::Work(ContextWork::Replay(replay)));
            return;
        };
        if replay.selects(item) {
            let previous = replay.previous.replace(item.id().to_owned());
            let item = item.clone();
            self.replay_item(update, step, previous, item);
        }
        replay.cursor += 1;
        replay.emitted = replay.cursor;
        *budget -= 1;
        update.work = Some(ContextWork::Replay(replay));
    }

    /// Places a replayed item after `previous`: in the desired context when the server already
    /// holds it, else as a create.
    fn replay_item(
        &mut self,
        update: &mut CtxUpdate,
        step: &mut Step,
        previous: Option<String>,
        item: ChatItem,
    ) {
        if self.remote.get(item.id()).is_none() {
            let previous = previous.unwrap_or_else(|| ROOT_ITEM.to_owned());
            self.create_ctx_item(&mut step.frames, update, Some(previous), item);
            return;
        }
        if let Ok(removed) = self.desired.delete(item.id()) {
            self.discard_context_item(removed);
        }
        let anchor = previous.filter(|previous| previous != ROOT_ITEM);
        let _ = self.desired.insert(anchor.as_deref(), item);
    }

    /// Retires finished work, then parks the update while it still waits, or retires it.
    fn settle_update(&mut self, mut update: CtxUpdate) {
        let target_len = update.target.len();
        if let Some(work) = update.work.take_if(|work| work.is_complete(target_len)) {
            self.retired.push_back(Retired::Work(work));
        }
        if !update.pending() {
            self.retired.push_back(Retired::Update(update));
            return;
        }
        if update.work.is_none() && !update.target.is_empty() {
            self.retired.push_back(Retired::Context(std::mem::take(&mut update.target)));
        }
        self.ctx_update = Some(update);
    }

    /// A parked context update starts once budget and ownership capacity allow, and no overlay or
    /// server items are in progress.
    fn pending_context_admitted(&self, budget: usize) -> bool {
        self.pending_context.is_some()
            && budget > 0
            && self.retired.len() <= RETIRED_CONTEXTS - 3
            && !self.ctx_update.as_ref().is_some_and(|update| {
                matches!(update.work, Some(ContextWork::Overlay(_) | ContextWork::ServerItems(_)))
            })
    }

    /// Releases the commands deferred behind context work, within the page and the budget.
    fn release_deferred_commands(&mut self, now: Ts, mut budget: usize, step: &mut Step) {
        while step.frames.len() < CONTEXT_PAGE && budget > 0 {
            if self.ctx_update.as_ref().is_some_and(|update| update.work.is_some()) {
                break;
            }
            let Some(command) = self.deferred_commands.pop_front() else { break };
            budget -= 1;
            match command {
                DeferredCommand::Response(req_id) => self.resend_deferred_reply(&req_id, step),
                DeferredCommand::Media(command) => {
                    if self.release_deferred_media(now, command, budget, step).is_break() {
                        return;
                    }
                }
                DeferredCommand::Item(item) => step.merge(self.submit_context_item(*item)),
            }
        }
    }

    fn resend_deferred_reply(&mut self, req_id: &str, step: &mut Step) {
        self.deferred_replies.remove(req_id);
        if !self.pending_replies.contains(req_id) {
            return;
        }
        if let Some(frame) = self.reply_frames.get(req_id).or_else(|| self.say_frames.get(req_id)) {
            step.frames.push(frame.clone());
        }
    }

    /// Applies a deferred media command. A truncation ends the continuation: it may have spent
    /// the remaining allowance preparing edits.
    fn release_deferred_media(
        &mut self,
        now: Ts,
        command: super::Command,
        budget: usize,
        step: &mut Step,
    ) -> ControlFlow<()> {
        self.deferred_media_count -= 1;
        self.deferred_media_bytes -= media_bytes(&command);
        let truncation = matches!(&command, super::Command::Truncate(_));
        step.merge(self.apply_command(now, command, budget));
        if truncation { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }

    pub(super) fn queue_context_item(&mut self, now: Ts, item: ChatItem) -> Step {
        if self.desired.get(item.id()).is_some()
            || self.deferred_commands.iter().any(|command| matches!(command, DeferredCommand::Item(queued) if queued.id() == item.id()))
        { return Step::default(); }
        if self
            .deferred_commands
            .iter()
            .filter(|command| matches!(command, DeferredCommand::Item(_)))
            .count()
            == QUEUED_CONTEXT_ITEMS
        {
            return Step {
                events: vec![lk_agents_abi::llm::realtime::Event::Error(self.error(
                    now,
                    "realtime context append capacity exceeded",
                    false,
                ))],
                recycle: true,
                ..Default::default()
            };
        }
        self.deferred_commands.push_back(DeferredCommand::Item(Box::new(item)));
        Step::default()
    }

    pub(super) fn submit_context_item(&mut self, item: ChatItem) -> Step {
        if self.desired.get(item.id()).is_some() {
            return Step::default();
        }
        let mut update = self.ctx_update.take().unwrap_or_default();
        update.tail_creates.insert(item.id().to_owned());
        let mut frames = Vec::new();
        self.create_ctx_item(&mut frames, &mut update, None, item);
        self.ctx_update = Some(update);
        Step { frames, ..Default::default() }
    }

    pub(super) fn defer_reply(&mut self, req_id: String, replay: bool) {
        if self.deferred_replies.insert(req_id.clone()) {
            let command = DeferredCommand::Response(req_id);
            if replay {
                self.deferred_commands.push_front(command);
            } else {
                self.deferred_commands.push_back(command);
            }
        }
    }

    pub(super) fn forget_deferred_reply(&mut self, req_id: &str) {
        self.deferred_replies.remove(req_id);
        self.deferred_commands
            .retain(|command| !matches!(command, DeferredCommand::Response(id) if id == req_id));
    }

    pub(super) fn queue_context_media(&mut self, now: Ts, command: super::Command) -> Step {
        let bytes = media_bytes(&command);
        if self.deferred_media_count == DEFERRED_MEDIA_COMMANDS
            || bytes > DEFERRED_MEDIA_BYTES.saturating_sub(self.deferred_media_bytes)
        {
            return Step {
                events: vec![lk_agents_abi::llm::realtime::Event::Error(self.error(
                    now,
                    "realtime context media capacity exceeded",
                    false,
                ))],
                recycle: true,
                ..Default::default()
            };
        }
        self.deferred_media_count += 1;
        self.deferred_media_bytes += bytes;
        self.deferred_commands.push_back(DeferredCommand::Media(command));
        Step::default()
    }

    pub(super) fn replay_context(&mut self) {
        if let Some(update) = &mut self.ctx_update
            && let Some(ContextWork::ServerItems(server)) = &mut update.work
        {
            let source = std::mem::take(&mut self.remote);
            if let Some(retired) = server.reconnect(self.connection_generation - 1, source)
                && retired.revision() != 0
            {
                self.retired.push_back(Retired::Remote(retired));
            }
            return;
        }
        let acknowledged = std::mem::take(&mut self.remote);
        if acknowledged.revision() != 0 {
            self.retired.push_back(Retired::Remote(acknowledged));
        }
        self.start_context_replay();
    }

    fn start_context_replay(&mut self) {
        if let Some(update) = &mut self.ctx_update
            && let Some(ContextWork::Prepare(prepare)) = &mut update.work
        {
            prepare.replay = true;
            return;
        }
        let desired = self.take_desired();
        let mut update = self.ctx_update.take().unwrap_or_default();
        let old_work = update.work.take();
        if let Some(ContextWork::Overlay(mut overlay)) = old_work {
            overlay.add_remote(desired);
            update.work = Some(ContextWork::Overlay(overlay));
        } else if old_work.is_none() {
            if !update.target.is_empty() {
                self.retired.push_back(Retired::Context(std::mem::take(&mut update.target)));
            }
            update.work = Some(ContextWork::Overlay(Box::new(OverlayWork::new(
                desired,
                OverlaySeed::replay_only(),
            ))));
        } else {
            let (unsent_from, preserve_empty, empty_allowed) = match old_work {
                Some(ContextWork::Replay(mut replay)) => {
                    let empty_allowed = std::mem::take(&mut replay.empty_allowed);
                    let flags = (Some(replay.emitted), replay.preserve_empty, empty_allowed);
                    self.retired.push_back(Retired::Work(ContextWork::Replay(replay)));
                    flags
                }
                _ => (None, true, BTreeSet::new()),
            };
            let seed = OverlaySeed {
                target: std::mem::take(&mut update.target),
                touched: std::mem::take(&mut update.touched),
                unsent_from,
                tail_creates: std::mem::take(&mut update.tail_creates),
                preserve_empty,
                empty_allowed,
            };
            update.work = Some(ContextWork::Overlay(Box::new(OverlayWork::new(desired, seed))));
        }
        // Repeated reconnects resume the same overlay. It already owns the original event
        // maps; no context operation emits during this phase, so no new maps need replacing.
        if matches!(&update.work, Some(ContextWork::Overlay(overlay)) if overlay.retired.is_none())
        {
            let retired = RetiredContextEvents::take(&mut update);
            if let Some(ContextWork::Overlay(overlay)) = &mut update.work {
                overlay.retired = Some(retired);
            }
        }
        self.ctx_update = Some(update);
    }
}

// 128 commands cover 2.56 seconds of 20 ms audio; bytes also bound unusually large frames.
const DEFERRED_MEDIA_COMMANDS: usize = 128;
const DEFERRED_MEDIA_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub(super) enum DeferredCommand {
    Response(String),
    Media(super::Command),
    Item(Box<ChatItem>),
}

fn media_bytes(command: &super::Command) -> usize {
    match command {
        super::Command::PushAudio(audio) => audio.frame.data.len(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::dialect::OPENAI;
    use crate::realtime::testfix::{machine, msg};

    #[test]
    fn repeated_reconnect_retains_the_unsettled_event_owner() {
        let mut machine = machine(OPENAI);
        let now = Ts::from_nanos(0);
        machine.open(now);
        for index in 0..300 {
            machine.submit_context_item(msg(&index.to_string(), "pending"));
        }
        machine.reconnect(now);
        let Some(ContextWork::Overlay(overlay)) =
            machine.ctx_update.as_ref().and_then(|update| update.work.as_ref())
        else {
            panic!("large projection must still be paged");
        };
        assert_eq!(overlay.retired.as_ref().unwrap().events.len(), 300);
        machine.reconnect(now);
        let Some(ContextWork::Overlay(overlay)) =
            machine.ctx_update.as_ref().and_then(|update| update.work.as_ref())
        else {
            panic!("second reconnect must resume the same projection");
        };
        assert_eq!(overlay.retired.as_ref().unwrap().events.len(), 300);
        assert!(machine.ctx_update.as_ref().unwrap().by_event_id.is_empty());
    }
}
