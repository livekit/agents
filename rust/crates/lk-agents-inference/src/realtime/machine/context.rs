//! Desired and acknowledged realtime contexts, operation preparation and replay.

use super::{Outgoing, RealtimeMachine, Step};
use lk_agents_abi::{ChatContext, ChatItem, Ts};
use std::collections::{BTreeMap, BTreeSet};

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
        if !self.closing && self.retired.len() >= RETIRED_CONTEXTS {
            step.merge(self.close(now));
            step.fatal =
                Some(self.error(now, "realtime context retirement capacity exceeded", false));
            return true;
        }
        false
    }

    /// A function output must name a call already submitted in the intended conversation.
    pub(super) fn holds_call(&self, call_id: &str) -> bool {
        !self.deleted_calls.iter().any(|id| id == call_id) && (self.desired.contains_call(call_id) || self.ctx_update.as_ref().and_then(|update| update.work.as_ref()).is_some_and(|work| matches!(work, ContextWork::ServerItems(server) if server.contains_call(call_id, &self.remote))))
    }

    pub(super) fn update_ctx(&mut self, now: Ts, chat_ctx: ChatContext) -> Step {
        self.deferred_commands.retain(|command| !matches!(command, DeferredCommand::Item(_)));
        if self.ctx_update.as_ref().is_some_and(|update| {
            matches!(update.work, Some(ContextWork::Overlay(_) | ContextWork::ServerItems(_)))
        }) {
            if let Some(previous) = self.pending_context.replace(chat_ctx) {
                self.retired.push_back(Retired::Context(previous));
            }
            return self.continue_context(now);
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
        self.continue_context(now)
    }

    pub(super) fn truncate_ctx(
        &mut self,
        now: Ts,
        edit: lk_agents_abi::llm::realtime::Truncate,
    ) -> Step {
        let mut update = self.ctx_update.take().unwrap_or_default();
        update.work = Some(ContextWork::Prepare(Box::new(PrepareWork::truncate(self, edit))));
        self.ctx_update = Some(update);
        self.continue_context(now)
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
        let event_id = self.ids.short("chat_ctx_delete_");
        frames.push(Outgoing::ItemDelete { event_id: event_id.clone(), item_id: id.to_owned() });
        update.touched.insert(id.to_owned());
        update.awaiting_delete.insert(id.to_owned());
        update.register_event(event_id, true, id.to_owned(), self.ids.next.get());
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
            Some("root") => None,
            Some(id) => Some(id.to_owned()),
            None => self.desired.tail_id().map(str::to_owned),
        };
        if let Ok(item) = self.desired.delete(&id) {
            self.discard_context_item(item);
        }
        let _ = self.desired.insert(anchor.as_deref(), item.clone());
        let event_id = self.ids.short("chat_ctx_create_");
        frames.push(Outgoing::ItemCreate { event_id: event_id.clone(), item, previous_item_id });
        update.touched.insert(id.clone());
        update.awaiting_create.insert(id.clone());
        update.register_event(event_id, false, id, self.ids.next.get());
    }

    pub(super) fn ctx_event_is_latest(&self, id: &str, sequence: u64) -> bool {
        !self.ctx_update.as_ref().is_some_and(|update| {
            [false, true].into_iter().any(|kind| {
                update
                    .events_by_item
                    .get(&(kind, id.to_owned()))
                    .and_then(|events| events.last_key_value())
                    .is_some_and(|(&latest, _)| latest > sequence)
            })
        })
    }

    pub(super) fn reject_ctx_item(&mut self, is_delete: bool, id: &str, sequence: u64) {
        let newer = !self.ctx_event_is_latest(id, sequence);
        if newer {
            return;
        }
        if !is_delete && let Ok(item) = self.desired.delete(id) {
            self.discard_context_item(item);
        }
        let delete_pending =
            self.ctx_update.as_ref().is_some_and(|update| update.awaiting_delete.contains(id));
        if self.desired.get(id).is_none()
            && (is_delete || !delete_pending)
            && let Some(item) = self.remote.get(id)
        {
            let item = item.clone();
            let previous = self
                .remote
                .previous_id(id)
                .filter(|id| self.desired.get(id).is_some())
                .map(str::to_owned);
            let _ = self.desired.insert(previous.as_deref(), item);
        }
    }

    pub(super) fn settle_ctx_item(&mut self, is_delete: bool, item_id: &str) -> Step {
        self.settle_ctx_event(is_delete, item_id, None)
    }

    pub(super) fn settle_ctx_event(
        &mut self,
        is_delete: bool,
        item_id: &str,
        sequence: Option<u64>,
    ) -> Step {
        let Some(update) = &mut self.ctx_update else { return Step::default() };
        let key = (is_delete, item_id.to_owned());
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
            let set =
                if is_delete { &mut update.awaiting_delete } else { &mut update.awaiting_create };
            set.remove(item_id);
            if !update.awaiting_create.contains(item_id)
                && !update.awaiting_delete.contains(item_id)
            {
                update.touched.remove(item_id);
            }
        }
        if !update.pending() {
            let update = self.ctx_update.take().unwrap();
            self.retired.push_back(Retired::Update(update));
        }
        Step::default()
    }
}

#[derive(Debug, Default)]
pub(super) struct CtxUpdate {
    /// `_item_create_future` / `_item_delete_future`, keyed by item id.
    pub(super) awaiting_create: BTreeSet<String>,
    pub(super) awaiting_delete: BTreeSet<String>,
    /// `_chat_ctx_event_futures`, keyed by client `event_id`: an update sends a delete and a
    /// create under the same item id, so only the event id says which one a rejection answers.
    pub(super) by_event_id: BTreeMap<String, (bool, String, u64)>,
    events_by_item: BTreeMap<(bool, String), BTreeMap<u64, String>>,
    /// The context it brings the server to, and the items it deletes or creates: what a
    /// reconnection replays when the lost connection had not confirmed it.
    pub(super) target: ChatContext,
    pub(super) touched: BTreeSet<String>,
    tail_creates: BTreeSet<String>,
    work: Option<ContextWork>,
}

impl CtxUpdate {
    fn register_event(&mut self, event_id: String, is_delete: bool, id: String, sequence: u64) {
        self.events_by_item
            .entry((is_delete, id.clone()))
            .or_default()
            .insert(sequence, event_id.clone());
        self.by_event_id.insert(event_id, (is_delete, id, sequence));
    }

    pub(super) fn pending(&self) -> bool {
        self.work.is_some() || !self.awaiting_create.is_empty() || !self.awaiting_delete.is_empty()
    }
}

/// Context preparation and submission share a bounded continuation budget.
const CONTEXT_PAGE: usize = 64;
const RETIRED_CONTEXTS: usize = 4;
const QUEUED_CONTEXT_ITEMS: usize = 32;

#[derive(Debug)]
pub(super) enum ContextWork {
    Replay(Box<ReplayWork>),
    Overlay(Box<OverlayWork>),
    ServerItems(Box<ServerItemsWork>),
    Prepare(Box<PrepareWork>),
}

#[derive(Debug, Default)]
struct RetiredContextEvents {
    creates: BTreeSet<String>,
    deletes: BTreeSet<String>,
    events: BTreeMap<String, (bool, String, u64)>,
    touched: BTreeSet<String>,
    tails: BTreeSet<String>,
    inverse: BTreeMap<(bool, String), BTreeMap<u64, String>>,
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

    fn step(&mut self) -> bool {
        self.creates.pop_first().is_some()
            || self.deletes.pop_first().is_some()
            || self.events.pop_first().is_some()
            || self.touched.pop_first().is_some()
            || self.tails.pop_first().is_some()
            || retirement::pop_inverse(&mut self.inverse)
    }
}

#[derive(Debug)]
pub(super) struct ReplayWork {
    phase: ReplayPhase,
    cursor: usize,
    emitted: usize,
    write: usize,
    selected: Vec<bool>,
    calls: BTreeSet<String>,
    seen: BTreeSet<String>,
    preserve_empty: bool,
    empty_allowed: BTreeSet<String>,
    retired_update: Option<RetiredContextEvents>,
    all_selected: bool,
    previous: Option<String>,
    retired_items: Vec<ChatItem>,
}

#[derive(Debug, Clone, Copy)]
enum ReplayPhase {
    Calls,
    Emit,
    Compact,
    Retire,
    Cleanup,
}

impl ReplayWork {
    fn new(preserve_empty: bool) -> Self {
        Self {
            phase: ReplayPhase::Calls,
            cursor: 0,
            emitted: 0,
            write: 0,
            selected: Vec::new(),
            calls: BTreeSet::new(),
            seen: BTreeSet::new(),
            preserve_empty,
            empty_allowed: BTreeSet::new(),
            retired_update: None,
            all_selected: true,
            previous: None,
            retired_items: Vec::new(),
        }
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

    fn retire_step(&mut self) -> bool {
        match self {
            Self::Context { items, .. } => items.next().is_some(),
            Self::Mirror { remote, .. } => remote.pop_front().is_some(),
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

impl OverlayWork {
    fn new(
        remote: impl Into<ProjectionItems>,
        target: ChatContext,
        touched: BTreeSet<String>,
        unsent_from: Option<usize>,
        tail_creates: BTreeSet<String>,
        preserve_empty: bool,
        empty_allowed: BTreeSet<String>,
    ) -> Self {
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

    fn step(&mut self) -> Option<ChatContext> {
        loop {
            match self.phase {
                OverlayPhase::Index => {
                    if let Some(item) = self.target.items.get(self.cursor) {
                        self.target_ids.insert(
                            item.id().to_owned(),
                            self.touched.contains(item.id())
                                || self.unsent_from.is_some_and(|from| self.cursor >= from),
                        );
                        self.cursor += 1;
                        return None;
                    }
                    self.target_items = Some(std::mem::take(&mut self.target.items).into_iter());
                    self.phase = OverlayPhase::Remote;
                }
                OverlayPhase::Remote => {
                    if let Some(item) = self.remote.next() {
                        if matches!(&item, ChatItem::Message(message) if message.content.is_empty())
                        {
                            self.empty_allowed.insert(item.id().to_owned());
                        }
                        if !self.touched.contains(item.id())
                            && !self.target_ids.get(item.id()).copied().unwrap_or(false)
                        {
                            self.add_node(item, self.tail);
                        } else {
                            self.retired_items.push(item);
                        }
                        return None;
                    }
                    self.previous = None;
                    self.phase = OverlayPhase::Target;
                }
                OverlayPhase::Target => {
                    if let Some(item) = self.target_items.as_mut().and_then(Iterator::next) {
                        if let Some(&index) = self.positions.get(item.id()) {
                            self.previous = Some(index);
                            self.retired_items.push(item);
                        } else if self.target_ids.get(item.id()).copied().unwrap_or(false) {
                            let previous = if self.tail_creates.contains(item.id()) {
                                self.tail
                            } else {
                                self.previous
                            };
                            self.add_node(item, previous);
                        } else {
                            self.retired_items.push(item);
                        }
                        return None;
                    }
                    self.phase = OverlayPhase::Extra;
                }
                OverlayPhase::Extra => {
                    if let Some(items) = self.extra.front_mut() {
                        if let Some(item) = items.next() {
                            if matches!(&item, ChatItem::Message(message) if message.content.is_empty())
                            {
                                self.empty_allowed.insert(item.id().to_owned());
                            }
                            if !self.positions.contains_key(item.id()) {
                                let index = self.nodes.len();
                                self.add_node(item, self.tail);
                                if self.collect_at.is_none() && !self.collected.is_empty() {
                                    self.collect_at = Some(index);
                                }
                            } else {
                                self.retired_items.push(item);
                            }
                            return None;
                        }
                        self.retired_sources.push(self.extra.pop_front().unwrap());
                        continue;
                    }
                    if self.collected.is_empty() {
                        self.collect_at = self.head;
                    }
                    self.phase = OverlayPhase::Collect;
                }
                OverlayPhase::Collect => {
                    if !self.extra.is_empty() {
                        self.phase = OverlayPhase::Extra;
                        continue;
                    }
                    if let Some(index) = self.collect_at {
                        let node = &mut self.nodes[index];
                        self.collect_at = node.next;
                        self.collected.push(node.item.take().expect("context node collected once"));
                        return None;
                    }
                    self.phase = OverlayPhase::Cleanup;
                }
                OverlayPhase::Cleanup => {
                    return Some(ChatContext::new(std::mem::take(&mut self.collected)));
                }
            }
        }
    }
}

impl RealtimeMachine {
    /// The runtime schedules this work below ready controls and only with write-queue capacity.
    pub fn context_pages_pending(&self) -> bool {
        self.ctx_update.as_ref().is_some_and(|update| update.work.is_some())
            || !self.deferred_commands.is_empty()
            || self.pending_context.is_some()
    }

    pub fn context_work_pending(&self) -> bool {
        self.context_pages_pending() || self.has_context_retirement()
    }

    /// Continue at most one context page; transport admission precedes the next continuation.
    pub fn continue_context(&mut self, _now: Ts) -> Step {
        let mut step = Step::default();
        let mut budget = CONTEXT_PAGE;
        let mut replay_after_server_items = false;
        let mut server_items_failed = false;
        if self.closing {
            return step;
        }
        if let Some(mut update) = self.ctx_update.take() {
            while budget > 0 {
                let Some(work) = update.work.take() else { break };
                match work {
                    ContextWork::ServerItems(mut server) => {
                        budget -= 1;
                        if server.failed {
                            server_items_failed = true;
                            update.work = Some(ContextWork::ServerItems(server));
                            break;
                        }
                        if server.advance(self) {
                            replay_after_server_items = server.replay;
                            update.work = server.prior.take().map(|prior| *prior);
                            self.retired.push_back(Retired::Work(ContextWork::ServerItems(server)));
                            if replay_after_server_items {
                                break;
                            }
                        } else {
                            update.work = Some(ContextWork::ServerItems(server));
                        }
                    }
                    ContextWork::Prepare(mut prepare) => {
                        budget -= 1;
                        let previous_frames = step.frames.len();
                        let done = prepare.advance(self, &mut update, &mut step);
                        if prepare.replay {
                            while step.frames.len() > previous_frames {
                                prepare.retire_frame(step.frames.pop().unwrap());
                            }
                        }
                        if prepare.replay_ready() {
                            let source = std::mem::take(&mut self.desired);
                            self.retired
                                .push_back(Retired::Context(std::mem::take(&mut update.target)));
                            let retired = RetiredContextEvents::take(&mut update);
                            let mut overlay = OverlayWork::new(
                                source,
                                ChatContext::empty(),
                                BTreeSet::new(),
                                None,
                                BTreeSet::new(),
                                true,
                                BTreeSet::new(),
                            );
                            overlay.retired = Some(retired);
                            self.retired.push_back(Retired::Work(ContextWork::Prepare(prepare)));
                            update.work = Some(ContextWork::Overlay(Box::new(overlay)));
                        } else if !done {
                            update.work = Some(ContextWork::Prepare(prepare));
                        } else {
                            self.retired.push_back(Retired::Work(ContextWork::Prepare(prepare)));
                        }
                    }
                    ContextWork::Overlay(mut overlay) => {
                        budget -= 1;
                        if let Some(target) = overlay.step() {
                            update.target = target;
                            let mut replay = Box::new(ReplayWork::new(overlay.preserve_empty));
                            replay.empty_allowed = std::mem::take(&mut overlay.empty_allowed);
                            self.retired.push_back(Retired::Work(ContextWork::Overlay(overlay)));
                            update.work = Some(ContextWork::Replay(replay));
                        } else {
                            update.work = Some(ContextWork::Overlay(overlay));
                        }
                    }
                    ContextWork::Replay(mut replay) => match replay.phase {
                        ReplayPhase::Calls => {
                            if let Some(item) = update.target.items.get(replay.cursor) {
                                replay.selected.push(false);
                                if let ChatItem::FunctionCall(call) = item {
                                    replay.calls.insert(call.call_id.clone());
                                }
                                replay.cursor += 1;
                                budget -= 1;
                            } else {
                                replay.phase = ReplayPhase::Emit;
                                replay.cursor = 0;
                            }
                            update.work = Some(ContextWork::Replay(replay));
                        }
                        ReplayPhase::Emit => {
                            if let Some(item) = update.target.items.get(replay.cursor) {
                                let valid = (!matches!(item, ChatItem::Message(message) if message.content.is_empty())
                                    || replay.preserve_empty
                                    || replay.empty_allowed.contains(item.id()))
                                    && !matches!(item, ChatItem::FunctionCallOutput(output) if !replay.calls.contains(&output.call_id));
                                let selected = valid && replay.seen.insert(item.id().to_owned());
                                replay.selected[replay.cursor] = selected;
                                replay.all_selected &= selected;
                                if selected {
                                    let previous = replay
                                        .previous
                                        .clone()
                                        .unwrap_or_else(|| "root".to_owned());
                                    replay.previous = Some(item.id().to_owned());
                                    let item = item.clone();
                                    if self.remote.get(item.id()).is_some() {
                                        if let Ok(removed) = self.desired.delete(item.id()) {
                                            self.discard_context_item(removed);
                                        }
                                        let anchor = (previous != "root").then_some(previous);
                                        let _ = self.desired.insert(anchor.as_deref(), item);
                                    } else {
                                        self.create_ctx_item(
                                            &mut step.frames,
                                            &mut update,
                                            Some(previous),
                                            item,
                                        );
                                    }
                                }
                                replay.cursor += 1;
                                replay.emitted = replay.cursor;
                                budget -= 1;
                                update.work = Some(ContextWork::Replay(replay));
                            } else {
                                if replay.all_selected {
                                    replay.phase = ReplayPhase::Cleanup;
                                } else {
                                    replay.phase = ReplayPhase::Compact;
                                    replay.cursor = 0;
                                }
                                update.work = Some(ContextWork::Replay(replay));
                            }
                        }
                        ReplayPhase::Compact => {
                            if replay.cursor < update.target.len() {
                                if replay.selected[replay.cursor] {
                                    update.target.items.swap(replay.write, replay.cursor);
                                    replay.write += 1;
                                }
                                replay.cursor += 1;
                                budget -= 1;
                            } else {
                                replay.phase = ReplayPhase::Retire;
                            }
                            update.work = Some(ContextWork::Replay(replay));
                        }
                        ReplayPhase::Retire => {
                            if update.target.len() > replay.write {
                                replay.retired_items.push(update.target.items.pop().unwrap());
                                budget -= 1;
                                update.work = Some(ContextWork::Replay(replay));
                            } else {
                                replay.phase = ReplayPhase::Cleanup;
                                update.work = Some(ContextWork::Replay(replay));
                            }
                        }
                        ReplayPhase::Cleanup => {
                            self.retired.push_back(Retired::Work(ContextWork::Replay(replay)));
                        }
                    },
                }
            }
            if update.work.is_none() && !update.target.is_empty() {
                self.retired.push_back(Retired::Context(std::mem::take(&mut update.target)));
            }
            if update.pending() {
                self.ctx_update = Some(update);
            } else {
                self.retired.push_back(Retired::Update(update));
            }
        }
        if server_items_failed {
            step.merge(self.close(_now));
            step.fatal =
                Some(self.error(_now, "realtime server item source capacity exceeded", false));
            return step;
        }
        if replay_after_server_items {
            self.start_context_replay();
            return step;
        }
        if self.enforce_retirement_capacity(_now, &mut step) {
            return step;
        }
        if self.pending_context.is_some()
            && !self
                .ctx_update
                .as_ref()
                .is_some_and(|update| matches!(update.work, Some(ContextWork::Overlay(_))))
        {
            let target = self.pending_context.take().unwrap();
            step.merge(self.update_ctx(_now, target));
        }
        if !self.ctx_update.as_ref().is_some_and(|update| update.work.is_some()) {
            let mut commands = 0;
            while step.frames.len() < CONTEXT_PAGE && commands < CONTEXT_PAGE {
                let Some(command) = self.deferred_commands.pop_front() else { break };
                commands += 1;
                match command {
                    DeferredCommand::Response(req_id) => {
                        self.deferred_replies.remove(&req_id);
                        if self.pending_replies.contains(&req_id)
                            && let Some(frame) = self
                                .reply_frames
                                .get(&req_id)
                                .or_else(|| self.say_frames.get(&req_id))
                        {
                            step.frames.push(frame.clone());
                        }
                    }
                    DeferredCommand::Media(command) => {
                        self.deferred_media_count -= 1;
                        self.deferred_media_bytes -= media_bytes(&command);
                        step.merge(self.apply_command(_now, command));
                    }
                    DeferredCommand::Item(item) => step.merge(self.submit_context_item(*item)),
                }
            }
        }
        step
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
            if let Some(retired) = server.reconnect(self.connection_generation - 1, source) {
                self.retired.push_back(Retired::Remote(retired));
            }
            return;
        }
        let acknowledged = std::mem::take(&mut self.remote);
        self.retired.push_back(Retired::Remote(acknowledged));
        self.start_context_replay();
    }

    fn start_context_replay(&mut self) {
        if let Some(update) = &mut self.ctx_update
            && let Some(ContextWork::Prepare(prepare)) = &mut update.work
        {
            prepare.replay = true;
            return;
        }
        let desired = std::mem::take(&mut self.desired);
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
                ChatContext::empty(),
                BTreeSet::new(),
                None,
                BTreeSet::new(),
                true,
                BTreeSet::new(),
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
            let target = std::mem::take(&mut update.target);
            let touched = std::mem::take(&mut update.touched);
            let tails = std::mem::take(&mut update.tail_creates);
            update.work = Some(ContextWork::Overlay(Box::new(OverlayWork::new(
                desired,
                target,
                touched,
                unsent_from,
                tails,
                preserve_empty,
                empty_allowed,
            ))));
        }
        if matches!(update.work, Some(ContextWork::Overlay(_))) {
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
