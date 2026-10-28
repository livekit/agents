//! Desired and acknowledged realtime contexts, operation preparation and replay.

use super::{DELETED_CALLS_REMEMBERED, Outgoing, RealtimeMachine, Step};
use lk_agents_abi::{ChatContext, ChatItem, Ts, compute_chat_ctx_diff};
use std::collections::{BTreeMap, BTreeSet};

impl RealtimeMachine {
    /// The conversation the server is being brought to has the call `call_id`.
    pub(super) fn holds_call(&self, call_id: &str) -> bool {
        !self.deleted_calls.iter().any(|c| c == call_id)
            && self
                .expected_ctx()
                .items
                .iter()
                .any(|i| matches!(i, ChatItem::FunctionCall(c) if c.call_id == call_id))
    }

    /// Bring the server to `chat_ctx`; an update still unconfirmed is superseded, its
    /// confirmations still settling.
    pub(super) fn update_ctx(&mut self, now: Ts, chat_ctx: ChatContext) -> Step {
        self.deferred_commands.retain(|command| !matches!(command, DeferredCommand::Item(_)));
        if self.held.is_none() && self.ctx_update.is_none() && self.remote.items().is_empty() {
            let length = chat_ctx.len();
            self.ctx_update = Some(CtxUpdate {
                target: chat_ctx,
                mirror_revision: Some(self.remote.revision()),
                work: Some(ContextWork::Replay(ReplayWork::new(length, false))),
                ..Default::default()
            });
            return self.continue_context(now);
        }
        let mut append_after = None;
        let mut bulk_append = None;
        // Tool outputs still pass through call-pair validation. A held transcript can add an
        // item independently of the requested context, so it also uses the general path.
        if self.held.is_none()
            && !chat_ctx.items.iter().any(|item| matches!(item, ChatItem::FunctionCallOutput(_)))
        {
            let current = self
                .ctx_update
                .as_ref()
                .map_or(self.remote.items(), |update| update.target.items.as_slice());
            let mirror_matches = self.ctx_update.as_ref().is_none_or(|update| {
                !matches!(update.work, Some(ContextWork::Overlay(_)))
                    && !matches!(&update.work, Some(ContextWork::Replay(replay)) if matches!(replay.phase, ReplayPhase::Compact | ReplayPhase::Retire))
                    && (update.mirror_revision == Some(self.remote.revision())
                        || self.remote.items().iter().filter(|item| !update.touched.contains(item.id())).eq(
                            update.target.items.iter().enumerate().filter(|(at, item)| {
                                !update.touched.contains(item.id()) && !update.unsent_from().is_some_and(|from| *at >= from)
                            }).map(|(_, item)| item),
                        ))
            });
            if mirror_matches && chat_ctx.items.starts_with(current) {
                if chat_ctx.len() == current.len() {
                    if let Some(update) = &mut self.ctx_update {
                        update.mirror_revision = Some(self.remote.revision());
                    }
                    return Step::default();
                }
                if chat_ctx.len() == current.len() + 1
                    && let Some(ChatItem::Message(message)) = chat_ctx.items.last()
                    && !message.content.is_empty()
                    && !current.iter().any(|item| item.id() == message.id)
                {
                    append_after = Some(current.last().map(|item| item.id().to_owned()));
                } else if chat_ctx.len() > current.len() + 1
                    && chat_ctx.items[current.len()..].iter().all(|item| matches!(item, ChatItem::Message(message) if !message.content.is_empty()))
                { bulk_append = Some(current.len()); }
            }
        }
        if let Some(previous) = append_after {
            let mut update = self.ctx_update.take().unwrap_or_default();
            update.mirror_revision = Some(self.remote.revision());
            if let Some(ContextWork::Replay(replay)) = &mut update.work {
                if matches!(replay.phase, ReplayPhase::Calls | ReplayPhase::Emit) {
                    replay.selected.push(false);
                    update.target = chat_ctx;
                    self.ctx_update = Some(update);
                    return self.continue_context(now);
                }
            }
            let mut frames = Vec::new();
            self.create_ctx_item(
                &mut frames,
                &mut update,
                Some(previous.unwrap_or_else(|| "root".to_owned())),
                chat_ctx.items.last().unwrap().clone(),
            );
            update.target = chat_ctx;
            self.ctx_update = Some(update);
            return Step { frames, ..Default::default() };
        }
        if let Some(start) = bulk_append {
            let mut update = self.ctx_update.take().unwrap_or_default();
            update.mirror_revision = Some(self.remote.revision());
            let actual_start = update.unsent_from().unwrap_or(start);
            let preserve_empty = match &update.work {
                Some(ContextWork::Replay(replay)) => replay.preserve_empty,
                _ => true,
            };
            let mut replay = ReplayWork::new(chat_ctx.len(), preserve_empty);
            replay.start = actual_start;
            replay.emitted = actual_start;
            update.target = chat_ctx;
            update.work = Some(ContextWork::Replay(replay));
            self.ctx_update = Some(update);
            return self.continue_context(now);
        }
        if let Some(update) = &mut self.ctx_update {
            if let Some(ContextWork::Replay(replay)) = update.work.take() {
                update.target.items.truncate(replay.emitted);
            }
        }
        if self.held.is_none()
            && self.remote.items().is_empty()
            && self
                .ctx_update
                .as_ref()
                .is_some_and(|update| update.target.is_empty() && update.awaiting_create.is_empty())
        {
            let mut update = self.ctx_update.take().unwrap_or_default();
            update.work = Some(ContextWork::Replay(ReplayWork::new(chat_ctx.len(), false)));
            update.target = chat_ctx;
            update.mirror_revision = Some(self.remote.revision());
            self.ctx_update = Some(update);
            return self.continue_context(now);
        }
        let (frames, mut update) = self.chat_ctx_events(&chat_ctx);
        if let Some(old) = self.ctx_update.take() {
            update.awaiting_create.extend(old.awaiting_create);
            update.awaiting_delete.extend(old.awaiting_delete);
            update.by_event_id.extend(old.by_event_id);
            update.touched.extend(old.touched);
        }
        self.ctx_update = update.pending().then_some(update);
        Step { frames, ..Default::default() }
    }

    /// The conversation the server is being brought to: the confirmed mirror, with the context
    /// update still awaiting confirmation applied.
    pub(super) fn expected_ctx(&self) -> ChatContext {
        match &self.ctx_update {
            Some(update) => {
                self.remote.with_update(&update.target, &update.touched, &update.tail_creates)
            }
            None => self.remote.to_chat_ctx(),
        }
    }

    /// `_create_update_chat_ctx_events`: what the server is being brought to (`expected_ctx`)
    /// diffed against the agent's context. Diffed against the bare mirror, an update sent while
    /// another was unconfirmed deleted again what that one had deleted, and was refused.
    pub(super) fn chat_ctx_events(&mut self, chat_ctx: &ChatContext) -> (Vec<Outgoing>, CtxUpdate) {
        let mut chat_ctx = std::borrow::Cow::Borrowed(chat_ctx);

        // xAI: the held transcript is not in the agent's context yet; a plain diff would delete
        // the user's turn. Re-insert it at the mirror's position.
        if let Some(held) = &self.held
            && let Some(pos) = self.remote.items().iter().position(|i| i.id() == held.item_id)
            && chat_ctx.get_by_id(&held.item_id).is_none()
        {
            let target_positions: BTreeMap<&str, usize> = chat_ctx
                .items
                .iter()
                .enumerate()
                .rev()
                .map(|(index, item)| (item.id(), index))
                .collect();
            let at = self.remote.items()[..pos]
                .iter()
                .rev()
                .find_map(|prev| target_positions.get(prev.id()).map(|i| i + 1))
                .unwrap_or(0);
            drop(target_positions);
            chat_ctx.to_mut().items.insert(at, self.remote.items()[pos].clone());
        }

        let remote_ctx = self.expected_ctx();
        // Empty content can be a local placeholder or a remote item with non-text content; keep
        // the remote ones, or every committed audio turn is deleted on the next update.
        let remote_by_id: BTreeMap<&str, &ChatItem> =
            remote_ctx.items.iter().rev().map(|item| (item.id(), item)).collect();
        // An output whose call neither the context nor the conversation holds would be refused
        // (`invalid_tool_call_id`): the context's own record of a call it dropped.
        let calls: BTreeSet<&str> = chat_ctx
            .items
            .iter()
            .chain(&remote_ctx.items)
            .filter_map(|i| match i {
                ChatItem::FunctionCall(c) => Some(c.call_id.as_str()),
                _ => None,
            })
            .collect();
        let filtered = ChatContext::new(
            chat_ctx
                .items
                .iter()
                .filter(|i| {
                    i.kind() != lk_agents_abi::ItemKind::Message
                        || i.as_message().is_some_and(|m| !m.content.is_empty())
                        || remote_by_id.contains_key(i.id())
                })
                .filter(|i| !matches!(i, ChatItem::FunctionCallOutput(o) if !calls.contains(o.call_id.as_str())))
                .cloned()
                .collect(),
        );
        let diff = compute_chat_ctx_diff(&remote_ctx, &filtered);
        let filtered_by_id: BTreeMap<&str, &ChatItem> =
            filtered.items.iter().rev().map(|item| (item.id(), item)).collect();
        for id in &diff.to_remove {
            if let Some(ChatItem::FunctionCall(c)) = remote_by_id.get(id.as_str()) {
                if self.deleted_calls.len() == DELETED_CALLS_REMEMBERED {
                    self.deleted_calls.pop_front();
                }
                self.deleted_calls.push_back(c.call_id.clone());
            }
        }
        for (_, id) in &diff.to_create {
            if let Some(ChatItem::FunctionCall(c)) = filtered_by_id.get(id.as_str()) {
                self.deleted_calls.retain(|id| *id != c.call_id);
            }
        }
        let mut frames = Vec::new();
        let mut update = CtxUpdate {
            awaiting_create: BTreeSet::new(),
            awaiting_delete: BTreeSet::new(),
            by_event_id: BTreeMap::new(),
            target: ChatContext::empty(),
            mirror_revision: Some(self.remote.revision()),
            touched: BTreeSet::new(),
            ..Default::default()
        };

        let delete = |frames: &mut Vec<Outgoing>, update: &mut CtxUpdate, id: &str| {
            let event_id = self.ids.short("chat_ctx_delete_");
            let frame = Outgoing::ItemDelete { event_id: event_id.clone(), item_id: id.to_owned() };
            frames.push(frame);
            update.touched.insert(id.to_owned());
            update.awaiting_delete.insert(id.to_owned());
            update.by_event_id.insert(event_id, (true, id.to_owned()));
        };

        for id in &diff.to_remove {
            delete(&mut frames, &mut update, id);
        }

        for (prev, id) in &diff.to_create {
            if let Some(&item) = filtered_by_id.get(id.as_str()) {
                self.create_ctx_item(
                    &mut frames,
                    &mut update,
                    Some(prev.clone().unwrap_or_else(|| "root".to_owned())),
                    item.clone(),
                );
            }
        }

        for (prev, id) in &diff.to_update {
            // Empty content means "not synced down"; recreating would replace a server-side audio
            // turn with an empty text message.
            let remote_is_empty = remote_by_id
                .get(id.as_str())
                .and_then(|item| item.as_message())
                .is_some_and(|m| m.content.is_empty());
            if remote_is_empty {
                continue;
            }
            delete(&mut frames, &mut update, id);
            if let Some(&item) = filtered_by_id.get(id.as_str()) {
                self.create_ctx_item(
                    &mut frames,
                    &mut update,
                    Some(prev.clone().unwrap_or_else(|| "root".to_owned())),
                    item.clone(),
                );
            }
        }

        update.target = filtered;
        (frames, update)
    }

    pub(super) fn create_ctx_item(
        &self,
        frames: &mut Vec<Outgoing>,
        update: &mut CtxUpdate,
        previous_item_id: Option<String>,
        item: ChatItem,
    ) {
        let id = item.id().to_owned();
        let event_id = self.ids.short("chat_ctx_create_");
        frames.push(Outgoing::ItemCreate { event_id: event_id.clone(), item, previous_item_id });
        update.touched.insert(id.clone());
        update.awaiting_create.insert(id.clone());
        update.by_event_id.insert(event_id, (false, id));
    }

    /// The server acked a create or a delete. Emits [`Event::ChatCtxUpdated`] once the last one
    /// lands (the `asyncio.gather` in `update_chat_ctx`).
    pub(super) fn settle_ctx_item(&mut self, is_delete: bool, item_id: &str) -> Step {
        let Some(update) = &mut self.ctx_update else { return Step::default() };
        let set = if is_delete { &mut update.awaiting_delete } else { &mut update.awaiting_create };
        if !set.remove(item_id) {
            return Step::default();
        }
        update.by_event_id.retain(|_, (d, id)| !(*d == is_delete && id == item_id));
        if update.pending() {
            return Step::default();
        }
        self.ctx_update = None;
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
    pub(super) by_event_id: BTreeMap<String, (bool, String)>,
    /// The context it brings the server to, and the items it deletes or creates: what a
    /// reconnection replays when the lost connection had not confirmed it.
    pub(super) target: ChatContext,
    mirror_revision: Option<u64>,
    pub(super) touched: BTreeSet<String>,
    tail_creates: BTreeSet<String>,
    work: Option<ContextWork>,
}

impl CtxUpdate {
    fn unsent_from(&self) -> Option<usize> {
        match &self.work {
            Some(ContextWork::Replay(replay)) => Some(replay.emitted),
            _ => None,
        }
    }

    pub(super) fn pending(&self) -> bool {
        self.work.is_some() || !self.awaiting_create.is_empty() || !self.awaiting_delete.is_empty()
    }
}

/// Context preparation and submission share a bounded continuation budget.
const CONTEXT_PAGE: usize = 64;
const QUEUED_CONTEXT_ITEMS: usize = 32;

#[derive(Debug)]
enum ContextWork {
    Replay(ReplayWork),
    Overlay(OverlayWork),
}

#[derive(Debug, Default)]
struct RetiredContextEvents {
    creates: BTreeSet<String>,
    deletes: BTreeSet<String>,
    events: BTreeMap<String, (bool, String)>,
    touched: BTreeSet<String>,
    tails: BTreeSet<String>,
}

impl RetiredContextEvents {
    fn take(update: &mut CtxUpdate) -> Self {
        Self {
            creates: std::mem::take(&mut update.awaiting_create),
            deletes: std::mem::take(&mut update.awaiting_delete),
            events: std::mem::take(&mut update.by_event_id),
            touched: std::mem::take(&mut update.touched),
            tails: std::mem::take(&mut update.tail_creates),
        }
    }

    fn step(&mut self) -> bool {
        self.creates.pop_first().is_some()
            || self.deletes.pop_first().is_some()
            || self.events.pop_first().is_some()
            || self.touched.pop_first().is_some()
            || self.tails.pop_first().is_some()
    }
}

#[derive(Debug)]
struct ReplayWork {
    phase: ReplayPhase,
    cursor: usize,
    emitted: usize,
    write: usize,
    selected: Vec<bool>,
    calls: BTreeSet<String>,
    seen: BTreeSet<String>,
    preserve_empty: bool,
    empty_allowed: BTreeSet<String>,
    retired_ids: BTreeMap<String, usize>,
    retired_update: Option<RetiredContextEvents>,
    all_selected: bool,
    start: usize,
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
    fn new(length: usize, preserve_empty: bool) -> Self {
        Self {
            phase: ReplayPhase::Calls,
            cursor: 0,
            emitted: 0,
            write: 0,
            selected: vec![false; length],
            calls: BTreeSet::new(),
            seen: BTreeSet::new(),
            preserve_empty,
            empty_allowed: BTreeSet::new(),
            retired_ids: BTreeMap::new(),
            retired_update: None,
            all_selected: true,
            start: 0,
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
struct OverlayWork {
    phase: OverlayPhase,
    remote: std::vec::IntoIter<ChatItem>,
    target: ChatContext,
    target_items: Option<std::vec::IntoIter<ChatItem>>,
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
    extra: std::collections::VecDeque<std::vec::IntoIter<ChatItem>>,
    preserve_empty: bool,
    empty_allowed: BTreeSet<String>,
    retired: Option<RetiredContextEvents>,
}

impl OverlayWork {
    fn new(
        remote: ChatContext,
        target: ChatContext,
        touched: BTreeSet<String>,
        unsent_from: Option<usize>,
        tail_creates: BTreeSet<String>,
        preserve_empty: bool,
        empty_allowed: BTreeSet<String>,
    ) -> Self {
        let capacity = remote.len() + target.len();
        Self {
            phase: OverlayPhase::Index,
            remote: remote.items.into_iter(),
            target,
            target_items: None,
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
            preserve_empty,
            empty_allowed,
            retired: None,
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

    fn add_remote(&mut self, remote: ChatContext) {
        if !remote.is_empty() {
            self.extra.push_back(remote.items.into_iter());
            if matches!(self.phase, OverlayPhase::Cleanup) {
                self.phase = OverlayPhase::Extra;
            }
        }
    }

    fn contains(&self, id: &str) -> bool {
        self.positions.contains_key(id)
            || self.target_ids.contains_key(id)
            || self.target.items.iter().any(|item| item.id() == id)
            || self
                .target_items
                .as_ref()
                .is_some_and(|items| items.as_slice().iter().any(|item| item.id() == id))
            || self.remote.as_slice().iter().any(|item| item.id() == id)
            || self.extra.iter().any(|items| items.as_slice().iter().any(|item| item.id() == id))
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
                        } else if self.target_ids.get(item.id()).copied().unwrap_or(false) {
                            let previous = if self.tail_creates.contains(item.id()) {
                                self.tail
                            } else {
                                self.previous
                            };
                            self.add_node(item, previous);
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
                            }
                            return None;
                        }
                        self.extra.pop_front();
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
                    if self.target_ids.pop_first().is_some()
                        || self.touched.pop_first().is_some()
                        || self.tail_creates.pop_first().is_some()
                    {
                        return None;
                    }
                    if self.retired.as_mut().is_some_and(RetiredContextEvents::step) {
                        return None;
                    }
                    // Even collected nodes contain a drop-bearing ChatItem slot. Retire the
                    // owned history incrementally instead of walking it all on this final step.
                    if self.nodes.pop().is_some() {
                        return None;
                    }
                    return Some(ChatContext::new(std::mem::take(&mut self.collected)));
                }
            }
        }
    }
}

impl RealtimeMachine {
    /// The runtime schedules this work below ready controls and only with write-queue capacity.
    pub(super) fn context_pages_pending(&self) -> bool {
        self.ctx_update.as_ref().is_some_and(|update| update.work.is_some())
    }

    pub fn context_work_pending(&self) -> bool {
        self.ctx_update.as_ref().is_some_and(|update| update.work.is_some())
            || !self.deferred_commands.is_empty()
    }

    /// Continue at most one context page; transport admission precedes the next continuation.
    pub fn continue_context(&mut self, _now: Ts) -> Step {
        let mut step = Step::default();
        if self.closing {
            return step;
        }
        if let Some(mut update) = self.ctx_update.take() {
            let mut budget = CONTEXT_PAGE;
            while budget > 0 {
                let Some(work) = update.work.take() else { break };
                match work {
                    ContextWork::Overlay(mut overlay) => {
                        budget -= 1;
                        if let Some(target) = overlay.step() {
                            let length = target.len();
                            update.target = target;
                            update.mirror_revision = Some(0);
                            let mut replay = ReplayWork::new(length, overlay.preserve_empty);
                            replay.empty_allowed = std::mem::take(&mut overlay.empty_allowed);
                            replay.retired_ids = std::mem::take(&mut overlay.positions);
                            update.work = Some(ContextWork::Replay(replay));
                        } else {
                            update.work = Some(ContextWork::Overlay(overlay));
                        }
                    }
                    ContextWork::Replay(mut replay) => match replay.phase {
                        ReplayPhase::Calls => {
                            if replay
                                .retired_update
                                .as_mut()
                                .is_some_and(RetiredContextEvents::step)
                                || replay.retired_ids.pop_first().is_some()
                            {
                                budget -= 1;
                                update.work = Some(ContextWork::Replay(replay));
                                continue;
                            }
                            if let Some(item) = update.target.items.get(replay.cursor) {
                                if let ChatItem::FunctionCall(call) = item {
                                    replay.calls.insert(call.call_id.clone());
                                }
                                if replay.cursor < replay.start {
                                    let selected = update.touched.contains(item.id())
                                        || self.remote.get(item.id()).is_some();
                                    replay.selected[replay.cursor] = selected;
                                    replay.all_selected &= selected;
                                    if selected {
                                        replay.seen.insert(item.id().to_owned());
                                    }
                                }
                                replay.cursor += 1;
                                budget -= 1;
                            } else {
                                replay.phase = ReplayPhase::Emit;
                                replay.cursor = replay.start;
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
                                    let previous = update.target.items[..replay.cursor]
                                        .iter()
                                        .enumerate()
                                        .rev()
                                        .find(|(index, _)| replay.selected[*index])
                                        .map(|(_, item)| item.id().to_owned())
                                        .unwrap_or_else(|| "root".to_owned());
                                    let item = item.clone();
                                    self.create_ctx_item(
                                        &mut step.frames,
                                        &mut update,
                                        Some(previous),
                                        item,
                                    );
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
                                update.target.items.pop();
                                budget -= 1;
                                update.work = Some(ContextWork::Replay(replay));
                            } else {
                                replay.phase = ReplayPhase::Cleanup;
                                update.work = Some(ContextWork::Replay(replay));
                            }
                        }
                        ReplayPhase::Cleanup => {
                            if replay.seen.pop_first().is_some()
                                || replay.calls.pop_first().is_some()
                                || replay.empty_allowed.pop_first().is_some()
                            {
                                budget -= 1;
                                update.work = Some(ContextWork::Replay(replay));
                            }
                        }
                    },
                }
            }
            self.ctx_update = update.pending().then_some(update);
        }
        if !self.context_pages_pending() {
            let mut commands = 0;
            while step.frames.len() < CONTEXT_PAGE && commands < CONTEXT_PAGE {
                let Some(command) = self.deferred_commands.pop_front() else { break };
                commands += 1;
                match command {
                    DeferredCommand::Response(req_id) => {
                        self.deferred_replies.remove(&req_id);
                        if self.pending_replies.contains(&req_id) {
                            if let Some(frame) = self
                                .reply_frames
                                .get(&req_id)
                                .or_else(|| self.say_frames.get(&req_id))
                            {
                                step.frames.push(frame.clone());
                            }
                        }
                    }
                    DeferredCommand::Media(command) => {
                        self.deferred_media_count -= 1;
                        self.deferred_media_bytes -= media_bytes(&command);
                        step.merge(self.apply_command(_now, command));
                    }
                    DeferredCommand::Item(item) => step.merge(self.submit_context_item(item)),
                }
            }
        }
        step
    }

    pub(super) fn context_acknowledged(&mut self, before: u64, is_delete: bool, id: &str) {
        if let Some(update) = &mut self.ctx_update {
            let awaited = if is_delete { &update.awaiting_delete } else { &update.awaiting_create };
            if update.mirror_revision == Some(before) && awaited.contains(id) {
                update.mirror_revision = Some(self.remote.revision());
            }
        }
    }

    pub(super) fn queue_context_item(&mut self, now: Ts, item: ChatItem) -> Step {
        if self.remote.get(item.id()).is_some()
            || self.ctx_update.as_ref().is_some_and(|update| update.target.get_by_id(item.id()).is_some()
                || update.work.as_ref().is_some_and(|work| matches!(work, ContextWork::Overlay(overlay) if overlay.contains(item.id()))))
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
        self.deferred_commands.push_back(DeferredCommand::Item(item));
        Step::default()
    }

    pub(super) fn submit_context_item(&mut self, item: ChatItem) -> Step {
        if self.remote.get(item.id()).is_some()
            || self
                .ctx_update
                .as_ref()
                .is_some_and(|update| update.target.get_by_id(item.id()).is_some())
        {
            return Step::default();
        }
        let mut update = self.ctx_update.take().unwrap_or_default();
        if update.mirror_revision != Some(self.remote.revision()) {
            update.mirror_revision = None;
        }
        update.tail_creates.insert(item.id().to_owned());
        update.target.items.push(item.clone());
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
        let old_revision = self.remote.revision();
        let remote = std::mem::take(&mut self.remote).into_chat_ctx();
        let mut update = self.ctx_update.take().unwrap_or_default();
        let old_work = update.work.take();
        if old_work.is_none() && update.mirror_revision == Some(old_revision) {
            // This target already is the full intended conversation. Moving it avoids an
            // unnecessary overlay and bulk retirement of a second history allocation.
            let mut replay = ReplayWork::new(update.target.len(), true);
            replay.retired_update = Some(RetiredContextEvents::take(&mut update));
            update.mirror_revision = Some(0);
            update.work = Some(ContextWork::Replay(replay));
        } else if let Some(ContextWork::Overlay(mut overlay)) = old_work {
            overlay.add_remote(remote);
            update.work = Some(ContextWork::Overlay(overlay));
        } else if update.target.is_empty()
            && update.awaiting_create.is_empty()
            && update.touched.is_empty()
        {
            update.target = remote;
            update.mirror_revision = Some(0);
            update.work = Some(ContextWork::Replay(ReplayWork::new(update.target.len(), true)));
        } else {
            let (unsent_from, preserve_empty, empty_allowed) = match old_work {
                Some(ContextWork::Replay(replay)) => {
                    (Some(replay.emitted), replay.preserve_empty, replay.empty_allowed)
                }
                _ => (None, true, BTreeSet::new()),
            };
            let target = std::mem::take(&mut update.target);
            update.mirror_revision = None;
            let touched = std::mem::take(&mut update.touched);
            let tail_creates = std::mem::take(&mut update.tail_creates);
            update.work = Some(ContextWork::Overlay(OverlayWork::new(
                remote,
                target,
                touched,
                unsent_from,
                tail_creates,
                preserve_empty,
                empty_allowed,
            )));
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
    Item(ChatItem),
}

fn media_bytes(command: &super::Command) -> usize {
    match command {
        super::Command::PushAudio(audio) => audio.frame.data.len(),
        _ => 0,
    }
}
