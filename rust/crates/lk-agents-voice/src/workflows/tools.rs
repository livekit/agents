use lk_agents_abi::{Modality, ToolFlags, ToolSpec};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct ToolBoxInner {
    tools: Vec<ToolSpec>,
    revision: u64,
}

/// The task's live tool list, shared with the driver (which builds the LLM request from it and
/// hands a changed list to the host, which re-sends it to a realtime session).
#[derive(Debug, Clone, Default)]
pub struct ToolBox {
    inner: Arc<Mutex<ToolBoxInner>>,
}

impl ToolBox {
    pub fn new(tools: Vec<ToolSpec>) -> Self {
        ToolBox { inner: Arc::new(Mutex::new(ToolBoxInner { tools, revision: 0 })) }
    }

    pub fn snapshot(&self) -> Vec<ToolSpec> {
        self.inner.lock().unwrap().tools.clone()
    }

    pub fn revision(&self) -> u64 {
        self.inner.lock().unwrap().revision
    }

    /// `await self.update_tools(current_tools)`. Returns the new [`revision`](Self::revision).
    pub fn replace(&self, tools: Vec<ToolSpec>) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.tools = tools;
        g.revision += 1;
        g.revision
    }

    /// `[t for t in self.tools if t.id != name] + [tool]`: appends, so tool order changes as in
    /// the reference.
    pub fn upsert_last(&self, tool: ToolSpec) -> u64 {
        let mut current = self.snapshot();
        current.retain(|t| t.name != tool.name);
        current.push(tool);
        self.replace(current)
    }
}

/// The values each `_build_confirm_tool(...)` closure captured, tagged with the
/// [`ToolBox::revision`] it was installed at. A call resolves to the newest install at or before
/// the turn's revision; otherwise a same-turn `update_x` would be confirmed unread.
#[derive(Debug, Clone)]
pub struct ConfirmSlot<C> {
    installs: Vec<(u64, C)>,
}

impl<C> Default for ConfirmSlot<C> {
    fn default() -> Self {
        ConfirmSlot { installs: Vec::new() }
    }
}

impl<C> ConfirmSlot<C> {
    /// Bounded; a turn ends before the next begins, so eight is ample.
    const KEEP: usize = 8;

    /// `revision` is [`ToolBox::revision`] *after* the `update_tools` that installed the tool.
    pub fn install(&mut self, revision: u64, captured: C) {
        self.installs.push((revision, captured));
        if self.installs.len() > Self::KEEP {
            self.installs.remove(0);
        }
    }

    pub fn captured_for(&self, turn_revision: u64) -> Option<&C> {
        self.installs.iter().rev().find(|(r, _)| *r <= turn_revision).map(|(_, c)| c)
    }
}

/// `_confirmation_required`. Tri-state: `None` defers to the input modality (text: no confirmation).
pub fn confirmation_required(require_confirmation: Option<bool>, modality: Modality) -> bool {
    match require_confirmation {
        Some(v) => v,
        None => modality == Modality::Audio,
    }
}

/// `IGNORE_ON_ENTER` when `require_explicit_ask`, so the model must ask before filling.
pub fn update_tool_flags(require_explicit_ask: bool) -> ToolFlags {
    ToolFlags { ignore_on_enter: require_explicit_ask }
}
