//! `_on_exit_task`: the `on_exit` each agent owes, run once by whichever of `drain()`, a handoff or
//! the close leaves the agent first.

use crate::agent_activity::hooks::Hooks;
use crate::agent_session::HookId;

/// The driving agent's `on_exit`, and the ones claimed but not yet asked of the host.
#[derive(Default)]
pub(crate) struct OnExit {
    current: State,
    /// Claimed and not yet asked of the host, with the agent each is for.
    claimed: Vec<(HookId, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    Running,
    Exiting(HookId),
    /// The close stopped waiting for it.
    Abandoned(HookId),
    Exited,
}

/// The `on_exit` a leaving agent owes: issued now, already running, or none (it ran).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    Issue(HookId),
    Wait(HookId),
    Done,
}

impl Exit {
    pub(crate) fn hook(self) -> Option<HookId> {
        match self {
            Exit::Issue(hook) | Exit::Wait(hook) => Some(hook),
            Exit::Done => None,
        }
    }
}

impl OnExit {
    /// The driving agent's `on_exit` running now.
    pub(crate) fn exiting(&self) -> Option<HookId> {
        match self.current {
            State::Exiting(hook) | State::Abandoned(hook) => Some(hook),
            State::Running | State::Exited => None,
        }
    }

    pub(crate) fn take_claimed(&mut self) -> Vec<(HookId, String)> {
        std::mem::take(&mut self.claimed)
    }

    pub(crate) fn ended(&mut self, hook: HookId) {
        if self.current == State::Exiting(hook) {
            self.current = State::Exited;
        }
    }

    /// The close: an exit still running is no longer waited on.
    pub(crate) fn abandon(&mut self) {
        if let State::Exiting(hook) = self.current {
            self.current = State::Abandoned(hook);
        }
    }

    /// The swap: the arriving agent owes its own `on_exit`.
    pub(crate) fn swapped(&mut self) {
        self.current = State::Running;
    }
}

/// The driving agent being left: its `on_exit`, claimed by the first to leave it.
pub(crate) struct Leave<'a> {
    pub(crate) on_exit: &'a mut OnExit,
    pub(crate) hooks: &'a mut Hooks,
    pub(crate) agent: &'a str,
}

impl Leave<'_> {
    /// The exit owed, running from now on when it is issued here: asked of the host as soon as
    /// the claimant returns, before anything it raised.
    pub(crate) fn claim(&mut self) -> Exit {
        match self.on_exit.current {
            State::Running => {
                let hook = self.hooks.mint();
                self.on_exit.current = State::Exiting(hook);
                self.on_exit.claimed.push((hook, self.agent.to_owned()));
                Exit::Issue(hook)
            }
            State::Exiting(hook) => Exit::Wait(hook),
            State::Abandoned(_) | State::Exited => Exit::Done,
        }
    }
}
