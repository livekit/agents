//! `_on_exit_task`: the `on_exit` each agent owes, run once by whichever of `drain()`, a handoff or
//! the close leaves the agent first.

use crate::agent_activity::hooks::Hooks;
use crate::agent_session::{HookId, NewActivity, PreviousActivity};

/// The driving agent's exit, and the exits of agents paused under an inline task, innermost last.
#[derive(Default)]
pub(crate) struct OnExit {
    current: State,
    paused: Vec<State>,
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

    /// The `on_exit` `hook` ended: every agent's exit it was is over.
    pub(crate) fn ended(&mut self, hook: HookId) {
        for exit in std::iter::once(&mut self.current).chain(self.paused.iter_mut()) {
            if *exit == State::Exiting(hook) {
                *exit = State::Exited;
            }
        }
    }

    /// The close: no exit still running is waited on.
    pub(crate) fn abandon(&mut self) {
        for exit in std::iter::once(&mut self.current).chain(self.paused.iter_mut()) {
            if let State::Exiting(hook) = *exit {
                *exit = State::Abandoned(hook);
            }
        }
    }

    /// The swap: a paused agent keeps its exit beneath the arriving one and gets it back on
    /// resume.
    pub(crate) fn swapped(&mut self, previous: PreviousActivity, next: NewActivity) {
        match (previous, next) {
            (PreviousActivity::Pause, NewActivity::Start) => {
                let paused = std::mem::take(&mut self.current);
                self.paused.push(paused);
            }
            (PreviousActivity::Close, NewActivity::Start) => self.current = State::Running,
            (_, NewActivity::Resume) => self.current = self.paused.pop().unwrap_or_default(),
        }
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
