//! `AgentSession.agent_state` and `user_state`: who is speaking, the user's turn claims, and the
//! away timer that runs while both listen.

use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::{AgentState, UserState};

use super::effects::{Effects, Timer};
use crate::events::SessionEvent;

pub(crate) struct AgentUserState {
    /// Moved only by [`AgentUserState::set_agent_state`].
    agent: AgentState,
    /// Moved only by [`AgentUserState::update_user_state`].
    user: UserState,
    /// `timers.user_away`: `None` is no away timer at all.
    away_timeout: Option<Duration>,
    /// Whether `Timer::UserAway` is armed, so a restart cancels only a timer that exists.
    away_armed: bool,
    /// While one runs the away timer stays off (#6883).
    running_tool_calls: u32,
    /// `_user_turn_claims`: while > 0 the user state is pinned to `Speaking`.
    claims: u32,
    /// The room's linked participant left: no away window until someone is linked again.
    unlinked: bool,
}

impl AgentUserState {
    pub(crate) fn new(away_timeout: Option<Duration>) -> Self {
        AgentUserState {
            agent: AgentState::Initializing,
            user: UserState::Listening,
            away_timeout,
            away_armed: false,
            running_tool_calls: 0,
            claims: 0,
            unlinked: false,
        }
    }

    pub(crate) fn agent_state(&self) -> AgentState {
        self.agent
    }

    pub(crate) fn someone_speaking(&self) -> bool {
        self.agent == AgentState::Speaking || self.user == UserState::Speaking
    }

    pub(crate) fn user_turn_claims(&self) -> u32 {
        self.claims
    }

    /// The previous state when it changed. Writes the state event itself, so no caller can move
    /// the state without telling the host.
    pub(crate) fn set_agent_state(
        &mut self,
        next: AgentState,
        eff: &mut Effects,
    ) -> Option<AgentState> {
        if self.agent == next {
            return None;
        }
        let old = std::mem::replace(&mut self.agent, next);
        eff.emit(SessionEvent::AgentStateChanged { old, new: next });
        Some(old)
    }

    /// The event, when the state changed. The away timer is reconciled by the loop, not here.
    pub(crate) fn update_user_state(&mut self, at: Ts, next: UserState) -> Effects {
        let mut eff = Effects::new();
        // Pinned to `Speaking` while a claim is held: a VAD `Listening` mid-claim would end the
        // turn under the holder (`AgentSession._update_user_state`).
        if (self.claims > 0 && next != UserState::Speaking) || self.user == next {
            return eff;
        }
        let old = std::mem::replace(&mut self.user, next);
        eff.emit(SessionEvent::UserStateChanged { old, new: next, at });
        eff
    }

    /// `_claim_user_turn`'s enter: the depth after. The first claim writes `Speaking`.
    pub(crate) fn claim_user_turn(&mut self, at: Ts) -> (u32, Effects) {
        self.claims += 1;
        if self.claims != 1 {
            return (self.claims, Effects::new());
        }
        (self.claims, self.update_user_state(at, UserState::Speaking))
    }

    /// `_claim_user_turn`'s exit. Saturating: wrapping would pin the user to `Speaking`. The last
    /// release re-derives the state from `user_speaking`.
    pub(crate) fn release_user_turn(&mut self, at: Ts, user_speaking: bool) -> (u32, Effects) {
        self.claims = self.claims.saturating_sub(1);
        if self.claims != 0 {
            return (self.claims, Effects::new());
        }
        let next = if user_speaking { UserState::Speaking } else { UserState::Listening };
        (self.claims, self.update_user_state(at, next))
    }

    /// Both participants silent: the only state in which the away deadline runs. `Initializing`
    /// counts as the agent's silence.
    fn both_listening(&self) -> bool {
        matches!(self.agent, AgentState::Listening | AgentState::Initializing)
            && self.user == UserState::Listening
    }

    /// `_set_user_away_timer`: cancel (only if armed), then re-arm unless a guard says not to.
    /// Called on every state edge, a final transcript while both listen, and the last tool end.
    pub(crate) fn reconcile_away_timer(&mut self, now: Ts, closing: bool) -> Effects {
        let mut eff = Effects::new();
        if closing {
            return eff;
        }
        if self.away_armed {
            self.away_armed = false;
            eff.cancel(Timer::UserAway);
        }
        let Some(timeout) = self.away_timeout else { return eff };
        if !self.both_listening() || self.running_tool_calls > 0 || self.unlinked {
            return eff;
        }
        self.away_armed = true;
        eff.arm(Timer::UserAway, now.saturating_add(timeout));
        eff
    }

    pub(crate) fn on_user_away(&mut self, now: Ts) -> Effects {
        self.away_armed = false;
        self.update_user_state(now, UserState::Away)
    }

    /// `_user_input_transcribed`'s timer half. `away` returns to `listening`; `listening` under a
    /// listening agent pushes the deadline out (writing `listening` would be a no-op edge and the
    /// away timeout would fire mid-conversation).
    pub(crate) fn on_final_transcript(&mut self, now: Ts, closing: bool) -> Effects {
        match self.user {
            UserState::Away => self.update_user_state(now, UserState::Listening),
            UserState::Listening if self.both_listening() => {
                self.reconcile_away_timer(now, closing)
            }
            UserState::Listening | UserState::Speaking => Effects::new(),
        }
    }

    /// `reset_away_timer`: activity outside the conversation (a participant linked, a key pressed)
    /// brings an away user back, or restarts the window while both listen.
    pub(crate) fn reset_away_timer(&mut self, now: Ts, closing: bool) -> Effects {
        if closing || self.away_timeout.is_none() {
            return Effects::new();
        }
        self.on_final_transcript(now, closing)
    }

    /// The room linked a participant (`true`) or its linked participant left (`false`). Leaving
    /// cancels the away window; a link resets it.
    pub(crate) fn on_link(&mut self, now: Ts, linked: bool, closing: bool) -> Effects {
        self.unlinked = !linked;
        if linked {
            return self.reset_away_timer(now, closing);
        }
        let mut eff = Effects::new();
        if std::mem::take(&mut self.away_armed) {
            eff.cancel(Timer::UserAway);
        }
        eff
    }

    /// A tool call started: the away window is held off (and cancelled) until the last one lands.
    pub(crate) fn on_tool_call_started(&mut self, now: Ts, closing: bool) -> Effects {
        self.running_tool_calls += 1;
        if !self.away_armed {
            return Effects::new();
        }
        self.reconcile_away_timer(now, closing)
    }

    /// A tool call ended; the last one to land while both listen restarts the away window
    /// (#6883). Saturating.
    pub(crate) fn on_tool_call_ended(&mut self, now: Ts, closing: bool) -> Effects {
        self.running_tool_calls = self.running_tool_calls.saturating_sub(1);
        if self.running_tool_calls != 0 || !self.both_listening() {
            return Effects::new();
        }
        self.reconcile_away_timer(now, closing)
    }

    /// The loop cancels every armed timer at the close's edge, the away timer among them.
    pub(crate) fn on_close_edge(&mut self) {
        self.away_armed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_session::effects::Effect;

    fn armed(eff: &Effects) -> bool {
        eff.transitions.iter().any(|e| {
            matches!(
                e,
                crate::agent_session::effects::Transition::Effect(Effect::Arm(Timer::UserAway, _))
            )
        })
    }

    // Catches: the away timer ignoring the room's link. The linked participant leaving must stop
    // the window (no `on_user_away` into an empty room), a later state edge must not re-arm it,
    // and a key pressed or a new link must bring an away user back.
    #[test]
    fn the_away_window_follows_the_link_and_the_keypad() {
        let t = Ts::from_nanos(0);
        let mut s = AgentUserState::new(Some(Duration::from_secs(15)));
        assert!(armed(&s.reconcile_away_timer(t, false)));
        let eff = s.on_link(t, false, false);
        assert!(eff.transitions.iter().any(|e| matches!(
            e,
            crate::agent_session::effects::Transition::Effect(Effect::Cancel(Timer::UserAway))
        )));
        assert!(!armed(&s.reconcile_away_timer(t, false)), "re-armed with nobody linked");
        assert!(armed(&s.on_link(t, true, false)), "a link restarts the window");
        let _ = s.on_user_away(t);
        assert_eq!(s.user, UserState::Away);
        let _ = s.reset_away_timer(t, false);
        assert_eq!(s.user, UserState::Listening, "a key pressed brings the user back");
    }
}
