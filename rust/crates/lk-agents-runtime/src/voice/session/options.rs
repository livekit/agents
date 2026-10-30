//! `_core.SessionOptions`: `AgentSession(**kwargs)`'s options before the session holds them. The
//! kwargs cross once; the layer they became, and every `update_options` since, is merged by voice's
//! rules.

use lk_agents_abi::Maybe;
use lk_agents_abi::voice::session_options::{
    Expressive, ResolvedSessionOptions, SessionConnOptions, SessionKwargs, SessionLegs,
    UpdateKwargs,
};
use lk_agents_voice::agent_activity::options::{
    AgentSessionOptions, ConfigStack, LegSet, resolved_session_options, session_layer, update_layer,
};
use parking_lot::{Mutex, MutexGuard};
use std::sync::Arc;

/// Shared with the session built over it: a later `update` before `start()` is read at assembly.
#[derive(Clone)]
pub struct SessionOptions(Arc<Mutex<State>>);

struct State {
    /// `conn_options=`: each leg's retry budget and timeout.
    conn: SessionConnOptions,
    /// `expressive=`, then each `update_options(expressive=)`.
    expressive: Expressive,
    session: AgentSessionOptions,
}

impl SessionOptions {
    fn state(&self) -> MutexGuard<'_, State> {
        self.0.lock()
    }

    pub fn session(&self) -> AgentSessionOptions {
        self.state().session
    }

    pub fn conn(&self) -> SessionConnOptions {
        self.state().conn
    }

    /// The session's `expressive=` as of now.
    pub fn expressive(&self) -> Expressive {
        self.state().expressive.clone()
    }
}

impl lk_agents_abi::voice::session::SessionOptions for SessionOptions {
    fn new(kwargs: SessionKwargs) -> Self {
        let session = session_layer(&kwargs);
        let state =
            State { conn: kwargs.conn_options, expressive: kwargs.expressive.clone(), session };
        SessionOptions(Arc::new(Mutex::new(state)))
    }

    fn update(&self, update: UpdateKwargs) {
        let patch = update_layer(&update);
        let mut state = self.state();
        state.session.merge(&patch);
        if let Maybe::Given(expressive) = update.expressive {
            state.expressive = expressive;
        }
    }

    fn resolved(&self, legs: SessionLegs) -> ResolvedSessionOptions {
        let stack = ConfigStack { session: self.session(), agent: None };
        resolved_session_options(&stack.resolve(&LegSet::from(legs)), &stack.session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::session::SessionOptions as _;
    use lk_agents_abi::voice::session_options::{TurnDetection, TurnDetectionKwarg};

    fn legs() -> SessionLegs {
        SessionLegs { stt: true, vad: true, llm: true, tts: true, ..SessionLegs::none() }
    }

    // Catches: `session.options` before `start()` missing an `update_options`, or a detector
    // object passed to `update_options` changing the resolved mode.
    #[test]
    fn options_read_before_start_follow_every_update() {
        let options = SessionOptions::new(crate::voice::session::tests::option_kwargs());
        let mode = |options: &SessionOptions| options.resolved(legs()).turn_detection_mode;
        options.update(UpdateKwargs {
            turn_detection: Maybe::Given(TurnDetectionKwarg::Mode(TurnDetection::Manual)),
            expressive: Maybe::Given(Expressive::default()),
            ..Default::default()
        });
        assert_eq!(mode(&options).as_deref(), Some("manual"));
        options.update(UpdateKwargs {
            turn_detection: Maybe::Given(TurnDetectionKwarg::Detector { streaming: true }),
            ..Default::default()
        });
        assert_eq!(mode(&options).as_deref(), Some("manual"));
        options.update(UpdateKwargs {
            turn_detection: Maybe::Given(TurnDetectionKwarg::Null),
            ..Default::default()
        });
        assert_ne!(mode(&options).as_deref(), Some("manual"));
    }
}
