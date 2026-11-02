//! `_core.SessionOptions`: `AgentSession(**kwargs)`'s options before the session holds them. The
//! kwargs cross once; the layer they became, and every `update_options` since, is merged by voice's
//! rules.

use lk_agents_abi::Maybe;
use lk_agents_abi::voice::session_options::{
    Expressive, ResolvedSessionOptions, SessionConnOptions, SessionKwargs, SessionLegs,
    UpdateKwargs,
};
use lk_agents_voice::agent_activity::options::{
    ConfigStack, OverrideLayer, resolved_session_options, session_layer, update_layer,
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
    /// The session layer and the `override_options` layers held before `start()`.
    layers: ConfigStack,
    /// The last `override_options` layer id minted, before or after `start()`.
    last_layer: u64,
    providers: super::providers::ProviderRegistry,
}

impl SessionOptions {
    fn state(&self) -> MutexGuard<'_, State> {
        self.0.lock()
    }

    /// The session layer and the held override layers the session starts with; the session
    /// holds those layers from now on, released through it.
    pub fn take_layers(&self) -> ConfigStack {
        let mut state = self.state();
        let layers = state.layers.clone();
        state.layers.overrides.clear();
        layers
    }

    pub fn conn(&self) -> SessionConnOptions {
        self.state().conn
    }

    /// The session's `expressive=` as of now.
    pub fn expressive(&self) -> Expressive {
        self.state().expressive.clone()
    }

    /// A fresh `override_options` layer id.
    pub fn next_layer_id(&self) -> u64 {
        let mut state = self.state();
        state.last_layer += 1;
        state.last_layer
    }
}

impl lk_agents_abi::voice::session::SessionOptions for SessionOptions {
    fn new(kwargs: SessionKwargs) -> Self {
        let layers = ConfigStack { session: session_layer(&kwargs), ..Default::default() };
        let state = State {
            conn: kwargs.conn_options,
            expressive: kwargs.expressive.clone().given().unwrap_or_default(),
            layers,
            last_layer: 0,
            providers: Default::default(),
        };
        SessionOptions(Arc::new(Mutex::new(state)))
    }

    fn update(&self, update: UpdateKwargs) {
        let patch = update_layer(&update);
        let mut state = self.state();
        state.layers.session.merge(&patch);
        if let Maybe::Given(expressive) = update.expressive {
            state.expressive = expressive;
        }
    }

    fn override_options(&self, update: UpdateKwargs) -> u64 {
        let id = self.next_layer_id();
        let options = update_layer(&update);
        self.state().layers.overrides.push(OverrideLayer { id, options });
        id
    }

    fn release_override(&self, layer: u64) {
        self.state().layers.overrides.retain(|held| held.id != layer);
    }

    fn update_providers(
        &self,
        providers: super::providers::Providers,
    ) -> super::providers::Providers {
        let (changed, retired) = { self.state().providers.update(providers) };
        drop(retired);
        changed
    }

    fn override_providers(
        &self,
        layer: u64,
        providers: super::providers::Providers,
    ) -> super::providers::Providers {
        let (changed, retired) = { self.state().providers.push(layer, providers) };
        drop(retired);
        changed
    }

    fn release_providers(&self, layer: u64) -> super::providers::Providers {
        let (changed, retired) = { self.state().providers.release(layer) };
        drop(retired);
        changed
    }

    fn resolved(&self, legs: SessionLegs) -> ResolvedSessionOptions {
        let layers = self.state().layers.clone();
        resolved_session_options(&layers.resolve(&legs), &layers.session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::session::SessionOptions as _;
    use lk_agents_abi::voice::session_options::{TurnDetection, TurnDetectionKwarg};

    #[test]
    fn replaced_provider_references_are_released_outside_the_options_lock() {
        use lk_agents_abi::{HostRef, RefRegistry};
        use std::collections::BTreeMap;
        use std::sync::Weak;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Registry {
            state: Weak<Mutex<State>>,
            released: AtomicUsize,
        }
        impl RefRegistry for Registry {
            fn release(&self, _: u64) {
                let state = self.state.upgrade().unwrap();
                assert!(
                    state.try_lock().is_some(),
                    "host reference dropped under the options lock"
                );
                self.released.fetch_add(1, Ordering::Relaxed);
            }
        }

        let options = SessionOptions::new(crate::voice::session::tests::option_kwargs());
        let registry =
            Arc::new(Registry { state: Arc::downgrade(&options.0), released: AtomicUsize::new(0) });
        let host_registry: Arc<dyn RefRegistry> = registry.clone();
        let providers =
            |id| BTreeMap::from([("tts".to_owned(), Some(HostRef::new(id, &host_registry)))]);
        drop(options.update_providers(providers(1)));
        drop(options.override_providers(10, providers(2)));
        drop(options.update_providers(providers(3)));
        assert_eq!(registry.released.load(Ordering::Relaxed), 1);
        drop(options.release_providers(10));
        assert_eq!(registry.released.load(Ordering::Relaxed), 2);
        drop(options.update_providers(BTreeMap::from([("tts".to_owned(), None)])));
        assert_eq!(registry.released.load(Ordering::Relaxed), 3);
    }

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
