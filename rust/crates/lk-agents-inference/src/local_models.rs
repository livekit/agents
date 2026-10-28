//! The local models (the VAD and the end-of-turn mini model) as a build links them in. A native
//! provider registers one [`LocalModels`] at link time and the core calls it directly; with none
//! linked, both come from the host's `livekit.local_inference`, one host call per inference.
//!
//! A provider is a crate linked into the final artifact beside the binding:
//!
//! ```ignore
//! lk_agents_inference::local_models::inventory::submit! {
//!     lk_agents_inference::local_models::LocalModels { init, vad, eot }
//! }
//! ```

use std::sync::Arc;

pub use inventory;

use crate::eot::LocalEotModel;
use crate::vad::VadModelFactory;
use crate::vad::VadModel;

/// One native provider of the local models.
pub struct LocalModels {
    /// Loads the selected weights now (idempotent, safe from any thread), so a forked worker
    /// shares them and the first stream does not pay for them.
    pub init: fn(vad: bool, eot: bool),
    /// A fresh VAD model with its own recurrent state: 16 kHz windows of
    /// [`VAD_WINDOW_SAMPLES`](crate::vad::machine::VAD_WINDOW_SAMPLES).
    pub vad: fn() -> Box<dyn VadModel>,
    /// The end-of-turn model, shared by every stream.
    pub eot: fn() -> Arc<dyn LocalEotModel>,
}

inventory::collect!(LocalModels);

/// The linked provider, if any.
pub fn native() -> Option<&'static LocalModels> {
    inventory::iter::<LocalModels>.into_iter().next()
}

struct NativeFactory(&'static LocalModels);

impl VadModelFactory for NativeFactory {
    fn create(&self) -> Box<dyn VadModel> {
        (self.0.vad)()
    }
}

/// The linked provider's VAD models, or `host`'s.
pub fn vad_factory_or(host: Arc<dyn VadModelFactory>) -> Arc<dyn VadModelFactory> {
    native().map_or(host, |p| Arc::new(NativeFactory(p)) as Arc<dyn VadModelFactory>)
}

/// Whether the build links a native provider of the local models.
#[lk_agents_abi::host(export(name = "local_models_native", pid = false))]
pub fn is_native() -> bool {
    native().is_some()
}

/// Loads the native models' weights; `false` when no provider is linked.
#[lk_agents_abi::host(export(name = "local_models_init", pid = false))]
pub fn init(vad: bool, eot: bool) -> bool {
    native().map(|p| (p.init)(vad, eot)).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Loud;

    impl VadModel for Loud {
        fn predict(&mut self, _window: &[i16]) -> (f32, f64) {
            (1.0, 0.0)
        }
        fn reset(&mut self) {}
    }

    struct Half;

    impl LocalEotModel for Half {
        fn predict(&self, _pcm: &[i16]) -> Result<f64, String> {
            Ok(0.5)
        }
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    static INITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    inventory::submit! {
        LocalModels {
            init: |_, _| {
                INITS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            },
            vad: || Box::new(Loud),
            eot: || Arc::new(Half),
        }
    }

    struct HostFactory;

    impl VadModelFactory for HostFactory {
        fn create(&self) -> Box<dyn VadModel> {
            panic!("the host's model must not be built when a native one is linked")
        }
    }

    // Catches: a linked provider not found, or the host's model still preferred over it.
    #[test]
    fn a_linked_provider_is_preferred_over_the_host() {
        assert!(is_native());
        assert!(init(true, false));
        assert_eq!(INITS.load(std::sync::atomic::Ordering::SeqCst), 1);
        let mut model = vad_factory_or(Arc::new(HostFactory)).create();
        assert_eq!(model.predict(&[0; 512]).0, 1.0);
        assert_eq!((native().unwrap().eot)().predict(&[]), Ok(0.5));
    }
}
