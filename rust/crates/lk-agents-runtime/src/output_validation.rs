//! Project host validation into the shared run's verdict without taking acceptance policy.

use lk_agents_abi::HostRef;
use lk_agents_abi::agent::ValidatedOutput;
use lk_agents_abi::errors::HostError;
use lk_agents_core::agent_run::OutputVerdict;

/// Keep the live value and its optional persistent representation separate. Voice retains
/// the live value; direct/text execution can additionally retain the serialized output.
pub(crate) fn project(
    result: Result<ValidatedOutput, HostError>,
) -> (OutputVerdict, Option<HostRef>, Option<serde_json::Value>) {
    match result {
        Ok(ValidatedOutput::Accepted { text, value, serialized }) => {
            (OutputVerdict::Accepted { text }, value, serialized.map(|output| output.value))
        }
        Ok(ValidatedOutput::Retry(reason)) => (OutputVerdict::Retry(reason), None, None),
        Ok(ValidatedOutput::Failed(error)) => (OutputVerdict::Failed(error), None, None),
        Err(error) => (OutputVerdict::Failed(error.to_string()), None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::agent::SerializedOutput;
    use lk_agents_abi::errors::{CapturedError, RefRegistry};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Registry(AtomicUsize);

    impl RefRegistry for Registry {
        fn release(&self, _: u64) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn accepted_live_value_and_serialized_null_are_independent() {
        let registry = Arc::new(Registry::default());
        let port: Arc<dyn RefRegistry> = registry.clone();
        let value = HostRef::new(7, &port);
        let (verdict, retained, serialized) = project(Ok(ValidatedOutput::Accepted {
            text: Some("accepted".into()),
            value: Some(value.clone()),
            serialized: Some(SerializedOutput { value: serde_json::Value::Null }),
        }));
        assert_eq!(verdict, OutputVerdict::Accepted { text: Some("accepted".into()) });
        assert_eq!(retained.as_ref(), Some(&value));
        assert_eq!(serialized, Some(serde_json::Value::Null));
        drop(value);
        assert_eq!(registry.0.load(Ordering::Relaxed), 0, "projection retains the live value");
        drop(retained);
        assert_eq!(registry.0.load(Ordering::Relaxed), 1);

        let (_, retained, serialized) = project(Ok(ValidatedOutput::Accepted {
            text: None,
            value: Some(HostRef::new(8, &port)),
            serialized: None,
        }));
        assert_eq!(retained.as_ref().map(HostRef::id), Some(8));
        assert_eq!(serialized, None);
    }

    #[test]
    fn failed_validation_releases_the_host_error_instead_of_retaining_an_output() {
        let registry = Arc::new(Registry::default());
        let port: Arc<dyn RefRegistry> = registry.clone();
        let error =
            HostError::captured("ValidatorError", "host failed", CapturedError::new(9, &port));
        let (verdict, retained, serialized) = project(Err(error));
        assert_eq!(verdict, OutputVerdict::Failed("ValidatorError: host failed".into()));
        assert_eq!(retained, None);
        assert_eq!(serialized, None);
        assert_eq!(registry.0.load(Ordering::Relaxed), 1);
    }
}
