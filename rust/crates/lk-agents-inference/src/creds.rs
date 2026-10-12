//! Credential resolution, shared verbatim by `inference/{llm,stt,tts}.py`.

use crate::token::TokenError;

/// `api_key if is_given(api_key) else os.getenv("LIVEKIT_INFERENCE_API_KEY",
/// os.getenv("LIVEKIT_API_KEY", ""))`. A set-but-empty `LIVEKIT_INFERENCE_API_KEY` shadows
/// `LIVEKIT_API_KEY`, and an explicit `Some("")` is taken as given.
pub fn resolve_credential(
    explicit: Option<&str>,
    inference_env: Option<&str>,
    livekit_env: Option<&str>,
) -> String {
    if let Some(value) = explicit {
        return value.to_owned();
    }
    // `.or()`, not `.filter(|v| !v.is_empty()).or()`: presence, not truthiness.
    inference_env.or(livekit_env).unwrap_or("").to_owned()
}

pub fn resolve_api_key(explicit: Option<&str>) -> Result<String, TokenError> {
    let inference = std::env::var("LIVEKIT_INFERENCE_API_KEY").ok();
    let livekit = std::env::var("LIVEKIT_API_KEY").ok();
    let key = resolve_credential(explicit, inference.as_deref(), livekit.as_deref());
    if key.is_empty() {
        return Err(TokenError::MissingApiKey);
    }
    Ok(key)
}

pub fn resolve_api_secret(explicit: Option<&str>) -> Result<String, TokenError> {
    let inference = std::env::var("LIVEKIT_INFERENCE_API_SECRET").ok();
    let livekit = std::env::var("LIVEKIT_API_SECRET").ok();
    let secret = resolve_credential(explicit, inference.as_deref(), livekit.as_deref());
    if secret.is_empty() {
        return Err(TokenError::MissingApiSecret);
    }
    Ok(secret)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `inference.filter(non_empty).or(livekit)` (an empty inference key must shadow), or
    // an explicit empty string treated as "not given".
    #[test]
    fn explicit_then_inference_then_general_and_empty_counts_as_set() {
        for (explicit, inference, general, want) in [
            (None, Some(""), Some("general"), ""),
            (None, None, Some("general"), "general"),
            (None, Some("specific"), Some("general"), "specific"),
            (None, None, None, ""),
            (Some(""), Some("specific"), Some("general"), ""),
            (Some("mine"), Some("specific"), None, "mine"),
        ] {
            assert_eq!(resolve_credential(explicit, inference, general), want);
        }
    }
}

/// The two ladders as the host calls them (`inference/_utils.py`).
#[cfg(feature = "python")]
mod python {
    use lk_agents_abi::python::check_pid;
    use pyo3::prelude::*;

    /// The api-key ladder: `explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else `LIVEKIT_API_KEY`.
    ///
    /// Matches `os.getenv(A, os.getenv(B, ""))`: a *set but empty* `LIVEKIT_INFERENCE_API_KEY`
    /// shadows `LIVEKIT_API_KEY`. Returns `""` when nothing resolves; the caller raises with its own
    /// message, and tests the explicit value (truthiness vs `is_given`) itself.
    #[pyfunction]
    #[pyo3(signature = (explicit = None))]
    pub fn inference_resolve_api_key(explicit: Option<&str>) -> PyResult<String> {
        check_pid()?;
        Ok(super::resolve_api_key(explicit).unwrap_or_default())
    }

    /// [`inference_resolve_api_key`] for the secret.
    #[pyfunction]
    #[pyo3(signature = (explicit = None))]
    pub fn inference_resolve_api_secret(explicit: Option<&str>) -> PyResult<String> {
        check_pid()?;
        Ok(super::resolve_api_secret(explicit).unwrap_or_default())
    }

    crate::python::register_fn!(inference_resolve_api_key, inference_resolve_api_secret);
}
