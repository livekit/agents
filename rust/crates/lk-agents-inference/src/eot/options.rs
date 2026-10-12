//! Constructor selection, shared by every host. Ambient environment is supplied as data.

/// The turn detector a host constructs: `(model, cloud, warnings)`.
#[lk_agents_abi::host(export(name = "eot_select"))]
pub fn select_wire(
    version: Option<&str>,
    hosted: bool,
    dev: bool,
    url: &str,
    key: &str,
    secret: &str,
    local_fallback: bool,
) -> Result<(&'static str, bool, Vec<String>), String> {
    let mut cloud = version.map_or(hosted || dev, |v| v == "v1");
    let mut warnings = vec![];
    if cloud {
        let missing: Vec<_> = [
            (url, "LIVEKIT_INFERENCE_URL"),
            (key, "LIVEKIT_API_KEY"),
            (secret, "LIVEKIT_API_SECRET"),
        ]
        .into_iter()
        .filter(|(value, _)| value.is_empty())
        .map(|(_, name)| name)
        .collect();
        if !missing.is_empty() {
            let missing = missing.join(", ");
            if version.is_some() {
                return Err(format!(
                    "TurnDetector(version='v1') requires {missing} (env or constructor argument)."
                ));
            }
            warnings.push(format!("LIVEKIT_INFERENCE_URL is set but {missing} missing; falling back to the turn-detector-v1-mini model"));
            cloud = false;
        }
    }
    let model = if cloud { "turn-detector-v1" } else { "turn-detector-v1-mini" };
    if !local_fallback && !cloud {
        warnings.push(format!(
            "local_fallback=False has no effect on the {model} model, which runs locally by design"
        ));
    }
    Ok((model, cloud, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_cloud_requires_credentials_but_auto_degrades() {
        assert!(select_wire(Some("v1"), false, false, "url", "", "secret", true).is_err());
        let (model, cloud, warnings) = select_wire(None, true, false, "url", "", "secret", true).unwrap();
        assert!(!cloud);
        assert_eq!(model, "turn-detector-v1-mini");
        assert_eq!(warnings.len(), 1);
        assert!(select_wire(None, false, true, "url", "key", "secret", true).unwrap().1);
        assert!(!select_wire(Some("v1-mini"), true, true, "url", "key", "secret", true).unwrap().1);
        assert_eq!(select_wire(None, false, false, "", "", "", false).unwrap().2.len(), 1);
    }
}
