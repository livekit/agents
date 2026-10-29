//! Transport URL classification and ephemeral operation identifiers.

/// Is this a LiveKit Cloud URL? Hostname suffix only, so a path or query cannot fake it.
pub fn is_cloud_url(url: &str) -> bool {
    let Some(host) = host_of(url) else { return false };
    host.ends_with(".livekit.cloud") || host.ends_with(".livekit.run")
}

/// Minimal `urlparse().hostname`: scheme, optional userinfo, strip port and IPv6 brackets, lowercase.
pub(crate) fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r)?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    if authority.is_empty() {
        return None;
    }
    let host = if let Some(end) = authority.strip_prefix('[').and_then(|a| a.find(']')) {
        &authority[1..=end]
    } else {
        authority.split(':').next()?
    };
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

/// `utils.shortuuid(prefix)`: `prefix + uuid.uuid4().hex[:12]`, twelve hex characters, all before
/// the version nibble, so six random bytes are equivalent.
pub fn shortuuid(prefix: &str) -> String {
    use rand::RngCore;
    use std::fmt::Write as _;
    let mut bytes = [0u8; 6];
    rand::rng().fill_bytes(&mut bytes);
    let mut out = String::with_capacity(prefix.len() + 12);
    out.push_str(prefix);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// `_observability_url(livekit_url)` (`job.py:58`). The env override wins unvalidated; otherwise
/// only LiveKit Cloud URLs get one, at `https://` + the ws url's hostname.
pub fn observability_url(livekit_url: &str, env_override: Option<&str>) -> Option<String> {
    if let Some(u) = env_override.filter(|u| !u.is_empty()) {
        return Some(u.to_string());
    }
    let host = host_of(livekit_url)?;
    is_cloud_url(livekit_url).then(|| format!("https://{host}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    // Catches: matching the suffix against the whole URL, an undotted suffix, or the port or
    // userinfo leaking into the hostname.
    #[test]
    fn only_the_hostname_decides_cloud() {
        for (url, cloud) in [
            ("wss://my-project.livekit.cloud", true),
            ("https://x.livekit.run/rtc?token=1", true),
            ("wss://p.livekit.cloud:443/x", true),
            ("wss://user:pw@p.livekit.cloud", true),
            ("https://evil.com/?x=.livekit.cloud", false),
            ("https://notlivekit.cloud", false),
            ("not a url", false),
            ("wss://[::1]:7880", false),
        ] {
            assert_eq!(is_cloud_url(url), cloud, "{url}");
        }
    }
    #[test]
    fn the_observability_url_is_cloud_only_unless_overridden() {
        assert_eq!(
            observability_url("wss://project.livekit.cloud", None).as_deref(),
            Some("https://project.livekit.cloud")
        );
        // Port and path are dropped; the host is lowercased.
        assert_eq!(
            observability_url("wss://Project.LiveKit.Cloud:443/rtc", None).as_deref(),
            Some("https://project.livekit.cloud")
        );
        assert_eq!(observability_url("ws://localhost:7880", None), None);
        assert_eq!(observability_url("wss://sfu.example.com", None), None);
        assert_eq!(
            observability_url("ws://localhost:7880", Some("http://collector:4318")).as_deref(),
            Some("http://collector:4318")
        );
        assert_eq!(
            observability_url("wss://project.livekit.cloud", Some("")).as_deref(),
            Some("https://project.livekit.cloud")
        );
    }
}
