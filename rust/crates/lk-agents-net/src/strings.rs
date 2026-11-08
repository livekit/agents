//! Transport URL classification and ids in the protocol's `guid` format.

/// Is this a LiveKit Cloud URL? Hostname suffix only, so a path or query cannot fake it.
pub fn is_cloud_url(url: &str) -> bool {
    let Some(host) = host_of(url) else { return false };
    host.ends_with(".livekit.cloud") || host.ends_with(".livekit.run")
}

/// Is this LiveKit's staging cloud? Hostname suffix only, like [`is_cloud_url`].
pub fn is_staging_cloud_url(url: &str) -> bool {
    host_of(url).is_some_and(|host| host.ends_with(".staging.livekit.cloud"))
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

/// The protocol's id alphabet (`guid.IDAlphabet`, shortuuid's default): no `0`, `1`, `I`, `O`, `l`.
pub const ID_ALPHABET: &[u8; 57] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// The random characters after an id's prefix (`guid.Size`).
pub const ID_SIZE: usize = 12;

/// A new id in the protocol's format (`guid.New`): `prefix` then [`ID_SIZE`] random characters of
/// [`ID_ALPHABET`].
///
/// 57^12 ~= 1.2e21 ids per prefix (about 70 bits): two of a billion ids collide with probability
/// ~4e-4, and a collision is even odds only after ~4e10 ids.
pub fn shortuuid(prefix: &str) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut id = String::with_capacity(prefix.len() + ID_SIZE);
    id.push_str(prefix);
    id.extend(
        (0..ID_SIZE).map(|_| char::from(ID_ALPHABET[rng.random_range(0..ID_ALPHABET.len())])),
    );
    id
}

/// Is `id` exactly `prefix` then [`ID_SIZE`] characters of [`ID_ALPHABET`]
/// (`guid.IsValidWithPrefix`)?
pub fn is_valid_id(prefix: &str, id: &str) -> bool {
    id.strip_prefix(prefix)
        .is_some_and(|rest| rest.len() == ID_SIZE && rest.bytes().all(|b| ID_ALPHABET.contains(&b)))
}

/// The host's [`shortuuid`].
pub struct Ids;

impl lk_agents_abi::types::ids::Ids for Ids {
    fn shortuuid(prefix: String) -> String {
        shortuuid(&prefix)
    }
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
    // Catches: staging decided by a substring anywhere in the URL, so a path or query fakes it.
    #[test]
    fn only_the_hostname_decides_staging() {
        assert!(is_staging_cloud_url("wss://p.staging.livekit.cloud"));
        assert!(!is_staging_cloud_url("wss://p.livekit.cloud/x.staging.livekit.cloud"));
        assert!(!is_staging_cloud_url("https://example.com?u=.staging.livekit.cloud"));
    }

    // Catches: an id outside the protocol's format (the server refuses a dispatch's thread id that
    // `guid.IsValidWithPrefix` rejects), or a validator that accepts a wrong length or character.
    #[test]
    fn ids_follow_the_protocol_guid_format() {
        let id = shortuuid("AT_");
        assert!(is_valid_id("AT_", &id), "{id}");
        assert_ne!(shortuuid("AT_"), id);
        for bad in
            ["AT_", "AT_abcdefghijk", "AT_abcdefghijkmn", "AT_0bcdefghijkm", "XX_abcdefghijkm"]
        {
            assert!(!is_valid_id("AT_", bad), "{bad}");
        }
        assert!(is_valid_id("", &shortuuid("")));
    }

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
