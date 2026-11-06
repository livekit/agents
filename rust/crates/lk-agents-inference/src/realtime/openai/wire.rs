//! Typed OpenAI-compatible realtime frames, with deployment-specific session layouts.

use crate::realtime::dialect::{NUM_CHANNELS, Protocol, SAMPLE_RATE};
use base64::Engine as _;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::llm::realtime::RealtimeProvider;
use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role};
use serde_json::Value;
use std::time::SystemTime;

mod client;
mod server;
pub use crate::realtime::protocol::{
    Incoming, ResponseDone, ServerError, TranscriptionCompleted, TranscriptionUsage,
};
pub use client::*;
pub use server::{DecodeError, decode_text};

mod role {
    use lk_agents_abi::Role;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(role: &Role, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(role.as_str())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Role, D::Error> {
        let value = String::deserialize(deserializer)?;
        Role::from_wire(&value)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown role {value:?}")))
    }
}
#[cfg(test)]
pub use server::{decode, decode_at};

/// Azure's deployment, for the URL.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Azure {
    pub deployment: Option<String>,
}

/// `process_base_url`: `http(s)` -> `ws(s)` as a prefix swap; `/realtime` appended only to an empty
/// or recognised base path (xAI's `/v1/realtime` is untouched). Not Azure: `model` is added as a
/// query parameter unless present. Azure: the deployment as `model`.
pub fn process_base_url(url: &str, model: &str, azure: Option<&Azure>) -> String {
    let url = if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        url.to_owned()
    };

    let (base, query) = match url.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (url.as_str(), None),
    };

    // Everything up to the first `/` after `://` is the authority.
    let (authority, path) = match base.find("://") {
        Some(i) => match base[i + 3..].find('/') {
            Some(j) => base.split_at(i + 3 + j),
            None => (base, ""),
        },
        None => (base, ""),
    };

    let stripped = path.trim_end_matches('/');
    let path = match azure {
        Some(_) => match stripped {
            "" | "/openai" => format!("{stripped}/v1/realtime"),
            "/openai/v1" => "/openai/v1/realtime".to_owned(),
            _ => path.to_owned(),
        },
        None if path.is_empty() || matches!(stripped, "" | "/v1" | "/openai" | "/openai/v1") => {
            format!("{stripped}/realtime")
        }
        None => path.to_owned(),
    };

    let mut params: Vec<(String, String)> = query
        .into_iter()
        .flat_map(|q| q.split('&'))
        .filter(|kv| !kv.is_empty())
        .map(|kv| match kv.split_once('=') {
            Some((k, v)) => (k.to_owned(), v.to_owned()),
            None => (kv.to_owned(), String::new()),
        })
        .collect();
    let has = |params: &[(String, String)], k: &str| params.iter().any(|(p, _)| p == k);
    match azure {
        Some(Azure { deployment: Some(d) }) if !has(&params, "model") => {
            params.push(("model".into(), d.clone()));
        }
        Some(_) => {}
        None if !has(&params, "model") => params.push(("model".into(), model.to_owned())),
        None => {}
    }
    let query: Vec<String> = params
        .into_iter()
        .map(|(k, v)| if v.is_empty() { k } else { format!("{k}={v}") })
        .collect();
    if query.is_empty() {
        format!("{authority}{path}")
    } else {
        format!("{authority}{path}?{}", query.join("&"))
    }
}

/// Deployment adjustments applied before serializing typed client frames.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WireShape {
    pub protocol: Protocol,
    pub beta_event_names: bool,
    pub drop_model: bool,
}

impl WireShape {
    pub fn outbound(&self, frame: &mut ClientEvent) {
        match frame {
            ClientEvent::SessionUpdate { session, .. } => {
                if self.drop_model {
                    session.drop_model();
                }
            }
            ClientEvent::ItemCreate { item, .. } if self.protocol == Protocol::Beta => item.beta(),
            _ => {}
        }
    }
}

/// Where `utils.shortuuid(prefix)` comes from; a port so wire tests can predict `event_id`s.
pub trait IdGen: Send + Sync {
    fn short(&self, prefix: &str) -> String;
}

/// Production: `utils.shortuuid`, six random bytes hex-encoded behind the prefix.
pub struct RandomIds;

impl IdGen for RandomIds {
    fn short(&self, prefix: &str) -> String {
        crate::shortuuid(prefix)
    }
}

pub struct SeqIds(std::sync::atomic::AtomicU64);

impl Default for SeqIds {
    fn default() -> Self {
        SeqIds(std::sync::atomic::AtomicU64::new(0))
    }
}

impl IdGen for SeqIds {
    fn short(&self, prefix: &str) -> String {
        let n = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        format!("{prefix}{n}")
    }
}

const MAX_CALL_ID_LEN: usize = 32;

/// `_shorten_call_id`: OpenAI caps `call_id` at 32 characters. FNV-1a as 32 hex characters, not
/// the reference's SHA-256 prefix; only determinism matters, since both ends of a pair go
/// through this function.
pub fn shorten_call_id(call_id: &str) -> String {
    if call_id.len() <= MAX_CALL_ID_LEN {
        return call_id.to_owned();
    }
    let mut a: u64 = 0xcbf2_9ce4_8422_2325;
    let mut b: u64 = 0x9e37_79b9_7f4a_7c15;
    for (i, byte) in call_id.bytes().enumerate() {
        a = (a ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        b = (b ^ ((byte as u64) << (i % 8))).wrapping_mul(0x100_0000_01b3);
    }
    format!("{a:016x}{b:016x}")
}

const PCM_SAMPLE_BYTES: usize = std::mem::size_of::<i16>();

pub fn audio_frame(pcm: Vec<u8>) -> lk_agents_abi::AudioFrame {
    let samples = (pcm.len() / PCM_SAMPLE_BYTES) as u32;
    lk_agents_abi::AudioFrame {
        data: pcm.into(),
        sample_rate: SAMPLE_RATE,
        num_channels: NUM_CHANNELS,
        samples_per_channel: samples,
    }
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
