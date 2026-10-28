//! A record position anchored at an official, opaque stream offset. Upstream pages are immutable
//! prefixes, but their size can change, so resuming replays and counts records across pages.
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use lk_agents_storage::StorageError;

pub(super) const MAX_BYTES: usize = 16 * 1024 * 1024;
pub(super) const MAX_RECORDS: usize = 1_000_000;
const PREFIX: &str = "as1.";
const MAX_CURSOR: usize = 4096;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Position {
    pub anchor: String,
    pub consumed: usize,
}

impl Position {
    pub fn at(anchor: String) -> Result<Self, StorageError> {
        let position = Self { anchor, consumed: 0 };
        position.validate()?;
        Ok(position)
    }

    fn validate(&self) -> Result<(), StorageError> {
        if self.anchor.is_empty() || self.anchor.len() > 2048 || self.consumed > MAX_RECORDS {
            return Err(invalid("invalid record position"));
        }
        // Reserve space for the largest valid count, including JSON escaping of the opaque
        // anchor, so every cursor we emit also satisfies the decoder's wire-size limit.
        let largest = Self { anchor: self.anchor.clone(), consumed: MAX_RECORDS };
        if serde_json::to_vec(&largest).expect("position serializes").len()
            > (MAX_CURSOR - PREFIX.len()) / 4 * 3
        {
            return Err(invalid("record cursor exceeds budget"));
        }
        Ok(())
    }

    pub fn decode(cursor: &str) -> Result<Self, StorageError> {
        if cursor.len() > MAX_CURSOR {
            return Err(invalid("record cursor exceeds budget"));
        }
        let payload =
            cursor.strip_prefix(PREFIX).ok_or_else(|| invalid("unknown record cursor version"))?;
        let bytes =
            URL_SAFE_NO_PAD.decode(payload).map_err(|_| invalid("malformed record cursor"))?;
        let position: Self =
            serde_json::from_slice(&bytes).map_err(|_| invalid("malformed record cursor"))?;
        position.validate()?;
        Ok(position)
    }

    pub fn encode(&self) -> String {
        format!(
            "{PREFIX}{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(self).expect("position serializes"))
        )
    }
}

pub(super) fn invalid(message: &str) -> StorageError {
    StorageError::Decode(format!("stream cursor: {message}"))
}
