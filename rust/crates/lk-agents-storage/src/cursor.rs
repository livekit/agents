use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::StorageError;

#[derive(Serialize, Deserialize)]
pub(crate) struct Cursor<P> {
    listing: serde_json::Value,
    position: P,
}

impl<P: Serialize + DeserializeOwned> Cursor<P> {
    pub(crate) fn position(
        token: Option<&str>,
        listing: impl Serialize,
        initial: P,
    ) -> Result<P, StorageError> {
        let Some(token) = token.filter(|token| !token.is_empty()) else {
            return Ok(initial);
        };
        let invalid = || StorageError::Invalid("invalid cursor for this listing".into());
        let bytes = URL_SAFE_NO_PAD.decode(token).map_err(|_| invalid())?;
        let cursor: Self = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
        if cursor.listing != serde_json::to_value(listing).map_err(|_| invalid())? {
            return Err(invalid());
        }
        Ok(cursor.position)
    }
}

impl<P: Serialize> Cursor<P> {
    pub(crate) fn encode(listing: impl Serialize, position: P) -> String {
        let cursor = Self {
            listing: serde_json::to_value(listing).expect("a cursor listing is JSON"),
            position,
        };
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&cursor).expect("a cursor position is JSON"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_bound_to_the_complete_listing() {
        let listing = ("tasks", Some("partition_key:*:one"), "updated");
        let token = Cursor::encode(listing, (42_i64, "task:one"));
        assert_eq!(
            Cursor::position(Some(&token), listing, (0_i64, String::new())).unwrap(),
            (42, "task:one".into())
        );
        assert!(Cursor::position(Some(&token), ("items", listing.1, listing.2), 0_i64).is_err());
        assert!(Cursor::position(Some("broken"), listing, 0_i64).is_err());
    }
}
