//! Persisted chat records derive their codec from the chat declarations. Live SDK media
//! must first be made durable by the application; serializing an in-process handle is refused.

use crate::MediaRef;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for MediaRef {
    fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("live media cannot be persisted; supply durable media first"))
    }
}

impl<'de> Deserialize<'de> for MediaRef {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom("persisted state cannot contain a live media handle"))
    }
}
