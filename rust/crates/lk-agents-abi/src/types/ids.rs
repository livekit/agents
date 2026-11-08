//! Opaque ids in the protocol's `guid` format (`protocol/utils/guid`): a prefix then twelve
//! characters of the shortuuid alphabet, as LiveKit Cloud mints its own.

#[crate::host(functions)]
pub trait Ids {
    /// A new id: `prefix` then twelve random characters.
    #[host(function)]
    fn shortuuid(#[host(default = "")] prefix: String) -> String;
}
