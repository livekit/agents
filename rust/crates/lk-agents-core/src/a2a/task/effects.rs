//! Work produced by [`super::machine::TaskMachine`], executed by runtime's A2A door.
//! `Execute` carries the existing storage/handler [`super::Operation`] contract.

use super::{Answer, Operation};
use lk_agents_abi::storage::StorageError;

#[derive(Debug)]
pub enum A2aTaskEffect {
    Execute { id: u64, operation: Operation },
    Finished(Result<Answer, StorageError>),
}
