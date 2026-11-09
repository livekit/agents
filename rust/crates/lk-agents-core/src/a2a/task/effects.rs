use super::{Answer, Operation};
use lk_agents_abi::storage::StorageError;

#[derive(Debug)]
pub enum A2aTaskEffect {
    Execute { id: u64, operation: Operation },
    Finished(Result<Answer, StorageError>),
}
