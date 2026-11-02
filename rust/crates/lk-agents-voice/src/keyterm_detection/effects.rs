//! Work requested by the domain state machine.
use lk_agents_abi::Vocabulary;

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum KeytermEffect {
    Request { req: u64, prompt: String },
    Push(Vec<String>),
    Cancel { req: u64 },
}
