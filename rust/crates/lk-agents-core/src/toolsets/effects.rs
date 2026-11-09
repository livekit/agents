#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolsetEffect {
    Setup(u64),
    CancelSetup,
    Grant(u64),
    Fail(u64),
    Released(u64),
    Close,
}
