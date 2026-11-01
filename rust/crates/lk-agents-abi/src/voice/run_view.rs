//! The host's recorded view of one native voice run.

use super::events::RunQuestion;
use crate::{ChatItem, HostRef};

#[crate::host(record(module = "session_wire", emit, transparent))]
#[derive(Clone)]
pub enum VoiceRunOutput {
    Text(String),
    Value(HostRef),
}

#[crate::host(class(name = "_core_VoiceRun"))]
pub trait VoiceRun: Sized + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn record(&mut self, item: ChatItem);
    #[host(call)]
    fn settle(&mut self, status: String, output: Option<HostRef>, question: Option<RunQuestion>);
    #[host(getter)]
    fn status(&self) -> String;
    #[host(getter)]
    fn output(&self) -> Option<VoiceRunOutput>;
    #[host(getter)]
    fn question(&self) -> Option<String>;
    #[host(getter)]
    fn verbatim(&self) -> Option<String>;
    #[host(getter)]
    fn items(&self) -> Vec<ChatItem>;
}
