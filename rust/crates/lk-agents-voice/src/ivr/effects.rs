use crate::agent_session::SpeechCommand;
use crate::trace::Step;

pub enum IvrEffect {
    Journal(Step),
    Reply(SpeechCommand),
}
