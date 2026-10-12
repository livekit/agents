use crate::generation::Instructions;
use crate::workflows::{Task, TaskCtx, ToolBox, ToolError, ToolReply};
use async_trait::async_trait;
use lk_agents_abi::{Modality, ToolSpec};
use serde_json::Value;

pub struct Outcome<V> {
    pub reply: ToolReply,
    /// `Some` once the task completed on this call: the erased value or its `ToolError`.
    pub settled: Option<Result<V, ToolError>>,
    /// The live tool list, when this call changed it.
    pub tools: Option<Vec<ToolSpec>>,
}

/// The object-safe half of a native workflow; `V` is the host's erased success value.
#[async_trait]
pub trait TaskDriver<V>: Send {
    fn instructions(&self) -> Instructions;
    fn declared_tools(&self) -> Vec<ToolSpec>;
    async fn call(&mut self, turn: &str, modality: Modality, name: &str, args: Value)
    -> Outcome<V>;
}

/// A task's success type erased to the host's value, supplied by the host.
pub type Erase<T, V> = Box<dyn Fn(T) -> Result<V, ToolError> + Send + Sync>;

pub struct NativeTask<T: Task, V> {
    task: T,
    tools: ToolBox,
    /// One [`TaskCtx`] per turn, keyed on the speech-handle id: its tool-revision snapshot is the
    /// stale-confirmation defence.
    turn: Option<(String, TaskCtx<T::Output>)>,
    erase: Erase<T::Output, V>,
}

impl<T: Task, V> NativeTask<T, V>
where
    T::Output: Clone,
{
    pub fn new(task: T, erase: Erase<T::Output, V>) -> Self {
        // Seeded from `Task::tools()`: confirmation flows filter the live list by name.
        let tools = ToolBox::new(task.tools());
        NativeTask { task, tools, turn: None, erase }
    }
}

impl<T: Task> NativeTask<T, T::Output>
where
    T::Output: Clone,
{
    pub fn typed(task: T) -> Self {
        NativeTask::new(task, Box::new(Ok))
    }
}

#[async_trait]
impl<T: Task, V: Send + 'static> TaskDriver<V> for NativeTask<T, V>
where
    T::Output: Clone,
{
    fn instructions(&self) -> Instructions {
        self.task.instructions()
    }

    fn declared_tools(&self) -> Vec<ToolSpec> {
        self.task.tools()
    }

    async fn call(
        &mut self,
        turn: &str,
        modality: Modality,
        name: &str,
        args: Value,
    ) -> Outcome<V> {
        if !matches!(&self.turn, Some((id, _)) if id == turn) {
            self.turn = Some((turn.to_owned(), TaskCtx::new(modality, self.tools.clone())));
        }
        let ctx = &mut self.turn.as_mut().expect("set immediately above").1;

        let before = self.tools.revision();
        let reply = self.task.call(name, args, ctx).await;
        let tools = (self.tools.revision() != before).then(|| self.tools.snapshot());

        let settled = ctx.take_outcome().map(|outcome| match outcome {
            Ok(value) => (self.erase)(value),
            Err(e) => Err(e),
        });

        Outcome { reply, settled, tools }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflows::ctx::ToolErrorKind;
    use crate::workflows::builtins::credit_card::{
        CARD_COLLECTION_RESTART, GetCardNumberConfig, GetCardNumberTask,
    };
    use crate::workflows::builtins::email_address::{GetEmailConfig, GetEmailTask};

    fn email_driver() -> NativeTask<GetEmailTask, <GetEmailTask as Task>::Output> {
        NativeTask::typed(GetEmailTask::new(GetEmailConfig::default()))
    }

    // Catches: a `TaskCtx` per call instead of per turn (a same-batch confirmation would succeed).
    #[tokio::test]
    async fn a_confirm_from_the_installing_turn_is_not_honoured() {
        let mut d = email_driver();
        let out = d
            .call(
                "speech-1",
                Modality::Audio,
                "update_email_address",
                serde_json::json!({"email": "a@b.co"}),
            )
            .await;
        assert!(
            matches!(out.reply, ToolReply::Text(_)),
            "audio defaults to requiring confirmation"
        );
        assert!(out.tools.is_some(), "update_tools ran, so the host must be handed the new list");
        assert!(out.settled.is_none(), "not complete until confirmed");

        // Same turn: the confirm tool did not exist when this turn was planned.
        let same = d
            .call("speech-1", Modality::Audio, "confirm_email_address", serde_json::json!({}))
            .await;
        assert!(matches!(same.reply, ToolReply::Error(_)), "a same-turn confirm is not bound");
        assert!(same.settled.is_none(), "and it must not complete the task");

        // A later turn planned against the new list confirms.
        let later = d
            .call("speech-2", Modality::Audio, "confirm_email_address", serde_json::json!({}))
            .await;
        assert!(matches!(later.reply, ToolReply::Silent));
        assert!(later.settled.is_some(), "the confirmed address completes the task");
    }

    // Catches: seeding the tool box empty instead of from `Task::tools()`.
    #[tokio::test]
    async fn updating_tools_keeps_the_built_ins() {
        let mut d = email_driver();
        let out = d
            .call(
                "t",
                Modality::Audio,
                "update_email_address",
                serde_json::json!({"email": "a@b.co"}),
            )
            .await;
        let names: Vec<String> =
            out.tools.expect("the list changed").into_iter().map(|s| s.name).collect();
        assert!(names.contains(&"update_email_address".to_owned()));
        assert!(names.contains(&"decline_email_capture".to_owned()));
        assert!(names.contains(&"confirm_email_address".to_owned()));
    }

    // Catches: `require_confirmation`'s third state not deferring to the input modality.
    #[tokio::test]
    async fn text_input_completes_without_a_confirmation_step() {
        let mut d = email_driver();
        let out = d
            .call(
                "t",
                Modality::Text,
                "update_email_address",
                serde_json::json!({"email": "a@b.co"}),
            )
            .await;
        assert!(matches!(out.reply, ToolReply::Silent), "nothing left to say");
        let settled = out.settled.expect("text input completes on the first valid address");
        assert_eq!(settled.expect("with the address").email_address, "a@b.co");
        assert!(out.tools.is_none(), "and no confirm tool is installed");
    }

    // Catches: dropping the `ToolError` subclass on the way back.
    #[tokio::test]
    async fn a_card_subclass_keeps_its_class_name() {
        let mut d = NativeTask::typed(GetCardNumberTask::new(GetCardNumberConfig::default()));
        let out = d
            .call(
                "t",
                Modality::Audio,
                "restart_card_collection",
                serde_json::json!({"reason": "wrong card"}),
            )
            .await;
        let settled = out.settled.expect("declining completes the task");
        let err = settled.err().expect("with a failure");
        assert_eq!(
            err.kind,
            ToolErrorKind::Subclass {
                class_name: CARD_COLLECTION_RESTART.to_owned(),
                reason: "wrong card".to_owned(),
            }
        );
    }
}
