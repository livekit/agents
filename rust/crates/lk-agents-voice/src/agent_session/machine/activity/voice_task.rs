use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::agent_session::{
    AgentUpdate, Answer, GenerateReplyRequest, RequestId, SpeechCommand, VoiceDriver,
};
use crate::trace::{HandoffStep, InterruptCause};
use lk_agents_abi::voice::session::VoiceTaskValue;
use lk_agents_abi::{HostRef, Maybe, Ts};

// The parent's tool keeps running while the task holds the floor; the parent is restored
// without its `on_exit` or `on_enter` however the task ends.

/// Minted per session, in entry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoiceTaskId(pub u64);

/// `VoiceTask.run()` as the core takes it.
#[derive(Debug)]
pub struct VoiceTaskRun {
    /// The task's agent, installed the way `update_agent` installs one.
    pub update: Box<AgentUpdate>,
    /// The agent has no `on_enter` of its own: the task opens with a reply instead.
    pub opens_with_reply: bool,
    pub call: RequestId,
    pub dtmf: Option<crate::workflows::machine::dtmf::DtmfCollector>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_session::effects::VoiceEffect;
    use crate::workflows::machine::dtmf::{DtmfCollector, DtmfEvent, DtmfTaskEvent};
    use std::time::Duration;

    #[test]
    fn keypad_completion_restores_the_parent_and_stops_collection_before_answering() {
        let mut driver = crate::agent_session::tests::driver();
        let parent = AgentUpdate {
            agent: "parent".into(),
            instructions: Default::default(),
            dynamic_instructions: None,
            toolset: Default::default(),
            chat_ctx: None,
            llm: Maybe::NotGiven,
            tts: Maybe::NotGiven,
            stt: Maybe::NotGiven,
            vad: Maybe::NotGiven,
            options: Default::default(),
            expressive: None,
            output_nodes: Default::default(),
            call: None,
            activation: None,
        };
        driver.activity.voice_tasks.stack.push(Suspended {
            id: VoiceTaskId(1),
            call: RequestId(77),
            awaiting: SpeechId(1),
            agent: "agent".into(),
            parent: Box::new(parent),
            dtmf: true,
        });
        let collector =
            DtmfCollector::new(2, false, Duration::from_secs(1), DtmfEvent::Pound).unwrap();
        driver.dtmf_step(DtmfTaskEvent::Start { collector, call: None });
        for digit in ["1", "2", "#"] {
            driver.dtmf_step(DtmfTaskEvent::Digit(digit.into()));
        }
        let effects = driver.settle_step();
        assert_eq!(driver.activity.agent, "parent");
        assert!(!driver.activity.voice_tasks.running());
        assert!(effects.iter().any(|effect| matches!(effect,
            VoiceEffect::Answer(RequestId(77), Answer::VoiceTask(Ok(VoiceTaskValue::Digits(digits)))) if digits == "1 2"
        )));
        driver.dtmf_step(DtmfTaskEvent::Digit("3".into()));
        let effects = driver.settle_step();
        assert!(!effects.iter().any(|effect| matches!(effect, VoiceEffect::Answer(..))));
        assert!(
            !driver.timers.deadline(crate::agent_session::effects::Timer::DtmfDebounce).is_some()
        );
    }

    /// A task's end restores the parent's calls and results, those landed meanwhile included,
    /// in causal order.
    /// https://github.com/livekit/agents/issues/8307 https://github.com/livekit/agents/issues/5284
    /// https://github.com/livekit/agents/issues/4792 https://github.com/livekit/agents/pull/6084
    #[test]
    fn task_restoration_preserves_causal_history() {
        use crate::agent::{Source, Visibility};
        use crate::agent_session::tests::agent_update;
        use crate::agent_session::{SayText, SpeechCommand};
        use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput};
        let at = std::time::UNIX_EPOCH;
        let call = |id: &str| -> ChatItem {
            FunctionCall {
                id: format!("{id}-item"),
                call_id: id.into(),
                name: id.into(),
                arguments: "{}".into(),
                created_at: at,
                extra: Default::default(),
                group_id: None,
            }
            .into()
        };
        let output = |id: &str| -> ChatItem {
            let mut o = FunctionCallOutput::new(format!("{id}-out"), id, "done", false, at);
            o.name = id.into();
            o.into()
        };
        let view = |driver: &VoiceDriver| -> Vec<String> {
            let ctx = driver.session.chat_ctx.agent_chat_ctx();
            ctx.items.iter().map(|i| i.id().to_owned()).collect()
        };

        let mut driver = crate::agent_session::tests::driver();
        driver.on_speech(
            Ts::ZERO,
            SpeechCommand::Say {
                text: SayText::Text("One moment.".into()),
                allow_interruptions: None,
                add_to_chat_ctx: false,
                call: None,
            },
        );
        driver.settle_step();
        let parent_speech = driver.activity.speeches().next().expect("the parent's speech");
        let (tool_call, tool_output) = (
            Source::ToolCall { speech: parent_speech, step: 1 },
            Source::ToolOutput { speech: parent_speech, step: 1 },
        );
        // The parent's step calls `collect` (which awaits the task) and `lookup` in parallel.
        driver.commit_item(call("collect"), tool_call.clone(), Visibility::Both);
        driver.commit_item(call("lookup"), tool_call, Visibility::Both);

        driver.activity.voice_tasks.stack.push(Suspended {
            id: VoiceTaskId(1),
            call: RequestId(77),
            awaiting: parent_speech,
            agent: "task".into(),
            parent: Box::new(agent_update("agent", None)),
            dtmf: false,
        });
        driver.update_agent(Ts::ZERO, agent_update("task", None));
        driver.settle_step();
        // While the task drives, its own pair is recorded and the parent's sibling finishes.
        driver.commit_item(call("task-done"), Source::HostCtx, Visibility::Both);
        driver.commit_item(output("task-done"), Source::HostCtx, Visibility::Both);
        driver.commit_item(output("lookup"), tool_output.clone(), Visibility::Both);
        assert_eq!(
            view(&driver),
            ["task-done-item", "task-done-out"],
            "the task sees its own pair"
        );

        driver.end_voice_tasks(Ts::ZERO, 0, None, Ok(VoiceTaskValue::Digits("1".into())));
        driver.settle_step();
        assert_eq!(driver.activity.agent, "agent");
        driver.commit_item(output("collect"), tool_output, Visibility::Both);
        assert_eq!(
            view(&driver),
            ["collect-item", "lookup-item", "lookup-out", "collect-out"],
            "the parent's calls and results, each once, in the order they happened"
        );
    }
}

#[derive(Debug, lk_agents_abi::Vocabulary)]
#[vocabulary(kind = VoiceTaskCommandKind)]
pub enum VoiceTaskCommand {
    /// From the tool that awaits it (`VoiceCommandOrigin::Tool`).
    Run(Box<VoiceTaskRun>),
    /// The awaiter went away: its task ends, and every task it started.
    Cancel { call: RequestId },
}

pub use lk_agents_abi::voice::session::VoiceTaskEnded;

/// The tasks running, innermost last.
#[derive(Default)]
pub(crate) struct VoiceTasks {
    next: u64,
    stack: Vec<Suspended>,
}

struct Suspended {
    id: VoiceTaskId,
    call: RequestId,
    /// The parent's speech whose tool awaits the task: kept from interruptions while it waits.
    awaiting: SpeechId,
    agent: String,
    /// The parent as it drove: what the task's end installs again.
    parent: Box<AgentUpdate>,
    dtmf: bool,
}

impl VoiceTasks {
    pub(crate) fn running(&self) -> bool {
        !self.stack.is_empty()
    }

    /// A task waits on this speech's tool.
    pub(crate) fn awaits(&self, speech: SpeechId) -> bool {
        self.stack.iter().any(|t| t.awaiting == speech)
    }

    /// The parents held for their restore, whose providers stay held.
    pub(crate) fn parents(&self) -> impl Iterator<Item = &AgentUpdate> {
        self.stack.iter().map(|t| &*t.parent)
    }
}

impl VoiceDriver {
    pub(crate) fn on_voice_task(&mut self, now: Ts, cmd: VoiceTaskCommand) {
        match cmd {
            VoiceTaskCommand::Run(run) => self.enter_voice_task(now, *run),
            VoiceTaskCommand::Cancel { call } => {
                let tasks = &self.activity.voice_tasks.stack;
                if let Some(i) = tasks.iter().position(|t| t.call == call) {
                    self.end_voice_tasks(now, i, None, Err(VoiceTaskEnded::Cancelled));
                }
            }
        }
    }

    /// The task's agent takes the floor: the parent's frame is kept, the task's installed, and the
    /// task opens with its `on_enter` or, without one, a reply. Refused outside a tool of the
    /// driving agent and while the agent changes, drains or the session closes.
    fn enter_voice_task(&mut self, now: Ts, run: VoiceTaskRun) {
        let VoiceTaskRun { mut update, opens_with_reply, call, dtmf } = run;
        let driving = self.activity.agent.as_str();
        let awaiting = self.tool_speech(&self.command_origin).filter(|s| {
            self.activity.speech_scheduler.contains(*s)
                && self.activity.speech_agent(*s) == Some(driving)
        });
        let refused = if self.session.closing.is_some() {
            Some(VoiceTaskEnded::Closing)
        } else if self.activity.in_window() || self.activity.drain.active() {
            Some(VoiceTaskEnded::AgentChanging)
        } else if awaiting.is_none() {
            Some(VoiceTaskEnded::NotInTool)
        } else {
            None
        };
        let (Some(awaiting), None) = (awaiting, &refused) else {
            let why = refused.unwrap_or(VoiceTaskEnded::NotInTool);
            self.record_trace(HandoffStep::TaskRefused { why: why.clone() });
            return self.answer(Some(call), Answer::VoiceTask(Err(why)));
        };
        let tasks = &mut self.activity.voice_tasks;
        tasks.next += 1;
        let id = VoiceTaskId(tasks.next);
        let parent = Box::new(self.driving_frame(&update));
        let agent = update.agent.clone();
        self.record_trace(HandoffStep::TaskEntered {
            task: id,
            parent: parent.agent.clone(),
            agent: agent.clone(),
            speech: awaiting,
        });
        update.call = None;
        // The parent's other speeches end: it is silent while it waits.
        self.cut_speeches_of(now, &parent.agent, Some(awaiting));
        self.arriving(&agent, update.activation.take());
        self.agent_suspended(&parent.agent, &agent);
        let suspended = Suspended { id, call, awaiting, agent, parent, dtmf: dtmf.is_some() };
        self.activity.voice_tasks.stack.push(suspended);
        self.update_agent(now, *update);
        self.activity.on_exit.swapped();
        if let Some(collector) = dtmf {
            self.dtmf_step(crate::workflows::machine::dtmf::DtmfTaskEvent::Start {
                collector,
                call: None,
            });
        }
        if opens_with_reply || self.activity.hook_bound.is_none() {
            self.agent_updated();
        }
        if opens_with_reply {
            let reply = SpeechCommand::GenerateReply(GenerateReplyRequest {
                user_input: None,
                instructions: None,
                allow_interruptions: None,
                tool_choice: None,
                chat_ctx: None,
                toolset: None,
                tool_names: None,
                on_enter: true,
                call: None,
            });
            self.on_speech(now, reply);
        } else {
            self.issue_on_enter(now);
        }
    }

    /// What `agent` still has to say as the floor leaves it: every speech it made but `except` and
    /// the ones a task awaits, queued, playing or in its tools (which finish, with no follow-up).
    fn cut_speeches_of(&mut self, now: Ts, agent: &str, except: Option<SpeechId>) {
        let own: Vec<SpeechId> = (self.activity.speech_scheduler.speech_ids())
            .filter(|s| Some(*s) != except && !self.activity.voice_tasks.awaits(*s))
            .filter(|s| self.activity.speech_agent(*s) == Some(agent))
            .collect();
        for s in own {
            self.force_interrupt_speech(now, s, InterruptCause::VoiceTask);
        }
    }

    /// The driving agent as `update` would replace it: every part `update` gives, so installing
    /// it undoes exactly what `update` changed.
    fn driving_frame(&self, update: &AgentUpdate) -> AgentUpdate {
        let a = &self.activity;
        let s = &self.session;
        fn kept<R>(given: bool, current: Option<R>) -> Maybe<Option<R>> {
            if given { Maybe::Given(current) } else { Maybe::NotGiven }
        }
        AgentUpdate {
            agent: a.agent.clone(),
            instructions: a.instructions.clone(),
            dynamic_instructions: a.dynamic_instructions,
            toolset: (*a.toolset).clone(),
            chat_ctx: None,
            llm: kept(update.llm.is_given(), s.llm.clone()),
            tts: kept(update.tts.is_given(), s.tts.clone()),
            stt: kept(update.stt.is_given(), s.stt.clone()),
            vad: kept(update.vad.is_given(), s.vad.clone()),
            options: s.config.stack.agent.unwrap_or_default(),
            expressive: s.expressive.agent.clone(),
            output_nodes: a.output_nodes.clone(),
            call: None,
            activation: None,
        }
    }

    /// A tool of `speech` completed a task with its validated value, or failed it: the task whose
    /// agent made the speech ends with it, the tasks it started first. An end no task owns is a
    /// plain output.
    pub(crate) fn end_voice_task(
        &mut self,
        now: Ts,
        speech: SpeechId,
        end: Result<HostRef, VoiceTaskEnded>,
    ) {
        let Some(agent) = self.activity.speech_agent(speech) else { return };
        let tasks = &self.activity.voice_tasks.stack;
        if let Some(i) = tasks.iter().rposition(|t| t.agent == agent) {
            self.end_voice_tasks(now, i, Some(speech), end.map(VoiceTaskValue::Value));
        }
    }

    /// A speech whose tool awaited a task ended under it: the task can return to nobody.
    pub(crate) fn voice_task_speech_released(&mut self, now: Ts, speech: SpeechId) {
        let tasks = &self.activity.voice_tasks.stack;
        if let Some(i) = tasks.iter().position(|t| t.awaiting == speech) {
            self.end_voice_tasks(now, i, None, Err(VoiceTaskEnded::Cancelled));
        }
    }

    /// The close ends every task before its `on_exit`: the agent it exits is the root one.
    pub(crate) fn close_voice_tasks(&mut self, now: Ts) {
        if self.activity.voice_tasks.running() {
            self.end_voice_tasks(now, 0, None, Err(VoiceTaskEnded::Closing));
        }
    }

    /// Tasks `from..` end, innermost first: each one's speeches stop, its parent is installed
    /// again (no hooks), and its awaiter is answered. A task ended with the last one is cancelled,
    /// unless the session closes. `speech` is the speech that completed or failed the last one.
    fn end_voice_tasks(
        &mut self,
        now: Ts,
        from: usize,
        speech: Option<SpeechId>,
        end: Result<VoiceTaskValue, VoiceTaskEnded>,
    ) {
        let closing = matches!(end, Err(VoiceTaskEnded::Closing));
        let mut end = Some(end);
        while self.activity.voice_tasks.stack.len() > from {
            let task = self.activity.voice_tasks.stack.pop().expect("stack longer than from");
            if task.dtmf {
                self.dtmf_step(crate::workflows::machine::dtmf::DtmfTaskEvent::Stop);
            }
            let last = self.activity.voice_tasks.stack.len() == from;
            let answer = if last {
                end.take().expect("last task consumes its result once")
            } else if closing {
                Err(VoiceTaskEnded::Closing)
            } else {
                Err(VoiceTaskEnded::Cancelled)
            };
            self.record_trace(HandoffStep::TaskEnded {
                task: task.id,
                how: answer.as_ref().map(|_| ()).map_err(Clone::clone),
                restored: task.parent.agent.clone(),
            });
            self.cut_speeches_of(now, &task.agent, speech);
            self.agent_resumed(&task.agent, &task.parent.agent);
            self.update_agent(now, *task.parent);
            self.activity.on_exit.swapped();
            self.answer(Some(task.call), Answer::VoiceTask(answer));
        }
    }

    pub(crate) fn dtmf_task_finished(
        &mut self,
        now: Ts,
        outcome: crate::workflows::machine::dtmf::DtmfOutcome,
    ) {
        use crate::workflows::machine::dtmf::DtmfOutcome;
        let Some(index) = self.activity.voice_tasks.stack.iter().rposition(|task| task.dtmf) else {
            return;
        };
        let result = match outcome {
            DtmfOutcome::Complete(result) => Ok(VoiceTaskValue::Digits(result.user_input)),
            DtmfOutcome::Incomplete { message } => Err(VoiceTaskEnded::Failed(message)),
            DtmfOutcome::Stopped => return,
        };
        self.end_voice_tasks(now, index, None, result);
    }
}
