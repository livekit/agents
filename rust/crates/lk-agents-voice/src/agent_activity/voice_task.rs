//! `VoiceTask.run()`: from inside a tool of the agent holding the floor, a task's agent takes it.
//! The parent waits in memory, its tool still running, and is put back without its `on_exit` or
//! `on_enter` once the task completes, is cancelled, or the session closes.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::agent_session::{AgentUpdate, Answer, Driver, RequestId, SpeechCommand};
use crate::trace::{HandoffStep, InterruptCause};
use lk_agents_abi::{HostRef, Maybe, Ts};

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
}

#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum VoiceTaskCommand {
    /// From the tool that awaits it (`Origin::Tool`).
    Run(Box<VoiceTaskRun>),
    /// The awaiter went away: its task ends, and every task it started.
    Cancel { call: RequestId },
}

/// Why `VoiceTask.run()` returned no value. A task ended under its awaiter cancels the await, as
/// the close cancels a tool's body.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, lk_agents_abi::Vocabulary)]
pub enum VoiceTaskEnded {
    #[error(
        "VoiceTask.run() must be awaited inside a tool of the agent holding the floor of a live \
         VoiceSession"
    )]
    NotInTool,
    #[error("VoiceTask.run() cannot start while the session's agent is changing or draining")]
    AgentChanging,
    #[error("the session closed before the VoiceTask completed")]
    #[host(kind = "asyncio.CancelledError")]
    Closing,
    #[error("the VoiceTask was cancelled before it completed")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
}

/// How a task ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskEnd {
    /// A tool of the task's agent returned a validated `Completed(value)`.
    Completed,
    /// Its awaiter went away, its awaiting speech ended, or the task that started it ended.
    Cancelled,
    Closing,
}

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

impl Driver {
    pub(crate) fn on_voice_task(&mut self, now: Ts, cmd: VoiceTaskCommand, out: &mut Out) {
        match cmd {
            VoiceTaskCommand::Run(run) => self.enter_voice_task(now, *run, out),
            VoiceTaskCommand::Cancel { call } => {
                let tasks = &self.activity.voice_tasks.stack;
                if let Some(i) = tasks.iter().position(|t| t.call == call) {
                    self.end_voice_tasks(now, i, TaskEnd::Cancelled, None, out);
                }
            }
        }
    }

    /// The task's agent takes the floor: the parent's frame is kept, the task's installed, and the
    /// task opens with its `on_enter` or, without one, a reply. Refused outside a tool of the
    /// driving agent and while the agent changes, drains or the session closes.
    fn enter_voice_task(&mut self, now: Ts, run: VoiceTaskRun, out: &mut Out) {
        let VoiceTaskRun { mut update, opens_with_reply, call } = run;
        let driving = self.activity.agent.as_str();
        let awaiting = self.tool_speech(&self.command_origin).filter(|s| {
            self.activity.speech_scheduler.contains(*s)
                && self.activity.speech_agent(*s) == Some(driving)
        });
        let refused = if self.session.closing.is_some() {
            Some(VoiceTaskEnded::Closing)
        } else if self.in_window() || self.activity.drain.active() {
            Some(VoiceTaskEnded::AgentChanging)
        } else if awaiting.is_none() {
            Some(VoiceTaskEnded::NotInTool)
        } else {
            None
        };
        let (Some(awaiting), None) = (awaiting, refused) else {
            let why = refused.unwrap_or(VoiceTaskEnded::NotInTool);
            out.trace(HandoffStep::TaskRefused { why });
            return out.answer(Some(call), Answer::VoiceTask(Err(why)));
        };
        let tasks = &mut self.activity.voice_tasks;
        tasks.next += 1;
        let id = VoiceTaskId(tasks.next);
        let parent = Box::new(self.driving_frame(&update));
        let agent = update.agent.clone();
        out.trace(HandoffStep::TaskEntered {
            task: id,
            parent: parent.agent.clone(),
            agent: agent.clone(),
            speech: awaiting,
        });
        update.call = None;
        // The parent's other speeches end: it is silent while it waits.
        self.cut_speeches_of(now, &parent.agent, Some(awaiting), out);
        let mut spans = Vec::new();
        let wall = self.step.wall_secs_at(now);
        self.session.trace.agent_suspended(wall, &parent.agent, &mut spans);
        spans.into_iter().for_each(|ev| out.machine().emit(ev));
        let suspended = Suspended { id, call, awaiting, agent, parent };
        self.activity.voice_tasks.stack.push(suspended);
        self.update_agent(now, *update, out);
        self.activity.on_exit.swapped();
        if opens_with_reply || self.activity.hook_bound.is_none() {
            let mut spans = Vec::new();
            self.session.trace.agent_updated(wall, &mut spans);
            spans.into_iter().for_each(|ev| out.machine().emit(ev));
        }
        if opens_with_reply {
            let reply = SpeechCommand::GenerateReply {
                user_input: None,
                instructions: None,
                allow_interruptions: None,
                tool_choice: None,
                chat_ctx: None,
                toolset: None,
                on_enter: true,
                call: None,
            };
            self.on_speech(now, reply, out);
        } else {
            self.issue_on_enter(now, out);
        }
    }

    /// What `agent` still has to say as the floor leaves it: every speech it made but `except` and
    /// the ones a task awaits, queued, playing or in its tools (which finish, with no follow-up).
    fn cut_speeches_of(&mut self, now: Ts, agent: &str, except: Option<SpeechId>, out: &mut Out) {
        let own: Vec<SpeechId> = (self.activity.speech_scheduler.speech_ids())
            .filter(|s| Some(*s) != except && !self.activity.voice_tasks.awaits(*s))
            .filter(|s| self.activity.speech_agent(*s) == Some(agent))
            .collect();
        for s in own {
            let _ = self.interrupt_speech(now, s, InterruptCause::VoiceTask, true, out);
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
            dynamic_instructions: a.dynamic_instructions.clone(),
            toolset: (*a.toolset).clone(),
            tool_retries: a.tool_retries,
            chat_ctx: None,
            llm: kept(update.llm.is_given(), s.llm.clone()),
            tts: kept(update.tts.is_given(), s.tts.clone()),
            stt: kept(update.stt.is_given(), s.stt.clone()),
            vad: kept(update.vad.is_given(), s.vad.clone()),
            options: s.config.stack.agent.unwrap_or_default(),
            expressive: s.expressive.agent.clone(),
            output_nodes: a.output_nodes.clone(),
            call: None,
        }
    }

    /// A tool of `speech` completed a task with its validated value: the task whose agent made the
    /// speech ends with it, the tasks it started first. A completion no task owns is a plain
    /// output.
    pub(crate) fn complete_voice_task(
        &mut self,
        now: Ts,
        speech: SpeechId,
        value: HostRef,
        out: &mut Out,
    ) {
        let Some(agent) = self.activity.speech_agent(speech) else { return };
        let tasks = &self.activity.voice_tasks.stack;
        if let Some(i) = tasks.iter().rposition(|t| t.agent == agent) {
            self.end_voice_tasks(now, i, TaskEnd::Completed, Some((speech, value)), out);
        }
    }

    /// A speech whose tool awaited a task ended under it: the task can return to nobody.
    pub(crate) fn voice_task_speech_released(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let tasks = &self.activity.voice_tasks.stack;
        if let Some(i) = tasks.iter().position(|t| t.awaiting == speech) {
            self.end_voice_tasks(now, i, TaskEnd::Cancelled, None, out);
        }
    }

    /// The close ends every task before its `on_exit`: the agent it exits is the root one.
    pub(crate) fn close_voice_tasks(&mut self, now: Ts, out: &mut Out) {
        if self.activity.voice_tasks.running() {
            self.end_voice_tasks(now, 0, TaskEnd::Closing, None, out);
        }
    }

    /// Tasks `from..` end, innermost first: each one's speeches stop, its parent is installed
    /// again (no hooks), and its awaiter is answered. `how` is the last one's; a task ended with
    /// it is cancelled, unless the session closes. `completed` is the speech that completed the
    /// last one, which ends on its own, and the value.
    fn end_voice_tasks(
        &mut self,
        now: Ts,
        from: usize,
        how: TaskEnd,
        completed: Option<(SpeechId, HostRef)>,
        out: &mut Out,
    ) {
        let (speech, mut value) = completed.map_or((None, None), |(s, v)| (Some(s), Some(v)));
        while self.activity.voice_tasks.stack.len() > from {
            let task = self.activity.voice_tasks.stack.pop().expect("stack longer than from");
            let last = self.activity.voice_tasks.stack.len() == from;
            let how = match how {
                TaskEnd::Completed if !last => TaskEnd::Cancelled,
                other => other,
            };
            let answer = match how {
                TaskEnd::Completed => value.take().ok_or(VoiceTaskEnded::Cancelled),
                TaskEnd::Cancelled => Err(VoiceTaskEnded::Cancelled),
                TaskEnd::Closing => Err(VoiceTaskEnded::Closing),
            };
            out.trace(HandoffStep::TaskEnded {
                task: task.id,
                how,
                restored: task.parent.agent.clone(),
            });
            self.cut_speeches_of(now, &task.agent, speech, out);
            self.update_agent(now, *task.parent, out);
            self.activity.on_exit.swapped();
            out.answer(Some(task.call), Answer::VoiceTask(answer));
        }
    }
}
