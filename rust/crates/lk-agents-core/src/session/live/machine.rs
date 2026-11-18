use super::*;
use crate::tool_calls::RunningToolExecution;
use lk_agents_abi::errors::{HostError, HostErrorKind};
use lk_agents_abi::storage::StorageError;

impl Live {
    pub(in crate::session) fn new(origin: LiveTurnOrigin) -> Self {
        Self {
            foreground: Foreground::Running(Turn { id: 1, origin, input: TurnInput::Initial }),
            queue: VecDeque::new(),
            bodies: BTreeMap::new(),
            identities: BTreeMap::new(),
            next_id: 1,
            observation: None,
            cancelled: BTreeSet::new(),
        }
    }

    pub(in crate::session) fn origin(&self) -> Option<&LiveTurnOrigin> {
        match &self.foreground {
            Foreground::Running(turn) | Foreground::Committing(turn) => Some(&turn.origin),
            Foreground::Idle | Foreground::Closed => None,
        }
    }

    pub(in crate::session) fn enqueue(
        &mut self,
        origin: LiveTurnOrigin,
        human: FrozenHuman,
    ) -> Result<(), StorageError> {
        if matches!(self.foreground, Foreground::Closed) {
            return Err(StorageError::Invalid("the live conversation owner has ended".into()));
        }
        if self.queue.len() >= MAX_QUEUED_TURNS {
            return Err(StorageError::Unavailable("the live conversation inbox is full".into()));
        }
        if origin.task_ids.iter().any(|task| self.cancelled.contains(task)) {
            return Err(StorageError::Cancelled("the source business task was canceled".into()));
        }
        if origin.task_ids.is_empty() || origin.request_id.is_empty() {
            return Err(StorageError::Invalid(
                "a live turn requires its admitted task identity".into(),
            ));
        }
        if self.origin().is_some_and(|current| current.request_id == origin.request_id)
            || self.queue.iter().any(|turn| turn.origin.request_id == origin.request_id)
        {
            return Err(StorageError::Invalid("the live request is already admitted".into()));
        }
        self.next_id += 1;
        self.queue.push_back(Turn {
            id: self.next_id,
            origin,
            input: TurnInput::Human(Box::new(human)),
        });
        Ok(())
    }

    pub(in crate::session) fn prepared(&mut self, key: BodyKey, identity: ArgumentIdentity) {
        self.identities.insert(key, identity);
    }

    /// Admits an identified body. The owned-body limit is retryable local pressure.
    pub(in crate::session) fn start_body(
        &mut self,
        key: BodyKey,
        frame_id: FrameId,
        call: FunctionCall,
    ) -> Result<(), HostError> {
        if self.bodies.contains_key(&key) {
            return Ok(());
        }
        if self.bodies.len() >= MAX_LIVE_BODIES {
            return Err(HostError {
                kind: HostErrorKind::Unavailable,
                ..HostError::new("the live conversation owns too many tools")
            });
        }
        let origin = self
            .origin()
            .cloned()
            .ok_or_else(|| HostError::new("a body has no admitted foreground turn"))?;
        let identity = self
            .identities
            .remove(&key)
            .ok_or_else(|| HostError::new("a body has no validated argument identity"))?;
        self.next_id += 1;
        self.bodies.insert(
            key,
            Body {
                execution: self.next_id,
                frame_id,
                origin,
                call,
                identity,
                state: BodyState::Running,
                cancellation_requested: false,
            },
        );
        Ok(())
    }

    /// Only an admitted host body releases its model step. A free-form handler progress
    /// message has neither this key nor its validated identity.
    pub(in crate::session) fn detach(&mut self, key: &BodyKey) -> bool {
        let Some(body) = self.bodies.get_mut(key) else {
            return false;
        };
        if !matches!(body.state, BodyState::Running) {
            return false;
        }
        body.state = BodyState::Detached;
        true
    }

    /// Returns true when the result belongs to a detached body, including a repeated report.
    /// Its delivery cannot be sent to whichever newer request currently owns the floor.
    pub(in crate::session) fn settle(&mut self, key: &BodyKey, report: ToolReport) -> bool {
        if matches!(report.outcome, ToolOutcome::Update { .. }) {
            return self.detached(key);
        }
        self.completed(key, Some(report))
    }

    pub(in crate::session) fn lost(&mut self, key: &BodyKey) -> bool {
        self.completed(key, None)
    }

    fn completed(&mut self, key: &BodyKey, report: Option<ToolReport>) -> bool {
        if matches!(self.foreground, Foreground::Closed) {
            // A closed owner has nobody to deliver to: the body is joined, nothing more.
            self.bodies.remove(key);
            return false;
        }
        let Some(body) = self.bodies.get_mut(key) else {
            return false;
        };
        match body.state {
            BodyState::Running => {
                self.bodies.remove(key);
                false
            }
            BodyState::Detached => {
                body.state = BodyState::Ready(report);
                let origin = body.origin.clone();
                self.next_id += 1;
                self.queue.push_back(Turn {
                    id: self.next_id,
                    origin,
                    input: TurnInput::Delivery(vec![key.clone()]),
                });
                true
            }
            BodyState::Ready(_) | BodyState::Delivering(_) => true,
        }
    }

    pub(in crate::session) fn output_source(&self, key: &BodyKey) -> Option<(&FrameId, &str)> {
        let body = self.bodies.get(key)?;
        matches!(body.state, BodyState::Running | BodyState::Detached)
            .then_some((&body.frame_id, body.call.id.as_str()))
    }

    pub(in crate::session) fn running(&self, current: &str) -> Vec<RunningToolExecution> {
        self.bodies
            .iter()
            .filter_map(|((invocation, _), body)| {
                let external = invocation != current || !matches!(body.state, BodyState::Running);
                (external && matches!(body.state, BodyState::Running | BodyState::Detached)).then(
                    || RunningToolExecution {
                        execution: body.execution,
                        call: body.call.clone(),
                        identity: body.identity.clone(),
                    },
                )
            })
            .collect()
    }

    pub(in crate::session) fn commit_started(&mut self) -> Option<(u64, LiveTurnOrigin, bool)> {
        let old = std::mem::replace(&mut self.foreground, Foreground::Idle);
        let Foreground::Running(turn) = old else {
            self.foreground = old;
            return None;
        };
        let completing = match &turn.input {
            TurnInput::Delivery(keys) => keys.as_slice(),
            TurnInput::Initial | TurnInput::Human(_) | TurnInput::Cancelled => &[],
        };
        let working = self.bodies.iter().any(|(key, body)| {
            body.origin.task_ids == turn.origin.task_ids && !completing.contains(key)
        });
        let result = (turn.id, turn.origin.clone(), working);
        self.foreground = Foreground::Committing(turn);
        Some(result)
    }

    /// Accepted output stays owned until the exact turn's storage/task propagation commits.
    pub(in crate::session) fn committed(&mut self, id: u64) -> bool {
        if !matches!(&self.foreground, Foreground::Committing(turn) if turn.id == id) {
            return false;
        }
        let Foreground::Committing(turn) =
            std::mem::replace(&mut self.foreground, Foreground::Idle)
        else {
            return false;
        };
        if let TurnInput::Delivery(keys) = turn.input {
            for key in keys {
                self.bodies.remove(&key);
            }
        }
        true
    }

    pub(in crate::session) fn awaiting_commit(&self, id: u64) -> bool {
        matches!(&self.foreground, Foreground::Committing(turn) if turn.id == id)
    }

    pub(in crate::session) fn cancellation(&self, execution: u64) -> Option<BodyKey> {
        self.bodies
            .iter()
            .find_map(|(key, body)| (body.execution == execution).then(|| key.clone()))
    }

    pub(in crate::session) fn owns_task(&self, task: &str) -> bool {
        self.origin().is_some_and(|origin| origin.task_ids.iter().any(|id| id == task))
            || self.queue.iter().any(|turn| turn.origin.task_ids.iter().any(|id| id == task))
            || self.bodies.values().any(|body| body.origin.task_ids.iter().any(|id| id == task))
    }

    pub(in crate::session) fn current_cancelled(&self) -> bool {
        self.origin()
            .is_some_and(|origin| origin.task_ids.iter().any(|id| self.cancelled.contains(id)))
    }

    pub(in crate::session) fn cancel_tasks(
        &mut self,
        tasks: &[String],
        current: Option<&str>,
    ) -> Vec<BodyKey> {
        self.cancelled.extend(tasks.iter().cloned());
        for turn in &mut self.queue {
            if turn.origin.task_ids.iter().any(|id| self.cancelled.contains(id))
                && matches!(turn.input, TurnInput::Human(_))
            {
                turn.input = TurnInput::Cancelled;
            }
        }
        self.bodies
            .iter_mut()
            .filter_map(|(key, body)| {
                if !body.origin.task_ids.iter().any(|id| self.cancelled.contains(id))
                    || body.cancellation_requested
                {
                    return None;
                }
                body.cancellation_requested = true;
                (matches!(body.state, BodyState::Running | BodyState::Detached)
                    && !(current == Some(key.0.as_str())
                        && matches!(body.state, BodyState::Running)))
                .then(|| key.clone())
            })
            .collect()
    }

    pub(in crate::session) fn owns_call(&self, frame: &FrameId, call: &str) -> bool {
        self.bodies.values().any(|body| &body.frame_id == frame && body.call.id == call)
    }

    pub(in crate::session) fn pending_for_frame(&self, frame: &FrameId) -> bool {
        self.bodies
            .values()
            .any(|body| &body.frame_id == frame && !matches!(body.state, BodyState::Delivering(_)))
    }

    pub(in crate::session) fn next(
        &mut self,
        active: &FrameId,
        completed_child: bool,
    ) -> Option<Turn> {
        if !matches!(self.foreground, Foreground::Idle) {
            return None;
        }
        // A local child can hold the current agent context. The old body keeps its own
        // result until that context returns; it cannot steal a child's frame or ancestor stack.
        let eligible = self.queue.iter().position(|turn| match &turn.input {
            TurnInput::Delivery(keys) => keys
                .iter()
                .all(|key| self.bodies.get(key).is_some_and(|body| &body.frame_id == active)),
            // An accepted child remains intact until its owned bodies settle. Starting a new
            // model here would replace that accepted output before it reaches its parent.
            TurnInput::Human(_) => !completed_child,
            TurnInput::Initial | TurnInput::Cancelled => true,
        })?;
        let turn = self.queue.remove(eligible)?;
        let input = match turn.input {
            TurnInput::Delivery(keys) => {
                for key in &keys {
                    if let Some(body) = self.bodies.get_mut(key)
                        && let BodyState::Ready(outcome) = &body.state
                    {
                        body.state = BodyState::Delivering(outcome.clone());
                    }
                }
                TurnInput::Delivery(keys)
            }
            input => input,
        };
        // The caller takes only the input payload; ownership remains until commit.
        let returned = Turn { id: turn.id, origin: turn.origin.clone(), input };
        self.foreground = Foreground::Running(Turn {
            id: turn.id,
            origin: turn.origin,
            input: match &returned.input {
                TurnInput::Delivery(keys) => TurnInput::Delivery(keys.clone()),
                TurnInput::Initial | TurnInput::Human(_) | TurnInput::Cancelled => {
                    TurnInput::Initial
                }
            },
        });
        Some(returned)
    }

    pub(in crate::session) fn delivered_outputs(&self, keys: &[BodyKey]) -> Vec<Delivery> {
        keys.iter()
            .filter_map(|key| {
                let body = self.bodies.get(key)?;
                let BodyState::Delivering(outcome) = &body.state else {
                    return None;
                };
                Some(Delivery {
                    invocation_id: key.0.clone(),
                    frame_id: body.frame_id.clone(),
                    call: body.call.clone(),
                    report: outcome.clone(),
                })
            })
            .collect()
    }

    pub(in crate::session) fn detached(&self, key: &BodyKey) -> bool {
        self.bodies.get(key).is_some_and(|body| !matches!(body.state, BodyState::Running))
    }

    pub(in crate::session) fn forget_invocation_preparations(&mut self, invocation: &str) {
        self.identities.retain(|(owner, _), _| owner != invocation);
    }

    pub(in crate::session) fn close(&mut self) -> Vec<BodyKey> {
        self.foreground = Foreground::Closed;
        self.observation = None;
        self.queue.clear();
        self.identities.clear();
        self.bodies.keys().cloned().collect()
    }

    /// No turn runs or waits and no body is owned; a closed owner is idle once its bodies end.
    pub(in crate::session) fn idle(&self) -> bool {
        matches!(self.foreground, Foreground::Idle | Foreground::Closed)
            && self.queue.is_empty()
            && self.bodies.is_empty()
    }
}
