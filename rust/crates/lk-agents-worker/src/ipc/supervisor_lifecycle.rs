use super::Exit;
use std::time::Duration;

#[derive(Clone, Copy)]
pub(super) struct Initialization {
    pub started: Duration,
    pub deadline: Duration,
}

#[derive(Clone, Copy)]
pub(super) enum Job {
    Idle,
    Pending { request_id: u64, deadline: Duration },
    Running,
}

#[derive(Clone, Copy)]
pub(super) enum Active {
    Starting { deadline: Duration },
    Initializing(Initialization),
    Ready(Job),
}

#[derive(Clone, Copy)]
pub(super) enum Work {
    Uninitialized,
    Initializing(Initialization),
    Ready(Job),
}

#[derive(Clone, Copy)]
pub(super) enum Ping {
    Ready,
    Unanswered,
}

#[derive(Clone, Copy)]
pub(super) enum Health {
    Idle,
    Monitoring {
        next_ping: Duration,
        ping: Ping,
        pong_deadline: Option<Duration>,
        next_sample: Option<Duration>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Child {
    Running,
    ShuttingDown,
}

#[derive(Clone, Copy)]
pub(super) enum Close {
    Ack { deadline: Duration, child: Child },
    ShuttingDown { deadline: Duration },
    Exit { deadline: Duration },
    Waiting(Child),
}

impl Close {
    pub fn acknowledged(self, deadline: Duration) -> Self {
        match self {
            Self::Ack { child: Child::Running, .. } => Self::ShuttingDown { deadline },
            Self::Ack { child: Child::ShuttingDown, .. } => Self::Exit { deadline },
            phase => phase,
        }
    }

    pub fn shutting_down(self, deadline: Duration) -> Self {
        match self {
            Self::Ack { deadline, .. } => Self::Ack { deadline, child: Child::ShuttingDown },
            Self::ShuttingDown { .. } => Self::Exit { deadline },
            Self::Exit { .. } => self,
            Self::Waiting(_) => Self::Waiting(Child::ShuttingDown),
        }
    }

    pub fn deadline(self) -> Option<Duration> {
        match self {
            Self::Ack { deadline, .. }
            | Self::ShuttingDown { deadline }
            | Self::Exit { deadline } => Some(deadline),
            Self::Waiting(_) => None,
        }
    }

    pub fn child(self) -> Child {
        match self {
            Self::Ack { child, .. } | Self::Waiting(child) => child,
            Self::ShuttingDown { .. } => Child::Running,
            Self::Exit { .. } => Child::ShuttingDown,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Cleanup {
    At(Duration),
    Reported,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Stop {
    NotRequested,
    Requested,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Reader {
    Open,
    Closed,
}

#[derive(Clone, Copy)]
pub(super) enum Drain {
    Until(Duration),
    Closed,
}

#[derive(Clone, Copy)]
pub(super) enum Lifecycle {
    Active { phase: Active, health: Health },
    Closing { phase: Close, cleanup: Cleanup, stop: Stop, reader: Reader, work: Work },
    Exited { exit: Exit, reader: Drain, work: Work, child: Child },
}

impl Lifecycle {
    pub fn exit(self) -> Option<Exit> {
        match self {
            Self::Exited { exit, .. } => Some(exit),
            _ => None,
        }
    }

    pub fn initialization(self) -> Option<Initialization> {
        match self.work() {
            Work::Initializing(init) => Some(init),
            _ => None,
        }
    }

    pub fn job(self) -> Job {
        match self.work() {
            Work::Ready(job) => job,
            _ => Job::Idle,
        }
    }

    pub fn work(self) -> Work {
        match self {
            Self::Active { phase: Active::Starting { .. }, .. } => Work::Uninitialized,
            Self::Active { phase: Active::Initializing(init), .. } => Work::Initializing(init),
            Self::Active { phase: Active::Ready(job), .. } => Work::Ready(job),
            Self::Closing { work, .. } | Self::Exited { work, .. } => work,
        }
    }

    pub fn reader_open(self) -> bool {
        matches!(
            self,
            Self::Active { .. }
                | Self::Closing { reader: Reader::Open, .. }
                | Self::Exited { reader: Drain::Until(_), .. }
        )
    }
}
