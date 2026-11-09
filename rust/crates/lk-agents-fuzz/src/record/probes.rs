use super::Observed;

// Injected hang and panic faults, and the host waits the driver armed to see whether they end.

/// What a fake did instead of answering: panicked as a provider future can, or never answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    Panic,
    Hang,
}

/// Where a fault was injected: one host call or provider stream step the fakes answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    ChatOpen,
    ChatNext,
    ChatClose,
    SttOpen,
    SttNext,
    SttClose,
    VadNext,
    VadClose,
    TtsOpen,
    TtsNext,
    TtsClose,
    Predict,
    ToolBody,
    Node,
    /// A hook the host never answers (a hook has no future of the fake's to panic in).
    Hook,
}

impl Site {
    /// A stream's close: a fault there ends the stream as far as its provider is concerned.
    pub fn is_close(self) -> bool {
        matches!(self, Site::ChatClose | Site::SttClose | Site::VadClose | Site::TtsClose)
    }
}

/// One host wait the driver armed: on a speech handle, or one of the host's own closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    Created(u64),
    Scheduled(u64),
    Playout(u64),
    Generation(u64),
    LatestGeneration(u64),
    StepAuthorized(u64),
    Reported(u64),
    /// `aclose()` then `close_io()`.
    SessionClose,
    /// `SessionStorage.aclose()`.
    StoreClose,
    /// The avatar worker's own close.
    AvatarClose,
    /// `BackgroundAudioPlayer.aclose()`.
    BackgroundClose,
}

/// An armed wait: when, and when it resolved, if it did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Probe {
    pub wait: Wait,
    pub armed: Observed,
    pub resolved: Option<Observed>,
}
