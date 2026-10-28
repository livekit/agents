//! What handling an [`Event`] reads and writes, for partial-order sampling: two events race when
//! they touch one object and one of them writes it. A pure function of the event, read off the
//! driver's router (`Driver::handle` and the handlers it calls); where a handler's reach is
//! unclear it writes more, never less, since a false independence is what loses an
//! interleaving's weight. `tests/pos.rs commutation` holds it to the session: two events it calls
//! independent must leave the same session in either order.
use lk_agents_abi::resources::ResourceId;
use lk_agents_voice::agent_activity::realtime::RealtimeWake;
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::{
    ActivityCommand, Command, Event, Input, Sent, SpeechCommand, TurnCommand,
};
use lk_agents_voice::data_plane::ForwarderCommand;
use lk_agents_voice::generation::GenEvent;

/// One piece of the session's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Object {
    /// Everything: the close, a handoff, a timer.
    All,
    /// One speech: its handle, steps and playout.
    Speech(SpeechId),
    /// The scheduler: the queue, the current speech, its authorization, the agent's state.
    Floor,
    /// The user's turn: recognition, endpointing, the user's state.
    Turn,
    /// The running agent's conversation record.
    Chat,
    /// The running agent's configuration: instructions, tools, models, options.
    Agent,
    /// The provider legs the core holds and which streams are live.
    Legs,
    /// One recognition stream.
    Stream(ResourceId),
    /// The data plane.
    Forwarder,
    /// The realtime socket's machine.
    Realtime,
    /// The room link.
    Room,
    /// The host hooks in flight.
    Hooks,
    /// The tool calls in flight.
    Tools,
    /// The close's phase.
    Close,
    /// Unrecoverable errors counted toward the close.
    Errors,
    Keyterms,
    Dtmf,
    Recorder,
    Video,
    Run,
    Fillers,
    /// The session's span tree, which every step on a turn, a speech or the conversation
    /// also writes.
    Spans,
    /// The idle waiters and holds.
    Waiters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Access {
    pub object: Object,
    pub write: bool,
}

/// Two access sets race: one object in both, written by at least one (POS*: reads commute).
pub fn races(a: &[Access], b: &[Access]) -> bool {
    a.iter().any(|x| {
        b.iter().any(|y| {
            (x.write || y.write)
                && (x.object == y.object || x.object == Object::All || y.object == Object::All)
        })
    })
}

use Object::*;

fn w(objects: &[Object]) -> Vec<Access> {
    objects.iter().map(|&object| Access { object, write: true }).collect()
}

fn with_reads(mut accesses: Vec<Access>, reads: &[Object]) -> Vec<Access> {
    accesses.extend(reads.iter().map(|&object| Access { object, write: false }));
    accesses
}

pub fn accesses(event: &Event) -> Vec<Access> {
    let mut accesses = match event {
        Event::Wake => w(&[All]),
        Event::Forwarder(ForwarderCommand::ReportCensus) => with_reads(Vec::new(), &[Forwarder]),
        Event::Forwarder(_) => w(&[Forwarder]),
        Event::Input(input) => input_accesses(input),
    };
    let spans = |a: &Access| {
        a.write
            && matches!(
                a.object,
                Speech(_) | Floor | Turn | Chat | Agent | Hooks | Tools | Close | Realtime
            )
    };
    if accesses.iter().any(spans) {
        accesses.push(Access { object: Spans, write: true });
    }
    accesses
}

/// A speech's own event: it writes the speech and reads the scheduler that holds it.
fn speech(id: SpeechId, writes: &[Object]) -> Vec<Access> {
    let mut accesses = w(&[Speech(id)]);
    accesses.extend(w(writes));
    with_reads(accesses, &[Floor, Agent])
}

fn input_accesses(input: &Input) -> Vec<Access> {
    let recognition = |stream: Option<ResourceId>| {
        let mut accesses = w(&[Turn, Floor, Chat]);
        accesses.extend(stream.map(|s| Access { object: Stream(s), write: true }));
        with_reads(accesses, &[Legs, Agent])
    };
    match input {
        Input::Host(Sent { command, .. }) => command_accesses(command),
        Input::Stt(tagged) => {
            if tagged.item.kind == lk_agents_abi::SpeechEventType::RecognitionUsage {
                with_reads(w(&[Turn]), &[Legs])
            } else {
                recognition(Some(tagged.stream))
            }
        }
        Input::Vad(_) | Input::Eot(..) | Input::Overlap(_) => recognition(None),
        Input::Room(lk_agents_rtc::RoomEvent::Close { .. }) => w(&[All]),
        Input::Room(_) => Vec::new(),
        Input::RoomIo(ev) => match ev {
            lk_agents_voice::room_io::RoomIoEvent::ParticipantUnlinked(_) => w(&[All]),
            lk_agents_voice::room_io::RoomIoEvent::ParticipantLinked(_)
            | lk_agents_voice::room_io::RoomIoEvent::KeyPressed => w(&[Turn, Room]),
            lk_agents_voice::room_io::RoomIoEvent::TextInput(_)
            | lk_agents_voice::room_io::RoomIoEvent::DeleteRoomRequested(_) => w(&[Room]),
        },
        Input::Gen(ev) => gen_accesses(ev),
        Input::Realtime(wake) => {
            let mut accesses = w(&[Realtime, Floor, Turn, Chat, Tools]);
            match wake {
                RealtimeWake::ToolsDone { speech, .. } | RealtimeWake::Heard { speech, .. } => {
                    accesses.push(Access { object: Speech(*speech), write: true })
                }
                RealtimeWake::Closed | RealtimeWake::Ended => {
                    accesses.push(Access { object: Close, write: true })
                }
                RealtimeWake::Event(_) => {}
            }
            with_reads(accesses, &[Agent])
        }
        Input::ChannelEnded(_) => with_reads(w(&[Legs, Turn, Floor, Errors, Close]), &[Agent]),
        Input::SttRetried(_) => w(&[Errors, Close]),
        Input::StreamOpened(_) => with_reads(w(&[Legs, Keyterms]), &[Close]),
        // A sampled frame mints the ids an item would, and pushes it to the model.
        Input::Video(_) => with_reads(w(&[Video, Chat, Realtime]), &[Turn, Close]),
        Input::InputStarted { .. } => w(&[Legs]),
        Input::GateSilenced(_) => Vec::new(),
        Input::InputCensus(_) => with_reads(Vec::new(), &[Close]),
    }
}

fn gen_accesses(ev: &GenEvent) -> Vec<Access> {
    match ev {
        GenEvent::FirstFrame { speech: id } | GenEvent::PlaybackStarted { speech: id } => {
            speech(*id, &[Floor, Turn])
        }
        GenEvent::Done { speech: id, .. }
        | GenEvent::StepSpoken { speech: id, .. }
        | GenEvent::Failed { speech: id, .. } => speech(*id, &[Floor, Turn, Chat, Tools]),
        GenEvent::ToolsFinished { speech: id, .. } => speech(*id, &[Floor, Chat, Tools, Hooks]),
        GenEvent::ToolStep { speech: id, .. } => speech(*id, &[Floor]),
        GenEvent::ChatFinished { speech: id, error, .. } => {
            speech(*id, if error.is_some() { &[Floor] } else { &[] })
        }
        GenEvent::LaunchReleased { speech: id } => with_reads(Vec::new(), &[Speech(*id), Floor]),
        GenEvent::ToolCallStarted { speech: id, .. } => speech(*id, &[Tools, Fillers]),
        GenEvent::ToolCallEnded { speech: id, .. } => speech(*id, &[Tools, Floor, Chat, Fillers]),
        GenEvent::ProviderFailed { speech: id, .. } => speech(*id, &[Errors, Close]),
        GenEvent::KeytermAnswer { .. } => with_reads(w(&[Keyterms]), &[Legs]),
        // A forwarded event may open or close a span.
        GenEvent::Emit(_) => w(&[Spans]),
        GenEvent::Trace(_) => Vec::new(),
    }
}

fn command_accesses(command: &Command) -> Vec<Access> {
    match command {
        Command::Speech(cmd) => match cmd {
            SpeechCommand::Say { .. } | SpeechCommand::GenerateReply { .. } => {
                with_reads(w(&[Floor, Chat]), &[Agent, Turn, Close])
            }
            SpeechCommand::Interrupt { .. } => w(&[Floor, Turn]),
            SpeechCommand::InterruptSpeech { speech, .. }
            | SpeechCommand::SetAllowInterruptions { speech, .. } => w(&[Speech(*speech), Floor]),
            SpeechCommand::CommitUserTurn { .. } => w(&[Turn, Floor, Chat]),
            SpeechCommand::WaitForIdle { .. } => with_reads(w(&[Waiters]), &[Floor, Turn]),
            SpeechCommand::WaitForIdleAndHold { .. } | SpeechCommand::ReleaseIdleHold { .. } => {
                with_reads(w(&[Waiters, Floor]), &[Turn])
            }
            SpeechCommand::ClaimUserTurn | SpeechCommand::ReleaseUserTurn => {
                w(&[Turn, Floor, Waiters])
            }
            SpeechCommand::EndCall { speech } => w(&[Speech(*speech), Close]),
            SpeechCommand::SetRunOutput { speech, .. } => w(&[Speech(*speech), Run]),
            SpeechCommand::ToolUpdate { .. } => w(&[Tools, Floor, Chat]),
            SpeechCommand::FillerStart { speech, .. } => {
                with_reads(w(&[Fillers]), &[Speech(*speech), Floor])
            }
            SpeechCommand::FillerStop { .. } => w(&[Fillers]),
            SpeechCommand::PauseAuthorization | SpeechCommand::ResumeAuthorization => w(&[Floor]),
        },
        Command::Activity(cmd) => match cmd {
            ActivityCommand::Update(_) | ActivityCommand::Drain { .. } => w(&[All]),
            ActivityCommand::UpdateInstructions(_)
            | ActivityCommand::UpdateTools(_)
            | ActivityCommand::UpdateExpressive(_) => w(&[Agent, Chat]),
            ActivityCommand::SetAudioInput(_) => w(&[Legs, Turn]),
            ActivityCommand::SetVideoInput(_) => w(&[Video]),
            ActivityCommand::SetIoEnabled { .. } => w(&[Legs, Turn, Floor]),
            ActivityCommand::UpdateChatCtx(_) => w(&[Chat, Realtime]),
            ActivityCommand::SetAudioOutput(_) => w(&[Legs, Floor]),
            ActivityCommand::SetTranscriptionOutput(_) => w(&[Legs]),
            ActivityCommand::UpdateModels { .. } => w(&[Legs, Agent, Realtime]),
        },
        Command::Turn(cmd) => match cmd {
            TurnCommand::Commit { .. } | TurnCommand::Clear => w(&[Turn, Floor, Chat]),
            TurnCommand::Configure(_) => w(&[Agent, Turn, Floor]),
            TurnCommand::SkipReplies(_) => w(&[Turn]),
        },
        Command::Video(_) => w(&[Video, Chat, Realtime]),
        Command::Close(_) => w(&[All]),
        Command::HookDone { .. } => w(&[Hooks, Floor, Turn, Chat, Tools]),
        Command::HostMetrics(_) => Vec::new(),
        Command::Ivr(lk_agents_voice::agent_session::IvrCommand::AddTools(_)) => w(&[Agent]),
        Command::Ivr(_) => Vec::new(),
        Command::SetStaticKeyterms(_) => w(&[Keyterms]),
        Command::Recorder(_) => w(&[Recorder]),
        Command::Dtmf(_) => w(&[Dtmf, Floor]),
        Command::Run(_) => w(&[Run, Floor, Chat, Agent]),
        Command::VoiceTask(_) => w(&[All]),
        Command::AmdSpan(_) => w(&[Spans]),
    }
}
