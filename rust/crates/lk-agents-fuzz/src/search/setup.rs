//! The setup: what a session is built with.
use fastrand::Rng;
use lk_agents_abi::voice::session_options::TextTransformSet;

use crate::{
    drive::{
        AmdAnswer, AmdSetup, ConnSetup, Faults, KeytermDetectionSetup, NodeSetup, PreConnect,
        ProcessorFaults, RealtimeSetup, RoomLegs, RoomSetup, Setup, TtsAlignment,
    },
    input::{AGENT_TURNS, DELAYS_MS, NAMES, TurnMode},
};

/// How often, in a hundred sessions, the pipeline has a recogniser, and has a voice.
const VOICED: u8 = 90;

/// Sample a session to run the case against from the seed's generator alone: every switch a coin,
/// every three-way switch a die. Nothing carries over from one seed to the next, so a seed
/// reproduces its run on its own.
pub fn sample_setup(rng: &mut Rng) -> Setup {
    let names = |rng: &mut Rng| -> Vec<String> {
        (0..rng.usize(0..4)).map(|_| NAMES[rng.usize(0..NAMES.len())].to_owned()).collect()
    };
    let ms = |rng: &mut Rng| DELAYS_MS[rng.usize(0..DELAYS_MS.len())];
    let turn = match rng.usize(0..3) {
        0 => TurnMode::Default,
        1 => TurnMode::Stt,
        _ => TurnMode::Manual,
    };
    // A voice agent hears and speaks: a session without a recogniser or a voice is the rare one.
    let mut setup = Setup {
        stt: rng.u8(0..100) < VOICED,
        vad: rng.bool(),
        eot: rng.bool(),
        interruption: rng.bool(),
        dynamic_endpointing: rng.bool(),
        turn,
        preemptive: rng.bool(),
        tools: if rng.bool() { names(rng) } else { vec![] },
        max_tool_steps: rng.bool().then(|| rng.u32(0..4)),
        instructions: AGENT_TURNS[rng.usize(0..AGENT_TURNS.len())].to_owned(),
        max_unrecoverable_errors: rng.u32(0..4),
        hook_bound_ms: rng.bool().then(|| ms(rng)),
        realtime: rng
            .bool()
            .then(|| RealtimeSetup { turn_detection: rng.bool(), ..RealtimeSetup::default() }),
        // Three ways: no navigator, one the host started, one navigating from the start.
        ivr: match rng.usize(0..3) {
            0 => None,
            1 => Some(false),
            _ => Some(true),
        },
        amd: rng.bool().then_some(AmdSetup::default()),
        keyterms: if rng.bool() { names(rng) } else { vec![] },
        keyterm_detection: None,
        nodes: None,
        room: rng.bool().then(|| RoomSetup {
            close_on_disconnect: rng.bool(),
            delete_room_on_close: rng.bool(),
        }),
        tts: rng.u8(0..100) < VOICED,
        connect_ms: rng.bool().then(|| ms(rng)),
        video: rng.bool(),
        room_camera: false,
        failing_processors: None,
        sync_transcription: rng.bool(),
        text_output_raises: false,
        lagging_output: false,
        pausable_output: rng.bool(),
        speech_gap_ms: rng.bool().then(|| ms(rng)),
        user_turn_max_words: rng.bool().then(|| rng.u32(1..12)),
        user_turn_max_ms: rng.bool().then(|| ms(rng)),
        transcription_timeout_ms: rng.bool().then(|| ms(rng)),
        text_transforms: match rng.usize(0..4) {
            0 | 1 => None,
            2 => Some(None),
            _ => Some(Some(TextTransformSet {
                filter_markdown: rng.bool(),
                filter_emoji: rng.bool(),
            })),
        },
        room_legs: rng.bool().then(|| RoomLegs {
            pre_connect: match rng.usize(0..4) {
                0 => None,
                1 => Some(PreConnect::Sent),
                2 => Some(PreConnect::Never),
                _ => Some(PreConnect::Malformed),
            },
        }),
        console: false,
        video_sampler: false,
        background_audio: false,
        avatar: false,
        avatar_reports_start: false,
        avatar_fps: 25,
        avatar_latency_ms: 20,
        stt_model: 0,
        tts_latency: None,
        conn: None,
        chunked_tts: false,
        tts_alignment: TtsAlignment::None,
        aligned_transcript: None,
        agent_aligned_transcript: None,
        tools_refuse_cancel: false,
        stray: 0.0,
        storage_lost_at: None,
        vad_stall_ms: None,
    };
    // `lk agent console` runs on the host's legs only.
    setup.console = setup.room_legs.is_none() && rng.u8(0..100) < 30;
    // Retired (the deprecated turn keywords), still drawn so every seed keeps its setup.
    rng.u8(0..100);
    setup.video_sampler = setup.video && rng.bool();
    setup.background_audio = setup.room_legs.is_some() && rng.bool();
    setup.avatar = setup.room_legs.is_some() && rng.u8(0..100) < 30;
    setup.avatar_reports_start = setup.avatar && rng.bool();
    if let Some(amd) = setup.amd.as_mut() {
        *amd = AmdSetup {
            interrupt_on_machine: rng.bool(),
            ivr_detection: rng.bool(),
            wait_until_finished: rng.bool(),
            own_stt: rng.bool(),
            // Most callees pick up; one call in eight rings out with nobody joining.
            answer: match rng.usize(0..8) {
                0..=4 => AmdAnswer::Track,
                5 | 6 => AmdAnswer::Sip,
                _ => AmdAnswer::Missing,
            },
        };
    }
    // Drawn last, so a seed draws the same setup as before the switch existed.
    setup.keyterm_detection = (rng.u8(0..100) < 35).then(|| KeytermDetectionSetup {
        stt_takes_keyterms: rng.u8(0..100) < 80,
        turn_interval: rng.u32(0..3),
        max_keyterms: rng.bool().then(|| rng.usize(0..4)),
        timeout_ms: rng.bool().then(|| DELAYS_MS[rng.usize(0..DELAYS_MS.len())]),
    });
    // The room's camera, drawn last too. The host's camera wins over the room's, as
    // `session.input.video` does: a seed that draws the room's has none of its own.
    setup.room_camera = setup.room_legs.is_some() && rng.u8(0..100) < 40;
    setup.video &= !setup.room_camera;
    setup.nodes = (rng.u8(0..100) < 30).then(|| NodeSetup {
        llm: rng.bool(),
        tts: rng.bool(),
        stt: rng.bool(),
        llm_text: None,
        transcription: false,
    });
    // The measured recogniser the transcripts follow, drawn last too.
    setup.stt_model = rng.usize(0..super::STT_MODELS.len());
    // Most voices take a real voice's time to start; some answer at once.
    setup.tts_latency = (rng.u8(0..100) < 80).then(|| rng.u64(..));
    // The realtime server's detector and its own dice, drawn last too.
    if let Some(realtime) = setup.realtime.as_mut() {
        realtime.semantic = rng.bool();
        realtime.seed = rng.u64(..);
        // Half the models call tools at the recorded rate, the rest as often as their prompt
        // has them.
        realtime.tool_calls = rng.bool().then(|| rng.u8(0..=100));
        realtime.text_only = rng.u8(0..100) < 25;
        realtime.fallback = rng.u8(0..100) < 30;
        // A duplex model now and then, from the model's own dice: the draws after it are kept.
        realtime.duplex = realtime.seed % 100 < 15;
    }
    // The legs' retry budgets, and a voice that does not stream, drawn last too.
    setup.conn = rng.bool().then(|| ConnSetup {
        stt: rng.u32(0..4),
        llm: rng.u32(0..4),
        tts: rng.u32(0..4),
        interval_ms: DELAYS_MS[rng.usize(0..DELAYS_MS.len())],
    });
    setup.chunked_tts = rng.u8(0..100) < 30;
    // drawn last, so every other part of a seed's setup stays what it was
    setup.avatar_fps = if rng.bool() { 25 } else { 40 };
    setup.avatar_latency_ms = rng.u64(1..1000 / setup.avatar_fps);
    // The opening agent's transcription node, drawn last too; the agents a handoff installs flip
    // it and their other nodes by name (`overrides`). The realtime audio node is not drawn yet:
    // what the node yields after the core flushed the segment lands in a segment of its own.
    if let Some(nodes) = setup.nodes.as_mut() {
        nodes.transcription = rng.bool();
    }
    // The voice's word alignment, and the session's and the agents' say on using it, drawn last
    // too. Most aligning voices stamp their words; some advertise it and send none.
    setup.tts_alignment = match rng.u8(0..100) {
        0..40 => TtsAlignment::None,
        40..85 => TtsAlignment::Words,
        _ => TtsAlignment::Missing,
    };
    let said = |rng: &mut Rng| match rng.usize(0..3) {
        0 => None,
        1 => Some(false),
        _ => Some(true),
    };
    setup.aligned_transcript = said(rng);
    setup.agent_aligned_transcript = said(rng);
    // Drawn last: one setup in ten has tool bodies that refuse their cancellation.
    setup.tools_refuse_cancel = rng.u8(0..100) < 10;
    // Drawn last: one setup in ten stores its lane and loses the lease at one of its first answers.
    setup.storage_lost_at = (rng.u8(0..100) < 10).then(|| rng.u32(1..6));
    // Drawn last: a third of the room's microphones go through processors that fail. A broken
    // model, license or connection fails a while, at worst for the rest of the call; a glitch
    // fails one frame now and then; a processor flapping at the edge of its budget is the rarest.
    // Gain control is on in half of them.
    let faults = |rng: &mut Rng| match rng.u8(0..100) {
        0..50 => Faults::Span {
            from: rng.u32(0..60),
            frames: if rng.u8(0..4) == 0 { u32::MAX } else { rng.u32(1..250) },
        },
        50..85 => Faults::Rare { from: rng.u32(0..60), every: rng.u32(20..200) },
        _ => {
            let period = rng.u32(2..32);
            Faults::Flapping { period, fails: rng.u32(1..period) }
        }
    };
    setup.failing_processors = (setup.room_legs.is_some() && rng.u8(0..100) < 33).then(|| {
        ProcessorFaults { denoiser: faults(rng), gain_control: rng.bool().then(|| faults(rng)) }
    });
    setup
}
