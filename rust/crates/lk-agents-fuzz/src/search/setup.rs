//! The setup: what a session is built with.
use fastrand::Rng;
use lk_agents_abi::voice::session_options::TextTransformSet;

use crate::{
    drive::{
        AmdAnswer, AmdSetup, ConnSetup, KeytermDetectionSetup, NodeSetup, PreConnect, RealtimeSetup, RoomLegs,
        RoomSetup, Setup,
    },
    input::{AGENT_TURNS, DELAYS_MS, NAMES, TurnMode},
};

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
    let mut setup = Setup {
        stt: rng.bool(),
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
        realtime: rng.bool().then(|| RealtimeSetup {
            turn_detection: rng.bool(),
            ..RealtimeSetup::default()
        }),
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
        tts: rng.bool(),
        connect_ms: rng.bool().then(|| ms(rng)),
        video: rng.bool(),
        room_camera: false,
        sync_transcription: rng.bool(),
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
        stt_model: 0,
        tts_latency: None,
        conn: None,
        chunked_tts: false,
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
            answer: match rng.usize(0..4) {
                0 | 1 => AmdAnswer::Track,
                2 => AmdAnswer::Sip,
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
    });
    // The recorded recogniser the transcripts follow, drawn last too.
    setup.stt_model = rng.usize(0..crate::corpus::STT_MODELS.len());
    // Most voices take their recorded time to start; some answer at once.
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
    }
    // The legs' retry budgets, and a voice that does not stream, drawn last too.
    setup.conn = rng.bool().then(|| ConnSetup {
        stt: rng.u32(0..4),
        llm: rng.u32(0..4),
        tts: rng.u32(0..4),
        interval_ms: DELAYS_MS[rng.usize(0..DELAYS_MS.len())],
    });
    setup.chunked_tts = rng.u8(0..100) < 30;
    setup
}
