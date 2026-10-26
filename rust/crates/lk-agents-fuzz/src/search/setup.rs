//! The setup: what a session is built with.
use fastrand::Rng;
use lk_agents_abi::voice::session_options::TextTransformSet;

use crate::{
    drive::{PreConnect, RealtimeSetup, RoomLegs, RoomSetup, Setup},
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
    Setup {
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
        realtime: rng.bool().then(|| RealtimeSetup { turn_detection: rng.bool() }),
        // Three ways: no navigator, one the host started, one navigating from the start.
        ivr: match rng.usize(0..3) {
            0 => None,
            1 => Some(false),
            _ => Some(true),
        },
        amd: rng.bool(),
        keyterms: if rng.bool() { names(rng) } else { vec![] },
        room: rng.bool().then(|| RoomSetup {
            close_on_disconnect: rng.bool(),
            delete_room_on_close: rng.bool(),
        }),
        tts: rng.bool(),
        connect_ms: rng.bool().then(|| ms(rng)),
        video: rng.bool(),
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
    }
}
