use fastrand::Rng;
use std::time::Duration;

use crate::{
    drive::{
        ConnSetup, Faults, KeytermDetectionSetup, NodeSetup, PreConnect, ProcessorFaults,
        RealtimeSetup, RoomLegs, RoomSetup, Setup, StorageSetup, TtsAlignment,
    },
    fakes::a2a_peer::{CancelReply, PeerEnd, PeerScript},
    fakes::{CaptureFault, UnreportedClear},
    input::{AGENT_TURNS, DELAYS_MS, NAMES, TurnMode},
};

// Draw new switches after the existing ones. Replaying an archived campaign still requires
// its original source and configuration; new provider requests change a seed's execution.

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
        input_routing: false,
        remote: None,
        max_tool_steps: rng.bool().then(|| rng.u32(0..4)),
        instructions: AGENT_TURNS[rng.usize(0..AGENT_TURNS.len())].to_owned(),
        max_unrecoverable_errors: rng.u32(0..4),
        hook_bound_ms: rng.bool().then(|| ms(rng)),
        realtime: rng
            .bool()
            .then(|| RealtimeSetup { turn_detection: rng.bool(), ..RealtimeSetup::default() }),
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
        transcription_output_raises: false,
        lagging_output: false,
        output_fault: None,
        unreported_clear: None,
        pausable_output: rng.bool(),
        speech_gap_ms: rng.bool().then(|| ms(rng)),
        user_turn_max_words: rng.bool().then(|| rng.u32(1..12)),
        user_turn_max_ms: rng.bool().then(|| ms(rng)),
        // A niche setting: a few cases, most keep automatic turns.
        push_to_talk: rng.u8(0..100) < 3,
        transcription_timeout_ms: rng.bool().then(|| ms(rng)),
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
        stt_speech: Default::default(),
        amd: None,
        tts_latency: None,
        conn: None,
        chunked_tts: false,
        tts_alignment: TtsAlignment::None,
        tts_markup: "",
        expressive_model: false,
        tts_pacing: false,
        tools_refuse_cancel: false,
        stray: 0.0,
        storage: None,
        vad_stall_ms: None,
        stress: None,
    };
    // `lk agent console` runs on the host's legs only.
    setup.console = setup.room_legs.is_none() && rng.u8(0..100) < 30;
    // Retired (the deprecated turn keywords), still drawn so every seed keeps its setup.
    rng.u8(0..100);
    setup.video_sampler = setup.video && rng.bool();
    setup.background_audio = setup.room_legs.is_some() && rng.bool();
    setup.avatar = setup.room_legs.is_some() && rng.u8(0..100) < 30;
    setup.avatar_reports_start = setup.avatar && rng.bool();
    setup.keyterm_detection = (rng.u8(0..100) < 35).then(|| KeytermDetectionSetup {
        stt_takes_keyterms: rng.u8(0..100) < 80,
        turn_interval: rng.u32(0..3),
        max_keyterms: rng.bool().then(|| rng.usize(0..4)),
        timeout_ms: rng.bool().then(|| DELAYS_MS[rng.usize(0..DELAYS_MS.len())]),
    });
    // The host's camera wins over the room's, as `session.input.video` does.
    setup.room_camera = setup.room_legs.is_some() && rng.u8(0..100) < 40;
    setup.video &= !setup.room_camera;
    setup.nodes = (rng.u8(0..100) < 30).then(|| NodeSetup {
        llm: rng.bool(),
        tts: rng.bool(),
        stt: rng.bool(),
        llm_text: None,
        transcription: false,
    });
    setup.stt_model = rng.usize(0..super::STT_MODELS.len());
    // Most voices take a real voice's time to start; some answer at once.
    setup.tts_latency = (rng.u8(0..100) < 80).then(|| rng.u64(..));
    if let Some(realtime) = setup.realtime.as_mut() {
        realtime.semantic = rng.bool();
        realtime.seed = rng.u64(..);
        // Half the models call tools at the recorded rate, the rest as often as their prompt
        // has them.
        realtime.tool_calls = rng.bool().then(|| rng.u8(0..=100));
        realtime.text_only = rng.u8(0..100) < 25;
        realtime.fallback = rng.u8(0..100) < 30;
        // From the model's own dice, so the draws after it are kept.
        realtime.duplex = realtime.seed % 100 < 15;
    }
    setup.conn = rng.bool().then(|| ConnSetup {
        stt: rng.u32(0..4),
        llm: rng.u32(0..4),
        tts: rng.u32(0..4),
        interval_ms: DELAYS_MS[rng.usize(0..DELAYS_MS.len())],
    });
    setup.chunked_tts = rng.u8(0..100) < 30;
    setup.avatar_fps = if rng.bool() { 25 } else { 40 };
    setup.avatar_latency_ms = rng.u64(1..1000 / setup.avatar_fps);
    // The opening agent's transcription node; handoff agents flip nodes by name (`overrides`).
    if let Some(nodes) = setup.nodes.as_mut() {
        nodes.transcription = rng.bool();
    }
    // Most aligning voices stamp their words; some advertise alignment and send none.
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
    // Retired (the session's and an agent's aligned-transcript option), still drawn so every
    // seed keeps its setup.
    let _ = (said(rng), said(rng));
    setup.tools_refuse_cancel = rng.u8(0..100) < 10;
    // Most calls store their partition; half of those over a faulty database.
    setup.storage = (rng.u8(0..100) < 80)
        .then(|| StorageSetup { faults: rng.bool(), ..StorageSetup::default() });
    // A broken model, license or connection fails a span; a glitch fails a rare frame; a
    // processor flapping at the edge of its budget is the rarest.
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
    // A call in four is stressed: its fakes panic or hang now and then, its waits probed.
    setup.stress = (rng.u8(0..4) == 0).then(|| rng.u64(..));
    // The voice's dialect, drawn last so every seed keeps the setup it drew before: under one, the
    // fake model writes reasoning and markup into its replies.
    setup.tts_markup =
        ["", "", "cartesia", "inworld", "xai", "fishaudio", "gemini"][rng.usize(0..7)];
    setup.expressive_model = !setup.tts_markup.is_empty();
    // pacing, as production runs it by default, drawn after the dialect for the same reason
    setup.tts_pacing = rng.bool();
    // How the recogniser reports speech, drawn after pacing for the same reason: with its own
    // detection or without; one with it may transcribe a speech only once it ended.
    let provider_vad = rng.u8(0..100) < 70;
    setup.stt_speech = crate::fakes::SttSpeech {
        provider_vad,
        transcribes_after_speech_ends: provider_vad && rng.u8(0..100) < 35,
    };
    if let Some(seed) = setup.stress {
        misbehaving_sink(&mut setup, &mut Rng::with_seed(seed ^ 0x7369_6e6b));
    }
    // Cover enabled correlation alongside the deliberately disabled path. This model has
    // its own request stream; the frontend never acquires implicit response authority.
    setup.input_routing = rng.u8(0..100) < 70;
    // ASR is a separate provider job: its final may trail the reply or a later user's final.
    // Draw after the existing switches so the rest of each seed's setup remains reproducible.
    if let Some(realtime) = setup.realtime.as_mut().filter(|r| !r.duplex) {
        match rng.u8(0..4) {
            0 => {
                realtime.delays.transcription = Some(20_000);
                realtime.delays.transcription_partial = Some(100);
            }
            1 => {
                realtime.delays.first_transcription = Some(25_000);
                realtime.delays.transcription = Some(100);
            }
            _ => {}
        }
    }
    // A frontend in five delegates to a remote A2A peer through its remote tool, drawn last so
    // every seed keeps the setup it drew before. The peer's faults are its own dice.
    if rng.u8(0..100) < 20 {
        setup.remote = Some(peer_script(rng));
        setup.tools.push(crate::fakes::remote_tool::REMOTE_TOOL.into());
    }
    // AMD through its executor on a call in ten, drawn after everything else. The answerer then
    // answers the executor's model requests instead of the gate (`search::conversation::answerer`).
    // A realtime call mostly detects turns on the client, as AMD requires; the rest are refused.
    if rng.u8(0..100) < 10 {
        let realtime = setup.realtime.is_some();
        if let Some(server) = setup.realtime.as_mut().filter(|_| rng.u8(0..100) < 75) {
            server.turn_detection = false;
            server.duplex = false;
        }
        setup.amd = Some(amd(rng, setup.stt, realtime));
        // Mostly a caller in the room to listen to; otherwise AMD finds no participant.
        if setup.room_legs.is_none() && !setup.console && rng.u8(0..100) < 80 {
            setup.room_legs = Some(RoomLegs { pre_connect: None });
        }
    }
    setup
}

/// AMD's models and options for a run of the executor: a decision model or an LLM, AMD's own
/// recogniser or not, an opening, and a stage the application handles. A realtime model hears the
/// caller itself, so AMD may bring its own recogniser without session STT.
fn amd(rng: &mut Rng, stt: bool, realtime: bool) -> crate::fakes::amd::AmdSetup {
    let mut options = crate::fakes::amd::options();
    match rng.u8(0..6) {
        0 => options.opening_text = Some(crate::fakes::amd::OPENING_TEXT.into()),
        1 => options.opening_instructions = Some(crate::fakes::amd::OPENING_INSTRUCTIONS.into()),
        _ => {}
    }
    match rng.u8(0..8) {
        0 => options.screening_instructions = None,
        1 => options.voicemail_instructions = None,
        2 => options.ivr_instructions = None,
        _ => {}
    }
    crate::fakes::amd::AmdSetup {
        decision: rng.u8(0..100) < 25,
        own_stt: (stt || realtime) && rng.bool(),
        inference: rng.bool(),
        options,
        ..Default::default()
    }
}

/// How the remote peer works and misbehaves: slow work, progress bursts, questions, a lost
/// acknowledgement after commit, a dropped stream, a failed task, a refused or lost cancel.
fn peer_script(rng: &mut Rng) -> PeerScript {
    PeerScript {
        work_ms: [0, 50, 400, 3_000, 20_000][rng.usize(0..5)],
        progress: [0, 1, 1, 3, 8][rng.usize(0..5)],
        asks: rng.u8(0..3),
        end: if rng.u8(0..5) == 0 { PeerEnd::Failed } else { PeerEnd::Completed },
        lost_ack: rng.u8(0..4) == 0,
        drop_after: (rng.u8(0..4) == 0).then(|| rng.u8(0..4)),
        cancel: [CancelReply::Confirmed, CancelReply::Refused, CancelReply::Lost][rng.usize(0..3)],
    }
}

/// One stressed call in eight has an audio output that misbehaves the way custom and remote ones
/// do: a frame it never takes, or a clear it never reports (late, or not at all). Never both, and
/// never outside a stressed call, so the calls that model a well-behaved room stay the norm. Drawn
/// from the stress seed, so it leaves every other switch of a seed as it was.
fn misbehaving_sink(setup: &mut Setup, rng: &mut Rng) {
    if rng.u8(0..8) != 0 {
        return;
    }
    if rng.bool() {
        setup.output_fault = Some(CaptureFault { at: rng.u32(0..200), hangs: rng.bool() });
    } else {
        let late = rng.bool().then(|| Duration::from_millis(rng.u64(500..9_000)));
        setup.unreported_clear = Some(UnreportedClear { late });
    }
}
