//! Background sound selection and per-frame fade progression.

pub use lk_agents_abi::voice::background_audio::{AudioConfig, LoopingStreamError};
use std::f64::consts::PI;

// Kept small because `clear_queue()` would cut ambient sounds abruptly: a sound removed from the
// mixer drains out within this window instead.
pub const AUDIO_SOURCE_BUFFER_MS: u32 = 400;

/// The published track's name; `aclose` resolves by name since a reconnect makes the cached SID
/// stale (#5698).
pub const TRACK_NAME: &str = "background_audio";

/// Resolved by name: `BuiltinAudioClip.path()` is public API, and the files are git-LFS objects
/// that `include_bytes!` could silently embed as pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinAudioClip {
    CityAmbience,
    ForestAmbience,
    OfficeAmbience,
    CrowdedRoom,
    KeyboardTyping,
    KeyboardTyping2,
    HoldMusic,
}

impl BuiltinAudioClip {
    /// The enum's Python value, the file name (public API).
    pub fn file_name(self) -> &'static str {
        match self {
            BuiltinAudioClip::CityAmbience => "city-ambience.ogg",
            BuiltinAudioClip::ForestAmbience => "forest-ambience.ogg",
            BuiltinAudioClip::OfficeAmbience => "office-ambience.ogg",
            BuiltinAudioClip::CrowdedRoom => "crowded-room.ogg",
            BuiltinAudioClip::KeyboardTyping => "keyboard-typing.ogg",
            BuiltinAudioClip::KeyboardTyping2 => "keyboard-typing2.ogg",
            // The one shipped name with an underscore; not a typo.
            BuiltinAudioClip::HoldMusic => "hold_music.ogg",
        }
    }
}

/// `_select_sound_from_list`: pick one config by probability, or none, from one uniform `draw` in
/// `[0, 1)` the host supplies. Probabilities summing below one leave the rest to silence.
pub fn select_sound(sounds: &[AudioConfig], draw: f64) -> Option<usize> {
    let total: f64 = sounds.iter().map(|s| s.probability).sum();
    if total <= 0.0 {
        return None;
    }
    // A total below 1.0 is a chance of silence, by design: the draw lands past every sound.
    if total < 1.0 && draw > total {
        return None;
    }

    let normalize_factor = if total <= 1.0 { 1.0 } else { total };
    let mut cumulative = 0.0;
    for (i, sound) in sounds.iter().enumerate() {
        if sound.probability <= 0.0 {
            continue;
        }
        cumulative += sound.probability / normalize_factor;
        if draw <= cumulative {
            return Some(i);
        }
    }
    // Float error can leave `r` above the final sum: the last entry wins.
    Some(sounds.len() - 1)
}

/// `None` when no change is needed, else one value per sample with volume and the equal-power fade
/// baked in. Mixed-width as numpy: `f64` ramp, `f32` accumulator.
pub fn frame_gain(
    t: u64,
    n: usize,
    stop_t: Option<u64>,
    fade_in: f64,
    fade_out: f64,
    sample_rate: u32,
    volume: f64,
) -> Option<Vec<f32>> {
    let sr = f64::from(sample_rate);
    let fade_in_n = if fade_in > 0.0 { (fade_in * sr) as u64 } else { 0 };
    let fade_out_n = if fade_out > 0.0 { (fade_out * sr) as u64 } else { 0 };
    let needs_fade_in = fade_in_n > 0 && t < fade_in_n;
    let needs_fade_out = fade_out_n > 0 && stop_t.is_some();
    if !needs_fade_in && !needs_fade_out && volume == 1.0 {
        return None;
    }

    let mut gain = vec![volume as f32; n];
    if needs_fade_in {
        for (i, g) in gain.iter_mut().enumerate() {
            let phase = ((t + i as u64) as f64 / fade_in_n as f64).clamp(0.0, 1.0);
            *g *= (phase * (PI / 2.0)).sin() as f32;
        }
    }
    if let Some(stop_t) = stop_t
        && fade_out_n > 0
    {
        for (i, g) in gain.iter_mut().enumerate() {
            let idx = (t + i as u64) as f64;
            let phase = ((idx - stop_t as f64) / fade_out_n as f64).clamp(0.0, 1.0);
            *g *= (phase * (PI / 2.0)).cos() as f32;
        }
    }
    Some(gain)
}

/// Apply a per-sample gain to an interleaved i16 frame (`np.repeat(gain, num_channels)`).
pub fn apply_gain(pcm: &[i16], gain: &[f32], num_channels: u16) -> Vec<i16> {
    let ch = num_channels.max(1) as usize;
    pcm.iter()
        .enumerate()
        .map(|(i, &s)| {
            let g = gain.get(i / ch).copied().unwrap_or(1.0);
            // `np.clip(...)` then `.astype(np.int16)`: saturate, then truncate toward zero.
            (f32::from(s) * g).clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameStep {
    /// `None`: pass the frame through untouched.
    pub gain: Option<Vec<f32>>,
    /// The fade-out has run its length; stop pulling frames after this one.
    pub last: bool,
}

/// Per-sound playback cursor (`_gen_wrapper`'s `t`, `stop_t`). The stop instant is latched once, at
/// the frame the stop was first seen.
#[derive(Debug, Clone, PartialEq)]
pub struct Playback {
    volume: f64,
    fade_in: f64,
    fade_out: f64,
    /// Cumulative samples (per channel) emitted so far.
    t: u64,
    stop_t: Option<u64>,
}

impl Playback {
    pub fn new(cfg: &AudioConfig) -> Self {
        Playback {
            volume: cfg.volume,
            fade_in: cfg.fade_in,
            fade_out: cfg.fade_out,
            t: 0,
            stop_t: None,
        }
    }

    /// Advance by one frame. With `fade_out == 0` nothing is latched; the driver drops the stream.
    pub fn on_frame(
        &mut self,
        samples_per_channel: usize,
        sample_rate: u32,
        stop_requested: bool,
    ) -> FrameStep {
        if self.stop_t.is_none() && self.fade_out > 0.0 && stop_requested {
            self.stop_t = Some(self.t);
        }
        let gain = frame_gain(
            self.t,
            samples_per_channel,
            self.stop_t,
            self.fade_in,
            self.fade_out,
            sample_rate,
            self.volume,
        );
        self.t += samples_per_channel as u64;
        let last = match self.stop_t {
            Some(stop_t) => self.t - stop_t >= (self.fade_out * f64::from(sample_rate)) as u64,
            None => false,
        };
        FrameStep { gain, last }
    }
}
