use super::*;

const RATE: u32 = 24_000;

fn tone(ms: u32) -> Vec<i16> {
    (0..RATE / 1000 * ms).map(|i| ((i as f64 * 0.12).sin() * 8000.0) as i16).collect()
}

fn silence(ms: u32) -> Vec<i16> {
    vec![0; (RATE / 1000 * ms) as usize]
}

fn detect(parts: &[Vec<i16>]) -> Vec<Pause> {
    let mut d = PauseDetector::new(RATE, 1);
    let mut out: Vec<Pause> = parts.iter().flat_map(|p| d.push(p.iter().copied())).collect();
    out.extend(d.finish());
    out
}

// Catches: a pause missed, mistimed, or a short gap (a stop consonant) taken for one.
#[test]
fn reports_the_leading_inner_and_trailing_silences() {
    let got = detect(&[
        silence(120),
        tone(600),
        silence(300),
        tone(400),
        silence(60),
        tone(400),
        silence(200),
    ]);
    let at = |p: &Pause| ((p.start * 100.0).round() as i64, (p.end * 100.0).round() as i64);
    assert_eq!(got.iter().map(at).collect::<Vec<_>>(), [(0, 12), (72, 102), (188, 208)]);
}

// Catches: a level fixed for one gain, so a quiet voice reads as silence throughout.
#[test]
fn the_threshold_follows_the_voice_s_own_level() {
    let quiet: Vec<i16> = tone(800).iter().map(|s| s / 40).collect();
    let got = detect(&[quiet.clone(), silence(300), quiet]);
    assert_eq!(got.len(), 1, "{got:?}");
    assert!((got[0].start - 0.8).abs() < 0.02 && (got[0].end - 1.1).abs() < 0.02, "{got:?}");
}

// Catches: a window split across two pushes counted twice or lost.
#[test]
fn pushes_of_any_size_give_the_same_pauses() {
    let audio: Vec<i16> = [tone(500), silence(400), tone(500)].concat();
    let whole = detect(std::slice::from_ref(&audio));
    let odd: Vec<Vec<i16>> = audio.chunks(337).map(<[i16]>::to_vec).collect();
    assert_eq!(detect(&odd), whole);
}

// Catches: a window taken for exactly 10 ms at a rate it does not divide (220 samples at 22.05
// kHz), so after half a minute a pause is placed past the audio received and read as its end.
#[test]
fn pauses_are_timed_in_the_audio_s_own_samples() {
    let rate = 22_050;
    let tone = |secs: usize| -> Vec<i16> {
        (0..rate * secs).map(|i| ((i as f64 * 0.12).sin() * 8000.0) as i16).collect()
    };
    let mut d = PauseDetector::new(rate as u32, 1);
    let found = d.push([tone(30), vec![0; rate / 2], tone(1)].concat());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        (found[0].start - 30.0).abs() < 0.011 && (found[0].end - 30.5).abs() < 0.011,
        "{found:?}"
    );
}
