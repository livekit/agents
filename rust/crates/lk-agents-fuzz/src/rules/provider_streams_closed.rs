//! After the session closed, every provider stream it opened was explicitly closed.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Kind, Phase, Recording},
};

pub struct ProviderStreamsClosed;

impl Rule for ProviderStreamsClosed {
    const CONTRACT: Contract = Contract {
        id: "provider_streams_closed",
        guarantee: "Every LLM, TTS and STT stream opened during the session is closed by the time the session is.",
        examples: &[closed_stream],
        counterexamples: &[dropped_stream],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // Streams are owed their close by the time the session is closed, not before: a run that
        // has not gone past the host's Close still has every right to hold them open.
        if r.phase() < Phase::Closing {
            return;
        }
        let inference = r.inference_view();
        for kind in LEGS {
            for stream in inference.streams(kind) {
                // A stream the product opened must reach the provider's `close`, not just be
                // dropped; a real plugin would leak its connection otherwise.
                check.expect(
                    &stream,
                    stream.closed_at().is_some(),
                    format!("{kind:?} stream {} was never closed", stream.nth()),
                );
            }
        }
    }
}

/// The three kinds a provider holds a connection for. Everything else a run opens is the
/// driver's own bookkeeping and has nobody to leak to.
const LEGS: [Kind; 3] = [Kind::Chat, Kind::Tts, Kind::Stt];

/// A TTS stream was opened and closed.
fn closed_stream() -> Recording {
    Authored::new().tts().closed(Kind::Tts, 0).driver_finished().done()
}

/// A TTS stream was opened and the session closed without closing it.
fn dropped_stream() -> Recording {
    Authored::new().tts().driver_finished().done()
}
