use lk_agents_abi::HostRef;
use lk_agents_abi::TtsCapabilities;
use lk_agents_abi::tokenize::bindings::SentenceTokenizerLeg;
use lk_agents_abi::tts::stream_pacer::SentenceStreamPacer;
use lk_agents_abi::tts::types::{SharedTtsOptions, TtsOptions};
use parking_lot::Mutex;

/// `_core.TtsOptionsCell`: a TTS's options, which its host reads back and replaces, and each
/// synthesis reads when its stage opens. Nothing waits on a change, so a plain lock holds them.
pub struct OptionsCell(Mutex<Held>);

struct Held {
    /// The tokenizer as the host gave it, read back as `tts.options.sentence_tokenizer`.
    sentence_tokenizer: HostRef,
    options: TtsOptions,
}

impl SharedTtsOptions for OptionsCell {
    fn current(&self) -> TtsOptions {
        self.0.lock().options.clone()
    }
}

impl lk_agents_abi::tts::types::TtsOptionsCell for OptionsCell {
    fn new(
        sentence_tokenizer: HostRef,
        leg: SentenceTokenizerLeg,
        capabilities: TtsCapabilities,
    ) -> Self {
        let options = TtsOptions { sentence_tokenizer: leg, pacing: capabilities.default_pacing() };
        OptionsCell(Mutex::new(Held { sentence_tokenizer, options }))
    }

    fn sentence_tokenizer(&self) -> HostRef {
        self.0.lock().sentence_tokenizer.clone()
    }

    fn pacing(&self) -> Option<SentenceStreamPacer> {
        self.0.lock().options.pacing
    }

    fn set(
        &self,
        sentence_tokenizer: HostRef,
        leg: SentenceTokenizerLeg,
        pacing: Option<SentenceStreamPacer>,
    ) {
        let options = TtsOptions { sentence_tokenizer: leg, pacing };
        *self.0.lock() = Held { sentence_tokenizer, options };
    }
}
