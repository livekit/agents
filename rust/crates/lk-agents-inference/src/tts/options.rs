use lk_agents_abi::HostRef;
use lk_agents_abi::TtsCapabilities;
use lk_agents_abi::tts::stream_pacer::SentenceStreamPacer;
use lk_agents_abi::tts::types::{OptionsTokenizer, SharedTtsOptions, TtsOptions};
use parking_lot::Mutex;

/// `_core.TtsOptionsCell`: a TTS's options, which its host reads back and replaces, and each
/// synthesis reads when its stage opens. Nothing waits on a change, so a plain lock holds them.
pub struct OptionsCell(Mutex<Held>);

struct Held {
    /// The tokenizer as the host gave it, read back as `tts.options.sentence_tokenizer`; `None`
    /// for the options of a TTS no host built.
    sentence_tokenizer: Option<HostRef>,
    options: TtsOptions,
}

impl OptionsCell {
    /// Split by `sentence_tokenizer`, paced as a TTS with `capabilities` is by default.
    pub fn host(sentence_tokenizer: OptionsTokenizer, capabilities: TtsCapabilities) -> Self {
        let options = TtsOptions {
            sentence_tokenizer: sentence_tokenizer.leg,
            pacing: capabilities.default_pacing(),
        };
        OptionsCell(Mutex::new(Held { sentence_tokenizer: Some(sentence_tokenizer.host), options }))
    }

    /// The options of a TTS no host built.
    pub fn native(options: TtsOptions) -> Self {
        OptionsCell(Mutex::new(Held { sentence_tokenizer: None, options }))
    }
}

impl SharedTtsOptions for OptionsCell {
    fn current(&self) -> TtsOptions {
        self.0.lock().options.clone()
    }
}

impl lk_agents_abi::tts::types::TtsOptionsCell for OptionsCell {
    fn new(sentence_tokenizer: OptionsTokenizer, capabilities: TtsCapabilities) -> Self {
        OptionsCell::host(sentence_tokenizer, capabilities)
    }

    fn sentence_tokenizer(&self) -> Option<HostRef> {
        self.0.lock().sentence_tokenizer.clone()
    }

    fn pacing(&self) -> Option<SentenceStreamPacer> {
        self.0.lock().options.pacing
    }

    fn set(&self, sentence_tokenizer: OptionsTokenizer, pacing: Option<SentenceStreamPacer>) {
        let options = TtsOptions { sentence_tokenizer: sentence_tokenizer.leg, pacing };
        *self.0.lock() = Held { sentence_tokenizer: Some(sentence_tokenizer.host), options };
    }
}
